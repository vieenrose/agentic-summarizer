"""SPEC §5.2 G4: project per-meeting wall clock on the reference device.

**Why this is a module and not arithmetic in a markdown file.** G4 is the one gate whose
verdict was, until now, reconstructed by hand from a benchmark table plus a remembered decode
length. That is exactly the shape CLAUDE.md trap 11 names as this project's most common
failure -- a number with no artifact behind it -- and it produced a wrong verdict twice:

* The decode rate was taken from `tg190` with no `-d`, i.e. decode from an EMPTY cache. A
  reading step never decodes there: its prompt is SYS + MEMORY + CHUNK ~3,400 tokens, so every
  decoded token attends over 3,400+ tokens of KV, where the device measures 9.87 t/s against
  12.57 at depth 0. **26% slower at the depth the system actually runs at.** That alone moved
  the verdict from 19.0 min (PASS) to 20.4 min (FAIL).
* The decode LENGTH was inherited from `qwen-tools-v5` (~190 tokens/step) and applied to every
  later checkpoint. It is not a constant. The first RAFT pool's targets run 1.45x longer, which
  by itself moves a meeting from 20.3 to 22.5 minutes -- so a checkpoint can fail G4 purely by
  recording more, and the anti-starvation work does exactly that.

So the projection now takes decode length from the RUN being scored, and the throughput
constants carry the depth they were measured at.

**Prefill is measured at depth 0 and that is correct**, not an oversight: SPEC §4.1 lets no
conversation history cross steps, so every step's prompt is built from empty. Only decode
happens deep.

**What this does NOT model.** Throughput here is nominal, single-run. `runs/g4-device-measured.md`
records that thermal throttling is not observable at 0.8B/Q8 over 29.5 minutes (prefill -0.2%,
decode -2.4%) but that transient decode stalls from process contention hit 2 of 14 rounds,
costing 13-37% of decode. A projection is a floor; the worst observed round runs ~11% over it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Reno 7 5G (CPH2371), `rl-v3` Q8_0, `-C 0xFF -t 8`, llama.cpp `15586e2d7`.
#: Measured 2026-09-05 via `ssh -o ProxyJump=raspberrypi user@100.122.78.108`.
#: Prefill at depth 0 (a step's prompt is always built from empty).
DEVICE_PREFILL_TPS = 58.15
#: Decode at depth 3400 -- the depth a reading step's tokens are actually emitted at.
#: The d-sweep behind it: d0 12.57, d1000 11.83, d2000 10.90, d3000 9.99, d3400 9.87.
DEVICE_DECODE_TPS = 9.87
#: SPEC §5.2 G4.
CEILING_MINUTES = 20.0


@dataclass(frozen=True)
class LatencyProjection:
    steps: float
    prefill_tokens_per_step: float
    decode_tokens_per_step: float
    seconds_per_step: float
    synthesis_seconds: float
    minutes: float

    @property
    def passes(self) -> bool:
        return self.minutes <= CEILING_MINUTES

    @property
    def margin(self) -> float:
        """Fraction of the ceiling left over. Negative when over budget."""
        return (CEILING_MINUTES - self.minutes) / CEILING_MINUTES


def project(
    *,
    steps: float,
    prefill_tokens_per_step: float,
    decode_tokens_per_step: float,
    synthesis_prefill_tokens: float = 1000.0,
    synthesis_decode_tokens: float = 250.0,
    prefill_tps: float = DEVICE_PREFILL_TPS,
    decode_tps: float = DEVICE_DECODE_TPS,
) -> LatencyProjection:
    """Project a meeting's wall clock from its MEASURED token profile.

    Every argument is a quantity the eval run can report, so nothing here is remembered.
    """
    per_step = prefill_tokens_per_step / prefill_tps + decode_tokens_per_step / decode_tps
    synth = synthesis_prefill_tokens / prefill_tps + synthesis_decode_tokens / decode_tps
    return LatencyProjection(
        steps=steps,
        prefill_tokens_per_step=prefill_tokens_per_step,
        decode_tokens_per_step=decode_tokens_per_step,
        seconds_per_step=per_step,
        synthesis_seconds=synth,
        minutes=(per_step * steps + synth) / 60.0,
    )
