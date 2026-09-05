"""Pins `arcsum.evalkit.latency` (SPEC §5.2 G4).

G4's verdict was reconstructed by hand from a benchmark table and a remembered decode length
until 2026-09-05, and was wrong twice for two different reasons. These tests pin both, because
each was a plausible-looking number that decided a ship question.
"""

from __future__ import annotations

from arcsum.evalkit.latency import (
    CEILING_MINUTES,
    DEVICE_DECODE_TPS,
    DEVICE_PREFILL_TPS,
    project,
)


def test_reproduces_the_corrected_reading_step_and_meeting_figures() -> None:
    """The measured `rl-v3` profile: 3,400-token prompt, ~190-token tool call, 15.2 steps.
    Corrected per-step is 77.7 s and the meeting is 20.4 min -- OVER the 20.00 ceiling."""
    p = project(steps=15.2, prefill_tokens_per_step=3400, decode_tokens_per_step=190)
    assert round(p.seconds_per_step, 1) == 77.7
    assert 20.2 < p.minutes < 20.6
    assert not p.passes
    assert p.margin < 0


def test_the_depth_zero_decode_rate_is_what_produced_the_wrong_PASS() -> None:
    """Not a hypothetical: the recorded 19.0 min used `tg190` with no `-d`, which measures
    decode from an EMPTY cache at 12.57 t/s. A reading step decodes after its own 3,400-token
    prompt, at 9.87. Substituting the wrong rate flips the gate, which is the whole lesson."""
    wrong = project(
        steps=15.2, prefill_tokens_per_step=3400, decode_tokens_per_step=190, decode_tps=12.57
    )
    right = project(steps=15.2, prefill_tokens_per_step=3400, decode_tokens_per_step=190)
    assert wrong.passes and not right.passes
    assert wrong.minutes < CEILING_MINUTES < right.minutes


def test_a_checkpoint_can_fail_G4_purely_by_RECORDING_MORE() -> None:
    """Decode length is not a device constant, it is a property of the checkpoint. The first
    RAFT pool's targets run 1.45x longer than gold, and that alone costs ~2 minutes -- so the
    anti-starvation work and G4 pull against each other, and inheriting `qwen-tools-v5`'s ~190
    tokens for a later checkpoint hides it."""
    lean = project(steps=15.2, prefill_tokens_per_step=3400, decode_tokens_per_step=190)
    rich = project(steps=15.2, prefill_tokens_per_step=3400, decode_tokens_per_step=190 * 1.45)
    assert rich.minutes - lean.minutes > 2.0
    assert not rich.passes


def test_fewer_larger_steps_trade_decode_overhead_for_prefill() -> None:
    """The 8k/6400-token mitigation, computed rather than remembered: each step re-sends ~800
    tokens of SYS + memory, so 15.2 -> 5.9 steps removes most of that repeated overhead."""
    small = project(steps=15.2, prefill_tokens_per_step=3400, decode_tokens_per_step=190)
    large = project(steps=5.9, prefill_tokens_per_step=6400, decode_tokens_per_step=300)
    assert large.minutes < small.minutes


def test_the_device_constants_carry_the_depth_they_were_measured_at() -> None:
    """A regression guard on the constants themselves. If someone re-measures decode without
    `-d` the value drifts up toward 12.57 and the gate silently starts passing again."""
    assert DEVICE_PREFILL_TPS > DEVICE_DECODE_TPS
    assert DEVICE_DECODE_TPS < 10.5, "decode must be the DEPTH-3400 rate, not the depth-0 rate"
