"""Token-based, line-atomic chunking (SPEC §4.1).

Chunk size ~2,500 tokens follows from SPEC §4.1's context budget, not preference:
~250 SYS + <=600 memory + ~2,500 chunk ~= 3,500, inside a 4k window.

"Chunking is token-based over the whole transcript, not segment-aligned" (SPEC §4.1), and
boundaries snap to line boundaries because v2 lines are atomic. The one deliberate
exception is a line too long to fit alone, which must be split or the packer stalls.

The tokenizer is injected as a plain `Callable[[str], int]`, never imported. That is what
lets the whole suite run with no `transformers` and no weights, and it is why `Chunk`
carries its own measured `tokens`: the prior project's `Chunk.is_content_rich` called the
module-level heuristic even when the caller had injected a real tokenizer, so the
NOP-collapse guard and the budget silently disagreed about how big a chunk was.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from arcsum.tokens import heuristic_token_len
from arcsum.transcript import SEP, Utterance

#: SPEC §4.1. Raised to 6400 on 2026-08-31 and REVERTED to 2500 the same day, on
#: measurement. Keep this at 2500 unless all three findings below are addressed.
#:
#: The 8k case was real on latency and is preserved in `runs/g4-device-measured.md`:
#: peak RSS costs only +50 MB (Qwen3.5-0.8B is hybrid linear-attention, 6 of 24 layers
#: full attention, so its KV cache is nearly context-independent), and though prefill is
#: 16% slower per token, steps/meeting fall 15.2 -> 5.9 and the meeting goes 19.1 -> 16.9
#: min — margin against the 20-minute ceiling 5% -> 16%. That margin is the only measured
#: lever on G4's real exposure (process-contention stalls, worst case 21.6 min).
#:
#: **It was reverted anyway, because three things broke at 6400** (`runs/v5-8k/`):
#:   1. NOP supervision collapses. Teacher NOP share 38.2% -> **1.4%** (18 of 1,326 steps
#:      over the 200-meeting pilot): a 6,400-token chunk almost always overlaps SOME
#:      item-covered span, so honest NOP targets nearly vanish. Trap 1 records what a
#:      NOP-starved pool produces — a checkpoint that cannot abstain and churns
#:      DROP + near-identical re-ADD. `mix_phase4` can downsample NOP, never manufacture it.
#:   2. G3 rouge1 FAILS. `qwen-tools-v5` at 6400 over the 40 held-out meetings: 21/19,
#:      +0.003, p=0.875, down from 28/12, +0.069, p=0.017. rouge2/rougeL survive at
#:      roughly half their margin. (Off-distribution serving, so not decisive alone —
#:      but it is the direction trap 6 and `runs/chunk1500/` both predict.)
#:   3. Every G1 probe scenario collapses to ONE chunk, so the gate silently measures
#:      nothing. Pinned now by tests over BOTH probe sets; see `probe_data.py`.
#:
#: Note 8k SERVING context (`n_ctx`) is a separate knob and remains 8192 — it is only the
#: transcript-per-step that reverts.
#:
#: **Chunk size is baked into the SFT distribution.** Changing it REQUIRES regenerating
#: supervision and retraining; serving an old checkpoint at a new chunk size is
#: off-distribution and was the confound that muddied `runs/chunk1500/`. Keep it a
#: parameter everywhere so trace generation and deployment cannot silently disagree.
CHUNK_TOKENS = 2500

#: Lines re-shown at the head of the next chunk, so a decision spanning a boundary is not
#: cut in half.
OVERLAP_LINES = 2

#: Minimum room worth re-splitting a line into rather than ending the chunk early. Tuned
#: at 2048 in the prior project; re-checked here at 2500.
SPLIT_SLACK = 64

#: The packer charges one token per line for the newline that joins it to the next. It is
#: charged for the first line too, which over-counts a chunk by one — deliberately
#: conservative, since an over-estimate keeps the real window safe.
NEWLINE_COST = 1

#: A chunk is "content-rich" if it is at least this fraction of the budget. Expressed as a
#: FRACTION, not the prior project's absolute 120 tokens: because the heuristic counts ~1
#: token per CJK character, an absolute threshold is ~4x stricter in zh than in en — an
#: asymmetry nobody chose. Calibrate on the pilot (SPEC §8 risk 3).
CONTENT_RICH_FRAC = 0.25


@dataclass(frozen=True, slots=True)
class Chunk:
    """One reading step's worth of transcript."""

    index: int
    utterances: tuple[Utterance, ...]
    #: Measured by the packer with the INJECTED counter, so every consumer reads the same
    #: number the budget used.
    tokens: int

    def render(self) -> str:
        return "\n".join(u.render() for u in self.utterances)

    def is_content_rich(
        self, *, min_frac: float = CONTENT_RICH_FRAC, budget: int = CHUNK_TOKENS
    ) -> bool:
        """Enough substance to be worth an op (SPEC §8 risk 3).

        Short back-channel exchanges ("嗯", "對", "好") are not content-rich, and answering
        NOP on them is correct behaviour rather than collapse. Only content-rich chunks
        count toward the NOP-collapse guard.
        """
        return self.tokens >= min_frac * budget

    def __len__(self) -> int:
        return len(self.utterances)


def _split_long(u: Utterance, budget: int, token_len: Callable[[str], int]) -> list[Utterance]:
    """Split one over-long utterance into pieces that each fit `budget` *as packed*.

    Every piece re-emits the speaker prefix, because v2's speaker field is mandatory
    (SPEC §2) — so the per-piece overhead is `token_len(f"{speaker}: ")`. One further
    token is reserved for the newline the packer charges per line: without it, a piece
    sized to exactly `budget` costs `budget + 1` once packed and could never be placed.

    Splits on spaces for latin text and character-wise for CJK, which has no spaces to
    split on. Returns `[u]` unchanged when it already fits or cannot be usefully divided.
    """
    overhead = token_len(f"{u.speaker}{SEP}") + NEWLINE_COST
    room = budget - overhead
    if room <= 0 or token_len(u.render()) + NEWLINE_COST <= budget:
        return [u]

    words = u.text.split(" ")
    units = words if len(words) > 1 else list(u.text)
    joiner = " " if len(words) > 1 else ""

    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = joiner.join([*current, unit])
        if current and token_len(candidate) > room:
            pieces.append(joiner.join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        pieces.append(joiner.join(current))

    return [Utterance(u.speaker, piece) for piece in pieces if piece] or [u]


def iter_chunks(
    utterances: list[Utterance],
    *,
    budget: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_LINES,
    token_len: Callable[[str], int] = heuristic_token_len,
) -> Iterator[Chunk]:
    """Pack utterances into token-budgeted, line-atomic chunks.

    Two phases. Any line that cannot fit on its own is pre-split, so the packer can never
    stall. Then whole rendered lines accumulate greedily; on overflow, if the remaining
    room still exceeds `SPLIT_SLACK` the offending line is re-split in place rather than
    ending the chunk early — on long-turn transcripts (up to ~2.6k chars/line) not doing
    this wasted ~27% of the chunk in the prior project.
    """
    if not utterances:
        return

    lines: list[Utterance] = []
    for u in utterances:
        lines.extend(_split_long(u, budget, token_len))

    index = 0
    i = 0
    while i < len(lines):
        current: list[Utterance] = []
        used = 0
        while i < len(lines):
            cost = token_len(lines[i].render()) + NEWLINE_COST
            if current and used + cost > budget:
                room = budget - used
                pieces = _split_long(lines[i], room, token_len) if room > SPLIT_SLACK else []
                if len(pieces) > 1:
                    lines[i : i + 1] = pieces
                    continue
                break
            current.append(lines[i])
            used += cost
            i += 1

        if not current:
            break

        yield Chunk(index, tuple(current), used)
        index += 1

        if i >= len(lines):
            break
        # Rewind for overlap, clamped so the cursor always advances.
        i = max(i - overlap, i - len(current) + 1)
