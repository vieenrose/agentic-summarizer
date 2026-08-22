"""Item-to-chunk alignment for §4.2's per-step supervision (SPEC §2.2 stage 1 out-of-
band offsets; §4.2 steps 1-3).

A chunk has no inherent connection to a MeetingBank item's boundaries -- it is
"however many lines fit in ~2,500 tokens," decided purely by `chunker.iter_chunks`.
Knowing each line's real source time span (from `corpus.meetingbank`'s offset-tracking
functions) is what lets us ask, after the fact, "which item(s) does this chunk's time
range actually cover?" -- so the teacher can be handed the item's real, human-authored
minute instead of inventing what mattered.

`chunk_offset_spans` deliberately mirrors `chunker.iter_chunks`'s packing algorithm
line for line rather than calling it, so that this module never has to touch (and risk
regressing) the harness's own chunking code merely to expose one extra piece of
bookkeeping real inference has no use for. `test_supervision_align.py` pins that the
two stay in lockstep: same `budget`/`overlap`/`token_len`, same per-chunk line counts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from arcsum.chunker import CHUNK_TOKENS, NEWLINE_COST, OVERLAP_LINES, SPLIT_SLACK, _split_long
from arcsum.tokens import heuristic_token_len
from arcsum.transcript import Utterance


def chunk_offset_spans(
    utterances: Sequence[Utterance],
    offsets: Sequence[tuple[float, float]],
    *,
    budget: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_LINES,
    token_len: Callable[[str], int] = heuristic_token_len,
) -> list[tuple[float, float]]:
    """One `(start_sec, end_sec)` per chunk `iter_chunks(utterances, budget=budget,
    overlap=overlap, token_len=token_len)` would yield, in the same order.

    `offsets[i]` is `utterances[i]`'s `(start_sec, end_sec)` span. A line produced by
    splitting an over-long utterance inherits its parent's whole span unchanged --
    there is no sub-utterance timing to split further, and SPEC's own item-boundary
    classification is already an approximation (§2.2), so this does not add a new
    source of error at the scale that matters.
    """
    if len(utterances) != len(offsets):
        raise ValueError(
            f"utterances and offsets must be the same length ({len(utterances)} != {len(offsets)})"
        )
    if not utterances:
        return []

    lines: list[Utterance] = []
    line_spans: list[tuple[float, float]] = []
    for u, span in zip(utterances, offsets, strict=True):
        pieces = _split_long(u, budget, token_len)
        lines.extend(pieces)
        line_spans.extend([span] * len(pieces))

    spans: list[tuple[float, float]] = []
    index = 0
    i = 0
    while i < len(lines):
        current: list[Utterance] = []
        current_spans: list[tuple[float, float]] = []
        used = 0
        while i < len(lines):
            cost = token_len(lines[i].render()) + NEWLINE_COST
            if current and used + cost > budget:
                room = budget - used
                pieces = _split_long(lines[i], room, token_len) if room > SPLIT_SLACK else []
                if len(pieces) > 1:
                    lines[i : i + 1] = pieces
                    line_spans[i : i + 1] = [line_spans[i]] * len(pieces)
                    continue
                break
            current.append(lines[i])
            current_spans.append(line_spans[i])
            used += cost
            i += 1

        if not current:
            break

        starts = [s for s, _e in current_spans]
        ends = [e for _s, e in current_spans]
        spans.append((min(starts), max(ends)))
        index += 1

        if i >= len(lines):
            break
        i = max(i - overlap, i - len(current) + 1)

    return spans


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    type: str
    summary: str
    start_sec: float
    end_sec: float


def overlapping_items(chunk_span: tuple[float, float], items: Sequence[Item]) -> list[Item]:
    """Items whose `[start_sec, end_sec)` interval overlaps `chunk_span`, in the order
    `items` was given (SPEC §4.2: "in meeting order")."""
    c_start, c_end = chunk_span
    return [it for it in items if it.start_sec < c_end and it.end_sec > c_start]
