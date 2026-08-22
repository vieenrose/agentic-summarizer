"""Pins `supervision.align` (SPEC §2.2 stage 1 out-of-band offsets; §4.2 item-to-chunk
alignment): chunk_offset_spans must stay in lockstep with the real chunker.iter_chunks,
and overlapping_items must find the items a chunk's time span actually covers.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from arcsum.chunker import iter_chunks
from arcsum.supervision.align import Item, chunk_offset_spans, overlapping_items
from arcsum.tokens import heuristic_token_len
from arcsum.transcript import Utterance


def _linear_offsets(
    utterances: list[Utterance], *, seconds_per_line: float = 1.0
) -> list[tuple[float, float]]:
    """A simple, deterministic offset schedule: line i spans [i, i+1) seconds."""
    return [(i * seconds_per_line, (i + 1) * seconds_per_line) for i in range(len(utterances))]


def test_chunk_offset_spans_matches_iter_chunks_line_counts_short_input() -> None:
    utterances = [Utterance("S1", f"line {i}") for i in range(10)]
    offsets = _linear_offsets(utterances)

    chunks = list(iter_chunks(utterances, budget=50, overlap=0, token_len=heuristic_token_len))
    spans = chunk_offset_spans(
        utterances, offsets, budget=50, overlap=0, token_len=heuristic_token_len
    )

    assert len(spans) == len(chunks)


def test_chunk_offset_spans_covers_the_whole_meeting_with_no_overlap() -> None:
    utterances = [Utterance("S1", f"line number {i}") for i in range(20)]
    offsets = _linear_offsets(utterances)

    spans = chunk_offset_spans(
        utterances, offsets, budget=30, overlap=0, token_len=heuristic_token_len
    )

    assert spans[0][0] == 0.0
    assert spans[-1][1] == 20.0
    # Spans must be contiguous (no gap) when there is no overlap.
    for (_, end), (next_start, _) in pairwise(spans):
        assert end == next_start


def test_chunk_offset_spans_with_overlap_matches_a_real_multichunk_iter_chunks_run() -> None:
    """A property test against the real budget/overlap defaults: same number of
    chunks, and every chunk's line count from iter_chunks matches what
    chunk_offset_spans consumed to produce its span (checked indirectly via count)."""
    utterances = [
        Utterance("S1", f"a somewhat longer utterance number {i} for packing") for i in range(60)
    ]
    offsets = _linear_offsets(utterances)

    chunks = list(iter_chunks(utterances, budget=100, overlap=2, token_len=heuristic_token_len))
    spans = chunk_offset_spans(
        utterances, offsets, budget=100, overlap=2, token_len=heuristic_token_len
    )

    assert len(spans) == len(chunks)
    # Every chunk's rendered content is a substring of [first_line, last_line] in the
    # source utterance order -- so its span's start must be <= its end.
    for start, end in spans:
        assert start <= end


def test_chunk_offset_spans_a_split_long_line_inherits_its_parent_span() -> None:
    """An over-long utterance gets fragmented by iter_chunks's own splitting logic;
    every fragment must carry the SAME span as the whole original utterance."""
    long_text = "word " * 500
    utterances = [Utterance("S1", long_text)]
    offsets = [(10.0, 20.0)]

    spans = chunk_offset_spans(
        utterances, offsets, budget=50, overlap=0, token_len=heuristic_token_len
    )

    assert len(spans) > 1  # the long utterance really did get split into multiple chunks
    for start, end in spans:
        assert (start, end) == (10.0, 20.0)


def test_chunk_offset_spans_empty_input() -> None:
    assert chunk_offset_spans([], []) == []


def test_chunk_offset_spans_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        chunk_offset_spans([Utterance("S1", "hi")], [])


# --- overlapping_items ---------------------------------------------------------------


def _item(item_id: str, start: float, end: float) -> Item:
    return Item(
        item_id=item_id,
        type="Ordinance",
        summary=f"summary for {item_id}",
        start_sec=start,
        end_sec=end,
    )


def test_overlapping_items_finds_items_within_the_chunk_span() -> None:
    items = [_item("a", 0, 10), _item("b", 15, 25), _item("c", 100, 110)]
    result = overlapping_items((5, 20), items)
    assert [it.item_id for it in result] == ["a", "b"]


def test_overlapping_items_excludes_items_entirely_outside_the_span() -> None:
    items = [_item("a", 0, 10)]
    assert overlapping_items((20, 30), items) == []


def test_overlapping_items_touching_but_not_overlapping_boundaries_excluded() -> None:
    """An item ending exactly when the chunk starts (or vice versa) shares no real
    time with it -- half-open interval semantics, not inclusive on both ends."""
    items = [_item("a", 0, 10)]
    assert overlapping_items((10, 20), items) == []


def test_overlapping_items_preserves_meeting_order() -> None:
    items = [_item("b", 5, 6), _item("a", 1, 2)]  # deliberately out of chrono order
    result = overlapping_items((0, 100), items)
    assert [it.item_id for it in result] == ["b", "a"]  # order of `items`, not sorted


def test_overlapping_items_empty_items_list() -> None:
    assert overlapping_items((0, 100), []) == []
