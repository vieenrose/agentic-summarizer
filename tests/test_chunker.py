"""Pins SPEC §4.1's chunking contract.

Guards a latent bug carried by the prior project: `Chunk.is_content_rich` measured the
chunk with the module-level heuristic even when the caller had injected a real tokenizer,
so the NOP-collapse guard and the budget disagreed about chunk size. Here the packer
records `Chunk.tokens` with the injected counter and every consumer reads that.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from arcsum.chunker import CHUNK_TOKENS, Chunk, iter_chunks
from arcsum.tokens import heuristic_token_len
from arcsum.transcript import Utterance


def meeting(n: int, words_per_line: int = 20) -> list[Utterance]:
    return [Utterance(f"S{i % 4 + 1}", "很好 " * words_per_line) for i in range(n)]


def test_empty_transcript_yields_no_chunks() -> None:
    assert list(iter_chunks([])) == []


def test_every_line_appears_in_some_chunk() -> None:
    """Nothing may be silently dropped — an unread line is an unsummarized decision."""
    utts = meeting(200)
    seen = {u.text for c in iter_chunks(utts) for u in c.utterances}
    assert seen == {u.text for u in utts}


def test_chunk_respects_budget() -> None:
    for chunk in iter_chunks(meeting(300), budget=500):
        assert chunk.tokens <= 500


def test_boundaries_fall_on_line_boundaries() -> None:
    """SPEC §4.1: "snapped to the nearest line boundary, since §2 lines are atomic"."""
    utts = meeting(100)
    originals = {u.text for u in utts}
    for chunk in iter_chunks(utts, budget=400):
        for u in chunk.utterances:
            assert u.text in originals


def test_chunks_are_contiguous_with_overlap() -> None:
    """Consecutive chunks share their overlap lines, so a boundary cannot hide a decision."""
    chunks = list(iter_chunks(meeting(120), budget=400, overlap=2))
    assert len(chunks) > 2
    for a, b in pairwise(chunks):
        tail = [u.text for u in a.utterances[-2:]]
        head = [u.text for u in b.utterances[:2]]
        assert set(head) & set(tail)


def test_chunk_indices_are_sequential_from_zero() -> None:
    assert [c.index for c in iter_chunks(meeting(80), budget=400)] == list(
        range(len(list(iter_chunks(meeting(80), budget=400))))
    )


@pytest.mark.parametrize("words_per_line", [200, 600, 1200])
def test_over_long_line_is_split_and_each_piece_keeps_its_speaker(words_per_line: int) -> None:
    """v2's speaker field is mandatory, so every split piece must re-emit it (SPEC §2)."""
    utts = [Utterance("S1", "很好 " * words_per_line)]
    chunks = list(iter_chunks(utts, budget=300))
    assert len(chunks) >= 1
    for chunk in chunks:
        for u in chunk.utterances:
            assert u.speaker == "S1"
            assert u.render().startswith("S1: ")
        assert chunk.tokens <= 300


def test_a_single_over_long_line_still_terminates() -> None:
    """The pre-split phase exists so the packer can never stall on one huge line."""
    utts = [Utterance("S1", "很" * 5000)]
    chunks = list(iter_chunks(utts, budget=200))
    assert chunks
    assert sum(len(c) for c in chunks) >= 1


def test_cjk_line_splits_character_wise() -> None:
    """CJK has no spaces to split on; character-wise accumulation is the fallback."""
    utts = [Utterance("S1", "決" * 800)]
    chunks = list(iter_chunks(utts, budget=250))
    joined = "".join(u.text for c in chunks for u in c.utterances)
    assert set(joined) == {"決"}


def test_chunk_tokens_uses_the_injected_tokenizer() -> None:
    """The budget instrument must be the passed one, never a hidden default."""
    calls: list[str] = []

    def counting(text: str) -> int:
        calls.append(text)
        return heuristic_token_len(text)

    chunks = list(iter_chunks(meeting(20), budget=400, token_len=counting))
    assert calls, "the injected counter was never called"
    assert all(c.tokens <= 400 for c in chunks)


def test_chunk_tokens_reflects_a_double_counting_tokenizer() -> None:
    """A different counter must produce different packing — proof it is really injected."""
    utts = meeting(40)
    normal = list(iter_chunks(utts, budget=800, token_len=heuristic_token_len))
    doubled = list(iter_chunks(utts, budget=800, token_len=lambda t: 2 * heuristic_token_len(t)))
    assert len(doubled) > len(normal)


def test_is_content_rich_uses_the_measured_tokens() -> None:
    """The guard reads the same number the budget used (the prior project's latent bug)."""
    rich = Chunk(0, (Utterance("S1", "x"),), tokens=int(0.5 * CHUNK_TOKENS))
    thin = Chunk(1, (Utterance("S1", "嗯"),), tokens=10)
    assert rich.is_content_rich()
    assert not thin.is_content_rich()


def test_is_content_rich_is_scale_free_across_scripts() -> None:
    """Expressed as a fraction of budget, zh and en chunks are comparable by construction."""
    at_threshold = Chunk(0, (), tokens=int(0.25 * 1000))
    assert at_threshold.is_content_rich(budget=1000)
    assert not Chunk(0, (), tokens=int(0.25 * 1000) - 1).is_content_rich(budget=1000)


def test_render_joins_lines_with_newlines() -> None:
    chunk = Chunk(0, (Utterance("S1", "a"), Utterance("S2", "b")), tokens=4)
    assert chunk.render() == "S1: a\nS2: b"
    assert len(chunk) == 2
