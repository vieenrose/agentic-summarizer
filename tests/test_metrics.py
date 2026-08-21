"""Pins SPEC §5's reference metrics: character-level ROUGE-1/2/L, Coverage/Density,
and length. All golden values in `tests/fixtures/rouge/cases.json` are hand-computed
or independently derived — never produced by calling the functions under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcsum.metrics.reference import (
    Length,
    coverage,
    density,
    extractive_fragments,
    length,
    rouge_l,
    rouge_n,
)
from arcsum.tokens import TOKENIZE_VERSION, heuristic_token_len

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "rouge" / "cases.json").read_text())


def test_fixture_declares_the_current_tokenize_version() -> None:
    assert FIXTURE["tokenize_version"] == TOKENIZE_VERSION


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_rouge1_matches_golden_fixtures(case: dict) -> None:
    score = rouge_n(case["candidate"], case["reference"], n=1)
    exp = case["rouge1"]
    assert score.precision == pytest.approx(exp["precision"]), case["why"]
    assert score.recall == pytest.approx(exp["recall"]), case["why"]
    assert score.f1 == pytest.approx(exp["f1"]), case["why"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_rouge2_matches_golden_fixtures(case: dict) -> None:
    score = rouge_n(case["candidate"], case["reference"], n=2)
    exp = case["rouge2"]
    assert score.precision == pytest.approx(exp["precision"]), case["why"]
    assert score.recall == pytest.approx(exp["recall"]), case["why"]
    assert score.f1 == pytest.approx(exp["f1"]), case["why"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_rougeL_matches_golden_fixtures(case: dict) -> None:
    score = rouge_l(case["candidate"], case["reference"])
    exp = case["rougeL"]
    assert score.precision == pytest.approx(exp["precision"]), case["why"]
    assert score.recall == pytest.approx(exp["recall"]), case["why"]
    assert score.f1 == pytest.approx(exp["f1"]), case["why"]


def test_rouge_records_the_tokenize_version() -> None:
    score = rouge_n("a", "a", n=1)
    assert score.tokenize_version == TOKENIZE_VERSION


def test_rouge_scores_are_never_out_of_bounds() -> None:
    for n in (1, 2):
        for c in FIXTURE["cases"]:
            score = rouge_n(c["candidate"], c["reference"], n=n)
            assert 0.0 <= score.precision <= 1.0
            assert 0.0 <= score.recall <= 1.0
            assert 0.0 <= score.f1 <= 1.0


def test_rouge_n_multiset_not_set_intersection() -> None:
    """A candidate repeating an n-gram the reference also repeats must not be
    under-counted by naive set intersection (which would give overlap=1, not 2)."""
    score = rouge_n("a a a", "a a", n=1)
    assert score.recall == pytest.approx(1.0)  # both of the reference's 2 'a's covered


def test_rouge_n_with_n_larger_than_either_sequence_is_zero_not_a_crash() -> None:
    score = rouge_n("a", "a", n=5)
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


# --- extractive_fragments / coverage / density ---------------------------------------


def test_extractive_fragments_hand_verified_case() -> None:
    """source=[a,b,c,d], summary=[a,b,x,c,d]. Fragment 1: 'a b' (len 2, from source[0:2]).
    Fragment 2 (after the unmatched 'x'): 'c d' (len 2, from source[2:4])."""
    fragments = extractive_fragments(["a", "b", "c", "d"], ["a", "b", "x", "c", "d"])
    assert fragments == [["a", "b"], ["c", "d"]]


def test_extractive_fragments_allows_out_of_order_matches() -> None:
    """A summary fragment need not appear at the source position the PREVIOUS fragment
    did -- the source is rescanned from the start each time."""
    fragments = extractive_fragments(["b", "a"], ["a", "b"])
    assert fragments == [["a"], ["b"]]


def test_extractive_fragments_no_overlap_at_all() -> None:
    assert extractive_fragments(["a", "b"], ["x", "y"]) == []


def test_extractive_fragments_empty_summary() -> None:
    assert extractive_fragments(["a", "b"], []) == []


def test_coverage_hand_verified_case() -> None:
    """source=a b c d, summary=a b x c d -> covered=4 of 5 summary tokens -> 0.8."""
    assert coverage("a b c d", "a b x c d") == pytest.approx(0.8)


def test_density_hand_verified_case() -> None:
    """Same case: fragments of length 2 and 2 -> (2^2+2^2)/5 = 1.6."""
    assert density("a b c d", "a b x c d") == pytest.approx(1.6)


def test_coverage_is_one_for_a_fully_extractive_summary() -> None:
    assert coverage("a b c d", "a b c d") == pytest.approx(1.0)


def test_coverage_is_zero_for_an_empty_summary() -> None:
    assert coverage("a b c d", "") == 0.0


def test_density_is_zero_for_an_empty_summary() -> None:
    assert density("a b c d", "") == 0.0


def test_coverage_is_zero_for_a_wholly_abstractive_summary() -> None:
    assert coverage("a b c d", "x y z") == 0.0


def test_density_is_not_bounded_to_one() -> None:
    """A summary that is one long verbatim copy has density >> 1, unlike coverage."""
    long_source = "a b c d e f g h i j"
    assert density(long_source, long_source) > 1.0


def test_coverage_and_density_use_the_normative_char_tokenizer() -> None:
    """CJK must be tokenised per-character, not word-split, for this to mean anything."""
    assert coverage("同意搬到 B 棟", "同意搬到 B 棟") == pytest.approx(1.0)


# --- length ------------------------------------------------------------------------------


def test_length_reports_chars_and_tokens() -> None:
    result = length("同意搬到 B 棟", token_len=heuristic_token_len)
    assert isinstance(result, Length)
    assert result.chars == len("同意搬到 B 棟")
    assert result.tokens == heuristic_token_len("同意搬到 B 棟")


def test_length_uses_the_injected_tokenizer() -> None:
    calls: list[str] = []

    def counting(text: str) -> int:
        calls.append(text)
        return 42

    result = length("some text", token_len=counting)
    assert result.tokens == 42
    assert calls == ["some text"]


def test_length_of_empty_text() -> None:
    result = length("", token_len=heuristic_token_len)
    assert result.chars == 0
    assert result.tokens == 0
