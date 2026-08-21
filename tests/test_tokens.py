"""Pins SPEC §5's normative character-level tokenization.

Guards the prior project's recurring defect: THREE drifted CJK-range implementations
(`chunker.heuristic_token_len` U+3000-U+9FFF, `index.tokenise` U+3400-U+9FFF,
`guards._tokens` U+4E00-U+9FFF) gave three different answers to "is this CJK", while
SPEC §5 makes the answer normative and comparison-critical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcsum import tokens

FIXTURES = Path(__file__).parent / "fixtures" / "chartok" / "cases.json"
_FIXTURE = json.loads(FIXTURES.read_text(encoding="utf-8"))
_CASES = _FIXTURE["cases"]


def test_fixture_declares_the_current_tokenize_version() -> None:
    """A tokenizer change must not land without regenerating and reviewing the goldens.

    This is the mechanism that makes "normative" operational rather than aspirational.
    """
    assert _FIXTURE["tokenize_version"] == tokens.TOKENIZE_VERSION


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_char_tokens_matches_golden_fixtures(case: dict) -> None:
    assert tokens.char_tokens(case["text"]) == case["tokens"], case["why"]


def test_char_tokens_is_total() -> None:
    """Never raises, for any input — a metric that can crash cannot be normative."""
    for text in ("", " ", "\n\t", "。、；", "　", "\x00", "🎉", "a" * 5000, "測" * 5000):
        assert isinstance(tokens.char_tokens(text), list)


def test_fullwidth_and_halfwidth_tokenise_identically() -> None:
    """NFKC folding is what makes an ordinance number stable across width variants."""
    assert tokens.char_tokens("ＣＢ　１１８６１８") == tokens.char_tokens("CB 118618")


def test_punctuation_never_becomes_a_token() -> None:
    """Counting punctuation as tokens is conservative for a budget and wrong for ROUGE."""
    assert tokens.char_tokens("好，就。搬！到？") == tokens.char_tokens("好 就 搬 到")


def test_is_cjk_is_the_single_source_of_truth() -> None:
    """Every CJK question routes through one predicate, over one set of ranges."""
    assert tokens.is_cjk("我") and tokens.is_cjk("㐀") and tokens.is_cjk("か")
    assert not tokens.is_cjk("a")
    assert not tokens.is_cjk("1")
    # Punctuation and fullwidth forms are NOT ideographs — normalisation handles them.
    assert not tokens.is_cjk("，")
    assert not tokens.is_cjk("。")
    assert not tokens.is_cjk("Ｂ")


def test_lexical_tokens_uses_cjk_bigrams_and_latin_unigrams() -> None:
    """Word-splitting Chinese gives one giant token and collapses every overlap score."""
    assert tokens.lexical_tokens("我們搬到") == {"我 們", "們 搬", "搬 到"}
    assert "ordinance" in tokens.lexical_tokens("通過 Ordinance")


def test_lexical_tokens_and_char_tokens_agree_on_what_is_cjk() -> None:
    """The two consumers of the range table must not drift apart again."""
    text = "議會通過 CB 118618 號提案"
    cjk_chars = {t for t in tokens.char_tokens(text) if tokens.is_cjk(t[0])}
    latin = {t for t in tokens.char_tokens(text) if not tokens.is_cjk(t[0])}
    assert latin <= tokens.lexical_tokens(text)
    for bigram in tokens.lexical_tokens(text) - latin:
        assert all(part in cjk_chars for part in bigram.split(" "))


def test_bigrams_are_empty_below_two_tokens() -> None:
    assert tokens.bigrams([]) == []
    assert tokens.bigrams(["我"]) == []
    assert tokens.bigrams(["我", "們"]) == ["我 們"]


def test_heuristic_over_estimates_so_packed_chunks_fit() -> None:
    """The budget estimator must never under-count, or a chunk overflows the real window."""
    assert tokens.heuristic_token_len("我們搬到B棟") >= 6
    assert tokens.heuristic_token_len("") == 0
    # ~1 token per CJK char, ~4 chars per latin token.
    assert tokens.heuristic_token_len("測" * 100) == 100
    assert tokens.heuristic_token_len("a" * 100) == 25


def test_token_len_name_identifies_the_instrument() -> None:
    """A silent instrument swap between trace generation and inference is a real bug class."""
    assert tokens.token_len_name(tokens.heuristic_token_len) == "heuristic"


def test_core_package_has_no_third_party_imports() -> None:
    """The zero-core-dependency property, asserted rather than trusted.

    `transformers` must be imported lazily inside `hf_token_len`, never at module scope.
    """
    source = (Path(__file__).parent.parent / "src" / "arcsum" / "tokens.py").read_text(
        encoding="utf-8"
    )
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "__future__" not in line
    ]
    for line in module_level:
        assert "transformers" not in line, f"third-party import at module scope: {line}"
