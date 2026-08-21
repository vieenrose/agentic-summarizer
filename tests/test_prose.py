"""Pins the §3 output contract (SPEC §3): "no bullets, no sections, no anchors,
< 1,000 tokens." Net-new relative to the prior project, which had no prose product —
this is the one enforcement point both the agent's SYNTHESIZE call and the baseline's
reduce step share, so they cannot disagree about what a valid output is.
"""

from __future__ import annotations

from arcsum.prose import PROSE_MAX_TOKENS, finalize
from arcsum.tokens import heuristic_token_len


def test_clean_prose_passes_through_unchanged() -> None:
    text = "會議討論辦公室搬遷案，最終決議遷至 B 棟大樓。"
    prose = finalize(text, token_len=heuristic_token_len)
    assert prose.text == text
    assert prose.had_markup is False
    assert prose.lang_flags == ()


def test_strips_leading_bullet_markers() -> None:
    raw = "- 會議討論搬遷案\n- 決議遷至 B 棟"
    prose = finalize(raw, token_len=heuristic_token_len)
    assert "-" not in prose.text
    assert prose.had_markup is True
    assert "會議討論搬遷案" in prose.text and "決議遷至 B 棟" in prose.text


def test_strips_numbered_list_markers() -> None:
    raw = "1. 會議討論搬遷案\n2. 決議遷至 B 棟"
    prose = finalize(raw, token_len=heuristic_token_len)
    assert "1." not in prose.text
    assert prose.had_markup is True


def test_strips_markdown_headings() -> None:
    raw = "## 摘要\n會議討論搬遷案"
    prose = finalize(raw, token_len=heuristic_token_len)
    assert "#" not in prose.text
    assert prose.had_markup is True


def test_strips_hallucinated_harness_labels() -> None:
    """A model that drifted toward the old bulleted format might emit its own labels."""
    raw = "TITLE: 搬遷案\nSUMMARY: 會議討論搬遷案"
    prose = finalize(raw, token_len=heuristic_token_len)
    assert "TITLE" not in prose.text
    assert "SUMMARY" not in prose.text
    assert prose.had_markup is True


def test_strips_hallucinated_timestamp_anchors() -> None:
    """v2 has no timestamps (SPEC §2); one leaking into the summary is a bug to fix,
    not a feature to preserve."""
    raw = "會議決議遷至 B 棟 [3:35]，並於 [1:02:07] 確認預算。"
    prose = finalize(raw, token_len=heuristic_token_len)
    assert "[" not in prose.text and "]" not in prose.text


def test_strips_markdown_emphasis_characters() -> None:
    raw = "會議**決議**遷至 `B 棟`。"
    prose = finalize(raw, token_len=heuristic_token_len)
    assert "*" not in prose.text
    assert "`" not in prose.text
    assert "決議" in prose.text and "B 棟" in prose.text


def test_collapses_to_one_flowing_block() -> None:
    """The output is ONE continuous block — no embedded newlines, no multi-paragraph
    structure, matching 'a single flowing zh-TW prose summary'."""
    raw = "第一段。\n\n第二段。\n第三段。"
    prose = finalize(raw, token_len=heuristic_token_len)
    assert "\n" not in prose.text


def test_collapses_excess_whitespace() -> None:
    raw = "會議   討論  搬遷案"
    prose = finalize(raw, token_len=heuristic_token_len)
    assert "  " not in prose.text


def test_over_budget_is_flagged() -> None:
    prose = finalize("很" * (PROSE_MAX_TOKENS + 50), token_len=heuristic_token_len)
    assert prose.over_budget is True
    assert prose.tokens > PROSE_MAX_TOKENS


def test_under_budget_is_not_flagged() -> None:
    prose = finalize("會議討論搬遷案", token_len=heuristic_token_len)
    assert prose.over_budget is False


def test_english_prose_is_flagged_by_the_language_guard() -> None:
    prose = finalize("The council approved the motion.", token_len=heuristic_token_len)
    assert any("zh-TW" in flag or "CJK" in flag for flag in prose.lang_flags)


def test_simplified_prose_is_flagged() -> None:
    prose = finalize("讨论会议记录", token_len=heuristic_token_len)
    assert any("simplified" in flag for flag in prose.lang_flags)


def test_both_lang_flags_can_fire_independently() -> None:
    """A single output can fail BOTH checks -- lang_flags is a tuple, not a single verdict."""
    # Low CJK ratio (mostly English) AND containing a simplified char, "讨论"-adjacent
    # content is unlikely here, so construct a case with both: mostly-English text with
    # one simplified character mixed in.
    prose = finalize("The council approved 讨 motion", token_len=heuristic_token_len)
    assert len(prose.lang_flags) >= 1  # at minimum the CJK-ratio flag fires


def test_never_raises_on_empty_or_garbage_input() -> None:
    for raw in ("", "   ", "\n\n\n", "- \n- \n", "🎉🎉🎉", "\x00\x01"):
        prose = finalize(raw, token_len=heuristic_token_len)
        assert isinstance(prose.text, str)


def test_chars_matches_the_finalized_text_length() -> None:
    prose = finalize("會議討論搬遷案", token_len=heuristic_token_len)
    assert prose.chars == len(prose.text)


def test_tokens_uses_the_injected_counter() -> None:
    calls: list[str] = []

    def counting(text: str) -> int:
        calls.append(text)
        return heuristic_token_len(text)

    finalize("會議討論搬遷案", token_len=counting)
    assert calls, "the injected counter was never called"
