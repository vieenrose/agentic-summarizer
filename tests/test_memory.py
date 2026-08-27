"""Pins SPEC §4.1's ARC+POINTS memory contract.

Guards a genuine gap in the prior project: its caps were count-only, never token-length.
Here an over-length ARC or point is REFUSED, never truncated — SPEC §4.2's replay rule is
"never half-applied into the corpus", and a truncated write is exactly that.
"""

from __future__ import annotations

from arcsum.memory import (
    ARC_TOKENS,
    MIN_PREFIX_TOKENS,
    POINT_TOKENS,
    POINTS_CAP,
    Memory,
    Point,
    normalize,
    spread,
)
from arcsum.render import render_memory
from arcsum.tokens import heuristic_token_len

# --- spread(): ported verbatim from the prior project; only the cap changes -----------


def test_spread_is_identity_under_cap() -> None:
    items = list(range(5))
    assert spread(items, 10) == items


def test_spread_keeps_endpoints_never_head_truncates() -> None:
    """SPEC §4.1: "evenly spread, never head-truncated ... drops the end of the meeting,
    where decisions land"."""
    items = list(range(20))
    result = spread(items, 5)
    assert result[0] == 0
    assert result[-1] == 19
    assert result == sorted(result)


def test_spread_cap_one_keeps_the_latest() -> None:
    assert spread(list(range(10)), 1) == [9]


def test_spread_cap_zero_is_empty() -> None:
    assert spread(list(range(10)), 0) == []


def test_spread_returns_distinct_items_even_on_rounding_collisions() -> None:
    for n in range(2, 30):
        for cap in range(1, n + 1):
            result = spread(list(range(n)), cap)
            assert len(result) == min(n, cap)
            assert len(set(result)) == len(result)


# --- normalize() ------------------------------------------------------------------


def test_normalize_folds_whitespace_and_case() -> None:
    assert normalize("  Hello   World  ") == normalize("hello world")


def test_normalize_folds_cjk_punctuation() -> None:
    """Whitespace-free zh has nothing for '.join(split())' to collapse; punctuation
    folding is what makes two points differing only in punctuation dedup."""
    assert normalize("通過議案，並公告。") == normalize("通過議案並公告")


# --- Memory.is_empty -----------------------------------------------------------------


def test_is_empty_true_on_a_fresh_memory() -> None:
    assert Memory().is_empty() is True


def test_is_empty_false_with_only_an_arc() -> None:
    """STRICT emptiness (both slots), not a 'thin memory' heuristic -- an arc alone is
    real information. `agent.synthesize_memory` keys a hard behavioural guarantee off
    this, so the boundary must be unambiguous."""
    m = Memory()
    m.set_arc("會議討論辦公室搬遷案。")
    assert m.is_empty() is False


def test_is_empty_false_with_only_a_point() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟", chunk=0)
    assert m.is_empty() is False


def test_is_empty_true_again_after_the_only_point_is_dropped() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟", chunk=0)
    m.drop_point("同意搬到")
    assert m.is_empty() is True


# --- Memory.set_arc ------------------------------------------------------------------


def test_set_arc_accepts_text_within_budget() -> None:
    m = Memory()
    assert m.set_arc("會議討論辦公室搬遷案。") is None
    assert m.arc == "會議討論辦公室搬遷案。"


def test_set_arc_refuses_empty() -> None:
    m = Memory()
    assert m.set_arc("   ") == "empty arc"
    assert m.arc == ""


def test_overlong_arc_is_refused_not_truncated() -> None:
    m = Memory()
    m.set_arc("原始摘要")
    reason = m.set_arc("很" * (ARC_TOKENS + 10))
    assert reason is not None
    assert "too long" in reason
    assert m.arc == "原始摘要"  # refusal leaves the previous arc standing, not a truncation


def test_set_arc_collapses_internal_whitespace() -> None:
    m = Memory()
    m.set_arc("會議  討論\n搬遷")
    assert m.arc == "會議 討論 搬遷"


# --- Memory.add_point ------------------------------------------------------------------


def test_add_point_accepts_text_within_budget() -> None:
    m = Memory()
    assert m.add_point("同意搬到 B 棟", chunk=0) is None
    assert [p.text for p in m.points] == ["同意搬到 B 棟"]


def test_add_point_refuses_empty() -> None:
    m = Memory()
    assert m.add_point("", chunk=0) == "empty point"
    assert m.points == []


def test_overlong_point_is_refused_not_truncated() -> None:
    m = Memory()
    reason = m.add_point("很" * (POINT_TOKENS + 5), chunk=0)
    assert reason is not None
    assert "too long" in reason
    assert m.points == []


def test_add_point_refuses_exact_duplicate() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟", chunk=0)
    reason = m.add_point("同意搬到 B 棟", chunk=1)
    assert reason == "duplicate point"
    assert len(m.points) == 1


def test_add_point_refuses_duplicate_up_to_punctuation_and_whitespace() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟", chunk=0)
    assert m.add_point("同意搬到 B 棟。", chunk=1) == "duplicate point"


def test_points_cap_is_sixteen() -> None:
    assert POINTS_CAP == 16


# --- Memory.find / drop_point ------------------------------------------------------


def test_find_matches_a_unique_prefix() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟大樓", chunk=0)
    assert m.find("同意搬到") == 0


def test_find_refuses_short_prefix_below_the_token_floor() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟大樓", chunk=0)
    # Two ideographs is below MIN_PREFIX_TOKENS=4 — refuse rather than risk mismatching.
    assert m.find("同意") is None


def test_prefix_floor_is_measured_in_char_tokens_not_characters() -> None:
    """4 char_tokens means 4 ideographs in zh but can be 4 latin WORDS in embedded latin —
    the point of expressing the floor this way rather than as a raw character count."""
    assert MIN_PREFIX_TOKENS == 4
    m = Memory()
    m.add_point("Council approved the budget plan", chunk=0)
    assert m.find("Council approved the budget") == 0


def test_find_refuses_ambiguous_prefix() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟", chunk=0)
    m.add_point("同意搬到 C 棟", chunk=1)
    assert m.find("同意搬到") is None


def test_drop_point_removes_the_uniquely_matched_point() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟大樓", chunk=0)
    m.add_point("預算通過", chunk=0)
    assert m.drop_point("同意搬到") is None
    assert [p.text for p in m.points] == ["預算通過"]


def test_drop_point_refuses_when_nothing_matches() -> None:
    m = Memory()
    m.add_point("預算通過", chunk=0)
    reason = m.drop_point("完全不相關的字串")
    assert reason == "prefix did not match exactly one point"
    assert len(m.points) == 1


def test_drop_point_refuses_ambiguous_prefix_and_leaves_memory_unchanged() -> None:
    m = Memory()
    m.add_point("同意搬到 B 棟", chunk=0)
    m.add_point("同意搬到 C 棟", chunk=1)
    reason = m.drop_point("同意搬到")
    assert reason == "prefix did not match exactly one point"
    assert len(m.points) == 2


# --- caps / enforce_caps -------------------------------------------------------------


def test_enforce_caps_spreads_overflowing_points() -> None:
    m = Memory()
    for i in range(20):
        m.add_point(f"第{i}項決議內容", chunk=i)
    m.enforce_caps()
    assert len(m.points) == POINTS_CAP
    assert m.points[0].text == "第0項決議內容"
    assert m.points[-1].text == "第19項決議內容"


def test_enforce_caps_is_idempotent() -> None:
    m = Memory()
    for i in range(20):
        m.add_point(f"第{i}項決議內容", chunk=i)
    m.enforce_caps()
    once = [p.text for p in m.points]
    m.enforce_caps()
    assert [p.text for p in m.points] == once


def test_enforce_caps_is_a_noop_under_cap() -> None:
    m = Memory()
    m.add_point("唯一的一項", chunk=0)
    m.enforce_caps()
    assert len(m.points) == 1


# --- token_len injection ---------------------------------------------------------------


def test_memory_uses_the_injected_tokenizer_for_caps() -> None:
    """A caller passing the real MiniCPM5 tokenizer must have its caps measured by it,
    not by a hidden default -- the prior project's exact class of latent divergence."""

    def generous(_text: str) -> int:
        return 1  # everything is "1 token" under this counter

    m = Memory(token_len=generous)
    assert m.set_arc("很" * 500) is None  # would be refused under the real counter
    assert m.add_point("很" * 500, chunk=0) is None


def test_default_token_len_is_the_heuristic() -> None:
    m = Memory()
    assert m.token_len is heuristic_token_len


def test_clone_carries_the_injected_tokenizer() -> None:
    def custom(text: str) -> int:
        return len(text)

    m = Memory(token_len=custom)
    m.add_point("x", chunk=0)
    cloned = m.clone()
    assert cloned.token_len is custom
    assert cloned.points == m.points
    # Mutating the clone must not mutate the original.
    cloned.points.append(Point("y", chunk=1))
    assert len(m.points) == 1


def test_clone_is_independent_of_the_original_points_list() -> None:
    m = Memory()
    m.add_point("a", chunk=0)
    cloned = m.clone()
    cloned.add_point("b", chunk=1)
    assert [p.text for p in m.points] == ["a"]
    assert [p.text for p in cloned.points] == ["a", "b"]


def test_prompt_tokens_reflects_the_rendered_memory() -> None:
    m = Memory()
    m.set_arc("會議摘要")
    m.add_point("一項決議", chunk=0)
    assert m.prompt_tokens() > 0
    assert m.prompt_tokens() == m.token_len(render_memory(m))
