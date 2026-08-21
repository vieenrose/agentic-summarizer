"""Pins the zh-TW output-language guard — net-new (SPEC §8 risk 5 has a number attached:
the prior project measured 23.2% English leakage on real zh ASR input).

Both checks deliberately err toward NOT firing: a false positive silently drops a true
decision, which is worse here than an occasional miss.
"""

from __future__ import annotations

from arcsum.lang import (
    MIN_CJK_RATIO_POINT,
    MIN_CJK_RATIO_PROSE,
    check_zh_tw,
    cjk_ratio,
    simplified_hits,
)


def test_cjk_ratio_pure_zh_is_one() -> None:
    assert cjk_ratio("同意搬到辦公室") == 1.0


def test_cjk_ratio_pure_english_is_zero() -> None:
    assert cjk_ratio("the council approved the motion") == 0.0


def test_cjk_ratio_empty_text_is_one() -> None:
    """An empty string has no evidence of English leakage; a separate empty check is
    the caller's job, not this one's."""
    assert cjk_ratio("") == 1.0
    assert cjk_ratio("   ") == 1.0


def test_cjk_ratio_ignores_whitespace_in_the_denominator() -> None:
    assert cjk_ratio("同 意 搬 到") == cjk_ratio("同意搬到")


def test_cjk_ratio_mixed_text_is_between_zero_and_one() -> None:
    ratio = cjk_ratio("通過 Ordinance 第 123 號")
    assert 0.0 < ratio < 1.0


def test_cjk_ratio_counts_ordinance_ids_as_non_cjk() -> None:
    """A point that is mostly an ordinance ID legitimately has a low CJK ratio — this
    is exactly why MIN_CJK_RATIO_POINT is looser than MIN_CJK_RATIO_PROSE."""
    ratio = cjk_ratio("CB 118618")
    assert ratio == 0.0


def test_simplified_hits_detects_curated_characters() -> None:
    # "会议" (meeting) itself contains two curated simplified characters (会, 议), on
    # top of "讨论" (discuss) — all four are legitimately flagged.
    assert simplified_hits("讨论会议记录") == {"讨", "论", "会", "议"}


def test_simplified_hits_empty_on_traditional_text() -> None:
    assert simplified_hits("討論會議記錄") == set()


def test_simplified_hits_empty_on_pure_english() -> None:
    assert simplified_hits("the council approved") == set()


def test_simplified_detector_excludes_ambiguous_characters() -> None:
    """`于`/`后`/`划` are valid in BOTH scripts and are deliberately excluded — the
    errs-toward-not-firing doctrine."""
    assert simplified_hits("终于划船之后") & {"于", "后", "划"} == set()


def test_check_zh_tw_accepts_pure_zh_tw_prose() -> None:
    assert (
        check_zh_tw("會議討論辦公室搬遷，決議遷至 B 棟。", min_cjk_ratio=MIN_CJK_RATIO_PROSE)
        is None
    )


def test_check_zh_tw_refuses_english_prose() -> None:
    reason = check_zh_tw("The council approved the motion.", min_cjk_ratio=MIN_CJK_RATIO_PROSE)
    assert reason is not None
    assert "zh-TW" in reason


def test_check_zh_tw_refuses_simplified_prose() -> None:
    reason = check_zh_tw(
        "討論會議記錄後，通過決議。".replace("討論", "讨论"), min_cjk_ratio=MIN_CJK_RATIO_PROSE
    )
    assert reason is not None
    assert "simplified" in reason


def test_check_zh_tw_point_threshold_is_looser_than_prose() -> None:
    """A point built mostly of an ordinance ID passes at the point threshold (ratio
    ~0.38) but would fail the stricter prose threshold (0.70)."""
    text = "CB 118618 案通過表決"
    assert check_zh_tw(text, min_cjk_ratio=MIN_CJK_RATIO_POINT) is None
    assert check_zh_tw(text, min_cjk_ratio=MIN_CJK_RATIO_PROSE) is not None


def test_check_zh_tw_checks_language_before_script() -> None:
    """A low-CJK-ratio English string should report the ratio reason, not silently pass
    just because it happens to contain no simplified characters."""
    reason = check_zh_tw("approved", min_cjk_ratio=MIN_CJK_RATIO_PROSE)
    assert reason is not None
    assert "CJK ratio" in reason
