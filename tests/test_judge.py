"""Judge evidence retrieval, verdict aggregation, and the budget guard.

No network: the client is stubbed. What is tested is the part that decides what a judge can
conclude — retrieval and aggregation — plus the guards that stop a runaway bill.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from judge import (  # noqa: E402
    DISQUALIFIED,
    PANEL,
    BulletVerdict,
    JudgeBudgetExceeded,
    Spend,
    TogetherJudge,
    _score_from,
    parse_bullets,
)

from voxsum.index import SNIPPET_CHARS, TranscriptIndex, tokenise
from voxsum.transcript import Utterance, parse_transcript

TRANSCRIPT = (
    "[0:00] S1: Next item is the vendor contract.\n"
    "[0:30] S2: I have circulated the details.\n"
    "[1:00] S1: For now we reject the vendor contract.\n"
    "[1:30] S2: I can rework the numbers.\n"
    "[2:00] S3: Can we also cover the coffee machine?\n"
    "[2:30] S1: No, out of scope today.\n"
    "[3:00] S1: With the reworked numbers, the vendor contract is approved.\n"
)

NOTES = (
    "TITLE: Vendor contract\n"
    "SUMMARY:\n- Vendor contract approved after rework [3:00]\n"
    "DECISIONS:\n- Vendor contract approved [3:00]\n"
    "ACTIONS:\n-\n"
    "OPEN:\n-\n"
    "TOPICS:\n- Vendor contract [0:00]\n"
)


@pytest.fixture
def index() -> TranscriptIndex:
    return TranscriptIndex(parse_transcript(TRANSCRIPT))


# --- tokenisation --------------------------------------------------------------

def test_cjk_uses_character_bigrams() -> None:
    # Word-splitting Chinese yields one giant token and every overlap score collapses.
    toks = tokenise("倉庫整併方案通過")
    assert "倉庫" in toks and "整併" in toks
    assert len(toks) > 3


def test_latin_uses_words() -> None:
    assert tokenise("Vendor contract approved") == {"vendor", "contract", "approved"}


def test_mixed_script_gets_both() -> None:
    toks = tokenise("vendor 廠商合約 approved")
    assert "vendor" in toks and "廠商" in toks


# --- snippet extraction --------------------------------------------------------

def test_snippet_is_taken_at_the_best_matching_window() -> None:
    """A 2.6k-char monologue must not be truncated from the start.

    VCSum zh lines run this long; a head-truncated snippet would hand the judge the opening
    pleasantries of a monologue whose decision lands 2k chars later.
    """
    filler = "we discussed many procedural matters at length. " * 60
    line = filler + "the vendor contract is hereby approved. " + filler
    idx = TranscriptIndex([Utterance(0, "S1", line)])
    snippet = idx.snippet(0, "vendor contract approved")
    assert "vendor contract" in snippet
    assert len(snippet) <= SNIPPET_CHARS


def test_short_line_is_returned_whole(index: TranscriptIndex) -> None:
    assert index.snippet(0, "vendor") == "Next item is the vendor contract."


# --- evidence modes ------------------------------------------------------------

def test_anchor_mode_is_neighbourhood_only(index: TranscriptIndex) -> None:
    ev = index.evidence_for("Vendor contract approved", 180, mode="anchor")
    assert ev and all(e.from_anchor_neighbourhood for e in ev)
    assert all(abs(e.anchor - 180) <= 3 * 30 for e in ev)


def test_claim_mode_searches_the_whole_transcript(index: TranscriptIndex) -> None:
    """A true claim anchored at the wrong line must still be findable (§7.1)."""
    ev = index.evidence_for("Vendor contract approved", 0, mode="claim")
    assert any(e.anchor == 180 for e in ev), "the approval line must be retrievable"
    assert any(not e.from_anchor_neighbourhood for e in ev)


def test_claim_mode_puts_neighbourhood_first(index: TranscriptIndex) -> None:
    ev = index.evidence_for("Vendor contract approved", 180, mode="claim")
    assert ev[0].from_anchor_neighbourhood


def test_anchor_mode_with_a_bad_anchor_returns_nothing(index: TranscriptIndex) -> None:
    # A bullet anchored to a non-existent line *should* fail FAITH-anchor.
    assert index.evidence_for("anything", 9999, mode="anchor") == []


def test_evidence_is_capped(index: TranscriptIndex) -> None:
    assert len(index.evidence_for("vendor contract", 180, mode="claim", limit=6)) <= 6


def test_unknown_mode_rejected(index: TranscriptIndex) -> None:
    with pytest.raises(ValueError, match="unknown evidence mode"):
        index.evidence_for("x", 0, mode="vibes")


def test_search_excludes_requested_anchors(index: TranscriptIndex) -> None:
    hits = index.search("vendor contract", exclude={0})
    assert all(e.anchor != 0 for e in hits)


# --- notes parsing -------------------------------------------------------------

def test_parse_bullets_extracts_section_text_and_anchor() -> None:
    bullets = parse_bullets(NOTES)
    assert ("DECISIONS", "Vendor contract approved", 180) in bullets
    # TITLE is not a bullet section and empty sections contribute nothing.
    assert all(s != "TITLE" for s, _, _ in bullets)
    assert len(bullets) == 3


def test_parse_bullets_tolerates_a_missing_anchor() -> None:
    (only,) = parse_bullets("DECISIONS:\n- Something with no anchor\n")
    assert only == ("DECISIONS", "Something with no anchor", None)


def test_parse_bullets_ignores_unparseable_clock() -> None:
    (only,) = parse_bullets("OPEN:\n- Bad clock [99:99]\n")
    assert only[2] is None


# --- verdict aggregation -------------------------------------------------------

def test_majority_of_three_families() -> None:
    v = BulletVerdict("DECISIONS", "x", 0, "claim")
    v.verdicts = {"a": "SUPPORTED", "b": "SUPPORTED", "c": "CONTRADICTED"}
    assert v.majority == "SUPPORTED"


def test_tie_falls_to_the_most_severe() -> None:
    """0% inversions is a product requirement — a split must not average it away."""
    v = BulletVerdict("DECISIONS", "x", 0, "claim")
    v.verdicts = {"a": "SUPPORTED", "b": "CONTRADICTED"}
    assert v.majority == "CONTRADICTED"

    v.verdicts = {"a": "SUPPORTED", "b": "UNSUPPORTED"}
    assert v.majority == "UNSUPPORTED"


def test_no_verdicts_is_missing_not_supported() -> None:
    assert BulletVerdict("DECISIONS", "x", 0, "claim").majority == "MISSING"


# --- score parsing -------------------------------------------------------------

def test_last_match_wins_per_key() -> None:
    """§7.2: a judge that restates then decides must be read at its final answer."""
    text = "Draft: COVER: 2\nSYNTH: 2\nOn reflection:\nCOVER: 4\nSYNTH: 5"
    assert _score_from(text) == (4, 5)


def test_missing_scores_are_none() -> None:
    assert _score_from("I cannot score this.") == (None, None)


def test_out_of_range_scores_are_not_accepted() -> None:
    assert _score_from("COVER: 9\nSYNTH: 0") == (None, None)


# --- budget and safety guards --------------------------------------------------

def test_spend_accounting_uses_the_price_table() -> None:
    spend = Spend()
    spend.add("openai/gpt-oss-20b", 1_000_000, 1_000_000)
    assert spend.usd == pytest.approx(0.25)  # 0.05 in + 0.20 out
    spend.add("Prism-ML/Ternary-Bonsai-27B", 1_000_000, 1_000_000)
    assert spend.usd == pytest.approx(0.25), "the free judge must cost nothing"


def test_budget_guard_stops_before_calling() -> None:
    client = TogetherJudge(api_key="x", budget_usd=0.01)
    client.spend.add("openai/gpt-oss-20b", 1_000_000, 1_000_000)  # $0.25, over budget
    with pytest.raises(JudgeBudgetExceeded, match="stopping"):
        client("openai/gpt-oss-20b", "sys", "user")


def test_disqualified_judge_is_refused() -> None:
    """gemma-3n-E4B answered SUPPORTED to a planted inversion in 4 tokens."""
    client = TogetherJudge(api_key="x", budget_usd=1.0)
    with pytest.raises(ValueError, match="planted-inversion probe"):
        client("google/gemma-3n-E4B-it", "sys", "user")
    assert "google/gemma-4-E4B-it" in DISQUALIFIED


def test_missing_api_key_fails_fast() -> None:
    with pytest.raises(SystemExit, match="TOGETHER_API_KEY"):
        TogetherJudge(api_key="")


def test_panel_is_three_distinct_families() -> None:
    models = set(PANEL.values())
    assert len(models) == 3
    # None of them may be the student's or the teacher's family (both Gemma).
    assert not any("gemma" in m.lower() for m in models)


def test_claim_mode_reserves_slots_for_retrieved_evidence(index: TranscriptIndex) -> None:
    """Regression: the neighbourhood used to consume every slot.

    A +/-3 neighbourhood yields up to 7 lines, so `(near + found)[:6]` discarded the
    whole-transcript search entirely and made claim mode identical to anchor mode.
    """
    ev = index.evidence_for("Vendor contract approved", 0, mode="claim", limit=6)
    # This transcript has only one line beyond the neighbourhood that matches lexically, so
    # the budget cannot fill; what matters is that the retrieved slot survives at all.
    assert any(not e.from_anchor_neighbourhood for e in ev), "search results were discarded"
    assert any(e.anchor == 180 for e in ev), "the approval line is 6 lines from the anchor"


def test_claim_mode_caps_the_neighbourhood_when_search_has_hits() -> None:
    """With enough lexical matches, the neighbourhood gets at most half the budget.

    Backfill may exceed that when the search comes up short (see the test below), so the
    cap is only observable on a transcript where retrieval can actually fill its half.
    """
    lines = [Utterance(i * 30, "S1", f"the vendor contract was discussed in round {i}")
             for i in range(40)]
    idx = TranscriptIndex(lines)
    ev = idx.evidence_for("vendor contract discussed", 0, mode="claim", limit=6)
    assert len(ev) == 6
    assert sum(1 for e in ev if e.from_anchor_neighbourhood) == 3
    assert sum(1 for e in ev if not e.from_anchor_neighbourhood) == 3


def test_claim_mode_still_leads_with_the_neighbourhood(index: TranscriptIndex) -> None:
    ev = index.evidence_for("Vendor contract approved", 180, mode="claim", limit=6)
    assert ev[0].from_anchor_neighbourhood


def test_claim_mode_reaches_support_at_anchor_plus_two(index: TranscriptIndex) -> None:
    """A support line at anchor +/-2 or +/-3 must not be dropped from claim mode.

    `near[:3]` is anchor-first (anchor, then +/-1), so a bullet whose true support sits at
    anchor +/-2 or +/-3 is outside the shown slice but still inside FAITH-anchor's +/-3
    window. Excluding the WHOLE neighbourhood from the search would make that line invisible
    to claim mode entirely; only the lines actually shown may be excluded.
    """
    lines = [Utterance(i * 30, "S1", f"the vendor contract was discussed in round {i}")
             for i in range(20)]
    lines[5] = Utterance(150, "S1", "Vendor contract approved for the new supplier arrangement")
    idx = TranscriptIndex(lines)
    # anchor at 60 (idx 2); the approval line at 150 is anchor+3 (idx 5).
    ev = idx.evidence_for("Vendor contract approved", 60, mode="claim", limit=6)
    assert any(e.anchor == 150 for e in ev), "support at anchor+3 was dropped by the split"


def test_claim_mode_backfills_when_one_source_is_short(index: TranscriptIndex) -> None:
    # A bullet with no lexical match anywhere: the neighbourhood must still fill the budget.
    ev = index.evidence_for("zzzz qqqq unmatchable", 90, mode="claim", limit=6)
    assert len(ev) == 6 and all(e.from_anchor_neighbourhood for e in ev)


def test_anchor_mode_is_unchanged_by_the_fix(index: TranscriptIndex) -> None:
    ev = index.evidence_for("Vendor contract approved", 180, mode="anchor", limit=6)
    assert ev and all(e.from_anchor_neighbourhood for e in ev)


def test_anchor_mode_leads_with_the_anchor_line(index: TranscriptIndex) -> None:
    # FAITH-anchor asks "does the anchored line support it", so the anchor line itself
    # must be the first snippet the judge reads.
    ev = index.evidence_for("Vendor contract approved", 180, mode="anchor", limit=6)
    assert ev[0].anchor == 180


def test_claim_mode_keeps_the_anchor_line_when_search_has_hits() -> None:
    """The anchor line must survive the claim-mode budget split.

    The anchor line states the claim, and many other lines also match "approved", so the
    whole-transcript search fills its half of the budget with no backfill. If the budget
    took `near[:3]` on an ascending neighbourhood, the anchor line (position 3 of 7) would
    be dropped and a correctly-anchored bullet would read as unsupported in claim mode.
    """
    lines = [
        Utterance(i * 30, "S1", f"the committee reviewed budgets and approved item {i}")
        for i in range(20)
    ]
    lines[5] = Utterance(150, "S1", "Vendor contract approved for the new supplier arrangement")
    idx = TranscriptIndex(lines)
    ev = idx.evidence_for("Vendor contract approved", 150, mode="claim", limit=6)
    assert len(ev) == 6
    assert ev[0].anchor == 150, "the anchor line should lead the claim-mode evidence"
    assert any(e.anchor == 150 for e in ev), "the anchor line was dropped by the split"
    # Retrieval still gets its reserved share; this is the separation §7.1 exists to draw.
    assert any(not e.from_anchor_neighbourhood for e in ev)


def test_evidence_order_is_pinned() -> None:
    """Order is a measured variance source (0.60 FAITH, 30% verdict flips), not a style choice.

    Pinning it in code means a comparison cannot silently measure presentation instead of the
    systems. Changing EVIDENCE_ORDER invalidates comparison with previously recorded numbers.
    """
    from voxsum.index import EVIDENCE_ORDER

    assert EVIDENCE_ORDER == "anchor_first"


def test_evidence_order_is_deterministic(index: TranscriptIndex) -> None:
    a = index.evidence_for("Vendor contract approved", 180, mode="claim")
    b = index.evidence_for("Vendor contract approved", 180, mode="claim")
    assert [e.anchor for e in a] == [e.anchor for e in b]
