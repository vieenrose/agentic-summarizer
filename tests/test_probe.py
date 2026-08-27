"""Pins SPEC §5.2's G1 revision probe -- the cheap, synthetic, diagnostic check that
must pass before any corpus-scale evaluation runs.
"""

from __future__ import annotations

import pytest

from arcsum.chunker import CHUNK_TOKENS, iter_chunks
from arcsum.probe import ProbeResult, probe_meetings, score_probe
from arcsum.tokens import heuristic_token_len


def test_probe_meetings_returns_at_least_two_independent_scenarios() -> None:
    """A pass on one meeting alone would risk being an artifact of one particular pair
    of words rather than genuine revision-handling."""
    meetings = probe_meetings()
    assert len(meetings) >= 2
    assert len({m.name for m in meetings}) == len(meetings)  # names are distinct


def test_probe_meetings_each_have_early_and_late_decisions() -> None:
    for m in probe_meetings():
        assert m.early_decision
        assert m.late_decision
        assert m.early_decision != m.late_decision
        assert m.subject_terms
        assert m.distractor_terms


def test_stale_summary_fails_even_with_subject_variants() -> None:
    """`subject_terms` accepts alternative surface forms per concept (a real Phase-2
    false-negative fix). That widening must never make a STALE summary passable: the
    failure G1 exists to catch is the early decision standing alone as current, which
    `states_earlier_as_current` decides independently of any subject wording.
    """
    stale = {
        "office_move_reversal": "本次會議審議搬遷案，決議搬遷至 B 棟大樓的預算案已經通過。",
        "budget_approval_reversal": "本次會議審議下一季行銷預算案，兩百萬元經表決後核准通過。",
    }
    for m in probe_meetings():
        result = score_probe(stale[m.name], m)
        assert result.states_earlier_as_current is True
        assert result.passed is False


def test_probe_meetings_transcripts_are_nonempty() -> None:
    for m in probe_meetings():
        assert len(m.utterances) > 0


def test_probe_transcripts_span_multiple_chunks() -> None:
    """SPEC §5.2's G1 asks whether a LATER CHUNK can overturn an earlier conclusion. A
    single-chunk transcript cannot test that at all -- no memory crosses a step, `DROP`
    is never exercised, and the agent arm degenerates into a one-shot summariser.

    This is a real defect that shipped: the original probe transcripts were ~120 tokens
    against a 2500-token budget, so every G1 number measured up to 2026-08-27 was
    measuring the wrong mechanism entirely. Pinned here so it cannot return silently.
    """
    for m in probe_meetings():
        chunks = list(iter_chunks(m.utterances, budget=CHUNK_TOKENS, token_len=heuristic_token_len))
        assert len(chunks) > 1, (
            f"{m.name} fits in {len(chunks)} chunk(s) at budget={CHUNK_TOKENS}: the probe "
            "would not exercise cross-chunk revision at all"
        )


def test_probe_reversal_lands_in_a_later_chunk_than_the_decision() -> None:
    """Multi-chunk alone is not enough -- the planted decision and its reversal must fall
    in DIFFERENT chunks, or the revision still happens inside one step."""
    for m in probe_meetings():
        chunks = list(iter_chunks(m.utterances, budget=CHUNK_TOKENS, token_len=heuristic_token_len))

        # Chunk windows may overlap, so take the FIRST chunk containing each marker.
        def first_chunk_containing(word: str, chunks=chunks) -> int | None:
            for i, c in enumerate(chunks):
                if any(word in u.text for u in c.utterances):
                    return i
            return None

        dec = first_chunk_containing(m.early_decision)
        rev = first_chunk_containing(m.late_decision)
        assert dec is not None, f"{m.name}: early decision word never appears"
        assert rev is not None, f"{m.name}: late decision word never appears"
        assert rev > dec, (
            f"{m.name}: reversal first appears in chunk {rev}, decision in chunk {dec} -- "
            "the reversal must land in a LATER chunk for G1 to test revision"
        )


@pytest.fixture
def office_move():
    return next(m for m in probe_meetings() if m.name == "office_move_reversal")


def test_missing_space_around_a_latin_cjk_subject_term_still_passes(office_move) -> None:
    """Measured 2026-08-27: a correct, faithful summary scored a false FAIL because it
    rendered "B 棟" without the space ("B棟") -- spacing noise around a Latin-letter
    term, not a faithfulness difference."""
    prose_no_space = "本次會議討論辦公室搬遷案，市府建議撤回搬遷至B棟，維持原辦公室。"
    result = score_probe(prose_no_space, office_move)
    assert result.states_later is True
    assert result.passed is True


# --- score_probe: the correct summary passes ------------------------------------------


def test_correct_summary_passes(office_move) -> None:
    good_prose = "會議討論辦公室搬遷案，原先議決搬遷至 B 棟，但因故已撤回該決議，維持原辦公室。"
    result = score_probe(good_prose, office_move)
    assert result.states_later is True
    assert result.states_earlier_as_current is False
    assert result.distractor_absent is True
    assert result.passed is True


# --- score_probe: each failure mode individually ---------------------------------------


def test_stale_summary_fails_states_earlier_as_current(office_move) -> None:
    """The exact failure mode map-reduce cannot avoid: no later chunk can overturn an
    earlier window's independent summary."""
    stale_prose = "會議討論辦公室搬遷案，議決通過搬遷至 B 棟。"
    result = score_probe(stale_prose, office_move)
    assert result.states_earlier_as_current is True
    assert result.passed is False


def test_summary_that_omits_the_outcome_fails_states_later(office_move) -> None:
    """Mentioning the subject without ever stating its outcome is not a pass either."""
    vague_prose = "會議討論了辦公室搬遷案的相關細節。"
    result = score_probe(vague_prose, office_move)
    assert result.states_later is False
    assert result.passed is False


def test_summary_with_the_distractor_fails(office_move) -> None:
    prose_with_distractor = (
        "會議討論辦公室搬遷案，已撤回搬遷 B 棟的決議，另外會議中間也安排了十分鐘的休息時間。"
    )
    result = score_probe(prose_with_distractor, office_move)
    assert result.states_later is True
    assert result.states_earlier_as_current is False
    assert result.distractor_absent is False
    assert result.passed is False


# --- score_probe: the narrated-reversal case must NOT be penalised --------------------


def test_narrating_the_reversal_with_both_words_present_still_passes(office_move) -> None:
    """A summary saying 'originally approved, but later rescinded' legitimately
    contains BOTH the early and late words -- that must not be flagged as stale."""
    narrated = "辦公室搬遷案原本通過，確定搬遷至 B 棟，但後來因故撤回，維持現狀。"
    result = score_probe(narrated, office_move)
    assert result.states_later is True
    assert result.states_earlier_as_current is False
    assert result.passed is True


# --- score_probe: an appending (non-revising) agent fails ---------------------------


def test_appending_agent_fails_the_probe(office_move) -> None:
    """The scenario SPEC §5.2 exists to catch: a system that ADDs the reversal without
    ever superseding the earlier state, so both statements stand side by side with no
    resolution -- since the earlier decision's word appears without the later one
    resolving it in isolation, this must fail exactly like the stale case."""
    only_early = "辦公室搬遷案議決通過，確定遷至 B 棟。"
    result = score_probe(only_early, office_move)
    assert result.passed is False


def test_revising_agent_passes(office_move) -> None:
    revised = "辦公室搬遷案最終撤回，不再搬遷至 B 棟。"
    result = score_probe(revised, office_move)
    assert result.states_later is True
    assert result.states_earlier_as_current is False


# --- second scenario: different vocabulary, same mechanism ---------------------------


def test_budget_scenario_correct_summary_passes() -> None:
    budget = next(m for m in probe_meetings() if m.name == "budget_approval_reversal")
    good_prose = "討論行銷預算案，原核准兩百萬預算，但財務部發現錯誤後已駁回，需重新提案。"
    result = score_probe(good_prose, budget)
    assert result.passed is True


def test_budget_scenario_stale_summary_fails() -> None:
    budget = next(m for m in probe_meetings() if m.name == "budget_approval_reversal")
    stale = "討論行銷預算案，核准兩百萬預算用於宣傳。"
    result = score_probe(stale, budget)
    assert result.passed is False


# --- ProbeResult.passed is a pure AND of the three signals ----------------------------


@pytest.mark.parametrize(
    ("later", "earlier_current", "distractor_absent", "expected"),
    [
        (True, False, True, True),
        (False, False, True, False),
        (True, True, True, False),
        (True, False, False, False),
        (False, True, False, False),
    ],
)
def test_probe_result_passed_is_the_conjunction(
    later: bool, earlier_current: bool, distractor_absent: bool, expected: bool
) -> None:
    result = ProbeResult("x", later, earlier_current, distractor_absent)
    assert result.passed is expected
