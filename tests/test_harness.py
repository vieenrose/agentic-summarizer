"""Harness conformance — STATE, ops, guards, chunker, render (CLAUDE.md §3-§6).

Everything here runs against a scripted fake model (a list of op strings), so §6 is
provable with no GPU and no model weights.
"""

from __future__ import annotations

import pytest

from voxsum.chunker import Chunk, heuristic_token_len, iter_chunks
from voxsum.guards import NOP_COLLAPSE_K, apply_ops, match_anchor
from voxsum.ops import Add, Cmp, Del, Malformed, Nop, Title, Upd, parse_ops, render_op
from voxsum.render import render_state
from voxsum.state import CAPS, Bullet, NotesState, spread
from voxsum.transcript import Utterance, clock_to_sec, parse_transcript

EXAMPLE = (
    "[0:00] S1: Let us discuss the office move.\n"
    "[2:30] S2: I propose we move to Building B.\n"
    "[5:10] S1: Agreed, Building B it is.\n"
    "[6:02] S2: I will circulate the move checklist by Friday.\n"
    "[7:40] S1: Parking allocation is still open.\n"
)


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(0, tuple(parse_transcript(EXAMPLE)))


@pytest.fixture
def state() -> NotesState:
    return NotesState()


# --- spread() ------------------------------------------------------------------

def test_spread_keeps_endpoints_never_head_truncates() -> None:
    items = list(range(10))
    got = spread(items, 4)
    assert len(got) == 4
    assert got[0] == 0 and got[-1] == 9, "endpoints must survive"
    assert got == sorted(got), "order preserved"


def test_spread_is_identity_under_cap() -> None:
    assert spread([1, 2, 3], 5) == [1, 2, 3]


def test_spread_cap_one_keeps_the_latest() -> None:
    # The end of a meeting is where decisions land; cap-1 must not keep the opening.
    assert spread([1, 2, 3], 1) == [3]


@pytest.mark.parametrize("n,cap", [(7, 3), (10, 9), (100, 6), (5, 4), (3, 2)])
def test_spread_returns_distinct_items(n: int, cap: int) -> None:
    got = spread(list(range(n)), cap)
    assert len(got) == cap == len(set(got))


# --- NOTES v2 rendering --------------------------------------------------------

def test_render_all_sections_present_and_ordered(state: NotesState) -> None:
    out = render_state(state)
    assert out.splitlines() == [
        "TITLE:",
        "SUMMARY:", "-",
        "DECISIONS:", "-",
        "ACTIONS:", "-",
        "OPEN:", "-",
        "TOPICS:", "-",
    ]


def test_render_matches_spec_example(state: NotesState) -> None:
    state.set_title("Office move decision")
    state.add("SUMMARY", "Move to Building B agreed after discussion", 310)
    state.add("DECISIONS", "Relocate the office to Building B", 310)
    state.add("ACTIONS", "S2: circulate the move checklist (due: Friday)", 362)
    state.add("OPEN", "Parking allocation for Building B", 460)
    state.add("TOPICS", "Office move", 0)
    assert render_state(state) == (
        "TITLE: Office move decision\n"
        "SUMMARY:\n- Move to Building B agreed after discussion [5:10]\n"
        "DECISIONS:\n- Relocate the office to Building B [5:10]\n"
        "ACTIONS:\n- S2: circulate the move checklist (due: Friday) [6:02]\n"
        "OPEN:\n- Parking allocation for Building B [7:40]\n"
        "TOPICS:\n- Office move [0:00]\n"
    )


def test_render_enforces_caps(state: NotesState) -> None:
    for i in range(CAPS["SUMMARY"] + 4):
        state.add("SUMMARY", f"bullet {i}", i)
    assert render_state(state).count("- bullet") == CAPS["SUMMARY"]


# --- STATE mutations -----------------------------------------------------------

def test_upd_keeps_slot_order(state: NotesState) -> None:
    state.add("DECISIONS", "First decision", 0)
    state.add("DECISIONS", "Second decision", 60)
    assert state.update("DECISIONS", "First de", "First decision revised", 120) is None
    # Revising must not reorder the timeline.
    assert [b.text for b in state.bullets("DECISIONS")] == [
        "First decision revised",
        "Second decision",
    ]


def test_prefix_must_be_unambiguous(state: NotesState) -> None:
    state.add("OPEN", "Parking allocation east", 0)
    state.add("OPEN", "Parking allocation west", 60)
    # Ambiguity is a refusal — editing the wrong bullet is how inversions happen.
    assert state.update("OPEN", "Parking", "merged", 120) is not None
    assert state.find("OPEN", "Parking") is None
    assert state.find("OPEN", "Parking allocation e") == 0


def test_prefix_below_minimum_never_matches(state: NotesState) -> None:
    state.add("TOPICS", "Budget", 0)
    assert state.find("TOPICS", "Bud") is None  # < 6 chars


def test_duplicate_bullets_refused(state: NotesState) -> None:
    assert state.add("TOPICS", "Office move", 0) is None
    assert state.add("TOPICS", "  office   MOVE ", 60) == "duplicate bullet"


def test_delete_removes_matched_bullet(state: NotesState) -> None:
    state.add("OPEN", "Parking allocation", 0)
    assert state.delete("OPEN", "Parking allocation") is None
    assert state.bullets("OPEN") == []


def test_compact_truncates_to_cap(state: NotesState) -> None:
    bullets = [Bullet(f"rewritten {i}", i) for i in range(CAPS["TOPICS"] + 3)]
    assert state.compact("TOPICS", bullets) is None
    assert len(state.bullets("TOPICS")) == CAPS["TOPICS"]


# --- op parsing ----------------------------------------------------------------

def test_parse_text_grammar_example() -> None:
    ops = parse_ops(
        "ADD DECISIONS - Budget increase approved at 10% [32:14]\n"
        "UPD SUMMARY «Budget increase» -> Budget increase approved at 10% "
        "after CFO revision [32:14]\n"
        "NOP"
    )
    assert isinstance(ops[0], Add)
    assert ops[0].section == "DECISIONS" and ops[0].anchor == clock_to_sec("32:14")
    assert ops[0].bullet == "Budget increase approved at 10%"
    assert isinstance(ops[1], Upd) and ops[1].prefix == "Budget increase"
    assert isinstance(ops[2], Nop)


def test_parse_del_and_cmp() -> None:
    ops = parse_ops("DEL OPEN «Parking allocation»\nCMP TOPICS\n- one [0:00]\n- two [1:00]")
    assert isinstance(ops[0], Del) and ops[0].prefix == "Parking allocation"
    assert isinstance(ops[1], Cmp) and len(ops[1].bullets) == 2
    assert ops[1].bullets[1].anchor == 60


def test_parse_functiongemma_call_format() -> None:
    raw = (
        "<start_function_call>call:ADD{section:<escape>DECISIONS<escape>,"
        "bullet:<escape>Relocate to Building B<escape>,"
        "anchor:<escape>5:10<escape>}<end_function_call>"
    )
    (op,) = parse_ops(raw)
    assert isinstance(op, Add)
    assert (op.section, op.bullet, op.anchor) == ("DECISIONS", "Relocate to Building B", 310)


def test_parse_functiongemma_upd_and_nop() -> None:
    raw = (
        "<start_function_call>call:UPD{section:<escape>SUMMARY<escape>,"
        "prefix:<escape>Budget increase<escape>,bullet:<escape>Approved after revision<escape>,"
        "anchor:<escape>32:14<escape>}<end_function_call>\n"
        "<start_function_call>call:NOP{}<end_function_call>"
    )
    ops = parse_ops(raw)
    assert isinstance(ops[0], Upd) and ops[0].prefix == "Budget increase"
    assert isinstance(ops[1], Nop)


def test_malformed_never_raises_and_is_recorded() -> None:
    ops = parse_ops("ADD NOSUCHSECTION - hello [0:00]\nplease summarise the meeting\nADD")
    assert all(isinstance(o, Malformed) for o in ops)
    assert "unknown section" in ops[0].reason


def test_parse_empty_output() -> None:
    assert parse_ops("") == []


def test_render_op_round_trips() -> None:
    for line in (
        "ADD DECISIONS - Budget approved [32:14]",
        "UPD SUMMARY «Budget increase» -> Approved after revision [32:14]",
        "DEL OPEN «Parking allocation»",
        "NOP",
    ):
        (op,) = parse_ops(line)
        assert render_op(op) == line


# --- guards --------------------------------------------------------------------

def test_anchor_outside_chunk_falls_to_matcher(state: NotesState, chunk: Chunk) -> None:
    ops = parse_ops("ADD DECISIONS - Relocate the office to Building B [99:99]")
    outcome = apply_ops(state, ops, chunk)
    # The op still lands, the anchor resolves to a real chunk line, and the substitution
    # is logged. Which line the matcher picks is its business — that it is real is ours.
    assert outcome.applied == 1
    bullet = state.bullets("DECISIONS")[0]
    assert chunk.has_line(bullet.anchor)
    assert "99:99" not in bullet.text, "an unparseable clock must not leak into the notes"
    assert "matcher" in outcome.results[0].reason


def test_valid_anchor_is_preserved(state: NotesState, chunk: Chunk) -> None:
    apply_ops(state, parse_ops("ADD OPEN - Parking allocation [7:40]"), chunk)
    assert state.bullets("OPEN")[0].anchor == 460


def test_missing_anchor_is_matched(state: NotesState, chunk: Chunk) -> None:
    apply_ops(state, parse_ops("ADD ACTIONS - circulate the move checklist"), chunk)
    assert state.bullets("ACTIONS")[0].anchor == 362


def test_temporal_guard_blocks_inversion(state: NotesState, chunk: Chunk) -> None:
    # Later bullet says approved; a new bullet claiming rejected is an inversion.
    state.add("DECISIONS", "Building B move approved", 310)
    ops = parse_ops("ADD DECISIONS - Building B move rejected [0:00]")
    outcome = apply_ops(state, ops, chunk)
    assert outcome.applied == 0
    assert "temporal guard" in outcome.results[0].reason
    assert len(state.bullets("DECISIONS")) == 1


def test_temporal_guard_allows_revision_via_upd(state: NotesState, chunk: Chunk) -> None:
    # The decision chain rejected -> approved is exactly what UPD is for.
    state.add("DECISIONS", "Building B move rejected", 0)
    outcome = apply_ops(
        state,
        parse_ops("UPD DECISIONS «Building B move» -> Building B move approved [5:10]"),
        chunk,
    )
    assert outcome.applied == 1
    assert state.bullets("DECISIONS")[0].text == "Building B move approved"


def test_temporal_guard_ignores_unrelated_subjects(state: NotesState, chunk: Chunk) -> None:
    state.add("DECISIONS", "Catering contract approved", 310)
    outcome = apply_ops(state, parse_ops("ADD DECISIONS - Parking plan rejected [0:00]"), chunk)
    assert outcome.applied == 1


def test_temporal_guard_only_covers_decisions_and_actions(state: NotesState, chunk: Chunk) -> None:
    state.add("SUMMARY", "Building B move approved", 310)
    outcome = apply_ops(state, parse_ops("ADD SUMMARY - Building B move rejected [0:00]"), chunk)
    assert outcome.applied == 1, "SUMMARY is not timeline-guarded"


def test_ops_apply_in_emission_order(state: NotesState, chunk: Chunk) -> None:
    outcome = apply_ops(
        state,
        parse_ops(
            "ADD DECISIONS - Move to Building B [2:30]\n"
            "UPD DECISIONS «Move to» -> Move to Building B confirmed [5:10]"
        ),
        chunk,
    )
    assert outcome.applied == 2
    assert state.bullets("DECISIONS")[0].text == "Move to Building B confirmed"


def test_malformed_op_is_not_fatal(state: NotesState, chunk: Chunk) -> None:
    outcome = apply_ops(state, parse_ops("garbage line\nADD TOPICS - Office move [0:00]"), chunk)
    assert outcome.applied == 1 and len(outcome.malformed) == 1


def test_valid_op_rate_excludes_nop(state: NotesState, chunk: Chunk) -> None:
    outcome = apply_ops(state, parse_ops("NOP\nADD TOPICS - Office move [0:00]"), chunk)
    assert outcome.valid_op_rate == 1.0


def test_nop_collapse_trips_after_k_content_rich_chunks(chunk: Chunk) -> None:
    rich = Chunk(0, tuple(parse_transcript(EXAMPLE * 6)))
    assert rich.is_content_rich()
    state = NotesState()
    assert not apply_ops(state, parse_ops("NOP"), rich, consecutive_nops=0).nop_collapse
    assert apply_ops(
        state, parse_ops("NOP"), rich, consecutive_nops=NOP_COLLAPSE_K - 1
    ).nop_collapse


def test_nop_on_thin_chunk_is_not_collapse() -> None:
    thin = Chunk(0, (Utterance(0, "S1", "mm-hm"), Utterance(3, "S2", "right")))
    assert not thin.is_content_rich()
    outcome = apply_ops(NotesState(), parse_ops("NOP"), thin, consecutive_nops=99)
    assert not outcome.nop_collapse, "an empty chunk deserves NOP"


def test_title_op_sets_title(state: NotesState, chunk: Chunk) -> None:
    outcome = apply_ops(state, parse_ops("TITLE: Office move decision"), chunk)
    assert outcome.applied == 1 and state.title == "Office move decision"
    assert isinstance(parse_ops("TITLE: x")[0], Title)


def test_match_anchor_prefers_lexical_overlap(chunk: Chunk) -> None:
    assert match_anchor(chunk, "parking allocation") == 460
    assert match_anchor(chunk, "循環 checklist Friday") == 362


def test_match_anchor_on_empty_chunk() -> None:
    assert match_anchor(Chunk(0, ()), "anything") is None


# --- chunker -------------------------------------------------------------------

def test_chunks_are_contiguous_with_overlap() -> None:
    lines = [Utterance(i * 10, "S1", f"utterance number {i} " * 12) for i in range(60)]
    chunks = list(iter_chunks(lines, budget=256, overlap=2))
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.utterances[0].start <= prev.utterances[-1].start, "overlap present"
        assert nxt.utterances[-1].start > prev.utterances[-1].start, "cursor advances"


def test_every_line_appears_in_some_chunk() -> None:
    lines = [Utterance(i * 10, "S1", f"line {i}") for i in range(25)]
    covered = {u.start for c in iter_chunks(lines, budget=64) for u in c.utterances}
    assert covered == {u.start for u in lines}


def test_chunk_respects_budget() -> None:
    lines = [Utterance(i * 10, "S1", "word " * 40) for i in range(30)]
    for c in iter_chunks(lines, budget=300):
        cost = sum(heuristic_token_len(u.render()) + 1 for u in c.utterances)
        assert cost <= 300 or len(c) == 1


def test_over_long_line_is_split_keeping_its_timestamp() -> None:
    # VCSum zh lines reach ~2.6k chars — longer than a whole chunk.
    long = Utterance(90, "S1", "很" * 2600)
    chunks = list(iter_chunks([long], budget=512))
    assert len(chunks) > 1
    assert all(u.start == 90 for c in chunks for u in c.utterances)
    assert "".join(u.text for c in chunks for u in c.utterances) == long.text


def test_chunk_has_line_is_exact() -> None:
    c = Chunk(0, tuple(parse_transcript(EXAMPLE)))
    assert c.has_line(310) and not c.has_line(311)


def test_empty_transcript_yields_no_chunks() -> None:
    assert list(iter_chunks([])) == []
