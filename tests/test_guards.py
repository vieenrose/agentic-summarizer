"""Pins the applier (`apply_ops`) and its deterministic guards (SPEC §4.1, §4.2, §8).

The headline design decision guarded here: the contradiction guard's ordering signal is
INVERTED relative to the prior project, because v2 has no timestamps and every point in
memory necessarily came from an earlier-or-same read of the transcript. The guard fires
against points from a STRICTLY EARLIER chunk only — same-step points make no ordering
claim, and a DROP earlier in the same step's emission order clears the way for an ADD
that would otherwise contradict.
"""

from __future__ import annotations

import inspect

from arcsum.chunker import CHUNK_TOKENS, Chunk
from arcsum.guards import (
    CONTRADICTION_OVERLAP,
    NOP_COLLAPSE_K,
    AppliedOp,
    Outcome,
    apply_ops,
    contradiction,
    polarity,
)
from arcsum.memory import Memory, Point
from arcsum.ops import Add, Arc, Drop, Malformed, Nop, render_op
from arcsum.transcript import Utterance


def rich_chunk(index: int = 0) -> Chunk:
    """A content-rich chunk: tokens comfortably above CONTENT_RICH_FRAC * CHUNK_TOKENS."""
    return Chunk(index, (Utterance("S1", "x"),), tokens=int(0.9 * CHUNK_TOKENS))


def thin_chunk(index: int = 0) -> Chunk:
    return Chunk(index, (Utterance("S1", "嗯"),), tokens=5)


# --- AppliedOp / Outcome ------------------------------------------------------------


def test_applied_op_log_line_ok() -> None:
    assert AppliedOp(Nop(), True).log_line() == "[ok] NOP"


def test_applied_op_log_line_dropped_shows_the_reason() -> None:
    line = AppliedOp(Add("x"), False, "empty point").log_line()
    assert line.startswith("[dropped: empty point]")


def test_outcome_applied_excludes_nop() -> None:
    outcome = Outcome([AppliedOp(Nop(), True), AppliedOp(Add("x"), True)])
    assert outcome.applied == 1


def test_outcome_valid_op_rate_excludes_nop_from_numerator_and_denominator() -> None:
    """Counting NOP into only one side has been a real bug, twice, in the prior
    project's history — this is the regression it guards against."""
    outcome = Outcome(
        [
            AppliedOp(Nop(), True),
            AppliedOp(Nop(), True),
            AppliedOp(Add("a"), True),
            AppliedOp(Add("b"), False, "duplicate point"),
        ]
    )
    assert outcome.valid_op_rate == 0.5  # 1 of 2 non-NOP ops, NOT 1 of 4


def test_outcome_valid_op_rate_is_none_with_no_non_nop_ops() -> None:
    assert Outcome([AppliedOp(Nop(), True)]).valid_op_rate is None
    assert Outcome([]).valid_op_rate is None


def test_outcome_malformed_lists_only_malformed_entries() -> None:
    m = Malformed("garbage", "does not match the op grammar")
    outcome = Outcome([AppliedOp(Nop(), True), AppliedOp(m, False, m.reason)])
    assert [r.op for r in outcome.malformed] == [m]


# --- polarity() -----------------------------------------------------------------------


def test_polarity_classifies_known_pairs() -> None:
    assert polarity("議案通過") == 1
    assert polarity("議案否決") == -1
    assert polarity("案件核准") == 1
    assert polarity("案件駁回") == -1


def test_polarity_unknown_text_is_zero() -> None:
    assert polarity("大家討論了很久") == 0


def test_polarity_negatives_win_on_a_substring_collision() -> None:
    """'不通過' contains '通過' as a substring; the negative reading must win."""
    assert polarity("提案不通過") == -1


# --- apply_ops: ADD/DROP/ARC/NOP basics --------------------------------------------


def test_add_succeeds_and_appends_to_memory() -> None:
    memory = Memory()
    outcome = apply_ops(memory, [Add("同意搬到 B 棟")], rich_chunk())
    assert outcome.applied == 1
    assert [p.text for p in memory.points] == ["同意搬到 B 棟"]


def test_drop_succeeds_and_removes_from_memory() -> None:
    memory = Memory()
    memory.add_point("同意搬到 B 棟大樓", chunk=0)
    outcome = apply_ops(memory, [Drop("同意搬到")], rich_chunk())
    assert outcome.applied == 1
    assert memory.points == []


def test_arc_succeeds_and_sets_the_arc() -> None:
    memory = Memory()
    outcome = apply_ops(memory, [Arc("會議討論辦公室搬遷")], rich_chunk())
    assert outcome.applied == 1
    assert memory.arc == "會議討論辦公室搬遷"


def test_nop_always_succeeds_and_never_touches_memory() -> None:
    memory = Memory()
    memory.add_point("既有決議", chunk=0)
    memory.set_arc("既有摘要")
    outcome = apply_ops(memory, [Nop()], rich_chunk())
    assert outcome.results == [AppliedOp(Nop(), True)]
    assert [p.text for p in memory.points] == ["既有決議"]
    assert memory.arc == "既有摘要"


def test_malformed_always_fails_with_its_parsed_reason() -> None:
    memory = Memory()
    bad = Malformed("garbled text", "does not match the op grammar")
    outcome = apply_ops(memory, [bad], rich_chunk())
    assert outcome.results == [AppliedOp(bad, False, "does not match the op grammar")]


def test_refused_op_leaves_memory_unchanged() -> None:
    """SPEC §4.2: 'never half-applied into the corpus'. `parse_ops` would normally turn
    an empty ADD body into `Malformed` before it ever reaches here, but constructing
    `Add("")` directly exercises the applier's own defense-in-depth via `Memory.add_point`."""
    memory = Memory()
    outcome = apply_ops(memory, [Add("")], rich_chunk())
    assert outcome.applied == 0
    assert memory.points == []


def test_ops_apply_in_emission_order() -> None:
    """A step's own ADD must be visible to its own later DROP."""
    memory = Memory()
    outcome = apply_ops(memory, [Add("同意搬到 B 棟"), Drop("同意搬到")], rich_chunk())
    assert outcome.applied == 2
    assert memory.points == []


def test_ops_apply_in_emission_order_reversed_fails_cleanly() -> None:
    """DROP before an ADD that would create the point cannot possibly match anything."""
    memory = Memory()
    outcome = apply_ops(memory, [Drop("同意搬到"), Add("同意搬到 B 棟")], rich_chunk())
    assert outcome.results[0].applied is False
    assert outcome.results[1].applied is True
    assert [p.text for p in memory.points] == ["同意搬到 B 棟"]


def test_enforce_caps_runs_after_a_step_even_without_a_separate_call() -> None:
    memory = Memory()
    ops = [Add(f"第{i}項決議內容") for i in range(20)]
    apply_ops(memory, ops, rich_chunk())
    assert len(memory.points) == 16  # POINTS_CAP, applied automatically


# --- language guard integration -----------------------------------------------------


def test_add_refuses_english_text() -> None:
    memory = Memory()
    outcome = apply_ops(memory, [Add("the council approved the motion")], rich_chunk())
    assert outcome.applied == 0
    assert memory.points == []
    assert "zh-TW" in outcome.results[0].reason


def test_arc_refuses_simplified_text() -> None:
    memory = Memory()
    outcome = apply_ops(memory, [Arc("讨论会议记录")], rich_chunk())
    assert outcome.applied == 0
    assert memory.arc == ""


def test_lang_check_false_bypasses_the_language_guard() -> None:
    """Escape hatch used to test the OTHER guards in isolation."""
    memory = Memory()
    outcome = apply_ops(
        memory, [Add("the council approved the motion")], rich_chunk(), lang_check=False
    )
    assert outcome.applied == 1


# --- contradiction guard: the redesigned inversion check -----------------------------


def test_contradiction_ignores_unknown_polarity_text() -> None:
    memory = Memory()
    memory.points.append(Point("很好的討論內容", chunk=0))
    assert contradiction(memory, "另一段沒有極性的文字", chunk_index=5) is None


def test_contradiction_ignores_unrelated_subjects() -> None:
    memory = Memory()
    memory.points.append(Point("預算案通過", chunk=0))
    assert contradiction(memory, "人事案否決", chunk_index=5) is None


def test_contradiction_fires_against_an_earlier_chunk() -> None:
    memory = Memory()
    memory.points.append(Point("同意搬遷辦公室通過", chunk=0))
    reason = contradiction(memory, "同意搬遷辦公室否決", chunk_index=5)
    assert reason is not None
    assert "同意搬遷辦公室通過" in reason


def test_contradiction_does_not_fire_between_same_step_points() -> None:
    """No ordering claim can be made between two points from the same chunk."""
    memory = Memory()
    memory.points.append(Point("同意搬遷辦公室通過", chunk=5))
    assert contradiction(memory, "同意搬遷辦公室否決", chunk_index=5) is None


def test_contradiction_does_not_fire_when_existing_polarity_matches() -> None:
    memory = Memory()
    memory.points.append(Point("同意搬遷辦公室通過", chunk=0))
    assert contradiction(memory, "同意搬遷辦公室核准", chunk_index=5) is None


def test_contradiction_does_not_fire_when_existing_polarity_is_unknown() -> None:
    # No polarity marker in the existing text at all (not even the positive "同意"),
    # so its polarity is genuinely unknown -- the guard must not fire either direction.
    memory = Memory()
    memory.points.append(Point("搬遷辦公室仍在討論中", chunk=0))
    assert contradiction(memory, "搬遷辦公室否決", chunk_index=5) is None


def test_contradiction_respects_the_overlap_threshold() -> None:
    """A weak lexical overlap must not trigger a false 'contradiction'."""
    memory = Memory()
    memory.points.append(Point("搬遷案通過", chunk=0))
    # Almost no shared bigrams with '搬遷案' -- should not fire despite opposite polarity.
    assert contradiction(memory, "完全不同的其他案否決", chunk_index=5) is None


def test_add_via_apply_ops_is_refused_by_the_contradiction_guard() -> None:
    memory = Memory()
    memory.points.append(Point("同意搬遷辦公室通過", chunk=0))
    outcome = apply_ops(memory, [Add("同意搬遷辦公室否決")], rich_chunk(index=5))
    assert outcome.applied == 0
    assert "同意搬遷辦公室通過" in outcome.results[0].reason
    # The refused ADD must not have touched memory.
    assert [p.text for p in memory.points] == ["同意搬遷辦公室通過"]


def test_drop_then_add_in_one_step_clears_the_contradiction() -> None:
    """The escape hatch: DROP earlier in the SAME step's emission order removes the
    point before the contradicting ADD's check runs."""
    memory = Memory()
    memory.points.append(Point("同意搬遷辦公室通過", chunk=0))
    outcome = apply_ops(
        memory, [Drop("同意搬遷辦公室通過"), Add("同意搬遷辦公室否決")], rich_chunk(index=5)
    )
    assert outcome.applied == 2
    assert [p.text for p in memory.points] == ["同意搬遷辦公室否決"]


def test_add_add_in_the_same_step_does_not_trigger_the_guard() -> None:
    """Two contradicting ADDs emitted within one step (no DROP between them) both
    succeed -- the guard's scope is cross-step inversions, not within-chunk narration."""
    memory = Memory()
    outcome = apply_ops(
        memory, [Add("同意搬遷辦公室通過"), Add("同意搬遷辦公室否決")], rich_chunk(index=5)
    )
    assert outcome.applied == 2
    assert {p.text for p in memory.points} == {"同意搬遷辦公室通過", "同意搬遷辦公室否決"}


def test_contradiction_overlap_constant_is_ported_unchanged() -> None:
    assert CONTRADICTION_OVERLAP == 0.34


# --- NOP-collapse guard: detect and record, never repair ----------------------------


def test_single_nop_on_a_content_rich_chunk_does_not_collapse() -> None:
    memory = Memory()
    outcome = apply_ops(memory, [Nop()], rich_chunk(), consecutive_nops=0)
    assert outcome.nop_collapse is False


def test_k_consecutive_nops_on_rich_chunks_flags_collapse() -> None:
    memory = Memory()
    outcome = apply_ops(memory, [Nop()], rich_chunk(), consecutive_nops=NOP_COLLAPSE_K - 1)
    assert outcome.nop_collapse is True


def test_below_k_consecutive_nops_does_not_collapse() -> None:
    memory = Memory()
    outcome = apply_ops(memory, [Nop()], rich_chunk(), consecutive_nops=NOP_COLLAPSE_K - 2)
    assert outcome.nop_collapse is False


def test_nop_on_a_thin_chunk_is_never_a_collapse() -> None:
    """A genuinely empty chunk deserves NOP -- collapse only applies to content-rich ones."""
    memory = Memory()
    outcome = apply_ops(memory, [Nop()], thin_chunk(), consecutive_nops=NOP_COLLAPSE_K)
    assert outcome.nop_collapse is False


def test_a_substantive_op_never_collapses_regardless_of_the_running_count() -> None:
    memory = Memory()
    outcome = apply_ops(
        memory, [Add("同意搬到 B 棟")], rich_chunk(), consecutive_nops=NOP_COLLAPSE_K
    )
    assert outcome.nop_collapse is False


def test_a_step_with_only_refused_ops_still_counts_as_not_substantive() -> None:
    """A step where every op was refused has made no real progress, same as a NOP."""
    memory = Memory()
    outcome = apply_ops(
        memory,
        [Malformed("garbage", "does not match the op grammar")],
        rich_chunk(),
        consecutive_nops=NOP_COLLAPSE_K - 1,
    )
    assert outcome.nop_collapse is True


def test_apply_ops_uses_the_actual_chunking_budget_for_richness() -> None:
    """A real bug this test caught: `Chunk.is_content_rich()` defaults its `budget`
    param to CHUNK_TOKENS (2500), so a caller running `iter_chunks` at a smaller custom
    budget (e.g. 500) must pass that SAME budget into `apply_ops`, or the richness
    threshold silently compares against the wrong denominator and NOP-collapse can
    never fire even on a chunk that is fully packed for its actual budget."""
    small_budget = 500
    # Packed to ~90% of the SMALL budget -- content-rich there, but far below 25% of
    # the default CHUNK_TOKENS=2500 (625), so the bug this guards against would make
    # this chunk register as NOT rich unless `budget` is threaded through correctly.
    chunk = Chunk(0, (Utterance("S1", "x"),), tokens=int(0.9 * small_budget))
    memory = Memory()
    outcome = apply_ops(
        memory, [Nop()], chunk, consecutive_nops=NOP_COLLAPSE_K - 1, budget=small_budget
    )
    assert outcome.nop_collapse is True


def test_apply_ops_accepts_no_model_callable() -> None:
    """The guard REPORTS a coverage gap; it must never repair it by calling another
    model (e.g. the map-reduce baseline) from inside the harness. Structurally enforced:
    `apply_ops`'s signature has no parameter through which a model could be reached."""
    params = inspect.signature(apply_ops).parameters
    assert not any("model" in name for name in params)


def test_render_op_still_works_for_ops_seen_through_applied_op() -> None:
    result = AppliedOp(Add("x"), True)
    assert render_op(result.op) == "ADD - x"
