"""Pins SPEC §4.2's supervision-construction discipline: gold-edit replay validation,
completion-only SFT samples, split-by-meeting, the NOP-share cap, and the aggregate
report definitions.
"""

from __future__ import annotations

import pytest

from arcsum.agent import run_agent
from arcsum.chunker import Chunk
from arcsum.memory import Memory
from arcsum.supervision.report import report
from arcsum.supervision.sft import (
    DEFAULT_MAX_NOP_FRAC,
    MixedPromptVersionError,
    SftSample,
    build_samples,
    check_single_prompt_version,
    downsample_nop,
    drop_bearing_share,
    drop_share,
    late_step_share,
    nop_share,
    oversample_drop,
    oversample_late_steps,
    split_by_meeting,
)
from arcsum.supervision.traces import all_replayed_cleanly, replay_sequence, replay_step
from arcsum.transcript import Utterance
from conftest import Scripted


def chunk(index: int = 0) -> Chunk:
    return Chunk(index, (Utterance("S1", "討論內容"),), tokens=400)


# --- replay_step -----------------------------------------------------------------------


def test_replay_step_ok_on_a_clean_add() -> None:
    memory = Memory()
    result = replay_step(memory, "ADD - 同意搬到 B 棟", chunk())
    assert result.ok is True
    assert result.failures == ()
    assert [p.text for p in memory.points] == ["同意搬到 B 棟"]


def test_replay_step_fails_on_a_drop_prefix_matching_nothing() -> None:
    """SPEC §4.2: "DROP prefixes must match an existing point"."""
    memory = Memory()
    result = replay_step(memory, "DROP «不存在的項目»", chunk())
    assert result.ok is False
    assert len(result.failures) == 1


def test_replay_step_fails_on_a_malformed_op() -> None:
    """SPEC §4.2: "ops must parse"."""
    memory = Memory()
    result = replay_step(memory, "UPD SUMMARY «x» -> y", chunk())  # UPD does not exist in v2
    assert result.ok is False


def test_replay_step_fails_on_a_refused_op_not_just_a_malformed_one() -> None:
    """An op that PARSES but is REFUSED (duplicate) must also count as a failure."""
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    result = replay_step(memory, "ADD - 同意搬到 B 棟", chunk())
    assert result.ok is False


def test_replay_step_respects_the_caps_via_spread_not_a_failure() -> None:
    """SPEC §4.2: "the resulting memory must respect the caps" -- an over-cap ADD is
    auto-corrected by spread(), not a replay failure in itself."""
    memory = Memory()
    for i in range(16):
        memory.add_point(f"第{i}項決議", chunk=i)
    result = replay_step(memory, "ADD - 第17項決議", chunk(16))
    assert result.ok is True
    assert len(memory.points) == 16  # capped, not 17


def test_replay_step_nop_always_succeeds() -> None:
    memory = Memory()
    result = replay_step(memory, "NOP", chunk())
    assert result.ok is True


# --- replay_sequence / all_replayed_cleanly ---------------------------------------------


def test_replay_sequence_applies_in_step_order_against_one_shared_memory() -> None:
    targets = [
        ("ADD - 同意搬到 B 棟", chunk(0)),
        ("DROP «同意搬到»", chunk(1)),
        ("ADD - 撤回搬遷案", chunk(2)),
    ]
    memory, results = replay_sequence(targets)
    assert [r.ok for r in results] == [True, True, True]
    assert [p.text for p in memory.points] == ["撤回搬遷案"]


def test_replay_sequence_identifies_exactly_which_step_failed() -> None:
    targets = [
        ("ADD - 同意搬到 B 棟", chunk(0)),
        ("DROP «不存在»", chunk(1)),  # fails
        ("ADD - 另一項決議", chunk(2)),  # succeeds independently
    ]
    _memory, results = replay_sequence(targets)
    assert [r.ok for r in results] == [True, False, True]


def test_all_replayed_cleanly_is_false_if_any_step_failed() -> None:
    targets = [("ADD - a", chunk(0)), ("DROP «不存在»", chunk(1))]
    _memory, results = replay_sequence(targets)
    assert all_replayed_cleanly(results) is False


def test_all_replayed_cleanly_is_true_when_every_step_succeeded() -> None:
    targets = [("ADD - 同意搬到 B 棟", chunk(0)), ("NOP", chunk(1))]
    _memory, results = replay_sequence(targets)
    assert all_replayed_cleanly(results) is True


def test_all_replayed_cleanly_of_an_empty_sequence_is_true() -> None:
    assert all_replayed_cleanly([]) is True


# --- build_samples -----------------------------------------------------------------------


def meeting_utts(n: int = 5) -> list[Utterance]:
    return [Utterance(f"S{i % 2 + 1}", "很好 " * 20) for i in range(n)]


def test_build_samples_one_per_reading_step() -> None:
    model = Scripted(("ADD - 一項決議", "NOP"))
    trace = run_agent(meeting_utts(60), model, budget=500, synthesize=False)
    samples = build_samples("m1", trace)
    assert len(samples) == len(trace.steps)
    assert all(s.meeting == "m1" for s in samples)


def test_build_samples_includes_a_synthesis_sample_when_present() -> None:
    # ADD (not NOP) so memory is non-empty: an all-NOP run leaves memory empty, which
    # `synthesize_memory` short-circuits without calling the model -- covered below.
    model = Scripted(("ADD - 同意搬到 B 棟", "會議討論搬遷案，最終決議遷至 B 棟。"))
    trace = run_agent(meeting_utts(5), model)
    samples = build_samples("m1", trace)
    assert len(samples) == len(trace.steps) + 1
    synth_sample = samples[-1]
    assert synth_sample.is_nop is False
    assert synth_sample.completion == "會議討論搬遷案，最終決議遷至 B 棟。"


def test_build_samples_omits_a_guarded_empty_memory_synthesis() -> None:
    """An all-NOP run's synthesis was never generated -- `raw` is "" and `prose.text`
    is a fixed constant -- so it must not enter the training pool at all. Emitting it
    would either inject an empty-completion row or teach the model to reproduce a
    hardcoded string the deterministic guard already handles."""
    model = Scripted(default="NOP")
    trace = run_agent(meeting_utts(5), model)
    assert trace.synthesis is not None
    assert trace.synthesis.skipped_empty_memory is True

    samples = build_samples("m1", trace)
    assert len(samples) == len(trace.steps)  # reading steps only, no synthesis row
    assert all(s.completion != "" for s in samples)


def test_build_samples_completion_is_the_raw_target_not_the_prompt() -> None:
    """Completion-only masking needs prompt and completion kept separate -- training
    on the prompt teaches the model to reproduce transcripts, not the task."""
    model = Scripted(("ADD - 一項決議",))
    trace = run_agent(meeting_utts(5), model, synthesize=False)
    sample = build_samples("m1", trace)[0]
    assert sample.completion == "ADD - 一項決議"
    assert sample.prompt != sample.completion
    assert "MEMORY:" in sample.prompt


def test_build_samples_records_is_nop_per_step() -> None:
    model = Scripted(("NOP", "ADD - 一項決議"))
    trace = run_agent(meeting_utts(90), model, budget=500, synthesize=False)
    samples = build_samples("m1", trace)
    assert samples[0].is_nop is True
    assert samples[1].is_nop is False


# --- check_single_prompt_version ---------------------------------------------------------


def test_check_single_prompt_version_passes_for_a_uniform_pool() -> None:
    samples = [
        SftSample("m1", 0, "sys-v1", "sys", "prompt", "completion", is_nop=False),
        SftSample("m1", 1, "sys-v1", "sys", "prompt", "completion", is_nop=False),
    ]
    assert check_single_prompt_version(samples) == "sys-v1"


def test_check_single_prompt_version_refuses_a_mixed_pool() -> None:
    samples = [
        SftSample("m1", 0, "sys-v1", "sys", "prompt", "completion", is_nop=False),
        SftSample("m2", 0, "sys-v2", "sys", "prompt", "completion", is_nop=False),
    ]
    with pytest.raises(MixedPromptVersionError, match="mixed prompt versions"):
        check_single_prompt_version(samples)


def test_check_single_prompt_version_refuses_an_empty_pool() -> None:
    with pytest.raises(MixedPromptVersionError, match="no samples"):
        check_single_prompt_version([])


# --- downsample_nop / nop_share ---------------------------------------------------------


def make_pool(n_nop: int, n_non_nop: int) -> list[SftSample]:
    samples = [
        SftSample(f"m{i}", 0, "sys-v1", "sys", "prompt", "NOP", is_nop=True) for i in range(n_nop)
    ]
    samples += [
        SftSample(f"m{i}", 1, "sys-v1", "sys", "prompt", "ADD - x", is_nop=False)
        for i in range(n_non_nop)
    ]
    return samples


def test_nop_share_reports_the_actual_fraction() -> None:
    pool = make_pool(n_nop=30, n_non_nop=70)
    assert nop_share(pool) == pytest.approx(0.30)


def test_nop_share_of_empty_pool_is_none() -> None:
    assert nop_share([]) is None


def test_downsample_nop_leaves_a_pool_already_under_the_cap_unchanged() -> None:
    pool = make_pool(n_nop=10, n_non_nop=90)  # 10% NOP, well under 35%
    result = downsample_nop(pool, max_nop_frac=0.35)
    assert len(result) == len(pool)


def test_downsample_nop_caps_an_over_share_pool() -> None:
    pool = make_pool(n_nop=90, n_non_nop=10)  # 90% NOP
    result = downsample_nop(pool, max_nop_frac=0.35, seed=0)
    assert nop_share(result) <= 0.35 + 1e-9


def test_downsample_nop_never_removes_non_nop_samples() -> None:
    pool = make_pool(n_nop=90, n_non_nop=10)
    result = downsample_nop(pool, max_nop_frac=0.35, seed=0)
    assert sum(1 for s in result if not s.is_nop) == 10


def test_downsample_nop_is_deterministic_given_a_seed() -> None:
    pool = make_pool(n_nop=90, n_non_nop=10)
    a = downsample_nop(pool, max_nop_frac=0.35, seed=7)
    b = downsample_nop(pool, max_nop_frac=0.35, seed=7)
    assert a == b


def test_default_max_nop_frac_matches_spec_risk_3() -> None:
    assert DEFAULT_MAX_NOP_FRAC == 0.35


# --- oversample_drop -----------------------------------------------------------------


def make_drop_pool(n_drop: int, n_plain: int) -> list[SftSample]:
    samples = [
        SftSample(f"d{i}", 0, "sys-v1", "sys", "prompt", "DROP «x»\nADD - y", is_nop=False)
        for i in range(n_drop)
    ]
    samples += [
        SftSample(f"p{i}", 0, "sys-v1", "sys", "prompt", "ADD - x", is_nop=False)
        for i in range(n_plain)
    ]
    return samples


def test_drop_bearing_share_reports_the_actual_fraction() -> None:
    pool = make_drop_pool(n_drop=10, n_plain=90)
    assert drop_bearing_share(pool) == pytest.approx(0.10)


def test_drop_bearing_share_of_empty_pool_is_none() -> None:
    assert drop_bearing_share([]) is None


def test_oversample_drop_default_is_a_noop() -> None:
    pool = make_drop_pool(n_drop=10, n_plain=90)
    assert oversample_drop(pool) == pool


def test_oversample_drop_raises_the_share_toward_the_target() -> None:
    pool = make_drop_pool(n_drop=10, n_plain=90)  # 10% DROP-bearing
    result = oversample_drop(pool, target_drop_frac=0.4, seed=0)
    assert drop_bearing_share(result) == pytest.approx(0.4, abs=0.01)


def test_oversample_drop_never_removes_or_alters_original_samples() -> None:
    pool = make_drop_pool(n_drop=10, n_plain=90)
    result = oversample_drop(pool, target_drop_frac=0.4, seed=0)
    for s in pool:
        assert result.count(s) >= 1


def test_oversample_drop_leaves_a_pool_already_at_or_above_target_unchanged() -> None:
    pool = make_drop_pool(n_drop=50, n_plain=50)  # 50% DROP-bearing
    result = oversample_drop(pool, target_drop_frac=0.4, seed=0)
    assert result == pool


def test_oversample_drop_is_deterministic_given_a_seed() -> None:
    pool = make_drop_pool(n_drop=10, n_plain=90)
    a = oversample_drop(pool, target_drop_frac=0.4, seed=3)
    b = oversample_drop(pool, target_drop_frac=0.4, seed=3)
    assert a == b


def test_oversample_drop_of_a_pool_with_no_drop_rows_is_a_noop() -> None:
    pool = make_pool(n_nop=10, n_non_nop=90)  # no DROP anywhere
    assert oversample_drop(pool, target_drop_frac=0.4) == pool


# --- split_by_meeting ---------------------------------------------------------------------


def test_split_by_meeting_keeps_every_meetings_steps_together() -> None:
    samples = [
        SftSample("m1", 0, "sys-v1", "sys", "p", "c", is_nop=False),
        SftSample("m1", 1, "sys-v1", "sys", "p", "c", is_nop=False),
        SftSample("m2", 0, "sys-v1", "sys", "p", "c", is_nop=False),
    ]
    train, _valid = split_by_meeting(samples, valid_frac=0.5, seed=0)
    # m1's two steps must land in the SAME split -- never one train, one valid.
    m1_splits = {("train" if s in train else "valid") for s in samples if s.meeting == "m1"}
    assert len(m1_splits) == 1


def test_split_by_meeting_covers_every_sample_exactly_once() -> None:
    samples = [SftSample(f"m{i}", 0, "sys-v1", "sys", "p", "c", is_nop=False) for i in range(20)]
    train, valid = split_by_meeting(samples, valid_frac=0.2, seed=0)
    assert set(train) | set(valid) == set(samples)
    assert set(train) & set(valid) == set()


def test_split_by_meeting_is_deterministic() -> None:
    samples = [SftSample(f"m{i}", 0, "sys-v1", "sys", "p", "c", is_nop=False) for i in range(20)]
    a = split_by_meeting(samples, valid_frac=0.2, seed=3)
    b = split_by_meeting(samples, valid_frac=0.2, seed=3)
    assert a == b


def test_split_by_meeting_of_empty_pool() -> None:
    assert split_by_meeting([], valid_frac=0.2) == ([], [])


# --- drop_share --------------------------------------------------------------------------


def test_drop_share_counts_completions_containing_drop() -> None:
    samples = [
        SftSample("m1", 0, "sys-v1", "sys", "p", "ADD - x", is_nop=False),
        SftSample("m1", 1, "sys-v1", "sys", "p", "DROP «x»", is_nop=False),
        SftSample("m1", 2, "sys-v1", "sys", "p", "NOP", is_nop=True),  # excluded: is_nop
    ]
    assert drop_share(samples) == pytest.approx(0.5)


def test_drop_share_of_empty_pool_is_none() -> None:
    assert drop_share([]) is None


# --- supervision.report -----------------------------------------------------------------


def test_report_total_steps() -> None:
    model = Scripted(("ADD - 一項決議", "NOP"))
    trace = run_agent(meeting_utts(60), model, budget=500, synthesize=False)
    r = report([trace])
    assert r.total_steps == len(trace.steps)


def test_report_valid_op_rate_excludes_nop_from_both_sides() -> None:
    """The metric definition that regressed twice in the prior project."""
    model = Scripted(("NOP", "ADD - 一項決議", "ADD - "))  # 3rd is malformed (empty)
    trace = run_agent(meeting_utts(60), model, budget=500, synthesize=False)
    r = report([trace])
    assert r.valid_op_rate == pytest.approx(0.5)  # 1 of 2 NON-NOP ops applied


def test_report_drop_and_arc_share() -> None:
    model = Scripted(("ADD - 一項決議", "DROP «一項決議»", "ARC: 摘要"))
    trace = run_agent(meeting_utts(90), model, budget=500, synthesize=False)
    r = report([trace])
    assert r.drop_share == pytest.approx(1 / 3)
    assert r.arc_share == pytest.approx(1 / 3)


def test_report_aggregates_across_multiple_traces() -> None:
    model_a = Scripted(("ADD - 一項",))
    model_b = Scripted(("ADD - 二項",))
    trace_a = run_agent(meeting_utts(5), model_a, synthesize=False)
    trace_b = run_agent(meeting_utts(5), model_b, synthesize=False)
    r = report([trace_a, trace_b])
    assert r.total_steps == len(trace_a.steps) + len(trace_b.steps)


def test_report_veto_rate_counts_vetoed_ops_in_the_denominator() -> None:
    def veto_everything(_op, _chunk):
        return "judge: unsupported"

    model = Scripted(("ADD - 一項決議",))
    trace = run_agent(meeting_utts(5), model, op_filter=veto_everything, synthesize=False)
    r = report([trace])
    assert r.veto_rate == pytest.approx(1.0)


def test_report_with_no_traces_returns_none_for_every_rate() -> None:
    r = report([])
    assert r.valid_op_rate is None
    assert r.nop_rate_on_rich_chunks is None
    assert r.drop_share is None
    assert r.arc_share is None
    assert r.veto_rate is None
    assert r.total_steps == 0


# --- oversample_late_steps -------------------------------------------------------------


def make_step_pool(n_late: int, n_early: int, *, late_are_nops: bool = False) -> list[SftSample]:
    """`n_late` samples at step 30, `n_early` at step 0."""
    late = [
        SftSample(
            f"L{i}",
            30,
            "sys-v1",
            "sys",
            "prompt",
            "NOP" if late_are_nops else "ADD - x",
            is_nop=late_are_nops,
        )
        for i in range(n_late)
    ]
    early = [
        SftSample(f"E{i}", 0, "sys-v1", "sys", "prompt", "ADD - y", is_nop=False)
        for i in range(n_early)
    ]
    return late + early


def test_oversample_late_steps_default_is_a_noop() -> None:
    pool = make_step_pool(n_late=5, n_early=95)
    assert oversample_late_steps(pool) == pool


def test_oversample_late_steps_raises_the_share_toward_the_target() -> None:
    """Measured motivation: only 1.6% of gold steps sit at index 40+, and the student
    stops making progress deep in long meetings — re-emitting an identical ARC while the
    transcript has moved on."""
    pool = make_step_pool(n_late=5, n_early=95)  # 5% late
    result = oversample_late_steps(pool, min_step=25, target_frac=0.3, seed=0)
    assert late_step_share(result, min_step=25) == pytest.approx(0.3, abs=0.02)


def test_oversample_late_steps_never_removes_or_alters_original_samples() -> None:
    pool = make_step_pool(n_late=5, n_early=95)
    result = oversample_late_steps(pool, min_step=25, target_frac=0.3, seed=0)
    for s in pool:
        assert s in result
    assert len(result) > len(pool)


def test_oversample_late_steps_pushes_the_nop_share_back_up() -> None:
    """The compounding interaction the docstring warns about, pinned. Late steps are
    NOP-heavy (the teacher's NOP rate rises from 32% at steps 0-9 to 51% at 40+), so
    this knob partially undoes `downsample_nop`'s cap. `build_sft` reports every share
    after the fact precisely so this is set by measurement, not assumption.
    """
    pool = make_step_pool(n_late=5, n_early=95, late_are_nops=True)
    before = nop_share(pool)
    result = oversample_late_steps(pool, min_step=25, target_frac=0.3, seed=0)
    assert nop_share(result) > before
