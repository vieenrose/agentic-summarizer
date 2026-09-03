"""Pins the CURSOR-style step loop and the net-new SYNTHESIZE call (SPEC §4.1).

The headline invariant: NO CONVERSATION HISTORY CROSSES STEPS. `Scripted` records every
`(system, user)` pair it is called with, which is what makes that provable without any
weights.
"""

from __future__ import annotations

import pytest

from arcsum.agent import (
    EMPTY_MEMORY_PROSE,
    STEP_BUDGET,
    StepBudgetExceeded,
    Usage,
    run_agent,
    synthesize_memory,
)
from arcsum.chunker import Chunk
from arcsum.memory import Memory
from arcsum.ops import Add, Arc, Op
from arcsum.prompts import TOOLCALL_PROMPT_VERSION
from arcsum.tokens import heuristic_token_len
from arcsum.transcript import Utterance
from conftest import Scripted, saturated_memory


def meeting(n: int, words_per_line: int = 20) -> list[Utterance]:
    return [Utterance(f"S{i % 4 + 1}", "很好 " * words_per_line) for i in range(n)]


# --- Usage -----------------------------------------------------------------------------


def test_usage_record_accumulates() -> None:
    u = Usage()
    u.record(100, 20, 1.5)
    u.record(50, 10, 0.5)
    assert u.calls == 2
    assert u.prefill_tokens == 150
    assert u.decode_tokens == 30
    assert u.wall_seconds == 2.0


def test_usage_defaults_carry_the_spec_seven_fields() -> None:
    u = Usage()
    assert u.wall_seconds == 0.0
    assert u.peak_rss_bytes == 0


# --- run_agent: the no-history invariant --------------------------------------------


def test_no_history_crosses_steps() -> None:
    """Each call gets a fresh SYS + freshly-rendered MEMORY+CHUNK -- nothing from a
    prior call's response is visible except through memory itself. If conversation
    history accumulated by concatenating prior turns, a later prompt would show more
    than one MEMORY:/CHUNK: block; a stateless implementation always shows exactly one."""
    model = Scripted(tuple(f"ADD - 第{i}項" for i in range(30)))
    run_agent(meeting(400, words_per_line=200), model, budget=500, synthesize=False)
    assert len(model.calls) > 5
    for _sys, user in model.calls:
        assert user.count("MEMORY:") == 1
        assert user.count("CHUNK:") == 1


def test_memory_visible_to_the_next_step() -> None:
    """What the PRIOR step wrote to memory IS visible -- via the rendered MEMORY block,
    never via the raw response text."""
    model = Scripted(("ADD - 同意搬到 B 棟", "NOP"))
    run_agent(meeting(60, words_per_line=200), model, budget=500, synthesize=False)
    assert len(model.calls) >= 2
    _sys0, user0 = model.calls[0]
    _sys1, user1 = model.calls[1]
    assert "同意搬到 B 棟" not in user0  # not yet written when step 0 was prompted
    assert "同意搬到 B 棟" in user1  # visible to step 1 via the rendered memory


def test_step_prompt_order_is_memory_then_chunk() -> None:
    model = Scripted()
    run_agent(meeting(5), model, synthesize=False)
    _sys, user = model.calls[0]
    assert user.index("MEMORY:") < user.index("CHUNK:")


def test_sys_prompt_is_reused_byte_identically_across_steps() -> None:
    model = Scripted()
    run_agent(meeting(60, words_per_line=200), model, budget=500, synthesize=False)
    systems = {sys for sys, _user in model.calls}
    assert len(systems) == 1


# --- run_agent: step records -----------------------------------------------------------


def test_trace_has_one_step_per_chunk() -> None:
    model = Scripted()
    trace = run_agent(meeting(60, words_per_line=200), model, budget=500, synthesize=False)
    assert len(trace.steps) >= 2
    assert [s.index for s in trace.steps] == list(range(len(trace.steps)))


def test_step_records_the_exact_prompt_and_raw_response() -> None:
    model = Scripted(("ADD - 一項決議",))
    trace = run_agent(meeting(5), model, synthesize=False)
    step = trace.steps[0]
    assert step.system == model.calls[0][0]
    assert step.user == model.calls[0][1]
    assert step.raw == "ADD - 一項決議"


def test_step_ops_reflect_the_parsed_response() -> None:
    model = Scripted(("ADD - 一項決議\nNOP",))
    trace = run_agent(meeting(5), model, synthesize=False)
    assert [type(op).__name__ for op in trace.steps[0].ops] == ["Add", "Nop"]


def test_step_is_nop_true_when_every_op_is_nop() -> None:
    model = Scripted(("NOP",))
    trace = run_agent(meeting(5), model, synthesize=False)
    assert trace.steps[0].is_nop is True


def test_step_is_nop_false_when_a_substantive_op_is_present() -> None:
    model = Scripted(("ADD - 一項決議",))
    trace = run_agent(meeting(5), model, synthesize=False)
    assert trace.steps[0].is_nop is False


def test_step_is_nop_true_on_empty_response() -> None:
    model = Scripted(("",))
    trace = run_agent(meeting(5), model, synthesize=False)
    assert trace.steps[0].is_nop is True


def test_memory_before_captures_the_pre_step_state() -> None:
    model = Scripted(("ADD - 同意搬到 B 棟", "NOP"))
    trace = run_agent(meeting(60, words_per_line=200), model, budget=500, synthesize=False)
    assert "同意搬到 B 棟" not in trace.steps[0].memory_before
    assert "同意搬到 B 棟" in trace.steps[1].memory_before


# --- run_agent: budget guard ------------------------------------------------------------


def test_over_budget_step_raises_step_budget_exceeded() -> None:
    """Raise, never truncate -- a silently shortened chunk references lines the student
    never saw."""
    memory = saturated_memory()
    model = Scripted()
    with pytest.raises(StepBudgetExceeded):
        run_agent(meeting(5), model, memory=memory, step_budget=1, synthesize=False)


def test_step_budget_default_is_documented() -> None:
    assert STEP_BUDGET == 3800


def test_budget_uses_the_injected_tokenizer() -> None:
    """A caller passing a real tokenizer must have the budget check use it, not a
    hidden default."""

    def huge(_text: str) -> int:
        return 10_000  # any real prompt "costs" more than any plausible step_budget

    model = Scripted()
    with pytest.raises(StepBudgetExceeded):
        run_agent(meeting(5), model, token_len=huge, step_budget=100, synthesize=False)


def test_memory_caps_use_the_injected_tokenizer() -> None:
    """The prior project's exact class of latent divergence: a caller using the real
    tokenizer for chunking must not silently get memory caps measured by the heuristic."""

    def generous(_text: str) -> int:
        return 1  # everything is "1 token"

    model = Scripted(("ADD - " + "很" * 500,))  # would be refused under the real counter
    trace = run_agent(meeting(5), model, token_len=generous, synthesize=False)
    assert trace.memory.token_len is generous
    assert len(trace.memory.points) == 1


def test_run_agent_overrides_a_passed_memorys_token_len() -> None:
    def custom(text: str) -> int:
        return len(text)

    memory = Memory()  # defaults to heuristic
    model = Scripted()
    trace = run_agent(meeting(5), model, memory=memory, token_len=custom, synthesize=False)
    assert trace.memory.token_len is custom


# --- run_agent: op_filter / step_filter ----------------------------------------------


def test_op_filter_vetoes_an_op_before_it_reaches_memory() -> None:
    """A vetoed op must not enter memory either, or the next step conditions on a
    point the student was never taught to produce."""

    def veto_everything(_op: Op, _chunk: Chunk) -> str | None:
        return "judge: unsupported"

    model = Scripted(("ADD - 一項決議",))
    trace = run_agent(meeting(5), model, op_filter=veto_everything, synthesize=False)
    assert trace.memory.points == []
    assert trace.steps[0].vetoed == (("ADD - 一項決議", "judge: unsupported"),)


def test_op_filter_lets_unvetoed_ops_through() -> None:
    def veto_nothing(_op: Op, _chunk: Chunk) -> str | None:
        return None

    model = Scripted(("ADD - 一項決議",))
    trace = run_agent(meeting(5), model, op_filter=veto_nothing, synthesize=False)
    assert [p.text for p in trace.memory.points] == ["一項決議"]
    assert trace.steps[0].vetoed == ()


def test_step_filter_can_veto_ops_as_a_batch() -> None:
    def veto_all(ops: list[Op], _chunk: Chunk) -> tuple[list[Op], list[tuple[Op, str]]]:
        return [], [(op, "batch veto") for op in ops]

    model = Scripted(("ADD - 一項決議",))
    trace = run_agent(meeting(5), model, step_filter=veto_all, synthesize=False)
    assert trace.memory.points == []
    assert trace.steps[0].vetoed == (("ADD - 一項決議", "batch veto"),)


def test_op_filter_and_step_filter_compose() -> None:
    """op_filter runs first; step_filter sees only what op_filter kept."""

    def veto_arc(op: Op, _chunk: Chunk) -> str | None:
        return "no arcs" if isinstance(op, Arc) else None

    seen: list[list[Op]] = []

    def record_and_pass(ops: list[Op], _chunk: Chunk) -> tuple[list[Op], list[tuple[Op, str]]]:
        seen.append(list(ops))
        return ops, []

    model = Scripted(("ADD - 一項決議\nARC: 摘要",))
    run_agent(meeting(5), model, op_filter=veto_arc, step_filter=record_and_pass, synthesize=False)
    assert len(seen[0]) == 1
    assert isinstance(seen[0][0], Add)


# --- run_agent: metrics -----------------------------------------------------------------


def test_trace_usage_counts_calls_and_tokens() -> None:
    model = Scripted(("ADD - 一項決議", "NOP"))
    trace = run_agent(meeting(60, words_per_line=200), model, budget=500, synthesize=False)
    assert trace.usage.calls == len(trace.steps)
    assert trace.usage.prefill_tokens > 0
    assert trace.usage.decode_tokens > 0


def test_valid_op_rate_excludes_nop_from_both_sides() -> None:
    model = Scripted(("NOP", "ADD - 一項決議", "ADD - "))  # 3rd is malformed (empty)
    trace = run_agent(meeting(60, words_per_line=200), model, budget=500, synthesize=False)
    # non-NOP attempted: 1 Add (applied) + 1 Malformed (not applied) = 2; 1 applied.
    assert trace.valid_op_rate == pytest.approx(0.5)


def test_valid_op_rate_is_none_with_no_non_nop_ops() -> None:
    model = Scripted(("NOP", "NOP"))
    trace = run_agent(meeting(60, words_per_line=200), model, budget=500, synthesize=False)
    assert trace.valid_op_rate is None


def test_nop_rate_on_rich_chunks() -> None:
    model = Scripted(("NOP", "ADD - 一項決議", "NOP"))
    trace = run_agent(meeting(90, words_per_line=200), model, budget=500, synthesize=False)
    rich = [s for s in trace.steps if s.chunk.is_content_rich(budget=500)]
    assert rich  # sanity: this meeting/budget combo does produce rich chunks
    assert trace.nop_rate_on_rich_chunks == pytest.approx(
        sum(1 for s in rich if s.is_nop) / len(rich)
    )


def test_trace_records_the_actual_run_budget() -> None:
    """`nop_rate_on_rich_chunks` must measure richness against the budget THIS RUN
    used, not the module default -- the same class of bug `apply_ops`'s own `budget`
    parameter guards against, one level up."""
    model = Scripted(("NOP",))
    trace = run_agent(meeting(5), model, budget=500, synthesize=False)
    assert trace.budget == 500


def test_drop_rate_and_arc_rate() -> None:
    model = Scripted(("ADD - 一項決議", "DROP «一項決議»", "ARC: 摘要"))
    trace = run_agent(meeting(90, words_per_line=200), model, budget=500, synthesize=False)
    assert trace.drop_rate == pytest.approx(1 / 3)
    assert trace.arc_rate == pytest.approx(1 / 3)


# --- run_agent: on_step callback -----------------------------------------------------


def test_on_step_is_called_once_per_chunk() -> None:
    events: list[tuple[int, float, int]] = []
    model = Scripted(("ADD - 一項決議", "NOP"))
    run_agent(
        meeting(60, words_per_line=200),
        model,
        budget=500,
        synthesize=False,
        on_step=lambda i, s, n: events.append((i, s, n)),
    )
    assert len(events) >= 2
    assert events[0][0] == 0


# --- run_agent: nop_retry (opt-in, default off) --------------------------------------


def test_nop_retry_defaults_to_off() -> None:
    """A run with retries must never be silently compared against one without."""
    model = Scripted(tuple("NOP" for _ in range(20)))
    trace = run_agent(meeting(5), model, synthesize=False)
    assert all(not s.retried for s in trace.steps)


def test_nop_retry_reissues_the_call_on_a_collapse() -> None:
    """A model that answers NOP every time eventually triggers NOP_COLLAPSE_K
    consecutive NOPs on content-rich chunks; `nop_retry=True` re-asks that step once."""
    model = Scripted(tuple("NOP" for _ in range(30)))
    trace = run_agent(
        meeting(200, words_per_line=200), model, budget=500, synthesize=False, nop_retry=True
    )
    assert any(s.retried for s in trace.steps)


def test_nop_retry_uses_the_nudged_prompt() -> None:
    model = Scripted(tuple("NOP" for _ in range(30)))
    trace = run_agent(
        meeting(200, words_per_line=200), model, budget=500, synthesize=False, nop_retry=True
    )
    retried_steps = [s for s in trace.steps if s.retried]
    assert retried_steps
    assert "提醒" in retried_steps[0].user


def test_nop_retry_records_both_calls_in_usage() -> None:
    model_no_retry = Scripted(tuple("NOP" for _ in range(30)))
    trace_no_retry = run_agent(
        meeting(200, words_per_line=200), model_no_retry, budget=500, synthesize=False
    )
    model_retry = Scripted(tuple("NOP" for _ in range(30)))
    trace_retry = run_agent(
        meeting(200, words_per_line=200),
        model_retry,
        budget=500,
        synthesize=False,
        nop_retry=True,
    )
    assert trace_retry.usage.calls > trace_no_retry.usage.calls


# --- SYNTHESIZE -----------------------------------------------------------------------


def test_one_synthesis_call_after_the_last_chunk() -> None:
    """A dedicated synth_model decouples this from the reading model's canned-response
    queue length, which is otherwise coupled to the (here, unpredictable) chunk count.

    The reading step ADDs (rather than NOPing) so the memory is non-empty: an all-NOP
    run leaves memory empty, which `synthesize_memory` now deliberately short-circuits
    without calling the model at all -- covered separately below."""
    step_model = Scripted(default="ADD - 同意搬到 B 棟")
    synth_model = Scripted(("會議討論搬遷案，最終決議遷至 B 棟。",))
    trace = run_agent(
        meeting(60, words_per_line=200), step_model, synth_model=synth_model, budget=500
    )
    assert trace.synthesis is not None
    assert len(step_model.calls) == len(trace.steps)
    assert len(synth_model.calls) == 1  # exactly one synthesis call, however many steps


def test_synthesize_false_skips_the_synthesis_call() -> None:
    model = Scripted(("NOP",))
    trace = run_agent(meeting(5), model, synthesize=False)
    assert trace.synthesis is None
    assert len(model.calls) == len(trace.steps)


def test_synthesis_reads_only_the_final_memory_no_chunk() -> None:
    model = Scripted(("ADD - 同意搬到 B 棟", "會議摘要文字內容"))
    trace = run_agent(meeting(5), model)
    assert "CHUNK:" not in trace.synthesis.user
    assert "同意搬到 B 棟" in trace.synthesis.user


def test_synthesize_memory_produces_prose() -> None:
    model = Scripted(("會議討論搬遷案，最終決議遷至 B 棟。",))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len)
    assert synthesis.prose.text == "會議討論搬遷案，最終決議遷至 B 棟。"
    assert synthesis.attempts == 1


def test_synthesize_memory_retries_on_over_budget() -> None:
    too_long = "很" * 2000
    ok = "會議討論搬遷案，最終決議遷至 B 棟。"
    model = Scripted((too_long, ok))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert synthesis.attempts == 2
    assert synthesis.prose.text == ok


def test_synthesize_memory_retries_on_bad_language() -> None:
    english = "The council approved the motion."
    ok = "會議討論搬遷案，最終決議遷至 B 棟。"
    model = Scripted((english, ok))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert synthesis.attempts == 2
    assert synthesis.prose.lang_flags == ()


def test_synthesize_memory_gives_up_after_retries_exhausted() -> None:
    model = Scripted(("The council approved.", "Still English."))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert synthesis.attempts == 2
    assert synthesis.prose.lang_flags != ()


def test_synthesize_memory_zero_retries_makes_exactly_one_attempt() -> None:
    model = Scripted(("The council approved.",))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=0)
    assert synthesis.attempts == 1
    assert len(model.calls) == 1


def test_synthesize_memory_retries_on_ungrounded_number() -> None:
    fabricated = "會議決議已於 2019 年 11 月 1 日完成搬遷。"
    ok = "會議討論搬遷案，最終決議遷至 B 棟。"
    model = Scripted((fabricated, ok))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert synthesis.attempts == 2
    assert synthesis.prose.text == ok
    assert synthesis.ungrounded_numbers == ()


def test_synthesize_memory_nudges_the_retry_prompt_on_ungrounded_number() -> None:
    fabricated = "會議決議已於 2019 年 11 月 1 日完成搬遷。"
    ok = "會議討論搬遷案，最終決議遷至 B 棟。"
    model = Scripted((fabricated, ok))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert len(model.calls) == 2
    first_user, second_user = model.calls[0][1], model.calls[1][1]
    assert first_user == second_user.split("\n\n（提醒")[0]
    assert second_user != first_user


def test_synthesize_memory_a_number_present_in_memory_is_not_flagged() -> None:
    ok = "會議核准了兩百萬美元的預算，編號 2019 案。"
    model = Scripted((ok,))
    memory = Memory()
    memory.add_point("核准兩百萬美元預算，編號 2019 案", chunk=0)
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len)
    assert synthesis.attempts == 1
    assert synthesis.ungrounded_numbers == ()


def test_synthesize_memory_records_ungrounded_numbers_after_retries_exhausted() -> None:
    fabricated = "會議決議已於 2019 年 11 月 1 日完成搬遷。"
    still_fabricated = "會議決議已於 2020 年 3 月 5 日完成搬遷。"
    model = Scripted((fabricated, still_fabricated))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert synthesis.attempts == 2
    assert synthesis.ungrounded_numbers != ()


def test_synthesize_memory_records_usage() -> None:
    model = Scripted(("會議討論搬遷案，最終決議遷至 B 棟。",))
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    usage = Usage()
    synthesize_memory(memory, model, token_len=heuristic_token_len, usage=usage)
    assert usage.calls == 1
    assert usage.prefill_tokens > 0


# --- SYNTHESIZE: the empty-memory guard -------------------------------------------------


def test_empty_memory_never_calls_the_model_at_all() -> None:
    """The whole point: with no arc and no points there is nothing to summarise, so ANY
    generated prose is unfaithful by construction. Measured motivation -- the G1 probe
    fed the fine-tuned student two short meetings, both were NOP'd to an empty memory,
    and SYNTHESIZE invented fluent, specific, entirely fictional land-use-zoning
    summaries (the training corpus's most common topic, i.e. the model's own prior
    filling a vacuum)."""
    model = Scripted(("這是一段不該被產生的幻覺摘要。",))
    synthesis = synthesize_memory(Memory(), model, token_len=heuristic_token_len)
    assert model.calls == []  # not merely unused output -- never invoked
    assert synthesis.skipped_empty_memory is True
    assert synthesis.attempts == 0
    assert synthesis.raw == ""
    assert "這是一段不該被產生的幻覺摘要。" not in synthesis.prose.text


def test_empty_memory_returns_the_fixed_honest_statement() -> None:
    synthesis = synthesize_memory(Memory(), Scripted(), token_len=heuristic_token_len)
    assert synthesis.prose.text == EMPTY_MEMORY_PROSE
    # Still a valid Prose under the §3 contract -- zh-TW, in budget, no markup.
    assert synthesis.prose.lang_flags == ()
    assert synthesis.prose.over_budget is False


def test_empty_memory_guard_records_no_usage() -> None:
    """No call was made, so nothing may be billed to the usage tally -- a phantom call
    would corrupt SPEC §7's per-meeting call/latency accounting."""
    usage = Usage()
    synthesize_memory(Memory(), Scripted(), token_len=heuristic_token_len, usage=usage)
    assert usage.calls == 0
    assert usage.prefill_tokens == 0
    assert usage.decode_tokens == 0


def test_arc_only_memory_is_not_empty_and_still_calls_the_model() -> None:
    """The guard is STRICT (both slots empty), not a 'thin memory' heuristic: an arc
    with no points is real information and must still be summarised normally."""
    memory = Memory()
    memory.set_arc("會議討論搬遷案")
    model = Scripted(("會議討論搬遷案，最終決議遷至 B 棟。",))
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len)
    assert len(model.calls) == 1
    assert synthesis.skipped_empty_memory is False
    assert synthesis.attempts == 1


def test_points_only_memory_is_not_empty_and_still_calls_the_model() -> None:
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    model = Scripted(("會議討論搬遷案，最終決議遷至 B 棟。",))
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len)
    assert len(model.calls) == 1
    assert synthesis.skipped_empty_memory is False


def test_normal_synthesis_is_not_flagged_as_skipped() -> None:
    memory = Memory()
    memory.add_point("同意搬到 B 棟", chunk=0)
    synthesis = synthesize_memory(
        memory, Scripted(("會議摘要文字內容。",)), token_len=heuristic_token_len
    )
    assert synthesis.skipped_empty_memory is False


def test_run_agent_end_to_end_all_nop_yields_the_honest_statement_not_a_hallucination() -> None:
    """End-to-end reproduction of the exact G1 probe failure mode: every reading step
    NOPs, memory stays empty, and the run must now decline to invent content rather
    than emit a confident fabrication."""
    model = Scripted(default="NOP")
    trace = run_agent(meeting(5), model)
    assert trace.memory.is_empty()
    assert trace.synthesis.skipped_empty_memory is True
    assert trace.synthesis.prose.text == EMPTY_MEMORY_PROSE
    # Only the reading steps hit the model; no synthesis call was made.
    assert len(model.calls) == len(trace.steps)


def test_run_agent_uses_synth_model_when_given() -> None:
    # ADD (not NOP) so memory is non-empty and the synthesis call actually happens.
    step_model = Scripted(("ADD - 同意搬到 B 棟",))
    synth_model = Scripted(("會議討論搬遷案，最終決議遷至 B 棟。",))
    trace = run_agent(meeting(5), step_model, synth_model=synth_model)
    assert len(step_model.calls) == 1  # only reading steps
    assert len(synth_model.calls) == 1  # only the synthesis call
    assert trace.synthesis.prose.text == "會議討論搬遷案，最終決議遷至 B 棟。"


def test_run_agent_uses_the_same_model_for_synthesis_by_default() -> None:
    model = Scripted(("ADD - 同意搬到 B 棟", "會議討論搬遷案，最終決議遷至 B 棟。"))
    trace = run_agent(meeting(5), model)
    assert len(model.calls) == 2
    assert trace.synthesis is not None


# --- prompt/tokenize version discipline ------------------------------------------------


def test_trace_records_prompt_and_tokenize_version() -> None:
    model = Scripted()
    trace = run_agent(meeting(5), model, synthesize=False)
    assert trace.prompt_version == "sys-v2"
    assert trace.tokenize_version == "chartok-v1"


def test_trace_records_the_token_len_instrument_name() -> None:
    model = Scripted()
    trace = run_agent(meeting(5), model, synthesize=False)
    assert trace.token_len_name == "heuristic"


# --- worst-case prefill: a saturated memory -------------------------------------------


def test_saturated_memory_fits_the_step_budget() -> None:
    """The worst-case prefill overhead a step can carry must still fit -- SPEC §4.1's
    table (~250 SYS + <=600 memory + ~2,500 chunk) is a claim about the WORST case."""
    memory = saturated_memory()
    model = Scripted(("NOP",))
    trace = run_agent(meeting(5), model, memory=memory, synthesize=False)
    assert trace.steps[0].prompt_tokens <= STEP_BUDGET


def test_step_error_aborts_the_meeting_by_default() -> None:
    """Default stays fail-fast. A partially-read meeting is NOT comparable to a fully
    read one, so a paired experiment must lose the meeting rather than quietly score a
    summary built from fewer chunks than its opponent saw."""
    utterances = meeting(300)

    def boom(_sys: str, _user: str) -> str:
        raise RuntimeError("llama-server 500")

    with pytest.raises(RuntimeError):
        run_agent(utterances, boom, synthesize=False)


def test_step_error_skip_records_the_chunk_and_keeps_reading() -> None:
    """`on_step_error="skip"` is the PRODUCT behaviour: one failed step must not cost the
    whole summary. Measured — an eval run lost `AlamedaCC_11162021` to a single
    llama.cpp 500, taking paired n from 20 to 19 and withholding every G3 gate.

    The failure must be RECORDED, not merely survived: a summary built from a partial
    read has to say so, or coverage metrics silently describe a different meeting.
    """
    utterances = meeting(300)
    calls = {"n": 0}

    def flaky(_sys: str, _user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("llama-server 500")
        return "ADD - 市議會核准搬遷案"

    trace = run_agent(utterances, flaky, synthesize=False, on_step_error="skip")

    assert trace.failed_steps == [0]
    assert len(trace.steps) >= 1
    assert all(s.index != 0 for s in trace.steps)
    assert trace.memory.points  # the surviving steps still curated memory


def test_step_budget_exceeded_is_not_swallowed_by_skip() -> None:
    """A prompt over budget is a CONFIGURATION error every later step would hit too, not
    a transient server fault — skipping it would silently read nothing."""
    utterances = meeting(300)

    with pytest.raises(StepBudgetExceeded):
        run_agent(
            utterances,
            lambda _s, _u: "NOP",
            synthesize=False,
            step_budget=1,
            on_step_error="skip",
        )


def test_tool_protocol_lands_on_the_same_ops_and_stamps_its_own_version() -> None:
    """SPEC §4.1 v1.0. The step grammar changed; memory, guards and caps did not. A run
    must also stamp the tool-call version, because a tool-call trace and an edit-line
    trace are not
    comparable and mixing them in one eval would be silent."""
    utterances = meeting(60)
    call = (
        '<tool_call>{"name":"update_memory","arguments":'
        '{"arc":"會議脈絡","add":["市議會核准搬遷案"]}}</tool_call>'
    )

    trace = run_agent(utterances, lambda _s, _u: call, synthesize=False, protocol="tool")

    assert trace.prompt_version == TOOLCALL_PROMPT_VERSION
    assert trace.memory.arc == "會議脈絡"
    assert any("搬遷案" in p.text for p in trace.memory.points)


def test_unknown_protocol_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown protocol"):
        run_agent(meeting(10), lambda _s, _u: "NOP", synthesize=False, protocol="react")


def test_partial_read_is_declared_on_the_summary_itself() -> None:
    """`on_step_error="skip"` keeps a meeting alive through a transient failure, but the
    reader must be told the read was incomplete. `trace.failed_steps` serves the harness;
    the notice serves the person deciding whether to trust the minutes."""
    utterances = meeting(300)
    calls = {"n": 0}

    def flaky(_sys: str, user: str) -> str:
        if "MEMORY:" in user:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("llama-server 500")
            return "ADD - 市議會核准搬遷案"
        return "本次會議通過搬遷案。"

    trace = run_agent(utterances, flaky, on_step_error="skip")

    assert trace.failed_steps
    assert "未能讀取" in trace.synthesis.prose.text


def test_a_complete_read_carries_no_notice() -> None:
    trace = run_agent(
        meeting(300),
        lambda _s, u: "ADD - 市議會核准搬遷案" if "MEMORY:" in u else "本次會議通過搬遷案。",
    )

    assert trace.failed_steps == []
    assert "未能讀取" not in trace.synthesis.prose.text
