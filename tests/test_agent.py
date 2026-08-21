"""Pins the CURSOR-style step loop and the net-new SYNTHESIZE call (SPEC §4.1).

The headline invariant: NO CONVERSATION HISTORY CROSSES STEPS. `Scripted` records every
`(system, user)` pair it is called with, which is what makes that provable without any
weights.
"""

from __future__ import annotations

import pytest

from arcsum.agent import (
    STEP_BUDGET,
    StepBudgetExceeded,
    Usage,
    run_agent,
    synthesize_memory,
)
from arcsum.chunker import Chunk
from arcsum.memory import Memory
from arcsum.ops import Add, Arc, Op
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
    queue length, which is otherwise coupled to the (here, unpredictable) chunk count."""
    step_model = Scripted(default="NOP")
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
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert synthesis.attempts == 2
    assert synthesis.prose.text == ok


def test_synthesize_memory_retries_on_bad_language() -> None:
    english = "The council approved the motion."
    ok = "會議討論搬遷案，最終決議遷至 B 棟。"
    model = Scripted((english, ok))
    memory = Memory()
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert synthesis.attempts == 2
    assert synthesis.prose.lang_flags == ()


def test_synthesize_memory_gives_up_after_retries_exhausted() -> None:
    model = Scripted(("The council approved.", "Still English."))
    memory = Memory()
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=1)
    assert synthesis.attempts == 2
    assert synthesis.prose.lang_flags != ()


def test_synthesize_memory_zero_retries_makes_exactly_one_attempt() -> None:
    model = Scripted(("The council approved.",))
    memory = Memory()
    synthesis = synthesize_memory(memory, model, token_len=heuristic_token_len, retries=0)
    assert synthesis.attempts == 1
    assert len(model.calls) == 1


def test_synthesize_memory_records_usage() -> None:
    model = Scripted(("會議討論搬遷案，最終決議遷至 B 棟。",))
    memory = Memory()
    usage = Usage()
    synthesize_memory(memory, model, token_len=heuristic_token_len, usage=usage)
    assert usage.calls == 1
    assert usage.prefill_tokens > 0


def test_run_agent_uses_synth_model_when_given() -> None:
    step_model = Scripted(("NOP",))
    synth_model = Scripted(("會議討論搬遷案，最終決議遷至 B 棟。",))
    trace = run_agent(meeting(5), step_model, synth_model=synth_model)
    assert len(step_model.calls) == 1  # only reading steps
    assert len(synth_model.calls) == 1  # only the synthesis call
    assert trace.synthesis.prose.text == "會議討論搬遷案，最終決議遷至 B 棟。"


def test_run_agent_uses_the_same_model_for_synthesis_by_default() -> None:
    model = Scripted(("NOP", "會議討論搬遷案，最終決議遷至 B 棟。"))
    trace = run_agent(meeting(5), model)
    assert len(model.calls) == 2
    assert trace.synthesis is not None


# --- prompt/tokenize version discipline ------------------------------------------------


def test_trace_records_prompt_and_tokenize_version() -> None:
    model = Scripted()
    trace = run_agent(meeting(5), model, synthesize=False)
    assert trace.prompt_version == "sys-v1"
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
