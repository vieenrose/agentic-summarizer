"""CURSOR loop, prompts, and the screen — driven by a scripted fake model.

No GPU, no weights: the fake model is a list of op strings, which is what makes §6 and the
screen logic provable in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from screen import score_meeting, screen_model  # noqa: E402

from voxsum.agent import STEP_BUDGET, StepBudgetExceeded, run_cursor
from voxsum.chunker import heuristic_token_len
from voxsum.prompts import PROMPT_VERSION, build_step_prompt, function_declarations, system_prompt
from voxsum.screenset import screen_meetings
from voxsum.state import NotesState
from voxsum.transcript import Utterance, parse_transcript

EXAMPLE = (
    "[0:00] S1: Let us discuss the office move.\n"
    "[2:30] S2: I propose we move to Building B.\n"
    "[5:10] S1: Agreed, Building B it is.\n"
)


class Scripted:
    """Replays canned responses, recording the prompts it was given."""

    def __init__(self, *responses: str, default: str = "NOP") -> None:
        self.responses = list(responses)
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.responses.pop(0) if self.responses else self.default


# --- prompts -------------------------------------------------------------------

def test_system_prompt_languages_and_version() -> None:
    assert "ADD <SECTION>" in system_prompt("en")
    assert "ADD <SECTION>" in system_prompt("zh-TW")
    assert "只回覆編輯指令" in system_prompt("zh-TW")
    assert PROMPT_VERSION == "sys-v1"


def test_system_prompt_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        system_prompt("fr")


def test_declarations_carry_functiongemma_shape() -> None:
    decl = function_declarations()
    # The trigger phrase is load-bearing per FunctionGemma's documented format.
    assert decl.startswith("You are a model that can do function calling")
    assert "<start_function_declaration>" in decl and "<end_function_declaration>" in decl
    for op in ("ADD", "UPD", "DEL", "TITLE", "NOP"):
        assert f"declaration:{op}{{" in decl
    assert "<escape>" in decl


def test_system_prompt_can_include_declarations() -> None:
    assert "<start_function_declaration>" in system_prompt("en", declarations=True)
    assert "<start_function_declaration>" not in system_prompt("en")


def test_step_prompt_order_is_state_then_chunk() -> None:
    from voxsum.chunker import Chunk

    prompt = build_step_prompt(NotesState(), Chunk(0, tuple(parse_transcript(EXAMPLE))))
    assert prompt.index("STATE:") < prompt.index("CHUNK:")
    assert "[2:30] S2: I propose we move to Building B." in prompt


# --- the loop ------------------------------------------------------------------

def test_loop_applies_ops_and_records_steps() -> None:
    model = Scripted(
        "TITLE: Office move decision\nADD TOPICS - Office move [0:00]",
        "ADD DECISIONS - Move to Building B [2:30]",
        "UPD DECISIONS «Move to» -> Move to Building B agreed [5:10]",
    )
    trace = run_cursor(parse_transcript(EXAMPLE), model, budget=20)
    assert trace.state.title == "Office move decision"
    assert trace.state.bullets("DECISIONS")[0].text == "Move to Building B agreed"
    assert len(trace.steps) == len(model.calls)
    assert trace.prompt_version == PROMPT_VERSION


def test_no_history_crosses_steps() -> None:
    """STATE is the entire memory (CLAUDE.md §4) — each call gets exactly one chunk."""
    model = Scripted(default="NOP")
    run_cursor(parse_transcript(EXAMPLE), model, budget=20)
    assert len(model.calls) > 1
    for _, user in model.calls:
        assert user.count("CHUNK:") == 1
        assert user.count("STATE:") == 1


def test_state_visible_to_next_step() -> None:
    model = Scripted("ADD DECISIONS - Move to Building B [0:00]", default="NOP")
    run_cursor(parse_transcript(EXAMPLE), model, budget=20)
    # The second step must see the first step's bullet in its STATE block.
    assert "Move to Building B" in model.calls[1][1]


def test_over_budget_step_raises_rather_than_truncating() -> None:
    lines = [Utterance(i * 10, "S1", "word " * 200) for i in range(4)]
    with pytest.raises(StepBudgetExceeded, match="rather than truncating"):
        run_cursor(lines, Scripted(), budget=4096, step_budget=64)


def test_budget_uses_injected_tokenizer() -> None:
    """The budget must be decided by the caller's tokenizer, not the heuristic."""
    calls: list[str] = []

    def counting(text: str) -> int:
        calls.append(text)
        return heuristic_token_len(text)

    run_cursor(parse_transcript(EXAMPLE), Scripted(), budget=64, token_len=counting)
    assert calls, "injected token_len was never consulted"


def test_metrics_are_reported() -> None:
    model = Scripted(
        "ADD TOPICS - Office move [0:00]",  # anchored natively
        "ADD DECISIONS - Building B [99:99]",  # falls to the matcher
        default="NOP",
    )
    trace = run_cursor(parse_transcript(EXAMPLE), model, budget=20)
    assert trace.valid_op_rate == 1.0
    assert trace.anchor_rate_raw == pytest.approx(0.5)


def test_malformed_output_lowers_valid_op_rate() -> None:
    model = Scripted("please summarise the meeting", default="NOP")
    trace = run_cursor(parse_transcript(EXAMPLE), model, budget=20)
    assert trace.valid_op_rate == 0.0


def test_coverage_gap_recorded_on_nop_collapse() -> None:
    lines = [
        Utterance(i * 30, "S1", f"substantive discussion point number {i} " * 12)
        for i in range(9)
    ]
    trace = run_cursor(lines, Scripted(default="NOP"), budget=256)
    assert trace.coverage_gaps, "K consecutive NOPs over rich chunks must be logged"
    assert trace.nop_rate_on_rich_chunks == 1.0


def test_default_step_budget_fits_the_spec_window() -> None:
    assert STEP_BUDGET == 4096


# --- screen set and scoring ----------------------------------------------------

def test_screen_meetings_plant_the_required_facts() -> None:
    for meeting in screen_meetings():
        assert meeting.rejected_at < meeting.approved_at, "chain must be rejected -> approved"
        assert len(meeting.deadlines_at) == 2
        assert meeting.line_at(meeting.trap_at) is not None
        assert meeting.line_at(meeting.approved_at) is not None
        assert meeting.subject_terms and meeting.trap_terms


class PerfectAgent:
    """A model that handles the planted chain correctly: revises rather than appends.

    Bullets must be written in the meeting's own language — the scorer looks for the
    planted subject terms, and polarity detection is per-language too.
    """

    # (subject, prefix, rejected, approved, action) per language.
    WORDING = {
        "en": (
            "warehouse consolidation",
            "wareho",
            "warehouse consolidation plan rejected as it stands",
            "warehouse consolidation plan approved after recosting",
            "owner: deliver warehouse consolidation milestone",
        ),
        "zh-TW": (
            "倉庫整併",
            "倉庫整併方案",
            "倉庫整併方案否決",
            "倉庫整併方案通過",
            "負責人：完成倉庫整併里程碑",
        ),
    }

    def __init__(self, meeting) -> None:
        self.meeting = meeting
        _, self.prefix, self.rejected, self.approved, self.action = self.WORDING[meeting.lang]

    def __call__(self, system: str, user: str) -> str:
        m = self.meeting
        ops: list[str] = []
        for u in m.utterances:
            stamp = f"[{u.clock}]"
            if f"{stamp} " not in user:
                continue
            if u.start == m.rejected_at:
                ops.append(f"ADD DECISIONS - {self.rejected} {stamp}")
            elif u.start == m.approved_at:
                ops.append(f"UPD DECISIONS «{self.prefix}» -> {self.approved} {stamp}")
            elif u.start in m.deadlines_at:
                # Distinct text per deadline: an identical bullet is refused as duplicate.
                ops.append(f"ADD ACTIONS - {self.action} {u.clock} {stamp}")
        return "\n".join(ops) if ops else "NOP"


def test_perfect_agent_passes_g1() -> None:
    for meeting in screen_meetings():
        trace = run_cursor(
            list(meeting.utterances), PerfectAgent(meeting), lang=meeting.lang, budget=96
        )
        result = score_meeting(trace, meeting)
        assert result.chain_correct, result.summary()
        assert result.both_deadlines, result.summary()
        assert result.fully_anchored, result.summary()
        assert result.trap_absent, result.summary()
        assert result.passed_g1
        assert result.revised_at_contradiction


def test_appending_agent_fails_the_chain() -> None:
    """The failure the screen exists to catch: both states asserted, notes self-contradictory.

    A naive keyword check would pass this — 'approved' is present. The chain check must not.
    """

    class Appender(PerfectAgent):
        def __call__(self, system: str, user: str) -> str:
            out = super().__call__(system, user)
            return out.replace(f"UPD DECISIONS «{self.prefix}» ->", "ADD DECISIONS -")

    meeting = screen_meetings()[0]
    trace = run_cursor(list(meeting.utterances), Appender(meeting), lang=meeting.lang, budget=96)
    result = score_meeting(trace, meeting)
    assert not result.revised_at_contradiction
    assert not result.passed_g1


def test_trap_topic_is_detected() -> None:
    meeting = screen_meetings()[0]

    class TrapAgent(PerfectAgent):
        def __call__(self, system: str, user: str) -> str:
            out = super().__call__(system, user)
            if f"[{meeting.line_at(meeting.trap_at).clock}] " in user:
                clock = meeting.line_at(meeting.trap_at).clock
                return f"ADD TOPICS - Office coffee machine budget [{clock}]"
            return out

    trace = run_cursor(list(meeting.utterances), TrapAgent(meeting), budget=96)
    result = score_meeting(trace, meeting)
    assert not result.trap_absent
    assert not result.passed_g1


def test_nop_only_agent_fails_everything() -> None:
    meeting = screen_meetings()[0]
    trace = run_cursor(list(meeting.utterances), Scripted(default="NOP"), budget=96)
    result = score_meeting(trace, meeting)
    assert not result.passed_g1
    assert not result.chain_correct and not result.both_deadlines


def test_screen_model_runs_both_languages() -> None:
    results = screen_model(Scripted(default="NOP"), budget=96)
    assert {r.lang for r, _, _ in results} == {"en", "zh-TW"}
    assert all(r.prompt_version == PROMPT_VERSION for r, _, _ in results)


def test_valid_op_rate_never_exceeds_one() -> None:
    """Regression: NOP counted in the numerator but not the denominator gave >100%."""
    model = Scripted(
        "NOP\nADD TOPICS - Office move [0:00]",
        "NOP\nADD OPEN - Parking [0:00]\ngarbage line",
        default="NOP",
    )
    trace = run_cursor(parse_transcript(EXAMPLE), model, budget=20)
    assert trace.valid_op_rate is not None
    assert 0.0 <= trace.valid_op_rate <= 1.0
    # Two ADDs applied, one malformed line refused.
    assert trace.valid_op_rate == pytest.approx(2 / 3)


# --- op filter (judge-filtered trace generation) --------------------------------

def test_op_filter_vetoes_before_state() -> None:
    """A vetoed op must not reach STATE, or the next step conditions on a bullet the
    student was never taught to produce."""
    model = Scripted(
        "ADD DECISIONS - unverifiable claim [0:00]\nADD TOPICS - Office move [0:00]",
        default="NOP",
    )

    def veto_decisions(op, chunk):
        return "judge: UNSUPPORTED" if getattr(op, "section", None) == "DECISIONS" else None

    trace = run_cursor(parse_transcript(EXAMPLE), model, budget=20, op_filter=veto_decisions)
    assert trace.state.bullets("DECISIONS") == [], "vetoed op leaked into STATE"
    assert [b.text for b in trace.state.bullets("TOPICS")] == ["Office move"]
    assert trace.steps[0].vetoed == (("ADD DECISIONS - unverifiable claim [0:00]",
                                     "judge: UNSUPPORTED"),)


def test_op_filter_absent_keeps_everything() -> None:
    model = Scripted("ADD DECISIONS - kept [0:00]", default="NOP")
    trace = run_cursor(parse_transcript(EXAMPLE), model, budget=20)
    assert len(trace.state.bullets("DECISIONS")) == 1
    assert all(not s.vetoed for s in trace.steps)


def test_vetoed_ops_are_not_counted_as_malformed() -> None:
    """A veto is a quality decision, not a protocol failure — it must not depress GT1."""
    model = Scripted("ADD TOPICS - dropped [0:00]", default="NOP")
    trace = run_cursor(
        parse_transcript(EXAMPLE), model, budget=20, op_filter=lambda op, c: "judge: UNSUPPORTED"
    )
    assert trace.valid_op_rate is None, "no ops reached the harness, so GT1 has no denominator"


def test_step_filter_judges_ops_together() -> None:
    """Per-op vetoing serialises network calls; a step filter sees them all at once.

    Measured: the judge filter added ~19s per step as 5 sequential HTTPS round-trips while
    both GPUs sat idle. The ops of a step are independent, so they can go concurrently.
    """
    seen: list[int] = []

    def batch(ops, chunk):
        seen.append(len(ops))
        kept = [o for o in ops if getattr(o, "section", None) != "DECISIONS"]
        vetoed = [
            (o, "judge: UNSUPPORTED")
            for o in ops
            if getattr(o, "section", None) == "DECISIONS"
        ]
        return kept, vetoed

    model = Scripted(
        "ADD DECISIONS - dropped [0:00]\nADD TOPICS - kept [0:00]\nADD OPEN - kept too [0:00]",
        default="NOP",
    )
    trace = run_cursor(parse_transcript(EXAMPLE), model, budget=20, step_filter=batch)
    assert seen[0] == 3, "all of a step's ops must reach the filter in one call"
    assert trace.state.bullets("DECISIONS") == []
    assert len(trace.state.bullets("TOPICS")) == 1
    assert trace.steps[0].vetoed == (("ADD DECISIONS - dropped [0:00]", "judge: UNSUPPORTED"),)


def test_step_filter_takes_precedence_over_op_filter() -> None:
    model = Scripted("ADD TOPICS - x [0:00]", default="NOP")
    trace = run_cursor(
        parse_transcript(EXAMPLE),
        model,
        budget=20,
        op_filter=lambda op, c: "should not run",
        step_filter=lambda ops, c: (ops, []),
    )
    assert len(trace.state.bullets("TOPICS")) == 1
