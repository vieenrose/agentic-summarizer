"""The CURSOR loop (CLAUDE.md §4).

One evolving STATE, one chunk per step, no conversation history across steps. The model is
a plain callable — `(system, user) -> str` — so the loop is identical for the teacher, the
student, and a scripted fake in tests.

The per-step budget is normative (PLAN.md §2c): `token_len` is injected and the loop
refuses to send an over-budget prompt rather than silently truncating it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .chunker import CHUNK_TOKENS, Chunk, heuristic_token_len, iter_chunks
from .guards import Outcome, apply_ops
from .ops import Add, Nop, Op, Upd, parse_ops, render_op
from .prompts import PROMPT_VERSION, build_step_prompt, system_prompt
from .state import NotesState
from .transcript import Utterance

__all__ = [
    "OpFilter",
    "Step",
    "StepBudgetExceeded",
    "StepFilter",
    "Trace",
    "Usage",
    "run_cursor",
]

# SYS ~250 + STATE <= 600 + CHUNK 2048 ~= 2.9k in, inside the 4k window (CLAUDE.md §8).
STEP_BUDGET = 4096

ModelFn = Callable[[str, str], str]

#: Veto hook for candidate ops, applied *before* they reach STATE. Returns None to keep the
#: op, or a reason string to drop it. Used by trace generation to keep only bullets a judge
#: can verify — see `train/gen_traces.py`. Kept out of `guards.py` on purpose: the guards are
#: deterministic and model-free (CLAUDE.md §5.3), and a judge is neither.
OpFilter = Callable[[Op, Chunk], str | None]

#: Step-level form of the same hook: given all of a step's ops, return (kept, [(op, reason)]).
#: Preferred when the veto costs a network round-trip — a step's ops are independent, so they
#: can be verified concurrently, and a per-op filter forces them into sequence.
StepFilter = Callable[[list[Op], Chunk], tuple[list[Op], list[tuple[Op, str]]]]


class StepBudgetExceeded(RuntimeError):
    """A step's prompt exceeded the student's per-step budget (PLAN.md §2c).

    Raised rather than truncated: a silently shortened chunk produces a trace whose ops
    reference lines the student never saw, which is worse than a loud failure.
    """


@dataclass
class Usage:
    """Token and call accounting — the instrument GT4 is measured with (CLAUDE.md §7.4).

    Both arms (CURSOR and the map-reduce baseline) fill this in the same way, with the same
    `token_len`, so a prefill comparison is like-for-like rather than two estimates.
    """

    calls: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0

    def record(self, prefill: int, decode: int) -> None:
        self.calls += 1
        self.prefill_tokens += prefill
        self.decode_tokens += decode


@dataclass(frozen=True, slots=True)
class Step:
    """One step's full record — enough to regenerate or audit it later."""

    index: int
    system: str
    user: str
    raw: str
    ops: tuple[Op, ...]
    outcome: Outcome
    prompt_tokens: int
    state_before: str
    chunk: Chunk
    #: (rendered op, reason) for ops the filter vetoed before they reached STATE.
    vetoed: tuple[tuple[str, str], ...] = ()

    @property
    def is_nop(self) -> bool:
        return all(isinstance(o, Nop) for o in self.ops) if self.ops else True


@dataclass
class Trace:
    """The result of one meeting: final STATE plus every step."""

    state: NotesState
    steps: list[Step] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    coverage_gaps: list[int] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def valid_op_rate(self) -> float | None:
        """Meeting-level GT1: applied / scored op lines, pooled across steps.

        NOP is excluded from numerator *and* denominator. Counting it in only one — as an
        earlier version did via `Outcome.applied` — yields rates above 100% on any step
        that mixes NOP with real ops.
        """
        scored = [
            r
            for s in self.steps
            for r in s.outcome.results
            if not isinstance(r.op, Nop)
        ]
        if not scored:
            return None
        return sum(1 for r in scored if r.applied) / len(scored)

    @property
    def nop_rate_on_rich_chunks(self) -> float | None:
        rich = [s for s in self.steps if s.chunk.is_content_rich()]
        return sum(1 for s in rich if s.is_nop) / len(rich) if rich else None

    @property
    def anchor_rate_raw(self) -> float | None:
        """Fraction of applied ADD/UPD ops whose own anchor resolved (no matcher).

        Reported separately from the post-matcher rate because it is the honest signal of
        whether the model can copy a timestamp (PLAN.md §5).
        """
        # Only ADD/UPD carry anchors. NOP and TITLE are applied with no reason set, so
        # counting every applied op would score them as native anchors and inflate this.
        scored = [
            r
            for s in self.steps
            for r in s.outcome.results
            if r.applied and isinstance(r.op, (Add, Upd))
        ]
        if not scored:
            return None
        native = sum(1 for r in scored if not (r.reason and "matcher" in r.reason))
        return native / len(scored)


def run_cursor(
    utterances: list[Utterance],
    model: ModelFn,
    *,
    lang: str = "en",
    declarations: bool = False,
    budget: int = CHUNK_TOKENS,
    step_budget: int = STEP_BUDGET,
    token_len=heuristic_token_len,
    state: NotesState | None = None,
    op_filter: OpFilter | None = None,
    step_filter: StepFilter | None = None,
) -> Trace:
    """Stream `utterances` past `model`, curating one NOTES state.

    `token_len` should be the *student's* tokenizer for anything normative — the default
    heuristic is for tests and quick local runs only.

    `op_filter` vetoes candidate ops before they are applied. Filtering here rather than at
    write time is deliberate: a vetoed op must not enter STATE either, or the next step
    conditions on a bullet the student was never taught to produce, and the trace stops
    being on-policy for the state trajectory the student will actually see.
    """
    sys = system_prompt(lang, declarations=declarations)
    trace = Trace(state=state or NotesState())
    consecutive_nops = 0

    for chunk in iter_chunks(utterances, budget=budget, token_len=token_len):
        state_before = build_step_prompt(trace.state, chunk)
        prompt_tokens = token_len(sys) + token_len(state_before)
        if prompt_tokens > step_budget:
            raise StepBudgetExceeded(
                f"step {chunk.index}: {prompt_tokens} tokens > budget {step_budget}. "
                "Lower the chunk budget rather than truncating the prompt."
            )

        rendered_state = build_step_prompt(trace.state, Chunk(chunk.index, ()))
        raw = model(sys, state_before)
        trace.usage.record(prompt_tokens, token_len(raw))
        ops = parse_ops(raw)

        vetoed: list[tuple[str, str]] = []
        if step_filter is not None:
            ops, rejected = step_filter(ops, chunk)
            vetoed = [(render_op(op), reason) for op, reason in rejected]
        elif op_filter is not None:
            kept: list[Op] = []
            for op in ops:
                reason = op_filter(op, chunk)
                if reason is None:
                    kept.append(op)
                else:
                    vetoed.append((render_op(op), reason))
            ops = kept

        outcome = apply_ops(trace.state, ops, chunk, consecutive_nops=consecutive_nops)

        step = Step(
            index=chunk.index,
            system=sys,
            user=state_before,
            raw=raw,
            ops=tuple(ops),
            outcome=outcome,
            prompt_tokens=prompt_tokens,
            state_before=rendered_state,
            chunk=chunk,
            vetoed=tuple(vetoed),
        )
        trace.steps.append(step)

        substantive = outcome.applied > 0 and not step.is_nop
        consecutive_nops = 0 if substantive else consecutive_nops + 1
        if outcome.nop_collapse:
            # Logged for the coverage fallback (CLAUDE.md §6.3); the fallback itself is
            # baseline.py's job, kept out of the agent protocol by design.
            trace.coverage_gaps.append(chunk.index)
            consecutive_nops = 0

    return trace
