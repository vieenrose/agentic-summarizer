"""The CURSOR-style step loop, plus the net-new `SYNTHESIZE` call (SPEC §4.1).

The transcript is read as a stream. The harness owns the memory; the model only emits
edit lines. **No conversation history crosses steps** — memory is the entire
carry-forward, which is what keeps each step's context constant-size and learnable at
1B. Every call is a fresh `(system, user) -> str` invocation; nothing about a prior
call's prompt or response is visible to the next one except through the memory itself.

**The `SYNTHESIZE` call has no analogue in the prior project.** That project's final
output was a deterministic render of the accumulated bullets. Here the transcript being
exhausted triggers one additional, differently-shaped model call: `synthesize_memory`
turns the final memory into the SPEC §3 prose product. Its output-length regime
(<1,000 tokens of prose) is entirely different from a reading step's (~150 tokens of
edit lines), which is why it is a separate public function rather than one more branch
of the reading loop — `agent_or_teacher(sys, user) -> str` stays the one seam both calls
share, but SYS text, sampling, and validation all differ.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from arcsum.chunker import CHUNK_TOKENS, Chunk, iter_chunks
from arcsum.guards import Outcome, apply_ops
from arcsum.memory import Memory
from arcsum.ops import Arc, Drop, Nop, Op, parse_ops, render_op
from arcsum.prompts import (
    PROMPT_VERSION,
    build_memory_view,
    build_step_prompt,
    build_synth_prompt,
    step_system_prompt,
    synth_system_prompt,
)
from arcsum.prose import Prose, finalize, ungrounded_numbers
from arcsum.tokens import TOKENIZE_VERSION, heuristic_token_len, token_len_name
from arcsum.transcript import Utterance

#: SPEC §4.1's table: ~250 SYS + <=600 memory + ~2,500 chunk ~= 3,500. Set above that
#: measured total, mirroring the headroom the prior project's STEP_BUDGET left above its
#: own ~2.9k estimate — real prompts vary, and this must fail loud, not truncate.
STEP_BUDGET = 3800

#: A one-line nudge appended to the user turn on a `nop_retry` re-ask. Deliberately
#: minimal: it must not change the op grammar or the model's task, only press on the
#: judgment call.
_NOP_RETRY_NUDGE = (
    "\n\n（提醒：請再次檢視這段內容是否包含值得記錄的重點。"
    "若有，請輸出至少一項 ADD 或 ARC 指令，而非 NOP。）"
)

#: A one-line nudge appended to the synthesis user turn when `ungrounded_numbers`
#: flags the previous attempt. Same minimal-pressure principle as `_NOP_RETRY_NUDGE`:
#: name the failure, don't change the task.
_GROUNDING_RETRY_NUDGE = (
    "\n\n（提醒：摘要中只能使用上述記憶中出現的數字、日期與金額，不要新增記憶中沒有的具體細節。）"
)

ModelFn = Callable[[str, str], str]
OpFilter = Callable[[Op, Chunk], "str | None"]
StepFilter = Callable[[list[Op], Chunk], tuple[list[Op], list[tuple[Op, str]]]]


class StepBudgetExceeded(RuntimeError):
    """Raised rather than silently truncating a prompt.

    A silently shortened chunk would produce a trace whose ops reference lines the
    student never saw, which is worse than a loud failure — lower the chunk budget
    instead of catching this.
    """


@dataclass
class Usage:
    calls: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    #: SPEC §7 needs both of these; the prior project's `Usage` carried neither.
    wall_seconds: float = 0.0
    peak_rss_bytes: int = 0

    def record(self, prefill: int, decode: int, seconds: float = 0.0) -> None:
        self.calls += 1
        self.prefill_tokens += prefill
        self.decode_tokens += decode
        self.wall_seconds += seconds


@dataclass(frozen=True, slots=True)
class Step:
    index: int
    system: str
    user: str
    raw: str
    ops: tuple[Op, ...]
    outcome: Outcome
    prompt_tokens: int
    memory_before: str
    chunk: Chunk
    vetoed: tuple[tuple[str, str], ...] = ()
    seconds: float = 0.0
    #: True if a `nop_retry` re-ask replaced this step's original attempt.
    retried: bool = False

    @property
    def is_nop(self) -> bool:
        """True if every op (after filtering) was `NOP`, or nothing parsed at all."""
        return all(isinstance(op, Nop) for op in self.ops)


#: Returned verbatim by `synthesize_memory` when the memory is empty, INSTEAD of
#: calling the model. Deliberately states only what is actually known.
EMPTY_MEMORY_PROSE = "本次會議沒有記錄到具體的決議或討論重點。"


@dataclass(frozen=True, slots=True)
class Synthesis:
    system: str
    user: str
    raw: str
    prose: Prose
    attempts: int
    #: True when the model was NEVER CALLED because the memory was empty, and
    #: `prose.text` is the fixed `EMPTY_MEMORY_PROSE` string rather than generated
    #: output. Distinct from `attempts == 0` being merely incidental: downstream
    #: scoring must be able to tell "the system declined to invent content" apart
    #: from "the system produced a summary", since the two mean opposite things
    #: about a run.
    skipped_empty_memory: bool = False
    #: Arabic-digit spans `prose.ungrounded_numbers` flagged as absent from the memory
    #: on the FINAL attempt — non-empty means the guard fired and retries did not clear
    #: it (or `retries=0`). Recorded rather than silently swallowed, same "detect and
    #: record" discipline as `guards.py`'s outcomes: a partial guard that fails open
    #: must still leave a visible trace of what it caught.
    ungrounded_numbers: tuple[str, ...] = ()


@dataclass
class Trace:
    memory: Memory
    steps: list[Step] = field(default_factory=list)
    synthesis: Synthesis | None = None
    prompt_version: str = PROMPT_VERSION
    tokenize_version: str = TOKENIZE_VERSION
    token_len_name: str = "heuristic"
    #: The actual chunking budget this run used. `nop_rate_on_rich_chunks` MUST measure
    #: richness against this, not the module default — the same class of bug as
    #: `apply_ops`'s `budget` parameter guards against, one level up.
    budget: int = CHUNK_TOKENS
    #: Chunk indices where `NOP_COLLAPSE_K` consecutive NOPs fired on a content-rich
    #: chunk. Reported, never repaired — see `guards.apply_ops`'s docstring for why.
    coverage_gaps: list[int] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def valid_op_rate(self) -> float | None:
        """Applied / attempted across the whole trace, `Nop` excluded from both sides."""
        non_nop = [
            r for step in self.steps for r in step.outcome.results if not isinstance(r.op, Nop)
        ]
        if not non_nop:
            return None
        return sum(1 for r in non_nop if r.applied) / len(non_nop)

    @property
    def nop_rate_on_rich_chunks(self) -> float | None:
        """SPEC §8 risk 3's monitored quantity: NOP share on chunks worth an op."""
        rich = [s for s in self.steps if s.chunk.is_content_rich(budget=self.budget)]
        if not rich:
            return None
        return sum(1 for s in rich if s.is_nop) / len(rich)

    @property
    def drop_rate(self) -> float | None:
        """Share of attempted non-NOP ops that were `DROP` — a revision-share proxy."""
        non_nop = [
            r for step in self.steps for r in step.outcome.results if not isinstance(r.op, Nop)
        ]
        if not non_nop:
            return None
        return sum(1 for r in non_nop if isinstance(r.op, Drop)) / len(non_nop)

    @property
    def arc_rate(self) -> float | None:
        """Share of attempted non-NOP ops that were `ARC`."""
        non_nop = [
            r for step in self.steps for r in step.outcome.results if not isinstance(r.op, Nop)
        ]
        if not non_nop:
            return None
        return sum(1 for r in non_nop if isinstance(r.op, Arc)) / len(non_nop)


def run_agent(
    utterances: list[Utterance],
    model: ModelFn,
    *,
    synth_model: ModelFn | None = None,
    budget: int = CHUNK_TOKENS,
    step_budget: int = STEP_BUDGET,
    token_len: Callable[[str], int] = heuristic_token_len,
    memory: Memory | None = None,
    op_filter: OpFilter | None = None,
    step_filter: StepFilter | None = None,
    on_step: Callable[[int, float, int], None] | None = None,
    synthesize: bool = True,
    nop_retry: bool = False,
) -> Trace:
    """Read `utterances` chunk by chunk, curating `memory` via edit lines, then
    (unless `synthesize=False`) run the final `SYNTHESIZE` call.

    `token_len` is the single source of truth for this whole run: it drives chunking,
    it is force-assigned onto the memory object (whether newly created or passed in), and
    it measures every prompt. A caller using the real MiniCPM5 tokenizer for chunking
    must not silently get memory caps measured by the heuristic instead — that is exactly
    the class of divergence this design exists to prevent.
    """
    mem = memory if memory is not None else Memory()
    mem.token_len = token_len

    sys = step_system_prompt()
    trace = Trace(
        memory=mem,
        prompt_version=PROMPT_VERSION,
        tokenize_version=TOKENIZE_VERSION,
        token_len_name=token_len_name(token_len),
        budget=budget,
    )
    consecutive_nops = 0

    def call(
        chunk: Chunk, user: str
    ) -> tuple[str, float, list[Op], list[tuple[str, str]], Outcome]:
        started = time.monotonic()
        raw = model(sys, user)
        elapsed = time.monotonic() - started
        trace.usage.record(token_len(sys) + token_len(user), token_len(raw), elapsed)

        ops = parse_ops(raw)
        vetoed: list[tuple[str, str]] = []
        if op_filter is not None:
            kept: list[Op] = []
            for op in ops:
                op_reason = op_filter(op, chunk)
                if op_reason is None:
                    kept.append(op)
                else:
                    vetoed.append((render_op(op), op_reason))
            ops = kept
        if step_filter is not None:
            ops, filtered = step_filter(ops, chunk)
            vetoed.extend((render_op(op), reason) for op, reason in filtered)

        outcome = apply_ops(
            trace.memory, ops, chunk, consecutive_nops=consecutive_nops, budget=budget
        )
        return raw, elapsed, ops, vetoed, outcome

    for chunk in iter_chunks(utterances, budget=budget, token_len=token_len):
        memory_before = build_memory_view(trace.memory)
        user = build_step_prompt(trace.memory, chunk)
        prompt_tokens = token_len(sys) + token_len(user)
        if prompt_tokens > step_budget:
            raise StepBudgetExceeded(
                f"step {chunk.index}: {prompt_tokens} tokens > budget {step_budget}. "
                "Lower the chunk budget rather than truncating the prompt."
            )

        raw, elapsed, ops, vetoed, outcome = call(chunk, user)
        retried = False
        if nop_retry and outcome.nop_collapse:
            retry_user = user + _NOP_RETRY_NUDGE
            raw2, elapsed2, ops2, vetoed2, outcome2 = call(chunk, retry_user)
            raw, elapsed, ops, vetoed, outcome = raw2, elapsed + elapsed2, ops2, vetoed2, outcome2
            user = retry_user
            retried = True

        step = Step(
            index=chunk.index,
            system=sys,
            user=user,
            raw=raw,
            ops=tuple(ops),
            outcome=outcome,
            prompt_tokens=prompt_tokens,
            memory_before=memory_before,
            chunk=chunk,
            vetoed=tuple(vetoed),
            seconds=elapsed,
            retried=retried,
        )
        trace.steps.append(step)

        if on_step is not None:
            on_step(chunk.index, elapsed, len(ops))

        substantive = outcome.applied > 0
        consecutive_nops = 0 if substantive else consecutive_nops + 1
        if outcome.nop_collapse:
            trace.coverage_gaps.append(chunk.index)
            consecutive_nops = 0

    if synthesize:
        trace.synthesis = synthesize_memory(
            trace.memory, synth_model or model, token_len=token_len, usage=trace.usage
        )

    return trace


def synthesize_memory(
    memory: Memory,
    model: ModelFn,
    *,
    token_len: Callable[[str], int] = heuristic_token_len,
    usage: Usage | None = None,
    retries: int = 1,
) -> Synthesis:
    """The final `SYNTHESIZE` call: memory -> SPEC §3 prose.

    A separate public function (not folded into `run_agent`'s loop) so supervision can
    build the synthesis training target from a stored final memory without replaying a
    transcript, and so tests can drive it with no transcript at all.

    Retries (up to `retries` extra attempts) ONLY on a hard contract failure — over the
    token budget, a language-guard flag, or `prose.ungrounded_numbers` flagging a
    fabricated digit span (see that function's docstring for what it does and does not
    catch) — never on stylistic dissatisfaction, since there is no deterministic way to
    judge "good enough" prose here.

    **An empty memory short-circuits: the model is not called at all.** With no arc and
    no points there is, by construction, nothing to summarise — so ANY generated prose
    is unfaithful, not merely low-quality, and no amount of retrying or better sampling
    can make it faithful. Measured, not hypothetical: the G1 revision probe (2026-08-26,
    `runs/sft-pilot-v1`) fed the fine-tuned student two short meetings, the reading
    steps NOP'd both, and `SYNTHESIZE` then produced fluent, specific, entirely
    fictional summaries about land-use zoning — the most common agenda topic in the
    training corpus, i.e. the model's own prior filling a vacuum. This guard is
    therefore a correctness invariant of the same family as `guards.apply_ops` refusing
    an op rather than repairing it, not a quality heuristic.

    Note what this does and does not fix: it makes an empty memory produce an HONEST
    output instead of a fabricated one. It does not make the memory correct — a memory
    that is empty when it should not be is an upstream curation failure, and this guard
    deliberately does not paper over it (`skipped_empty_memory` is set precisely so it
    stays visible).
    """
    sys = synth_system_prompt()
    user = build_synth_prompt(memory)

    if memory.is_empty():
        return Synthesis(
            system=sys,
            user=user,
            raw="",
            prose=finalize(EMPTY_MEMORY_PROSE, token_len=token_len),
            attempts=0,
            skipped_empty_memory=True,
        )

    grounding = memory.arc + " " + " ".join(p.text for p in memory.points)

    raw = ""
    prose: Prose | None = None
    flagged: tuple[str, ...] = ()
    attempts = 0
    nudged = False
    for _ in range(retries + 1):
        attempts += 1
        started = time.monotonic()
        raw = model(sys, user)
        elapsed = time.monotonic() - started
        if usage is not None:
            usage.record(token_len(sys) + token_len(user), token_len(raw), elapsed)
        prose = finalize(raw, token_len=token_len)
        flagged = ungrounded_numbers(prose.text, grounding)
        if not prose.over_budget and not prose.lang_flags and not flagged:
            break
        if flagged and not nudged:
            user = user + _GROUNDING_RETRY_NUDGE
            nudged = True

    assert prose is not None  # loop runs at least once (retries + 1 >= 1)
    return Synthesis(
        system=sys, user=user, raw=raw, prose=prose, attempts=attempts, ungrounded_numbers=flagged
    )
