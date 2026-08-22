"""Per-step teacher supervision (SPEC §4.2 steps 1-3): converting MeetingBank's real,
human-authored item minutes into ADD/DROP/ARC/NOP edit lines against a real, live
`Memory` object -- walking chunks in the same order and using the same harness
(`ops.parse_ops`, `guards.apply_ops`) the deployed agent uses, so a gold sequence
produced here is, by construction, a sequence the real harness can replay.

**Grounding, not free-run summarization.** The deployed model (and `agent.run_agent`)
never sees anything beyond the transcript itself. The teacher here is deliberately
shown MORE: the human-authored minute(s) overlapping the current chunk (SPEC §4.2 step
1's "narrow, grounded conversion task"), which items have already concluded (for the
ARC line, step 3), and a short look at the immediately upcoming item (so a decision
reversed almost immediately is caught as a DROP+ADD rather than two disconnected ADDs
-- SPEC step 1: "that foresight is used only to emit DROP for points a later segment
supersedes"). None of this grounding is part of the DEPLOYED prompt shape
(`prompts.build_step_prompt`) -- it exists only here, offline, to build supervision.

**Uncovered chunks (step 2) are not blanket-NOP'd.** SPEC step 2: MeetingBank's ~60s/
~10-word filter excluded real content, not just filler. An uncovered chunk is shown
its nearest neighbouring items and asked to judge continuity, not simply told to skip.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import lru_cache

from arcsum.agent import ModelFn, Step, Trace
from arcsum.chunker import CHUNK_TOKENS, Chunk, iter_chunks
from arcsum.guards import apply_ops
from arcsum.memory import Memory
from arcsum.ops import parse_ops
from arcsum.prompts import _STEP_SYS, PROMPT_VERSION, build_step_prompt, step_system_prompt
from arcsum.render import render_memory
from arcsum.supervision.align import Item, chunk_offset_spans, overlapping_items
from arcsum.tokens import TOKENIZE_VERSION, heuristic_token_len, token_len_name
from arcsum.transcript import Utterance

#: Bump on any change to the grounding text below -- this governs teacher-generation
#: reproducibility, separately from `prompts.PROMPT_VERSION`, which governs the
#: DEPLOYED model's own prompt shape and must never depend on how supervision for it
#: happened to be built.
TEACHER_PROMPT_VERSION = "teacher-v1"

_TEACHER_SYS = (
    _STEP_SYS
    + """

你現在的任務是為訓練資料生成標準答案，因此除了 CHUNK 之外，你還會看到這段內容對應的正式議程摘要
（來自會議記錄，內容真實可靠）。請將這份摘要的重點，依照上述格式轉換成 ADD／ARC 指令，
而不是自行從 CHUNK 內容重新歸納重點。若這份摘要的內容推翻了先前記憶中的某項重點，請先 DROP
該重點，再 ADD 新的重點。"""
)

#: Appended to a retry attempt's user turn, never the first attempt's -- the model is
#: called at temperature 0, so a bare repeat of the identical prompt would only ever
#: reproduce the identical (already-failed) output.
_RETRY_NUDGE = (
    "\n\n（提醒：上一次輸出的格式或語言有誤，請重新輸出。務必只使用繁體中文，"
    "不要夾雜英文字詞，也不要輸出不完整的句子。）"
)


class MissingExtraError(ImportError):
    """Raised in place of a bare `ModuleNotFoundError`, naming the extra to install.
    Subclasses `ImportError` so an `except ImportError` caller still catches it."""


@lru_cache(maxsize=1)
def _converter():
    try:
        import opencc
    except ImportError as exc:
        raise MissingExtraError(
            "supervision.teacher.to_traditional needs the 'supervision' extra "
            "(pip install 'arcsum-agentic[supervision]')"
        ) from exc
    return opencc.OpenCC("s2twp")


def to_traditional(text: str) -> str:
    """Deterministically convert any simplified characters in `text` to Taiwan-
    standard traditional (opencc `s2twp`: MOE standard, phrase-aware).

    Measured necessary, not assumed: the real Qwen3.8-27B teacher drifts into
    simplified characters under retry pressure even when its prompt explicitly says
    "全部使用繁體中文書寫" (Phase 1 pilot, 2026-08-22). Applied unconditionally to every
    teacher completion (not just retries) because it is a safe no-op on text that is
    already correct traditional Chinese -- character-for-character conversion of a
    character that has no simplified-only form leaves it untouched.

    `s2twp` over the plainer `s2t` deliberately: `s2t` maps some already-correct
    modern Taiwan usage to obscure classical variants a native reader would flag as
    wrong (e.g. 核准 -> 覈准), which `s2twp`'s Taiwan-vocabulary awareness avoids.

    The `OpenCC` instance is cached (`lru_cache`) rather than rebuilt per call -- this
    runs once per teacher completion, across thousands of calls in a real corpus run.
    """
    return _converter().convert(text)


_UNCOVERED_SUFFIX = """

這段 CHUNK 沒有對應的正式議程摘要。以下提供鄰近議程項目作為參考：如果這段內容與鄰近議程項目
的業務屬於同一件事的延續，請依該項目摘要記錄相應的 ADD／ARC 指令；如果這段內容是純粹的程序性
發言（例如點名、宣布休會、程序性轉場等），請回覆 NOP。"""


def teacher_step_system_prompt(*, covered: bool) -> str:
    return _TEACHER_SYS if covered else _TEACHER_SYS + _UNCOVERED_SUFFIX


def _render_items(items: Sequence[Item]) -> str:
    return "\n".join(f"- [{it.type}] {it.summary}" for it in items)


def build_teacher_step_prompt(
    memory: Memory,
    chunk: Chunk,
    *,
    grounding_items: Sequence[Item] = (),
    concluded_items: Sequence[Item] = (),
    next_item: Item | None = None,
) -> str:
    """The teacher's user turn: MEMORY, then CHUNK (same fixed order as the deployed
    `prompts.build_step_prompt`), then the grounding sections a live inference call
    would never have."""
    parts = [f"MEMORY:\n{render_memory(memory)}\nCHUNK:\n{chunk.render()}\n"]

    if grounding_items:
        parts.append(f"對應議程摘要：\n{_render_items(grounding_items)}\n")
    else:
        neighbours = []
        if concluded_items:
            neighbours.append(
                f"前一項已結束議程：[{concluded_items[-1].type}] {concluded_items[-1].summary}"
            )
        if next_item is not None:
            neighbours.append(f"下一項議程：[{next_item.type}] {next_item.summary}")
        if neighbours:
            parts.append("鄰近議程項目：\n" + "\n".join(neighbours) + "\n")

    if concluded_items:
        parts.append(f"已結束議程項目（供更新 ARC 摘要參考）：\n{_render_items(concluded_items)}\n")

    if next_item is not None and grounding_items:
        parts.append(
            f"下一項議程（供研判是否有重點將被推翻）：\n- [{next_item.type}] {next_item.summary}\n"
        )

    return "\n".join(parts)


def generate_step(
    memory: Memory,
    chunk: Chunk,
    teacher: ModelFn,
    *,
    grounding_items: Sequence[Item] = (),
    concluded_items: Sequence[Item] = (),
    next_item: Item | None = None,
    consecutive_nops: int = 0,
    budget: int = CHUNK_TOKENS,
    token_len: Callable[[str], int] = heuristic_token_len,
    max_attempts: int = 2,
) -> tuple[Step, Memory]:
    """One step's worth of gold supervision. Returns `(step, memory_after)` --
    `memory` itself is never mutated; the caller commits `memory_after` explicitly.

    **The teacher is called with the grounded prompt; the `Step` records the PLAIN
    deployed-shape prompt.** Grounding exists only to help the teacher produce a good
    completion -- it must never leak into the stored (prompt, completion) pair, or a
    student trained on it would learn to expect grounding text inference never
    provides. `system`/`user`/`prompt_tokens` below are exactly what
    `prompts.step_system_prompt`/`build_step_prompt` would build for a live inference
    call; only the argument to `teacher(...)` differs.

    **Retries on a failed replay (SPEC §4.2: "a sequence that fails replay is
    regenerated or dropped -- never half-applied into the corpus").** Each attempt
    applies against a FRESH `memory.clone()`, never the caller's real object, so a
    partially-bad first attempt can never leave stray mutations behind for a retry to
    compound on. If every attempt still fails to replay cleanly, the returned
    `memory_after` is `memory` UNCHANGED (a "dropped" step's ops touch nothing) and
    `step.retried` is `True` so the caller can exclude it from the training pool.

    **The retry prompt is nudged, not just repeated.** `teacher` is called with
    `temperature=0` in practice (SPEC §5.1's forced-greedy doctrine, mirrored here for
    reproducibility) -- an identical retry prompt against a deterministic model
    reproduces the identical (still-bad) output. `_RETRY_NUDGE` is appended on every
    attempt after the first so the input genuinely differs, the same fix
    `agent._NOP_RETRY_NUDGE` already applies to the live inference loop's own retry.
    """
    teacher_sys = teacher_step_system_prompt(covered=bool(grounding_items))
    teacher_user = build_teacher_step_prompt(
        memory,
        chunk,
        grounding_items=grounding_items,
        concluded_items=concluded_items,
        next_item=next_item,
    )
    student_sys = step_system_prompt()
    student_user = build_step_prompt(memory, chunk)
    memory_before = render_memory(memory)
    prompt_tokens = token_len(student_sys) + token_len(student_user)

    step: Step | None = None
    for attempt in range(max_attempts):
        candidate = memory.clone()
        attempt_user = teacher_user + _RETRY_NUDGE if attempt > 0 else teacher_user
        started = time.monotonic()
        raw = teacher(teacher_sys, attempt_user)
        elapsed = time.monotonic() - started

        ops = parse_ops(raw)
        outcome = apply_ops(candidate, ops, chunk, consecutive_nops=consecutive_nops, budget=budget)

        step = Step(
            index=chunk.index,
            system=student_sys,
            user=student_user,
            raw=raw,
            ops=tuple(ops),
            outcome=outcome,
            prompt_tokens=prompt_tokens,
            memory_before=memory_before,
            chunk=chunk,
            seconds=elapsed,
            retried=attempt > 0,
        )
        if replay_step_cleanly(step):
            return step, candidate

    assert step is not None  # max_attempts >= 1, loop runs at least once
    return step, memory


def generate_meeting_supervision(
    utterances: list[Utterance],
    offsets: list[tuple[float, float]],
    items: list[Item],
    teacher: ModelFn,
    *,
    budget: int = CHUNK_TOKENS,
    token_len: Callable[[str], int] = heuristic_token_len,
    memory: Memory | None = None,
) -> Trace:
    """Walk one meeting's chunks in order, generating grounded per-step supervision
    for each (SPEC §4.2 steps 1-3). Returns a real `agent.Trace` -- the same shape
    `agent.run_agent` produces from live inference -- with `trace.synthesis` left
    unset: SPEC §4.2 step 4's target is the already human-validated composed summary,
    not something to (re)generate here.
    """
    mem = memory if memory is not None else Memory()
    mem.token_len = token_len

    trace = Trace(
        memory=mem,
        prompt_version=PROMPT_VERSION,
        tokenize_version=TOKENIZE_VERSION,
        token_len_name=token_len_name(token_len),
        budget=budget,
    )

    spans = chunk_offset_spans(utterances, offsets, budget=budget, token_len=token_len)
    consecutive_nops = 0

    for chunk, span in zip(
        iter_chunks(utterances, budget=budget, token_len=token_len), spans, strict=True
    ):
        overlapping = overlapping_items(span, items)
        concluded = [it for it in items if it.end_sec <= span[1]]
        upcoming = [it for it in items if it.start_sec >= span[1]]
        next_item = upcoming[0] if upcoming else None

        step, trace.memory = generate_step(
            trace.memory,
            chunk,
            teacher,
            grounding_items=overlapping,
            concluded_items=concluded,
            next_item=next_item,
            consecutive_nops=consecutive_nops,
            budget=budget,
            token_len=token_len,
        )
        trace.steps.append(step)
        trace.usage.record(
            token_len(step.system) + token_len(step.user), token_len(step.raw), step.seconds
        )

        substantive = step.outcome.applied > 0
        consecutive_nops = 0 if substantive else consecutive_nops + 1
        if step.outcome.nop_collapse:
            trace.coverage_gaps.append(chunk.index)
            consecutive_nops = 0

    return trace


def replay_step_cleanly(step: Step) -> bool:
    """SPEC §4.2's validation rule: "ops must parse, DROP prefixes must match an
    existing point, and the resulting memory must respect the caps. A sequence that
    fails replay is regenerated or dropped." True iff every non-NOP op in this step
    was actually applied -- reuses `Outcome.valid_op_rate` (already excludes NOP from
    both sides) rather than re-deriving the same NOP-exclusion logic a second time."""
    rate = step.outcome.valid_op_rate
    return rate is None or rate == 1.0
