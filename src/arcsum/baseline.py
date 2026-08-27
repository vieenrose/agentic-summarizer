"""The fair map-reduce baseline (SPEC §5.2) — the opponent the agent architecture must
beat to earn its complexity.

**Fair, not weak.** Same model, same `iter_chunks`/`token_len` instrument, same §3 prose
contract via `prose.finalize` as the agent. The only architectural difference is what
SPEC §5.2 names as the point of comparison: **no state carried across steps.** Each map
call sees only its own chunk — `build_map_prompt` takes a `Chunk` and nothing else, no
memory concept exists in its vocabulary at all — which is the property that makes this a
structurally different opponent rather than a disguised copy of the agent.

**One reduce call, not one per section** (SPEC §5.2: "concatenate the chunk summaries,
one final compress pass") — never the prior project's per-over-cap-section reduce.
Skipped entirely when there is at most one window: compressing a single summary into
itself buys nothing and would cost the baseline a call the agent's `SYNTHESIZE` doesn't
have an equivalent of at that scale. A second reduce attempt is allowed on a contract
failure (over budget or bad language), mirroring `agent.synthesize_memory`'s own
default retry allowance — so `reduce_calls` is 0, 1, or 2, kept symmetric with the
agent arm's own worst case rather than pinned to a fixed count.

**Deterministic fallback, never a lossier retry.** If the reduce output still fails the
§3 contract after one retry, concatenate the (already individually cleaned) window
summaries rather than attempting to compress further — a failed shrink must not delete
the meeting's decisions.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from arcsum.agent import ModelFn, Usage
from arcsum.chunker import CHUNK_TOKENS, Chunk, iter_chunks
from arcsum.prompts import (
    PROMPT_VERSION,
    build_map_prompt,
    build_reduce_prompt,
    map_system_prompt,
    reduce_system_prompt,
)
from arcsum.prose import Prose, finalize
from arcsum.tokens import TOKENIZE_VERSION, heuristic_token_len, token_len_name
from arcsum.transcript import Utterance


@dataclass(frozen=True, slots=True)
class BaselineResult:
    prose: Prose
    window_summaries: tuple[str, ...]
    usage: Usage
    windows: int
    reduce_calls: int
    prompt_version: str = PROMPT_VERSION
    tokenize_version: str = TOKENIZE_VERSION
    token_len_name: str = "heuristic"
    #: True when the reduce call was skipped because its own rendered prompt would
    #: have exceeded `reduce_context_tokens` before any call was attempted -- distinct
    #: from `reduce_calls == 0` at a single window, which means there was nothing to
    #: reduce at all. SPEC §7's kill-criterion measurement needs to tell these apart:
    #: this one means the baseline itself cannot run at the declared deploy context on
    #: this meeting, at any generation quality.
    reduce_skipped_overflow: bool = False
    #: How many hierarchical folding passes ran before the final reduce. 0 means the
    #: window summaries fit one reduce call directly (the short-meeting case); >0 means
    #: they were folded in batches first. Recorded because a silently-degraded control
    #: arm is exactly what invalidated the first G3 run.
    reduce_passes: int = 0
    #: Output-side compress passes run because the reduced prose still broke SPEC §3's
    #: token cap. >0 means the cap was only met after re-compressing the summary itself.
    compress_passes: int = 0


def summarise_window(
    chunk: Chunk,
    model: ModelFn,
    *,
    token_len: Callable[[str], int] = heuristic_token_len,
    usage: Usage | None = None,
) -> str:
    """One independent map call — no state carried in, no state carried out. Returns
    the CLEANED text (via `prose.finalize`), so a bulleted window summary does not leak
    bullets into the reduce step's input.
    """
    sys = map_system_prompt()
    user = build_map_prompt(chunk)
    started = time.monotonic()
    try:
        raw = model(sys, user)
    except Exception:
        # Deterministic fallback, same principle the reduce step already follows: a
        # window that cannot be summarised must not delete the window's content, and
        # must not take the whole meeting down with it.
        #
        # Measured 2026-08-27: llama.cpp returns a 500 when the model emits an invalid
        # UTF-8 byte, refusing the whole response over one bad character. It is
        # deterministic at temperature 0 with `cache_prompt: false`, so no retry escapes
        # it, and it strikes the map call far more often than the agent's reading steps
        # because map generates long prose while a reading step emits short op lines.
        # Two of twenty meetings died this way — and since SPEC §5.2's comparison is
        # PAIRED, each loss cost the agent arm a meeting too, dropping the pool to n=18
        # and withholding every G3 gate for `n < min_n`. A server defect was deciding
        # whether a gate got a verdict.
        #
        # Falling back to the window's own text keeps the meeting scoreable. Note the
        # direction: raw transcript text is MORE extractive than a real summary, so this
        # favours the baseline on ROUGE/coverage/density. That is deliberate — a
        # workaround for a defect on the control arm must not be one that flatters the
        # treatment. Streaming was rejected for the opposite reason: it returns content
        # truncated at the bad byte, silently shortening only the baseline's output.
        if usage is not None:
            usage.record(token_len(sys) + token_len(user), 0, time.monotonic() - started)
        return finalize(chunk.render(), token_len=token_len).text
    elapsed = time.monotonic() - started
    if usage is not None:
        usage.record(token_len(sys) + token_len(user), token_len(raw), elapsed)
    return finalize(raw, token_len=token_len).text


def _reduce_once(
    summaries: tuple[str, ...], model: ModelFn, *, token_len: Callable[[str], int], usage: Usage
) -> Prose:
    sys = reduce_system_prompt()
    user = build_reduce_prompt(summaries)
    started = time.monotonic()
    raw = model(sys, user)
    elapsed = time.monotonic() - started
    usage.record(token_len(sys) + token_len(user), token_len(raw), elapsed)
    return finalize(raw, token_len=token_len)


#: Safety stop for the hierarchical reduce. Each pass strictly shrinks the level (a
#: pass that fails to is detected and abandoned), so this is a belt-and-braces bound
#: against pathological input, not an expected limit.
MAX_REDUCE_PASSES = 4

#: Bound on the output-side compress loop (see `run_map_reduce`). Small: each pass is a
#: whole model call, and a model that will not shorten in two tries will not in five.
MAX_COMPRESS_PASSES = 2


def _partition_to_fit(
    summaries: tuple[str, ...], *, token_len: Callable[[str], int], context: int
) -> list[tuple[str, ...]]:
    """Greedily group `summaries` so each group's OWN rendered reduce prompt fits
    `context`. Measured with the real builders, not estimated, for the same reason
    `chunker` packs against the injected counter: a guessed size drifts from the one
    the server actually enforces.

    A single summary that alone overflows `context` is emitted as its own group; the
    caller detects the resulting lack of progress and falls back rather than looping.
    """
    sys_tokens = token_len(reduce_system_prompt())
    groups: list[tuple[str, ...]] = []
    current: list[str] = []
    for summary in summaries:
        trial = [*current, summary]
        if current and sys_tokens + token_len(build_reduce_prompt(tuple(trial))) > context:
            groups.append(tuple(current))
            current = [summary]
        else:
            current = trial
    if current:
        groups.append(tuple(current))
    return groups


def run_map_reduce(
    utterances: list[Utterance],
    model: ModelFn,
    *,
    reduce_model: ModelFn | None = None,
    budget: int = CHUNK_TOKENS,
    token_len: Callable[[str], int] = heuristic_token_len,
    reduce_context_tokens: int | None = None,
) -> BaselineResult:
    """Map each chunk independently, then one reduce call over all window summaries.

    **`reduce_context_tokens` guards against the reduce call's own unbounded size.**
    `build_reduce_prompt` concatenates EVERY window summary with no cap -- fine for a
    short meeting, but measured to overflow a real 4096-token deploy context on 7 of
    20 real meetings sampled (up to 43 windows). Passing the deployed model's actual
    context ceiling here makes that failure structural and checked BEFORE a call is
    attempted, rather than an `HTTPError` from the server after burning a request:
    when the rendered reduce prompt would exceed it, the reduce call is skipped
    entirely and the existing deterministic concatenation fallback is used instead --
    the same fallback already used when two real reduce attempts both fail the
    contract, on the same "never delete decisions" principle. `None` (the default)
    preserves the old unbounded behaviour for callers that already run at a large
    enough context for it not to matter.
    """
    usage = Usage()
    window_summaries = tuple(
        summarise_window(chunk, model, token_len=token_len, usage=usage)
        for chunk in iter_chunks(utterances, budget=budget, token_len=token_len)
    )

    reduce_skipped_overflow = False
    reduce_calls = 0
    reduce_passes = 0
    compress_passes = 0

    if len(window_summaries) <= 1:
        prose = finalize(window_summaries[0] if window_summaries else "", token_len=token_len)
    else:
        reducer = reduce_model or model
        level = window_summaries

        def prompt_tokens(summaries: tuple[str, ...]) -> int:
            return token_len(reduce_system_prompt()) + token_len(build_reduce_prompt(summaries))

        # Hierarchical reduce: fold the window summaries in context-sized batches until
        # they fit ONE final reduce call. Without this, the overflow fallback became the
        # COMMON path rather than a rare safety net -- measured 2026-08-27, 11 of 20
        # held-out meetings, mean 3,695 tokens and max 12,540 against SPEC §3's <1,000
        # cap, because the "baseline" was emitting concatenated map summaries and never
        # reducing at all. That silently turned §5.2's deliberately FAIR control arm into
        # a strawman and made G3 meaningless in BOTH directions: the agent "won" ROUGE
        # against invalid over-length output, and "lost" coverage/density purely because
        # raw concatenation is more extractive than any real summary.
        if reduce_context_tokens is not None:
            while (
                prompt_tokens(level) > reduce_context_tokens
                and len(level) > 1
                and reduce_passes < MAX_REDUCE_PASSES
            ):
                groups = _partition_to_fit(
                    level, token_len=token_len, context=reduce_context_tokens
                )
                if len(groups) >= len(level):
                    break  # a lone summary overflows: folding cannot shrink this further
                level = tuple(
                    g[0]
                    if len(g) == 1
                    else _reduce_once(g, reducer, token_len=token_len, usage=usage).text
                    for g in groups
                )
                reduce_calls += sum(1 for g in groups if len(g) > 1)
                reduce_passes += 1

        if len(level) <= 1:
            # Folded all the way down; there is no pair left to reduce.
            prose = finalize(level[0] if level else "", token_len=token_len)
        elif reduce_context_tokens is not None and prompt_tokens(level) > reduce_context_tokens:
            prose = finalize(" ".join(level), token_len=token_len)
            reduce_skipped_overflow = True
        else:
            prose = _reduce_once(level, reducer, token_len=token_len, usage=usage)
            reduce_calls += 1
            if prose.over_budget or prose.lang_flags:
                retried = _reduce_once(level, reducer, token_len=token_len, usage=usage)
                reduce_calls += 1
                if not retried.over_budget and not retried.lang_flags:
                    prose = retried
                else:
                    # Deterministic fallback: never delete decisions by attempting a
                    # third, equally-unreliable compress. Concatenate what the previous
                    # level already produced, individually cleaned, and validate the
                    # SHAPE (not the length) of that concatenation.
                    prose = finalize(" ".join(level), token_len=token_len)

            # Folding bounds the reduce INPUT; nothing yet bounds its OUTPUT, and the
            # model can still emit an over-length summary from an in-context prompt
            # (measured 2026-08-27: 5 of 20 held-out meetings still broke SPEC §3's
            # <1,000-token cap, up to 2,181, even after hierarchical folding fixed the
            # far worse concatenation bug). Compressing the PROSE ITSELF is always
            # in-context by construction -- it is one already-short document -- so
            # unlike another reduce over `level` it cannot re-overflow.
            while prose.over_budget and compress_passes < MAX_COMPRESS_PASSES:
                compressed = _reduce_once((prose.text,), reducer, token_len=token_len, usage=usage)
                compress_passes += 1
                if compressed.lang_flags or not compressed.text:
                    break  # a broken compress is worse than an over-long summary
                if token_len(compressed.text) >= token_len(prose.text):
                    prose = compressed if not compressed.over_budget else prose
                    break  # not converging -- stop rather than loop on a stubborn model
                prose = compressed

    return BaselineResult(
        prose=prose,
        window_summaries=window_summaries,
        usage=usage,
        windows=len(window_summaries),
        reduce_calls=reduce_calls,
        prompt_version=PROMPT_VERSION,
        tokenize_version=TOKENIZE_VERSION,
        token_len_name=token_len_name(token_len),
        reduce_skipped_overflow=reduce_skipped_overflow,
        reduce_passes=reduce_passes,
        compress_passes=compress_passes,
    )
