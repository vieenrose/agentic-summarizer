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
    raw = model(sys, user)
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


def run_map_reduce(
    utterances: list[Utterance],
    model: ModelFn,
    *,
    reduce_model: ModelFn | None = None,
    budget: int = CHUNK_TOKENS,
    token_len: Callable[[str], int] = heuristic_token_len,
) -> BaselineResult:
    """Map each chunk independently, then one reduce call over all window summaries."""
    usage = Usage()
    window_summaries = tuple(
        summarise_window(chunk, model, token_len=token_len, usage=usage)
        for chunk in iter_chunks(utterances, budget=budget, token_len=token_len)
    )

    if len(window_summaries) <= 1:
        prose = finalize(window_summaries[0] if window_summaries else "", token_len=token_len)
        reduce_calls = 0
    else:
        reducer = reduce_model or model
        prose = _reduce_once(window_summaries, reducer, token_len=token_len, usage=usage)
        reduce_calls = 1
        if prose.over_budget or prose.lang_flags:
            retried = _reduce_once(window_summaries, reducer, token_len=token_len, usage=usage)
            reduce_calls = 2
            if not retried.over_budget and not retried.lang_flags:
                prose = retried
            else:
                # Deterministic fallback: never delete decisions by attempting a third,
                # equally-unreliable compress. Concatenate what the map step already
                # produced, individually cleaned, and validate the SHAPE (not the
                # length) of that concatenation.
                prose = finalize(" ".join(window_summaries), token_len=token_len)

    return BaselineResult(
        prose=prose,
        window_summaries=window_summaries,
        usage=usage,
        windows=len(window_summaries),
        reduce_calls=reduce_calls,
        prompt_version=PROMPT_VERSION,
        tokenize_version=TOKENIZE_VERSION,
        token_len_name=token_len_name(token_len),
    )
