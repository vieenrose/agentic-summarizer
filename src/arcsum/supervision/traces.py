"""Gold-edit replay validation (SPEC §4.2's "Validation").

SPEC §4.2: "Every gold edit sequence is replayed through the real harness before use:
ops must parse, `DROP` prefixes must match an existing point, and the resulting
memory must respect the caps. A sequence that fails replay is regenerated or
dropped — never half-applied into the corpus."

**This module does not drive a live teacher.** Generating candidate targets means
calling an offline teacher model over a real transcript — a network/weights-dependent
corpus-construction script (a future `cli/gen_traces.py`), not a unit-testable library
function. What belongs here, and is the actual enforcement mechanism SPEC §4.2
demands, is replay: given a candidate raw target string (the teacher's edit-line
output for one step) and the memory state it was generated against, does it apply
cleanly through the REAL `arcsum.ops`/`arcsum.guards` pipeline — the identical code
path inference will use, not a re-implementation of it?
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from arcsum.chunker import CHUNK_TOKENS, Chunk
from arcsum.guards import apply_ops
from arcsum.memory import Memory
from arcsum.ops import parse_ops
from arcsum.tokens import heuristic_token_len


@dataclass(frozen=True, slots=True)
class ReplayResult:
    ok: bool
    #: One entry per op that failed to apply — empty when `ok` is True. Each entry is
    #: the applier's own refusal reason, so a failure is immediately diagnosable
    #: without re-deriving what went wrong.
    failures: tuple[str, ...] = field(default_factory=tuple)


def replay_step(
    memory: Memory, raw_target: str, chunk: Chunk, *, budget: int = CHUNK_TOKENS
) -> ReplayResult:
    """Apply one step's candidate gold target to `memory` IN PLACE, through the exact
    code path `agent.run_agent` uses at inference time. A malformed op, a refused op
    (duplicate, over-length, a language-guard failure, a contradiction), or a `DROP`
    that matched nothing all count as a replay failure — never half-applied.
    """
    ops = parse_ops(raw_target)
    outcome = apply_ops(memory, ops, chunk, budget=budget)
    failures = tuple(r.log_line() for r in outcome.results if not r.applied)
    return ReplayResult(ok=not failures, failures=failures)


def replay_sequence(
    targets: Sequence[tuple[str, Chunk]],
    *,
    token_len: Callable[[str], int] = heuristic_token_len,
    budget: int = CHUNK_TOKENS,
) -> tuple[Memory, list[ReplayResult]]:
    """Replay a whole meeting's candidate gold targets, in step order, against one
    shared memory — exactly the carry-forward invariant SPEC §4.1 defines. Returns
    the final memory and one `ReplayResult` per step, so a caller can identify exactly
    which step failed rather than only knowing the meeting as a whole did.

    Does NOT track `consecutive_nops`/NOP-collapse across steps: that is a distinct
    diagnostic (`supervision.report`, over real `agent.Trace` objects), not a replay
    validity question. Every step here starts the guard's counter fresh — correct for
    "does this op sequence apply cleanly", not intended to reproduce the collapse
    signal.
    """
    memory = Memory(token_len=token_len)
    results = [
        replay_step(memory, raw_target, chunk, budget=budget) for raw_target, chunk in targets
    ]
    return memory, results


def all_replayed_cleanly(results: Sequence[ReplayResult]) -> bool:
    """SPEC §4.2: a sequence that fails replay at ANY step is regenerated or dropped
    in its entirety — never half-applied. This is the single boolean gate."""
    return all(r.ok for r in results)
