"""Shared test fakes — the substitution seams that let the whole suite run with no GPU,
no weights, and no network.

`ModelFn = Callable[[str, str], str]` is the entire model abstraction (`arcsum.agent`),
so a plain scripted callable stands in for the teacher, the student, and any local test
double with zero mocking machinery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from arcsum.memory import ARC_TOKENS, POINT_TOKENS, POINTS_CAP, Memory


@dataclass
class Scripted:
    """Replays canned responses in order (falling back to `default` once exhausted) and
    RECORDS every `(system, user)` pair it was called with — which is what makes
    "no conversation history crosses steps" and "memory is visible to the next step"
    provable at all, with no weights anywhere.
    """

    responses: tuple[str, ...] = ()
    default: str = "NOP"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        idx = len(self.calls) - 1
        return self.responses[idx] if idx < len(self.responses) else self.default


def saturated_memory(*, token_len: Callable[[str], int] | None = None) -> Memory:
    """A memory filled to both caps: ARC at its token limit, POINTS at its count and
    per-entry token limits. Used to measure the WORST-CASE prefill overhead a step can
    carry, rather than the best case an empty memory would understate.
    """
    memory = Memory(token_len=token_len) if token_len is not None else Memory()
    memory.set_arc("很" * (ARC_TOKENS - 2))  # margin: exact-cap heuristic rounding
    for i in range(POINTS_CAP):
        # Each point must be unique (add_point refuses duplicates) and near, but under,
        # the per-point token cap regardless of which token_len is injected.
        memory.add_point(f"第{i:03d}項" + "很" * (POINT_TOKENS - 5), chunk=0)
    return memory
