"""Deterministic memory -> prompt-block rendering (SPEC §4.1).

    ARC: <text|->
    POINTS:
    - <point>
    ...

Unconditional shape: every slot always renders something, so no byte-level difference in
the prompt is contingent on whether the memory happens to be empty. An empty ARC renders
`ARC: -`, mirroring POINTS' `-` for an empty list — one rule, pinned by a test, rather
than the prior project's incidental `f"TITLE: {title}".rstrip()`, which rendered
differently depending on whether the title was set.
"""

from __future__ import annotations

from arcsum.memory import Memory

EMPTY = "-"


def render_memory(memory: Memory, *, enforce_caps: bool = True) -> str:
    """Render the ARC+POINTS block. Caps applied by default — this is what the model sees.

    `enforce_caps=False` exists only for diagnostics (e.g. inspecting an over-cap memory
    before `enforce_caps()` runs); every production caller uses the default.
    """
    m = memory
    if enforce_caps:
        m = memory.clone()
        m.enforce_caps()

    lines = [f"ARC: {m.arc or EMPTY}"]
    lines.append("POINTS:")
    if m.points:
        # `[id]` rather than `-`: SPEC §4.1 v1.1 addresses points by id, and the model
        # can only use an id it can see. Text-prefix addressing is what produced the
        # DROP + near-identical re-ADD churn measured at 28.2% of steps.
        lines.extend(f"[{p.pid}] {p.text}" for p in m.points)
    else:
        lines.append(EMPTY)
    return "\n".join(lines) + "\n"
