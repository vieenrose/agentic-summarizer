"""Deterministic NOTES v2 renderer (CLAUDE.md §3, §6.5).

The final notes are rendered from STATE by the harness — the model never emits NOTES
directly. Sections are always all present and in fixed order; an empty section is exactly
`-` on one line.

`render_state` is the product output. `render_for_prompt` is the compact STATE view shown
to the model each step; both derive from the same state so they cannot drift.
"""

from __future__ import annotations

from .state import BULLET_SECTIONS, NotesState

__all__ = ["EMPTY_SECTION", "render_state", "render_for_prompt"]

EMPTY_SECTION = "-"


def render_state(state: NotesState, *, enforce_caps: bool = True) -> str:
    """Render NOTES v2. Caps are applied by default — this is the shipping output."""
    if enforce_caps:
        state = state.clone()
        state.enforce_caps()

    lines = [f"TITLE: {state.title}".rstrip()]
    for section in BULLET_SECTIONS:
        lines.append(f"{section}:")
        bullets = state.bullets(section)
        lines.extend(b.render() for b in bullets) if bullets else lines.append(EMPTY_SECTION)
    return "\n".join(lines) + "\n"


def render_for_prompt(state: NotesState) -> str:
    """The STATE block shown to the model (<= ~600 tok by construction of the caps).

    Identical format to the product output, so the model only ever learns one shape.
    """
    return render_state(state, enforce_caps=True)
