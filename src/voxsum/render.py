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


def promote_decision_summaries(state: NotesState, lang: str = "en") -> int:
    """Deterministic harness-side mitigation for the measured zero-DECISIONS class
    (the student puts decision-shaped content in SUMMARY and never emits DECISIONS
    ops — VoxSum round-5.2, their op-level audit). Any SUMMARY bullet whose text
    matches the commitment lexicon is promoted into DECISIONS with its anchor,
    guarded by the dedup check and the section cap. Returns the promoted count.
    """
    from .highlight import is_commit_line

    promoted = 0
    for b in list(state.bullets("SUMMARY")):
        if is_commit_line(b.text, lang):
            reason = state.add("DECISIONS", b.text, b.anchor)
            if reason is None:
                promoted += 1
    return promoted


def render_state(state: NotesState, *, enforce_caps: bool = True, lang: str = "en",
                 promote_decisions: bool = False) -> str:
    """Render NOTES v2. Caps are applied by default — this is the shipping output."""
    if enforce_caps:
        state = state.clone()
        state.enforce_caps()
    if promote_decisions:
        promote_decision_summaries(state, lang)

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
