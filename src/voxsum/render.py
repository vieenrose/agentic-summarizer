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


def enforce_decision_chain(state: NotesState) -> int:
    """Deterministic chain guard: across DECISIONS and SUMMARY, when two bullets
    share a subject and carry opposite polarities (rejected vs approved), keep the
    LATEST and drop the older one. The model's measured over-ADD failure leaves the
    stale rejection beside the approval (in DECISIONS and/or SUMMARY); the harness
    owns the final word (spec §6) and resolves the timeline. Returns the number
    dropped."""
    from voxsum.guards import _polarity

    dropped = 0
    # one decision timeline: SUMMARY bullets first (older), then DECISIONS
    entries = [(b, "SUMMARY") for b in state.bullets("SUMMARY")] + \
              [(b, "DECISIONS") for b in state.bullets("DECISIONS")]
    for i, (b, sec) in enumerate(entries):
        pi = _polarity(b.text)
        if pi not in (1, -1):
            continue
        for j in range(i):
            bj, secj = entries[j]
            if bj.anchor is not None and b.anchor is not None and bj.anchor >= b.anchor:
                continue
            if _polarity(bj.text) == -pi and _subject_overlap(b.text, bj.text):
                state.delete(secj, _prefix_of(bj.text))
                dropped += 1
                entries = [(x, s) for s in ("SUMMARY", "DECISIONS")
                           for x in state.bullets(s)]
                break
    return dropped


def _subject_overlap(a: str, b: str) -> bool:
    """Cheap subject-overlap test for the chain guard: significant token overlap
    beyond the polarity words themselves."""
    import re
    toks_a = {t for t in re.findall(r"[\w\u4e00-\u9fff]+", a.lower()) if len(t) > 1}
    toks_b = {t for t in re.findall(r"[\w\u4e00-\u9fff]+", b.lower()) if len(t) > 1}
    inter = toks_a & toks_b
    return len(inter) >= 2 or (inter and len(inter) >= max(1, min(len(toks_a), len(toks_b)) // 2))


def _prefix_of(text: str) -> str:
    return text[:6]


def render_state(state: NotesState, *, enforce_caps: bool = True, lang: str = "en",
                 promote_decisions: bool = False, enforce_chain: bool = False) -> str:
    """Render NOTES v2. Caps are applied by default — this is the shipping output."""
    if enforce_caps:
        state = state.clone()
        state.enforce_caps()
    if promote_decisions:
        promote_decision_summaries(state, lang)
    if enforce_chain:
        enforce_decision_chain(state)

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
