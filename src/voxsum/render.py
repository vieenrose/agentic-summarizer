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


def promote_decision_summaries(
    state: NotesState, lang: str = "en",
    evidence_lookup: callable | None = None,
) -> tuple[int, int]:
    """Deterministic promotion of decision-shaped SUMMARY bullets into DECISIONS.

    Two gates (VoxSum round-5.5): (1) the SPECIFIC token that triggered the lexicon
    match must appear in the evidence lines at the bullet's anchor — the claim the
    promotion makes, checked for free; (2) the caller may also run the model
    verifier against whole-transcript evidence. A refused bullet stays in SUMMARY
    (the model did say it; the guard decides what gets promoted, not what gets
    censored). Returns (promoted, refused).
    """
    from .highlight import is_commit_line, commit_token

    promoted = refused = 0
    for b in list(state.bullets("SUMMARY")):
        if not is_commit_line(b.text, lang):
            continue
        token = commit_token(b.text, lang)
        if evidence_lookup is not None and token:
            evidence = evidence_lookup(b.anchor) if b.anchor is not None else ""
            if token not in evidence:
                refused += 1
                continue
        reason = state.add("DECISIONS", b.text, b.anchor)
        if reason is not None:
            continue
        drop_reason = state.delete("SUMMARY", b.text[:24]) or state.delete("SUMMARY", b.text[:6])
        promoted += 1
    return promoted, refused


def enforce_decision_chain(state: NotesState) -> int:
    """Deterministic chain guard: across DECISIONS and SUMMARY, when two bullets
    share a subject and carry opposite polarities (rejected vs approved), keep the
    LATEST and drop the older one. The model's measured over-ADD failure leaves the
    stale rejection beside the approval (in DECISIONS and/or SUMMARY); the harness
    owns the final word (spec §6) and resolves the timeline. Returns the number
    dropped."""
    from voxsum.guards import _polarity

    dropped = 0
    # one decision timeline across SUMMARY and DECISIONS, ORDER-INDEPENDENT: for any
    # opposing-polarity pair on one subject, drop the OLDER (by anchor), wherever it
    # sits in the section ordering.
    changed = True
    while changed:
        changed = False
        entries = [(b, "SUMMARY") for b in state.bullets("SUMMARY")] + \
                  [(b, "DECISIONS") for b in state.bullets("DECISIONS")]
        for i, (b, sec) in enumerate(entries):
            pi = _polarity(b.text)
            if pi not in (1, -1):
                continue
            for j, (bj, secj) in enumerate(entries):
                if i == j:
                    continue
                if b.anchor is None or bj.anchor is None:
                    continue
                if _polarity(bj.text) != -pi or not _subject_overlap(b.text, bj.text):
                    continue
                # drop the OLDER bullet, whichever section it is in
                older_sec, older_text = (sec, b.text) if b.anchor <= bj.anchor else (secj, bj.text)
                reason = state.delete(older_sec, older_text[:24])
                if reason:
                    reason = state.delete(older_sec, older_text)
                dropped += 1
                changed = True
                break
            if changed:
                break
    return dropped


def _subject_overlap(a: str, b: str) -> bool:
    """Subject-overlap test for the chain guard. En: word tokens; zh: character
    bigrams (the whole zh string is one whitespace-free run, so word tokenization
    yields an empty intersection). Polarity words (the rejected/approved verbs) are
    excluded so the subject, not the decision, carries the overlap."""
    import re

    def toks(text: str) -> set:
        s = set()
        # en words
        s |= {w for w in re.findall(r"[a-zA-Z]{3,}", text.lower())}
        # zh bigrams (and the lone char as a fallback for 2-char subjects)
        cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
        s |= {cjk[i:i + 2] for i in range(len(cjk) - 1)}
        s |= {ch for ch in cjk}
        return s

    ta, tb = toks(a), toks(b)
    inter = ta & tb
    # the overlap must be SUBSTANTIAL relative to the smaller subject, and exceed
    # what the polarity verb alone would contribute
    return len(inter) >= 3 and len(inter) >= min(len(ta), len(tb)) // 3


def render_state(state: NotesState, *, enforce_caps: bool = True, lang: str = "en",
                 promote_decisions: bool = False, enforce_chain: bool = False,
                 enforce_lang: bool = False,
                 evidence_lookup: callable | None = None) -> str:
    """Render NOTES v2. Caps are applied by default — this is the shipping output."""
    if enforce_caps:
        state = state.clone()
        state.enforce_caps()
    if promote_decisions:
        promote_decision_summaries(state, lang, evidence_lookup=evidence_lookup)
    if enforce_chain:
        enforce_decision_chain(state)
    if enforce_lang:
        enforce_output_language(state, lang)

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


_HAN_RE = __import__("re").compile(r"[\u4e00-\u9fff]")
_LATIN_RE = __import__("re").compile(r"[A-Za-z]")


def enforce_output_language(state, lang: str) -> int:
    """Round-6 guard: a zh run must produce zh bullets.

    The loanword flip (VoxSum round 6): a zh transcript with heavy English
    technical vocabulary can push the model into English bullets — and when it
    leaves the source language it also leaves the source content (their Cerebras
    episode: invented content, anchors resolving to unrelated lines). A bullet
    whose Han fraction is under 50% in a zh run is dropped and logged; English
    loanwords inside a zh bullet (DDR4, wafer) survive the 50% threshold.
    Returns the number of bullets dropped.
    """
    if not lang.startswith("zh"):
        return 0
    dropped = 0
    for section in BULLET_SECTIONS:
        for b in list(state.bullets(section)):
            body = b.text.split("[")[0]
            han = len(_HAN_RE.findall(body))
            latin = len(_LATIN_RE.findall(body))
            if han + latin >= 4 and han / (han + latin) < 0.5:
                if not state.delete(section, b.text[:24]) and not state.delete(section, b.text):
                    continue
                dropped += 1
    return dropped
