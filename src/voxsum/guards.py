"""Guards — the harness owns the final word (CLAUDE.md §6).

Every op the model emits passes through `apply_ops`. Nothing is trusted:

1. **Anchor validation** — an ADD/UPD anchor must resolve to a line *in the current
   chunk*; otherwise the bullet falls to the deterministic matcher (logged).
2. **Temporal guard** — ops touching DECISIONS/ACTIONS are cross-checked against the
   time-sorted decision/action timeline; a contradiction is dropped. This is the
   0%-inversions backstop, and it is why an inversion cannot be a model failure alone.
3. **NOP-collapse** — K consecutive NOPs over content-rich chunks trips the coverage
   fallback (the caller acts on `Outcome.nop_collapse`).
4. **Malformed ops** — logged, never fatal.

Ops are applied in emission order so a step's own `ADD` is visible to its later `UPD`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunker import Chunk
from .ops import Add, Cmp, Del, Malformed, Nop, Op, Title, Upd, render_op
from .state import NotesState
from .transcript import sec_to_clock

__all__ = ["AppliedOp", "Outcome", "NOP_COLLAPSE_K", "apply_ops", "match_anchor"]

# K consecutive NOPs over content-rich chunks -> coverage fallback (CLAUDE.md §6.3).
NOP_COLLAPSE_K = 3

# Sections whose ops are timeline-checked. These are the ones an inversion would corrupt.
TIMELINE_SECTIONS = ("DECISIONS", "ACTIONS")

# Polarity markers. Deliberately small and explicit: a wrong "contradiction" verdict
# silently drops a true decision, so this errs toward not firing.
_NEGATIVE = (
    "reject", "rejected", "denied", "declined", "not approved", "cancel", "cancelled",
    "postpone", "postponed", "deferred", "on hold", "blocked", "vetoed", "withdrawn",
    "否決", "駁回", "拒絕", "取消", "延後", "暫緩", "擱置", "不通過", "未通過",
)
_POSITIVE = (
    "approve", "approved", "agreed", "accepted", "confirmed", "signed off", "go ahead",
    "greenlit", "adopted", "ratified", "passed",
    "通過", "核准", "批准", "同意", "確認", "採納", "定案",
)


@dataclass(frozen=True, slots=True)
class AppliedOp:
    op: Op
    applied: bool
    reason: str | None = None

    def log_line(self) -> str:
        verdict = "ok" if self.applied else f"dropped: {self.reason}"
        return f"[{verdict}] {render_op(self.op)}"


@dataclass
class Outcome:
    """Result of one step. `results` preserves emission order for the op log."""

    results: list[AppliedOp] = field(default_factory=list)
    nop_collapse: bool = False

    @property
    def applied(self) -> int:
        return sum(1 for r in self.results if r.applied)

    @property
    def valid_op_rate(self) -> float | None:
        """Fraction of op lines that parsed AND validated (GT1). None if no scored ops.

        NOP is excluded from both numerator and denominator: it is always a valid answer
        (CLAUDE.md §5.0), so counting it would inflate the rate on quiet chunks. Whether
        NOP was *appropriate* is the separate NOP-collapse metric.
        """
        scored = [r for r in self.results if not isinstance(r.op, Nop)]
        if not scored:
            return None
        return sum(1 for r in scored if r.applied) / len(scored)

    @property
    def malformed(self) -> list[AppliedOp]:
        return [r for r in self.results if isinstance(r.op, Malformed)]


def match_anchor(chunk: Chunk, text: str) -> int | None:
    """Deterministic anchor fallback: the chunk line with the best lexical overlap.

    Used when the model's anchor does not resolve. Returns a real line's start, or None
    for an empty chunk — an unanchored bullet is later resolved by the ANCHOR sweep.
    """
    if not chunk.utterances:
        return None
    target = _tokens(text)
    if not target:
        return chunk.utterances[0].start
    best, best_score = chunk.utterances[0].start, -1.0
    for u in chunk.utterances:
        cand = _tokens(u.text)
        if not cand:
            continue
        score = len(target & cand) / len(target | cand)
        if score > best_score:
            best, best_score = u.start, score
    return best


def _tokens(text: str) -> set[str]:
    """Word tokens for en; character bigrams for CJK (CLAUDE.md §7.2)."""
    words = {w for w in "".join(c if c.isalnum() else " " for c in text.casefold()).split() if w}
    cjk = [c for c in text if "一" <= c <= "鿿"]
    words |= {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
    return words


def _polarity(text: str) -> int:
    """+1 affirmative, -1 negative, 0 unknown. Negatives win — 'not approved' is negative."""
    low = text.casefold()
    neg = any(m in low for m in _NEGATIVE)
    pos = any(m in low for m in _POSITIVE)
    if neg:
        return -1
    return 1 if pos else 0


def _contradicts_timeline(
    state: NotesState, section: str, bullet: str, anchor: int | None
) -> str | None:
    """Reject an ADD that states the opposite of a *later* bullet about the same subject.

    The rule is directional: the meeting's later word wins. Revising an earlier bullet is
    what UPD is for and is always allowed; asserting a stale opposite as a *new* bullet is
    the inversion this exists to stop.
    """
    if section not in TIMELINE_SECTIONS or anchor is None:
        return None
    polarity = _polarity(bullet)
    if polarity == 0:
        return None
    subject = _tokens(bullet)
    if not subject:
        return None
    for existing in state.bullets(section):
        if existing.anchor is None or existing.anchor <= anchor:
            continue
        other = _polarity(existing.text)
        if other == 0 or other == polarity:
            continue
        other_tokens = _tokens(existing.text)
        overlap = len(subject & other_tokens) / max(len(subject | other_tokens), 1)
        if overlap >= 0.34:  # same subject, opposite polarity, and the other one is later
            return (
                f"contradicts later {section} bullet at "
                f"[{sec_to_clock(existing.anchor)}] (temporal guard)"
            )
    return None


def _resolve_anchor(
    chunk: Chunk, bullet: str, anchor: int | None
) -> tuple[int | None, str | None]:
    """Validate an anchor against the current chunk, falling back to the matcher."""
    if anchor is not None and chunk.has_line(anchor):
        return anchor, None
    note = (
        f"anchor [{sec_to_clock(anchor)}] not in chunk; used matcher"
        if anchor is not None
        else "no anchor emitted; used matcher"
    )
    return match_anchor(chunk, bullet), note


def apply_ops(
    state: NotesState,
    ops: list[Op],
    chunk: Chunk,
    *,
    consecutive_nops: int = 0,
) -> Outcome:
    """Validate and apply a step's ops in place. Returns the per-op verdicts.

    `consecutive_nops` is the count *before* this step; the caller keeps the running
    tally and passes it in so the collapse guard stays stateless here.
    """
    outcome = Outcome()
    substantive = False

    for op in ops:
        match op:
            case Nop():
                outcome.results.append(AppliedOp(op, True))

            case Malformed(_, reason):
                outcome.results.append(AppliedOp(op, False, reason))

            case Title(title):
                reason = state.set_title(title)
                outcome.results.append(AppliedOp(op, reason is None, reason))
                substantive = substantive or reason is None

            case Add(section, bullet, anchor):
                resolved, note = _resolve_anchor(chunk, bullet, anchor)
                if contradiction := _contradicts_timeline(state, section, bullet, resolved):
                    outcome.results.append(AppliedOp(op, False, contradiction))
                    continue
                reason = state.add(section, bullet, resolved)
                outcome.results.append(AppliedOp(op, reason is None, reason or note))
                substantive = substantive or reason is None

            case Upd(section, prefix, bullet, anchor):
                resolved, note = _resolve_anchor(chunk, bullet, anchor)
                reason = state.update(section, prefix, bullet, resolved)
                if reason == "prefix did not match exactly one bullet":
                    # Deterministic fallback: the model wants this bullet in the state
                    # but misjudged the op type (UPD against an empty/mismatched
                    # prefix). Honor the intent as an ADD; the timeline guard still
                    # vetoes contradictory DECISIONS/ACTIONS, and state.add still
                    # rejects duplicates. Logged in `reason` for full transparency.
                    if contradiction := _contradicts_timeline(state, section, bullet, resolved):
                        outcome.results.append(AppliedOp(op, False, contradiction))
                        continue
                    reason2 = state.add(section, bullet, resolved)
                    outcome.results.append(
                        AppliedOp(op, reason2 is None, reason2 or f"upd-as-add: {reason}")
                    )
                    substantive = substantive or reason2 is None
                    continue
                outcome.results.append(AppliedOp(op, reason is None, reason or note))
                substantive = substantive or reason is None

            case Del(section, prefix):
                reason = state.delete(section, prefix)
                outcome.results.append(AppliedOp(op, reason is None, reason))
                substantive = substantive or reason is None

            case Cmp(section, bullets):
                repaired = [
                    b.__class__(b.text, _resolve_anchor(chunk, b.text, b.anchor)[0])
                    for b in bullets
                ]
                reason = state.compact(section, repaired)
                outcome.results.append(AppliedOp(op, reason is None, reason))
                substantive = substantive or reason is None

            case _:
                outcome.results.append(AppliedOp(op, False, "unhandled op type"))

    state.enforce_caps()

    # NOP-collapse: only content-rich chunks count — a genuinely empty chunk deserves NOP.
    if not substantive and chunk.is_content_rich():
        outcome.nop_collapse = consecutive_nops + 1 >= NOP_COLLAPSE_K
    return outcome
