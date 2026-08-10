"""NOTES v2 state store — the agent's entire memory (CLAUDE.md §3, §4, §6.5).

One evolving NOTES state, curated by the model through edit ops. The harness owns the
final word: every mutation goes through here, caps are enforced by `spread()` (never
head-truncation), and a mutation that cannot be applied is refused with a reason rather
than half-applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .transcript import sec_to_clock

__all__ = [
    "CAPS",
    "SECTIONS",
    "Bullet",
    "MIN_PREFIX",
    "NotesState",
    "normalize",
    "spread",
]

# Fixed order, all always present (CLAUDE.md §3).
SECTIONS = ("TITLE", "SUMMARY", "DECISIONS", "ACTIONS", "OPEN", "TOPICS")
BULLET_SECTIONS = SECTIONS[1:]  # TITLE carries no bullets and no anchor

# Harness-enforced per-section caps (CLAUDE.md §3).
CAPS = {"SUMMARY": 5, "DECISIONS": 5, "ACTIONS": 6, "OPEN": 4, "TOPICS": 6}

# `«prefix»` = the first >= 6 characters of an existing STATE bullet (CLAUDE.md §5.0).
MIN_PREFIX = 6


@dataclass(frozen=True, slots=True)
class Bullet:
    """One anchored bullet. `anchor` is seconds; None only while awaiting the matcher."""

    text: str
    anchor: int | None = None

    def render(self) -> str:
        if self.anchor is None:
            return f"- {self.text}"
        return f"- {self.text} [{sec_to_clock(self.anchor)}]"


def normalize(text: str) -> str:
    """Comparison key for dedup and prefix matching — case- and space-insensitive."""
    return " ".join(text.split()).casefold()


def spread(items: list, cap: int) -> list:
    """Reduce to `cap` items by spreading evenly across the list, preserving order.

    Never head-truncation (CLAUDE.md §6.5): head-truncating a time-ordered section drops
    the end of the meeting, which is where decisions land. Endpoints are always kept.
    """
    n = len(items)
    if n <= cap:
        return list(items)
    if cap <= 0:
        return []
    if cap == 1:
        return [items[-1]]  # the latest state of the meeting beats the earliest
    picks = [round(i * (n - 1) / (cap - 1)) for i in range(cap)]
    # `round` can collide on short lists; walk forward to keep `cap` distinct indices.
    seen: list[int] = []
    for p in picks:
        while p in seen:
            p += 1
        seen.append(min(p, n - 1))
    return [items[i] for i in sorted(dict.fromkeys(seen))]


@dataclass
class NotesState:
    """The live NOTES. Mutations return a reason string on refusal, None on success."""

    title: str = ""
    sections: dict[str, list[Bullet]] = field(
        default_factory=lambda: {s: [] for s in BULLET_SECTIONS}
    )

    # --- lookups ---------------------------------------------------------------

    def bullets(self, section: str) -> list[Bullet]:
        if section not in self.sections:
            raise KeyError(f"unknown section: {section!r}")
        return self.sections[section]

    def find(self, section: str, prefix: str) -> int | None:
        """Index of the single bullet matching `prefix`, or None if absent/ambiguous.

        Ambiguity is a refusal, not a coin flip — silently editing the wrong bullet is
        how a correct decision becomes an inverted one.
        """
        key = normalize(prefix)
        if len(key) < MIN_PREFIX:
            return None
        hits = [i for i, b in enumerate(self.bullets(section)) if normalize(b.text).startswith(key)]
        return hits[0] if len(hits) == 1 else None

    def _duplicate(self, section: str, text: str, skip: int | None = None) -> bool:
        key = normalize(text)
        return any(
            normalize(b.text) == key for i, b in enumerate(self.bullets(section)) if i != skip
        )

    # --- mutations -------------------------------------------------------------

    def set_title(self, title: str) -> str | None:
        title = " ".join(title.split())
        if not title:
            return "empty title"
        self.title = title
        return None

    def add(self, section: str, text: str, anchor: int | None) -> str | None:
        text = " ".join(text.split())
        if not text:
            return "empty bullet"
        if self._duplicate(section, text):
            return "duplicate bullet"
        self.sections[section].append(Bullet(text, anchor))
        return None

    def update(self, section: str, prefix: str, text: str, anchor: int | None) -> str | None:
        idx = self.find(section, prefix)
        if idx is None:
            return "prefix did not match exactly one bullet"
        text = " ".join(text.split())
        if not text:
            return "empty bullet"
        if self._duplicate(section, text, skip=idx):
            return "duplicate bullet"
        old = self.sections[section][idx]
        # An UPD keeps its slot: revising a decision must not reorder the timeline.
        self.sections[section][idx] = replace(old, text=text, anchor=anchor)
        return None

    def delete(self, section: str, prefix: str) -> str | None:
        idx = self.find(section, prefix)
        if idx is None:
            return "prefix did not match exactly one bullet"
        del self.sections[section][idx]
        return None

    def compact(self, section: str, bullets: list[Bullet]) -> str | None:
        """Model-curated compaction: replace SECTION with <= cap rewritten bullets."""
        cap = CAPS[section]
        kept: list[Bullet] = []
        for b in bullets:
            text = " ".join(b.text.split())
            if not text or any(normalize(k.text) == normalize(text) for k in kept):
                continue
            kept.append(replace(b, text=text))
        if not kept:
            return "no usable bullets"
        self.sections[section] = kept[:cap]
        return None

    # --- output ----------------------------------------------------------------

    def enforce_caps(self) -> None:
        """Apply per-section caps via `spread()`. Idempotent; safe to call every step."""
        for section, cap in CAPS.items():
            if len(self.sections[section]) > cap:
                self.sections[section] = spread(self.sections[section], cap)

    def is_content_rich(self) -> bool:
        return any(self.sections[s] for s in BULLET_SECTIONS)

    def clone(self) -> NotesState:
        return NotesState(self.title, {s: list(b) for s, b in self.sections.items()})
