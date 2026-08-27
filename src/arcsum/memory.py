"""External memory: ARC + POINTS (SPEC §4.1).

Two slots, harness-owned, token-capped: `ARC` (<= 80 tokens, 1-3 sentences of
meeting-level through-line) and `POINTS` (<= 16 entries of <= 25 tokens each, a
key point / decision / commitment per entry). Total <= ~600 tokens by construction.

**Token-length caps, not just count caps.** The prior project's `NotesState` only
enforced a cap on the *number* of bullets; the text length itself was merely requested
in the prompt, never checked. SPEC §4.1 makes `ARC <= 80 tokens` and `POINTS entries
<= 25 tokens` normative, so this module measures and enforces both.

**Refuse, never truncate, on an over-length write.** SPEC §4.2's replay rule is "never
half-applied into the corpus" — a truncated ARC is exactly a half-application: it trains
the model that over-long output is silently repaired rather than rejected. A refusal is
a clean, gradient-bearing signal, and a refused `ARC` simply leaves the previous arc
standing (stale but valid), which beats a mid-sentence truncation.

**`token_len` lives on `Memory`, not threaded per call.** The caps are an invariant of
the memory object, matching the injection pattern already used by `chunker`. Every
caller must pass the SAME counter it uses for chunking, or a heuristic-measured refusal
at trace-generation time and a real-tokenizer refusal at inference time silently
diverge — the exact bug class this design exists to prevent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from arcsum.tokens import char_tokens, heuristic_token_len, normalise

#: SPEC §4.1.
ARC_TOKENS = 80
POINT_TOKENS = 25
POINTS_CAP = 16

#: Floor for `DROP «prefix»` disambiguation, expressed in the normative char tokenizer
#: so it means the same thing in zh (4 ideographs) as in embedded latin (4 words) —
#: unlike the prior project's `MIN_PREFIX = 6` latin chars, which was ~one weak word in
#: en but a highly specific phrase in zh. Ambiguity, not length, is what `find()` guards.
MIN_PREFIX_TOKENS = 4


def normalize(text: str) -> str:
    """Comparison key for dedup and prefix matching.

    Adds punctuation folding on top of the prior project's `" ".join(split())`: that
    alone is a no-op for whitespace-free zh-TW (there is nothing to collapse), so
    without folding, two points differing only in punctuation would never dedup.

    Punctuation is dropped outright, not folded to a space: for zh-TW, punctuation never
    plays whitespace's role of separating adjacent characters (unlike latin, hanzi need
    no space between them), so the target case — two points differing only in a comma or
    full stop — dedups correctly. `arcsum.tokens.normalise` runs NFKC first, which folds
    fullwidth CJK punctuation (e.g. `，`) to its halfwidth ASCII form (`,`) — so the drop
    set below must cover both forms, not just the fullwidth ones a raw zh-TW string
    would contain.
    """
    folded = "".join(ch for ch in normalise(text) if ch not in _DROPPED_PUNCT)
    return " ".join(folded.split()).casefold()


_DROPPED_PUNCT = set("。，、；：！？「」『』（）〈〉《》【】…—～") | set(",.;:!?\"'()<>[]{}~-")


def spread(items: list, cap: int) -> list:
    """Reduce to `cap` items by spreading evenly across the list, preserving order.

    Never head-truncation (SPEC §4.1): head-truncating a time-ordered list drops the end
    of the meeting, where decisions land. Endpoints are always kept. Ported verbatim from
    the prior project — its docstring is nearly a paraphrase of SPEC §4.1's overflow rule.
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


@dataclass(frozen=True, slots=True)
class Point:
    """One POINTS entry.

    `chunk` is the emitting chunk index — a diagnostic, NEVER rendered into the prompt
    or the product output. It exists to distinguish same-step points (no ordering claim)
    from cross-step ones, feeding `guards.contradiction` and supervision reporting.
    """

    text: str
    chunk: int = -1


@dataclass
class Memory:
    """The whole carry-forward across steps (SPEC §4.1). No conversation history
    crosses steps — this object IS the memory.

    Every mutation returns a reason string on refusal, `None` on success. Never a bool,
    never an exception — the reason is what makes a dropped op explainable in the trace.
    """

    arc: str = ""
    points: list[Point] = field(default_factory=list)
    #: The budget instrument. MUST match whatever counter chunked the transcript, or
    #: caps are being measured against the wrong tokenizer. `compare=False`/`repr=False`
    #: because it is an instrument, not part of the memory's logical state.
    token_len: Callable[[str], int] = field(default=heuristic_token_len, compare=False, repr=False)

    def is_empty(self) -> bool:
        """No arc AND no points — the memory carries no information at all.

        Lives here rather than being re-derived at call sites so "empty" has exactly
        one definition. Deliberately STRICT (both slots empty), not a "thin memory"
        heuristic: `agent.synthesize_memory` keys a hard behavioural guarantee off
        this, and a fuzzy threshold there would be a quality judgement rather than the
        correctness invariant it is meant to express.
        """
        return not self.arc and not self.points

    def set_arc(self, text: str) -> str | None:
        """Replace the arc note. Refuses on empty, over-length, or unchanged text.

        **Unchanged is a refusal, mirroring `add_point`'s "duplicate point".** Rewriting
        the arc to what it already says changes nothing, so reporting it as a successful
        substantive edit makes the step-level metrics dishonest: `guards.apply_ops` marks
        the step `substantive`, which suppresses the NOP-collapse detector for a step
        that did no work. Measured 2026-08-27 on `LongBeachCC_05232017` (53 chunks, the
        longest meeting in the eval set): steps 26-40 each re-emitted a byte-identical
        arc while the transcript had already moved on to an unrelated agenda item, and
        every one of them counted as real work. Comparing on the same normal form
        `add_point` uses, so the two ops agree on what "the same text" means.

        **Gold traces still replay cleanly (SPEC §4.2).** Checked against the
        `sft-dropv2` pool's 3,345 unique gold steps: 29 pair a redundant arc line with a
        real ADD/DROP, so the step stays substantive and only the arc op is refused, and
        just 3 consist of nothing but a no-op arc — those become non-substantive, which
        is the honest reading of a step that changed nothing.
        """
        cleaned = " ".join(text.split())
        if not cleaned:
            return "empty arc"
        n = self.token_len(cleaned)
        if n > ARC_TOKENS:
            return f"arc too long ({n} > {ARC_TOKENS} tokens)"
        if self.arc and normalize(cleaned) == normalize(self.arc):
            return "arc unchanged"
        self.arc = cleaned
        return None

    def add_point(self, text: str, chunk: int) -> str | None:
        """Append a point. Refuses on empty, over-length, or exact-duplicate text."""
        cleaned = " ".join(text.split())
        if not cleaned:
            return "empty point"
        n = self.token_len(cleaned)
        if n > POINT_TOKENS:
            return f"point too long ({n} > {POINT_TOKENS} tokens)"
        key = normalize(cleaned)
        if any(normalize(p.text) == key for p in self.points):
            return "duplicate point"
        self.points.append(Point(cleaned, chunk))
        return None

    def find(self, prefix: str) -> int | None:
        """Index of the single point matching `prefix`, or `None` if absent/ambiguous.

        Ambiguity is a refusal, not a coin flip — silently dropping the wrong point is
        how a correct decision becomes an inverted one.
        """
        key = normalize(prefix)
        if len(char_tokens(key)) < MIN_PREFIX_TOKENS:
            return None
        hits = [i for i, p in enumerate(self.points) if normalize(p.text).startswith(key)]
        return hits[0] if len(hits) == 1 else None

    def drop_point(self, prefix: str) -> str | None:
        """Remove the point uniquely matched by `prefix`. Refuses if none or ambiguous."""
        idx = self.find(prefix)
        if idx is None:
            return "prefix did not match exactly one point"
        del self.points[idx]
        return None

    def enforce_caps(self) -> None:
        """Apply the POINTS count cap via `spread()`. Idempotent; safe every step."""
        if len(self.points) > POINTS_CAP:
            self.points = spread(self.points, POINTS_CAP)

    def prompt_tokens(self) -> int:
        """Token cost of this memory as it will be rendered into the next prompt."""
        from arcsum.render import render_memory  # local import: avoids a render<->memory cycle

        return self.token_len(render_memory(self))

    def clone(self) -> Memory:
        """A deep-enough copy for speculative mutation (e.g. within `apply_ops`)."""
        return replace(self, points=list(self.points))
