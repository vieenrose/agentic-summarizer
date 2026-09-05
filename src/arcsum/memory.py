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

import re
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

#: Character-trigram containment at which two points are "the same point said twice", for
#: `synthesis_view`'s presentation-level dedup only. Chosen high: this collapses a rendering,
#: and merging two genuinely distinct decisions would lose content, which is the more
#: expensive error. At 0.6 the measured near-duplicate rate in the journal-shaped supervision
#: slice is 11.2% against 5.6% in the pre-journal one.
NEAR_DUPLICATE_CONTAINMENT = 0.6


def _trigrams(text: str) -> set[str]:
    s = normalize(text)
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else ({s} if s else set())


#: Numerals are the one thing that must never be folded away by a similarity test. Kept as a
#: local pattern rather than importing `evalkit.grounding`: `memory` is harness core and the
#: evaluation package depends on it, never the reverse.
_NUMERAL = re.compile(r"\d+|[零〇一二兩三四五六七八九十百千萬億兆]{1,}")


def _near_duplicate(a: str, b: str) -> bool:
    """Symmetric near-duplicate test.

    Symmetric on purpose — containment alone is directional, so a short point fully contained
    in a longer one scores 1.0 in one direction and much less in the other. Taking the MAX
    treats "同意搬到 B 棟" and "同意搬到 B 棟大樓並於三月完成" as the same point, which for a
    summary's purposes they are.

    **Points carrying different numerals are never duplicates, however similar the rest.**
    Measured while calibrating this: `第1項決議` and `第11項決議` score 0.667 and would have
    merged, and this corpus is full of agenda items, ordinance numbers and dollar amounts
    that differ in exactly one figure. Collapsing those loses a distinct decision — the
    expensive error — while failing to collapse a true duplicate merely leaves the redundancy
    this function is trying to reduce.
    """
    if set(_NUMERAL.findall(normalize(a))) != set(_NUMERAL.findall(normalize(b))):
        return False
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return ta == tb
    inter = len(ta & tb)
    return max(inter / len(ta), inter / len(tb)) >= NEAR_DUPLICATE_CONTAINMENT


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

    `pid` (SPEC §4.1 v1.1) is a stable integer identity, and unlike `chunk` it IS
    rendered, because the model addresses points by it. v1.0 addressed by text prefix,
    which required the model to reproduce a prefix of its own earlier phrasing; the
    resulting `DROP «X»` + near-identical `ADD «X'»` churn measured 28.2% of steps on
    real ASR. An id cannot be mis-addressed by a model that can see it numbered.
    0 means "unassigned" — only `Memory.add_point` mints ids, so a hand-built `Point`
    in a test cannot silently collide with a real one.
    """

    text: str
    chunk: int = -1
    pid: int = 0


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """A point that has left the WORKING SET, kept for `SYNTHESIZE` (SPEC §4.1 v1.1).

    **Nothing the model recorded is ever destroyed.** v1.0 deleted evicted and dropped
    points outright, and measured on the three longest held-out meetings the model
    correctly recorded 41, 23 and 27 points of which 80%, 65% and 48% were gone before
    synthesis ran. The journal is the fix, and it is free at read time because the model
    never sees it — only `SYNTHESIZE` does.

    `reason` distinguishes the three ways a point leaves, which the metrics need kept
    apart: `evicted` (cap overflow, the harness's choice), `dropped` (the model closed
    it out), `superseded` (the model replaced it, and `superseded_by` names the pid that
    replaced it). A reversal is `superseded`, and it is the case G1 exists to measure.
    """

    point: Point
    reason: str
    superseded_by: int = 0


@dataclass
class Memory:
    """The whole carry-forward across steps (SPEC §4.1). No conversation history
    crosses steps — this object IS the memory.

    Every mutation returns a reason string on refusal, `None` on success. Never a bool,
    never an exception — the reason is what makes a dropped op explainable in the trace.

    **v1.1 splits the memory in two.** `points` is the WORKING SET: bounded, rendered
    into every step's prompt, and therefore re-prefilled every step (~19% of the
    transcript again over 37 chunks — which is why it cannot simply be made bigger).
    `journal` is append-only, model-invisible, and unbounded; it is what `SYNTHESIZE`
    reads. The split exists because one slot was serving two jobs with opposite size
    requirements, and the small one won.
    """

    arc: str = ""
    points: list[Point] = field(default_factory=list)
    #: Append-only record of every point that has left the working set. The model NEVER
    #: sees this; `render.py` does not touch it and no step prompt includes it.
    journal: list[JournalEntry] = field(default_factory=list)
    #: Monotonic id source. Never reused, so a pid in the journal can always be traced
    #: back even after the point has left the working set.
    _next_pid: int = 0
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
        self._next_pid += 1
        self.points.append(Point(cleaned, chunk, self._next_pid))
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
        """Remove the point uniquely matched by `prefix`, retiring it to the journal.

        Retained for the v0.9 edit protocol and for tool-call rows that still address by
        text. New supervision should use `drop_id`/`revise_id` (SPEC §4.1 v1.1).
        """
        idx = self.find(prefix)
        if idx is None:
            return "prefix did not match exactly one point"
        self.journal.append(JournalEntry(self.points[idx], "dropped"))
        del self.points[idx]
        return None

    def _index_of(self, pid: int) -> int | None:
        return next((i for i, p in enumerate(self.points) if p.pid == pid), None)

    def drop_id(self, pid: int) -> str | None:
        """Retire the point with this id from the working set (SPEC §4.1 v1.1).

        Refusal names the id rather than the text, so a trace shows exactly what the
        model asked for — an id that never existed and one already retired are
        different mistakes and read differently in the metrics.
        """
        idx = self._index_of(pid)
        if idx is None:
            return f"no point with id {pid}"
        self.journal.append(JournalEntry(self.points[idx], "dropped"))
        del self.points[idx]
        return None

    def revise_id(self, pid: int, text: str) -> str | None:
        """Atomically supersede point `pid` with `text` (SPEC §4.1 v1.1).

        **This op exists because DROP-then-ADD is what produced churn.** v1.0 had no way
        to say "this point is now wrong, here is the correction" in one act, so revision
        was two ops the harness could not tell apart from a model rewriting what it
        already had — `guards.restates_dropped` fires on both by construction. Here the
        supersession is explicit, journalled with a `superseded_by` link, and therefore
        separable from churn in the data and in the metrics.

        Validated exactly as `add_point` validates, then applied as one unit: a refused
        revision leaves the memory untouched, never half-applied (SPEC §4.2).
        """
        idx = self._index_of(pid)
        if idx is None:
            return f"no point with id {pid}"
        cleaned = " ".join(text.split())
        if not cleaned:
            return "empty point"
        n = self.token_len(cleaned)
        if n > POINT_TOKENS:
            return f"point too long ({n} > {POINT_TOKENS} tokens)"
        old = self.points[idx]
        if normalize(cleaned) == normalize(old.text):
            return "revision unchanged"
        self._next_pid += 1
        new = Point(cleaned, old.chunk, self._next_pid)
        self.journal.append(JournalEntry(old, "superseded", superseded_by=new.pid))
        self.points[idx] = new
        return None

    def synthesis_view(self) -> list[JournalEntry]:
        """Everything the meeting produced, for `SYNTHESIZE` (SPEC §4.1 v1.1), with
        near-duplicates collapsed.

        Journal entries in the order they left, then the surviving working set as
        `reason="kept"`. Superseded points are retained WITH their link rather than
        filtered out: a summary that must report the final state still needs to know a
        reversal happened, which is precisely what G1 measures.

        **The dedup is not tidying — it repairs a measured regression.** `apply_ops` refuses
        only EXACT duplicate points, so near-duplicates accumulate; before v1.1 eviction hid
        most of them, and now nothing removes them because retiring a point no longer destroys
        it. Measured on the supervision slices: near-duplicate entries went **5.6% -> 11.2%**
        when the view became journal-shaped. `runs/v12-e3` then trained a coverage-gated
        teacher on that view, which dutifully restated both halves of every pair, and the
        student generalised "redundancy is correct output" into the reading step —
        **churn 3.5% -> 29.8%, 23 -> 197 events, paired p = 2.2e-07**.

        Collapsing at PRESENTATION rather than tightening the reading step's duplicate guard
        is deliberate: the guard would have to refuse an `ADD` at write time, when it cannot
        yet know whether the two points diverge later, and a wrongly refused `ADD` loses
        content permanently. Here nothing is lost — the journal still holds every point, and
        only the rendering merges them.

        A LATER entry wins over an earlier near-duplicate, because the later phrasing reflects
        the more complete reading of the meeting; superseded entries are exempt, since their
        whole purpose is to sit beside the text that replaced them.
        """
        entries = list(self.journal) + [JournalEntry(p, "kept") for p in self.points]
        out: list[JournalEntry] = []
        for e in entries:
            if e.reason == "superseded":
                out.append(e)
                continue
            dup = next((i for i, k in enumerate(out)
                        if k.reason != "superseded"
                        and _near_duplicate(k.point.text, e.point.text)), None)
            if dup is None:
                out.append(e)
            else:
                out[dup] = e
        return out

    def enforce_caps(self) -> None:
        """Apply the POINTS count cap via `spread()`, RETIRING the evicted points to the
        journal rather than deleting them (SPEC §4.1 v1.1). Idempotent; safe every step.

        `spread` still chooses WHICH points stay visible — evenly, never head-truncated,
        because dropping the tail of a time-ordered list drops the end of the meeting
        where decisions land. What changes in v1.1 is only that eviction costs
        working-set attention and not information.
        """
        if len(self.points) > POINTS_CAP:
            kept = spread(self.points, POINTS_CAP)
            keep_ids = {id(p) for p in kept}
            for p in self.points:
                if id(p) not in keep_ids:
                    self.journal.append(JournalEntry(p, "evicted"))
            self.points = kept

    def prompt_tokens(self) -> int:
        """Token cost of this memory as it will be rendered into the next prompt."""
        from arcsum.render import render_memory  # local import: avoids a render<->memory cycle

        return self.token_len(render_memory(self))

    def clone(self) -> Memory:
        """A deep-enough copy for speculative mutation (e.g. within `apply_ops`).

        The journal is copied too: `apply_ops` mutates a clone and keeps it only if the
        step succeeds, so a speculative eviction must not leak into the real journal.
        """
        return replace(self, points=list(self.points), journal=list(self.journal))
