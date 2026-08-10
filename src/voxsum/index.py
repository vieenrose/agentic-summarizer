"""Lexical search and snippet extraction for judge evidence (CLAUDE.md §5.3, §7.2).

The judge never sees a whole transcript in claim mode — it sees ≤ 6 snippets of ≤ 120
chars. How those are chosen decides what the judge *can* conclude, so two details from §7.2
are load-bearing:

* **Tokenisation is language-aware.** Word overlap for en; character bigrams for zh, which
  has no spaces — word-splitting Chinese produces one giant token and every score collapses
  to 0 or 1.
* **A snippet is extracted at the best-matching window *inside* the line.** VCSum zh lines
  run to ~2.6k chars. Truncating such a line at 120 chars from the start would hand the
  judge the opening pleasantries of a monologue whose decision is 2k chars later — the
  claim would read as unsupported when the transcript supports it perfectly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .transcript import Utterance, sec_to_clock

__all__ = ["EVIDENCE_ORDER", "Evidence", "SNIPPET_CHARS", "TranscriptIndex", "tokenise"]

#: Pinned evidence ordering. NOT cosmetic: judging identical bullets with identical evidence
#: under four orderings moved FAITH by 0.60 and flipped 30% of verdicts (RESULTS.md) — larger
#: than the 0.5 tie band the ship gates use. Ordering must therefore be identical across arms
#: and across runs, or a comparison partly measures presentation. Changing this value
#: invalidates comparison with every number recorded before the change; bump PROMPT_VERSION
#: alongside it if you ever do.
EVIDENCE_ORDER = "anchor_first"

SNIPPET_CHARS = 120
NEIGHBOURHOOD = 3  # +/- lines around the anchor (§7.1: FAITH-anchor is anchor +/- 3 lines)


def tokenise(text: str) -> set[str]:
    """Word tokens for latin script; character bigrams for CJK. Both, for mixed text."""
    lowered = text.casefold()
    words = {w for w in "".join(c if c.isalnum() else " " for c in lowered).split() if w}
    cjk = [c for c in text if "㐀" <= c <= "鿿" or "豈" <= c <= "﫿"]
    words |= {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
    return words


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass(frozen=True, slots=True)
class Evidence:
    """One snippet handed to the judge."""

    anchor: int
    text: str
    from_anchor_neighbourhood: bool

    def render(self) -> str:
        return f"[{sec_to_clock(self.anchor)}] {self.text}"


class TranscriptIndex:
    """Lexical index over one transcript. Deterministic; no model tokens (§5.3)."""

    def __init__(self, utterances: list[Utterance]) -> None:
        self.utterances = list(utterances)
        self._tokens = [tokenise(u.text) for u in self.utterances]
        self._by_anchor: dict[int, int] = {u.start: i for i, u in enumerate(self.utterances)}

    def __len__(self) -> int:
        return len(self.utterances)

    def line_index(self, anchor: int) -> int | None:
        return self._by_anchor.get(anchor)

    def snippet(self, line: int, query: str, *, width: int = SNIPPET_CHARS) -> str:
        """The `width`-char window of line `line` that best matches `query`.

        Scans at a stride rather than every offset: on a 2.6k-char line an exhaustive scan
        is wasteful, and the best window is not sensitive to a few characters of alignment.
        """
        text = self.utterances[line].text
        if len(text) <= width:
            return text
        q = tokenise(query)
        if not q:
            return text[:width]
        stride = max(width // 4, 1)
        best, best_score = text[:width], -1.0
        for start in range(0, len(text) - width + stride, stride):
            window = text[start : start + width]
            score = _overlap(q, tokenise(window))
            if score > best_score:
                best, best_score = window, score
        return best

    def neighbourhood(
        self, anchor: int, query: str, *, radius: int = NEIGHBOURHOOD
    ) -> list[Evidence]:
        """Lines within `radius` of the anchor, anchor line first. Empty if unresolved.

        Ordering is anchor-centred (anchor, then ±1, ±2, …) so the bullet's own claimed
        support line survives a partial budget: claim mode reserves half its slots for
        whole-transcript retrieval, and `near[:near_budget]` must still include the anchor
        line or a correctly-anchored bullet reads as unsupported in claim mode.
        """
        centre = self.line_index(anchor)
        if centre is None:
            return []
        lo, hi = max(centre - radius, 0), min(centre + radius + 1, len(self.utterances))
        order = sorted(range(lo, hi), key=lambda i: (abs(i - centre), i))
        return [
            Evidence(self.utterances[i].start, self.snippet(i, query), True)
            for i in order
        ]

    def search(
        self, query: str, *, top_k: int = 6, exclude: set[int] | None = None
    ) -> list[Evidence]:
        """Top-k lines across the WHOLE transcript by lexical overlap (claim mode)."""
        q = tokenise(query)
        if not q:
            return []
        skip = exclude or set()
        scored = [
            (_overlap(q, self._tokens[i]), i)
            for i in range(len(self.utterances))
            if self.utterances[i].start not in skip
        ]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            Evidence(self.utterances[i].start, self.snippet(i, query), False)
            for _, i in scored[:top_k]
        ]

    def evidence_for(
        self, bullet: str, anchor: int | None, *, mode: str = "claim", limit: int = 6
    ) -> list[Evidence]:
        """Evidence for one bullet.

        `claim` — anchor neighbourhood ∪ lexical top-k over the whole transcript, so a true
        claim anchored at the wrong line still scores as supported (§7.1: FAITH-claim and
        FAITH-anchor are reported separately precisely because they differ).

        `anchor` — the anchor neighbourhood only. A bullet whose anchor is wrong *should*
        fail this one.
        """
        if mode not in ("claim", "anchor"):
            raise ValueError(f"unknown evidence mode: {mode!r}")

        near = self.neighbourhood(anchor, bullet) if anchor is not None else []
        if mode == "anchor":
            return near[:limit]

        # Reserve slots for the retrieved half. A +/-3 neighbourhood yields up to 7 lines,
        # so a naive `(near + found)[:limit]` fills every slot from the neighbourhood and
        # silently discards the whole-transcript search — which made claim mode identical
        # to anchor mode, defeating the separation §7.1 exists to draw and inflating
        # UNSUPPORTED for any true claim whose support lies elsewhere in the meeting.
        # The anchor line is guaranteed a slot because `neighbourhood` is anchor-first.
        near_budget = max(limit // 2, 1)
        head = near[:near_budget]
        # Exclude only the lines actually shown from the search, not the whole
        # neighbourhood. Lines at anchor +/-2 or +/-3 are inside `near` (FAITH-anchor's own
        # window) but outside the anchor-first `head`, so excluding every near line would
        # drop a support line at +/-2/+/-3 from claim mode entirely — the bullet's true
        # evidence would reach neither the neighbourhood slice nor the retrieval.
        shown = {e.anchor for e in head}
        found = self.search(bullet, top_k=limit, exclude=shown)
        tail = found[: limit - near_budget]
        # Backfill from whichever side has spares, so a bullet is never under-evidenced
        # just because one source came up short. Dedup against what is already shown: the
        # whole-transcript search may now surface a +/-2/+/-3 line the spare neighbourhood
        # also carries.
        if len(head) + len(tail) < limit:
            have = shown | {e.anchor for e in tail}
            spare_near = [e for e in near[len(head) :] if e.anchor not in have]
            spare_found = [e for e in found[len(tail) :] if e.anchor not in have]
            for extra in (*spare_near, *spare_found):
                if len(head) + len(tail) >= limit:
                    break
                have.add(extra.anchor)
                tail.append(extra)
        # Neighbourhood first (EVIDENCE_ORDER = "anchor_first"): it is what FAITH-anchor
        # would see, so a judge reading in order encounters the anchored evidence before the
        # retrieved evidence. The ordering is pinned — see EVIDENCE_ORDER.
        return head + tail
