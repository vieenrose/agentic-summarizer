"""Line-keyed lexical retrieval for judge evidence (SPEC §5.1: "retrieved transcript
spans").

**Re-keyed on line number, not on a timestamp anchor.** The prior project's index
built a claim-vs-anchor split and a `±3-line` neighbourhood centred on each bullet's
anchor — none of which has a v2 analogue, since SPEC §3's output carries no anchors at
all. What survives is the pure lexical retrieval underneath it: a claim's evidence is
simply "the top-k most lexically similar transcript lines," full stop.

**`EVIDENCE_ORDER` is pinned, and this is not incidental.** In the prior project,
reordering IDENTICAL evidence — same lines, same content, different presentation
order — moved a judge's FAITH score by 0.60 and flipped 30% of verdicts, both larger
than any tie band the gates use. The finding is about presentation order, not about
anchors, so it survives the redesign untouched: evidence is always emitted
highest-lexical-score first, ties broken by original transcript line order, never left
to whatever a dict or set iteration happens to produce.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arcsum.tokens import lexical_tokens
from arcsum.transcript import Utterance

EVIDENCE_ORDER = "score_desc_then_line"


@dataclass(frozen=True, slots=True)
class Evidence:
    line: int
    text: str
    score: float

    def render(self) -> str:
        return self.text


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TranscriptIndex:
    """Lexical top-k retrieval over a whole transcript, keyed by line number."""

    def __init__(self, utterances: Sequence[Utterance]) -> None:
        self._lines = [u.render() for u in utterances]
        self._tokens = [lexical_tokens(u.text) for u in utterances]

    def search(self, query: str, *, top_k: int = 6) -> list[Evidence]:
        """Top-`k` lines by lexical overlap with `query`. Zero-overlap lines are
        excluded entirely — evidence with no lexical connection to the claim is not
        evidence. `EVIDENCE_ORDER`: score descending, ties broken by line ascending.
        """
        q = lexical_tokens(query)
        scored = [
            Evidence(line=i, text=self._lines[i], score=_overlap(q, toks))
            for i, toks in enumerate(self._tokens)
        ]
        scored = [e for e in scored if e.score > 0]
        scored.sort(key=lambda e: (-e.score, e.line))
        return scored[:top_k]
