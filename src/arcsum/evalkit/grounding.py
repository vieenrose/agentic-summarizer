"""Reference-free fabrication detection: does every specific claim in the summary appear
in the transcript it came from?

**Why a second faithfulness instrument at all.** G2 runs one LLM judge, and that judge has
a measured failure mode: gpt-oss can spend its whole budget in `reasoning_content` and
return empty `content`, which cost 21 of 40 baseline meetings and did so SYSTEMATICALLY on
the longest summaries (median 5,087 chars vs 562). The gate saw only the control arm's
shortest outputs and reported "14 vs 11, FAIL" for what was really 18 vs 109. A single
instrument whose failures correlate with the thing being measured cannot detect its own
bias; a second, independent one can.

This instrument is deliberately the OPPOSITE KIND of thing from an LLM judge: fully
deterministic, no model, no network, no budget, no retries, sub-second per meeting. It
cannot be subtle and does not try to be. It answers exactly one question — is this
specific token in the source — and it answers it the same way every time.

**What it generalises.** `prose.ungrounded_numbers` checks Arabic-digit spans against the
MEMORY. Its own docstring states two scope limits, and both are lifted here:

* it is blind to CJK numerals, while the same investigation found fabricated
  `二零一六年六月三十日` and `十五萬美元`; `CJK_NUMBER` covers those.
* it grounds against memory, not the transcript. Memory is an intermediate the model
  wrote, so a detail fabricated during the READING step is "grounded" by that definition
  and invisible. Grounding against the source transcript catches fabrication wherever in
  the pipeline it entered.

**What it deliberately does NOT do.** It does not check whether a true statement is
RELEVANT, whether a relation between two grounded entities is invented, or whether an
omission distorts. Those need a judge. This is the floor, not the ceiling: a summary can
score perfectly here and still be badly unfaithful. Report it beside G2, never instead.

**Known false positives, accepted.** A legitimately-present value reformatted between
source and summary (`兩百萬` -> `2000000`) reads as ungrounded, because grounding is a
literal containment test rather than a parse. `normalise_for_match` folds width and
punctuation, which removes the largest class of these, but not reformatting across numeral
systems. The rate is reported (`n_checked`) alongside the count so a reader can judge
whether a given delta is signal.

**Known false negatives, also accepted.** The claim token is the NUMERAL alone; units are
not part of it. So a summary asserting `十五萬美元` against a source that says `十五萬人`
grounds successfully — the figure is present, the unit was invented, and this instrument
cannot tell. Widening the token to include units would catch that at the cost of a much
larger class of false positives (the same figure legitimately carrying different units in
source and summary). The floor is chosen deliberately low: this exists to make fabrication
CHEAP to detect at all, beneath a judge that can weigh meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from arcsum.tokens import normalise

#: Arabic-digit runs of at least TWO digits. Single digits are excluded deliberately: they
#: appear constantly as ordinals and enumeration markers (`第 2 項`, `1.`) where a literal
#: containment test says nothing about faithfulness, and measured on the training pool they
#: dominated the flagged set with tokens like `'1'` and `'2'`. A one-digit fabrication is
#: real but rare; the noise it admits is not worth it.
ARABIC_NUMBER = re.compile(r"\d[\d,.]*\d")

#: CJK numerals, the class `prose.ungrounded_numbers` states it cannot see. Requires two
#: characters so that a bare 一 or 十 inside ordinary prose is not treated as a claim.
CJK_NUMBER = re.compile(r"[零一二三四五六七八九十百千萬億兆壹貳參肆伍陸柒捌玖拾佰仟]{2,}")

#: Latin/alphanumeric identifiers: case numbers, ordinance ids, model names. These are
#: high-value because they are exactly what a model invents when it pattern-matches a
#: document genre it half-remembers.
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


@dataclass(frozen=True)
class GroundingReport:
    """One meeting's grounding result.

    `ungrounded_rate` is the headline. `n_checked` must travel with it: a summary that
    asserts nothing specific scores a perfect 0.0 and is not thereby faithful — it is
    empty. Reading the rate without the denominator would reward exactly the abstention
    failure the real-ASR gate exists to catch.
    """

    meeting: str
    n_checked: int
    ungrounded: tuple[str, ...]

    @property
    def n_ungrounded(self) -> int:
        return len(self.ungrounded)

    @property
    def ungrounded_rate(self) -> float:
        return self.n_ungrounded / self.n_checked if self.n_checked else 0.0


def normalise_for_match(text: str) -> str:
    """Fold the differences that are formatting rather than fact.

    NFKC (via `arcsum.tokens.normalise`, the single source of truth for this project's
    normalisation) folds fullwidth digits onto ASCII, so a source `１２億` grounds a
    summary `12億`. Whitespace and the punctuation that CJK text sprinkles between digits
    are then removed, so `12,000` grounds `12000`.
    """
    folded = normalise(text)
    return re.sub(r"[\s,，、.。·\-–—]", "", folded)


def claims_in(text: str) -> tuple[str, ...]:
    """Every specific, checkable token in `text`, deduplicated and order-stable.

    Deduplicated because a summary repeating one fabricated figure three times is one
    fabrication, not three — counting occurrences would let verbosity dominate the metric.
    """
    seen: dict[str, None] = {}
    for pattern in (ARABIC_NUMBER, CJK_NUMBER, IDENTIFIER):
        for m in pattern.findall(text):
            key = normalise_for_match(m)
            if key:
                seen.setdefault(key, None)
    return tuple(seen)


def check(meeting: str, summary: str, source: str) -> GroundingReport:
    """Grounding of one summary against the transcript it was built from."""
    haystack = normalise_for_match(source)
    claims = claims_in(summary)
    return GroundingReport(
        meeting=meeting,
        n_checked=len(claims),
        ungrounded=tuple(c for c in claims if c not in haystack),
    )


@dataclass(frozen=True)
class GroundingSummary:
    """Aggregate over meetings, shaped for `metrics.stats.compare`'s paired protocol."""

    n_meetings: int
    total_checked: int
    total_ungrounded: int
    meetings_with_any: int
    per_meeting: tuple[GroundingReport, ...]

    @property
    def ungrounded_rate(self) -> float:
        return self.total_ungrounded / self.total_checked if self.total_checked else 0.0


def summarise(reports: list[GroundingReport]) -> GroundingSummary:
    return GroundingSummary(
        n_meetings=len(reports),
        total_checked=sum(r.n_checked for r in reports),
        total_ungrounded=sum(r.n_ungrounded for r in reports),
        meetings_with_any=sum(1 for r in reports if r.n_ungrounded),
        per_meeting=tuple(reports),
    )
