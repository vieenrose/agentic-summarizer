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

**The numeral-system false positive is FIXED, and it was not cosmetic.** This module used to
accept that a value reformatted between numeral systems (`兩百萬` vs `2000000`) read as
ungrounded. That looked like a rounding error and was not: the corpus writes figures in
Arabic while fluent zh-TW output writes them in CJK, so the mismatch fired SYSTEMATICALLY on
exactly the well-written summaries. Measured while building journal synthesis supervision, it
rejected 3 of 6 faithful teacher outputs — `六十` against a source `60`, `十二` against `12`,
`九十萬` against `90萬`. `cjk_to_int` + `numeral_forms` now fold both directions.

**Consequence for previously reported numbers**: the change is strictly more permissive, so
every ungrounded rate measured before it is an UPPER BOUND and is not comparable to one
measured after. Re-measure rather than compare across the boundary.

**Remaining false positives, accepted.** Approximations (`約五千多` against `5,200`) and
values that are computed rather than quoted still read as ungrounded. The rate is reported
(`n_checked`) alongside the count so a reader can judge whether a given delta is signal.

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
#:
#: `兩` and `〇` were missing until the numeral fold was built, and their absence silently
#: MIS-VALUED claims rather than merely missing them: `兩百萬` matched as `百萬`, which reads
#: as 1,000,000 — so a correct figure was compared against half its own value. Any character
#: `cjk_to_int` understands must appear here, or the two disagree about where a numeral starts.
CJK_NUMBER = re.compile(r"[零〇一二兩三四五六七八九十百千萬億兆壹貳參肆伍陸柒捌玖拾佰仟]{2,}")

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


#: CJK numeral characters, split by the role they play in a numeral. `兩` is included as a
#: digit because `兩百萬` is ordinary zh-TW for 2,000,000.
_CJK_DIGIT = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "壹": 1,
    "貳": 2,
    "參": 3,
    "肆": 4,
    "伍": 5,
    "陸": 6,
    "柒": 7,
    "捌": 8,
    "玖": 9,
}
_CJK_UNIT = {"十": 10, "百": 100, "千": 1000, "拾": 10, "佰": 100, "仟": 1000}
_CJK_BIG = {"萬": 10**4, "億": 10**8, "兆": 10**12}


def cjk_to_int(token: str) -> int | None:
    """The integer a CJK numeral denotes, or `None` if it is not one.

    Handles both ways zh writes numbers, because both occur in this corpus: MULTIPLICATIVE
    (`六十` = 60, `九十萬` = 900,000) whenever a unit character is present, and POSITIONAL
    (`二零一六` = 2016, how years are written) when the token is bare digits.
    """
    if not token:
        return None
    if any(c in _CJK_UNIT or c in _CJK_BIG for c in token):
        total = section = number = 0
        for ch in token:
            if ch in _CJK_DIGIT:
                number = _CJK_DIGIT[ch]
            elif ch in _CJK_UNIT:
                section += (number or 1) * _CJK_UNIT[ch]
                number = 0
            elif ch in _CJK_BIG:
                total += (section + number or 1) * _CJK_BIG[ch]
                section = number = 0
            else:
                return None
        return total + section + number
    if all(c in _CJK_DIGIT for c in token):
        return int("".join(str(_CJK_DIGIT[c]) for c in token))
    return None


def numeral_forms(value: int) -> set[str]:
    """The written forms a value plausibly takes, as normalised match keys.

    Beyond the plain integer this emits the MIXED forms zh actually uses — `900000` is
    written `90萬` far more often than in full — because grounding is a containment test
    and `90萬` does not contain `900000`.
    """
    forms = {str(value)}
    for ch, mag in _CJK_BIG.items():
        if value >= mag and value % mag == 0:
            forms.add(f"{value // mag}{ch}")
    return forms


def normalise_for_match(text: str) -> str:
    """Fold the differences that are formatting rather than fact.

    NFKC (via `arcsum.tokens.normalise`, the single source of truth for this project's
    normalisation) folds fullwidth digits onto ASCII, so a source `１２億` grounds a
    summary `12億`. Whitespace and the punctuation that CJK text sprinkles between digits
    are then removed, so `12,000` grounds `12000`.
    """
    folded = normalise(text)
    return re.sub(r"[\s,，、.。·\-–—]", "", folded)


def _haystack(source: str) -> str:
    """The source, plus an Arabic rendering of every CJK numeral it contains.

    **This is what makes the numeral test symmetric**, and it corrects a false-positive class
    the module previously documented as accepted. The corpus writes figures in Arabic
    (`60天`, `12個月`) while fluent zh-TW output writes them in CJK (`六十天`, `十二個月`), so
    a literal containment test called correct prose a fabrication. Measured while building
    journal synthesis supervision: this alone rejected 3 of 6 teacher outputs, every one of
    them faithful — `六十` against a source `60`, `十二` against `12`, `九十萬` against `90萬`.

    Appending the converted forms handles BOTH directions with one mechanism: a CJK claim is
    checked against its own Arabic variant here, and an Arabic claim matches the rendering
    appended for a CJK source. Doing it on the haystack rather than per claim keeps `check`
    a single containment test.

    Note this makes the instrument STRICTLY MORE PERMISSIVE — it can only turn a reported
    fabrication into a pass, never the reverse. Rates measured before this change are
    therefore upper bounds, not comparable point estimates.
    """
    base = normalise_for_match(source)
    extra = set()
    for m in CJK_NUMBER.findall(source):
        v = cjk_to_int(normalise_for_match(m))
        if v is not None:
            extra |= numeral_forms(v)
    return base + ("\n" + "\n".join(sorted(extra)) if extra else "")


def _claim_forms(claim: str) -> set[str]:
    """A claim plus the alternate numeral renderings that denote the same value."""
    forms = {claim}
    v = cjk_to_int(claim)
    if v is not None:
        forms |= numeral_forms(v)
    return forms


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
    """Grounding of one summary against the transcript it was built from.

    A claim counts as grounded when ANY of its equivalent numeral renderings appears; see
    `_haystack` for why the numeral-system fold is required rather than cosmetic.
    """
    haystack = _haystack(source)
    claims = claims_in(summary)
    return GroundingReport(
        meeting=meeting,
        n_checked=len(claims),
        ungrounded=tuple(c for c in claims if not any(f in haystack for f in _claim_forms(c))),
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
