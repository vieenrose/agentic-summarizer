"""Transcript format v2 (SPEC §2) — timestamp-free, speaker-mandatory.

    <speaker>: <text>

One utterance = one line is a hard rule. Parsing splits on the FIRST `": "`.

**`parse_line` is total: it never raises.** SPEC §2 justifies the mandatory speaker field
precisely because it "keeps `parse_line` total and unambiguous". A line that does not
conform still has to produce *something*, and the only answer that never invents a speaker
is the reserved label `UNK` with the whole line as text. Conformance checking is a separate
concern, handled by `validate_v2` — which is also exactly what SPEC §4.3's hard pass/fail
translation gate needs.

Deliberately absent, relative to the prior project's format v1: `clock_to_sec`,
`sec_to_clock`, and `Utterance.start`. v2 has no timestamps, so the whole anchor family is
gone — and with it the mm/ss-inversion bug class that once corrupted evidence placement.
"""

from __future__ import annotations

from dataclasses import dataclass

#: SPEC §2: "Never contains `: `, and is <= 40 chars."
MAX_SPEAKER_LEN = 40

#: SPEC §2's reserved literal label, used when diarization is unavailable. Also the
#: fallback speaker for a non-conforming line — `validate_v2` is what tells the two apart.
UNK = "UNK"

SEP = ": "


@dataclass(frozen=True, slots=True)
class Utterance:
    """One transcript line. Speaker is mandatory (SPEC §2); there is no timestamp."""

    speaker: str
    text: str

    def render(self) -> str:
        """The ONE definition of the v2 wire format.

        Everything that measures or emits a line goes through here, so the chunker's cost
        model and the prompt render cannot disagree about what a line costs.
        """
        return format_line(self.speaker, self.text)


def format_line(speaker: str, text: str) -> str:
    """Emit one v2 line. Text is emitted as-is; v2 has no escaping (SPEC §2)."""
    return f"{speaker}{SEP}{text}"


def parse_line(line: str) -> tuple[str, str]:
    """Split on the FIRST `": "` (SPEC §2). Returns `(speaker, text)`. NEVER raises.

    A non-conforming line — no `": "` at all, or a first `": "` beyond `MAX_SPEAKER_LEN`
    — yields `(UNK, <the whole line>)`. Falling back rather than raising is what makes the
    function total; use `validate_v2` when you need to know that it happened.

    The length bound is what stops a `": "` inside the *text* of an undiarized line from
    being mistaken for a speaker delimiter.
    """
    idx = line.find(SEP)
    if idx == -1 or idx > MAX_SPEAKER_LEN or idx == 0:
        return UNK, line
    return line[:idx], line[idx + len(SEP) :]


def parse_transcript(text: str) -> list[Utterance]:
    """Parse a whole v2 transcript. Blank lines are skipped. Never raises."""
    return [parse_line_to_utterance(raw) for raw in text.splitlines() if raw.strip()]


def parse_line_to_utterance(line: str) -> Utterance:
    speaker, body = parse_line(line)
    return Utterance(speaker, body)


@dataclass(frozen=True, slots=True)
class LineDefect:
    """One v2 conformance violation, with enough context to fix the emitter."""

    lineno: int
    reason: str
    excerpt: str


def validate_v2(text: str) -> list[LineDefect]:
    """Strict conformance check (SPEC §2). Returns every defect; empty means conforming.

    Separate from `parse_line` on purpose: parsing must be total for the harness, while a
    corpus importer and SPEC §4.3's translation gate need to *fail loudly* instead of
    silently accepting a 200-character speaker.
    """
    defects: list[LineDefect] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        excerpt = raw[:60]
        idx = raw.find(SEP)
        if idx == -1:
            defects.append(LineDefect(lineno, "no ': ' separator", excerpt))
            continue
        if idx == 0:
            defects.append(LineDefect(lineno, "empty speaker field", excerpt))
            continue
        if idx > MAX_SPEAKER_LEN:
            defects.append(
                LineDefect(lineno, f"speaker field longer than {MAX_SPEAKER_LEN} chars", excerpt)
            )
            continue
        if not raw[idx + len(SEP) :].strip():
            defects.append(LineDefect(lineno, "empty text", excerpt))
            continue
        if any(ch in raw for ch in ("\r", "\x00")):
            defects.append(LineDefect(lineno, "embedded control character", excerpt))
    return defects


def line_count_matches(src: str, dst: str) -> bool:
    """SPEC §4.3: "assert input and output line counts match" — a hard pass/fail.

    Format v2 is one utterance per line, so a document-level translation that merges or
    splits utterances corrupts the format silently. This is the check that catches it.
    Blank lines are ignored on both sides, since they carry no utterance.
    """
    return _nonblank(src) == _nonblank(dst)


def _nonblank(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())
