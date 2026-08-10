"""Transcript format v1 — parsing, clocks, and the Utterance record.

Normative reference: CLAUDE.md §2. One utterance = one line, no embedded newlines.

    [<start>] <speaker>: <text>
    [<start>] <text>              (no diarization)

`clock_to_sec` / `sec_to_clock` are the two primitives everything downstream depends
on. The mm<->ss-inverted formula is a known past bug that corrupted evidence
placement (CLAUDE.md §7), so both directions are tested against padding edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Utterance",
    "clock_to_sec",
    "sec_to_clock",
    "format_line",
    "parse_line",
    "parse_transcript",
]

# M:SS (leading unit unpadded, seconds zero-padded) or H:MM:SS from one hour.
_CLOCK_RE = re.compile(r"^(?:(\d+):([0-5]\d):([0-5]\d)|(\d+):([0-5]\d))$")

MAX_SPEAKER_LEN = 40


@dataclass(frozen=True, slots=True)
class Utterance:
    """One transcript line. `start` is seconds; the clock text is derived, never stored."""

    start: int
    speaker: str | None
    text: str

    @property
    def clock(self) -> str:
        return sec_to_clock(self.start)

    def render(self) -> str:
        return format_line(self.start, self.speaker, self.text)


def clock_to_sec(clock: str) -> int:
    """`M:SS` -> M*60+S; `H:MM:SS` -> H*3600+M*60+S. Raises ValueError on anything else.

    Accepts an optional surrounding `[...]` so callers can pass an anchor verbatim.
    """
    s = clock.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    m = _CLOCK_RE.match(s)
    if m is None:
        raise ValueError(f"not a v1 clock: {clock!r}")
    if m.group(1) is not None:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return int(m.group(4)) * 60 + int(m.group(5))


def sec_to_clock(sec: int) -> str:
    """Inverse of `clock_to_sec`. `M:SS` under one hour, `H:MM:SS` from one hour.

    Seconds and minutes-in-hour are zero-padded; the leading unit is unpadded.
    """
    if sec < 0:
        raise ValueError(f"negative timestamp: {sec}")
    minutes, seconds = divmod(int(sec), 60)
    if minutes < 60:
        return f"{minutes}:{seconds:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_line(start: int, speaker: str | None, text: str) -> str:
    """Emit one v1 line. Text is emitted as-is; no escaping exists in v1."""
    head = f"[{sec_to_clock(start)}] "
    return f"{head}{speaker}: {text}" if speaker else f"{head}{text}"


def parse_line(line: str) -> tuple[str, str | None, str]:
    """Split on the FIRST `] `, then the FIRST `: ` after it (CLAUDE.md §2).

    Returns `(timestamp, speaker | None, text)` with the timestamp as its clock text.
    Raises ValueError if the line is not v1.

    A speaker field never contains `] ` or `: ` and is <= 40 chars, so a `: ` belonging
    to the *text* of an undiarized line cannot be mistaken for a speaker delimiter.
    """
    if not line.startswith("["):
        raise ValueError(f"line does not start with '[': {line[:40]!r}")
    close = line.find("] ")
    if close == -1:
        raise ValueError(f"no '] ' found: {line[:40]!r}")
    timestamp = line[1:close]
    clock_to_sec(timestamp)  # validate; raises on malformed clocks
    rest = line[close + 2 :]

    colon = rest.find(": ")
    if colon != -1 and colon <= MAX_SPEAKER_LEN:
        return timestamp, rest[:colon], rest[colon + 2 :]
    return timestamp, None, rest


def parse_transcript(text: str) -> list[Utterance]:
    """Parse a whole v1 transcript. Blank lines are skipped; bad lines raise."""
    out: list[Utterance] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            clock, speaker, body = parse_line(raw)
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
        out.append(Utterance(clock_to_sec(clock), speaker, body))
    return out
