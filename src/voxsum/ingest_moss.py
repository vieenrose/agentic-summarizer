"""MOSS-Transcribe-Diarize 0.9B output -> transcript format v1.

MOSS emits one flat string of segments (its canonical format):

    [start][Sxx]transcribed speech[end][start][Sxx]...

with times in seconds (float). We convert to CLAUDE.md §2 v1 lines:

  * `[S01]` -> `S1`, renumbered by **first appearance** (v1 requires first-appearance
    order; MOSS labels are only relative within the audio, and their numbering does not
    have to start at 1 or be dense).
  * float seconds -> `M:SS` / `H:MM:SS` via `sec_to_clock` (floor, matching "utterance
    start only").
  * consecutive segments from the same speaker are merged into one utterance when the
    gap is small, because MOSS segments at phrase granularity while v1 wants utterances.
  * embedded newlines are collapsed — one utterance = one line is a hard rule.

Acoustic event annotations (MOSS can emit e.g. `(laughter)`) are kept verbatim by
default; pass `drop_events=True` to strip bracketed non-speech markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .transcript import Utterance, format_line

__all__ = ["MossSegment", "parse_moss_output", "moss_to_utterances", "moss_to_v1"]

# [1.23][S01]text[4.56]  — text runs up to the next '[' that opens a timestamp.
_SEG_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]"
    r"(?:\[(?P<spk>S\d+)\])?"
    r"(?P<text>.*?)"
    r"\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)
_EVENT_RE = re.compile(r"[(（\[]\s*(?:laughter|applause|music|noise|silence)\s*[)）\]]", re.I)
_WS_RE = re.compile(r"\s+")

# Merge same-speaker segments separated by less than this many seconds of silence.
DEFAULT_MERGE_GAP = 2.0


@dataclass(frozen=True, slots=True)
class MossSegment:
    start: float
    end: float
    speaker: str | None
    text: str


def parse_moss_output(raw: str, *, drop_events: bool = False) -> list[MossSegment]:
    """Parse MOSS's flat canonical string into segments, in emission order."""
    segments: list[MossSegment] = []
    for m in _SEG_RE.finditer(raw):
        text = m.group("text")
        if drop_events:
            text = _EVENT_RE.sub(" ", text)
        text = _WS_RE.sub(" ", text).strip()
        if not text:
            continue
        segments.append(
            MossSegment(
                start=float(m.group("start")),
                end=float(m.group("end")),
                speaker=m.group("spk"),
                text=text,
            )
        )
    return segments


def moss_to_utterances(
    segments: list[MossSegment],
    *,
    merge_gap: float | None = DEFAULT_MERGE_GAP,
    join: str = " ",
) -> list[Utterance]:
    """Renumber speakers by first appearance, merge adjacent same-speaker segments.

    `merge_gap=None` disables merging (one v1 line per MOSS segment).
    """
    label_map: dict[str, str] = {}

    def relabel(moss_label: str | None) -> str | None:
        if moss_label is None:
            return None
        if moss_label not in label_map:
            label_map[moss_label] = f"S{len(label_map) + 1}"
        return label_map[moss_label]

    out: list[Utterance] = []
    prev_end = 0.0
    for seg in sorted(segments, key=lambda s: s.start):
        speaker = relabel(seg.speaker)
        mergeable = (
            merge_gap is not None
            and out
            and out[-1].speaker == speaker
            and speaker is not None
            and seg.start - prev_end < merge_gap
        )
        if mergeable:
            last = out[-1]
            out[-1] = Utterance(last.start, last.speaker, f"{last.text}{join}{seg.text}")
        else:
            out.append(Utterance(int(seg.start), speaker, seg.text))
        prev_end = seg.end
    return out


def moss_to_v1(
    raw: str,
    *,
    merge_gap: float | None = DEFAULT_MERGE_GAP,
    drop_events: bool = False,
) -> str:
    """MOSS raw output -> a complete v1 transcript (no header, no footer, LF-terminated)."""
    utterances = moss_to_utterances(
        parse_moss_output(raw, drop_events=drop_events), merge_gap=merge_gap
    )
    return "".join(format_line(u.start, u.speaker, u.text) + "\n" for u in utterances)
