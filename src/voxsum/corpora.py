"""Public corpora -> transcript v1 (PLAN.md §1a).

**Read this before using any output of this module for FAITH-anchor.**

Neither available corpus carries timestamps:

* **QMSum** (`pszemraj/qmsum-cleaned`) — AMI/ICSI/committee meetings with real speaker
  labels (`Professor E:`), ~31k tokens each, no clock.
* **MeetingBank** (`huuuyeah/meetingbank`) — council proceedings, ~2k-char agenda-item
  segments, **no speaker labels and no clock** (the spec's own §7.8 caveat).

v1 requires a timestamp per line, so we synthesise one from a fixed speaking rate. The
result is **monotonic and internally consistent, but not true**: an anchor points at the
line that states a claim, and that relationship holds, yet the wall-clock value is
invented. Consequences, which the manifest records per meeting so they cannot be forgotten:

* fine for **training** — the student learns "copy the timestamp of the supporting line",
  which is exactly the skill, and the clock's truth is irrelevant to it;
* **not** fine for reporting FAITH-anchor as a measure of real-world anchor accuracy.
  On a synthetic clock it measures self-consistency only. Real anchors require audio
  through MOSS (`tools/transcribe_moss.py`), which is why §1a puts audio first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .transcript import Utterance

__all__ = [
    "SPEAKING_RATE_WPM",
    "MeetingRecord",
    "parse_meetingbank_transcript",
    "parse_qmsum_input",
    "synthesise_clock",
]

# Words per minute used to synthesise the clock. 150 wpm is ordinary meeting speech.
SPEAKING_RATE_WPM = 150

# AMI/ICSI annotation markers in QMSum text. They are transcription metadata, not speech.
_MARKERS = re.compile(r"\{(?:disfmarker|vocalsound|nonvocalsound|gap|pause|comment)\}")
_WS = re.compile(r"\s+")

# "Speaker Name: text" — a speaker field never contains ": " and is <= 40 chars (§2).
_TURN = re.compile(r"^(?P<speaker>[^:]{1,40}):\s*(?P<text>.+)$")

# Sentence-ish split for corpora that ship one undifferentiated block of prose.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True, slots=True)
class MeetingRecord:
    """One converted meeting plus the provenance a reported number needs."""

    meeting_id: str
    source: str
    lang: str
    utterances: tuple[Utterance, ...]
    #: False whenever the clock was synthesised — see the module docstring.
    authentic_clock: bool = False
    #: False when speaker fields were absent in the source (MeetingBank).
    authentic_speakers: bool = False
    split: str = "train"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_lines(self) -> int:
        return len(self.utterances)

    @property
    def duration(self) -> int:
        return self.utterances[-1].start if self.utterances else 0

    def render(self) -> str:
        return "".join(u.render() + "\n" for u in self.utterances)

    def manifest(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "source": self.source,
            "lang": self.lang,
            "split": self.split,
            "n_lines": self.n_lines,
            "duration_sec": self.duration,
            "authentic_clock": self.authentic_clock,
            "authentic_speakers": self.authentic_speakers,
            "notes": list(self.notes),
        }


def _clean(text: str) -> str:
    return _WS.sub(" ", _MARKERS.sub(" ", text)).strip()


def synthesise_clock(
    turns: list[tuple[str | None, str]], *, wpm: int = SPEAKING_RATE_WPM
) -> list[Utterance]:
    """Assign monotonically increasing starts from each turn's own length.

    Distinct starts are guaranteed: two lines sharing a timestamp would make `«prefix»`
    anchoring ambiguous, and `Chunk.has_line` could not tell them apart.
    """
    per_word = 60.0 / max(wpm, 1)
    out: list[Utterance] = []
    cursor = 0.0
    for speaker, text in turns:
        start = int(cursor)
        if out and start <= out[-1].start:
            start = out[-1].start + 1
        out.append(Utterance(start, speaker, text))
        cursor = max(cursor + len(text.split()) * per_word, float(start) + 1.0)
    return out


def parse_qmsum_input(raw: str, *, drop_first_line_query: bool = True) -> list[tuple[str, str]]:
    """QMSum `input` -> [(speaker, text)].

    The first line is the retrieval *query*, not speech — it must be dropped or it becomes
    a transcript line the model can be asked to anchor to.
    """
    lines = raw.splitlines()
    if drop_first_line_query and lines and not _TURN.match(lines[0].strip()):
        lines = lines[1:]

    turns: list[tuple[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = _TURN.match(line)
        if not m:
            # Continuation of the previous turn rather than a new speaker.
            if turns:
                speaker, text = turns[-1]
                turns[-1] = (speaker, _clean(f"{text} {line}"))
            continue
        text = _clean(m.group("text"))
        if text:
            turns.append((_clean(m.group("speaker")), text))
    return turns


def parse_meetingbank_transcript(raw: str, *, max_chars: int = 600) -> list[str]:
    """MeetingBank `transcript` -> sentence-ish lines, no speakers available.

    Splitting matters: one utterance per line is a hard rule, and a single 2k-char blob
    would give the whole segment one anchor, making every bullet point at the same line.
    """
    text = _clean(raw)
    if not text:
        return []
    lines: list[str] = []
    for sentence in _SENTENCE.split(text):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            lines.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            lines.append(sentence)
    return lines


def qmsum_record(meeting_id: str, raw: str, *, split: str = "train") -> MeetingRecord:
    turns = parse_qmsum_input(raw)
    return MeetingRecord(
        meeting_id=meeting_id,
        source="qmsum",
        lang="en",
        utterances=tuple(synthesise_clock([(s, t) for s, t in turns])),
        authentic_clock=False,
        authentic_speakers=True,
        split=split,
        notes=("clock synthesised at 150 wpm; not real time",),
    )


def meetingbank_record(meeting_id: str, raw: str, *, split: str = "train") -> MeetingRecord:
    lines = parse_meetingbank_transcript(raw)
    return MeetingRecord(
        meeting_id=meeting_id,
        source="meetingbank",
        lang="en",
        utterances=tuple(synthesise_clock([(None, t) for t in lines])),
        authentic_clock=False,
        authentic_speakers=False,
        split=split,
        notes=(
            "clock synthesised at 150 wpm; not real time",
            "no speaker labels in source: ACTIONS attribution is unavailable",
        ),
    )
