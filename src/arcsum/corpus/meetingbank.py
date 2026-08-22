"""MeetingBank word-level transcripts -> format v2 (SPEC §2.2 stage 1).

**Which release.** The authoritative source is the Zenodo release (record `7989108`,
`MeetingBank.zip`) — confirmed by direct inspection, not the paper. Each meeting's
transcript is `Audio&Transcripts/<city>/transcripts/<file>.transcript.json`:

    {"segments": [{"offset": int, "duration": int, "speaker": int,
                   "nbest": [{"text": str, "words": [...]}]}, ...]}

Segments are pre-sorted by `offset`; `speaker` is a small integer per meeting (not a
stable global id); `nbest` always has exactly one candidate in the corpus as shipped.
**The `huuuyeah/meetingbank` Hugging Face dataset is a stripped derivative** — a flat,
speakerless blob — and must not be used here; it satisfies none of this stage.

**Import is turn-merging, not word-splitting.** Each segment already carries a fully
formed utterance string (`nbest[0]["text"]`); the transform this module performs is
`master:src/voxsum/ingest_moss.py::moss_to_utterances`'s adapted for this shape — merge
CONSECUTIVE segments sharing the same speaker into one v2 line, relabel speakers
`S1…Sn` by first appearance (SPEC §2), and fall back to the reserved `UNK` label when a
segment carries no speaker at all. No gap gate: unlike the prior project's MOSS ingest
(`merge_gap=2.0`), word/segment-level ASR timing here gives no reason to split two
adjacent same-speaker segments into separate turns — SPEC §2.2 stage 1 says exactly
"group consecutive words by speaker into one line per turn," with no gap concept.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from arcsum.transcript import UNK, Utterance

#: MeetingBank uids contain spaces (e.g. "SeattleCityCouncil_03142016_CB 118618"), and
#: the id becomes a filename. A space there survives every Python path API and then
#: breaks the first unquoted shell expansion downstream — which is exactly how two
#: meetings were silently skipped mid-run in the prior project. Cuts harder now, at
#: 1,250 meetings instead of the 40 where it was originally found.
_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_id(uid: str) -> str:
    """Filename-safe meeting id. Never returns empty, even for a pathological uid."""
    cleaned = _UNSAFE_ID_CHARS.sub("_", str(uid)).strip("_")
    return cleaned or "meeting"


def extract_turns(segments: Sequence[dict]) -> list[tuple[object, str]]:
    """`(speaker_key, text)` pairs straight from the Zenodo segment schema, in the
    order the segments already carry (pre-sorted by `offset`).

    Segments with no non-empty candidate text are dropped — an ASR segment that
    transcribed to nothing carries no content to group into a turn.
    """
    turns: list[tuple[object, str]] = []
    for seg in segments:
        nbest = seg.get("nbest") or []
        if not nbest:
            continue
        text = (nbest[0].get("text") or "").strip()
        if not text:
            continue
        turns.append((seg.get("speaker"), text))
    return turns


def merge_consecutive_turns(turns: Iterable[tuple[object, str]]) -> list[Utterance]:
    """Merge consecutive same-speaker turns into one v2 `Utterance`; relabel speaker
    keys `S1…Sn` by first appearance (SPEC §2); a `None` key (no speaker at all) maps
    to the reserved literal `UNK`, never invented.

    Text from merged turns is joined with a single space — appropriate here because
    import happens BEFORE translation, on English words (SPEC §2.2 stage 1 precedes
    stage 2). A CJK-joining convention would be wrong for this stage.
    """
    label_map: dict[object, str] = {}

    def relabel(key: object) -> str:
        if key is None:
            return UNK
        if key not in label_map:
            label_map[key] = f"S{len(label_map) + 1}"
        return label_map[key]

    out: list[Utterance] = []
    for key, text in turns:
        speaker = relabel(key)
        if out and out[-1].speaker == speaker:
            out[-1] = Utterance(speaker, f"{out[-1].text} {text}")
        else:
            out.append(Utterance(speaker, text))
    return out


def import_meeting(transcript: dict) -> list[Utterance]:
    """Full pipeline: one Zenodo transcript JSON document -> v2 utterances."""
    return merge_consecutive_turns(extract_turns(transcript.get("segments") or []))


#: Zenodo segment `offset`/`duration` are .NET `TimeSpan` ticks (100 ns per tick) --
#: confirmed empirically: a transcript's top-level `duration` divided by this constant
#: equals `Metadata/MeetingBank.json`'s `VideoDuration` for the same meeting (both in
#: whole seconds), and per-word durations only make sense at this scale (~0.6 s/word,
#: not 0.6 s x 10^7). `itemInfo[*].startTime`/`endTime` are already in seconds, so
#: converting through this constant is what makes the two sources comparable at all.
TICKS_PER_SECOND = 10_000_000


def extract_turns_with_offsets(segments: Sequence[dict]) -> list[tuple[object, str, float, float]]:
    """Like `extract_turns`, but each turn also carries its `(start_sec, end_sec)` span.

    SPEC §2.2 stage 1: "discard all timing from the emitted line while retaining each
    line's source offset out-of-band, since §4.2 needs it to align items to chunks."
    This is that out-of-band value -- never emitted into the v2 wire format, and kept
    fully separate from `arcsum.transcript.Utterance`, which stays timestamp-free.
    """
    turns: list[tuple[object, str, float, float]] = []
    for seg in segments:
        nbest = seg.get("nbest") or []
        if not nbest:
            continue
        text = (nbest[0].get("text") or "").strip()
        if not text:
            continue
        start = (seg.get("offset") or 0) / TICKS_PER_SECOND
        end = start + (seg.get("duration") or 0) / TICKS_PER_SECOND
        turns.append((seg.get("speaker"), text, start, end))
    return turns


def merge_consecutive_turns_with_offsets(
    turns: Iterable[tuple[object, str, float, float]],
) -> list[tuple[Utterance, float, float]]:
    """Like `merge_consecutive_turns`, but returns `(Utterance, start_sec, end_sec)` --
    a merged line's span is the union of every segment folded into it (earliest start,
    latest end), since a merged turn's text is the concatenation of all of them.
    """
    label_map: dict[object, str] = {}

    def relabel(key: object) -> str:
        if key is None:
            return UNK
        if key not in label_map:
            label_map[key] = f"S{len(label_map) + 1}"
        return label_map[key]

    out: list[tuple[Utterance, float, float]] = []
    for key, text, start, end in turns:
        speaker = relabel(key)
        if out and out[-1][0].speaker == speaker:
            prev_u, prev_start, prev_end = out[-1]
            out[-1] = (
                Utterance(speaker, f"{prev_u.text} {text}"),
                prev_start,
                max(prev_end, end),
            )
        else:
            out.append((Utterance(speaker, text), start, end))
    return out


def import_meeting_with_offsets(transcript: dict) -> list[tuple[Utterance, float, float]]:
    """Full pipeline, offset-tracking variant: one Zenodo transcript JSON document ->
    `[(Utterance, start_sec, end_sec), ...]`, in the same order `import_meeting` would
    return its utterances alone. Used only for building §4.2's per-step supervision;
    the plain `import_meeting` remains what corpus import actually writes to disk."""
    return merge_consecutive_turns_with_offsets(
        extract_turns_with_offsets(transcript.get("segments") or [])
    )
