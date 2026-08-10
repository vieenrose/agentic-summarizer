"""Transcript v1 conformance — CLAUDE.md §2, §3.1, §7."""

from __future__ import annotations

import pytest

from voxsum.ingest_moss import moss_to_v1, parse_moss_output
from voxsum.transcript import (
    clock_to_sec,
    format_line,
    parse_line,
    parse_transcript,
    sec_to_clock,
)

# Padding edges named in the spec, plus the hour boundary.
CLOCKS = ["0:00", "0:07", "3:35", "9:59", "59:58", "1:00:00", "1:02:07", "10:00:00"]


@pytest.mark.parametrize("clock", CLOCKS)
def test_clock_round_trip(clock: str) -> None:
    assert sec_to_clock(clock_to_sec(clock)) == clock


def test_clock_values_are_not_mm_ss_inverted() -> None:
    # The known past bug: swapping minutes and seconds. 3:35 is 215s, never 3*1+35*60.
    assert clock_to_sec("3:35") == 215
    assert clock_to_sec("1:02:07") == 3727
    assert clock_to_sec("59:58") == 3598


@pytest.mark.parametrize("sec", range(0, 7300, 7))
def test_sec_round_trip(sec: int) -> None:
    assert clock_to_sec(sec_to_clock(sec)) == sec


def test_clock_accepts_bracketed_anchor() -> None:
    assert clock_to_sec("[14:30]") == 870


@pytest.mark.parametrize("bad", ["", "14:30:", "1:2:3", "0:60", "abc", "12", "1:0:00"])
def test_clock_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        clock_to_sec(bad)


def test_parse_line_diarized() -> None:
    assert parse_line("[2:30] S2: I propose we move to Building B.") == (
        "2:30",
        "S2",
        "I propose we move to Building B.",
    )


def test_parse_line_named_speaker() -> None:
    assert parse_line("[1:02:07] Chair Lin: Motion carries.") == (
        "1:02:07",
        "Chair Lin",
        "Motion carries.",
    )


def test_parse_line_undiarized_with_colon_in_text() -> None:
    # A ': ' beyond the 40-char speaker limit belongs to the text, not a speaker field.
    line = "[0:05] " + "we agreed on the following points and then said: yes"
    ts, speaker, text = parse_line(line)
    assert (ts, speaker) == ("0:05", None)
    assert text.endswith("said: yes")


def test_parse_line_text_may_contain_brackets() -> None:
    # Split on the FIRST '] ' only — later brackets are content.
    ts, speaker, text = parse_line("[5:10] S1: see [appendix A] for details")
    assert (ts, speaker, text) == ("5:10", "S1", "see [appendix A] for details")


def test_parse_line_zh() -> None:
    assert parse_line("[2:30] S2: 我建議搬到 B 棟大樓。") == ("2:30", "S2", "我建議搬到 B 棟大樓。")


@pytest.mark.parametrize("bad", ["0:00 S1: hi", "[0:00]S1: hi", "[x:yy] S1: hi"])
def test_parse_line_rejects_non_v1(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_line(bad)


def test_parse_transcript_example_en() -> None:
    text = (
        "[0:00] S1: Let us discuss the office move.\n"
        "[2:30] S2: I propose we move to Building B.\n"
        "[5:10] S1: Agreed, Building B it is.\n"
    )
    utterances = parse_transcript(text)
    assert [u.start for u in utterances] == [0, 150, 310]
    assert [u.speaker for u in utterances] == ["S1", "S2", "S1"]
    # Rendering is lossless.
    assert "".join(u.render() + "\n" for u in utterances) == text


def test_parse_transcript_reports_line_number() -> None:
    with pytest.raises(ValueError, match="line 2"):
        parse_transcript("[0:00] S1: ok\nbroken line\n")


def test_long_line_is_not_truncated() -> None:
    # VCSum zh lines run to ~2.6k chars; readers must not assume a max length.
    body = "很" * 2600
    _, _, text = parse_line(format_line(0, "S1", body))
    assert text == body


# --- MOSS ingestion -------------------------------------------------------------

MOSS_RAW = (
    "[0.48][S01]Welcome everyone[1.66]"
    "[12.26][S02]The new transcription pipeline is ready for evaluation[13.81]"
    "[14.36][S01]Great, include the diarization results in the report[18.76]"
)


def test_parse_moss_output() -> None:
    segments = parse_moss_output(MOSS_RAW)
    assert [s.speaker for s in segments] == ["S01", "S02", "S01"]
    assert segments[0].start == 0.48 and segments[0].end == 1.66
    assert segments[2].text == "Great, include the diarization results in the report"


def test_moss_to_v1_renumbers_and_floors() -> None:
    v1 = moss_to_v1(MOSS_RAW, merge_gap=None)
    assert v1 == (
        "[0:00] S1: Welcome everyone\n"
        "[0:12] S2: The new transcription pipeline is ready for evaluation\n"
        "[0:14] S1: Great, include the diarization results in the report\n"
    )
    parse_transcript(v1)  # output is valid v1


def test_moss_speaker_labels_renumbered_by_first_appearance() -> None:
    # MOSS labels are relative and need not start at 1 or be dense.
    raw = "[0.0][S07]first[1.0][2.0][S03]second[3.0][4.0][S07]third[5.0]"
    lines = moss_to_v1(raw, merge_gap=None).splitlines()
    assert [parse_line(line)[1] for line in lines] == [
        "S1",
        "S2",
        "S1",
    ]


def test_moss_merges_adjacent_same_speaker_segments() -> None:
    raw = "[0.0][S01]first part[1.0][1.5][S01]second part[2.5][30.0][S01]much later[31.0]"
    lines = moss_to_v1(raw, merge_gap=2.0).splitlines()
    assert lines == ["[0:00] S1: first part second part", "[0:30] S1: much later"]


def test_moss_collapses_embedded_newlines() -> None:
    # One utterance = one line is a hard rule.
    raw = "[0.0][S01]line one\nline two[1.0]"
    assert moss_to_v1(raw) == "[0:00] S1: line one line two\n"


def test_moss_undiarized_segments_have_no_speaker_field() -> None:
    assert moss_to_v1("[0.0]just audio[1.0]") == "[0:00] just audio\n"


def test_moss_drop_events() -> None:
    raw = "[0.0][S01](laughter) we can begin[2.0]"
    assert moss_to_v1(raw, drop_events=True) == "[0:00] S1: we can begin\n"
    assert "(laughter)" in moss_to_v1(raw)
