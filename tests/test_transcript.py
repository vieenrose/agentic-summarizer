"""Pins SPEC §2 (transcript format v2) and §4.3's line-count integrity gate.

Guards two decisions the prior project gives no precedent for, because v2 deletes the
branch it relied on (`speaker=None` for undiarized lines):
  * `parse_line` is TOTAL — a non-conforming line yields `(UNK, whole line)`, never a raise;
  * conformance is a SEPARATE function, so a corpus importer can fail loudly while the
    harness keeps parsing.
"""

from __future__ import annotations

import pytest

from arcsum.transcript import (
    MAX_SPEAKER_LEN,
    UNK,
    LineDefect,
    Utterance,
    format_line,
    line_count_matches,
    parse_line,
    parse_transcript,
    validate_v2,
)

SPEC_EXAMPLE = "S1: 我們來討論辦公室搬遷。\nS2: 我建議搬到 B 棟大樓。\nS1: 好，就搬到 B 棟。"


def test_parse_line_splits_on_the_first_colon_space() -> None:
    assert parse_line("S1: 我們來討論辦公室搬遷。") == ("S1", "我們來討論辦公室搬遷。")


def test_parse_line_splits_on_the_first_separator_not_a_later_one() -> None:
    """A `: ` inside the text must not re-split the line."""
    assert parse_line("S1: 本案的重點: 我們搬到 B 棟") == ("S1", "本案的重點: 我們搬到 B 棟")


@pytest.mark.parametrize(
    "line",
    [
        "",
        " ",
        "no separator at all",
        ":",
        ": leading separator",
        "S1:no space after colon",
        "本案的重點: 我們搬到 B 棟",  # first ': ' is inside a plausible speaker length
        "x" * (MAX_SPEAKER_LEN + 1) + ": text",
        "x" * 3000 + ": text",
        "S1: " + "很" * 2600,
        "\x00: text",
        "UNK: ",
    ],
)
def test_parse_line_never_raises(line: str) -> None:
    """SPEC §2 justifies the mandatory speaker field as keeping `parse_line` total."""
    speaker, text = parse_line(line)
    assert isinstance(speaker, str)
    assert isinstance(text, str)


def test_overlong_speaker_falls_back_to_unk_and_validate_reports_it() -> None:
    """The two halves of the decision, asserted together so they cannot drift apart."""
    line = "x" * (MAX_SPEAKER_LEN + 1) + ": text"
    assert parse_line(line) == (UNK, line)

    defects = validate_v2(line)
    assert len(defects) == 1
    assert "longer than" in defects[0].reason


def test_line_without_a_separator_falls_back_to_unk_with_the_whole_line_as_text() -> None:
    assert parse_line("just some prose") == (UNK, "just some prose")


def test_a_conforming_unk_line_parses_as_a_speaker() -> None:
    """`UNK` is a real reserved label (SPEC §2), not only a fallback."""
    assert parse_line("UNK: 我們搬到 B 棟") == ("UNK", "我們搬到 B 棟")
    assert validate_v2("UNK: 我們搬到 B 棟") == []


def test_render_round_trips_and_contains_no_newline() -> None:
    """One utterance = one line is a hard rule (SPEC §2)."""
    u = Utterance("S1", "我們搬到 B 棟")
    assert u.render() == "S1: 我們搬到 B 棟"
    assert "\n" not in u.render()
    assert parse_line(u.render()) == ("S1", "我們搬到 B 棟")


def test_format_line_emits_text_as_is() -> None:
    """v2 has no escaping — text is emitted verbatim (SPEC §2)."""
    assert format_line("S1", "a: b «c» [1:23]") == "S1: a: b «c» [1:23]"


def test_long_monologue_line_survives_parse_and_render() -> None:
    """SPEC §2: "readers must not assume a max line length"."""
    text = "很" * 2600
    u = Utterance("S1", text)
    assert parse_line(u.render()) == ("S1", text)


def test_parse_transcript_skips_blank_lines() -> None:
    parsed = parse_transcript("S1: a\n\n   \nS2: b\n")
    assert [(u.speaker, u.text) for u in parsed] == [("S1", "a"), ("S2", "b")]


def test_parse_transcript_matches_the_spec_example() -> None:
    parsed = parse_transcript(SPEC_EXAMPLE)
    assert [u.speaker for u in parsed] == ["S1", "S2", "S1"]
    assert parsed[1].text == "我建議搬到 B 棟大樓。"


def test_validate_v2_accepts_the_spec_example() -> None:
    assert validate_v2(SPEC_EXAMPLE) == []


@pytest.mark.parametrize(
    ("line", "reason_fragment"),
    [
        ("no separator at all", "no ': '"),
        (": text", "empty speaker"),
        ("S1: ", "empty text"),
        ("x" * 41 + ": text", "longer than"),
        ("S1: has a \x00 null", "control character"),
    ],
)
def test_validate_v2_reports_each_defect_kind(line: str, reason_fragment: str) -> None:
    defects = validate_v2(line)
    assert defects, f"expected a defect for {line[:30]!r}"
    assert reason_fragment in defects[0].reason


def test_validate_v2_reports_line_numbers_one_indexed() -> None:
    defects = validate_v2("S1: ok\nbroken line\nS2: ok")
    assert [d.lineno for d in defects] == [2]
    assert isinstance(defects[0], LineDefect)


def test_line_count_mismatch_is_detected() -> None:
    """SPEC §4.3: line-count integrity is a HARD pass/fail for the translation gate."""
    src = "S1: a\nS2: b\nS3: c"
    assert line_count_matches(src, "S1: x\nS2: y\nS3: z")
    # A document-level translation that merges two utterances into one:
    assert not line_count_matches(src, "S1: x y\nS3: z")
    # ...or splits one into two:
    assert not line_count_matches(src, "S1: x\nS1: x2\nS2: y\nS3: z")


def test_line_count_ignores_blank_lines_on_both_sides() -> None:
    """Blank lines carry no utterance, so they must not fail an otherwise-sound pair."""
    assert line_count_matches("S1: a\nS2: b", "S1: x\n\nS2: y\n\n")


def test_utterance_is_frozen() -> None:
    """Immutability is what lets chunks share utterances without defensive copying."""
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError is a dataclass detail
        Utterance("S1", "a").speaker = "S2"  # type: ignore[misc]
