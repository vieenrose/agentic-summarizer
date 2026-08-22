"""Pins the MeetingBank importer (SPEC §2.2 stage 1) and the corpus provenance/split
discipline (SPEC §2.2, §4.2).

Guards a mandatory-not-optional fix ported verbatim from the prior project: MeetingBank
uids contain spaces, which survive every Python path API and then break the first
unquoted shell expansion downstream — how two meetings were silently skipped mid-run.
"""

from __future__ import annotations

import pytest

from arcsum.corpus.manifest import (
    DEFAULT_EVAL_N,
    Manifest,
    MeetingRecord,
    carve_splits,
)
from arcsum.corpus.meetingbank import (
    TICKS_PER_SECOND,
    extract_turns,
    extract_turns_with_offsets,
    import_meeting,
    import_meeting_with_offsets,
    merge_consecutive_turns,
    merge_consecutive_turns_with_offsets,
    safe_id,
)
from arcsum.transcript import UNK

# --- safe_id --------------------------------------------------------------------------


def test_safe_id_sanitises_spaces() -> None:
    assert (
        safe_id("SeattleCityCouncil_03142016_CB 118618") == "SeattleCityCouncil_03142016_CB_118618"
    )


def test_safe_id_sanitises_arbitrary_punctuation() -> None:
    assert safe_id("a/b:c*d") == "a_b_c_d"


def test_safe_id_never_returns_empty() -> None:
    assert safe_id("") == "meeting"
    assert safe_id("***") == "meeting"


def test_safe_id_preserves_already_safe_ids() -> None:
    assert safe_id("DenverCityCouncil_05012017") == "DenverCityCouncil_05012017"


def test_safe_id_strips_leading_and_trailing_underscores_from_sanitisation() -> None:
    assert safe_id(" leading and trailing ") == "leading_and_trailing"


# --- extract_turns / merge_consecutive_turns / import_meeting -----------------------


def seg(speaker: object, text: str, *, offset: int = 0, duration: int = 0) -> dict:
    return {
        "offset": offset,
        "duration": duration,
        "speaker": speaker,
        "nbest": [{"text": text}],
    }


def test_extract_turns_pulls_speaker_and_text() -> None:
    segments = [seg(0, "hello"), seg(1, "world")]
    assert extract_turns(segments) == [(0, "hello"), (1, "world")]


def test_extract_turns_drops_empty_candidates() -> None:
    segments = [seg(0, "hello"), seg(0, "   "), seg(1, "world")]
    assert extract_turns(segments) == [(0, "hello"), (1, "world")]


def test_extract_turns_handles_missing_nbest() -> None:
    segments = [{"offset": 0, "duration": 0, "speaker": 0, "nbest": []}]
    assert extract_turns(segments) == []


def test_merge_consecutive_turns_relabels_by_first_appearance() -> None:
    turns = [(5, "a"), (2, "b"), (5, "c")]
    utts = merge_consecutive_turns(turns)
    # 5 seen first -> S1; 2 seen second -> S2; 5 recurs as S1 again (not merged, since
    # the intervening S2 turn breaks consecutiveness).
    assert [u.speaker for u in utts] == ["S1", "S2", "S1"]


def test_merge_consecutive_turns_merges_same_speaker_runs() -> None:
    turns = [(0, "first"), (0, "second"), (1, "third")]
    utts = merge_consecutive_turns(turns)
    assert len(utts) == 2
    assert utts[0].speaker == "S1"
    assert utts[0].text == "first second"
    assert utts[1].speaker == "S2"
    assert utts[1].text == "third"


def test_merge_consecutive_turns_maps_none_speaker_to_unk() -> None:
    turns = [(None, "unlabeled speech")]
    utts = merge_consecutive_turns(turns)
    assert utts[0].speaker == UNK


def test_merge_consecutive_turns_merges_consecutive_unk_segments() -> None:
    turns = [(None, "a"), (None, "b"), (0, "c")]
    utts = merge_consecutive_turns(turns)
    assert len(utts) == 2
    assert utts[0].speaker == UNK
    assert utts[0].text == "a b"


def test_merge_consecutive_turns_empty_input_yields_no_utterances() -> None:
    assert merge_consecutive_turns([]) == []


def test_import_meeting_full_pipeline() -> None:
    transcript = {
        "segments": [
            seg(0, "Please come to order."),
            seg(0, "We will now begin."),
            seg(1, "Thank you, Mr. Chair."),
        ]
    }
    utts = import_meeting(transcript)
    assert len(utts) == 2
    assert utts[0].speaker == "S1"
    assert utts[0].text == "Please come to order. We will now begin."
    assert utts[1].speaker == "S2"
    assert utts[1].text == "Thank you, Mr. Chair."


def test_import_meeting_handles_missing_segments_key() -> None:
    assert import_meeting({}) == []


def test_imported_utterances_render_as_valid_v2_lines() -> None:
    """The whole point of the import: output must satisfy format v2 (SPEC §2)."""
    from arcsum.transcript import parse_line

    transcript = {"segments": [seg(0, "hello there"), seg(1, "general kenobi")]}
    utts = import_meeting(transcript)
    for u in utts:
        rendered = u.render()
        assert "\n" not in rendered
        speaker, text = parse_line(rendered)
        assert speaker == u.speaker
        assert text == u.text


# --- MeetingRecord / manifest ----------------------------------------------------------


def test_meeting_record_manifest_shape() -> None:
    r = MeetingRecord("m1", split="train", translated_by="translategemma-27b")
    m = r.manifest()
    assert m == {
        "meeting_id": "m1",
        "split": "train",
        "translated_by": "translategemma-27b",
        "composed_by": None,
        "human_validated": False,
    }


# --- offset tracking (SPEC §2.2 stage 1's out-of-band offset, for §4.2 alignment) ------


def test_ticks_per_second_matches_the_dotnet_timespan_convention() -> None:
    """Confirmed empirically against a real meeting: transcript top-level `duration`
    (ticks) / TICKS_PER_SECOND == Metadata's `VideoDuration` (whole seconds)."""
    assert TICKS_PER_SECOND == 10_000_000


def test_extract_turns_with_offsets_converts_ticks_to_seconds() -> None:
    segments = [seg(0, "hello", offset=10_000_000, duration=5_000_000)]
    turns = extract_turns_with_offsets(segments)
    assert turns == [(0, "hello", 1.0, 1.5)]


def test_extract_turns_with_offsets_drops_empty_candidates_like_extract_turns() -> None:
    segments = [seg(0, "hello", offset=0, duration=0), seg(0, "   ", offset=10, duration=1)]
    turns = extract_turns_with_offsets(segments)
    assert len(turns) == 1


def test_merge_consecutive_turns_with_offsets_spans_the_whole_merged_run() -> None:
    turns = [
        (0, "first", 0.0, 1.0),
        (0, "second", 1.0, 2.5),
        (1, "third", 2.5, 3.0),
    ]
    merged = merge_consecutive_turns_with_offsets(turns)
    assert len(merged) == 2
    (u0, start0, end0) = merged[0]
    assert u0.text == "first second"
    assert (start0, end0) == (0.0, 2.5)
    (u1, start1, end1) = merged[1]
    assert u1.text == "third"
    assert (start1, end1) == (2.5, 3.0)


def test_merge_consecutive_turns_with_offsets_takes_the_max_end_not_the_last() -> None:
    """A merged run's end must be the LATEST end among its segments, not simply the
    last one processed -- ASR segments are not guaranteed non-overlapping."""
    turns = [(0, "a", 0.0, 5.0), (0, "b", 1.0, 2.0)]
    merged = merge_consecutive_turns_with_offsets(turns)
    (_u0, start0, end0) = merged[0]
    assert (start0, end0) == (0.0, 5.0)


def test_import_meeting_with_offsets_produces_the_same_utterances_as_import_meeting() -> None:
    """The offset-tracking variant must never diverge from the plain importer's
    utterance sequence -- offsets are additive metadata, not a different pipeline."""
    transcript = {
        "segments": [
            seg(0, "Please come to order.", offset=0, duration=20_000_000),
            seg(0, "We will now begin.", offset=20_000_000, duration=10_000_000),
            seg(1, "Thank you, Mr. Chair.", offset=30_000_000, duration=15_000_000),
        ]
    }
    plain = import_meeting(transcript)
    with_offsets = import_meeting_with_offsets(transcript)
    assert plain == [u for u, _start, _end in with_offsets]


def test_import_meeting_with_offsets_handles_missing_segments_key() -> None:
    assert import_meeting_with_offsets({}) == []


def test_meeting_record_ready_for_corpus_requires_all_three_stages() -> None:
    incomplete = MeetingRecord("m1", translated_by="x")
    assert incomplete.ready_for_corpus() is False

    complete = MeetingRecord("m1", translated_by="x", composed_by="y", human_validated=True)
    assert complete.ready_for_corpus() is True


def test_meeting_record_not_ready_without_human_validation() -> None:
    """SPEC §4: human validation is non-optional -- translated+composed alone is not enough."""
    r = MeetingRecord("m1", translated_by="x", composed_by="y", human_validated=False)
    assert r.ready_for_corpus() is False


# --- carve_splits -----------------------------------------------------------------------


def test_carve_splits_produces_exactly_eval_n_eval_meetings() -> None:
    ids = [f"m{i}" for i in range(100)]
    splits = carve_splits(ids, eval_n=20)
    assert sum(1 for v in splits.values() if v == "eval") == 20
    assert sum(1 for v in splits.values() if v == "train") == 80


def test_carve_splits_covers_every_id_exactly_once() -> None:
    ids = [f"m{i}" for i in range(50)]
    splits = carve_splits(ids, eval_n=10)
    assert set(splits.keys()) == set(ids)


def test_carve_splits_is_deterministic_across_calls() -> None:
    ids = [f"m{i}" for i in range(50)]
    a = carve_splits(ids, eval_n=10, seed=42)
    b = carve_splits(ids, eval_n=10, seed=42)
    assert a == b


def test_carve_splits_is_independent_of_input_order() -> None:
    ids = [f"m{i}" for i in range(50)]
    a = carve_splits(ids, eval_n=10, seed=42)
    b = carve_splits(list(reversed(ids)), eval_n=10, seed=42)
    assert a == b


def test_carve_splits_different_seeds_can_differ() -> None:
    ids = [f"m{i}" for i in range(50)]
    a = carve_splits(ids, eval_n=10, seed=1)
    b = carve_splits(ids, eval_n=10, seed=2)
    assert a != b


def test_carve_splits_rejects_eval_n_larger_than_the_corpus() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        carve_splits(["m0", "m1"], eval_n=5)


def test_default_eval_n_matches_spec_phase_1() -> None:
    assert DEFAULT_EVAL_N == 40


# --- Manifest ----------------------------------------------------------------------------


def test_manifest_set_split_creates_records_for_new_ids() -> None:
    m = Manifest()
    m.set_split({"a": "train", "b": "eval"})
    assert m.train_ids() == {"a"}
    assert m.eval_ids() == {"b"}


def test_manifest_set_split_preserves_existing_provenance_fields() -> None:
    m = Manifest()
    m.records["a"] = MeetingRecord("a", split="train", translated_by="x")
    m.set_split({"a": "eval"})
    assert m.records["a"].split == "eval"
    assert m.records["a"].translated_by == "x"


def test_manifest_to_list_is_sorted_by_meeting_id() -> None:
    m = Manifest()
    m.set_split({"z": "train", "a": "eval", "m": "train"})
    ids = [r["meeting_id"] for r in m.to_list()]
    assert ids == sorted(ids)


def test_eval_meetings_are_held_out_of_train() -> None:
    """The exact leak the prior project hit: eval and train ids must never overlap."""
    ids = [f"m{i}" for i in range(200)]
    splits = carve_splits(ids, eval_n=40)
    m = Manifest()
    m.set_split(splits)
    assert m.eval_ids() & m.train_ids() == set()
    assert len(m.eval_ids()) == 40
