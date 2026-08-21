"""Pins `arcsum.cli.import_corpus` (SPEC §2.2 stage 1): Zenodo transcript JSON -> v2
`.txt` files + a provenance manifest, using the exact record-7989108 shape
(`segments[].nbest[0].text`, `segments[].speaker`) documented in CLAUDE.md.
"""

from __future__ import annotations

import json

import pytest

from arcsum.cli.import_corpus import build_parser, import_directory, import_transcript_file, main


def _transcript(*turns: tuple[object, str]) -> dict:
    return {
        "segments": [{"speaker": speaker, "nbest": [{"text": text}]} for speaker, text in turns]
    }


def test_import_transcript_file_merges_consecutive_turns_and_relabels_speakers(tmp_path) -> None:
    path = tmp_path / "SeattleCityCouncil_03142016_CB 118618.transcript.json"
    path.write_text(
        json.dumps(_transcript((1, "Good morning."), (1, "Let's begin."), (2, "Thank you.")))
    )

    meeting_id, v2_text = import_transcript_file(path)

    assert meeting_id == "SeattleCityCouncil_03142016_CB_118618"
    assert v2_text == "S1: Good morning. Let's begin.\nS2: Thank you."


def test_import_transcript_file_sanitises_unsafe_filename_characters(tmp_path) -> None:
    path = tmp_path / "SeattleCityCouncil_03142016_CB 118618.transcript.json"
    path.write_text(json.dumps(_transcript((1, "hello"))))

    meeting_id, _ = import_transcript_file(path)

    assert " " not in meeting_id
    assert meeting_id != ""


def test_import_transcript_file_strips_the_full_transcript_json_suffix(tmp_path) -> None:
    """Real Zenodo filenames are compound (`<slug>.mp3.transcript.json`); `Path.stem`
    only strips the LAST suffix, which would otherwise leave a stray `.transcript` (or
    `.mp3.transcript`) baked into the meeting id -- breaking any later lookup against
    Metadata/MeetingBank.json's clean meeting-id keys. Caught by running this CLI
    against the real Zenodo release (SPEC §9 Phase 1 pilot staging)."""
    path = tmp_path / "longbeach_a6470ca4-93aa-4ae0-a9ae-32003669a8af.mp3.transcript.json"
    path.write_text(json.dumps(_transcript((1, "hello"))))

    meeting_id, _ = import_transcript_file(path)

    assert meeting_id == "longbeach_a6470ca4-93aa-4ae0-a9ae-32003669a8af.mp3"
    assert not meeting_id.endswith(".transcript")


def test_import_transcript_file_strips_suffix_for_a_meeting_id_style_filename(tmp_path) -> None:
    """The staging convention this project actually uses for the pilot corpus:
    `<meeting_id>.transcript.json`, where meeting_id itself contains no further dots."""
    path = tmp_path / "AlamedaCC_01072020.transcript.json"
    path.write_text(json.dumps(_transcript((1, "hello"))))

    meeting_id, _ = import_transcript_file(path)

    assert meeting_id == "AlamedaCC_01072020"


def test_import_transcript_file_falls_back_to_unk_with_no_speaker(tmp_path) -> None:
    path = tmp_path / "m1.transcript.json"
    path.write_text(json.dumps(_transcript((None, "no speaker here"))))

    _, v2_text = import_transcript_file(path)

    assert v2_text == "UNK: no speaker here"


def test_import_directory_writes_one_txt_per_meeting_and_carves_splits(tmp_path) -> None:
    src = tmp_path / "transcripts"
    src.mkdir()
    for i in range(5):
        (src / f"m{i}.transcript.json").write_text(json.dumps(_transcript((1, f"turn {i}"))))
    out = tmp_path / "corpus"

    manifest = import_directory(src, out, eval_n=2, seed=0)

    txt_files = sorted(out.glob("*.txt"))
    assert len(txt_files) == 5
    assert len(manifest.records) == 5
    assert len(manifest.eval_ids()) == 2
    assert len(manifest.train_ids()) == 3


def test_import_directory_records_are_unvalidated_by_default(tmp_path) -> None:
    src = tmp_path / "transcripts"
    src.mkdir()
    (src / "m0.transcript.json").write_text(json.dumps(_transcript((1, "hi"))))
    out = tmp_path / "corpus"

    manifest = import_directory(src, out, eval_n=1, seed=0)

    (record,) = manifest.records.values()
    assert record.translated_by is None
    assert record.composed_by is None
    assert record.human_validated is False
    assert record.ready_for_corpus() is False


def test_import_directory_ignores_non_transcript_files(tmp_path) -> None:
    src = tmp_path / "transcripts"
    src.mkdir()
    (src / "m0.transcript.json").write_text(json.dumps(_transcript((1, "hi"))))
    (src / "README.md").write_text("not a transcript")
    out = tmp_path / "corpus"

    manifest = import_directory(src, out, eval_n=1, seed=0)

    assert len(manifest.records) == 1


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_out() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcripts/"])


def test_main_writes_manifest_and_txt_files(tmp_path) -> None:
    src = tmp_path / "transcripts"
    src.mkdir()
    for i in range(3):
        (src / f"m{i}.transcript.json").write_text(json.dumps(_transcript((1, f"turn {i}"))))
    out = tmp_path / "corpus"

    rc = main([str(src), "--out", str(out), "--eval-n", "1"])

    assert rc == 0
    manifest_path = out / "manifest.json"
    assert manifest_path.exists()
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(records) == 3
    assert sum(1 for r in records if r["split"] == "eval") == 1
    assert len(list(out.glob("*.txt"))) == 3


def test_main_accepts_custom_manifest_path(tmp_path) -> None:
    src = tmp_path / "transcripts"
    src.mkdir()
    (src / "m0.transcript.json").write_text(json.dumps(_transcript((1, "hi"))))
    out = tmp_path / "corpus"
    manifest_path = tmp_path / "elsewhere" / "prov.json"

    rc = main([str(src), "--out", str(out), "--manifest", str(manifest_path), "--eval-n", "1"])

    assert rc == 0
    assert manifest_path.exists()
    assert not (out / "manifest.json").exists()


def test_main_is_deterministic_across_runs_with_the_same_seed(tmp_path) -> None:
    src = tmp_path / "transcripts"
    src.mkdir()
    for i in range(5):
        (src / f"m{i}.transcript.json").write_text(json.dumps(_transcript((1, f"turn {i}"))))
    out1 = tmp_path / "corpus1"
    out2 = tmp_path / "corpus2"

    main([str(src), "--out", str(out1), "--eval-n", "2", "--seed", "7"])
    main([str(src), "--out", str(out2), "--eval-n", "2", "--seed", "7"])

    assert json.loads((out1 / "manifest.json").read_text()) == json.loads(
        (out2 / "manifest.json").read_text()
    )
