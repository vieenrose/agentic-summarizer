"""Pins `arcsum.cli.build_sft`: `arcsum-gen-traces` JSONL rows -> a train/valid split,
using only the pool-level operations `supervision.sft` already defines and tests.
"""

from __future__ import annotations

import json

import pytest

from arcsum.cli.build_sft import build_parser, load_samples, main
from arcsum.supervision.sft import SftSample


def _row(meeting: str, step: int, *, is_nop: bool, prompt_version: str = "sys-v1") -> dict:
    return {
        "meeting": meeting,
        "step": step,
        "prompt_version": prompt_version,
        "system": "SYS",
        "prompt": f"prompt {meeting}-{step}",
        "completion": "NOP" if is_nop else f"ADD - point {meeting}-{step}",
        "is_nop": is_nop,
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_samples_decodes_into_sft_sample(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    _write_jsonl(path, [_row("m1", 0, is_nop=False)])

    samples = load_samples([path])

    assert samples == [
        SftSample(
            meeting="m1",
            step=0,
            prompt_version="sys-v1",
            system="SYS",
            prompt="prompt m1-0",
            completion="ADD - point m1-0",
            is_nop=False,
        )
    ]


def test_load_samples_merges_multiple_files_and_skips_blank_lines(tmp_path) -> None:
    path1 = tmp_path / "a.jsonl"
    path2 = tmp_path / "b.jsonl"
    path1.write_text(json.dumps(_row("m1", 0, is_nop=False)) + "\n\n", encoding="utf-8")
    path2.write_text(json.dumps(_row("m2", 0, is_nop=False)), encoding="utf-8")

    samples = load_samples([path1, path2])

    assert {s.meeting for s in samples} == {"m1", "m2"}


# --- main() --------------------------------------------------------------------------------


def test_main_writes_train_and_valid_split_by_meeting(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [_row(f"m{i}", 0, is_nop=False) for i in range(10)]
    _write_jsonl(path, rows)
    out_dir = tmp_path / "sft"

    rc = main([str(path), "--out-dir", str(out_dir), "--valid-frac", "0.2", "--seed", "0"])

    assert rc == 0
    train = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(line) for line in (out_dir / "valid.jsonl").read_text().splitlines()]
    train_meetings = {r["meeting"] for r in train}
    valid_meetings = {r["meeting"] for r in valid}
    assert train_meetings.isdisjoint(valid_meetings)
    assert len(valid_meetings) == 2  # round(0.2 * 10)
    assert train_meetings | valid_meetings == {f"m{i}" for i in range(10)}


def test_main_downsamples_excess_nop(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    # 1 non-NOP, 9 NOP -- far over any reasonable max_nop_frac.
    rows = [_row("m0", 0, is_nop=False)] + [_row(f"m{i}", 0, is_nop=True) for i in range(1, 10)]
    _write_jsonl(path, rows)
    out_dir = tmp_path / "sft"

    main(
        [
            str(path),
            "--out-dir",
            str(out_dir),
            "--valid-frac",
            "0.0",
            "--max-nop-frac",
            "0.5",
            "--seed",
            "0",
        ]
    )

    train = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(line) for line in (out_dir / "valid.jsonl").read_text().splitlines()]
    combined = train + valid
    nop_count = sum(1 for r in combined if r["is_nop"])
    non_nop_count = len(combined) - nop_count
    assert nop_count / len(combined) <= 0.5
    assert non_nop_count == 1


def test_main_refuses_a_mixed_prompt_version_pool(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [
        _row("m1", 0, is_nop=False, prompt_version="sys-v1"),
        _row("m2", 0, is_nop=False, prompt_version="sys-v2"),
    ]
    _write_jsonl(path, rows)
    out_dir = tmp_path / "sft"

    rc = main([str(path), "--out-dir", str(out_dir)])

    assert rc == 1
    assert not (out_dir / "train.jsonl").exists()


def test_main_returns_nonzero_with_no_samples(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    out_dir = tmp_path / "sft"

    rc = main([str(path), "--out-dir", str(out_dir)])

    assert rc == 1


def test_main_is_deterministic_with_a_fixed_seed(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [_row(f"m{i}", 0, is_nop=(i % 2 == 0)) for i in range(20)]
    _write_jsonl(path, rows)
    out1 = tmp_path / "sft1"
    out2 = tmp_path / "sft2"

    main([str(path), "--out-dir", str(out1), "--seed", "3"])
    main([str(path), "--out-dir", str(out2), "--seed", "3"])

    assert (out1 / "train.jsonl").read_text() == (out2 / "train.jsonl").read_text()
    assert (out1 / "valid.jsonl").read_text() == (out2 / "valid.jsonl").read_text()


# --- --target-drop-frac ---------------------------------------------------------------


def _drop_row(meeting: str, step: int) -> dict:
    return {
        "meeting": meeting,
        "step": step,
        "prompt_version": "sys-v1",
        "system": "SYS",
        "prompt": f"prompt {meeting}-{step}",
        "completion": "DROP «x»\nADD - y",
        "is_nop": False,
    }


def test_target_drop_frac_default_is_a_noop(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [_drop_row(f"d{i}", 0) for i in range(5)] + [
        _row(f"m{i}", 0, is_nop=False) for i in range(45)
    ]
    _write_jsonl(path, rows)
    out_dir = tmp_path / "sft"

    rc = main([str(path), "--out-dir", str(out_dir), "--valid-frac", "0.0"])

    assert rc == 0
    total = (out_dir / "train.jsonl").read_text().count("\n") + 1
    assert total == 50


def test_target_drop_frac_duplicates_drop_bearing_rows(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [_drop_row(f"d{i}", 0) for i in range(5)] + [
        _row(f"m{i}", 0, is_nop=False) for i in range(45)
    ]
    _write_jsonl(path, rows)
    out_dir = tmp_path / "sft"

    rc = main(
        [str(path), "--out-dir", str(out_dir), "--valid-frac", "0.0", "--target-drop-frac", "0.4"]
    )

    assert rc == 0
    train = (out_dir / "train.jsonl").read_text().splitlines()
    total = len(train)
    n_drop = sum(1 for line in train if "DROP" in line)
    assert n_drop / total == pytest.approx(0.4, abs=0.02)


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_out_dir() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["traces.jsonl"])
