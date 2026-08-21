"""Pins `arcsum.cli.device_bench`: parsing Phase 0b's checked-in artifact formats
(`llama-bench -o jsonl`, the hand-written RSS text file) and assembling the SPEC §7 G4
gate -- never re-deriving the wall-clock-per-meeting figure itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcsum.cli.device_bench import build_parser, main, parse_rss, parse_throughput, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_THROUGHPUT = REPO_ROOT / "runs" / "phase0b-2026-08-21" / "throughput.jsonl"
REAL_RSS = REPO_ROOT / "runs" / "phase0b-2026-08-21" / "rss.txt"


def _record(**overrides) -> dict:
    base = {
        "model_type": "llama ?B Q8_0",
        "cpu_mask": "0xFF",
        "n_prompt": 2500,
        "n_gen": 150,
        "n_depth": 0,
        "avg_ts": 23.5,
        "avg_ns": 112735733737,
    }
    base.update(overrides)
    return base


def test_parse_throughput_decodes_one_record_per_line() -> None:
    text = "\n".join(json.dumps(_record(n_depth=d)) for d in (0, 1000))
    records = parse_throughput(text)
    assert [r["n_depth"] for r in records] == [0, 1000]


def test_parse_throughput_skips_non_json_lines() -> None:
    text = "/data/local/tmp/bench/results.jsonl: 1 file pulled, 0 skipped.\n" + json.dumps(
        _record()
    )
    records = parse_throughput(text)
    assert len(records) == 1


def test_parse_throughput_preserves_llama_bench_fields_verbatim() -> None:
    record = _record(avg_ts=16.83)
    (parsed,) = parse_throughput(json.dumps(record))
    assert parsed == record


def test_parse_throughput_against_the_real_committed_fixture() -> None:
    text = REAL_THROUGHPUT.read_text(encoding="utf-8")
    records = parse_throughput(text)
    assert len(records) == 18
    assert {r["cpu_mask"] for r in records} == {"0xC0", "0x3F", "0xFF"}


def test_parse_rss_extracts_name_vmhwm_and_pss() -> None:
    text = "minicpm5-1b-q8_0 VmHWM: 1373368 kB Pss: 1309649 kB"
    (row,) = parse_rss(text)
    assert row.name == "minicpm5-1b-q8_0"
    assert row.vmhwm_kb == 1373368
    assert row.pss_kb == 1309649


def test_parse_rss_skips_unparseable_lines() -> None:
    text = "# a stray comment\n" + "minicpm5-1b-q4_0 VmHWM: 849200 kB Pss: 832604 kB"
    rows = parse_rss(text)
    assert len(rows) == 1


def test_parse_rss_against_the_real_committed_fixture() -> None:
    text = REAL_RSS.read_text(encoding="utf-8")
    rows = parse_rss(text)
    assert len(rows) == 3
    names = {r.name for r in rows}
    assert names == {"minicpm5-1b-q8_0", "minicpm5-1b-q4_0", "minicpm5-1b-cursor-p20.Q4_K_M"}


def test_summarize_groups_throughput_by_cpu_mask() -> None:
    records = [_record(cpu_mask="0xFF", n_depth=0), _record(cpu_mask="0xC0", n_depth=0)]
    summary = summarize(records, [])
    assert set(summary["throughput_by_mask"].keys()) == {"0xFF", "0xC0"}


def test_summarize_includes_rss_rows() -> None:
    from arcsum.cli.device_bench import RssRow

    summary = summarize([], [RssRow("q8_0", 1000, 900)])
    assert summary["rss"] == [{"name": "q8_0", "vmhwm_kb": 1000, "pss_kb": 900}]


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_a_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_withholds_gate_without_wall_clock(tmp_path, capsys) -> None:
    throughput_path = tmp_path / "throughput.jsonl"
    throughput_path.write_text(json.dumps(_record()), encoding="utf-8")
    rss_path = tmp_path / "rss.txt"
    rss_path.write_text("q8_0 VmHWM: 1000 kB Pss: 900 kB", encoding="utf-8")

    rc = main(["report", str(throughput_path), str(rss_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "WITHHELD" in out


def test_main_passes_gate_when_wall_clock_under_ceiling(tmp_path, capsys) -> None:
    throughput_path = tmp_path / "throughput.jsonl"
    throughput_path.write_text(json.dumps(_record()), encoding="utf-8")
    rss_path = tmp_path / "rss.txt"
    rss_path.write_text("q8_0 VmHWM: 1000 kB Pss: 900 kB", encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = main(
        [
            "report",
            str(throughput_path),
            str(rss_path),
            "--wall-clock-minutes",
            "12.9",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True
    assert "PASS" in capsys.readouterr().out


def test_main_fails_gate_when_wall_clock_over_ceiling(tmp_path) -> None:
    throughput_path = tmp_path / "throughput.jsonl"
    throughput_path.write_text(json.dumps(_record()), encoding="utf-8")
    rss_path = tmp_path / "rss.txt"
    rss_path.write_text("", encoding="utf-8")
    out_path = tmp_path / "report.json"

    main(
        [
            "report",
            str(throughput_path),
            str(rss_path),
            "--wall-clock-minutes",
            "22.5",
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is False


def test_main_against_the_real_committed_phase0b_artifacts(tmp_path) -> None:
    out_path = tmp_path / "report.json"
    rc = main(
        [
            "report",
            str(REAL_THROUGHPUT),
            str(REAL_RSS),
            "--wall-clock-minutes",
            "12.9",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True
    assert len(report["rss"]) == 3
