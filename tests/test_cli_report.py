"""Pins `arcsum.cli.report` (SPEC §5.2): assembling the ship decision from already-
scored JSONL records, without re-deriving any gate logic that `metrics.stats` already
owns.
"""

from __future__ import annotations

import json

import pytest

from arcsum.cli.report import build_parser, build_report, main, render_report


def _record(meeting_id: str, system: str, **fields) -> dict:
    return {"meeting_id": meeting_id, "system": system, **fields}


def test_build_report_counts_wins_losses_and_pairs() -> None:
    records = [
        _record("m1", "agent", rouge1=0.6, rouge2=0.3, rougeL=0.5, coverage=0.4, density=1.0),
        _record("m1", "baseline", rouge1=0.5, rouge2=0.3, rougeL=0.5, coverage=0.4, density=1.0),
        _record("m2", "agent", rouge1=0.4, rouge2=0.3, rougeL=0.5, coverage=0.4, density=1.0),
        _record("m2", "baseline", rouge1=0.5, rouge2=0.3, rougeL=0.5, coverage=0.4, density=1.0),
    ]
    report = build_report(records, treatment="agent", control="baseline", g1_passed=True)
    assert report["paired_count"] == 2
    rouge1 = next(c for c in report["comparisons"] if c["metric"] == "rouge1")
    assert rouge1["wins"] == 1
    assert rouge1["losses"] == 1
    assert rouge1["n"] == 2


def test_build_report_withholds_g3_below_min_n() -> None:
    records = [
        _record("m1", "agent", rouge1=0.6, rouge2=0.3, rougeL=0.5, coverage=0.4, density=1.0),
        _record("m1", "baseline", rouge1=0.5, rouge2=0.3, rougeL=0.5, coverage=0.4, density=1.0),
    ]
    report = build_report(records, treatment="agent", control="baseline", g1_passed=True)
    g3_rouge1 = next(g for g in report["gates"] if g["gate"] == "G3_rouge1")
    assert g3_rouge1["passed"] is None


def test_build_report_ships_baseline_when_g1_fails_regardless_of_scores() -> None:
    records = [
        _record("m1", "agent", rouge1=0.9, rouge2=0.9, rougeL=0.9, coverage=0.9, density=1.0),
        _record("m1", "baseline", rouge1=0.1, rouge2=0.1, rougeL=0.1, coverage=0.1, density=1.0),
    ]
    report = build_report(records, treatment="agent", control="baseline", g1_passed=False)
    assert report["decision"] == "ship the baseline"


def test_build_report_uses_inversions_field_for_g2() -> None:
    records = [
        _record("m1", "agent", inversions=0),
        _record("m1", "baseline", inversions=1),
        _record("m2", "agent", inversions=2),
        _record("m2", "baseline", inversions=0),
    ]
    report = build_report(records, treatment="agent", control="baseline", g1_passed=True)
    g2 = next(g for g in report["gates"] if g["gate"] == "G2_faithfulness")
    # treatment total=2, control total=1 -> 2 <= 1 is False
    assert g2["passed"] is False


def test_build_report_g4_withheld_without_wall_clock() -> None:
    report = build_report([], treatment="agent", control="baseline", g1_passed=True)
    g4 = next(g for g in report["gates"] if g["gate"] == "G4_budget")
    assert g4["passed"] is None


def test_build_report_g4_uses_supplied_wall_clock() -> None:
    report = build_report(
        [], treatment="agent", control="baseline", g1_passed=True, wall_clock_minutes=5.0
    )
    g4 = next(g for g in report["gates"] if g["gate"] == "G4_budget")
    assert g4["passed"] is True


def test_build_report_applies_tie_thresholds() -> None:
    records = [
        _record("m1", "agent", rouge1=0.51),
        _record("m1", "baseline", rouge1=0.50),
    ]
    without_threshold = build_report(
        records, treatment="agent", control="baseline", g1_passed=True, tie_thresholds={}
    )
    with_threshold = build_report(
        records,
        treatment="agent",
        control="baseline",
        g1_passed=True,
        tie_thresholds={"rouge1": 0.05},
    )
    r1_no_threshold = next(c for c in without_threshold["comparisons"] if c["metric"] == "rouge1")
    r1_threshold = next(c for c in with_threshold["comparisons"] if c["metric"] == "rouge1")
    assert r1_no_threshold["wins"] == 1
    assert r1_threshold["ties"] == 1


def test_render_report_mentions_decision_and_every_gate() -> None:
    report = build_report([], treatment="agent", control="baseline", g1_passed=True)
    text = render_report(report)
    assert "decision:" in text
    for g in report["gates"]:
        assert g["gate"] in text


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_exactly_one_of_g1_flags() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["a.jsonl", "--treatment", "agent", "--control", "baseline"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "a.jsonl",
                "--treatment",
                "agent",
                "--control",
                "baseline",
                "--g1-passed",
                "--g1-failed",
            ]
        )


def test_main_merges_multiple_score_files_and_writes_report(tmp_path, capsys) -> None:
    scored = tmp_path / "scored.jsonl"
    scored.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _record("m1", "agent", rouge1=0.6),
                _record("m1", "baseline", rouge1=0.5),
            ]
        ),
        encoding="utf-8",
    )
    judged = tmp_path / "judged.jsonl"
    judged.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _record("m1", "agent", inversions=0),
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "report.json"

    rc = main(
        [
            str(scored),
            str(judged),
            "--treatment",
            "agent",
            "--control",
            "baseline",
            "--g1-passed",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "decision:" in captured.out
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["treatment"] == "agent"
    assert report["paired_count"] == 1


def test_main_reads_tie_thresholds_file(tmp_path) -> None:
    scored = tmp_path / "scored.jsonl"
    scored.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _record("m1", "agent", rouge1=0.51),
                _record("m1", "baseline", rouge1=0.50),
            ]
        ),
        encoding="utf-8",
    )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"rouge1": 0.05}), encoding="utf-8")
    out_path = tmp_path / "report.json"

    main(
        [
            str(scored),
            "--treatment",
            "agent",
            "--control",
            "baseline",
            "--g1-passed",
            "--tie-thresholds",
            str(thresholds),
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    rouge1 = next(c for c in report["comparisons"] if c["metric"] == "rouge1")
    assert rouge1["ties"] == 1


def test_main_skips_blank_lines_in_input_files(tmp_path) -> None:
    scored = tmp_path / "scored.jsonl"
    scored.write_text(
        json.dumps(_record("m1", "agent", rouge1=0.5))
        + "\n\n"
        + json.dumps(_record("m1", "baseline", rouge1=0.5))
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "report.json"

    rc = main(
        [
            str(scored),
            "--treatment",
            "agent",
            "--control",
            "baseline",
            "--g1-passed",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
