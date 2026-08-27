"""`arcsum-report`: the SPEC §5.2 ship decision, computed from already-scored JSONL
files (produced by `arcsum-score` and/or `arcsum-judge`) plus the G1 probe verdict.

    arcsum-report scored.jsonl judged.jsonl \\
        --treatment agent --control baseline \\
        --g1-passed \\
        --tie-thresholds thresholds.json \\
        --wall-clock-minutes 12.4 \\
        --out report.json

This module owns no measurement of its own — it only assembles gate verdicts from
records the caller already produced, per `metrics.stats`'s doctrine that a gate's
inputs (paired scores, inversions, wall-clock, G1) are supplied, never re-derived here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arcsum.metrics.stats import (
    compare,
    count_inversions,
    gate_g2_faithfulness,
    gate_g3_quality,
    gate_g4_budget,
    load_scores,
    ship_decision,
)


def _read_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_report(
    records: list[dict],
    *,
    treatment: str,
    control: str,
    g1_passed: bool,
    tie_thresholds: dict[str, float] | None = None,
    wall_clock_minutes: float | None = None,
) -> dict:
    """Pure assembly, decoupled from file I/O so it is directly testable."""
    scores = load_scores(records)
    comparisons, paired_count = compare(
        scores, treatment, control, tie_thresholds=tie_thresholds or {}
    )

    # Paired, like `compare`: summing the two arms over different denominators
    # silently favours whichever has fewer judged records (see count_inversions).
    treatment_inversions = count_inversions(scores, treatment, paired_with=control)
    control_inversions = count_inversions(scores, control, paired_with=treatment)
    # PAIRED judged count, matching the paired inversion sums above: a meeting judged
    # for only one arm contributes to neither, so the gate withholds when the two arms
    # share no judged meetings rather than comparing across disjoint sets.
    judged_paired = sum(
        1
        for systems in scores.values()
        if (systems.get(treatment) or {}).get("inversions") is not None
        and (systems.get(control) or {}).get("inversions") is not None
    )
    g2 = gate_g2_faithfulness(
        treatment_inversions, control_inversions, judged_records=judged_paired
    )
    g3 = gate_g3_quality(comparisons)
    g4 = gate_g4_budget(wall_clock_minutes)

    gates = [g2, *g3, g4]
    decision = ship_decision(gates, g1_passed=g1_passed)

    return {
        "treatment": treatment,
        "control": control,
        "paired_count": paired_count,
        "g1_passed": g1_passed,
        "comparisons": [
            {
                "metric": c.metric,
                "n": c.n,
                "wins": c.wins,
                "losses": c.losses,
                "ties": c.ties,
                "mean_delta": c.mean_delta,
                "stderr": c.stderr,
                "p_value": c.p_value,
            }
            for c in comparisons
        ],
        "gates": [{"gate": g.gate, "passed": g.passed, "detail": g.detail} for g in gates],
        "decision": decision,
    }


def render_report(report: dict) -> str:
    lines = [
        f"paired meetings: {report['paired_count']} ({report['treatment']} vs {report['control']})",
        f"G1 (revision probe): {'PASS' if report['g1_passed'] else 'FAIL'}",
    ]
    for c in report["comparisons"]:
        lines.append(
            f"{c['metric']}: n={c['n']} wins={c['wins']} losses={c['losses']} "
            f"ties={c['ties']} mean_delta={c['mean_delta']:+.3f} "
            f"(SE={c['stderr']:.3f}) p={c['p_value']:.3f}"
        )
    for g in report["gates"]:
        verdict = "WITHHELD" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
        lines.append(f"{g['gate']}: {verdict} ({g['detail']})")
    lines.append(f"decision: {report['decision']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("scores", type=Path, nargs="+", help="JSONL score/judge record files")
    p.add_argument("--treatment", required=True, help="system name being evaluated")
    p.add_argument("--control", required=True, help="baseline system name")
    g1 = p.add_mutually_exclusive_group(required=True)
    g1.add_argument("--g1-passed", action="store_true", dest="g1_passed", default=None)
    g1.add_argument("--g1-failed", action="store_false", dest="g1_passed", default=None)
    p.add_argument(
        "--tie-thresholds", type=Path, default=None, help="JSON file: {metric: threshold}"
    )
    p.add_argument(
        "--wall-clock-minutes",
        type=float,
        default=None,
        help="measured wall-clock per meeting (SPEC §7 gate); omit if not yet measured",
    )
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = _read_records(args.scores)
    tie_thresholds = (
        json.loads(args.tie_thresholds.read_text(encoding="utf-8")) if args.tie_thresholds else {}
    )

    report = build_report(
        records,
        treatment=args.treatment,
        control=args.control,
        g1_passed=args.g1_passed,
        tie_thresholds=tie_thresholds,
        wall_clock_minutes=args.wall_clock_minutes,
    )

    print(render_report(report))
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
