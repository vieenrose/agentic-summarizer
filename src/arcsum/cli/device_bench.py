"""`arcsum-bench`: summarize Phase 0b device-measurement artifacts (SPEC §7, §9 Phase
0b) into the G4 wall-clock gate verdict.

    arcsum-bench report throughput.jsonl rss.txt --wall-clock-minutes 12.9 --out report.json

`throughput.jsonl` is `llama-bench -o jsonl` output, pulled verbatim off the reference
device (`runs/phase0b-2026-08-21/throughput.jsonl` is a real example) — parsed here with
NO reinterpretation of its schema. `rss.txt` is one `<name> VmHWM: <n> kB Pss: <n> kB`
line per quant, from `/proc/<pid>/status` + `smaps_rollup`.

**This tool does not compute the wall-clock-per-meeting figure itself.** Turning a
sparse throughput/depth sweep into a real per-meeting wall-clock estimate requires
domain judgement about how per-step KV depth actually compounds across an 11-step
reading phase — exactly the kind of interpretation that stayed a reviewed, documented
number in `SPEC.md` §7/§9 rather than a formula. `--wall-clock-minutes` is supplied by
the caller (that reviewed number) for the same reason `cli.report` takes it as an
input rather than re-deriving it: re-deriving it here would risk a second, silently
drifting answer to a question `SPEC.md` already has a reviewed one for.

**adb/ssh orchestration is deliberately out of scope.** The actual device session runs
over `ssh <user>@training-machine` + `adb shell`/`adb pull` (CLAUDE.md "Device access")
against hardware not reachable from wherever this CLI happens to run, and was one-shot,
reviewed shell work, not a reusable pipeline — automating it here would be untested
speculation about a machine this repository's test suite cannot see.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from arcsum.metrics.stats import gate_g4_budget

_RSS_LINE = re.compile(
    r"^(?P<name>\S+)\s+VmHWM:\s*(?P<vmhwm>\d+)\s*kB\s+Pss:\s*(?P<pss>\d+)\s*kB\s*$"
)


@dataclass(frozen=True, slots=True)
class RssRow:
    name: str
    vmhwm_kb: int
    pss_kb: int


def parse_throughput(text: str) -> list[dict]:
    """One `llama-bench -o jsonl` record per non-blank line. Records are returned
    exactly as llama-bench emits them -- no field is renamed or reinterpreted."""
    return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]


def parse_rss(text: str) -> list[RssRow]:
    """`<name> VmHWM: <n> kB Pss: <n> kB` per line. A line not matching that shape is
    skipped rather than raising -- this file is hand-appended-to during a live device
    session and may carry stray notes."""
    rows = []
    for line in text.splitlines():
        m = _RSS_LINE.match(line.strip())
        if m:
            rows.append(RssRow(m["name"], int(m["vmhwm"]), int(m["pss"])))
    return rows


def summarize(throughput: list[dict], rss: list[RssRow]) -> dict:
    by_mask: dict[str, list[dict]] = {}
    for record in throughput:
        by_mask.setdefault(record.get("cpu_mask", "?"), []).append(record)

    return {
        "throughput_by_mask": {
            mask: [
                {
                    "model_type": r.get("model_type"),
                    "n_depth": r.get("n_depth"),
                    "avg_ts": r.get("avg_ts"),
                    "avg_ns": r.get("avg_ns"),
                }
                for r in records
            ]
            for mask, records in sorted(by_mask.items())
        },
        "rss": [{"name": r.name, "vmhwm_kb": r.vmhwm_kb, "pss_kb": r.pss_kb} for r in rss],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="summarize throughput/RSS artifacts + the G4 gate")
    report.add_argument("throughput", type=Path, help="llama-bench -o jsonl output")
    report.add_argument("rss", type=Path, help="RSS text file (VmHWM/Pss per quant)")
    report.add_argument(
        "--wall-clock-minutes",
        type=float,
        default=None,
        help="the reviewed per-meeting wall-clock figure (SPEC §7); omit to withhold G4",
    )
    report.add_argument("--out", type=Path, default=None, help="write the JSON report here")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    throughput = parse_throughput(args.throughput.read_text(encoding="utf-8"))
    rss = parse_rss(args.rss.read_text(encoding="utf-8"))
    summary = summarize(throughput, rss)

    gate = gate_g4_budget(args.wall_clock_minutes)
    summary["gate"] = {"gate": gate.gate, "passed": gate.passed, "detail": gate.detail}

    for mask, records in summary["throughput_by_mask"].items():
        print(f"cpu_mask={mask}:")
        for r in records:
            print(f"  {r['model_type']} depth={r['n_depth']}: {r['avg_ts']:.2f} tok/s")
    for r in summary["rss"]:
        print(f"rss {r['name']}: VmHWM={r['vmhwm_kb']} kB Pss={r['pss_kb']} kB")
    verdict = "WITHHELD" if gate.passed is None else ("PASS" if gate.passed else "FAIL")
    print(f"{gate.gate}: {verdict} ({gate.detail})")

    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[bench] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
