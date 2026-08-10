#!/usr/bin/env python3
"""Aggregate op-trace statistics from gen_traces JSONL output (PLAN.md §2, §7.4).

    python tools/trace_report.py data/traces_v1_brokenfilter
    python tools/trace_report.py data/traces/train.jsonl data/traces/v2.jsonl

Accepts one or more `.jsonl` trace files (or directories, globbed for `*.jsonl`) and prints,
pooled across every record, the §7.4 operational metrics:

* step / meeting counts, split by `lang` and by source prefix (qmsum/mbank/synth);
* **valid-op rate** — applied ops over the ops submitted to the harness, with NOP excluded
  from both the numerator and the denominator (GT1). Counting NOP as a valid op has been a
  real bug in this repo twice;
* **anchor rate (raw)** — applied ADD/UPD ops that natively carried a resolvable `[m:ss]`,
  over applied ADD/UPD only. NOP and TITLE are never scored;
* **NOP share** — steps whose accepted ops are all NOP;
* **revision share** — steps whose accepted ops contain a UPD or DEL;
* **judge veto rate** — judge-vetoed ops over claim-bearing ops (ADD/UPD/CMP bullets),
  when the records carry veto info.

Ops are parsed with `voxsum.ops.parse_ops` — the harness's own parser — never a local regex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.ops import Add, Cmp, Del, Nop, Upd, parse_ops  # noqa: E402

SOURCES = ("qmsum", "mbank", "synth")


def _source_of(meeting: str) -> str:
    for prefix in SOURCES:
        if meeting.startswith(prefix):
            return prefix
    return "other"


def load_records(paths: list[str | Path]) -> tuple[list[dict], list[Path]]:
    """Expand files/directories and read every JSONL line. Never raises on a bad path."""
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"[report] skipping {path}: not a file or directory", file=sys.stderr)
    records: list[dict] = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records, files


def valid_op_counts(records: list[dict]) -> tuple[int, int]:
    """(applied, submitted) non-NOP ops — GT1's numerator and denominator.

    Submitted ops are parsed from `raw`; applied ops are the record's `target` (gen_traces
    writes exactly the ops the harness accepted). A judge-vetoed op never reached the
    harness, so it is subtracted from the submitted count — matching `Trace.valid_op_rate`
    in `agent.py`, which scores only the ops that reached `apply_ops`. NOP is excluded from
    BOTH sides: counting it as a success has been a real bug in this repo twice.
    """
    num = den = 0
    for rec in records:
        den += sum(1 for op in parse_ops(rec.get("raw", "")) if not isinstance(op, Nop))
        den = max(den - len(rec.get("vetoed") or []), 0)
        num += sum(1 for op in parse_ops(rec.get("target", "")) if not isinstance(op, Nop))
    return num, den


def valid_op_rate(records: list[dict]) -> float | None:
    num, den = valid_op_counts(records)
    return num / den if den else None


def anchor_counts(records: list[dict]) -> tuple[int, int]:
    """(native-anchored, applied) ADD/UPD ops — the raw anchor rate (§7.4).

    Only applied ADD/UPD carry anchors (agent.py `anchor_rate_raw`): NOP and TITLE never
    enter the denominator, and neither do CMP bullets, whose anchors would otherwise
    inflate the model's native-anchor signal.
    """
    num = den = 0
    for rec in records:
        for op in parse_ops(rec.get("target", "")):
            if isinstance(op, (Add, Upd)):
                den += 1
                if op.anchor is not None:
                    num += 1
    return num, den


def anchor_rate_raw(records: list[dict]) -> float | None:
    num, den = anchor_counts(records)
    return num / den if den else None


def nop_share(records: list[dict]) -> float | None:
    """Fraction of steps whose accepted ops are all NOP."""
    if not records:
        return None
    nops = 0
    for rec in records:
        accepted = parse_ops(rec.get("target", ""))
        if accepted and all(isinstance(op, Nop) for op in accepted):
            nops += 1
    return nops / len(records)


def revision_share(records: list[dict]) -> float | None:
    """Fraction of steps whose accepted ops contain a UPD or DEL."""
    if not records:
        return None
    rev = 0
    for rec in records:
        if any(isinstance(op, (Upd, Del)) for op in parse_ops(rec.get("target", ""))):
            rev += 1
    return rev / len(records)


def veto_rate(records: list[dict]) -> float | None:
    """Judge-vetoed ops over claim-bearing ops (ADD/UPD/CMP bullets) in `raw`.

    Vetoes only ever target claims — NOP/TITLE/DEL are never judged (gen_traces.py) — so
    those are out of scope here too. None when no claims were submitted, e.g. a run with
    no judge filter.
    """
    vetoed = sum(len(rec.get("vetoed") or []) for rec in records)
    judgeable = 0
    for rec in records:
        for op in parse_ops(rec.get("raw", "")):
            if isinstance(op, (Add, Upd)):
                judgeable += 1
            elif isinstance(op, Cmp):
                judgeable += len(op.bullets)
    return vetoed / judgeable if judgeable else None


def _rate(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def report(records: list[dict], files: list[Path] | None = None) -> str:
    """Render the aggregated report. Pure and deterministic — testable."""
    files = files or []
    n_steps = len(records)
    n_meetings = len({r.get("meeting") for r in records if r.get("meeting")})
    if n_steps == 0:
        return "[report] no records"

    by_lang: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for rec in records:
        lang = rec.get("lang") or "?"
        by_lang[lang] = by_lang.get(lang, 0) + 1
        source = _source_of(rec.get("meeting", ""))
        by_source[source] = by_source.get(source, 0) + 1

    def share(count: int) -> str:
        return f"{100.0 * count / n_steps:.1f}%"

    lines = [
        f"[report] {len(files)} file(s), {n_meetings} meetings, {n_steps} steps",
        "steps by lang:",
    ]
    lines += [f"  {k:6s} {v:5d} ({share(v)})" for k, v in sorted(by_lang.items())]
    lines.append("steps by source:")
    lines += [f"  {k:6s} {v:5d} ({share(v)})" for k, v in sorted(by_source.items())]

    v_num, v_den = valid_op_counts(records)
    a_num, a_den = anchor_counts(records)
    lines.append("metrics:")
    lines.append(
        f"  valid-op rate    {_rate(v_num / v_den) if v_den else 'n/a'}"
        + (f"   ({v_num}/{v_den})" if v_den else "")
    )
    lines.append(
        f"  anchor rate raw  {_rate(a_num / a_den) if a_den else 'n/a'}"
        + (f"   ({a_num}/{a_den})" if a_den else "")
    )
    lines += [
        f"  NOP share        {_rate(nop_share(records))}",
        f"  revision share   {_rate(revision_share(records))}",
        f"  judge veto rate  {_rate(veto_rate(records))}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "paths",
        nargs="+",
        help="trace .jsonl files, or directories to glob for *.jsonl",
    )
    args = p.parse_args(argv)
    records, files = load_records(args.paths)
    if not records:
        print("[report] no records found", file=sys.stderr)
        return 1
    print(report(records, files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
