"""Swap the replayable part of a pool's `SYNTHESIZE` slice for journal-shaped rows.

    python tools/swap_synth_slice.py --pool data/staging/sft_pool_v11.jsonl \\
        --synth data/staging/synth_journal.jsonl \\
        --out data/staging/sft_pool_v12.jsonl --report runs/pool-v12-report.json

**Why `clean_pool.py --synth` cannot do this.** That path substitutes by `(meeting, step)`,
which assumes the new slice has the same membership as the old one. It does not: journal
rows are rebuilt by REPLAYING a meeting's reading steps, so they exist only for meetings that
have reading steps, and they carry `step = -1` because a synthesis call has no chunk index.

**What must NOT be replaced, and this is the whole reason the tool is careful.** Measured on
`sft_pool_v11.jsonl`: 450 synthesis rows span 324 meetings, but only **175 of those meetings
have reading steps**. The other 149 are synthetic capability supervision — `hedge-*`,
reversal and deliberation scenarios — which have no transcript to replay and therefore no
journal to rebuild. They are also the highest-value rows per unit in the pool: **12 hedge
rows** (`tools/gen_hedge_synth.py`) are what fixed `qwen-tools-v4`'s deterministic polarity
inversion and recovered both failing G3 gates. Replacing the slice wholesale would delete
them and silently regress a fix that cost a full experiment to find.

So the rule is: a synthesis row is dropped **only if a journal-shaped row exists for its
meeting**. Everything else survives untouched, still v1.0-shaped, which is accepted — those
rows teach a specific behaviour on a small memory, and that behaviour is not what the journal
work is changing.

**The swap is also a net ADDITION.** The replay covers 245 meetings, of which only 175 had
synthesis supervision before, so ~70 meetings gain a synthesis row they never had.

Counts are reported before and after, per class, because this project's recorded lesson is
that reading the shares a build tool prints is necessary and not sufficient — `sft-dropv1`
landed below its NOP cap while reporting the cap it solved against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def is_synthesis(row: dict) -> bool:
    return row["prompt"].startswith("MEMORY:")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--synth", type=Path, required=True,
                   help="journal-shaped rows from tools/gen_journal_synth.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    pool = [json.loads(ln) for ln in args.pool.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    new = [json.loads(ln) for ln in args.synth.read_text(encoding="utf-8").splitlines()
           if ln.strip()]
    if not new:
        print("[swap] REFUSED: no journal rows supplied", file=sys.stderr)
        return 1

    versions = {r.get("prompt_version") for r in new}
    if len(versions) != 1:
        print(f"[swap] REFUSED: journal rows mix prompt versions {versions}", file=sys.stderr)
        return 1

    replace_for = {r["meeting"] for r in new}
    before_syn = [r for r in pool if is_synthesis(r)]

    kept: list[dict] = []
    dropped = 0
    for r in pool:
        if is_synthesis(r) and r["meeting"] in replace_for:
            dropped += 1
            continue
        kept.append(r)
    out_rows = kept + new

    after_syn = [r for r in out_rows if is_synthesis(r)]
    survivors = [r for r in before_syn if r["meeting"] not in replace_for]
    gained = len(replace_for - {r["meeting"] for r in before_syn})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "pool": str(args.pool), "synth": str(args.synth), "out": str(args.out),
        "rows_before": len(pool), "rows_after": len(out_rows),
        "synthesis_before": len(before_syn), "synthesis_after": len(after_syn),
        "synthesis_dropped": dropped,
        "journal_rows_added": len(new),
        "meetings_gaining_synthesis": gained,
        "synthetic_rows_preserved": len(survivors),
        "preserved_meeting_prefixes": sorted(
            {r["meeting"].split("-")[0] for r in survivors})[:12],
        "prompt_version": versions.pop(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
