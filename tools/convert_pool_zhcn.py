"""Convert an SFT pool to Simplified internals (`arcsum/simplified.py`).

    python tools/convert_pool_zhcn.py --pool data/staging/sft_pool_v12.jsonl \\
        --out data/staging/sft_pool_v12_zhcn.jsonl --report runs/zhcn-pool-report.json

**What this buys, measured on this project's corpora**: the student's own tokenizer spends
10.5% fewer tokens on Simplified than on Traditional (1.577 -> 1.761 chars/token over 354,995
characters). Chunking is token-based and wall-clock is steps x per-step latency, so that is
~10.5% fewer reading steps — 19.0 min -> ~17 min against G4's 20.00 min ceiling, whose
measured margin is currently 3%.

**Why the whole pool and not just the transcript.** The model's memory, its journal and its
own emitted points are all re-rendered into every subsequent prompt. Converting only the input
would leave the model reading Simplified chunks and writing Traditional points, mixing scripts
inside a single prompt and losing most of the saving on the part of the prompt that repeats.

**The `system` field is converted too, and inference must match.** Rows carry the system
prompt verbatim, so serving must apply `to_simplified(...)` to `prompts.*_system_prompt()` or
the fine-tune sees a prompt it was never trained on. That is the same train/serve mismatch
CLAUDE.md trap 10 records for the jinja flag, and it is equally silent.

**Refuses rather than degrading.** `simplified.converter()` returns identity when the extra is
missing, which is right for the harness and wrong here: writing an unconverted pool under a
converted filename would poison a training run invisibly. This tool checks `available()` and
exits non-zero.

**Numerals are verified, not assumed.** OpenCC touches Han characters, but a conversion bug
that altered a digit would corrupt every grounding measurement downstream, so the tool asserts
that the multiset of Arabic numbers is identical before and after on every row and reports the
count it checked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.simplified import (  # noqa: E402
    SCRIPT_VERSION,
    TO_SIMPLIFIED,
    available,
    converter,
)

ARABIC = re.compile(r"\d+")
FIELDS = ("system", "prompt", "completion")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    if not available():
        print("[zhcn] REFUSED: opencc is not installed. Install the `script` extra; "
              "writing an unconverted pool under a converted name would poison a run "
              "invisibly.", file=sys.stderr)
        return 1

    convert = converter(TO_SIMPLIFIED)
    rows = [json.loads(ln) for ln in args.pool.read_text(encoding="utf-8").splitlines()
            if ln.strip()]

    out_rows = []
    changed = numerals_checked = 0
    mismatches: list[str] = []
    for r in rows:
        new = dict(r)
        for f in FIELDS:
            if f not in r or not isinstance(r[f], str):
                continue
            src = r[f]
            dst = convert(src)
            if ARABIC.findall(src) != ARABIC.findall(dst):
                mismatches.append(f"{r.get('meeting')}:{f}")
            numerals_checked += len(ARABIC.findall(src))
            if dst != src:
                changed += 1
            new[f] = dst
        # Completions must still parse as the same structure they did before.
        new["script_version"] = SCRIPT_VERSION
        out_rows.append(new)

    if mismatches:
        print(f"[zhcn] REFUSED: conversion altered numerals in {len(mismatches)} fields, "
              f"e.g. {mismatches[:5]}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "pool": str(args.pool), "out": str(args.out),
        "rows": len(out_rows), "fields_changed": changed,
        "numerals_verified": numerals_checked,
        "script_version": SCRIPT_VERSION, "config": TO_SIMPLIFIED,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
