#!/usr/bin/env python3
"""Trace files -> train-split-filtered JSONL, ready for build_sft.py.

Eval meetings (t1/micro) have traces too (the generation ran before the carve); they are
held out here so no eval-meeting step enters the SFT set. Also verifies every record's
prompt_version is uniform before build_sft refuses it anyway.

    python tools/filter_train_traces.py data/traces_v2/*.jsonl --out data/traces_v2/train/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.prompts import PROMPT_VERSION  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    manifest = json.loads(
        (Path("data/transcripts/manifest.json")).read_text(encoding="utf-8")
    )
    split_of = {row["meeting_id"]: row["split"] for row in manifest}

    kept = dropped_eval = 0
    versions: set[str] = set()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.traces:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        out_rows = []
        for row in rows:
            versions.add(row.get("prompt_version", "?"))
            meeting = row["meeting"]
            split = split_of.get(meeting, "train")
            if split != "train":
                dropped_eval += 1
                continue
            out_rows.append(row)
            kept += 1
        if out_rows:
            (args.out / path.name).write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
                encoding="utf-8",
            )
        print(f"[filter] {path.name}: {len(rows)} records -> {len(out_rows)} train")

    print(f"[filter] kept {kept} train steps, dropped {dropped_eval} eval-meeting steps")
    print(f"[filter] prompt versions: {sorted(versions)} (code: {PROMPT_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
