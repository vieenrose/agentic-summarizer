#!/usr/bin/env python3
"""SFT dataset v3: source-aware NOP handling (fixes the v2 NOP-collapse).

v2 collapsed to NOP because the budget-128 traces are 96% NOP and taught "small chunk
-> NOP" at the screen's exact chunk size. Here the two distributions are mixed
deliberately:

* **standard traces** (waves 1-3, budget 2048): NOP capped at 35% of non-NOP steps;
* **b128 traces** (screen distribution): every *active* step kept (they are the scarce
  ADD/UPD/TITLE demonstrations at small-chunk scale) plus a small sampled NOP subset
  (3 per meeting, spread across the trajectory) so filler->NOP is still taught without
  dominating.

    python tools/build_sft_v3.py data/traces_v2 --out data/sft/train3.jsonl --valid-out data/sft/valid3.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

from build_sft import build_sample  # noqa: E402

NOP_PER_B128_MEETING = 2


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tracedir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--valid-out", type=Path, default=None)
    p.add_argument("--valid-frac", type=float, default=0.04)
    p.add_argument("--max-nop-frac", type=float, default=0.35, help="standard traces only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    standard: list[dict] = []
    b128: list[dict] = []
    for path in sorted(args.tracedir.glob("*.jsonl")):
        if "b128" in path.name:
            b128.extend(load(path))
        else:
            standard.extend(load(path))

    versions = {r.get("prompt_version") for r in standard + b128}
    if len(versions) != 1:
        raise SystemExit(f"mixed prompt versions: {sorted(versions)}")
    print(f"[v3] standard {len(standard)} steps, b128 {len(b128)} steps")

    # --- standard: NOP cap ---
    nops = [r for r in standard if r["is_nop"]]
    others = [r for r in standard if not r["is_nop"]]
    cap = int(len(others) * args.max_nop_frac / max(1 - args.max_nop_frac, 1e-6))
    if len(nops) > cap:
        rng.shuffle(nops)
        print(f"[v3] standard NOP downsample: {len(nops)} -> {cap}")
        nops = nops[:cap]
    standard = others + nops

    # --- b128: short screen-structured meetings are the target distribution; long
    # meetings' b128 traces are 96% NOP and collapse the student (v2/v3 lesson), so
    # only their *active* steps are kept. Short meetings keep all active + a NOP sample.
    active = [r for r in b128 if not r["is_nop"]]
    nops = [r for r in b128 if r["is_nop"]]
    per_meeting: dict[str, list[dict]] = {}
    for r in nops:
        per_meeting.setdefault(r["meeting"], []).append(r)
    short_nops: list[dict] = []
    for meeting, rows in sorted(per_meeting.items()):
        if len(rows) > 20:  # long meeting (82 chunks at b128): its NOPs poison; drop
            print(f"[v3] dropping {len(rows)} NOPs of long b128 meeting {meeting}")
            continue
        rows = sorted(rows, key=lambda r: r["step"])
        n = min(NOP_PER_B128_MEETING, len(rows))
        idx = {round(i * (len(rows) - 1) / max(n - 1, 1)) for i in range(n)}
        short_nops += [rows[i] for i in sorted(idx)]
    print(f"[v3] b128: {len(active)} active kept, {len(short_nops)} short-meeting NOPs kept")
    b128 = active + short_nops

    samples = [build_sample(r) for r in standard + b128]
    samples = [s for s in samples if s["completion"]]
    rng.shuffle(samples)

    valid: list[dict] = []
    if args.valid_out and args.valid_frac > 0:
        meetings = sorted({s["meeting"] for s in samples})
        rng.shuffle(meetings)
        n_valid = max(1, round(len(meetings) * args.valid_frac))
        held = set(meetings[:n_valid])
        valid = [s for s in samples if s["meeting"] in held]
        samples = [s for s in samples if s["meeting"] not in held]
        print(f"[v3] held out {len(held)} meetings ({len(valid)} steps)")

    for target, rows in ((args.out, samples), (args.valid_out, valid)):
        if target is None or not rows:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(
        ("nop" if s["is_nop"] else "revision" if s["has_revision"] else "add") for s in samples
    )
    total = max(len(samples), 1)
    print(f"[v3] {len(samples)} train samples -> {args.out}")
    for key in ("add", "revision", "nop"):
        print(f"[v3]   {key:9s} {counts[key]:6d}  {counts[key] / total:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
