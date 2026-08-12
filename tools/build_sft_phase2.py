#!/usr/bin/env python3
"""Phase-2 dataset for the en student: real-transcript adaptation (PLAN 0c).

The fabrication on real meetings is a training-distribution gap: the passing G1 mix is
~73% clean synthetic b128. Phase 2 continues training from the G1-passing checkpoint on
a REAL-heavy mix at low LR, so the protocol pattern survives while the model learns the
real-transcript shape (long noisy lines, no speakers, synthesized clocks).

Mix (en only, train split):
  * real 2048-budget steps (qmsum + meetingbank)  x3  — the eval distribution, where the
    fabrication was measured
  * real b128 steps (mbank, Y wave)                    — small-chunk real shape
  * combined b128 steps (the G1-pattern teachers) x0.5 — keeps the screen pattern alive

    python tools/build_sft_phase2.py --out data/sft/lfm-en-phase2.jsonl
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

from build_sft_qwen import build_sample  # noqa: E402  (text grammar — LFM path)

REAL_SOURCES = ("qmsum", "meetingbank")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tracedir", type=Path, default=Path("data/traces_v2"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    manifest = json.loads(
        (Path(__file__).resolve().parent.parent / "data/transcripts/manifest.json").read_text()
    )
    split_of = {r["meeting_id"]: r["split"] for r in manifest}
    source_of = {r["meeting_id"]: r["source"].split(":")[0] for r in manifest}

    standard_real: list[dict] = []
    b128_real: list[dict] = []
    b128_combined: list[dict] = []
    for path in sorted(args.tracedir.glob("*.jsonl")):
        rows = [r for r in load(path) if r.get("lang") == "en"]
        for r in rows:
            if split_of.get(r["meeting"], "train") != "train":
                continue
            src = source_of.get(r["meeting"], "")
            if "b128" in path.name:
                if src in REAL_SOURCES:
                    b128_real.append(r)
                elif r["meeting"].startswith("synth-en-combined"):
                    b128_combined.append(r)
            elif src in REAL_SOURCES:
                standard_real.append(r)

    rng = random.Random(args.seed)

    def _nop_cap(rows: list[dict], frac: float) -> list[dict]:
        nops = [r for r in rows if r["is_nop"]]
        others = [r for r in rows if not r["is_nop"]]
        cap = int(len(others) * frac / max(1 - frac, 1e-6))
        if len(nops) > cap:
            rng.shuffle(nops)
            nops = nops[:cap]
        return others + nops

    standard_real = _nop_cap(standard_real, 0.30)
    b128_real = _nop_cap(b128_real, 0.30)
    b128_combined = _nop_cap(b128_combined, 0.30)

    # Upsample: the eval distribution (real 2048) must dominate the adaptation.
    pool = standard_real * 3 + b128_real + b128_combined[: len(b128_combined) // 2]
    rng.shuffle(pool)
    samples = [build_sample(r) for r in pool]
    samples = [s for s in samples if s["completion"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for s in samples:
            sink.write(json.dumps(s, ensure_ascii=False) + "\n")

    counts = Counter(
        ("nop" if s["is_nop"] else "revision" if s["has_revision"] else "add") for s in samples
    )
    print(f"[phase2] {len(samples)} samples: {dict(counts)}")
    print(f"[phase2] real-2048 x3={len(standard_real)*3}  b128-real={len(b128_real)}  "
          f"combined/2={len(b128_combined)//2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
