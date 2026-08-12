"""Phase-5 (350M primary): p4 mix + the Z-wave traces (new real QMSum meetings).

Usage: python3 tools/build_sft_phase5.py
"""
import json, random, sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "train"); sys.path.insert(0, "tools")
from build_sft_qwen import build_sample

OUT = "data/sft/lfm-en-phase5.jsonl"
TRACE_WAVES = ["data/traces_v2/Z1_en.jsonl", "data/traces_v2/Z2_en.jsonl"]
BASE = "data/sft/lfm-en-phase4.jsonl"

def main() -> None:
    base = [json.loads(l) for l in open(BASE)]
    new_samples = []
    for w in TRACE_WAVES:
        if not Path(w).exists():
            print(f"skip {w} (missing)"); continue
        kept = 0
        for line in open(w):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r["meeting"].startswith("qmsum-"):
                continue
            if r["is_nop"] or not r.get("target"):
                continue
            s = build_sample(r)
            if s and s["completion"]:
                new_samples.append(s); kept += 1
        print(f"{w}: +{kept} samples")
    pool = base + new_samples
    random.Random(0).shuffle(pool)
    with open(OUT, "w") as f:
        for s in pool:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"phase-5: {len(pool)} (base {len(base)} + new {len(new_samples)})")

if __name__ == "__main__":
    main()
