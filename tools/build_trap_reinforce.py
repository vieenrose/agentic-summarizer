"""Trap-reinforcement augmentation: inject a trap-style line (a topic raised and
explicitly ruled out of scope) into the CHUNK of zh SFT rows, leaving the completion
unchanged. Teaches the model to NOT report the trap — the regression v2 showed on the
G1 screen (leaking 咖啡機 as "決定不討論...").

Distractor injection: the target stays valid for the chunk's real content, so the
trap is an extra irrelevant line the student must learn to ignore.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from voxsum.transcript import clock_to_sec, sec_to_clock

#: Varied trap topics so the student learns the *pattern*, not a specific word.
TRAPS = [
    "辦公室咖啡機預算", "茶水間裝潢", "團康活動經費", "停車位分配",
    "制服採購", "尾牙場地", "零食採購預算", "部門旅遊保險",
]
RAISE = [
    "我們今天也要討論{trap}嗎？",
    "順帶一提，{trap}能不能排進議程？",
    "還有一件事，{trap}要處理嗎？",
]
RULE_OUT = [
    "不，{trap}不在今天的議程。",
    "{trap}先放著，今天不討論。",
    "那個{trap}之後再說，先跳過。",
]


def inject_trap(prompt: str, rng: random.Random) -> str | None:
    """Insert a trap line + ruling-out into the CHUNK of `prompt`. None if the
    chunk has no parseable lines."""
    trap = rng.choice(TRAPS)
    m = re.search(r"CHUNK:\n(.*)", prompt, re.S)
    if not m:
        return None
    chunk = m.group(1)
    lines = chunk.rstrip("\n").split("\n")
    if not lines or not lines[0].startswith("["):
        return None
    # find the chunk's clock range
    starts = []
    for l in lines:
        mm = re.match(r"\[(\d+):(\d{2})(?::(\d{2}))?\]", l)
        if mm:
            s = int(mm.group(1)) * 60 + int(mm.group(2)) + (int(mm.group(3) or 0) * 3600)
            starts.append(s)
    if not starts:
        return None
    lo, hi = min(starts), max(starts)
    pos = len(lines) // 2  # insert midway
    # interpolate strictly between the surrounding lines' clocks so the result
    # stays monotonic (a transcript v1 rule the student must keep seeing).
    before = starts[pos - 1] if pos > 0 else lo
    after = starts[pos] if pos < len(lines) else hi
    gap = after - before
    if gap >= 20:
        t = before + max(5, gap // 3)
        t2 = t + min(5, gap // 3)
    else:
        t, t2 = hi + 30, hi + 35
    raise_line = f"[{sec_to_clock(t)}] S3: " + rng.choice(RAISE).format(trap=trap)
    rule_line = f"[{sec_to_clock(t2)}] S1: " + rng.choice(RULE_OUT).format(trap=trap)
    lines = lines[:pos] + [raise_line, rule_line] + lines[pos:]
    new_chunk = "\n".join(lines)
    return prompt[: m.start(1)] + new_chunk + "\n"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/sft/minicpm-p15d.jsonl")
    ap.add_argument("--out", default="data/sft/trap-reinforce.jsonl")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    zh = [r for r in rows if r.get("lang") == "zh-TW" and not r["completion"].strip().startswith("NOP")]
    rng = random.Random(9)
    rng.shuffle(zh)
    out = []
    for r in zh:
        if len(out) >= args.n:
            break
        p = inject_trap(r["prompt"], rng)
        if p is None:
            continue
        nr = dict(r)
        nr["prompt"] = p
        nr["trap_aug"] = True
        out.append(nr)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} trap-reinforced rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
