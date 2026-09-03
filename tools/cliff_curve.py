"""Does synthesis output length COLLAPSE as memory occupancy rises? (`SYNTHESIZE`, §4.1)

    python tools/cliff_curve.py --url http://127.0.0.1:8081 --label mixed-e3

**The failure this measures was found by reading a user's debug log, not by any gate.** A
real meeting left 15 points in memory and got a 116-character summary that mentioned none
of them. Sweeping occupancy against a FIXED point pool showed why: output length rose with
memory up to 12 points and fell off a cliff at 13.

    points   2     6     12    13    14    15
    v5       102   189   544   116   77    88

Past the cliff the model does not merely shorten -- it begins asserting content that is in
no point, so the failure is unfaithful as well as brief.

Diagnosed as EXPOSURE BIAS: every synthesis training row paired a TEACHER-authored memory
with its summary, while inference always presents a STUDENT-authored one. High-occupancy
student memories were simply off-distribution. `runs/selfdistil-e3` confirmed it -- swapping
the memories for student-authored ones, changing no targets and adding no rows, removed the
cliff entirely.

**Why this is a standalone tool and not part of an end-to-end run.** Driving `run_agent`
over real transcripts cannot isolate this: occupancy is whatever the reading step happens
to produce, so a checkpoint that reads less simply never reaches the cliff and looks fixed.
Holding ONE point pool constant and truncating it to each length is what makes the curve
attributable to synthesis alone. Points are taken from the pool's head, so every shorter
memory is a strict prefix of every longer one.

**WHICH pool decides whether the cliff appears at all — measured, 2026-09-02.** `POOL`
below is hand-written and well-formed, and on it `v5` shows NO cliff (237 / 1711 / 454 /
348 / 562 / 632 at 2/6/12/13/14/15). That is not a refutation of the cliff; it is the
exposure-bias diagnosis restating itself. A clean pool is TEACHER-shaped, which is the
distribution every synthesis training row came from, so the model is in-distribution and
copes. Real student memories are not clean: the extracted `ivod-17684` pool carries a
point truncated mid-phrase (`門檻高且未解`) and two near-duplicate pairs.

So `--pool-file` is the load-bearing path and `POOL` is only a smoke default. Pass a pool
dumped from a real `run_agent` memory (`{"arc": str, "points": [str]}`) for any curve you
intend to act on, and say which pool any reported number came from -- curves from
different pools are not comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.agent import synthesize_memory  # noqa: E402
from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.memory import Memory  # noqa: E402

ARC = "市議會審議年度預算案與多項市政提案，並就社區服務與公共安全議題進行討論。"

#: One fixed pool, ordered so that any prefix reads as a coherent meeting. Written to
#: resemble what the reading step actually emits (short, resolution-shaped, zh-TW) rather
#: than idealised points, so the curve reflects deployment conditions.
POOL = [
    "市議會通過 2019 年度總預算案，總額為 12 億元",
    "教育局提案增設三所公立幼兒園，預計 2020 年啟用",
    "交通局報告捷運藍線延伸工程進度落後六個月",
    "議員質詢市府對街友安置方案的執行成效",
    "衛生局宣布社區篩檢站將增加至十二處",
    "市府同意撥款 3000 萬元修繕老舊公有市場",
    "警察局提出加強夜間巡邏的人力配置計畫",
    "環保局說明垃圾焚化爐改建的環評時程",
    "議會決議將公園綠地維護預算提高百分之十五",
    "社會局報告長照服務使用人數較去年成長兩成",
    "工務局承諾在雨季前完成八條排水溝清淤",
    "議員要求市府公開招標文件以提升透明度",
    "文化局提案將舊火車站列為市定古蹟",
    "財政局說明地方稅收短徵的因應措施",
    "市議會通過設置青年創業貸款專案基金",
    "消防局報告新購兩輛雲梯車的交付時程",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--points", type=int, nargs="+", default=[2, 6, 12, 13, 14, 15])
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--label", default="")
    p.add_argument("--pool-file", type=Path, default=None,
                   help='{"arc": str, "points": [str]} dumped from a real run_agent memory')
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    arc, pool, pool_name = ARC, POOL, "builtin-clean"
    if args.pool_file:
        blob = json.loads(args.pool_file.read_text(encoding="utf-8"))
        arc, pool = blob["arc"], blob["points"]
        pool_name = blob.get("source", args.pool_file.stem)
    print(f"[cliff] pool={pool_name} ({len(pool)} points)", file=sys.stderr)

    rows = []
    for n in args.points:
        if n > len(pool):
            print(f"[cliff] REFUSED: n={n} exceeds the {len(pool)}-point pool "
                  f"'{pool_name}'", file=sys.stderr)
            return 1
        lens = []
        for seed in range(args.seeds):
            # A fresh client per seed: `cache_prompt: false` is mandatory here (trap 4 --
            # the prompt cache changes generation, and each occupancy shares a long prefix
            # with the last, which is exactly when the cache would contaminate the curve).
            model = LlamaServer(base_url=args.url, max_tokens=1200, repeat_penalty=1.1,
                                seed=seed, raw_completion=True,
                                extra={"cache_prompt": False})
            mem = Memory(arc=arc)
            for text in pool[:n]:
                mem.add_point(text, 0)
            syn = synthesize_memory(mem, model)
            lens.append(syn.prose.chars)
        rows.append({"points": n, "chars": lens,
                     "mean": round(sum(lens) / len(lens), 1)})
        print(f"[cliff] {n:>2} points -> {rows[-1]['mean']:.0f} chars {lens}",
              file=sys.stderr)

    summary = {"label": args.label, "url": args.url, "pool": pool_name,
               "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
               "seeds": args.seeds, "rows": rows}
    print(f"\n[cliff] {args.label}: " +
          " ".join(f"{r['points']}={r['mean']:.0f}" for r in rows), file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"[cliff] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
