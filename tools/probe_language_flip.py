"""Round 6 / the loanword hypothesis: at what Latin-character fraction does a zh
transcript flip the model's output language to English?

The VoxSum author's finding (real podcasts, uncontaminated): 22.3% Latin -> English
output; 4.9% and 3.5% -> Chinese. Their suggested probe, implemented here: take a zh
transcript, progressively substitute English technical terms, and measure the output
language of the resulting NOTES at each fraction.

Usage:
  python tools/probe_language_flip.py --base-url http://127.0.0.1:8098 \
      --transcript data/transcripts/meeting-zh-long.txt --lines 40
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.agent import run_cursor
from voxsum.backends.llama_server import LlamaServer
from voxsum.render import render_state
from voxsum.transcript import parse_transcript

#: zh -> English technical-term substitutions (loanwords as spoken in Taiwanese
#: tech meetings). Ordered by usefulness; greedy substitution by frequency.
ZH_EN = {
    "記憶體": "memory", "伺服器": "server", "晶圓": "wafer", "封裝": "packaging",
    "測試": "test", "客戶": "customer", "產品": "product", "價格": "price",
    "供應": "supply", "需求": "demand", "產能": "capacity", "訂單": "order",
    "市場": "market", "公司": "company", "工廠": "fab", "設備": "equipment",
    "技術": "technology", "資料": "data", "系統": "system", "平台": "platform",
    "晶片": "chip", "庫存": "inventory", "出貨": "shipment", "成本": "cost",
    "業務": "business", "報告": "report", "製造": "manufacture", "銷售": "sales",
    "標準": "standard", "規格": "spec", "樣品": "sample", "認證": "certification",
    "人工智慧": "AI", " inference": "inference", "資料中心": "datacenter",
    "人工": "manual", "智慧": "smart", "電腦": "computer", "軟體": "software",
    "硬體": "hardware", "網路": "network", "雲端": "cloud",
}

LATIN = re.compile(r"[A-Za-z]")
HAN = re.compile(r"[\u4e00-\u9fff]")


def latin_fraction(text: str) -> float:
    body = re.sub(r"\s", "", text)
    if not body:
        return 0.0
    return len(LATIN.findall(body)) / len(body)


def substitute(text: str, target: float) -> str:
    """Occurrence-level greedy substitution: replace the next most frequent
    remaining term-occurrence until `target` Latin fraction — smooth, and
    nested (variant at t1 is a prefix of variant at t2 > t1)."""
    import heapq

    occs = []  # (-count, term) heap — always substitute from the most frequent term
    counts = {k: text.count(k) for k in ZH_EN if text.count(k)}
    heap = [(-c, k) for k, c in counts.items()]
    heapq.heapify(heap)
    out = text
    while heap and latin_fraction(out) < target:
        _, k = heapq.heappop(heap)
        out = out.replace(k, ZH_EN[k], 1)  # one occurrence at a time
        counts[k] -= 1
        if counts[k] > 0:
            heapq.heappush(heap, (-counts[k], k))
    return out


def output_han_fraction(notes: str) -> float:
    bullets = [l for l in notes.splitlines() if l.startswith("- ")]
    body = "".join(b[2:].split("[")[0] for b in bullets)  # strip anchors
    han, latin = len(HAN.findall(body)), len(LATIN.findall(body))
    return han / (han + latin) if (han + latin) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8098")
    ap.add_argument("--transcript", type=Path,
                    default=Path("data/transcripts/meeting-zh-long.txt"))
    ap.add_argument("--lines", type=int, default=40)
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[6, 10, 14, 18, 22])
    ap.add_argument("--lang", default="zh-TW")
    args = ap.parse_args()

    lines = args.transcript.read_text(encoding="utf-8").splitlines()[: args.lines]
    model = LlamaServer(base_url=args.base_url, thinking=False,
                        send_thinking_kwarg=False, max_tokens=512)

    print(f"{'Latin% in':>10} {'actual':>7} {'Han% out':>9}  language")
    for target in args.targets:
        variant = substitute("\n".join(lines), target / 100.0)
        frac = latin_fraction("\n".join(l.split("] ", 1)[-1] for l in variant.splitlines()))
        utt = parse_transcript(variant)
        trace = run_cursor(utt, model, lang="zh-TW", budget=2048)
        notes = render_state(trace.state)
        han = output_han_fraction(notes)
        lang = "zh" if han >= 0.5 else "ENGLISH (flip)"
        print(f"{target:>9.0f}% {frac*100:>6.1f}% {han*100:>8.1f}%  {lang}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
