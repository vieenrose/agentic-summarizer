"""Round-6 training round: augment the zh corpus with the *real* ASR failure texture
measured on the podcast tier, so the student learns zh targets on code-switched,
garbled input (the train/deploy gap the tier exists to close).

Three augmentation classes, each mapped to a measured failure:
  1. Latin injection        -> the language flip (Cerebras 23.2% -> English)
  2. homophone/garble noise -> malformed ops + garble propagation (離岸封建)
  3. code-switch artifacts  -> simplified/traditional mixing (尽快/辦公室)

Surface-only: timestamps, speakers, and decision beats stay intact. The teacher
regenerates op targets (in Chinese) on the augmented transcripts.

Usage:
  python tools/build_real_asr_aug.py [--n-per-source 4]
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from voxsum.transcript import Utterance, parse_transcript

#: Real code-switch vocabulary extracted from the podcast tier (the actual terms
#: Taiwanese tech speech mixes in — the Cerebras episode's flip vocabulary).
LATIN_TERMS = {
    "人工智慧": "AI", "推論": "inference", "伺服器": "server", "晶片": "chip",
    "晶圓": "wafer", "記憶體": "memory", "模組": "module", "系統": "system",
    "平台": "platform", "每秒": "per second", "代幣": "token", "市場": "market",
    "客戶": "customer", "產品": "product", "再生能源": "renewable energy",
    "百分比": "percent", "資料中心": "datacenter", "雲端": "cloud",
    "軟體": "software", "硬體": "hardware", "半導體": "semiconductor",
    "封裝": "packaging", "測試": "test", "產能": "capacity", "訂單": "order",
    "供應": "supply", "需求": "demand", "成本": "cost", "價格": "price",
    "規格": "spec", "認證": "certification", "庫存": "inventory",
}

#: Real ASR garble pairs from the tier (homophone + simplified/traditional + the
#: specific garbles the model copied: 離岸封建 for 離岸風電, R一百 for RE100).
GARBLES = [
    ("風電", "封建"), ("客戶", "科特"), ("儘快", "尽快"), ("已", "以"),
    ("得", "的"), ("在", "再"), ("們", "門"), ("跟", "根"), ("對", "隊"),
    ("匯", "會"), ("計", "記"), ("RE100", "R一百"),
]

LATIN_RE = re.compile(r"[A-Za-z]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def latin_fraction(text: str) -> float:
    body = re.sub(r"\s", "", text)
    if not body:
        return 0.0
    return len(LATIN_RE.findall(body)) / len(body)


def inject_latin(text: str, target: float, rng: random.Random) -> str:
    """Occurrence-level greedy substitution to `target` Latin fraction (the probe
    that reproduces the flip)."""
    import heapq
    counts = {k: text.count(k) for k in LATIN_TERMS if text.count(k)}
    heap = [(-c, k) for k, c in counts.items()]
    heapq.heapify(heap)
    out = text
    while heap and latin_fraction(out) < target:
        _, k = heapq.heappop(heap)
        out = out.replace(k, LATIN_TERMS[k], 1)
        counts[k] -= 1
        if counts[k] > 0:
            heapq.heappush(heap, (-counts[k], k))
    return out


def inject_garble(text: str, rng: random.Random) -> str:
    for a, b in GARBLES:
        if a in text and rng.random() < 0.6:
            text = text.replace(a, b, 1)
    return text


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=3)
    ap.add_argument("--latin-targets", type=float, nargs="+", default=[0.16, 0.22])
    args = ap.parse_args()

    manifest = json.load(open("data/transcripts/manifest.json"))
    have = {r["meeting_id"] for r in manifest}
    zh_src = [r for r in manifest if r.get("lang") == "zh-TW" and r["split"] == "train"
              and not r["meeting_id"].startswith(("synth-zh-noisy", "synth-zh-aug"))]
    random.Random(7).shuffle(zh_src)
    added = 0
    for src in zh_src[:12]:
        utt = parse_transcript(Path(f"data/transcripts/{src['file']}").read_text())
        for variant, target in enumerate(args.latin_targets):
            rng = random.Random(1000 + src["meeting_id"].__hash__() % 1000 + variant)
            out = []
            for u in utt:
                t = inject_latin(u.text, target, rng)
                t = inject_garble(t, rng)
                out.append(Utterance(u.start, u.speaker, t))
            mid = f"synth-zh-rasr-{src['meeting_id'].split('-')[-1]}-{int(target*100)}"
            if mid in have:
                continue
            Path(f"data/transcripts/{mid}.txt").write_text(
                "".join(u.render() + "\n" for u in out))
            manifest.append({
                "meeting_id": mid, "source": "aug:real-asr", "lang": "zh-TW",
                "split": "train", "n_lines": len(out), "duration_sec": out[-1].start,
                "authentic_clock": False, "authentic_speakers": False,
                "latin_target": target,
            })
            added += 1
    json.dump(manifest, open("data/transcripts/manifest.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"added {added} real-ASR-augmented zh transcripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
