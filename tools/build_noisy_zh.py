"""Noisy-zh augmentation: ASR-style noise injected into zh synth meetings so the
student learns to emit ops on noisy input (the maintainer's real zh-TW procurement
meeting produced 1.25 ops/chunk — the zh pool is clean synth, so noise suppresses
op emission).

Noise classes (surface-only; decision beats stay intact):
  1. filler insertion  嗯 / 啊 / 那個 / 就是 / 然後
  2. disfluency        repeat a word fragment ("這個、這個方案")
  3. homophone error   common ASR confusions (的/得, 在/再, 他/她, 那/哪)
"""
import json, random, re, sys
from pathlib import Path

sys.path.insert(0, "src")
from voxsum.transcript import parse_transcript

FILLERS = ["嗯", "啊", "那個", "就是", "然後", "哦"]
HOMOPHONES = [("的", "得"), ("得", "的"), ("在", "再"), ("再", "在"),
              ("他", "她"), ("她", "他"), ("那", "哪"), ("們", "門"),
              ("已", "以"), ("以", "已"), ("跟", "根"), ("對", "隊")]

def noisy(text: str, rng: random.Random) -> str:
    # 1. filler insertion at a random boundary
    if rng.random() < 0.35:
        if len(text) >= 4:
            i = rng.randrange(1, len(text))
            text = text[:i] + rng.choice(FILLERS) + "，" + text[i:]
    # 2. disfluency: repeat the first 2 chars
    if rng.random() < 0.18 and len(text) >= 2:
        text = text[:2] + "、" + text
    # 3. homophone error
    if rng.random() < 0.25:
        for a, b in HOMOPHONES:
            if a in text:
                text = text.replace(a, b, 1)
                break
    return text

def main() -> int:
    manifest = json.load(open("data/transcripts/manifest.json"))
    have = {r["meeting_id"] for r in manifest}
    zh_combined = [r for r in manifest
                   if r["meeting_id"].startswith("synth-zh-combined")]
    random.Random(11).shuffle(zh_combined)
    added = 0
    for i, r in enumerate(zh_combined[:10]):
        mid = f"synth-zh-noisy-{i}"
        if mid in have:
            continue
        utt = parse_transcript(Path(f"data/transcripts/{r['file']}").read_text())
        rng = random.Random(100 + i)
        out = []
        for u in utt:
            from voxsum.transcript import Utterance
            out.append(Utterance(u.start, u.speaker, noisy(u.text, rng)))
        Path(f"data/transcripts/{mid}.txt").write_text("".join(u.render() + "\n" for u in out))
        manifest.append({
            "meeting_id": mid, "source": "synth:noisy", "lang": "zh-TW", "split": "train",
            "n_lines": len(out), "duration_sec": out[-1].start,
            "authentic_clock": False, "authentic_speakers": False,
            "notes": [f"ASR-noise augmentation of {r['meeting_id']} (fillers/disfluency/homophones)"],
            "tokens": r["tokens"], "file": f"{mid}.txt", "parent": r["meeting_id"],
        })
        added += 1
    json.dump(manifest, open("data/transcripts/manifest.json", "w"), ensure_ascii=False, indent=1)
    print("noisy zh meetings:", added)
    return 0

if __name__ == "__main__":
    main()
