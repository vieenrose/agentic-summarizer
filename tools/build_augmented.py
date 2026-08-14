"""Augmented meetings: inject the over-assertion counterfactual beats into REAL
transcripts, so the hard-class lessons sit in real context (the 3-line synth
meetings did not transfer to 500-line real ones — measured on T1).

The injected beats are generic-subject exchanges whose correct ops are NOP or
ADD OPEN — never DECISIONS/ACTIONS:
  soft-action   "we need to look into X" ... "no owner yet"
  either/or     "should we go with X or wait?" ... "not settled yet"
  supposed-to   "it was supposed to be X, but it is Y"  (the partial-quote trap)
"""
import json, sys, random
from pathlib import Path

sys.path.insert(0, "src")
from voxsum.transcript import parse_transcript, sec_to_clock
from voxsum.synth import build_meeting  # reuse _SUBJECTS/_TRAPS rotation via variant

BEATS_EN = {
    "soft": [
        "One more thing before we move on - we need to look into the quarterly report format.",
        "Right, that would help the planning.",
        "We will assign that next time. No owner yet.",
    ],
    "either": [
        "Should we go with the new office layout or wait?",
        "Both sides have good arguments.",
        "We have not settled that yet. Let us keep both options open.",
    ],
    "supposed": [
        "The report cover was supposed to be green, but it came out blue.",
        "I see. Let us check with the print shop.",
        "Yes, green or blue is still open.",
    ],
    "nofollow": [
        "We should email the slides to everyone.",
        "Oh, you don't have to email them, the folder has the slides.",
        "Right, no need to email anything then.",
    ],
    "intention": [
        "The intention is that everybody gets a copy of the notes.",
        "Right, that is the plan going forward.",
        "We will confirm when that starts.",
    ],
}

def inject(utterances, beats, at: int, speaker_map) -> list:
    """Insert 3 beat lines after index `at`, shifting the tail by 3*15s."""
    from voxsum.transcript import Utterance
    shift = len(beats) * 15
    out = []
    for i, u in enumerate(utterances):
        if i == at:
            base = u.start + 10
            for j, (spk, text) in enumerate(zip(speaker_map, beats)):
                out.append(Utterance(base + j * 15, spk, text))
        tail = u.start + (shift if i > at else 0)
        out.append(Utterance(tail, u.speaker, u.text))
    return out

def main() -> int:
    manifest = json.load(open("data/transcripts/manifest.json"))
    train_en = [r for r in manifest
                if r["split"] == "train" and r["lang"] == "en"
                and r["source"].split(":")[0] in ("qmsum", "meetingbank")
                and 200 <= r["n_lines"] <= 700 and r["authentic_speakers"]]
    random.Random(7).shuffle(train_en)
    existing = [m for m in manifest if m["meeting_id"].startswith("aug-en-")]
    base_idx = len(existing)
    pick = [r for r in train_en if r["meeting_id"] not in {e["parent"] for e in existing}][:8]
    if len(existing) >= 8:
        pick = [r for r in train_en if r["meeting_id"] not in {e["parent"] for e in existing}][:]
    added = 0
    out_manifest = []
    for i, r in enumerate(pick):
        mid = f"aug-en-{base_idx + i}"
        path = Path(f"data/transcripts/{r['file']}")
        utt = parse_transcript(path.read_text())
        speakers = list(dict.fromkeys(u.speaker for u in utt if u.speaker))
        speakers = speakers[:3] or ["S1", "S2", "S3"]
        # four injection points spread across the meeting
        a = len(utt) // 3
        b = int(len(utt) * 0.55)
        c = int(len(utt) * 0.8)
        u2 = inject(utt, BEATS_EN["soft"], a, speakers)
        u3 = inject(u2, BEATS_EN["either"], b, speakers)
        u4 = inject(u3, BEATS_EN["nofollow"], c, speakers)
        u5 = inject(u4, BEATS_EN["intention"], min(20, len(u4) // 4), speakers)
        u4 = u5
        # final monotonic renumber: strictly increasing starts, 15s minimum gap
        from voxsum.transcript import Utterance
        fixed = []
        prev = -15
        for u in u4:
            start = max(u.start, prev + 15)
            fixed.append(Utterance(start, u.speaker, u.text))
            prev = start
        out = Path(f"data/transcripts/{mid}.txt")
        out.write_text("".join(u.render() + "\n" for u in fixed))
        out_manifest.append({
            "meeting_id": mid, "source": "augmented", "lang": "en", "split": "train",
            "n_lines": len(fixed), "duration_sec": fixed[-1].start,
            "authentic_clock": False, "authentic_speakers": r["authentic_speakers"],
            "notes": ["real transcript + injected over-assertion counterfactual beats"],
            "tokens": r["tokens"], "file": f"{mid}.txt",
            "parent": r["meeting_id"],
        })
        added += 1
    manifest.extend(out_manifest)
    json.dump(manifest, open("data/transcripts/manifest.json", "w"), ensure_ascii=False, indent=1)
    print("augmented meetings:", added)
    return 0

if __name__ == "__main__":
    main()
