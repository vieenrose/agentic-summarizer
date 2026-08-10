#!/usr/bin/env python3
"""Expand the synthetic pool and carve eval tiers (T1/micro) out of the manifest.

The zh pool is only 16 meetings — holding out 10 for T1 + 3 for micro would leave 3 for
training. Generate 8 more en + 8 more zh revision-dense meetings (variants v2/v3 en,
v4/v5 zh), then mark eval meetings with a `split` field:

* t1   — 10 en QMSum (real, 12k-40k tok) + 10 zh synth (held out; VCSum unobtainable)
* micro — 3 en MeetingBank (small = cheap iteration) + 3 zh synth
* everything else stays train

The two qmsum meetings already baselined in runs/arms (n=2 pair) stay in T1 for
continuity with earlier measured numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.chunker import heuristic_token_len  # noqa: E402
from voxsum.synth import REVISION_KINDS, build_set  # noqa: E402
from voxsum.transcript import parse_transcript  # noqa: E402

OUT = Path("data/transcripts")
MANIFEST = OUT / "manifest.json"

T1_EN = [
    "qmsum-16abbdf7b3f2",   # already baselined (runs/arms)
    "qmsum-3f8b473ddd36",   # already baselined (runs/arms)
    "qmsum-46afb4f2ef60", "qmsum-4bfcff6d8771", "qmsum-6825a6ef4300",
    "qmsum-76d19929893a", "qmsum-8ac3acb7fe5e", "qmsum-a001c3a20024",
    "qmsum-bdb39cc06654", "qmsum-e75802cbf8d3",
]
T1_ZH = [f"synth-zh-{k}-{v}" for k in REVISION_KINDS for v in (0, 1)] + [
    "synth-zh-reversal-4", "synth-zh-reversal-5"
]
MICRO_EN = [
    "mbank-LongBeachCC_07192016_16-0661",   # 386 tok — cheapest possible iteration
    "mbank-LongBeachCC_09192017_17-0808",   # 570 tok
    "mbank-DenverCityCouncil_03142016_16-0110",  # 522 tok
]
MICRO_ZH = ["synth-zh-reassign-4", "synth-zh-withdraw-4", "synth-zh-withdraw-5"]

def main() -> int:
    # 1. Generate the new synthetic variants (v2/v3 en, v4/v5 zh).
    new_meetings = []
    for m in build_set(en_per_kind=4, zh_per_kind=6, chunk_budget=2048):
        v = int(m.meeting_id.rsplit("-", 1)[1])
        is_new = (m.lang == "en" and v >= 2) or (m.lang == "zh-TW" and v >= 4)
        if is_new:
            new_meetings.append(m)
    print(f"[data] {len(new_meetings)} new synthetic meetings")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    have = {x["meeting_id"] for x in manifest}
    for m in new_meetings:
        if m.meeting_id in have:
            continue
        path = OUT / f"{m.meeting_id}.txt"
        path.write_text(m.render(), encoding="utf-8")
        utt = parse_transcript(path.read_text(encoding="utf-8"))
        tok = sum(heuristic_token_len(u.text) for u in utt)
        manifest.append({
            "meeting_id": m.meeting_id,
            "source": f"synth:{m.kind}",
            "lang": m.lang,
            "split": "train",
            "n_lines": len(utt),
            "duration_sec": utt[-1].start,
            "authentic_clock": False,
            "authentic_speakers": True,
            "notes": [f"synthetic revision-dense meeting; expected op {m.expected_op}"],
            "tokens": tok,
            "file": path.name,
        })

    # 2. Carve the eval tiers (reset first: re-running must converge, not accumulate).
    split_of = {}
    for mid in T1_EN + T1_ZH:
        split_of[mid] = "t1"
    for mid in MICRO_EN + MICRO_ZH:
        split_of[mid] = "micro"
    changed = 0
    for row in manifest:
        if row["split"] in ("t1", "micro") and row["meeting_id"] not in split_of:
            row["split"] = "train"
            changed += 1
        if row["meeting_id"] in split_of:
            new = split_of[row["meeting_id"]]
            if row.get("split") != new:
                row["split"] = new
                changed += 1
    missing = [m for m in split_of if m not in {x["meeting_id"] for x in manifest}]
    if missing:
        print("WARNING: eval meetings missing from manifest:", missing)

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"[data] split changed for {changed} meetings; splits now:",
          dict(Counter(x["split"] for x in manifest)))
    return 0 if not missing else 1

if __name__ == "__main__":
    raise SystemExit(main())
