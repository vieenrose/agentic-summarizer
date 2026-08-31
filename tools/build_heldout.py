"""Carve a FRESH held-out eval set from MeetingBank meetings nothing has touched.

    python tools/build_heldout.py --n 20 --out data/heldout_en --seed 20260828

The 20-meeting eval set in `data/eval20_zh` has now been read six times — dropv2, dropv4,
dropv5, dropv6, chunk-1500, and the G1 probes. Every "PASS" claimed against it carries
that search history, and `sft-dropv6`'s 16/20 rouge1 in particular was reached after
choosing `sys-v2` partly on evidence from the same 20 meetings. It is no longer a clean
instrument, and there is nothing else held out: SPEC §9's Phase-1 split reserved exactly
one eval slice and the project has spent it.

This selects from the **1,000 annotated meetings that appear in neither `data/pilot`'s
manifest nor Phase 4's `data/p4_zh`**, so the result is untouched by any training pool or
any prior measurement.

Selection is a plain seeded random draw, NOT stratified to resemble the old eval set.
Matching the old set's length profile would import the very selection choices whose
influence this set exists to escape — and meeting length is exactly the variable the last
four builds were tuned around, so anchoring on it would be the worst possible choice.

Output is ENGLISH format-v2, the input to §2.2 stage 2 (translation). Nothing here can be
scored yet: zh-TW text and composed references are separate, expensive stages.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.corpus.meetingbank import import_meeting  # noqa: E402

ZIP = REPO / "data/raw/MeetingBank.zip"
META = "Metadata/MeetingBank.json"


def used_meeting_ids() -> set[str]:
    used: set[str] = set()
    man = REPO / "data/pilot/manifest.json"
    if man.exists():
        used |= {r["meeting_id"] for r in json.loads(man.read_text(encoding="utf-8"))}
    for d in ("data/p4_zh", "data/p4_en50", "data/eval20_zh"):
        p = REPO / d
        if p.is_dir():
            used |= {f.stem for f in p.glob("*.txt")}
    return used


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260828)
    args = p.parse_args(argv)

    z = zipfile.ZipFile(ZIP)
    meta = json.loads(z.read(META))
    used = used_meeting_ids()

    # Map clean meeting id -> transcript member. The transcript FILENAMES are opaque
    # (`full_100417V.mp3.transcript.json`, `longbeach_<uuid>.mp3.transcript.json`) and
    # carry no meeting id, so they cannot be matched by name — each metadata entry names
    # its own file in `Transcripts`. The zip holds 1,366 transcripts but only 1,250
    # meetings have `itemInfo`; a meeting with no gold items cannot get a composed
    # reference later, so it is not a candidate.
    by_basename = {
        Path(n).name: n
        for n in z.namelist()
        if "/transcripts/" in n and n.endswith(".transcript.json")
    }
    members = {}
    for mid, entry in meta.items():
        fname = entry.get("Transcripts")
        if isinstance(fname, str) and fname in by_basename:
            members[mid] = by_basename[fname]

    candidates = sorted(set(members) - used)
    print(
        f"[heldout] {len(meta)} annotated, {len(members)} transcripts, {len(used)} already "
        f"used -> {len(candidates)} candidates",
        file=sys.stderr,
    )
    if len(candidates) < args.n:
        print(f"[heldout] REFUSED: only {len(candidates)} candidates", file=sys.stderr)
        return 1

    picked = random.Random(args.seed).sample(candidates, args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for mid in sorted(picked):
        utterances = import_meeting(json.loads(z.read(members[mid])))
        (args.out / f"{mid}.txt").write_text(
            "\n".join(u.render() for u in utterances) + "\n", encoding="utf-8"
        )
        manifest.append(
            {
                "meeting_id": mid,
                "split": "heldout",
                "utterances": len(utterances),
                "items": len(meta[mid].get("itemInfo") or {}),
                "translated_by": None,
                "composed_by": None,
                "human_validated": False,
            }
        )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    tot = sum(m["utterances"] for m in manifest)
    print(
        f"[heldout] wrote {len(manifest)} meetings ({tot} utterances, "
        f"{sum(m['items'] for m in manifest)} gold items) -> {args.out}",
        file=sys.stderr,
    )
    print(f"[heldout] seed={args.seed}; selection is reproducible from this seed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
