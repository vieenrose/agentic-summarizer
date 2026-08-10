#!/usr/bin/env python3
"""Materialise transcripts + manifest into `data/` (PLAN.md §1a).

Sources, in the order §1a prefers them:

1. **audio** — real clock and real speakers, via `tools/transcribe_moss.py`. Not handled
   here; drop the resulting v1 files in `--out` and re-run with `--manifest-only`.
2. **synthetic revision-dense** (`voxsum.synth`) — the UPD/DEL demonstrations natural
   meetings do not supply, zh-TW oversampled per RESULTS.md.
3. **public corpora** (QMSum, MeetingBank) — real speech, **synthesised clock**. Fine for
   training; not a basis for reporting FAITH-anchor as real-world accuracy.

Every meeting lands in `manifest.json` with `authentic_clock` / `authentic_speakers`, so a
later reader cannot mistake a synthesised clock for a measured one.

    python tools/prepare_data.py --out data/transcripts --qmsum 40 --meetingbank 40
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.chunker import heuristic_token_len  # noqa: E402
from voxsum.corpora import MeetingRecord, meetingbank_record, qmsum_record  # noqa: E402
from voxsum.synth import build_set  # noqa: E402
from voxsum.transcript import parse_transcript  # noqa: E402

HUB = Path.home() / ".cache/huggingface/hub"


def _snapshot(dataset: str) -> Path | None:
    hits = sorted(glob.glob(str(HUB / f"datasets--{dataset}/snapshots/*")))
    return Path(hits[0]) if hits else None


def load_qmsum(limit: int, split: str = "validation") -> list[MeetingRecord]:
    root = _snapshot("pszemraj--qmsum-cleaned")
    if root is None:
        print("[data] qmsum not in the HF cache; skipping", file=sys.stderr)
        return []
    import hashlib

    import pyarrow.parquet as pq

    table = pq.read_table(str(root / f"data/{split}-00000-of-00001.parquet"))
    inputs = table.column("input").to_pylist()

    # QMSum ships one row per *query* over the same meeting — 272 rows carry only 35
    # distinct transcripts in the validation split, and the `id` field does not identify
    # the meeting (`va-sq-1`, `va-sq-2` … are different queries on the same recording).
    # Dedupe on the transcript body, or the same meeting enters the trace set dozens of
    # times and the student trains on near-duplicates.
    out: list[MeetingRecord] = []
    seen: set[str] = set()
    for raw in inputs:
        body = raw.split("\n", 1)[1] if "\n" in raw else raw
        digest = hashlib.sha1(body.encode()).hexdigest()[:12]
        if digest in seen:
            continue
        seen.add(digest)
        rec = qmsum_record(f"qmsum-{digest}", raw, split="train")
        if rec.n_lines >= 20:
            out.append(rec)
        if len(out) >= limit:
            break
    return out


def load_meetingbank(limit: int, split: str = "validation") -> list[MeetingRecord]:
    root = _snapshot("huuuyeah--meetingbank")
    if root is None:
        print("[data] meetingbank not in the HF cache; skipping", file=sys.stderr)
        return []
    out: list[MeetingRecord] = []
    with (root / f"{split}.json").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rec = meetingbank_record(f"mbank-{row['uid']}", row["transcript"], split="train")
            if rec.n_lines >= 20:
                out.append(rec)
            if len(out) >= limit:
                break
    return out


def load_synth() -> list[MeetingRecord]:
    out: list[MeetingRecord] = []
    for m in build_set():
        out.append(
            MeetingRecord(
                meeting_id=m.meeting_id,
                source=f"synth:{m.kind}",
                lang=m.lang,
                utterances=m.utterances,
                authentic_clock=False,
                authentic_speakers=True,
                split="train",
                notes=(
                    f"synthetic revision-dense meeting; expected op {m.expected_op}",
                    f"setup at {m.setup_at}s, revision at {m.revision_at}s",
                ),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data/transcripts"))
    p.add_argument("--qmsum", type=int, default=0, help="max QMSum meetings")
    p.add_argument("--meetingbank", type=int, default=0, help="max MeetingBank segments")
    p.add_argument("--no-synth", action="store_true")
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="re-scan --out (e.g. after adding MOSS transcripts) and rewrite the manifest",
    )
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[MeetingRecord] = []
    if not args.manifest_only:
        if not args.no_synth:
            records += load_synth()
        if args.qmsum:
            records += load_qmsum(args.qmsum)
        if args.meetingbank:
            records += load_meetingbank(args.meetingbank)

        for rec in records:
            text = rec.render()
            parse_transcript(text)  # vet: refuse to write anything that is not v1
            (args.out / f"{rec.meeting_id}.txt").write_text(text, encoding="utf-8")

    entries = []
    for path in sorted(args.out.glob("*.txt")):
        by_id = {r.meeting_id: r for r in records}
        rec = by_id.get(path.stem)
        text = path.read_text(encoding="utf-8")
        utterances = parse_transcript(text)
        entry = rec.manifest() if rec else {
            "meeting_id": path.stem,
            "source": "unknown (found on disk)",
            "lang": "?",
            "split": "?",
            "n_lines": len(utterances),
            "duration_sec": utterances[-1].start if utterances else 0,
            "authentic_clock": None,
            "authentic_speakers": None,
            "notes": ["provenance unrecorded — set it before using in any reported eval"],
        }
        entry["tokens"] = heuristic_token_len(text)
        entry["file"] = path.name
        entries.append(entry)

    (args.out / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_source: dict[str, int] = {}
    for e in entries:
        by_source[str(e["source"]).split(":")[0]] = by_source.get(
            str(e["source"]).split(":")[0], 0
        ) + 1
    total_tokens = sum(int(e["tokens"]) for e in entries)
    authentic = sum(1 for e in entries if e.get("authentic_clock"))
    print(f"[data] {len(entries)} meetings, {total_tokens:,} tokens -> {args.out}")
    for source, n in sorted(by_source.items()):
        print(f"[data]   {source:12s} {n:4d}")
    print(f"[data] authentic clock: {authentic}/{len(entries)} — the rest are synthesised")
    if authentic == 0:
        print(
            "[data] NOTE: no meeting has a real clock. FAITH-anchor on this set measures "
            "self-consistency, not real-world anchor accuracy (§1a).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
