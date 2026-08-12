#!/usr/bin/env python3
"""Harvest sweep DROP corrections as negative SFT signal (the committed next step).

The sweep drops bullets the whole-transcript judge cannot verify (fabrications,
request-as-decision). For each dropped bullet we reconstruct the earliest trace step
where it was emitted, and build a negative sample: SAME state+chunk, target WITHOUT the
wrong bullet. This teaches the student not to emit unverifiable content — converting
judge-time corrections into training signal.

Two hygiene rules:
* train meetings only (the eval tiers stay held out);
* the stale-state class is excluded: bullets that a LATER step UPD/DELs were true at
  emission time and needed revision, not suppression — those are already taught by the
  UPD demonstrations.

    python tools/harvest_negatives.py --out data/sft/lfm-en-negatives.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

from build_sft_qwen import build_sample  # noqa: E402 (text grammar)
from judge import _FAITH_SYS, faith_prompt, TogetherJudge  # noqa: E402
from voxsum.agent import run_cursor  # noqa: E402
from voxsum.backends.llama_server import LlamaServer  # noqa: E402
from voxsum.ops import Del, Nop, Upd, parse_ops  # noqa: E402
from voxsum.sweep import run_sweep  # noqa: E402
from voxsum.transcript import parse_transcript  # noqa: E402

REAL_SOURCES = ("qmsum", "meetingbank")


def load_trace_records(tracedir: Path) -> dict[str, list[dict]]:
    """meeting stem -> records (split-filtered to train already at build time)."""
    out: dict[str, list[dict]] = {}
    for path in sorted(tracedir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.setdefault(r["meeting"], []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r["chunk_start"])
    return out


def target_lines(target: str) -> list[str]:
    return [l for l in target.splitlines() if l.strip()]


def op_mentions(line: str, text: str) -> bool:
    """Does this target line carry the bullet (ADD/Upd/DEL)?"""
    for op in parse_ops(line):
        if isinstance(op, Nop):
            continue
        body = getattr(op, "bullet", getattr(op, "prefix", "")) or ""
        if body and text.casefold() in body.casefold():
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracedir", type=Path, default=Path("data/traces_v2"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--student-url", default="http://127.0.0.1:8093")
    p.add_argument("--judge", default="local:8090/gpt-oss-20b")
    p.add_argument("--max-meetings", type=int, default=0)
    args = p.parse_args(argv)

    manifest = json.loads(
        (Path(__file__).resolve().parent.parent / "data/transcripts/manifest.json").read_text()
    )
    meetings = [
        r for r in manifest
        if r["split"] == "train" and r["lang"] == "en"
        and r["source"].split(":")[0] in REAL_SOURCES
    ]
    meetings.sort(key=lambda r: -r["tokens"])
    if args.max_meetings:
        meetings = meetings[: args.max_meetings]

    records = load_trace_records(args.tracedir)
    student = LlamaServer(base_url=args.student_url, max_tokens=512, temperature=0.0)
    judge_client = TogetherJudge(api_key="local", budget_usd=10, max_tokens=14000)
    judge = lambda s, u: judge_client(args.judge, s, u)

    negatives: list[dict] = []
    stats = {"meetings": 0, "dropped": 0, "harvested": 0, "stale_skipped": 0, "no_record": 0}
    for m in meetings:
        mid = m["meeting_id"]
        utt = parse_transcript((Path("data/transcripts") / m["file"]).read_text())
        trace = run_cursor(utt, student, lang="en", declarations=False, budget=2048)
        sweep = run_sweep(
            trace.state, utt, judge,
            verify=True, anchor=False, budget=60,
            prompt_builder=lambda b, ev: faith_prompt(b, ev),
            apply_fix=False,
        )
        stats["meetings"] += 1
        dropped = [v.bullet for v in sweep.verified if v.verdict in ("DROP", "FIX")]
        stats["dropped"] += len(dropped)
        rows = records.get(mid, [])
        for bullet in dropped:
            # Earliest step emitting the bullet (the ADD step).
            hit = next(
                (r for r in rows if any(op_mentions(l, bullet) for l in target_lines(r["target"]))),
                None,
            )
            if hit is None:
                stats["no_record"] += 1
                continue
            # Stale-state class: a LATER step revises/drops this bullet -> skip.
            later = [
                r for r in rows if r["chunk_start"] > hit["chunk_start"]
                and any(
                    isinstance(o, (Upd, Del)) and bullet.casefold() in str(o).casefold()
                    for o in parse_ops(r["target"])
                )
            ]
            if later:
                stats["stale_skipped"] += 1
                continue
            kept = [
                l for l in target_lines(hit["target"]) if not op_mentions(l, bullet)
            ]
            if not kept:
                stats["stale_skipped"] += 1  # nothing left: bullet WAS the step's content
                continue
            rec = dict(hit)
            rec["target"] = "\n".join(kept)
            rec["meeting"] = f"{mid}~neg"
            rec["negative_of"] = bullet[:80]
            sample = build_sample(rec)
            if sample["completion"]:
                negatives.append(sample)
                stats["harvested"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for s in negatives:
            sink.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[neg] {stats}")
    print(f"[neg] {len(negatives)} negative samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
