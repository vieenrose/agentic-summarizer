#!/usr/bin/env python3
"""Harvest sweep DROP corrections as negative SFT signal — v2 (per-step states).

v1 could only harvest bullets that appeared in the TEACHER's trace targets (48/109);
the 53 "no_record" drops were the student's pure hallucinations with no teacher trace.
v2 uses the student's OWN per-step record (state_before + chunk + applied ops) to
reconstruct a negative sample for every dropped bullet the student actually emitted:

  user   = the exact STATE+CHUNK prompt of the step that emitted the bullet
  target = that step's APPLIED ops minus the op carrying the dropped bullet

Same hygiene as v1: train meetings only; the stale-state class (bullets a later step
UPD/DELs) is excluded — those were true at emission time and need revision, not
suppression.

    python tools/harvest_negatives.py --out data/sft/lfm-en-negatives2.jsonl
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
from judge import faith_prompt, TogetherJudge  # noqa: E402
from voxsum.agent import run_cursor  # noqa: E402
from voxsum.backends.llama_server import LlamaServer  # noqa: E402
from voxsum.ops import Del, Malformed, Nop, Upd, render_op  # noqa: E402
from voxsum.sweep import run_sweep  # noqa: E402
from voxsum.transcript import parse_transcript  # noqa: E402

REAL_SOURCES = ("qmsum", "meetingbank")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
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

    student = LlamaServer(base_url=args.student_url, max_tokens=512, temperature=0.0)
    judge_client = TogetherJudge(api_key="local", budget_usd=10, max_tokens=14000)
    judge = lambda s, u: judge_client(args.judge, s, u)

    negatives: list[dict] = []
    stats = {"meetings": 0, "dropped": 0, "harvested": 0, "stale_skipped": 0,
             "not_emitted": 0, "nothing_left": 0}
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
        for bullet in dropped:
            needle = bullet.casefold()
            # Earliest step whose raw EMITS the bullet with it APPLIED.
            hit = None
            for step in trace.steps:
                if needle not in step.raw.casefold():
                    continue
                applied = [r.op for r in step.outcome.results
                           if r.applied and not isinstance(r.op, Malformed)]
                if any(needle in str(o).casefold() for o in applied):
                    hit = (step, applied)
                    break
            if hit is None:
                stats["not_emitted"] += 1
                continue
            step, applied = hit
            # Stale-state class: a later step revises/drops this bullet.
            later_revises = False
            for later in trace.steps:
                if later.index <= step.index:
                    continue
                if any(isinstance(o, (Upd, Del)) and needle in str(o).casefold()
                       for o in later.ops):
                    later_revises = True
                    break
            if later_revises:
                stats["stale_skipped"] += 1
                continue
            kept = [render_op(o) for o in applied if needle not in str(o).casefold()]
            if not kept:
                stats["nothing_left"] += 1
                continue
            rec = {
                "meeting": f"{mid}~neg",
                "lang": "en",
                "step": step.index,
                "prompt_version": "sys-v1",
                "system": step.system,
                "user": step.user,
                "target": "\n".join(kept),
                "is_nop": False,
                "negative_of": bullet[:80],
            }
            sample = build_sample(rec)
            if sample["completion"]:
                negatives.append(sample)
                stats["harvested"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for s in negatives:
            sink.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[neg2] {stats}")
    print(f"[neg2] {len(negatives)} negative samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
