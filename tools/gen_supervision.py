"""Generate gold per-step supervision for a translated corpus (SPEC §4.2 steps 1-3).

    python tools/gen_supervision.py --corpus data/p4_zh --offsets data/p4_offsets \\
        --items data/p4_items_zh.json --out data/p4_supervision --urls 8083,8084

This is the GOLD path, not self-distillation: `supervision.teacher` walks each meeting's
chunks in order and asks the teacher to convert the aligned segment minutes into edit
lines, so the targets are grounded in MeetingBank's own item summaries rather than in
whatever a model happened to say. `cli/gen_traces.py` runs the agent loop instead and is
a different tool for a different job.

**Every trace is replayed through the real harness before it is written** (SPEC §4.2:
"every gold edit sequence must replay cleanly"). A step whose ops do not apply cleanly
is not silently kept -- it is counted and reported, because a target the harness would
refuse is a target that teaches the model to emit something the harness refuses.

`trace.synthesis` is deliberately left unset by `generate_meeting_supervision`: §4.2
step 4's synthesis target is the human-validated composed summary, which is a separate
artifact. These meetings therefore contribute STEP supervision only, which is exactly
what the long-meeting deficit needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.supervision.align import Item  # noqa: E402
from arcsum.supervision.sft import build_samples  # noqa: E402
from arcsum.supervision.teacher import (  # noqa: E402
    generate_meeting_supervision,
    replay_step_cleanly,
)
from arcsum.transcript import parse_transcript  # noqa: E402


def load_items(items_json: dict, meeting: str) -> list[Item]:
    out = []
    for it in items_json.get(meeting, []):
        if not it.get("summary_zh"):
            continue
        try:
            start = float(it.get("startTime") or 0.0)
            end = float(it.get("endTime") or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(
            Item(
                item_id=str(it.get("item_id")),
                type=str(it.get("type") or ""),
                summary=it["summary_zh"],
                start_sec=start,
                end_sec=end,
            )
        )
    return out


def run_meeting(args_tuple) -> dict:
    meeting, corpus, offsets_dir, items_json, url, out_dir = args_tuple
    dst = out_dir / f"{meeting}.jsonl"
    if dst.exists():
        return {"meeting": meeting, "skipped": True}

    utts = parse_transcript((corpus / f"{meeting}.txt").read_text(encoding="utf-8"))
    offs_raw = json.loads((offsets_dir / f"{meeting}.json").read_text(encoding="utf-8"))
    offsets = [(o["start_sec"], o["end_sec"]) for o in offs_raw]
    items = load_items(items_json, meeting)
    if len(offsets) != len(utts):
        return {
            "meeting": meeting,
            "error": f"offset/utterance mismatch {len(offsets)}/{len(utts)}",
        }
    if not items:
        return {"meeting": meeting, "error": "no gold items"}

    teacher = LlamaServer(base_url=url, max_tokens=512, seed=0, raw_completion=True,
                          extra={"chat_template_kwargs": {"enable_thinking": False}})
    try:
        trace = generate_meeting_supervision(utts, offsets, items, teacher)
    except Exception as exc:  # one meeting must not sink the batch
        return {"meeting": meeting, "error": repr(exc)[:160]}

    dirty = sum(1 for s in trace.steps if not replay_step_cleanly(s))
    samples = build_samples(meeting, trace)
    with dst.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({
                "meeting": s.meeting, "step": s.step, "prompt_version": s.prompt_version,
                "system": s.system, "prompt": s.prompt, "completion": s.completion,
                "is_nop": s.is_nop,
            }, ensure_ascii=False) + "\n")
    return {
        "meeting": meeting, "steps": len(trace.steps), "samples": len(samples),
        "nop": sum(1 for s in trace.steps if s.is_nop), "dirty_replays": dirty,
        "items": len(items),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--offsets", type=Path, required=True)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--urls", default="http://127.0.0.1:8083",
                   help="comma-separated llama-server base URLs; meetings are round-robined")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    urls = [u if u.startswith("http") else f"http://127.0.0.1:{u}" for u in args.urls.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    items_json = json.loads(args.items.read_text(encoding="utf-8"))
    meetings = sorted(p.stem for p in args.corpus.glob("*.txt"))
    jobs = [
        (m, args.corpus, args.offsets, items_json, urls[i % len(urls)], args.out)
        for i, m in enumerate(meetings)
    ]
    print(f"[supervision] {len(jobs)} meetings across {len(urls)} server(s)", file=sys.stderr)

    done = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_meeting, jobs):
            if r.get("skipped"):
                continue
            if r.get("error"):
                errors += 1
                print(
                    f"[supervision] FAILED {r['meeting']}: {r['error']}",
                    file=sys.stderr, flush=True,
                )
                continue
            done += 1
            print(
                f"[supervision] ({done}/{len(jobs)}) {r['meeting']} steps={r['steps']} "
                f"nop={r['nop']} items={r['items']} dirty={r['dirty_replays']}",
                file=sys.stderr, flush=True,
            )
    print(f"[supervision] done={done} errors={errors}", file=sys.stderr)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
