"""STATE-reading negatives (the maintainer's op-level finding).

On the real zh meeting the model re-proposed bullets STATE already contained —
~30% of its output, rejected by the dedup guard. The harvest only captures the
sweep's drops, never the dedup rejections; this tool captures them: for each step
where an op was rejected as duplicate, the negative sample is (same state+chunk,
target = the step's applied non-NOP ops, or NOP).

Usage:
    .venv/bin/python tools/build_state_negatives.py --out data/sft/state-negatives.jsonl \
        --base-url http://127.0.0.1:8098 --tokenizer openbmb/MiniCPM5-1B \
        data/transcripts/meeting-zh-long.txt [more transcripts...]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.agent import run_cursor
from voxsum.backends.llama_server import LlamaServer
from voxsum.chunker import heuristic_token_len
from voxsum.ops import Nop, render_op
from voxsum.prompts import PROMPT_VERSION, system_prompt
from voxsum.transcript import parse_transcript


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("transcripts", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8098")
    p.add_argument("--lang", default="zh-TW")
    p.add_argument("--budget", type=int, default=2048)
    args = p.parse_args(argv)

    student = LlamaServer(base_url=args.base_url, max_tokens=512, temperature=0.0)
    sys_block = system_prompt(args.lang)
    total = 0
    with args.out.open("w") as sink:
        for path in args.transcripts:
            utt = parse_transcript(path.read_text(encoding="utf-8"))
            trace = run_cursor(
                utt, student, lang=args.lang, budget=args.budget,
                token_len=heuristic_token_len,
            )
            for s in trace.steps:
                dupes = [
                    r for r in s.outcome.results
                    if not r.applied and r.reason and "duplicate" in r.reason
                ]
                if not dupes:
                    continue
                kept = [
                    r.op for r in s.outcome.results
                    if r.applied and not isinstance(r.op, Nop)
                ]
                target = "\n".join(render_op(op) for op in kept) or "NOP"
                row = {
                    "meeting": path.stem, "lang": args.lang, "step": s.index,
                    "prompt_version": PROMPT_VERSION,
                    "system": sys_block, "prompt": s.state_before,
                    "completion": target, "is_nop": not kept,
                    "has_revision": any("UPD" in l for l in target.splitlines()),
                    "prompt_tokens": s.prompt_tokens,
                }
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    print(f"state-reading negatives: {total} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
