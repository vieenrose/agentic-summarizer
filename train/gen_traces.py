#!/usr/bin/env python3
"""Teacher -> harness-validated op traces (PLAN.md §2, §2c).

Replays the real CURSOR loop with the teacher in the student's seat: `SYS + STATE_i +
CHUNK_i` in, ops out, validated by the real harness, applied, cursor advances. The STATE a
step conditions on was itself built by *accepted* ops, so the trace is on-policy for the
distribution the student will actually see.

Three §2c invariants are enforced here, not assumed:

* the teacher sees exactly the student's per-step prompt — no lookahead, no enlarged STATE;
* every step is asserted within budget using the **student's** tokenizer;
* each record stores the exact prompt it was generated from, so "on budget" is checkable.

    python train/gen_traces.py data/transcripts/*.txt --out data/traces/train.jsonl \\
        --lang en --tokenizer google/functiongemma-270m-it
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.agent import StepBudgetExceeded, run_cursor  # noqa: E402
from voxsum.ops import Malformed, Nop, render_op  # noqa: E402
from voxsum.prompts import PROMPT_VERSION  # noqa: E402
from voxsum.render import render_state  # noqa: E402
from voxsum.transcript import parse_transcript  # noqa: E402


def student_token_len(model_id: str) -> Callable[[str], int]:
    """Token counter using the *student's* tokenizer — the one the budget is defined by.

    A heuristic must never be what decides whether a step is on budget (PLAN.md §2c).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("transcripts", nargs="+", type=Path, help="transcript v1 files")
    p.add_argument("--out", type=Path, required=True, help="output JSONL")
    p.add_argument("--lang", default="en", choices=["en", "zh-TW"])
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--tokenizer", default="google/functiongemma-270m-it")
    p.add_argument("--budget", type=int, default=2048, help="chunk token budget")
    p.add_argument("--step-budget", type=int, default=4096, help="per-step prompt ceiling")
    p.add_argument("--grammar", action="store_true", help="constrain output with the op GBNF")
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable teacher reasoning. Default is thinking ON: the screen found it is "
        "what buys revise-don't-append, especially in zh-TW (RESULTS.md). Legitimate per "
        "PLAN.md §2c — extra compute on the same input; only op lines are kept.",
    )
    p.add_argument("--max-tokens", type=int, default=6144, help="output budget per step")
    p.add_argument(
        "--keep-nop",
        action="store_true",
        default=True,
        help="keep NOP steps (default: yes — NOP must be taught, not just tolerated)",
    )
    p.add_argument("--heuristic-tokens", action="store_true", help="skip transformers (tests only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from voxsum.backends.llama_server import OP_GRAMMAR, LlamaServer
    from voxsum.chunker import heuristic_token_len

    token_len = heuristic_token_len if args.heuristic_tokens else student_token_len(args.tokenizer)
    model = LlamaServer(
        base_url=args.base_url,
        grammar=OP_GRAMMAR if args.grammar else None,
        thinking=not args.no_thinking,
        max_tokens=args.max_tokens,
    )
    if not model.health():
        print(f"llama-server not reachable at {args.base_url}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = nop_steps = 0

    with args.out.open("w", encoding="utf-8") as sink:
        for path in args.transcripts:
            utterances = parse_transcript(path.read_text(encoding="utf-8"))
            print(f"[traces] {path.name}: {len(utterances)} lines", flush=True)
            try:
                trace = run_cursor(
                    utterances,
                    model,
                    lang=args.lang,
                    budget=args.budget,
                    step_budget=args.step_budget,
                    token_len=token_len,
                )
            except StepBudgetExceeded as exc:
                print(f"[traces] SKIPPED {path.name}: {exc}", file=sys.stderr)
                continue

            for step in trace.steps:
                # Keep only ops the harness accepted — the target must be applicable.
                accepted = [
                    r.op
                    for r in step.outcome.results
                    if r.applied and not isinstance(r.op, Malformed)
                ]
                dropped += sum(1 for r in step.outcome.results if not r.applied)
                if not accepted:
                    continue
                is_nop = all(isinstance(o, Nop) for o in accepted)
                if is_nop:
                    nop_steps += 1
                    if not args.keep_nop:
                        continue

                sink.write(
                    json.dumps(
                        {
                            "meeting": path.stem,
                            "lang": args.lang,
                            "step": step.index,
                            "prompt_version": PROMPT_VERSION,
                            # The exact prompt, so budget claims stay checkable (§2c).
                            "system": step.system,
                            "user": step.user,
                            "prompt_tokens": step.prompt_tokens,
                            "target": "\n".join(render_op(o) for o in accepted),
                            "raw": step.raw,
                            "is_nop": is_nop,
                            "chunk_start": step.chunk.start,
                            "chunk_end": step.chunk.end,
                            "content_rich": step.chunk.is_content_rich(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1

            notes = args.out.with_suffix(f".{path.stem}.notes.txt")
            notes.write_text(render_state(trace.state), encoding="utf-8")
            rate = trace.valid_op_rate
            raw = trace.anchor_rate_raw
            anchor = "n/a" if raw is None else f"{raw:.0%}"
            print(
                f"[traces] {path.name}: {len(trace.steps)} steps, "
                f"valid-op {'n/a' if rate is None else f'{rate:.0%}'}, "
                f"anchor(raw) {anchor}",
                flush=True,
            )

    print(f"[traces] kept {kept} steps ({nop_steps} NOP), dropped {dropped} ops -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
