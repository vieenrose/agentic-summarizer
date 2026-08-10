#!/usr/bin/env python3
"""Run both arms over the same meetings and render their notes (CLAUDE.md §7.3).

Paired by construction: identical meetings, identical chunk budget, identical model, one
`token_len` instrument. The only difference is the architecture — CURSOR curates one evolving
STATE; the baseline digests each window independently and reduces.

Emits `<meeting>.<arm>.notes.txt` plus a `usage.json` carrying the GT4 accounting, so the
prefill comparison comes from the same run as the quality comparison.

    python eval/run_arms.py data/transcripts/synth-*.txt --out runs/arms --lang en
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.agent import StepBudgetExceeded, run_cursor  # noqa: E402
from voxsum.baseline import run_map_reduce  # noqa: E402
from voxsum.prompts import PROMPT_VERSION  # noqa: E402
from voxsum.render import render_state  # noqa: E402
from voxsum.transcript import parse_transcript  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("transcripts", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lang", default="en", choices=["en", "zh-TW"])
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--budget", type=int, default=2048, help="chunk token budget, both arms")
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument(
        "--declarations",
        action="store_true",
        help="FunctionGemma tool declarations in SYS for the cursor arm — the student is "
        "trained with them, so eval must match (CLAUDE.md §7.8)",
    )
    p.add_argument("--arms", default="cursor,baseline")
    p.add_argument(
        "--heuristic-tokens",
        action="store_true",
        help="skip the student tokenizer (tests only): chunking must use the real "
        "tokenizer, or zh chunks overflow the 4k context (heuristic undercounts zh)",
    )
    args = p.parse_args(argv)

    from voxsum.backends.llama_server import LlamaServer
    from voxsum.chunker import heuristic_token_len

    token_len = heuristic_token_len
    if not args.heuristic_tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
        token_len = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
        print("[arms] chunking with the student tokenizer (real budget)", flush=True)

    # Greedy: paired eval must be reproducible; sampling noise belongs to the judge, not the arms.
    model = LlamaServer(
        base_url=args.base_url,
        thinking=not args.no_thinking,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    if not model.health():
        print(f"llama-server not reachable at {args.base_url}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    usage: list[dict] = []

    for path in args.transcripts:
        utterances = parse_transcript(path.read_text(encoding="utf-8"))
        for arm in arms:
            print(f"[arms] {path.stem} :: {arm} ({len(utterances)} lines)", flush=True)
            try:
                if arm == "cursor":
                    result = run_cursor(
                        utterances,
                        model,
                        lang=args.lang,
                        budget=args.budget,
                        token_len=token_len,
                        declarations=args.declarations,
                    )
                    state, use = result.state, result.usage
                    extra = {
                        "valid_op_rate": result.valid_op_rate,
                        "anchor_rate_raw": result.anchor_rate_raw,
                        "coverage_gaps": len(result.coverage_gaps),
                    }
                else:
                    result = run_map_reduce(
                        utterances,
                        model,
                        lang=args.lang,
                        budget=args.budget,
                        token_len=token_len,
                    )
                    state, use = result.state, result.usage
                    extra = {
                        "windows": result.windows,
                        "reduce_calls": result.reduce_calls,
                        "valid_bullet_rate": result.valid_bullet_rate,
                    }
            except StepBudgetExceeded as exc:
                print(f"[arms] SKIPPED {path.stem}/{arm}: {exc}", file=sys.stderr)
                continue

            (args.out / f"{path.stem}.{arm}.notes.txt").write_text(
                render_state(state), encoding="utf-8"
            )
            usage.append(
                {
                    "meeting_id": path.stem,
                    "arm": arm,
                    "lang": args.lang,
                    "prompt_version": PROMPT_VERSION,
                    "chunk_budget": args.budget,
                    "calls": use.calls,
                    "prefill_tokens": use.prefill_tokens,
                    "decode_tokens": use.decode_tokens,
                    **extra,
                }
            )
            print(
                f"[arms]   calls {use.calls}  prefill {use.prefill_tokens:,}  "
                f"decode {use.decode_tokens:,}",
                flush=True,
            )

    (args.out / "usage.json").write_text(
        json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # GT4 straight from this run, so the efficiency claim and the quality claim share data.
    for lang in {u["lang"] for u in usage}:
        cur = sum(u["prefill_tokens"] for u in usage if u["arm"] == "cursor" and u["lang"] == lang)
        base = sum(
            u["prefill_tokens"] for u in usage if u["arm"] == "baseline" and u["lang"] == lang
        )
        if not (cur and base):
            continue
        # GT4 is only meaningful at production chunk size over enough windows: CURSOR's SYS
        # is ~3x the map step's, so on a meeting that fits in one or two chunks the fixed
        # cost dominates and the ratio is an artifact of the test, not of the architecture.
        calls = sum(u["calls"] for u in usage if u["arm"] == "cursor" and u["lang"] == lang)
        ratio = cur / base
        if args.budget < 1024 or calls < 3:
            print(
                f"[arms] GT4 {lang}: {ratio:.2f}x — NOT REPORTABLE "
                f"(chunk budget {args.budget}, {calls} steps). Re-run at --budget 2048 "
                "over meetings long enough to span several chunks."
            )
            continue
        gate = "PASS" if ratio <= 1.25 else "FAIL"
        print(
            f"[arms] GT4 {lang}: prefill {cur:,} vs {base:,} = {ratio:.2f}x "
            f"({(ratio - 1) * 100:+.0f}%) {gate}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
