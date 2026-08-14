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
    p.add_argument(
        "--baseline-url",
        default=None,
        help="model for the baseline arm (a GENERAL model: the fine-tuned student is "
        "trained for the CURSOR prompt only and produces empty digests — measured). "
        "Defaults to --base-url.",
    )
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
        "--sweep",
        choices=["none", "verify", "anchor", "both"],
        default="none",
        help="final VERIFY/ANCHOR sweep on the cursor arm's state (CLAUDE.md §5.2) — "
        "the harness-side faithfulness backstop",
    )
    p.add_argument("--sweep-budget", type=int, default=20)
    p.add_argument("--sweep-judge", default="local:8090/gpt-oss-20b")
    p.add_argument(
        "--verify-url",
        default="",
        help="on-device in-stream verification: an LFM2.5-350m-verifier endpoint. Every "
        "ADD/UPD touching DECISIONS/ACTIONS is judged against the chunk's anchor "
        "neighborhood before application; UNSUPPORTED/CONTRADICTED ops are dropped.",
    )
    p.add_argument("--tokenizer", default="google/functiongemma-270m-it",
                  help="student tokenizer for chunk budgets (must match the served model)")
    p.add_argument(
        "--heuristic-tokens",
        action="store_true",
        help="skip the student tokenizer (tests only): chunking must use the real "
        "tokenizer, or zh chunks overflow the 4k context (heuristic undercounts zh)",
    )
    args = p.parse_args(argv)

    from voxsum.backends.llama_server import LlamaServer
    from voxsum.chunker import heuristic_token_len

    verify_filter = None
    if args.verify_url:
        from voxsum.ops import Add, Upd
        from voxsum.transcript import sec_to_clock
        from judge import _FAITH_SYS, faith_prompt

        verifier = LlamaServer(base_url=args.verify_url, max_tokens=8, temperature=0.0)

        class _Ev:
            def __init__(self, text: str):
                self.text = text

            def render(self) -> str:
                return self.text

        def verify_filter(op, chunk):
            if not isinstance(op, (Add, Upd)):
                return None
            if op.section not in ("DECISIONS", "ACTIONS"):
                return None
            anchor = getattr(op, "anchor", None)
            if anchor is None:
                return None
            near = [u for u in chunk.utterances if abs(u.start - anchor) <= 90]
            if not near:
                near = list(chunk.utterances)[:6]
            ev = [_Ev(f"[{sec_to_clock(u.start)}] {u.speaker + ': ' if u.speaker else ''}{u.text}") for u in near]
            raw = verifier(_FAITH_SYS, faith_prompt(op.bullet, ev)).upper()
            import re
            m = re.search(r"(SUPPORTED|CONTRADICTED|UNSUPPORTED)", raw)
            verdict = m.group(1) if m else "PARSE_FAIL"
            if verdict in ("UNSUPPORTED", "CONTRADICTED"):
                return f"in-stream verifier: {verdict}"
            return None

    token_len = heuristic_token_len
    if not args.heuristic_tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        token_len = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
        print(f"[arms] chunking with {args.tokenizer} (real budget)", flush=True)

    # Greedy: paired eval must be reproducible; sampling noise belongs to the judge, not the arms.
    model = LlamaServer(
        base_url=args.base_url,
        thinking=not args.no_thinking,
        max_tokens=args.max_tokens,
        temperature=0.0,
        send_thinking_kwarg=False,  # MiniCPM5 template inserts an empty <think> block otherwise
    )
    sweep_judge = None
    if args.sweep != "none":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
        from judge import TogetherJudge

        sweep_judge = TogetherJudge(api_key="local", budget_usd=5.0, max_tokens=14000)
        print(f"[arms] sweep={args.sweep} judge={args.sweep_judge} "
              f"budget={args.sweep_budget}", flush=True)

    baseline_model = (
        LlamaServer(
            base_url=args.baseline_url,
            thinking=not args.no_thinking,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        if args.baseline_url
        else model
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
                        op_filter=verify_filter,
                    )
                    state, use = result.state, result.usage
                    sweep_extra = {}
                    if sweep_judge is not None:
                        from voxsum.sweep import run_sweep

                        prompt_builder = None
                        if args.sweep in ("verify", "both"):
                            # Pin the FAITH judge's EXACT protocol: the sweep must drop
                            # exactly what the eval judge would flag (measured: a lenient
                            # sweep prompt let 3/12 inversions through).
                            from judge import _FAITH_SYS, faith_prompt

                            def prompt_builder(bullet, evidence):
                                return faith_prompt(bullet, evidence)

                        sweep = run_sweep(
                            state,
                            utterances,
                            lambda sys_p, user: sweep_judge(args.sweep_judge, sys_p, user),
                            verify=args.sweep in ("verify", "both"),
                            anchor=args.sweep in ("anchor", "both"),
                            budget=args.sweep_budget,
                            prompt_builder=prompt_builder,
                            # FIX rewrites from the local judge create inversions
                            # (measured twice); drop-only is the safe protocol.
                            apply_fix=False,
                        )
                        sweep_extra = {
                            "sweep_calls": sweep.calls,
                            "sweep_dropped": sweep.dropped,
                            "sweep_fixed": sweep.fixed,
                            "sweep_anchors_repaired": sweep.anchors_repaired,
                        }
                    extra = {
                        "valid_op_rate": result.valid_op_rate,
                        "anchor_rate_raw": result.anchor_rate_raw,
                        "coverage_gaps": len(result.coverage_gaps),
                        **sweep_extra,
                    }
                else:
                    result = run_map_reduce(
                        utterances,
                        baseline_model,
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

    # Merge, don't overwrite: tiers run en and zh as separate invocations and GT4 needs
    # both arms' accounting from the same run.
    usage_path = args.out / "usage.json"
    merged = usage
    if usage_path.exists():
        try:
            merged = json.loads(usage_path.read_text(encoding="utf-8")) + usage
        except Exception:
            pass
    usage_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

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
