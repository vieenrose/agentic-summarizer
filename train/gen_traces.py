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
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.agent import StepBudgetExceeded, run_cursor  # noqa: E402
from voxsum.index import TranscriptIndex  # noqa: E402
from voxsum.ops import Add, Cmp, Malformed, Nop, Upd, render_op  # noqa: E402
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


def make_judge_filter(
    utterances: list,
    judge,
    model: str,
    *,
    log: list[str] | None = None,
):
    """Veto candidate bullets a judge cannot verify from the transcript.

    Why this exists: the harness accepts any op that parses, anchors and survives the guards
    — it has no notion of whether the bullet is *true*. Measured on real QMSum meetings only
    53-58% of the teacher's own bullets are judge-verifiable (RESULTS.md), so an unfiltered
    trace set teaches the student to emit unverifiable bullets, and GT2 is a faithfulness
    gate. Filtering here keeps the SFT target clean.

    This does **not** breach PLAN.md §2c. The judge is not the teacher: the teacher still saw
    only its own chunk. The filter is harness-side, like a guard, and it only ever *removes*
    a target — it never feeds transcript knowledge back into the teacher's prompt.
    """
    from judge import _FAITH_SYS, _VERDICT, faith_prompt  # local: keeps eval/ optional

    # Evidence is limited to what the agent has already *seen* — lines up to the end of the
    # current chunk. Judging against the whole transcript looks obviously right and is
    # actively harmful here: at the setup chunk of a reversal meeting the correct bullet is
    # "plan rejected", and a whole-transcript judge marks it CONTRADICTED because the plan is
    # approved later. That veto keeps the bullet out of STATE, so at the revision chunk there
    # is nothing to revise and the teacher can only ADD — destroying the UPD demonstration the
    # revision-dense meetings exist to produce. Measured: zh synthetic revision share fell to
    # 10%, the lowest of any source, on the set built specifically to teach revision.
    #
    # A bullet asserting the state at time T is not falsified by an event at T+1. That is what
    # UPD is for.
    prefix_cache: dict[int, TranscriptIndex] = {}

    def index_upto(end: int) -> TranscriptIndex:
        if end not in prefix_cache:
            prefix_cache[end] = TranscriptIndex([u for u in utterances if u.start <= end])
        return prefix_cache[end]

    def verify(text: str, anchor: int | None, horizon: int) -> str | None:
        index = index_upto(horizon)
        evidence = index.evidence_for(text, anchor, mode="claim")
        raw = judge(model, _FAITH_SYS, faith_prompt(text, evidence))
        hits = _VERDICT.findall(raw)
        verdict = hits[-1].upper() if hits else "MISSING"
        return None if verdict == "SUPPORTED" else f"judge: {verdict}"

    def op_filter(op, chunk) -> str | None:
        # Only claims get judged. NOP/TITLE assert nothing about the meeting, and DEL removes
        # a bullet rather than adding one — vetoing a DEL would *preserve* a wrong bullet.
        try:
            horizon = chunk.end
            if isinstance(op, (Add, Upd)):
                return verify(op.bullet, op.anchor, horizon)
            if isinstance(op, Cmp):
                # Independent claims, so verify them together rather than in sequence.
                with ThreadPoolExecutor(max_workers=len(op.bullets) or 1) as pool:
                    reasons = list(
                        pool.map(lambda b: verify(b.text, b.anchor, horizon), op.bullets)
                    )
                return next((r for r in reasons if r is not None), None)
            return None
        except Exception as exc:  # a judge outage must not silently unfilter the run
            if log is not None:
                log.append(f"judge error, op kept unverified: {exc}")
            return None

    return op_filter


def make_batch_judge_filter(utterances, judge, model, *, log=None, workers: int = 8):
    """Judge every op of a step concurrently.

    Measured: teacher inference is ~20s per step and the judge filter added ~19s more, run as
    5 sequential HTTPS round-trips at ~3.8s each — with both GPUs idle throughout. The calls
    are independent (each verifies a different bullet), so serialising them was pure waste.
    Verifying a step's ops together removes almost all of it.

    Returns a *step* filter: given the step's ops, return the subset to keep plus the vetoes.
    """
    per_op = make_judge_filter(utterances, judge, model, log=log)

    def step_filter(ops, chunk):
        if not ops:
            return [], []
        with ThreadPoolExecutor(max_workers=min(workers, len(ops))) as pool:
            reasons = list(pool.map(lambda op: per_op(op, chunk), ops))
        kept = [op for op, reason in zip(ops, reasons, strict=False) if reason is None]
        vetoed = [
            (op, reason) for op, reason in zip(ops, reasons, strict=False) if reason is not None
        ]
        return kept, vetoed

    return step_filter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("transcripts", nargs="+", type=Path, help="transcript v1 files")
    p.add_argument("--out", type=Path, required=True, help="output JSONL")
    p.add_argument("--lang", default="en", choices=["en", "zh-TW"])
    p.add_argument(
        "--base-url",
        action="append",
        default=None,
        help="teacher endpoint; repeatable. With several, meetings run in parallel — one "
        "worker per endpoint. A whole-model server per GPU beats one layer-split server "
        "across both: no cross-PCIe activations, and real per-meeting parallelism.",
    )
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
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--keep-nop",
        action="store_true",
        default=True,
        help="keep NOP steps (default: yes — NOP must be taught, not just tolerated)",
    )
    p.add_argument("--heuristic-tokens", action="store_true", help="skip transformers (tests only)")
    p.add_argument(
        "--judge-filter",
        action="store_true",
        help="drop bullets a judge cannot verify before they reach STATE. Strongly "
        "recommended: only 53-58%% of the teacher's own bullets are verifiable on real "
        "meetings, and training on the rest teaches unverifiable output.",
    )
    p.add_argument("--judge-model", default="openai/gpt-oss-20b")
    p.add_argument("--judge-budget-usd", type=float, default=1.00)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from voxsum.backends.llama_server import OP_GRAMMAR, LlamaServer
    from voxsum.chunker import heuristic_token_len, iter_chunks

    token_len = heuristic_token_len if args.heuristic_tokens else student_token_len(args.tokenizer)
    urls = args.base_url or ["http://127.0.0.1:8080"]
    models = [
        LlamaServer(
            base_url=url,
            grammar=OP_GRAMMAR if args.grammar else None,
            thinking=not args.no_thinking,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        for url in urls
    ]
    for url, client in zip(urls, models, strict=False):
        if not client.health():
            print(f"llama-server not reachable at {url}", file=sys.stderr)
            return 2
    print(f"[traces] {len(models)} teacher endpoint(s): {', '.join(urls)}", flush=True)

    judge_client = None
    if args.judge_filter:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
        from judge import TogetherJudge

        judge_client = TogetherJudge(
            api_key=os.environ.get("TOGETHER_API_KEY", ""), budget_usd=args.judge_budget_usd
        )
    else:
        print(
            "[traces] WARNING: no judge filter. On real meetings ~45% of teacher bullets are "
            "unverifiable, and they will become SFT targets. Pass --judge-filter.",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = nop_steps = vetoed_total = 0

    # One endpoint per process. Meetings are independent, so the parallelism that matters is
    # run at the shell level: launch one `gen_traces.py` per teacher endpoint over disjoint
    # transcript sets, then merge the JSONL. `build_sft.py` already accepts several files.
    model = models[0]

    with args.out.open("w", encoding="utf-8") as sink:
        for path in args.transcripts:
            utterances = parse_transcript(path.read_text(encoding="utf-8"))
            n_chunks = len(list(iter_chunks(utterances, budget=args.budget, token_len=token_len)))
            print(f"[traces] {path.name}: {len(utterances)} lines, {n_chunks} chunks", flush=True)
            if n_chunks < 2:
                # A single-chunk meeting cannot teach revision: the model sees the setup and
                # the later contradiction together and can simply ADD the final state, so no
                # UPD/DEL is ever the correct answer. Revision-dense synthetic meetings are
                # short by design, so they need a smaller --budget than real transcripts.
                print(
                    f"[traces] WARNING {path.name} fits in one chunk at budget {args.budget}: "
                    "it teaches no cross-chunk revision. Lower --budget for short meetings.",
                    file=sys.stderr,
                )
            judge_errors: list[str] = []
            step_filter = (
                make_batch_judge_filter(
                    utterances, judge_client, args.judge_model, log=judge_errors
                )
                if judge_client is not None
                else None
            )
            try:
                trace = run_cursor(
                    utterances,
                    model,
                    lang=args.lang,
                    budget=args.budget,
                    step_budget=args.step_budget,
                    token_len=token_len,
                    step_filter=step_filter,
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
                            "vetoed": [{"op": o, "reason": r} for o, r in step.vetoed],
                            "reasoning": step.reasoning,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1

            vetoed_total += sum(len(s.vetoed) for s in trace.steps)
            for message in judge_errors[:3]:
                print(f"[traces] {message}", file=sys.stderr)

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
    if judge_client is not None:
        print(f"[traces] judge vetoed {vetoed_total} unverifiable bullets")
        print(f"[traces] {judge_client.spend.report()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
