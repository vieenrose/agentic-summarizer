"""`arcsum-gen-traces`: run the real agent step loop (SPEC §4.1) over a directory of
format-v2 transcripts against a `llama-server` instance, and write both the SFT-ready
step projection and a supervision diagnostic report.

    arcsum-gen-traces corpus/ --url http://127.0.0.1:8080 \\
        --out traces.jsonl --report-out report.json

`corpus/` holds one format-v2 `.txt` per meeting (e.g. `arcsum-import`'s output); the
filename stem is the meeting id. Each output line in `--out` is already shaped like
`supervision.sft.SftSample`'s fields (`meeting`, `step`, `prompt_version`, `system`,
`prompt`, `completion`, `is_nop`) — deliberately the SAME projection `build_samples`
would compute, so `arcsum-build-sft` can decode it directly with no intermediate
schema of its own.

**The supervision report is computed from the live `Trace` objects in this same
process, never re-derived from the serialized rows** — `supervision.report`'s own
docstring is explicit that its definitions must stay welded to real `Trace`/`Step`/
`Outcome` objects, not a parallel on-disk schema, because getting a rate's numerator
and denominator on opposite sides of a serialization boundary was a real, twice-
repeated bug in the prior project. `--report-out` is therefore the only place this
tool's supervision diagnostics live; there is no companion tool that reconstructs a
`Trace` from `--out` later.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from arcsum.agent import Trace, run_agent
from arcsum.backends.llama_server import LlamaServer
from arcsum.chunker import CHUNK_TOKENS
from arcsum.supervision.report import report as build_supervision_report
from arcsum.transcript import parse_transcript


def gen_trace_for_meeting(
    text: str,
    model: LlamaServer,
    *,
    synth_model: LlamaServer | None = None,
    budget: int = CHUNK_TOKENS,
    protocol: str = "edit",
) -> Trace:
    utterances = parse_transcript(text)
    return run_agent(utterances, model, synth_model=synth_model, budget=budget,
                     protocol=protocol, on_step_error="skip")


def trace_to_sft_rows(meeting_id: str, trace: Trace) -> list[dict]:
    """The exact projection `supervision.sft.build_samples` computes, inlined here so
    `--out` never depends on importing a training-only module for a shape this simple."""
    rows = [
        {
            "meeting": meeting_id,
            "step": step.index,
            "prompt_version": trace.prompt_version,
            "system": step.system,
            "prompt": step.user,
            "completion": step.raw,
            "is_nop": step.is_nop,
        }
        for step in trace.steps
    ]
    if trace.synthesis is not None:
        rows.append(
            {
                "meeting": meeting_id,
                "step": len(trace.steps),
                "prompt_version": trace.prompt_version,
                "system": trace.synthesis.system,
                "prompt": trace.synthesis.user,
                "completion": trace.synthesis.raw,
                "is_nop": False,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("corpus", type=Path, help="directory of format-v2 .txt transcripts")
    p.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server base URL")
    p.add_argument(
        "--synth-url",
        default=None,
        help="separate llama-server base URL for the SYNTHESIZE call (default: --url)",
    )
    p.add_argument("--max-tokens-step", type=int, default=512)
    p.add_argument("--max-tokens-synth", type=int, default=1200)
    p.add_argument("--budget", type=int, default=CHUNK_TOKENS, help="chunk token budget")
    # SPEC 4.1 step grammar. A tool-call student emits `<tool_call>{...}` and an
    # edit-line student emits ADD/DROP/ARC/NOP; running the wrong one records raw text the
    # parser rejects, so every row would be unusable while the tool still "succeeds".
    p.add_argument("--protocol", choices=("edit", "tool"), default="edit")
    # Extra llama-server body fields, e.g.
    #   --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
    # The text-only Qwen3.5 repos train with a CLOSED <think></think> in the prompt;
    # served without this the model emits <think> in place of the tool-call prefix and
    # every completion is unparseable (measured 2026-09-01, 0/27 on the probe).
    p.add_argument("--extra", default="")
    p.add_argument("--out", type=Path, required=True, help="write SFT-shaped JSONL rows here")
    p.add_argument(
        "--report-out", type=Path, default=None, help="write the JSON supervision report here"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = sorted(args.corpus.glob("*.txt"))
    if not paths:
        print(f"[gen-traces] no .txt files found under {args.corpus}", file=sys.stderr)
        return 1

    extra = json.loads(args.extra) if args.extra else {}
    model = LlamaServer(base_url=args.url, max_tokens=args.max_tokens_step, extra=extra)
    synth_model = LlamaServer(base_url=args.synth_url or args.url,
                              max_tokens=args.max_tokens_synth, extra=extra)

    traces: list[Trace] = []
    rows: list[dict] = []
    for path in paths:
        meeting_id = path.stem
        trace = gen_trace_for_meeting(
            path.read_text(encoding="utf-8"), model, synth_model=synth_model,
            budget=args.budget, protocol=args.protocol,
        )
        traces.append(trace)
        rows.extend(trace_to_sft_rows(meeting_id, trace))

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    supervision = build_supervision_report(traces)
    print(f"[gen-traces] {len(paths)} meetings, {len(rows)} rows -> {args.out}", file=sys.stderr)
    print(f"[gen-traces] {supervision}", file=sys.stderr)
    if args.report_out:
        args.report_out.write_text(
            json.dumps(asdict(supervision), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
