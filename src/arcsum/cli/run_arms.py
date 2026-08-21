"""`arcsum-run-arms`: run BOTH the agent and the fair map-reduce baseline (SPEC §5.2)
over the same corpus and model, producing `arcsum-score`-ready pairs files for each.

    arcsum-run-arms corpus/ --references refs.json --url http://127.0.0.1:8080 \\
        --out-agent agent_pairs.json --out-baseline baseline_pairs.json
    arcsum-score agent_pairs.json --system agent --out agent_scored.jsonl
    arcsum-score baseline_pairs.json --system baseline --out baseline_scored.jsonl

`corpus/` holds one format-v2 `.txt` per meeting; `refs.json` is
`{"<meeting_id>": "<reference summary text>"}`. Both arms run against the SAME model
and the SAME `iter_chunks`/`token_len` instrument (SPEC §5.2's "fair, not weak"
baseline) — this tool exists specifically so that guarantee is structural (one call
site constructs both arms' `LlamaServer`s from the same `--url`/`--budget`), not a
convention two separate scripts could silently drift apart on.

A meeting present in `corpus/` but missing from `refs.json` is skipped with a warning,
never silently scored against an empty reference — SPEC §5.2's comparison is paired
per meeting, and an unscoreable meeting must not enter either arm's pairs file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arcsum.agent import run_agent
from arcsum.backends.llama_server import LlamaServer
from arcsum.baseline import run_map_reduce
from arcsum.chunker import CHUNK_TOKENS
from arcsum.transcript import parse_transcript


def run_both_arms(
    corpus_dir: Path,
    references: dict[str, str],
    *,
    step_model: LlamaServer,
    synth_model: LlamaServer,
    reduce_model: LlamaServer,
    budget: int = CHUNK_TOKENS,
) -> tuple[list[dict], list[dict], list[str]]:
    """Returns `(agent_pairs, baseline_pairs, skipped_meeting_ids)`."""
    agent_pairs: list[dict] = []
    baseline_pairs: list[dict] = []
    skipped: list[str] = []

    for path in sorted(corpus_dir.glob("*.txt")):
        meeting_id = path.stem
        if meeting_id not in references:
            skipped.append(meeting_id)
            continue

        source = path.read_text(encoding="utf-8")
        utterances = parse_transcript(source)
        reference = references[meeting_id]

        trace = run_agent(utterances, step_model, synth_model=synth_model, budget=budget)
        agent_pairs.append(
            {
                "meeting_id": meeting_id,
                "source": source,
                "candidate": trace.synthesis.prose.text if trace.synthesis else "",
                "reference": reference,
            }
        )

        baseline = run_map_reduce(utterances, step_model, reduce_model=reduce_model, budget=budget)
        baseline_pairs.append(
            {
                "meeting_id": meeting_id,
                "source": source,
                "candidate": baseline.prose.text,
                "reference": reference,
            }
        )

    return agent_pairs, baseline_pairs, skipped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("corpus", type=Path, help="directory of format-v2 .txt transcripts")
    p.add_argument("--references", type=Path, required=True, help='JSON: {"<id>": "<ref>"}')
    p.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server base URL")
    p.add_argument("--synth-url", default=None, help="default: --url")
    p.add_argument("--reduce-url", default=None, help="default: --url")
    p.add_argument("--max-tokens-step", type=int, default=512)
    p.add_argument("--max-tokens-synth", type=int, default=1200)
    p.add_argument("--budget", type=int, default=CHUNK_TOKENS)
    p.add_argument("--out-agent", type=Path, required=True)
    p.add_argument("--out-baseline", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    references = json.loads(args.references.read_text(encoding="utf-8"))

    step_model = LlamaServer(base_url=args.url, max_tokens=args.max_tokens_step)
    synth_model = LlamaServer(base_url=args.synth_url or args.url, max_tokens=args.max_tokens_synth)
    reduce_model = LlamaServer(
        base_url=args.reduce_url or args.url, max_tokens=args.max_tokens_synth
    )

    agent_pairs, baseline_pairs, skipped = run_both_arms(
        args.corpus,
        references,
        step_model=step_model,
        synth_model=synth_model,
        reduce_model=reduce_model,
        budget=args.budget,
    )

    args.out_agent.write_text(
        json.dumps(agent_pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.out_baseline.write_text(
        json.dumps(baseline_pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if skipped:
        print(
            f"[run-arms] skipped {len(skipped)} meetings with no reference: {skipped}",
            file=sys.stderr,
        )
    print(
        f"[run-arms] {len(agent_pairs)} meetings -> {args.out_agent}, {args.out_baseline}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
