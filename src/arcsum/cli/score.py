"""`arcsum-score`: reference metrics (SPEC §5) for a batch of meetings, written as
JSONL score records compatible with `metrics.stats.load_scores`.

    arcsum-score pairs.json --system agent --out scored.jsonl

`pairs.json` is `[{"meeting_id", "source", "candidate", "reference"}, ...]`.
`source` is the whole transcript — Coverage/Density (SPEC §5) measure how extractive
the CANDIDATE is relative to the SOURCE TRANSCRIPT, not relative to the reference
summary, so it is a required field distinct from `reference`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from arcsum.metrics.reference import coverage, density, length, rouge_l, rouge_n
from arcsum.tokens import heuristic_token_len


def score_pair(
    source: str,
    candidate: str,
    reference: str,
    *,
    meeting_id: str,
    system: str,
    token_len: Callable[[str], int] = heuristic_token_len,
) -> dict:
    """One meeting's reference-metric record, in the `load_scores`-compatible shape
    (a `meeting_id` and `system` field, so it can be globbed straight into
    `metrics.stats.compare` without transformation)."""
    r1 = rouge_n(candidate, reference, n=1)
    r2 = rouge_n(candidate, reference, n=2)
    rl = rouge_l(candidate, reference)
    ln = length(candidate, token_len=token_len)
    return {
        "meeting_id": meeting_id,
        "system": system,
        "rouge1": r1.f1,
        "rouge2": r2.f1,
        "rougeL": rl.f1,
        "coverage": coverage(source, candidate),
        "density": density(source, candidate),
        "length_chars": ln.chars,
        "length_tokens": ln.tokens,
    }


def score_pairs(
    pairs: Sequence[dict], *, system: str, token_len: Callable[[str], int] = heuristic_token_len
) -> list[dict]:
    return [
        score_pair(
            p["source"],
            p["candidate"],
            p["reference"],
            meeting_id=p["meeting_id"],
            system=system,
            token_len=token_len,
        )
        for p in pairs
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "pairs", type=Path, help="JSON file: [{meeting_id, source, candidate, reference}, ...]"
    )
    p.add_argument("--system", required=True, help="system name stamped on every record")
    p.add_argument("--out", type=Path, required=True, help="write JSONL score records here")
    p.add_argument(
        "--tokenizer", default=None, help="HF model id for real token counts (default: heuristic)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))

    token_len = heuristic_token_len
    if args.tokenizer:
        from arcsum.tokens import hf_token_len

        token_len = hf_token_len(args.tokenizer)

    records = score_pairs(pairs, system=args.system, token_len=token_len)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[score] wrote {len(records)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
