"""`arcsum-judge`: SPEC §5.1 faithfulness scoring over a batch of (transcript, prose)
pairs, written as JSONL score records compatible with `metrics.stats.load_scores` and
`cli.report`'s `inversions` field.

    arcsum-judge cases.json --model local:8080/gemma-judge --out judged.jsonl

`cases.json` is `[{"meeting_id", "system", "transcript", "prose"}, ...]`. `transcript`
is v2 text (SPEC §2), parsed here via `transcript.parse_transcript`. Fully testable
with no network: `JudgeClient` talks to `urllib.request.urlopen`, which this module's
tests stub exactly as `test_judge.py` does for the client itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arcsum.judge.client import JudgeClient
from arcsum.judge.faith import judge_meeting
from arcsum.transcript import parse_transcript


def judge_case(case: dict, client: JudgeClient, *, model: str, votes: int = 3) -> dict:
    utterances = parse_transcript(case["transcript"])
    score = judge_meeting(case["prose"], utterances, client, model=model, votes=votes)
    return {
        "meeting_id": case["meeting_id"],
        "system": case["system"],
        "faith_claim": score.faith_claim,
        "inversions": score.inverted,
        "unsupported": score.unsupported,
        "claims": len(score.bullets),
    }


def judge_cases(
    cases: list[dict], client: JudgeClient, *, model: str, votes: int = 3
) -> list[dict]:
    return [judge_case(case, client, model=model, votes=votes) for case in cases]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "cases", type=Path, help="JSON file: [{meeting_id, system, transcript, prose}, ...]"
    )
    p.add_argument(
        "--model", required=True, help="local:<port>/<name> judge model (SPEC §5.1 refusal rules)"
    )
    p.add_argument("--votes", type=int, default=3, help="majority-vote repeats per claim")
    p.add_argument("--budget-usd", type=float, default=5.0)
    p.add_argument("--out", type=Path, required=True, help="write JSONL score records here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))

    client = JudgeClient(budget_usd=args.budget_usd)
    records = judge_cases(cases, client, model=args.model, votes=args.votes)

    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[judge] {client.spend.report()}", file=sys.stderr)
    print(f"[judge] wrote {len(records)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
