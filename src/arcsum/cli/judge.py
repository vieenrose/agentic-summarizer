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
from collections.abc import Callable
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


def case_key(case: dict) -> str:
    """The identity a record is resumed on. One definition, used by writer and reader."""
    return f"{case.get('system')}/{case.get('meeting_id')}"


def judge_cases(
    cases: list[dict],
    client: JudgeClient,
    *,
    model: str,
    votes: int = 3,
    sink: Callable[[dict], None] | None = None,
    fail_sink: Callable[[str, str], None] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Judge every case, isolating per-case failures. Returns `(records, failures)`,
    where `failures` maps `"<system>/<meeting_id>"` to the exception text.

    **One hard claim must not sink the batch.** Measured 2026-08-27: a single claim
    (a household-count figure the summary conflated between a state total and a county
    total -- exactly the subtle inversion the judge exists to catch) made the judge
    reason for 10,370 characters, exhaust its output budget before emitting a verdict,
    and raise; that aborted all 40 cases after ~16 had already been judged, discarding
    the completed work. `cli.run_arms` already learned this lesson and isolates per
    meeting per arm; this mirrors it.

    A failure is recorded, NOT scored as zero inversions: a case the judge could not
    evaluate is missing evidence, and silently counting it as clean would bias G2
    toward passing -- the same defect `gate_g2_faithfulness` withholds on when no
    records exist at all.

    **`sink` closes the same hole at the PROCESS boundary, where it reopened.** Isolating
    per case stopped one claim from discarding the batch, but the batch still lived only
    in memory until the run ended -- so anything that killed the process discarded it just
    as completely. That is not hypothetical here: the five-judge panel of 2026-09-05 ran
    hosted reasoning models at ~2.3 minutes per case, i.e. **over three hours per judge**,
    entirely uninspectable, with every completed case one `kill` away from being lost. A
    sink writes each record as it is produced, which buys durability and progress at once
    -- an on-disk line count is the only way to tell a slow judge from a wedged one.
    """
    records: list[dict] = []
    failures: dict[str, str] = {}
    for case in cases:
        key = case_key(case)
        try:
            record = judge_case(case, client, model=model, votes=votes)
        except Exception as exc:  # any judge failure is per-case data, not fatal
            failures[key] = f"{type(exc).__name__}: {exc}"
            if fail_sink is not None:
                fail_sink(key, failures[key])
            continue
        records.append(record)
        if sink is not None:
            sink(record)
    return records, failures


def completed_keys(path: Path) -> set[str]:
    """Keys already scored in an existing JSONL output, for `--resume`.

    Tolerates a truncated final line: a run killed mid-write leaves a partial record, and
    refusing to resume because of it would defeat the point of resuming at all.
    """
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add(f"{rec.get('system')}/{rec.get('meeting_id')}")
    return keys


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
    p.add_argument(
        "--max-tokens",
        type=int,
        default=JudgeClient.max_tokens,
        help="per-call output budget. A REASONING judge needs far more than the answer "
        "itself: gpt-oss-20b was measured emitting 10,370 characters of reasoning on a "
        "hard claim and returning empty content at the 3000 default, but answering "
        "cleanly at 6000. Raise this before suspecting the judge model.",
    )
    p.add_argument("--out", type=Path, required=True, help="write JSONL score records here")
    p.add_argument(
        "--out-failures", type=Path, default=None, help="optional: write per-case failures here"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip cases already present in --out and APPEND. A hosted reasoning judge runs "
        "~2.3 min/case, so an 80-case panel is a 3-hour job; without this, any interruption "
        "re-spends the whole budget to recompute what is already on disk.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))

    done = completed_keys(args.out) if args.resume else set()
    if done:
        cases = [c for c in cases if case_key(c) not in done]
        print(f"[judge] resuming: {len(done)} already scored, {len(cases)} to go", file=sys.stderr)

    client = JudgeClient(budget_usd=args.budget_usd, max_tokens=args.max_tokens)
    failures_so_far: dict[str, str] = {}

    def write_failures(key: str, err: str) -> None:
        # Rewritten whole each time: it is a small dict, and a failure file that only
        # appears at exit is exactly the durability hole `sink` exists to close.
        failures_so_far[key] = err
        if args.out_failures is not None:
            args.out_failures.write_text(
                json.dumps(failures_so_far, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # Line-buffered append: each record is on disk before the next case starts, so the
    # file is both a crash-safe log and a live progress meter (`wc -l`).
    with args.out.open("a" if done else "w", encoding="utf-8", buffering=1) as f:
        records, failures = judge_cases(
            cases,
            client,
            model=args.model,
            votes=args.votes,
            sink=lambda r: f.write(json.dumps(r, ensure_ascii=False) + "\n"),
            fail_sink=write_failures,
        )

    if args.out_failures is not None:
        args.out_failures.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"[judge] {client.spend.report()}", file=sys.stderr)
    print(f"[judge] wrote {len(records)} records -> {args.out}", file=sys.stderr)
    if failures:
        print(
            f"[judge] {len(failures)} case(s) FAILED and are absent from the output -- "
            "they are not scored as clean; see --out-failures",
            file=sys.stderr,
        )
        for key, err in sorted(failures.items()):
            print(f"[judge]   {key}: {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
