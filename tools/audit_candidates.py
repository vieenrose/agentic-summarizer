"""Audit what a RAFT reward SELECTS, before spending a training round finding out.

    python tools/audit_candidates.py --candidates data/staging/raft2_candidates.jsonl \
        --gold data/staging/sft_pool_v18.jsonl

**Why this exists.** Round 1 of RAFT produced a pool that looked excellent on the two things
it was audited for — grounding (6.4% ungrounded against gold's 45.1%) and abstention (NOP
22.4% against 48.2%) — and trained a checkpoint whose churn was **44.7%**, four times over
G7's ceiling, on both seeds. The defect was in the pool the whole time and in a column nobody
looked at: the kept rows carried **0.65 drops per add against gold's 0.37**, because the
reward credited every applied op at +1 and a bare `DROP` therefore counted as work.

Cost: two training runs, four GGUF exports and four evaluations to learn something computable
from the candidate file in about a second.

**What it reports, and why each column.** The comparison that matters is not "are the winners
good" but **"how does SELECTION move each statistic relative to the candidate pool"** — the
reward's whole job is to move them, and the direction of the move is the reward's signature:

* `drop/add` — the round-1 defect. Selection should pull this DOWN toward gold. Round 1 pushed
  it up to 0.65; the corrected reward pulls it to 0.38 against gold's 0.37.
* `arc/step` — the same exploit one op over. `ARC` replaces a single slot rather than
  accumulating, so if it is the cheapest full credit the policy will emit one every step.
  Round 1's pool ran 1.56x gold.
* `churn`, `ungrounded`, `near_duplicates` — the three penalties. Selection should drive all
  three toward zero; if it does not, the penalty is not biting.
* `decode_tokens` — G4 is a wall-clock gate, so length is a priced resource. Winners are
  expected to be somewhat LONGER than average (they record more) but the ratio is worth
  seeing, since a reward additive in applied ops silently selects the most verbose candidate.
* `adds/step` — the thing RAFT is actually for. It must go UP, or starvation is not being
  fixed and the whole exercise is a churn experiment.

**Read the WINNERS row against GOLD, not against zero.** Gold is the incumbent the sampler
had to beat; a pool that differs wildly from it in op mix is a pool whose trained behaviour
will differ wildly, whatever its headline rates say.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
from pathlib import Path

TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def _count(value: object) -> int:
    """`add`/`drop` may be a list or a bare scalar depending on what the model emitted.

    Assuming a list here raised `TypeError: object of type 'int' has no len()` on the first
    real candidate file — a model that emits `"drop": 3` instead of `"drop": [3]` is
    well-formed enough for the harness to apply, so the audit has to handle it too.
    """
    if value is None:
        return 0
    return len(value) if isinstance(value, list | tuple) else 1


def op_mix(rows: list[dict]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for row in rows:
        match = TOOL_CALL.search(row.get("completion", ""))
        if not match:
            continue
        try:
            args = json.loads(match.group(1))["arguments"]
        except Exception:
            continue
        counts["add"] += _count(args.get("add"))
        counts["drop"] += _count(args.get("drop"))
        counts["revise"] += _count(args.get("revise"))
        counts["arc"] += 1 if args.get("arc") else 0
        counts["steps"] += 1
    return counts


def _mean_present(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if key in r]
    return (sum(vals) / len(vals)) if vals else None


def profile(name: str, rows: list[dict]) -> dict:
    mix = op_mix(rows)
    steps = max(mix["steps"], 1)
    return {
        "arm": name,
        "n": len(rows),
        "adds_per_step": mix["add"] / steps,
        "drops_per_step": mix["drop"] / steps,
        "arc_per_step": mix["arc"] / steps,
        "revise_per_step": mix["revise"] / steps,
        "drop_over_add": mix["drop"] / max(mix["add"], 1),
        # Same caveat: these are SCORE fields. On gold pool rows they are absent, and a 0.0
        # would be read as "gold never churns" when the truth is "gold was never scored".
        # `replay` in `build_raft_pool`'s audit is where gold's real churn comes from.
        "churn_per_step": _mean_present(rows, "churn"),
        "ungrounded_per_step": _mean_present(rows, "ungrounded"),
        "near_dup_per_step": _mean_present(rows, "near_duplicates"),
        # `None`, not 0.0, when the field is absent. Gold POOL rows carry no score breakdown
        # -- they were never scored as candidates -- and printing 0.0 there reads as "gold
        # emits no tokens", which is exactly the kind of confidently wrong cell this tool
        # exists to catch.
        "decode_tokens": (
            st.mean([r["decode_tokens"] for r in rows if "decode_tokens" in r])
            if any("decode_tokens" in r for r in rows)
            else None
        ),
        "score": st.mean([r["score"] for r in rows]) if rows and "score" in rows[0] else 0.0,
    }


def _fmt(v: float | None, width: int, prec: int) -> str:
    return ("-").rjust(width) if v is None else format(v, f"{width}.{prec}f")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--candidates", type=Path, required=True, help="raft_reading --save-candidates")
    p.add_argument("--gold", type=Path, default=None, help="the source pool, as the incumbent")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["meeting"], row["step"])].append(row)
    winners = [max(v, key=lambda x: x["score"]) for v in groups.values()]

    arms = [profile("all candidates", rows), profile("WINNERS", winners)]
    if args.gold is not None:
        gold_rows = [
            json.loads(line)
            for line in args.gold.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keys = {(m, s) for m, s in groups}
        gold = [
            r
            for r in gold_rows
            if "<tool_call>" in r.get("completion", "")
            and (r.get("meeting"), r.get("step")) in keys
        ]
        arms.append(profile("gold (incumbent)", gold))

    cols = [
        "arm",
        "n",
        "adds_per_step",
        "drops_per_step",
        "drop_over_add",
        "arc_per_step",
        "churn_per_step",
        "ungrounded_per_step",
        "near_dup_per_step",
        "decode_tokens",
    ]
    print(
        f"{'arm':17}{'n':>7}{'add/st':>8}{'drop/st':>9}{'drop/add':>10}{'arc/st':>8}"
        f"{'churn':>8}{'ungrnd':>8}{'neardup':>9}{'decode':>8}"
    )
    for a in arms:
        print(
            f"{a['arm']:17}{a['n']:7}{a['adds_per_step']:8.2f}{a['drops_per_step']:9.2f}"
            f"{a['drop_over_add']:10.2f}{a['arc_per_step']:8.2f}"
            f"{_fmt(a['churn_per_step'], 8, 3)}{_fmt(a['ungrounded_per_step'], 8, 3)}"
            f"{_fmt(a['near_dup_per_step'], 9, 3)}"
            f"{'       -' if a['decode_tokens'] is None else format(a['decode_tokens'], '8.1f')}"
        )
    print("\nSELECTION should move drop/add DOWN toward gold, churn/ungrounded/near-dup toward")
    print("zero, and add/step UP. Round 1's broken reward pushed drop/add to 0.65 vs gold 0.37")
    print("and produced 44.7% churn at serving.")
    if args.json:
        args.json.write_text(
            json.dumps({"arms": arms, "columns": cols}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
