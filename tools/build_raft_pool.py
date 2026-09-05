"""Compose the RAFT training pool: on-policy reading rows + a grounded synthesis slice.

    python tools/build_raft_pool.py \
        --raft data/staging/raft_long.jsonl \
        --out data/staging/sft_pool_raft.jsonl \
        --report runs/raft-pool.json

`raft_reading.py` emits its winning READING rows and passes every other row in the source
pool through untouched. That passthrough is the part that needs a decision, because the
checkpoint this pool trains FROM (`rl-v3`) earned its synthesis quality from GRPO —
retention 0.929, 2.5% ungrounded — and the passed-through synthesis targets did not.

**What the reading rows fix, measured on the 882 steps where a gold row exists for the
same (meeting, step), both scored against the chunk the POLICY actually saw:**

| | specifics | ungrounded | empty/NOP | adds/step |
|---|---|---|---|---|
| RAFT-kept | 938 | **6.4%** | **22.4%** | 1.40 |
| gold | 949 | **45.1%** | 48.2% | 0.90 |

Two things follow. The obvious one is starvation: `rl-v3` NOPs 46.2% of chunks on the
held-out 40 and starves 17 of them (8 of the 10 meetings over 20 chunks), and gold teaches
exactly that rate. The less obvious one is that **the reading-step gold has the same
"target is not a function of its input" defect that was found and repaired for
`SYNTHESIZE` and never checked here**: §4.2 has the teacher convert aligned segment
MINUTES into ops, so a gold `ADD` may legitimately carry a figure the chunk does not
contain — but at serving time the model has only the chunk, so the row demonstrates
inventing detail from context it will not have. RAFT repairs it by construction: it
samples on-policy from the chunk and `step_reward.UNGROUNDED_PENALTY` prices the invention.

**The synthesis slice is FILTERED, and `runs/clean-e3`'s refutation of filtering does not
apply.** That experiment removed 37% of synthesis rows, concentrated at high occupancy —
the regime that already fails — and the model stopped asserting specifics at all. The
mechanism is absent here, measured by occupancy:

| entries in prompt | rows | kept | ungrounded |
|---|---|---|---|
| 0-8 | 106 | 74.5% | 34.5% |
| 9-16 | 285 | 76.8% | 18.1% |
| 17-28 | 64 | **100%** | **0.0%** |
| 29+ | 24 | **100%** | **0.0%** |

Every row above 16 entries survives, because those are the regenerated journal-shaped rows
(`gen_journal_synth.py`, 0 ungrounded across 2,015 specifics). The ungrounded rows are the
older low-occupancy teacher-written ones. So the filter removes exactly the rows that teach
fabrication and keeps exactly the rows that teach journal-scale synthesis — the opposite of
`clean-e3`'s cut. **Re-check this table if the source pool changes**; the filter is only
safe as long as the loss stays off the high-occupancy end.

Baseline map/reduce rows pass through untouched: SPEC §5.2 requires the control arm to run
the SAME model, so dropping them would make "same model" false and put a confound in the
ship decision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from arcsum.evalkit import grounding

#: Rows whose prompt starts with this are `SYNTHESIZE` (`build_synth_prompt` is
#: `build_memory_view`). Classifying by "has no CHUNK" instead sweeps up the baseline's
#: reduce rows -- done once in this project, and harmless only because there were 4 of them.
SYNTH_PREFIX = "MEMORY:"


def row_kind(row: dict) -> str:
    if "<tool_call>" in row.get("completion", ""):
        return "reading"
    if row.get("prompt", "").startswith(SYNTH_PREFIX):
        return "synthesize"
    if row.get("prompt", "").startswith("SUMMARIES:"):
        return "baseline_reduce"
    return "baseline_map"


def occupancy(prompt: str) -> int:
    """Memory entries visible in a synthesis prompt, counting both renderings."""
    return prompt.count("\n- ") + prompt.count("] ")


def build(rows: list[dict]) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    kinds: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    occ_kept: Counter[str] = Counter()
    occ_all: Counter[str] = Counter()

    for row in rows:
        kind = row_kind(row)
        kinds[kind] += 1
        if kind != "synthesize":
            kept.append(row)
            continue
        bucket = _bucket(occupancy(row["prompt"]))
        occ_all[bucket] += 1
        report = grounding.check("", row["completion"], row["prompt"])
        if report.n_ungrounded:
            dropped["synthesize_ungrounded"] += 1
            continue
        kept.append(row)
        occ_kept[bucket] += 1

    report = {
        "in_rows": len(rows),
        "out_rows": len(kept),
        "kinds_in": dict(kinds),
        "kinds_out": dict(Counter(row_kind(r) for r in kept)),
        "dropped": dict(dropped),
        "synthesis_retention_by_occupancy": {
            b: {"rows": occ_all[b], "kept": occ_kept[b]} for b in sorted(occ_all)
        },
    }
    return kept, report


def _bucket(n: int) -> str:
    if n <= 8:
        return "00-08"
    if n <= 16:
        return "09-16"
    if n <= 28:
        return "17-28"
    return "29+"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raft", type=Path, required=True, help="raft_reading.py output")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.raft.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept, report = build(rows)

    with args.out.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.report is not None:
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
