"""Remove measured pathologies from an SFT pool, instead of adding data to outweigh them.

    python tools/clean_pool.py --pool data/staging/sft_pool_mixed.jsonl \\
        --reversals data/staging/sft_pool_revfix.jsonl \\
        --max-ungrounded 2 --out data/staging/sft_pool_clean.jsonl

**Why this exists.** Every iteration since `qwen-tools-v3` ADDED supervision — synthetic
reversals, deliberation examples, hedge rows, detail rows, self-distilled synthesis — and
four of the last five moved one axis while breaking another. The pathologies the student
exhibits were then found IN the pool, counted with the harness's own guards, so they can be
removed rather than outweighed. Full measurement in `runs/next-iteration-plan.md`.

Four surgeries, each reported with a before/after count so a build can never silently skip
one (`build_sft`'s recorded lesson: reading the shares a tool reports is necessary, not
sufficient — check that what you asked for is what landed):

* **S1 grounding filter.** 39.7% of the specific claims in synthesis targets are absent
  from the memory those targets were written from — the target was composed from the
  whole-meeting gold summary rather than from the input, so it is not a function of its
  input. `--max-ungrounded` sets the per-row tolerance. **A filter alone cannot fully fix
  this**: dropping every offending row removes 45% of synthesis rows, concentrated at high
  occupancy, which is the regime that already fails. Filtering is step one; regenerating
  targets from the memory alone is the real repair, and is not done here.
* **S2 churn.** Rows where an applied `ADD` merely restates a point `DROP`ped in the same
  step, detected with `guards.restates_dropped` — the same function the harness uses at
  inference, so the training filter and the runtime detector cannot drift. Dropped, never
  rewritten: `mix_phase4.py`'s rule is that an unrepairable step is dropped, never turned
  into a NOP it did not earn.
* **S3 no-op ARC.** Targets re-emitting the previous step's ARC verbatim. The harness
  refuses every one as `arc unchanged`, so these teach an op rejected 100% of the time.
  The `arc` key is stripped and the row's `add`/`drop` kept — the row still teaches valid
  edits.
* **S4 reversal swap.** Replace the pool's reversal rows with a corrected set. The pool's
  originals lost the identifying detail in the replacement point (0 of 34 preserving
  `key_term`); the corrected rows preserve it 26/26.

**Deliberately NOT done here: reweighting or oversampling.** `sft-dropv3` held NOP and DROP
shares nearly constant and still regressed, because what changed was WHICH samples carried
those labels. This tool only removes and repairs; it never duplicates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.evalkit import grounding  # noqa: E402
from arcsum.guards import restates_dropped  # noqa: E402
from arcsum.memory import normalize  # noqa: E402

TOOL_CALL = re.compile(r'\{"name".*\}', re.S)


def is_reading(row: dict) -> bool:
    return "CHUNK:" in row["prompt"]


def parse_args_json(completion: str) -> dict | None:
    m = TOOL_CALL.search(completion)
    if not m:
        return None
    try:
        return json.loads(m.group(0)).get("arguments") or {}
    except (json.JSONDecodeError, AttributeError):
        return None


def render_call(args: dict) -> str:
    return ('<tool_call>{"name": "update_memory", "arguments": '
            + json.dumps(args, ensure_ascii=False) + "}</tool_call>")


def prompt_points(prompt: str) -> list[str]:
    body = prompt.split("POINTS:")[-1].split("\nCHUNK:")[0]
    return [ln[2:].strip() for ln in body.splitlines() if ln.startswith("- ")]


def dropped_texts(args: dict, prompt: str) -> list[str]:
    """Full text of each point this step's `drop` prefixes removed.

    Resolved against the memory rendered in the PROMPT rather than the prefix alone,
    because `restates_dropped` compares against what was actually removed — a prefix is by
    design too short to judge similarity on.
    """
    points = prompt_points(prompt)
    out = []
    for d in args.get("drop") or []:
        key = normalize(d)
        hit = next((p for p in points if normalize(p).startswith(key)), None)
        if hit:
            out.append(hit)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reversals", type=Path, default=None,
                   help="corrected reversal rows; replaces rows whose meeting starts 'rev'")
    p.add_argument("--synth", type=Path, default=None,
                   help="regenerated SYNTHESIZE rows from tools/regen_synth.py. Substituted "
                        "by (meeting, step), and REPAIR IS PREFERRED TO FILTERING: rows "
                        "supplied here bypass the --max-ungrounded cut, because deleting a "
                        "fabricating row deletes its content too, which starves the model "
                        "(measured, runs/clean-e3: specifics 26 -> 5 across 20 meetings)")
    p.add_argument("--max-ungrounded", type=int, default=2,
                   help="per synthesis row, specifics allowed to be absent from its memory")
    p.add_argument("--max-ungrounded-rate", type=float, default=None,
                   help="preferred over --max-ungrounded: an ABSOLUTE count penalises "
                        "high-occupancy rows mechanically, because a target written from "
                        "more memory asserts more claims. Measured: an absolute cap of 2 "
                        "retains 84%% of 1-5-point rows but only 65%% of 13-16-point ones "
                        "-- thinning exactly the regime the synthesis cliff lives in.")
    p.add_argument("--skip-churn", action="store_true")
    p.add_argument("--skip-arc", action="store_true")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    raw = args.pool.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(ln) for ln in raw if ln.strip()]
    before = {"total": len(rows),
              "reading": sum(1 for r in rows if is_reading(r)),
              "synthesis": sum(1 for r in rows if not is_reading(r))}

    # --- S4: swap reversal rows first, so later surgeries see the corrected ones ---
    swapped_in = swapped_out = 0
    if args.reversals:
        new_rev = [json.loads(ln) for ln in
                   args.reversals.read_text(encoding="utf-8").splitlines() if ln.strip()]
        new_rev = [r for r in new_rev if r["meeting"].startswith("rev")]
        kept = [r for r in rows if not r["meeting"].startswith("rev")]
        swapped_out, swapped_in = len(rows) - len(kept), len(new_rev)
        rows = kept + new_rev

    # --- S1b: substitute regenerated SYNTHESIZE targets ---------------------------
    # Done BEFORE the grounding filter so a repaired row is judged on its NEW target.
    regenerated: set[tuple[str, str]] = set()
    swapped_synth = 0
    if args.synth:
        new_syn = {}
        for ln in args.synth.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("regen"):
                new_syn[(r["meeting"], str(r["step"]))] = r
        for i, r in enumerate(rows):
            key = (r["meeting"], str(r["step"]))
            if r["prompt"].startswith("MEMORY:") and key in new_syn:
                rows[i] = new_syn[key]
                regenerated.add(key)
                swapped_synth += 1

    # --- S3: strip an ARC identical to the previous step's, per meeting ---
    arc_stripped = 0
    if not args.skip_arc:
        prev_arc: dict[str, str] = defaultdict(str)
        for r in sorted((r for r in rows if is_reading(r)),
                        key=lambda r: (r["meeting"], int(r["step"]))):
            a = parse_args_json(r["completion"])
            if not a or not a.get("arc"):
                continue
            if a["arc"] == prev_arc[r["meeting"]]:
                a.pop("arc")
                r["completion"] = render_call(a)
                arc_stripped += 1
            else:
                prev_arc[r["meeting"]] = a["arc"]

    # --- S2: drop churn rows ---
    churn_dropped = 0
    if not args.skip_churn:
        keep = []
        for r in rows:
            if not is_reading(r):
                keep.append(r)
                continue
            a = parse_args_json(r["completion"])
            if not a:
                keep.append(r)
                continue
            removed = dropped_texts(a, r["prompt"])
            if removed and any(restates_dropped(ad, removed) for ad in (a.get("add") or [])):
                churn_dropped += 1
                continue
            keep.append(r)
        rows = keep

    # --- S1: grounding filter on synthesis targets ---
    ungrounded_dropped = 0
    kept_by_occupancy: dict[str, list[int]] = defaultdict(list)
    keep = []
    for r in rows:
        if is_reading(r):
            keep.append(r)
            continue
        rep = grounding.check(r["meeting"], r["completion"], r["prompt"])
        pts = len(prompt_points(r["prompt"]))
        bucket = "1-5" if pts <= 5 else "6-9" if pts <= 9 else "10-12" if pts <= 12 else "13-16"
        over = (rep.ungrounded_rate > args.max_ungrounded_rate
                if args.max_ungrounded_rate is not None
                else rep.n_ungrounded > args.max_ungrounded)
        if (r["meeting"], str(r["step"])) in regenerated:
            over = False  # already repaired and verified; keep the content
        if over:
            ungrounded_dropped += 1
            kept_by_occupancy[bucket].append(0)
            continue
        kept_by_occupancy[bucket].append(1)
        keep.append(r)
    rows = keep

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    after = {"total": len(rows),
             "reading": sum(1 for r in rows if is_reading(r)),
             "synthesis": sum(1 for r in rows if not is_reading(r))}
    report = {
        "pool": str(args.pool), "out": str(args.out),
        "max_ungrounded": args.max_ungrounded,
        "max_ungrounded_rate": args.max_ungrounded_rate,
        "before": before, "after": after,
        "S1_ungrounded_synthesis_dropped": ungrounded_dropped,
        "S2_churn_rows_dropped": churn_dropped,
        "S3_noop_arc_stripped": arc_stripped,
        "S1b_synthesis_rows_regenerated": swapped_synth,
        "S4_reversal_rows_out": swapped_out, "S4_reversal_rows_in": swapped_in,
        "synthesis_retained_by_occupancy": {
            k: {"kept": sum(v), "of": len(v)} for k, v in sorted(kept_by_occupancy.items())},
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    # The occupancy table is the one to read: S1 concentrates its cuts at HIGH occupancy,
    # which is the regime the synthesis cliff already lives in. Thinning it further is a
    # real risk, and this is where it becomes visible rather than showing up as a gate
    # regression two hours later.
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
