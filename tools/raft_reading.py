"""RAFT for the READING step: sample N candidates per step, keep the best, emit SFT rows.

    # 1. serve the current policy
    llama-server -m runs/rl-v3/gguf_final/model.Q8_0.gguf --port 8089 --no-jinja
    # 2. sample and filter
    python tools/raft_reading.py --url http://127.0.0.1:8089 \\
        --pool data/staging/sft_pool_v18.jsonl --out data/staging/raft_reading.jsonl \\
        --samples 8 --report runs/raft-reading.json
    # 3. train on the kept rows with the EXISTING trainer, unchanged
    python tools/train_toolcalls.py --unsloth --train data/staging/raft_reading.jsonl ...

**What this attacks.** GRPO on `SYNTHESIZE` fixed retention and grounding and could not touch
CHURN, because a terminal reward carries no signal about which of ~15 reading steps caused it.
This scores each reading step against the harness's own `Outcome` — the same
`restates_dropped` detector `evalkit.behaviour` reports the metric with — so the training
signal and the gate cannot drift apart.

**Why RAFT and not more GRPO.** Rejection sampling needs ONLY the policy: no frozen reference
model, no KL term, no full-vocabulary logits for a loss. GRPO on this model OOMed at 4
generations because the 248k-vocab logits tensor is the binding constraint, which also blocked
larger groups and multi-GPU. Sampling is just inference against a served GGUF, so N is limited
by time rather than memory, and the kept rows are ordinary SFT rows that the existing trainer
consumes unchanged. Reported competitive with GRPO/PPO, with GRPO's edge attributed mainly to
discarding all-wrong prompts rather than to reward normalisation (arXiv 2504.11343).

**Sampling is ON-POLICY, and the first version of this tool was not — which made it useless
for its own purpose.** Sampling from the pool's stored prompts measured 0 churn removed over 45
steps, for a reason that is obvious afterwards: the pool is already 0% churn (verified with the
harness's own detector), so there is no churn in gold states to fix. **The model churns on the
states IT reaches**, not on the teacher's. So each step is prompted from the memory the policy
has actually built — `build_step_prompt(memory, chunk)`, the same call `run_agent` makes — and
memory advances with the WINNING candidate, not with gold.

**Gold competes as one of the candidates.** Advancing purely on the policy's own output would
let a trajectory drift somewhere the pool never labelled; including gold in the group means the
teacher wins whenever it deserves to, and the emitted row is whatever actually scored best.
Ties go to gold, so the teacher is the incumbent and has to be beaten, not merely matched.

**Zero-variance groups are dropped and counted.** If every sample scores identically there is
nothing to learn from the comparison; the literature reports 28-45% of groups collapsing this
way in standard GRPO, and this run measures its own rate rather than assuming.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.chunker import Chunk  # noqa: E402
from arcsum.guards import apply_ops  # noqa: E402
from arcsum.memory import Memory  # noqa: E402
from arcsum.prompts import (  # noqa: E402
    TOOLCALL_PROMPT_VERSION,
    build_step_prompt,
    tool_step_system_prompt,
)
from arcsum.rl.step_reward import op_shape, score_step  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402
from arcsum.toolcalls import parse_tool_calls  # noqa: E402
from arcsum.transcript import Utterance, parse_line  # noqa: E402


def chunk_from_row(row: dict) -> Chunk:
    """The REAL chunk the step read, recovered from the stored prompt.

    **A stub chunk silently poisons everything.** The first on-policy version built the prompt
    with a placeholder utterance, so the model was asked to summarise `S1: x` — and still
    emitted 80 confident `ADD`s, which the reward ranked highly because nothing checked them
    against a source. Every kept row would have paired a contentless prompt with detailed
    content: a machine for teaching hallucination.

    The chunk text is the tail of the stored prompt after `CHUNK:`; `parse_line` is total, so a
    line without a speaker still yields an utterance rather than raising.
    """
    body = row["prompt"].partition("CHUNK:")[2].strip("\n")
    # `parse_line` returns a (speaker, text) TUPLE, not an Utterance.
    utterances = [Utterance(*parse_line(ln)) for ln in body.splitlines() if ln.strip()]
    return Chunk(index=int(row["step"]), utterances=utterances, tokens=heuristic_token_len(body))


def evaluate(memory: Memory, raw: str, chunk: Chunk, chunk_text: str) -> tuple[float, dict, list]:
    """Apply a candidate to a CLONE and score what the harness did with it."""
    ops = parse_tool_calls(raw)
    if not ops:
        return float("-inf"), {}, []
    trial = memory.clone()
    outcome = apply_ops(trial, ops, chunk, lang_check=False)
    # `raw` is passed so the reward can price DECODE TOKENS. Without it the score is additive
    # in applied ops and the most verbose candidate wins: the first pass kept rows averaging
    # 78.9 completion tokens against gold's 54.4, a 1.45x decode cost that alone moves a
    # meeting from 20.3 to 22.5 minutes against G4's 20.00 ceiling.
    s = score_step(
        outcome, ops, chunk_text=chunk_text, chunk_has_content=bool(chunk_text.strip()), raw=raw
    )
    return (
        s.score,
        {
            "applied": s.applied,
            "refused": s.refused,
            "churn": s.churn,
            "malformed": s.malformed,
            "revised": s.revised,
            "ungrounded": s.ungrounded,
            "decode_tokens": s.decode_tokens,
            "near_duplicates": s.near_duplicates,
        },
        ops,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8089")
    p.add_argument(
        "--samples",
        type=int,
        default=8,
        help="candidates per step. Memory-free here — this is inference against a "
        "served model, so N trades against TIME, not VRAM as it does in GRPO.",
    )
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--meetings", type=int, default=0)
    p.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="only sample meetings with at least this many reading steps. Churn is a "
        "LONG-MEETING behaviour — it needs a memory full enough to re-add into — and "
        "measured on three 16-step meetings the policy churned ZERO times, so sampling "
        "short meetings spends generations where the target behaviour cannot occur.",
    )
    p.add_argument("--max-steps", type=int, default=0, help="cap steps per meeting; 0 = all")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    rows = [
        json.loads(ln) for ln in args.pool.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if "CHUNK:" in r["prompt"] and "tool_call" in r["completion"]:
            by[r["meeting"]].append(r)
    meetings = sorted(by)
    if args.min_steps:
        meetings = [m for m in meetings if len(by[m]) >= args.min_steps]
        print(f"[raft] {len(meetings)} meetings with >= {args.min_steps} steps", file=sys.stderr)
    if args.meetings:
        meetings = meetings[: args.meetings]

    stats = collections.Counter()
    kept_rows: list[dict] = []
    shapes_gold: collections.Counter = collections.Counter()
    shapes_kept: collections.Counter = collections.Counter()

    for i, meeting in enumerate(meetings, 1):
        steps = sorted(by[meeting], key=lambda x: int(x["step"]))
        memory = Memory(token_len=heuristic_token_len)
        system = tool_step_system_prompt()
        total = len(steps)
        for row in steps[: args.max_steps or None]:
            # step index comes from the row via chunk_from_row
            # ON-POLICY: the prompt renders the memory the POLICY has built, via the same
            # builder `run_agent` uses. The stored `row["prompt"]` shows the teacher's state
            # and is deliberately not used.
            chunk = chunk_from_row(row)
            chunk_text = "\n".join(f"{u.speaker}: {u.text}" for u in chunk.utterances)
            prompt = build_step_prompt(memory, chunk, total=total)
            gold_score, gold_detail, gold_ops = evaluate(
                memory, row["completion"], chunk, chunk_text
            )
            stats["steps"] += 1
            shapes_gold.update(op_shape(gold_ops))
            stats["gold_churn"] += gold_detail.get("churn", 0)
            # **Gold is scored against the POLICY's memory, not its own.** Once the trajectory
            # diverges, a gold `drop: [3]` refers to an id the policy never minted, and the
            # harness refuses it — so gold loses on a technicality rather than on quality, and
            # "improved" inflates. These two counters are what make that visible: if
            # `gold_refused` runs far above the on-trajectory baseline (10.8%, measured), the
            # improvement rate is an artifact of divergence and not evidence about the policy.
            stats["gold_refused"] += gold_detail.get("refused", 0)
            stats["gold_ops"] += gold_detail.get("applied", 0) + gold_detail.get("refused", 0)

            cands = []
            for k in range(args.samples):
                c = LlamaServer(
                    base_url=args.url,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=k,
                    raw_completion=True,
                    extra={"cache_prompt": False},
                )
                try:
                    raw = c(system, prompt)
                except Exception as exc:
                    # Record the CAUSE, not just a count. A bare tally read as "2.7% improved"
                    # on a run where 5,400 of 5,772 samples had failed — the server had been
                    # started with `-np 4`, which DIVIDES `-c` among slots, so every prompt
                    # over 2,048 tokens was rejected and the surviving 6% looked healthy.
                    stats["sample_errors"] += 1
                    stats[f"err::{type(exc).__name__}"] += 1
                    if stats["sample_errors"] <= 3:
                        print(f"[raft] sample error: {str(exc)[:200]}", file=sys.stderr)
                    continue
                sc, detail, ops = evaluate(memory, raw, chunk, chunk_text)
                stats["policy_churn"] += detail.get("churn", 0)
                stats["policy_ungrounded"] += detail.get("ungrounded", 0)
                stats["policy_refused"] += detail.get("refused", 0)
                stats["policy_ops"] += detail.get("applied", 0) + detail.get("refused", 0)
                cands.append((sc, detail, ops, raw))

            scores = [c[0] for c in cands if c[0] != float("-inf")]
            if not cands:
                stats["no_candidates"] += 1
            elif scores and max(scores) == min(scores) and len(scores) == len(cands):
                stats["zero_variance_groups"] += 1

            best = max(cands, key=lambda c: c[0]) if cands else None
            if best is not None and best[0] > gold_score:
                chosen_raw, chosen_ops, chosen_detail = best[3], best[2], best[1]
                stats["improved"] += 1
                stats["churn_removed"] += max(
                    0, gold_detail.get("churn", 0) - chosen_detail.get("churn", 0)
                )
                gain = round(best[0] - gold_score, 3)
            else:
                chosen_raw, chosen_ops, chosen_detail = (row["completion"], gold_ops, gold_detail)
                stats["gold_kept"] += 1
                gain = 0.0

            kept_rows.append(
                {
                    **row,
                    "prompt": prompt,
                    "system": system,
                    "completion": chosen_raw,
                    "prompt_version": TOOLCALL_PROMPT_VERSION,
                    "raft_gain": gain,
                }
            )
            shapes_kept.update(op_shape(chosen_ops))
            # Advance with what was CHOSEN — this is what makes the next step's state
            # on-policy rather than a replay of the teacher.
            apply_ops(memory, parse_tool_calls(chosen_raw), chunk, lang_check=False)

        if i % 10 == 0:
            print(
                f"[raft] {i}/{len(meetings)} meetings | improved {stats['improved']} "
                f"/ {stats['steps']} steps",
                file=sys.stderr,
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Non-reading rows pass through untouched so the output is a COMPLETE pool.
    with args.out.open("a", encoding="utf-8") as f:
        for r in rows:
            if not ("CHUNK:" in r["prompt"] and "tool_call" in r["completion"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "pool": str(args.pool),
        "out": str(args.out),
        "url": args.url,
        "samples": args.samples,
        "temperature": args.temperature,
        **stats,
        "improved_rate": round(stats["improved"] / max(stats["steps"], 1), 4),
        "gold_refusal_rate": round(stats["gold_refused"] / max(stats["gold_ops"], 1), 4),
        "policy_refusal_rate": round(stats["policy_refused"] / max(stats["policy_ops"], 1), 4),
        "zero_variance_rate": round(stats["zero_variance_groups"] / max(stats["steps"], 1), 4),
        "op_shape_gold": dict(shapes_gold),
        "op_shape_kept": dict(shapes_kept),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
