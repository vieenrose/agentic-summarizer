"""Merge Phase-4 long-meeting supervision into the pilot SFT pool (SPEC §4.2, §8 risk 3).

    python tools/mix_phase4.py --pilot data/staging/sft_pool.jsonl \\
        --new data/p4_supervision --out data/staging/sft_pool_p4.jsonl --min-nop 0.32

Two problems this exists to solve, both measured rather than anticipated.

**1. The new supervision is NOP-poor and would recreate dropv1's churn.** These are the
longest meetings, and their chunks are nearly all covered by gold item spans, so the
teacher emits edits almost everywhere: ~2% NOP against the pilot's 38.2%. Adding all
2,706 samples lands the pool at 23.6% NOP -- BELOW the 25.7% that produced dropv1's
churn regression (DROP followed by a near-identical re-ADD, burning 45 of a 53-step
meeting on one topic). `downsample_nop` cannot rescue this: it only ever LOWERS the NOP
share. So the merge is capped by arithmetic, admitting new samples only while the pool
stays at or above `--min-nop`.

Admission is ordered by DESCENDING STEP INDEX, because the deficit being repaired is
specifically late-step behaviour: the pilot has ~53 samples at index >= 40 (1.6%), and
even an 830-sample budget spent there is roughly a 15x increase in that regime.

**2. Roughly half the teacher's gold steps do not replay cleanly.** Measured over the
first 12 meetings: 48% of steps had at least one op the harness refuses -- dominated by
"point too long" (over POINT_TOKENS), then "duplicate point", then "arc unchanged".
SPEC §4.2 is explicit that "a sequence that fails replay is regenerated or dropped",
because a target the harness would refuse teaches the model to emit something the
harness refuses. Each completion is therefore replayed here and rewritten to contain
only the ops that actually applied. A step left with nothing is DROPPED rather than
rewritten to NOP -- turning a refused edit into a NOP would teach "nothing to record"
about a chunk the teacher thought was worth recording, which is a different and false
lesson.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.chunker import CHUNK_TOKENS, iter_chunks  # noqa: E402
from arcsum.guards import apply_ops  # noqa: E402
from arcsum.memory import Memory  # noqa: E402
from arcsum.ops import parse_ops, render_op  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402
from arcsum.transcript import parse_transcript  # noqa: E402


def clean_meeting(rows: list[dict], transcript: Path) -> tuple[list[dict], dict]:
    """Replay one meeting's gold rows and keep only the ops that actually applied."""
    utts = parse_transcript(transcript.read_text(encoding="utf-8"))
    chunks = list(iter_chunks(utts, budget=CHUNK_TOKENS, token_len=heuristic_token_len))
    mem = Memory(token_len=heuristic_token_len)
    kept: list[dict] = []
    stats = {"in": len(rows), "dropped_empty": 0, "ops_removed": 0}

    for r in sorted(rows, key=lambda x: x["step"]):
        if r["step"] >= len(chunks):
            continue
        ops = parse_ops(r["completion"])
        outcome = apply_ops(mem, ops, chunks[r["step"]])
        good = [a.op for a in outcome.results if a.applied]
        stats["ops_removed"] += sum(1 for a in outcome.results if not a.applied)
        if r["is_nop"]:
            kept.append(r)
            continue
        if not good:
            stats["dropped_empty"] += 1
            continue
        rendered = "\n".join(render_op(o) for o in good).strip()
        if not rendered:
            stats["dropped_empty"] += 1
            continue
        kept.append({**r, "completion": rendered})
    return kept, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True, help="dir of per-meeting jsonl")
    p.add_argument("--corpus", type=Path, default=REPO / "data/p4_zh")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-nop", type=float, default=0.32)
    p.add_argument(
        "--hold-nop",
        action="store_true",
        help="admit NOP and non-NOP together so the pool's NOP share is UNCHANGED",
    )
    args = p.parse_args(argv)

    pilot = [
        json.loads(ln)
        for ln in args.pilot.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    p_nop = sum(1 for r in pilot if r["is_nop"])
    print(f"[mix] pilot {len(pilot)} samples, NOP {100 * p_nop / len(pilot):.1f}%", file=sys.stderr)

    new: list[dict] = []
    agg = {"in": 0, "dropped_empty": 0, "ops_removed": 0}
    for f in sorted(args.new.glob("*.jsonl")):
        rows = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        cleaned, st = clean_meeting(rows, args.corpus / f"{f.stem}.txt")
        new.extend(cleaned)
        for k in agg:
            agg[k] += st[k]
    n_nop = sum(1 for r in new if r["is_nop"])
    print(
        f"[mix] new {agg['in']} -> {len(new)} after replay-clean "
        f"({agg['ops_removed']} ops removed, {agg['dropped_empty']} steps dropped empty), "
        f"NOP {100 * n_nop / max(1, len(new)):.1f}%",
        file=sys.stderr,
    )

    # Late steps first: that is the regime the pilot is missing.
    new.sort(key=lambda r: -r["step"])
    admitted: list[dict] = []
    nop = p_nop
    total = len(pilot)

    if args.hold_nop:
        # Measured on sft-dropv4: admitting late samples under a 32% FLOOR let the
        # pool's NOP drift 34.9% -> 32.0%, and that build regressed the SHORT meetings
        # (<400 lines: 10/11 wins -> 6/11) while fixing the long ones exactly as
        # intended (>=400 lines: 4/9 -> 8/9, mean +0.012 -> +0.217; corr(length,
        # change) = +0.671). A lower NOP share makes the model readier to edit, which
        # is the wrong trade on a meeting with few chunks. So hold the share EXACTLY:
        # take every new NOP sample, then as many late non-NOP samples as that budget
        # buys. This admits MORE long-meeting data than the floor did, not less.
        share = p_nop / len(pilot)
        new_nop = [r for r in new if r["is_nop"]]
        new_other = [r for r in new if not r["is_nop"]]
        # (p_nop + k_nop) / (len(pilot) + k_nop + k_other) == share
        k_nop = len(new_nop)
        k_other = round((p_nop + k_nop) / share) - len(pilot) - k_nop
        admitted = new_nop + new_other[: max(0, k_other)]
        nop = p_nop + k_nop
        total = len(pilot) + len(admitted)
    else:
        for r in new:
            nn, tt = nop + (1 if r["is_nop"] else 0), total + 1
            if nn / tt < args.min_nop:
                break
            admitted.append(r)
            nop, total = nn, tt

    merged = pilot + admitted
    with args.out.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    late_pilot = sum(1 for r in pilot if r["step"] >= 40)
    late_new = sum(1 for r in admitted if r["step"] >= 40)
    print(
        f"[mix] admitted {len(admitted)}/{len(new)} new samples "
        f"(min step index {min((r['step'] for r in admitted), default=0)})",
        file=sys.stderr,
    )
    print(
        f"[mix] merged {len(merged)} samples, NOP {100 * nop / total:.1f}% "
        f"(floor {100 * args.min_nop:.0f}%)",
        file=sys.stderr,
    )
    print(
        f"[mix] step index >=40: pilot {late_pilot} -> merged {late_pilot + late_new} "
        f"({(late_pilot + late_new) / max(1, late_pilot):.1f}x)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
