#!/usr/bin/env python3
"""Measure how much evidence ORDER changes judge verdicts (§7.3 variance source).

Fixing the retrieval bug changed anchor mode's evidence *order* without changing the evidence
*set*, and FAITH-anchor still moved 0.47 on one note-set. That is one accidental data point.
If order really is worth half a point on a 1-5 scale, it is a variance source larger than the
0.5 tie band the ship gates are judged against, and it must be held fixed across arms — or a
tier comparison partly measures presentation.

This judges the same bullets with the same evidence under several orderings and reports the
spread. Verdicts are compared per bullet, so disagreement is attributable rather than inferred
from aggregate scores.

    TOGETHER_API_KEY=... python eval/order_sensitivity.py runs/judged3/*.cursor.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import _FAITH_SYS, _VERDICT, TogetherJudge, faith_prompt  # noqa: E402

from voxsum.index import TranscriptIndex  # noqa: E402
from voxsum.transcript import parse_transcript, sec_to_clock  # noqa: E402

#: Orderings compared. `anchor_first` is what the code now produces and what we intend to pin.
ORDERINGS = ("anchor_first", "chronological", "reversed", "retrieved_first")


def reorder(evidence: list, ordering: str) -> list:
    if ordering == "anchor_first":
        return list(evidence)
    if ordering == "chronological":
        return sorted(evidence, key=lambda e: e.anchor)
    if ordering == "reversed":
        return list(reversed(evidence))
    if ordering == "retrieved_first":
        return [e for e in evidence if not e.from_anchor_neighbourhood] + [
            e for e in evidence if e.from_anchor_neighbourhood
        ]
    raise ValueError(f"unknown ordering: {ordering!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("scores", nargs="+", type=Path, help="judge.py output files")
    p.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    p.add_argument("--mode", default="claim", choices=["claim", "anchor"])
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--limit-bullets", type=int, default=12, help="per file, to bound cost")
    p.add_argument("--budget-usd", type=float, default=0.20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    client = TogetherJudge(
        api_key=os.environ.get("TOGETHER_API_KEY", ""), budget_usd=args.budget_usd
    )
    rows: list[dict] = []

    for path in args.scores:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "meeting_id" not in data:
            continue
        utterances = parse_transcript(
            (args.transcript_dir / f"{data['meeting_id']}.txt").read_text(encoding="utf-8")
        )
        index = TranscriptIndex(utterances)
        bullets = [b for b in data["bullets"] if b["mode"] == args.mode][: args.limit_bullets]
        print(f"[order] {data['meeting_id']}/{data['system']}: {len(bullets)} bullets", flush=True)

        for b in bullets:
            evidence = index.evidence_for(b["bullet"], b["anchor"], mode=args.mode)
            if len(evidence) < 2:
                continue
            anchor_txt = f" [{sec_to_clock(b['anchor'])}]" if b["anchor"] is not None else ""
            verdicts = {}
            for ordering in ORDERINGS:
                prompt = faith_prompt(b["bullet"] + anchor_txt, reorder(evidence, ordering))
                raw = client(args.model, _FAITH_SYS, prompt)
                hits = _VERDICT.findall(raw)
                verdicts[ordering] = hits[-1].upper() if hits else "MISSING"
            rows.append(
                {
                    "meeting": data["meeting_id"],
                    "system": data["system"],
                    "bullet": b["bullet"],
                    "verdicts": verdicts,
                    "stable": len(set(verdicts.values())) == 1,
                }
            )
            flag = "" if rows[-1]["stable"] else "  <-- ORDER-DEPENDENT"
            print(f"[order]   {list(verdicts.values())}{flag}", flush=True)

    if not rows:
        print("no bullets with >=2 evidence items", file=sys.stderr)
        return 2

    unstable = [r for r in rows if not r["stable"]]
    print()
    print(f"bullets judged            : {len(rows)}")
    print(f"order-dependent verdicts  : {len(unstable)} ({len(unstable) / len(rows):.0%})")
    for ordering in ORDERINGS:
        sup = sum(1 for r in rows if r["verdicts"][ordering] == "SUPPORTED")
        faith = 1 + 4 * sup / len(rows)
        print(f"  {ordering:16s} supported {sup:3d}/{len(rows)}  FAITH-equiv {faith:.2f}")
    faiths = [
        1 + 4 * sum(1 for r in rows if r["verdicts"][o] == "SUPPORTED") / len(rows)
        for o in ORDERINGS
    ]
    print(f"\nFAITH spread across orderings: {max(faiths) - min(faiths):.2f}")
    print("(compare with the 0.5 tie band the ship gates use)")

    if args.out:
        args.out.write_text(
            json.dumps({"rows": rows, "orderings": list(ORDERINGS)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
