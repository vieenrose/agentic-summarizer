#!/usr/bin/env python3
"""Calibrate FAITH against QMSum's own human reference summaries.

Our arms score FAITH-claim 2.9-3.3 of 5 with ~45% of bullets unverifiable. That number is
uninterpretable on its own: it could mean the notes are bad, or it could mean this corpus
and this metric simply do not permit better. So judge the *human* summary the dataset ships
("Summarize the whole meeting") through the identical pipeline and compare.

Fairness matters more than convenience here, so the comparison is deliberately like-for-like:

* the human summary is prose, so it is split into sentences and each sentence is treated as
  one claim — the same granularity our bullets are judged at;
* each sentence is assigned an anchor by the **same deterministic matcher** the harness uses
  for an unanchored bullet, so it receives neighbourhood + retrieved evidence exactly as our
  bullets do. Judging human sentences with retrieval only would hand them a different
  evidence budget and invalidate the comparison;
* same judge, same prompt, same claim mode.

A human score near ours means 53% verifiability is a property of the corpus and metric. A
human score much higher means it is a property of our teacher.

    TOGETHER_API_KEY=... python eval/calibrate_reference.py --meeting 16abbdf7b3f2
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import _FAITH_SYS, _VERDICT, TogetherJudge, faith_prompt  # noqa: E402

from voxsum.chunker import Chunk  # noqa: E402
from voxsum.guards import match_anchor  # noqa: E402
from voxsum.index import TranscriptIndex  # noqa: E402
from voxsum.transcript import parse_transcript, sec_to_clock  # noqa: E402

HUB = Path.home() / ".cache/huggingface/hub"
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def load_reference(meeting_hash: str, split: str = "validation") -> str | None:
    """The dataset's own whole-meeting summary for a meeting identified by body hash."""
    roots = sorted(glob.glob(str(HUB / "datasets--pszemraj--qmsum-cleaned/snapshots/*")))
    if not roots:
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(f"{roots[0]}/data/{split}-00000-of-00001.parquet")
    ids = table.column("id").to_pylist()
    inputs = table.column("input").to_pylist()
    outputs = table.column("output").to_pylist()
    for mid, raw, summary in zip(ids, inputs, outputs, strict=False):
        body = raw.split("\n", 1)[1] if "\n" in raw else raw
        if hashlib.sha1(body.encode()).hexdigest()[:12] != meeting_hash:
            continue
        if "-gq-" in mid:  # general query = "Summarize the whole meeting"
            return summary
    return None


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(" ".join(text.split())) if len(s.strip()) > 15]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--meeting", action="append", required=True, help="body hash, repeatable")
    p.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--budget-usd", type=float, default=0.20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    client = TogetherJudge(
        api_key=os.environ.get("TOGETHER_API_KEY", ""), budget_usd=args.budget_usd
    )
    results = []

    for meeting_hash in args.meeting:
        reference = load_reference(meeting_hash)
        if reference is None:
            print(f"[calib] no general-query reference for {meeting_hash}", file=sys.stderr)
            continue
        path = args.transcript_dir / f"qmsum-{meeting_hash}.txt"
        utterances = parse_transcript(path.read_text(encoding="utf-8"))
        index = TranscriptIndex(utterances)
        whole = Chunk(0, tuple(utterances))

        claims = sentences(reference)
        print(f"[calib] qmsum-{meeting_hash}: {len(claims)} reference sentences", flush=True)

        verdicts = []
        for claim in claims:
            # Same deterministic matcher the harness uses for an unanchored bullet, so the
            # reference gets the same evidence budget our bullets get.
            anchor = match_anchor(whole, claim)
            evidence = index.evidence_for(claim, anchor, mode="claim")
            text = claim + (f" [{sec_to_clock(anchor)}]" if anchor is not None else "")
            raw = client(args.model, _FAITH_SYS, faith_prompt(text, evidence))
            hits = _VERDICT.findall(raw)
            verdict = hits[-1].upper() if hits else "MISSING"
            verdicts.append({"claim": claim, "anchor": anchor, "verdict": verdict})
            print(f"[calib]   {verdict:13s} {claim[:70]!r}", flush=True)

        supported = sum(1 for v in verdicts if v["verdict"] == "SUPPORTED")
        n = len(verdicts) or 1
        faith = 1 + 4 * supported / n
        print(
            f"[calib] qmsum-{meeting_hash}: HUMAN reference {supported}/{n} supported "
            f"({supported / n:.0%}), FAITH-equivalent {faith:.2f}"
        )
        results.append(
            {
                "meeting": f"qmsum-{meeting_hash}",
                "source": "human_reference",
                "claims": n,
                "supported": supported,
                "supported_rate": supported / n,
                "faith_equivalent": faith,
                "verdicts": verdicts,
            }
        )

    print()
    print(client.spend.report())
    if results:
        mean = sum(r["supported_rate"] for r in results) / len(results)
        print(f"\nhuman reference mean supported rate: {mean:.0%}")
        print("our arms, same judge and pipeline: cursor 53-58%, baseline 47-48%")
    if args.out and results:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
