"""What the READING step actually captured, measured on the memory it leaves behind.

    python tools/measure_memory.py --url http://127.0.0.1:8081 --label mixed-e3

**Why this exists as a committed tool.** Self-distillation (`runs/selfdistil-e3`) fixed
the synthesis cliff, the independent probe and the real-ASR gate all at once, and was
still the wrong checkpoint to ship -- because it had quietly degraded the READING step,
and no gate in SPEC §5.2 looks at the reading step in isolation. G3 sees the prose, G2
sees the claims, the ASR gate sees the summary length. All of them see the reading step
only through synthesis, which can mask a thinner memory by writing about it more fluently.

The mechanism was specific: replacing teacher-authored memories with student-authored ones
in the synthesis rows made the STUDENT'S memory the memory prior, so the reading step
regressed toward producing what it already produced. That is a self-reinforcing loop, and
the only way to see it is to look at the memory directly.

The number that moved was measured inline in a terminal and never committed, which is the
same failure `asr_gate.py` was written to stop. So this reports a small vector, not one
number, and is meant to be run on BOTH the candidate and the incumbent in the same pass --
a remembered figure whose exact definition cannot be re-derived is not evidence.

`detail_chars` is the headline: mean characters per recorded point. It is a deliberately
crude proxy for "how much did this point actually say", and it is crude in a SAFE
direction -- a model that pads points with filler scores better on it, and every failure
mode seen so far (abstention, vagueness, collapse to topic labels) moves it DOWN. Read it
alongside `numerals`, which padding does not inflate.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.agent import run_agent  # noqa: E402
from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.transcript import parse_transcript  # noqa: E402

#: Digits, CJK numerals and the units they attach to. Counted because it is the one
#: signal a model cannot inflate by writing longer, vaguer points: a specific figure is
#: either carried out of the chunk or it is not.
NUMERAL = re.compile(r"[0-9]+|[一二三四五六七八九十百千萬億零壹貳參肆伍陸柒捌玖拾]+")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--corpus", type=Path, default=REPO / "data/heldout_zh")
    p.add_argument("--n", type=int, default=5, help="meetings, taken in sorted order")
    p.add_argument("--protocol", choices=("edit", "tool"), default="tool")
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    common = {"base_url": args.url, "seed": 0, "raw_completion": True,
              "extra": {"cache_prompt": False}}
    step = LlamaServer(max_tokens=256, **common)
    prose = LlamaServer(max_tokens=1200, repeat_penalty=1.1, **common)

    files = sorted(args.corpus.glob("*.txt"))[: args.n]
    if not files:
        print(f"[memory] REFUSED: no transcripts in {args.corpus}", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        trace = run_agent(
            parse_transcript(f.read_text(encoding="utf-8")), step, synth_model=prose,
            protocol=args.protocol, on_step_error="skip",
        )
        pts = [pt.text for pt in trace.memory.points]
        chars = [len(t) for t in pts]
        rows.append({
            "meeting": f.stem, "steps": len(trace.steps),
            "nop": sum(1 for s in trace.steps if s.is_nop),
            "points": len(pts),
            "detail_chars": round(st.mean(chars), 1) if chars else 0.0,
            "numerals": sum(len(NUMERAL.findall(t)) for t in pts),
            "arc_chars": len(trace.memory.arc or ""),
            "prose_chars": trace.synthesis.prose.chars,
        })
        r = rows[-1]
        print(f"[memory] {r['meeting']}: steps={r['steps']} pts={r['points']} "
              f"detail={r['detail_chars']} num={r['numerals']} prose={r['prose_chars']}",
              file=sys.stderr)

    steps = sum(r["steps"] for r in rows)
    summary = {
        "label": args.label, "protocol": args.protocol, "url": args.url,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": str(args.corpus), "n_meetings": len(rows),
        "nop_rate": round(sum(r["nop"] for r in rows) / steps, 3) if steps else None,
        "mean_points": round(st.mean(r["points"] for r in rows), 2),
        # The headline. Averaged over MEETINGS, not over points, so one long meeting
        # with many points cannot dominate a 5-meeting sample.
        "mean_detail_chars": round(st.mean(r["detail_chars"] for r in rows), 2),
        "mean_numerals": round(st.mean(r["numerals"] for r in rows), 2),
        "mean_prose_chars": round(st.mean(r["prose_chars"] for r in rows), 1),
        "rows": rows,
    }
    print(f"\n[memory] {args.label or args.protocol}: "
          f"points {summary['mean_points']} | detail {summary['mean_detail_chars']} ch "
          f"| numerals {summary['mean_numerals']} | prose {summary['mean_prose_chars']} ch",
          file=sys.stderr)

    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"[memory] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
