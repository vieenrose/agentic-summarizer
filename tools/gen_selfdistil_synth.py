"""Rewrite synthesis targets against the STUDENT'S OWN memories (self-distillation).

    python tools/gen_selfdistil_synth.py --traces data/selfdistil/traces.jsonl \
        --out data/selfdistil/synth.jsonl --url http://127.0.0.1:8083

**The defect this attacks, measured 2026-09-02.** Every synthesis row in the pool pairs a
TEACHER-authored memory with a summary. At inference the memory is STUDENT-authored. The
gap that opens is large and specific:

    >=15-point TEACHER memory (a training row):  model writes 449 chars
    >=15-point STUDENT memory (a real run):      model writes  88 chars

and past 12 points the student's summaries collapse outright (12 pts -> 544 chars,
13 -> 116) and begin fabricating -- inventing a surname at 14 points, and reporting a
committee member as 已故 (deceased) at 15 when memory said only "reappoint". Classic
exposure bias: the model is never trained on the inputs it actually sees.

**Not fixable by adding more of the same data.** The pool ALREADY teaches the right thing:
58.3% of its synthesis rows carry >=13 points, and their mean target is 641 chars against
355 for <=12. The signal is present and the model inverts it, so this rewrites the INPUTS
rather than adding examples.

The memories come from `cli/gen_traces.py` run with the shipped checkpoint; the targets
come from the 27B teacher, which writes the 600+ char summaries the pool needs. Rows whose
target is shorter than the memory warrants are dropped -- importing a short target here
would teach exactly the behaviour being fixed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.lang import MIN_CJK_RATIO_PROSE, check_zh_tw  # noqa: E402
from arcsum.prompts import TOOLCALL_PROMPT_VERSION, synth_system_prompt  # noqa: E402
from arcsum.supervision.teacher import to_traditional  # noqa: E402

#: Minimum target length per memory point. The pool's own high-occupancy rows average
#: ~40 chars/point (641 chars at >=13 points), so this floor is deliberately below what
#: good data looks like -- it rejects collapse, not brevity.
MIN_CHARS_PER_POINT = 25


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8083")
    p.add_argument("--min-points", type=int, default=3)
    args = p.parse_args(argv)

    lines = args.traces.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    synth = [
        r for r in rows
        if "MEMORY:" in r.get("prompt", "") and "CHUNK:" not in r.get("prompt", "")
    ]
    model = LlamaServer(base_url=args.url, max_tokens=1400, seed=0, temperature=0.3,
                        extra={"chat_template_kwargs": {"enable_thinking": False}})

    out, refused = [], 0
    for i, r in enumerate(synth):
        pts = re.findall(r"^- (.+)$", r["prompt"], re.M)
        if len(pts) < args.min_points:
            continue
        text = to_traditional(" ".join(model(synth_system_prompt(), r["prompt"]).split()))
        if len(text) < MIN_CHARS_PER_POINT * len(pts):
            refused += 1
            print(f"[sd] {i}: refused ({len(text)} chars for {len(pts)} points)", file=sys.stderr)
            continue
        if bad := check_zh_tw(text, min_cjk_ratio=MIN_CJK_RATIO_PROSE):
            refused += 1
            print(f"[sd] {i}: refused ({bad})", file=sys.stderr)
            continue
        out.append({"meeting": f"sd-{r.get('meeting', i)}", "step": 999,
                    "prompt_version": TOOLCALL_PROMPT_VERSION,
                    "system": synth_system_prompt(), "prompt": r["prompt"],
                    "completion": text, "is_nop": False})
        print(f"[sd] {i}: ok ({len(pts)} pts -> {len(text)} chars)", file=sys.stderr)

    with args.out.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[sd] wrote {len(out)}, refused {refused} -> {args.out}", file=sys.stderr)
    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())
