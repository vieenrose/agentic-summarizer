"""Measure a model's ONE-PASS summary of a long transcript, at several input lengths.

    python tools/onepass_probe.py --url http://127.0.0.1:8088 \\
        --transcript data/asr_eval_v1/dram-supply.txt \\
        --lengths 5000,12000,20000,32000 --out runs/onepass-htiny.json

**Why this exists as a tool rather than a shell one-liner.** Every one-pass candidate this
project has screened — granite-4.0-h-350m, granite-4.0-h-1b, granite-3.1-1b-a400m,
granite-3.1-3b-a800m — was measured ad hoc, and the numbers went into prose with no artifact
behind them. That is the failure this session already paid for twice: a "10.0 -> 2.8" that
re-ran as 10.4 -> 7.8, and a "3/27 and 12/27" that re-ran as 3/27 and 11/27. A screening
result that cannot be reproduced cannot be compared against the next candidate.

**The three numbers that decide a one-pass candidate**, and why length must be swept:

* `chars` — a model can be perfectly faithful by saying almost nothing. granite-3.1-3b-a800m
  measured 0% ungrounded at every length while emitting 2-7 specifics regardless of input
  size, which is not a summariser, it is an abstainer.
* `specifics` — how much concrete content it actually commits to.
* `ungrounded` — how much of that it invented.

Sweeping length is what separates "cannot summarise" from "cannot summarise THIS MUCH". A
model whose specifics count stays flat as input grows 6x is not degrading gracefully; it is
ignoring most of its context, and the flat line is the finding.

**Sampling follows CLAUDE.md trap 2.** This is a PROSE call, so `repeat_penalty=1.1` is set:
greedy decoding degenerates into repetition on prose, once emitting the same sentence eight
times. Screening a candidate without it measures the sampler, not the model — a mistake this
project made when first judging the granite MoE.

**Precision is the caller's choice and is recorded.** Two candidates were misjudged from
Q4_K_M alone before the user pointed out that a quantisation artifact was being read as a
model limitation. The correct protocol is f16/bf16 first, then quantise down; `--label`
carries the precision into the artifact so a later reader cannot lose it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.evalkit import grounding  # noqa: E402
from arcsum.prose import finalize  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402

#: Follows IBM's published summarization prompting guidance for Granite
#: (https://www.ibm.com/granite/docs/use-cases/prompt-engineering#summarization): state the
#: role, the source, the required form, and the grounding constraint explicitly.
SYSTEM = (
    "你是一個會議記錄助手。以下是一整場會議的逐字稿。\n"
    "請寫出一段流暢連貫的繁體中文摘要，不超過 1000 個字：\n"
    "- 不使用條列式、不加標題、不加時間戳記，只寫一段連續的文字。\n"
    "- 只能寫逐字稿中確實出現的內容，包括其中的數字、金額與日期；"
    "絕對不可加入逐字稿沒有提到的資訊。\n"
    "- 涵蓋會議的主要決議、討論重點與後續事項。\n"
    "- 全部使用繁體中文書寫。"
)


def truncate_to_tokens(text: str, budget: int) -> str:
    """Cut at a LINE boundary at or under `budget` heuristic tokens.

    Line-atomic like `chunker`, because slicing mid-utterance would hand the model a
    fragment no real caller would send and make the shortest lengths look worse than they are.
    """
    out: list[str] = []
    used = 0
    for line in text.splitlines():
        n = heuristic_token_len(line)
        if used + n > budget:
            break
        out.append(line)
        used += n
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--transcript", type=Path, required=True)
    p.add_argument("--lengths", default="5000,12000,20000,32000",
                   help="comma-separated heuristic-token budgets")
    p.add_argument("--label", required=True,
                   help="model AND precision, e.g. 'granite-4.0-h-tiny bf16'")
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    src = args.transcript.read_text(encoding="utf-8")
    full = heuristic_token_len(src)
    budgets = [int(x) for x in args.lengths.split(",") if x.strip()]

    # Identity is verified, never assumed: a server that failed to bind leaves the PREVIOUS
    # model answering on that port, and this session already recorded one measurement against
    # the wrong model that way.
    import urllib.request
    with urllib.request.urlopen(f"{args.url}/props", timeout=30) as r:
        props = json.load(r)
    model_path = props.get("model_path")
    print(f"[onepass] serving: {model_path}", file=sys.stderr)

    client = LlamaServer(base_url=args.url, max_tokens=args.max_tokens,
                         repeat_penalty=1.1, seed=0, raw_completion=True,
                         extra={"cache_prompt": False,
                                "chat_template_kwargs": {"enable_thinking": False}})

    rows = []
    for b in budgets:
        text = truncate_to_tokens(src, b)
        n_in = heuristic_token_len(text)
        t0 = time.time()
        try:
            out = finalize(client(SYSTEM, text), token_len=heuristic_token_len)
        except Exception as exc:
            rows.append({"budget": b, "input_tokens": n_in, "error": str(exc)})
            print(f"[onepass] {b}: ERROR {exc}", file=sys.stderr)
            continue
        dt = time.time() - t0
        g = grounding.check("", out.text, text)
        rows.append({
            "budget": b, "input_tokens": n_in, "chars": out.chars,
            "specifics": g.n_checked, "ungrounded": g.n_ungrounded,
            "ungrounded_rate": round(g.ungrounded_rate, 4),
            "lang_flags": list(out.lang_flags),
            "seconds": round(dt, 1),
            "text": out.text,
        })
        print(f"[onepass] {b:>6} tok in -> {out.chars:>4} chars, "
              f"{g.n_checked:>2} specifics, {g.n_ungrounded} ungrounded, {dt:.0f}s",
              file=sys.stderr)

    report = {
        "label": args.label, "model_path": model_path,
        "transcript": str(args.transcript), "transcript_tokens": full,
        "max_tokens": args.max_tokens, "repeat_penalty": 1.1, "seed": 0,
        "cache_prompt": False,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"[onepass] wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
