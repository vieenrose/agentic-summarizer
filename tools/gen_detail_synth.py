"""Generate SYNTHESIS rows that PRESERVE identifying details carried in memory.

    python tools/gen_detail_synth.py --pool data/staging/sft_pool_tools_v5.jsonl \\
        --out data/detail_synth --url http://127.0.0.1:8083

**The defect this closes, measured 2026-08-31.** Across the pool's 187 synthesis rows there
are 808 distinctive details in the MEMORY points (digits, 《named acts》, ordinal spans like
第二期). Only **379 — 46.9% — appear in the prose target.** The pool teaches, by majority,
that roughly half the identifying detail in memory is not worth carrying into the summary.

The model has learned exactly that. `tools/loss_map.py` on the corrected probe instrument:

    v6 reversal arm:  memory 70.4% -> prose 44.4%   (-26 points)
    v6 control arm:   memory 86.7% -> prose 60.0%   (-27 points)

Identical in both arms, and present with NO reversal anywhere in the transcript — so this is
not a revision problem and not a corpus gap. It is `gen_hedge_synth.py`'s shape exactly:
memory correct, prose lossy, majority signal in the pool pointing the wrong way. That fix
cost 12 rows and recovered two G3 gates on the way; see `runs/qwen-v2-heldout/RESULT.md`.

**Why this does not trade G1 against G3.** Checked before building: the held-out references
average 9.6 distinct details per summary against the agent's 5.2, so preserving detail moves
the output TOWARD the reference. Had it gone the other way this tool would be buying G1 at
G3's expense and should not have been written.

**Real memory states, not invented ones.** Every row reuses a MEMORY block the pool already
contains, so the input distribution is unchanged and only the target's detail-retention
policy differs. A row is kept only if the regenerated prose preserves at least
`--min-keep` of the details -- a target that dropped them would re-teach the defect.
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

#: What counts as an identifying detail: something a paraphrase would destroy. Digits, an
#: ordinal-prefixed span, or a 《》-quoted proper name. Deliberately narrow -- a loose
#: pattern would count ordinary words and make the retention rate meaningless.
DETAIL = re.compile(r"[0-9]+|第[一二三四五六七八九十百千0-9]+[期區號條屆年月天處件]|《[^》]{2,}》")

_SYS = """你是一個會議記錄助手。以下是目前累積的會議記憶（ARC 與 POINTS）。

請根據這份記憶，寫出一段流暢連貫的繁體中文摘要，不超過 1000 個字：
- 不使用條列式、不使用小標題、不加時間戳記。
- 只寫一段連續的文字，讀起來像一篇完整的會議摘要，而不是重點清單。
- **記憶中出現的具體識別資訊必須原封不動地保留**，包括數字、金額、天數、年限、
  案號、規格編號、條號、期別（例如「第二期」）、以及《》內的法案或文件名稱。
  這些是讀者辨認「究竟是哪一個案子」的依據，絕對不可以省略，也不可以改寫成
  「若干」「部分」「相關」這類籠統說法。
- 全部使用繁體中文書寫。"""


def details_in(text: str) -> set[str]:
    return {m for m in DETAIL.findall(text) if len(m) >= 2}


def memory_points(prompt: str) -> list[str]:
    return re.findall(r"^- (.+)$", prompt, re.M)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8083")
    p.add_argument("--min-details", type=int, default=2,
                   help="skip memory states carrying fewer details than this")
    p.add_argument("--min-keep", type=float, default=0.8,
                   help="reject a target preserving less than this share of details")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines()]
    synth = [r for r in rows
             if "MEMORY:" in r.get("prompt", "") and "CHUNK:" not in r.get("prompt", "")]
    model = LlamaServer(base_url=args.url, max_tokens=1200, seed=0, temperature=0.3,
                        extra={"chat_template_kwargs": {"enable_thinking": False}})

    out_rows: list[dict] = []
    refused = skipped = 0
    before_tot = before_keep = 0
    for i, r in enumerate(synth):
        if args.limit and len(out_rows) >= args.limit:
            break
        # `set().union(*gen)` over an empty generator is already the empty set; an
        # `or [set()]` guard here would never fire, since a generator is always truthy.
        wanted: set[str] = set().union(*(details_in(pt) for pt in memory_points(r["prompt"])))
        if len(wanted) < args.min_details:
            skipped += 1
            continue
        # Baseline: what the EXISTING target keeps. Reported so the shift is measured, not
        # assumed -- reading the share the pool already has is what made the hedge fix
        # legible rather than a hopeful guess.
        before_tot += len(wanted)
        before_keep += len(wanted & details_in(r.get("completion", "")))

        text = to_traditional(" ".join(model(_SYS, r["prompt"]).split()))
        kept = wanted & details_in(text)
        if len(kept) / len(wanted) < args.min_keep or len(text) < 60:
            refused += 1
            print(f"[detail] {i}: refused ({len(kept)}/{len(wanted)} details)", file=sys.stderr)
            continue
        if bad := check_zh_tw(text, min_cjk_ratio=MIN_CJK_RATIO_PROSE):
            refused += 1
            print(f"[detail] {i}: refused ({bad})", file=sys.stderr)
            continue
        out_rows.append({
            "meeting": f"detail-{r.get('meeting', i)}",
            "step": 999,
            # The tool-call pool is tagged `tools-v1` throughout, INCLUDING its synthesis
            # rows (whose completions are prose, not tool calls) -- `gen_hedge_synth`'s 12
            # rows were retagged on merge for the same reason. `train_toolcalls.py` refuses
            # a pool with mixed prompt versions, which is what caught this.
            "prompt_version": TOOLCALL_PROMPT_VERSION,
            "system": synth_system_prompt(),  # the REAL shipped prompt, not `_SYS`
            "prompt": r["prompt"],
            "completion": text,
            "is_nop": False,
        })
        print(f"[detail] {i}: ok ({len(kept)}/{len(wanted)})", file=sys.stderr)

    out = args.out / "detail_synth.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    base = before_keep / max(before_tot, 1)
    print(f"[detail] wrote {len(out_rows)}, refused {refused}, skipped {skipped} -> {out}\n"
          f"[detail] existing pool targets kept {before_keep}/{before_tot} ({base:.1%}) "
          f"of these details", file=sys.stderr)
    return 0 if out_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
