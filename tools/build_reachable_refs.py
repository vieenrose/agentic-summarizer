"""Compose G3 references FROM THE TRANSCRIPT, so the gate measures something reachable.

    python tools/build_reachable_refs.py --corpus data/heldout_zh \\
        --url http://127.0.0.1:8087 --out data/heldout_refs_reachable.json

**The defect this repairs.** Measured 2026-09-03 on the 40 held-out meetings: of 454
specific claims in `data/heldout_references.json`, **211 (46.5%) do not appear in the
transcript being summarised.** Restricted to Arabic numbers and Latin identifiers, where
numeral-system reformatting cannot be the excuse, still **160 (35%)** — values like `94009`,
`166513`, `2200000`, `2572928`. They are ordinance numbers and dollar amounts from
MeetingBank's MINUTES DOCUMENTS, which SPEC §2.2 stage 3 composed the references from. The
agent never sees them.

**Two consequences, and the second is the serious one.**

1. G3 has a ceiling no faithful agent can reach: a model that says only what it heard must
   diverge from a third to a half of the reference's concrete content.
2. **The correlation runs backwards.** `qwen-tools-v5` fabricates 33.3% of the specifics it
   asserts and passes 3/3 G3 gates; `spec-e3` fabricates 15.6% and passes 1/3. Surface
   overlap with a reference full of invented-from-elsewhere detail is best achieved by a
   model that invents plausible detail in the same style. **A gate that rewards
   fabrication is worse than no gate**, because it certifies the failure.

That is how `v5` was certified while churning on a quarter of its steps on real meetings.

**What this tool does NOT claim to fix.** G3 is a PAIRED comparison and both arms are
scored against the same references, so unreachable content penalises both — it does not
bias the agent-vs-baseline direction on its own. What it does do is reward VERBOSITY (the
baseline writes ~5,000 characters against the agent's ~250, so it collides with more
reference n-grams by construction) and punish terseness. Rebuilding the references does not
remove that asymmetry; it removes the unreachability.

**Structural-similarity caveat, stated rather than hidden.** A reference composed by
summarising a transcript necessarily resembles the map-reduce baseline's own algorithm more
than it resembles the agent's chunk-and-curate loop. This tool feeds the WHOLE transcript in
one pass, precisely so the reference is not a map-reduce artifact — but that caps coverage
at the meetings which fit the teacher's context (25 of 40 at a 36k budget). **Meetings that
do not fit are SKIPPED, not composed hierarchically**, because a hierarchically composed
reference would be exactly the baseline's output shape and would tilt the comparison. The
skipped set is longer meetings, which is a real selection bias and is recorded in the
output.

Every reference is verified: the ungrounded rate against its own transcript is measured and
written alongside, so the replacement is never assumed to be better than what it replaces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.evalkit import grounding  # noqa: E402
from arcsum.prose import finalize  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402

SYSTEM = (
    "你是一個會議記錄助手。以下是一整場會議的逐字稿。\n"
    "請寫出一段流暢連貫的繁體中文摘要，不超過 1000 個字：\n"
    "- 不使用條列式、不加標題、不加時間戳記，只寫一段連續的文字。\n"
    "- 只能寫逐字稿中確實出現的內容，包括其中的數字、金額與日期；"
    "絕對不可加入逐字稿沒有提到的資訊。\n"
    "- 涵蓋會議的主要決議、討論重點與後續事項。\n"
    "- 全部使用繁體中文書寫。"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8087", help="the TEACHER server")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-input-tokens", type=int, default=36000,
                   help="skip meetings above this; they would need hierarchical "
                        "composition, which is the baseline's own shape")
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    teacher = LlamaServer(base_url=args.url, max_tokens=args.max_tokens,
                          repeat_penalty=1.1, seed=0, raw_completion=True,
                          extra={"cache_prompt": False,
                                 "chat_template_kwargs": {"enable_thinking": False}})

    refs: dict[str, str] = {}
    rows, skipped = [], []
    for f in sorted(args.corpus.glob("*.txt")):
        src = f.read_text(encoding="utf-8")
        n_in = heuristic_token_len(src)
        if n_in > args.max_input_tokens:
            skipped.append({"meeting": f.stem, "tokens": n_in})
            print(f"[refs] SKIP {f.stem}: {n_in} tokens > {args.max_input_tokens}",
                  file=sys.stderr)
            continue
        try:
            out = finalize(teacher(SYSTEM, src), token_len=heuristic_token_len)
        except Exception as exc:  # one meeting must not lose the pass
            skipped.append({"meeting": f.stem, "error": str(exc)})
            print(f"[refs] ERROR {f.stem}: {exc}", file=sys.stderr)
            continue
        rep = grounding.check(f.stem, out.text, src)
        refs[f.stem] = out.text
        rows.append({"meeting": f.stem, "input_tokens": n_in, "chars": out.chars,
                     "specifics": rep.n_checked, "ungrounded": rep.n_ungrounded,
                     "lang_flags": list(out.lang_flags)})
        print(f"[refs] {f.stem}: {out.chars} chars, {rep.n_checked} specifics, "
              f"{rep.n_ungrounded} ungrounded", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")

    tot = sum(r["specifics"] for r in rows)
    ung = sum(r["ungrounded"] for r in rows)
    report = {
        "corpus": str(args.corpus), "out": str(args.out),
        "composed": len(rows), "skipped": len(skipped),
        "specifics": tot, "ungrounded": ung,
        "ungrounded_rate": round(ung / tot, 4) if tot else None,
        "rows": rows, "skipped_detail": skipped,
    }
    print(f"\n[refs] composed {len(rows)}, skipped {len(skipped)} | "
          f"ungrounded {ung}/{tot} "
          f"({(ung / tot if tot else 0):.1%}) — compare 46.5% for the originals",
          file=sys.stderr)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
