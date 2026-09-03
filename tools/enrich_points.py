"""Put the figures back into gold `ADD` targets, where the chunk actually contains them.

    python tools/enrich_points.py --pool data/staging/sft_pool_mixed.jsonl \\
        --url http://127.0.0.1:8087 --out data/staging/points_enriched.jsonl

**The deficit, measured 2026-09-03.** A meeting summariser that records *that* a budget was
discussed but not *what the figure was* has done a fraction of the job, and that is what
this system does:

    chunks containing at least one specific (number / identifier)   99%
    gold ADD targets carrying one                                   42%
    `qwen-tools-v5` memory points carrying one                      33%
    `s234-e3` ep2 memory points carrying one                        16%

**The student tracks its supervision** (33% against a 42% ceiling), so this is a data
problem, not a capability problem — no amount of retraining on the same pool moves it.

**And the token cap is NOT the cause**, which is what makes the repair cheap. ADD targets
that carry a specific and those that do not have the SAME length distribution — mean 17.6
vs 17.7 tokens against a `POINT_TOKENS` cap of 25, medians both 18, only 7-8% within a
token of the cap. There is roughly 7 tokens of headroom. The teacher is not dropping
figures to fit; it just writes the topic rather than the number.

**Every rewrite is verified, and fabrication is the thing being guarded against.** A
rewritten point is accepted only when every specific it adds is present in ITS OWN CHUNK,
it stays within the cap, and it remains zh-TW. Anything else keeps the original. The
failure mode to fear here is a pool that teaches the model to attach plausible-looking
figures to points, which would be strictly worse than the vagueness it replaces — so the
grounding check runs against the chunk, not against the model's own output.

Only targets that LACK a specific while their chunk HAS one are candidates; the other 42%
are already doing their job and are left untouched.
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
from arcsum.evalkit import grounding  # noqa: E402
from arcsum.lang import MIN_CJK_RATIO_POINT, check_zh_tw  # noqa: E402
from arcsum.memory import POINT_TOKENS  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402

TOOL_CALL = re.compile(r'\{"name".*\}', re.S)

SYSTEM = (
    "你是一個會議記錄助手。以下是一段會議逐字稿（CHUNK）與一條已寫好的重點。\n"
    "請改寫這條重點，把逐字稿中「與這條重點直接相關」的具體數字、金額、日期或編號加進去。\n"
    "規則：\n"
    f"- 只能使用逐字稿中確實出現的數字或編號，絕對不可自行推測或編造。\n"
    f"- 改寫後總長度不得超過 {POINT_TOKENS} 個字。\n"
    "- 保持原意，只補上具體數值，不要改變結論。\n"
    "- 若逐字稿中沒有與這條重點相關的具體數值，原封不動輸出原句。\n"
    "- 只輸出改寫後的那一句，不要加任何說明。\n"
    "- 全部使用繁體中文書寫。"
)


def parse_args_json(completion: str) -> dict | None:
    m = TOOL_CALL.search(completion)
    if not m:
        return None
    try:
        return json.loads(m.group(0)).get("arguments") or {}
    except (json.JSONDecodeError, AttributeError):
        return None


def render_call(a: dict) -> str:
    return ('<tool_call>{"name": "update_memory", "arguments": '
            + json.dumps(a, ensure_ascii=False) + "}</tool_call>")


def accept(new: str, old: str, chunk: str) -> tuple[bool, str]:
    """Is this rewrite safe to train on? Returns `(ok, reason)`."""
    new = " ".join(new.split())
    if not new or new == old:
        return False, "unchanged"
    if heuristic_token_len(new) > POINT_TOKENS:
        return False, "over cap"
    if check_zh_tw(new, min_cjk_ratio=MIN_CJK_RATIO_POINT):
        return False, "language"
    added = set(grounding.claims_in(new)) - set(grounding.claims_in(old))
    if not added:
        return False, "no specific added"
    hay = grounding.normalise_for_match(chunk)
    ungrounded = [c for c in added if c not in hay]
    if ungrounded:
        # The exact failure this tool must not create: a plausible figure with no support.
        return False, f"fabricated {ungrounded}"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8087", help="the TEACHER server")
    p.add_argument("--limit", type=int, default=0, help="0 = all candidates")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    raw = args.pool.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(ln) for ln in raw if ln.strip()]
    teacher = LlamaServer(base_url=args.url, max_tokens=96, seed=0, raw_completion=True,
                          extra={"cache_prompt": False,
                                 "chat_template_kwargs": {"enable_thinking": False}})

    stats = {"candidates": 0, "rewritten": 0, "rejected": 0}
    reasons: dict[str, int] = {}
    done = 0
    out_rows = []
    for r in rows:
        if "CHUNK:" not in r["prompt"]:
            out_rows.append(r)
            continue
        a = parse_args_json(r["completion"])
        adds = (a or {}).get("add") or []
        if not adds:
            out_rows.append(r)
            continue
        chunk = r["prompt"].split("CHUNK:", 1)[-1]
        chunk_specifics = set(grounding.claims_in(chunk))
        new_adds, changed = [], False
        for old in adds:
            has = bool(grounding.claims_in(old))
            if has or not chunk_specifics or (args.limit and done >= args.limit):
                new_adds.append(old)
                continue
            stats["candidates"] += 1
            done += 1
            try:
                cand = teacher(SYSTEM, f"CHUNK:\n{chunk}\n\n重點：{old}")
            except Exception as exc:  # one bad row must not lose the pass
                reasons["error"] = reasons.get("error", 0) + 1
                new_adds.append(old)
                print(f"[enrich] ERROR {exc}", file=sys.stderr)
                continue
            ok, why = accept(cand, old, chunk)
            reasons[why] = reasons.get(why, 0) + 1
            if ok:
                stats["rewritten"] += 1
                new_adds.append(" ".join(cand.split()))
                changed = True
            else:
                stats["rejected"] += 1
                new_adds.append(old)
        if changed:
            a["add"] = new_adds
            r = {**r, "completion": render_call(a), "enriched": True}
        out_rows.append(r)
        if stats["candidates"] and stats["candidates"] % 100 == 0 and changed:
            print(f"[enrich] {stats['candidates']} candidates, "
                  f"{stats['rewritten']} rewritten", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {**stats, "reject_reasons": reasons, "pool": str(args.pool),
              "out": str(args.out)}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
