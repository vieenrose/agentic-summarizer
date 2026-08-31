"""Build zh-TW reference summaries for a held-out slice (SPEC §2.2 stages 2b + 3).

    # stage 2b — translate the gold item summaries (TranslateGemma must be serving)
    python tools/build_heldout_refs.py items --corpus data/heldout_en \\
        --out data/heldout_items_zh.json --urls 8200,8201

    # stage 3 — compose one whole-meeting summary each (Qwen teacher must be serving)
    python tools/build_heldout_refs.py compose --items data/heldout_items_zh.json \\
        --out data/heldout_composed.json --url http://127.0.0.1:8082

A parameterised replacement for `data/raw/translate_items.py` + `data/raw/compose_pilot.py`,
whose paths, ports and output names were all hardcoded to the pilot. Same prompts, same
`prose.finalize` enforcement, so the references it produces are directly comparable to
`data/pilot_composed.json`.

**`finalize` is not cosmetic here.** It is the single enforcement point for SPEC §3's
output contract (<1,000 tokens, no bullets, no headings, zh-TW) and is the same function
the agent's SYNTHESIZE call and the baseline's reduce step both pass through. A reference
that skipped it would hold the model to a shape the reference itself does not have.

The two stages are separate subcommands because they need DIFFERENT models loaded, and
28 GB of TranslateGemma plus the Qwen teacher do not fit on one card at once.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from translate_corpus import translate_long  # noqa: E402

from arcsum.prose import finalize  # noqa: E402
from arcsum.supervision.teacher import to_traditional  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402

ZIP = REPO / "data/raw/MeetingBank.zip"
META = "Metadata/MeetingBank.json"

_COMPOSE_SYS = """你是一個會議記錄助手。以下是同一場會議依序列出的議程項目，包含類型標籤與摘要內容。

請將這些項目整合成一段流暢連貫的繁體中文摘要，不超過 1000 個字：
- 不使用條列式、不使用小標題、不加時間戳記。
- 只寫一段連續的文字，讀起來像一篇完整的會議摘要，而不是重點清單。
- 全部使用繁體中文書寫。"""


def cmd_items(args: argparse.Namespace) -> int:
    meta = json.loads(zipfile.ZipFile(ZIP).read(META))
    meetings = sorted(f.stem for f in args.corpus.glob("*.txt"))
    urls = [u if u.startswith("http") else f"http://127.0.0.1:{u}" for u in args.urls.split(",")]

    jobs: list[tuple[str, str, str, str]] = []
    for i, mid in enumerate(meetings):
        for item_id, it in (meta.get(mid, {}).get("itemInfo") or {}).items():
            summary = (it.get("Summary") or "").strip()
            if summary:
                jobs.append((mid, item_id, summary, urls[i % len(urls)]))
    print(f"[items] {len(jobs)} gold items over {len(meetings)} meetings", file=sys.stderr)

    def work(job: tuple[str, str, str, str]) -> tuple[str, str, str | None]:
        mid, item_id, summary, url = job
        return mid, item_id, translate_long(url, summary, max_tokens=2000)

    out: dict[str, list[dict]] = {}
    kept_en = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for mid, item_id, zh in ex.map(work, jobs):
            info = meta[mid]["itemInfo"][item_id]
            if zh and zh.strip():
                text = to_traditional(" ".join(zh.split()))
            else:
                # Keep the ENGLISH rather than dropping the item: a missing item silently
                # shortens the reference, which would flatter every system scored against
                # it. Visible instead in `kept_english` and to the CJK check downstream.
                text = " ".join(summary.split())
                kept_en += 1
            out.setdefault(mid, []).append(
                {
                    "item_id": item_id,
                    "type": info.get("type") or "",
                    "startTime": info.get("startTime"),
                    "endTime": info.get("endTime"),
                    "summary_zh": text,
                }
            )
    for mid in out:
        out[mid].sort(key=lambda r: (r["startTime"] is None, r["startTime"]))
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[items] wrote {sum(len(v) for v in out.values())} items "
          f"({kept_en} kept English) -> {args.out}", file=sys.stderr)
    return 0


def compose_one(url: str, items: list[dict], max_tokens: int) -> str:
    lines = [f"{i + 1}. [{it['type']}] {it['summary_zh']}" for i, it in enumerate(items)]
    body = {
        "messages": [
            {"role": "system", "content": _COMPOSE_SYS},
            {"role": "user", "content": "議程項目：\n" + "\n".join(lines)},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        # Qwen3.8 emits reasoning by default and can return EMPTY assistant content for
        # a long agenda, which is how 11 of 40 references came back blank on the first
        # pass. Same setting the teacher path already pins in `tools/gen_supervision.py`.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"].get("content") or ""


def cmd_compose(args: argparse.Namespace) -> int:
    items_by_meeting = json.loads(args.items.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    if args.out.exists():  # resumable: composing is the expensive stage
        out = json.loads(args.out.read_text(encoding="utf-8"))

    todo = [m for m in sorted(items_by_meeting) if m not in out]
    print(f"[compose] {len(todo)} meetings to compose ({len(out)} already done)", file=sys.stderr)
    empty: list[str] = []
    for n, mid in enumerate(todo, 1):
        raw = compose_one(args.url, items_by_meeting[mid], args.max_tokens)
        if not raw.strip():  # one retry, then record — never store a blank reference
            raw = compose_one(args.url, items_by_meeting[mid], args.max_tokens)
        prose = finalize(raw, token_len=heuristic_token_len)
        if not prose.text.strip():
            # An EMPTY reference is the most dangerous artifact this tool can emit: it
            # scores 0.0 against every candidate, so both arms tie at zero and the pair
            # silently becomes a no-op that dilutes the mean delta while still counting
            # toward n. Measured: 11 of 40 on the first pass, and the old flag check
            # (over_budget / lang_flags) did not catch a single one.
            empty.append(mid)
            print(f"[compose] EMPTY for {mid} after retry — not stored", file=sys.stderr)
            continue
        out[mid] = {
            "raw": prose.text,
            "text": prose.text,
            "tokens": prose.tokens,
            "over_budget": prose.over_budget,
            "lang_flags": list(prose.lang_flags),
            "had_markup": prose.had_markup,
            "n_items": len(items_by_meeting[mid]),
        }
        args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"[compose] ({n}/{len(todo)}) {mid}: {prose.tokens} tok "
              f"over_budget={prose.over_budget} flags={prose.lang_flags}", file=sys.stderr)
    bad = [m for m, v in out.items() if v["over_budget"] or v["lang_flags"]]
    blank = [m for m, v in out.items() if not v["text"].strip()]
    print(f"[compose] wrote {len(out)} references -> {args.out}; {len(bad)} flagged, "
          f"{len(blank)} blank-in-file, {len(empty)} refused this pass", file=sys.stderr)
    if blank or empty:
        print("[compose] REFUSED: blank references must be regenerated before scoring",
              file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("items", help="translate gold item summaries to zh-TW")
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--urls", default="8200,8201")
    a.add_argument("--workers", type=int, default=16)
    a.set_defaults(fn=cmd_items)

    b = sub.add_parser("compose", help="compose one whole-meeting reference per meeting")
    b.add_argument("--items", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--url", default="http://127.0.0.1:8082")
    b.add_argument("--max-tokens", type=int, default=8000)
    b.set_defaults(fn=cmd_compose)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
