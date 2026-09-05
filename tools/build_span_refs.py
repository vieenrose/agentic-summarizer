"""Compose G3 references for meetings of ANY length, from span-local gold minutes.

    python tools/build_span_refs.py --corpus data/heldout_zh \\
        --items data/heldout_items_zh.json --url http://127.0.0.1:8091 \\
        --out data/heldout_refs_span.json --report runs/span-refs-report.json

**The problem this solves.** `build_reachable_refs.py` composes a reference by reading the
WHOLE transcript in one teacher pass, so it silently drops every meeting above the teacher's
context. Measured on the held-out set, the consequence is not a minor coverage gap — it is a
perfect confound:

| reference set | n | median chunks | max | meetings over `POINTS_CAP` |
|---|---|---|---|---|
| one-pass reachable refs | 25 | 7 | 14 | **0** |
| the meetings it excludes | 15 | 23 | 37 | **10** |

So every agent-vs-baseline number to date was computed in the one regime where the working set
never overflows, the journal never fills, and the agent runs the v1.0 code path. SPEC §5.2.3
forbids resting a ship decision on that.

**Why not just use the gold minutes directly — REFUTED, do not retry.** MeetingBank's
`itemInfo[].Summary` is human-authored and exists for every meeting regardless of length, which
makes it look like the obvious answer. Measured against the transcripts they summarise, the
gold item summaries are **55.6% ungrounded (499 of 898 specifics)** — worse than the references
they would replace (38.3%) and far worse than the one-pass route (2.2%). They are written from
the MINUTES DOCUMENTS, so they carry ordinance numbers, dollar figures and department codes
(`（財務部門2410）`) that are never spoken aloud.

**What this does instead.** `itemInfo` aligns each item to a `startTime`/`endTime`. The spans
are small — median 366 s, 95th percentile ~3,256 s (~8k zh tokens) — so each item fits a
teacher's context no matter how long the meeting is. For each item the teacher rewrites its
gold minute using ONLY the transcript span it is aligned to, deleting whatever is not said
there. Rewritten items are concatenated in time order.

**Why this is not the baseline's own shape**, which is the objection that forbids hierarchical
composition: map-reduce *chooses what is salient* in each window, and that editorial choice is
precisely what is under test. Here the selection is fixed in advance by the human-authored item
list and the model may only REMOVE unreachable detail. The per-window structure is shared; the
judgement is not.

**The line->time alignment is reconstructed, not stored.** Format v2 is timestamp-free by
design (SPEC §2), so the mapping is rebuilt from the Zenodo release: segments are merged by
CONSECUTIVE SPEAKER, carrying `(min start, max end)`. That merge is what the v2 import applied,
verified exactly — 1213 raw segments -> 572 merged == 572 zh lines on `AlamedaCC_04162019`, and
likewise on `BostonCC_03022022` (534 -> 172) and `LongBeachCC_04172018` (1731 -> 919). A
mismatch is a hard error rather than an approximation, because a misaligned span would feed the
teacher the wrong evidence and silently produce a reference about the wrong agenda item.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.corpus.meetingbank import extract_turns_with_offsets  # noqa: E402
from arcsum.evalkit import grounding  # noqa: E402
from arcsum.prose import finalize  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402

#: **Reference LENGTH decides G3, so it is chosen deliberately rather than inherited.**
#: Measured (SPEC §5.2.4): against 273-character references the terse agent wins F1 32/8;
#: against ~870-character ones the verbose baseline wins — the same two systems, opposite
#: verdicts, from reference length alone. The arms' natural lengths here are ~310 (agent)
#: and ~874 (baseline) characters, so a reference sitting BETWEEN them favours neither by
#: construction. That is the target this prompt aims at.
#:
#: It cannot be met by asking for more items: MeetingBank annotates a median of **4** items
#: per meeting (max 14), so reference length is set by how fully each item is rendered, not
#: by how many there are. Hence a per-item length floor rather than a whole-summary one.
SYSTEM = (
    "你是一個會議記錄編輯。以下提供一段會議逐字稿，以及一則根據官方紀錄寫成的議程摘要。\n"
    "請改寫這則摘要，使它只包含逐字稿中確實說出來的內容：\n"
    "- 刪除逐字稿沒有提到的編號、金額、日期、部門代碼與人名。\n"
    "- 不可加入逐字稿沒有的任何資訊，也不可自行推論或計算。\n"
    "- 若這則摘要的內容在逐字稿中完全找不到，只回答「（無）」。\n"
    "- 用流暢的繁體中文書寫，不要條列，不要加標題。\n"
    "- 請寫得完整具體：說明是誰提出、決定了什麼、以及逐字稿中提到的關鍵細節與理由，"
    "約 80 到 120 個字。不要只用一句話帶過。"
)

EMPTY = "（無）"


def line_spans(zip_path: Path, meeting: str) -> list[tuple[float, float]]:
    """`(start, end)` per line of the v2 transcript, rebuilt from the Zenodo release."""
    with zipfile.ZipFile(zip_path) as z:
        meta = json.load(io.TextIOWrapper(z.open("Metadata/MeetingBank.json"),
                                          encoding="utf-8"))
        rec = meta.get(meeting)
        if rec is None:
            raise KeyError(f"{meeting} absent from MeetingBank.json")
        key = str(rec["Transcripts"])
        names = {n.split("/")[-1]: n for n in z.namelist()
                 if n.endswith(".transcript.json")}
        path = names.get(key.split("/")[-1])
        if path is None:
            stem = key.split(".")[0].split("/")[-1]
            hits = [v for k, v in names.items() if stem in k]
            if not hits:
                raise KeyError(f"no transcript file for {meeting}")
            path = hits[0]
        doc = json.load(io.TextIOWrapper(z.open(path), encoding="utf-8"))

    merged: list[list] = []
    for spk, _text, start, end in extract_turns_with_offsets(doc["segments"]):
        if merged and merged[-1][0] == spk:
            merged[-1][2] = end
        else:
            merged.append([spk, start, end])
    return [(m[1], m[2]) for m in merged]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--zip", type=Path, default=Path("data/raw/MeetingBank.zip"))
    p.add_argument("--url", required=True, help="the TEACHER server")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-span-tokens", type=int, default=12000)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    items = json.loads(args.items.read_text(encoding="utf-8"))
    teacher = LlamaServer(base_url=args.url, max_tokens=args.max_tokens,
                          repeat_penalty=1.1, seed=0, raw_completion=True,
                          extra={"cache_prompt": False,
                                 "chat_template_kwargs": {"enable_thinking": False}})

    refs: dict[str, str] = {}
    rows, skipped = [], []
    meetings = sorted(items)
    if args.limit:
        meetings = meetings[: args.limit]

    for i, meeting in enumerate(meetings, 1):
        src_path = args.corpus / f"{meeting}.txt"
        if not src_path.exists():
            skipped.append({"meeting": meeting, "why": "no transcript"})
            continue
        lines = src_path.read_text(encoding="utf-8").splitlines()
        try:
            spans = line_spans(args.zip, meeting)
        except KeyError as exc:
            skipped.append({"meeting": meeting, "why": str(exc)})
            continue
        if len(spans) != len(lines):
            # Hard error, never an approximation: a misaligned span feeds the teacher the
            # wrong evidence and yields a confident reference about the wrong agenda item.
            skipped.append({"meeting": meeting,
                            "why": f"alignment mismatch {len(spans)} spans vs {len(lines)} lines"})
            continue

        pieces, kept, dropped = [], 0, 0
        for item in sorted(items[meeting], key=lambda x: x.get("startTime", 0)):
            s, e = item.get("startTime"), item.get("endTime")
            if s is None or e is None:
                continue
            window = [ln for ln, (a, b) in zip(lines, spans, strict=True) if b >= s and a <= e]
            if not window:
                dropped += 1
                continue
            text = "\n".join(window)
            while heuristic_token_len(text) > args.max_span_tokens and len(window) > 8:
                window = window[: int(len(window) * 0.8)]
                text = "\n".join(window)
            user = f"逐字稿：\n{text}\n\n議程摘要：\n{item['summary_zh']}"
            try:
                out = finalize(teacher(SYSTEM, user), token_len=heuristic_token_len)
            except Exception as exc:
                skipped.append({"meeting": meeting, "item": item.get("item_id"),
                                "why": str(exc)})
                dropped += 1
                continue
            body = out.text.strip()
            if not body or EMPTY in body or body.startswith("（無"):
                dropped += 1
                continue
            pieces.append(body)
            kept += 1

        if not pieces:
            skipped.append({"meeting": meeting, "why": "no item survived rewriting"})
            continue
        ref = "".join(pieces)
        rep = grounding.check(meeting, ref, src_path.read_text(encoding="utf-8"))
        refs[meeting] = ref
        rows.append({"meeting": meeting, "chars": len(ref), "items_kept": kept,
                     "items_dropped": dropped, "specifics": rep.n_checked,
                     "ungrounded": rep.n_ungrounded})
        print(f"[span-refs] {i}/{len(meetings)} {meeting}: {len(ref)} chars, "
              f"{kept} items kept / {dropped} dropped, "
              f"{rep.n_ungrounded}/{rep.n_checked} ungrounded", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")

    tot = sum(r["specifics"] for r in rows)
    ung = sum(r["ungrounded"] for r in rows)
    report = {
        "corpus": str(args.corpus), "items": str(args.items), "out": str(args.out),
        "composed": len(rows), "skipped": len(skipped),
        "specifics": tot, "ungrounded": ung,
        "ungrounded_rate": round(ung / tot, 4) if tot else None,
        "rows": rows, "skipped_detail": skipped,
    }
    print(f"\n[span-refs] composed {len(rows)}, skipped {len(skipped)} | "
          f"ungrounded {ung}/{tot} ({(ung / tot if tot else 0):.1%}) — compare "
          f"55.6% for the raw gold minutes and 2.2% for the one-pass route",
          file=sys.stderr)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
