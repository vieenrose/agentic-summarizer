"""Generate synthetic DELIBERATION-WITHOUT-RESOLUTION supervision, to close the gap
`CLAUDE.md`'s "explicit STATED OUTCOME" finding identifies (2026-08-30).

    python tools/gen_deliberation.py --split train --out data/deliberation_train
    python tools/gen_deliberation.py --split probe --out data/deliberation_probe

**The gap, precisely.** Reading real zh-TW ASR (`data/ly_phase3_v2`), every trained
checkpoint requires a chunk to contain an explicit STATED OUTCOME before it will record
anything — open-ended debate, personal critique, and in-progress Q&A are NOP'd even when
substantive. Direct reads confirmed this is not length, not speaker count, not ASR noise:
`ivod-17666` is a legislator's clause-by-clause critique of a specific bill article,
substantively the SAME KIND of content as a curating example, but framed as ongoing
opinion rather than a landed resolution, and it is NOP'd anyway.

**Why this traces to the corpus, and why synthetic data is the fix.** MeetingBank's gold
items are always RESOLVED agenda-item outcomes ("City Council approved X") — never
mid-debate commentary. The model correctly generalised "record resolutions" from that
supervision; the fix is supervision that ALSO rewards recording a stated POSITION, request,
or concern that never resolves within the chunk, phrased as attribution rather than
outcome: `ADD - 委員質疑<subject>` / `ADD - 委員要求<subject>`, never `ADD - <subject>通過`.

Same disciplines as `tools/gen_reversals.py`, for the same reasons:

1. Gold is PLANTED from a structured plan, not read off the generated text.
2. `--split probe` is an INDEPENDENT validation set — different subjects, different
   speakers' stances, so a probe pass cannot be pattern match on the training scenarios.
3. Every sequence replays through the real harness (SPEC §4.2) before being kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.chunker import CHUNK_TOKENS, iter_chunks  # noqa: E402
from arcsum.guards import apply_ops  # noqa: E402
from arcsum.memory import Memory  # noqa: E402
from arcsum.ops import parse_ops, render_op  # noqa: E402
from arcsum.prompts import build_step_prompt, step_system_prompt  # noqa: E402
from arcsum.supervision.teacher import to_traditional  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402
from arcsum.transcript import parse_transcript  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    slug: str
    subject: str          #: the bill/issue under discussion
    stance_verb: str       #: how the speaker's position should be attributed: 質疑/主張/要求/批評
    position: str          #: the substance of the position, e.g. "條文未界定不利益範圍"
    speaker_role: str      #: "委員" / "官員" / "陳情人" — who is speaking
    fillers: tuple[str, ...]


TRAIN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("roadsafety", "道路交通安全條例修正案", "質疑", "條文對電動自行車速限規定不明確",
             "委員", ("號誌汰換期程", "行人穿越道設計", "違規記點制度")),
    Scenario("longtermcare", "長期照顧服務法修正案", "要求", "居家照顧員薪資保障應明訂於條文",
             "委員", ("照顧機構評鑑", "家屬喘息服務", "外籍看護聘僱")),
    Scenario("digitalprivacy", "個人資料保護法修正案", "批評", "主管機關對外洩事件裁罰標準過於寬鬆",
             "委員", ("資安稽核頻率", "跨境傳輸規範", "當事人請求權")),
    Scenario("farmsubsidy", "農業天然災害救助辦法", "主張", "現行救助金額未反映近年物價上漲",
             "委員", ("災損認定程序", "保險理賠銜接", "現勘人力不足")),
    Scenario("airquality", "空氣污染防制法修正案", "質疑", "工業區加嚴標準的緩衝期過長",
             "官員", ("監測站增設", "裁罰金額分級", "污染源即時揭露")),
    Scenario("laborsafety", "職業安全衛生法修正案", "要求", "承攬商違規應併罰原事業單位",
             "委員", ("危險作業通報", "職災補償標準", "教育訓練時數")),
    Scenario("watersupply", "自來水法修正案", "批評", "偏鄉供水改善進度落後既定期程",
             "陳情人", ("水質檢驗結果", "管線汰換經費", "備援水源規劃")),
    Scenario("cyberfraud", "詐欺犯罪危害防制條例", "主張", "金融機構應對可疑帳戶即時圈存",
             "委員", ("警示帳戶通報", "跨機關資料共享", "被害人求償機制")),
)

PROBE_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("pensionreform", "軍公教退休撫卹條例修正案", "質疑", "年資採計方式未考慮中斷服務情形",
             "委員", ("退撫基金收支", "延退誘因設計", "遺屬年金比例")),
    Scenario("forestprotect", "森林法修正案", "要求", "國有林地濫墾應加重刑責並溯及既往查處",
             "委員", ("巡山員人力配置", "空拍監測系統", "原民傳統領域劃設")),
    Scenario("telecomfraud", "電信網路詐騙防制條例", "批評", "電信業者對境外來電未落實強制標示",
             "陳情人", ("簡訊攔截技術", "門號實名查核", "被害申訴管道")),
    Scenario("hospitalstaff", "醫療法修正案", "主張", "急診壅塞應納入醫院評鑑扣分項目",
             "官員", ("護病比規範", "轉診獎勵措施", "夜間值班津貼")),
)

_ALL = {s.slug for s in TRAIN_SCENARIOS} | {s.slug for s in PROBE_SCENARIOS}
assert len(_ALL) == len(TRAIN_SCENARIOS) + len(PROBE_SCENARIOS), "scenario slugs overlap"
assert not ({s.subject for s in TRAIN_SCENARIOS} & {s.subject for s in PROBE_SCENARIOS})
assert not ({s.position for s in TRAIN_SCENARIOS} & {s.position for s in PROBE_SCENARIOS})

_WRITE_SYS = """你是一個立法院委員會會議逐字稿產生器。請寫出一段真實自然的繁體中文委員會質詢逐字稿，
語氣口語、可以有停頓詞、重複與插話，不要寫得像正式公文。

格式規則（非常重要）：
- 每一行的格式必須是「S<數字>: <發言內容>」，例如「S1: 主席，我想請教一下。」
- 不要加時間戳記、不要加標題、不要用條列式、不要加任何說明文字。
- 至少要有兩位不同的發言人。
- 全部使用繁體中文。"""


def _part_prompt(sc: Scenario, part: str) -> str:
    if part == "deliberation":
        return (
            f"請寫出一段立法院委員會質詢逐字稿，主題是「{sc.subject}」。"
            f"一位{sc.speaker_role}對這個法案提出意見：{sc.position}。"
            f"這位{sc.speaker_role}只是提出看法、質疑或要求，過程中沒有任何人裁示、沒有表決、"
            f"也沒有達成共識或結論——討論到一半就結束，就像真實委員會質詢常常沒有當場定案一樣。"
            f"請務必在逐字稿中明確出現「{sc.subject}」與「{sc.position}」這兩個內容。"
            "大約 20 到 30 行，可以有其他委員或官員的插話，但不要下結論。"
        )
    items = "、".join(sc.fillers)
    return (
        f"請寫出一段立法院委員會逐字稿，內容依序討論以下與前面無關的議題：{items}。"
        "同樣不要下結論或表決，只是討論過程。大約 20 到 30 行。"
    )


def _clean(raw: str) -> list[str]:
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        speaker, _, text = line.partition(":")
        speaker = speaker.strip()
        if not (speaker.startswith("S") and speaker[1:].isdigit()):
            continue
        text = " ".join(text.split())
        if text:
            out.append(f"{speaker}: {to_traditional(text)}")
    return out


def generate_meeting(model: LlamaServer, sc: Scenario) -> str | None:
    lines: list[str] = []
    delib_raw = ""
    for part in ("deliberation", "filler"):
        raw = model(_WRITE_SYS, _part_prompt(sc, part))
        if part == "deliberation":
            delib_raw = raw
        got = _clean(raw)
        if len(got) < 6:
            return None
        lines.extend(got)
    # Check the planted needles against the RAW model output, BEFORE `to_traditional`
    # (s2twp) has a chance to rewrite mainland lexical choices to Taiwan ones -- e.g.
    # 自行車 -> 腳踏車. That conversion is correct for the corpus but runs on the exact
    # planted phrase too, so checking the POST-conversion text made every generation
    # look like a miss (measured: 0/48 written on the first pass). The gold builder
    # still keys off the post-conversion `body`, since that is what the student sees.
    if sc.subject not in delib_raw or sc.position not in delib_raw:
        return None
    return "\n".join(lines) + "\n"


def build_gold(sc: Scenario, body: str) -> list[dict] | None:
    utterances = parse_transcript(body)
    chunks = list(iter_chunks(utterances, budget=CHUNK_TOKENS, token_len=heuristic_token_len))
    if not chunks:
        return None
    # `body` already went through `to_traditional` (see `generate_meeting`), so the
    # needles must too, or a lexical rewrite (自行車 -> 腳踏車) makes every chunk miss.
    subject_conv, position_conv = to_traditional(sc.subject), to_traditional(sc.position)
    deliberation_i = next(
        (c.index for c in chunks
         if subject_conv in c.render() and position_conv in c.render()),
        None,
    )
    if deliberation_i is None:
        return None

    # The point is attributed to the SPEAKER'S STANCE, never phrased as a resolution —
    # that distinction is the entire lesson: "委員質疑X" not "X應修正" or "X已通過".
    # No bill name: POINT_TOKENS=25 (arcsum.memory) and subject+position alone routinely
    # exceeds it (measured: 31 tokens, refused as "point too long" on every scenario).
    # The stance and its substance carry the lesson; which bill is incidental.
    point = f"{sc.speaker_role}{sc.stance_verb}{sc.position}"

    rows: list[dict] = []
    memory = Memory(token_len=heuristic_token_len)
    for c in chunks:
        if c.index != deliberation_i:
            continue  # filler chunks carry no planted gold; excluded rather than guessed
        completion = f"ADD - {point}"
        ops = parse_ops(completion)
        outcome = apply_ops(memory, ops, c)
        applied = [a.op for a in outcome.results if a.applied]
        if len(applied) != len(ops):
            return None
        rows.append({
            "meeting": f"delib-{sc.slug}",
            "step": c.index,
            "prompt_version": "sys-v2",
            "system": step_system_prompt(),
            "prompt": build_step_prompt(Memory(token_len=heuristic_token_len), c,
                                        total=len(chunks)),
            "completion": "\n".join(render_op(o) for o in applied),
            "is_nop": False,
        })
    return rows or None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "probe"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8082")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--rebuild-gold", action="store_true")
    args = p.parse_args(argv)

    scenarios = TRAIN_SCENARIOS if args.split == "train" else PROBE_SCENARIOS
    repeats = args.repeats if args.split == "train" else 1
    args.out.mkdir(parents=True, exist_ok=True)

    if args.rebuild_gold:
        rebuilt = refused = 0
        for sc in scenarios:
            for f in sorted(args.out.glob(f"{sc.slug}-*.txt")):
                rows = build_gold(sc, f.read_text(encoding="utf-8"))
                out = f.with_suffix(".jsonl")
                if rows is None:
                    refused += 1
                    out.unlink(missing_ok=True)
                    continue
                with out.open("w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                rebuilt += 1
        print(f"[delib] rebuilt gold for {rebuilt}, refused {refused}", file=sys.stderr)
        return 0

    written = failed = 0
    for sc in scenarios:
        for k in range(repeats):
            model = LlamaServer(base_url=args.url, max_tokens=2500, seed=k,
                                temperature=0.8 if repeats > 1 else 0.3,
                                extra={"chat_template_kwargs": {"enable_thinking": False}})
            body = generate_meeting(model, sc)
            if body is None:
                failed += 1
                print(f"[delib] {sc.slug}#{k}: generation refused", file=sys.stderr)
                continue
            name = f"{sc.slug}-{k}"
            (args.out / f"{name}.txt").write_text(body, encoding="utf-8")
            if args.split == "train":
                rows = build_gold(sc, body)
                if rows is None:
                    failed += 1
                    (args.out / f"{name}.txt").unlink()
                    print(f"[delib] {sc.slug}#{k}: gold refused", file=sys.stderr)
                    continue
                with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1
            print(f"[delib] {sc.slug}#{k}: ok", file=sys.stderr)

    print(f"[delib] wrote {written}, refused {failed} -> {args.out}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
