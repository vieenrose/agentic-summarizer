"""Build a human-review packet for SPEC §2.2 stage 4 / §5's human validation.

    python tools/build_review_packet.py --pairs runs/qwen-v2-heldout/g_agent_pairs.json \\
        --baseline runs/qwen-v2-heldout/g_baseline_pairs.json --n 12 \\
        --out runs/qwen-v2-heldout/review

**This is the one gate no tooling can close.** Every quality number in this project comes
from automated metrics and an LLM judge; SPEC requires a human to read real outputs and say
whether they are usable, and that has never been done. This script does the part a machine
can: sample fairly, strip the cues that would bias a reader, and lay the work out so it can
actually be completed in an hour.

Three things it does deliberately:

1. **Blind and order-randomised.** Each meeting shows two summaries as "A" and "B" with the
   assignment shuffled per meeting and recorded only in the answer key. A reviewer told
   which system produced which will confirm whichever they expect to win — and the agent's
   summaries are ~5x shorter than the baseline's, which is itself a strong unblinded cue.
2. **Stratified by meeting length, not random.** Length is the variable every measured
   difference tracks (the agent's margin is +0.202 on long meetings against +0.035 on
   short), so a uniform sample would under-represent exactly where the systems differ.
3. **Asks about usability, not preference.** "Which is better" invites taste. The questions
   are whether the summary states decisions correctly, whether anything in it is
   unsupported by the transcript, and whether the reviewer would send it to attendees —
   the last being the only question that matters for shipping.

The transcript is included in full for every sampled meeting. A faithfulness judgement made
without reading the source is not a faithfulness judgement.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

QUESTIONS = """\
1. 這份摘要所陳述的決議，是否與逐字稿一致？（是／部分／否）
2. 摘要中是否有逐字稿沒有支持的內容？（無／有，請指出）
3. 是否遺漏了逐字稿中重要的決議或承諾？（無／有，請指出）
4. 你會把這份摘要直接寄給與會者嗎？（會／需要修改／不會）
5. 其他意見：
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True, help="agent pairs json")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260830)
    args = p.parse_args(argv)

    agent = {r["meeting_id"]: r for r in json.loads(args.pairs.read_text(encoding="utf-8"))}
    base = {r["meeting_id"]: r for r in json.loads(args.baseline.read_text(encoding="utf-8"))}
    common = sorted(set(agent) & set(base))
    if len(common) < args.n:
        print(f"[review] REFUSED: only {len(common)} paired meetings", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    # Stratify by transcript length: short / medium / long, sampled evenly.
    by_len = sorted(common, key=lambda m: len(agent[m]["source"]))
    third = max(1, len(by_len) // 3)
    strata = [by_len[:third], by_len[third:2 * third], by_len[2 * third:]]
    picked: list[str] = []
    per = max(1, args.n // 3)
    for s in strata:
        picked += rng.sample(s, min(per, len(s)))
    for m in rng.sample(common, len(common)):
        if len(picked) >= args.n:
            break
        if m not in picked:
            picked.append(m)
    picked = picked[: args.n]

    args.out.mkdir(parents=True, exist_ok=True)
    key = []
    for i, mid in enumerate(picked, 1):
        first_is_agent = rng.random() < 0.5
        a_text = agent[mid]["candidate"] if first_is_agent else base[mid]["candidate"]
        b_text = base[mid]["candidate"] if first_is_agent else agent[mid]["candidate"]
        key.append({"item": i, "meeting_id": mid,
                    "A": "agent" if first_is_agent else "baseline",
                    "B": "baseline" if first_is_agent else "agent"})
        (args.out / f"{i:02d}_{mid}.md").write_text(
            f"# 審閱項目 {i} — {mid}\n\n"
            f"## 逐字稿\n\n```\n{agent[mid]['source']}\n```\n\n"
            f"## 摘要 A\n\n{a_text}\n\n### A 的問題\n\n{QUESTIONS}\n"
            f"## 摘要 B\n\n{b_text}\n\n### B 的問題\n\n{QUESTIONS}\n"
            "## 最後一題\n\nA 與 B 哪一份比較適合直接寄給與會者？為什麼？\n",
            encoding="utf-8",
        )

    (args.out / "ANSWER_KEY.json").write_text(
        json.dumps({"seed": args.seed, "note": "do not open before review is complete",
                    "assignments": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "README.md").write_text(
        f"# 人工審閱（SPEC §2.2 stage 4 / §5）\n\n"
        f"{len(picked)} 場會議，每場兩份摘要（A／B），系統身分已隱藏且順序隨機。\n"
        f"請逐份閱讀逐字稿後再作答；未讀逐字稿的忠實度判斷沒有意義。\n\n"
        f"完成後才開啟 `ANSWER_KEY.json`。\n\n"
        f"取樣：依逐字稿長度分層（短／中／長各約三分之一），seed={args.seed}。\n",
        encoding="utf-8")
    print(f"[review] {len(picked)} items -> {args.out} (blind, key withheld)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
