"""Generate SYNTHESIS training rows that PRESERVE question-form (`是否`) points.

    python tools/gen_hedge_synth.py --out data/hedge_synth --url http://127.0.0.1:8082

**The bug this closes.** `qwen-tools-v4` deterministically inverted a faithful reading-step
point: memory held `委員質疑國有林地濫墾是否應加重刑責` ("questions WHETHER penalties should
be increased") and synthesis wrote `認為...不應加重刑責` ("believes they should NOT be") —
asserting the opposite polarity as fact. Measured 1.3% of ADD points, but it is the wrong
kind of rare: a meeting-minutes tool must not flip a stated position.

**Why the fix must be training-side.** Two prompt-only interventions were tested and BOTH
produced byte-identical output: an instruction to synthesis to preserve question form, and
an instruction to the reading step to never emit `是否`. A fine-tuned checkpoint at
`temperature=0` does not respond to novel system-prompt text; behaviour is fixed by
training. See `runs/qwen-v2-heldout/RESULT.md`.

**What the existing pool actually teaches.** Of 175 synthesis rows, 5 carry `是否` in the
MEMORY input — and only 2 preserve it in the prose target. The other 3 DROP the point
entirely. So the pool's majority signal is "a question-form point is not worth carrying",
and with no trained behaviour for preserving it, inference improvises — badly. This tool
adds rows where `是否` in memory maps to `是否` preserved in prose, flipping that majority.

Prose targets are teacher-generated rather than templated: a templated target would teach
the phrasing but also teach template-shaped prose, and SPEC §3's output contract is fluent
connected text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.prompts import PROMPT_VERSION, synth_system_prompt  # noqa: E402
from arcsum.supervision.teacher import to_traditional  # noqa: E402

#: Each case is (arc, [points]) where exactly one point carries a `是否` question. Subjects
#: are deliberately DISJOINT from `gen_deliberation.py`'s train AND probe scenarios, so
#: this does not become a second way to memorise the probe.
CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("委員就漁港設施改善案提出質詢，並要求主管機關補充說明。",
     ("委員質疑漁港浚深工程是否應納入本年度預算", "漁業署說明現行工程排程與經費來源",
      "委員要求一個月內提出改善期程")),
    ("會議討論校園午餐食材採購規範，尚未作成決議。",
     ("委員詢問有機食材比例是否應提高至五成", "教育局說明現行供應商契約條件",
      "委員要求提供各縣市比較資料")),
    ("委員針對長照機構評鑑制度提出意見。",
     ("委員質疑評鑑結果是否應全面公開", "衛福部說明現行公開範圍與個資考量")),
    ("會議審查捷運延伸線可行性報告，各方意見分歧。",
     ("委員詢問延伸路線是否應優先考量人口稠密區", "交通部說明運量預估方法",
      "地方代表要求納入地方意見")),
    ("委員就公務人員退休年金調整方向提出質詢。",
     ("委員質疑所得替代率是否應再行檢討", "銓敘部說明現行精算結果")),
    ("會議討論再生能源躉購費率，尚在研議階段。",
     ("委員詢問費率是否應隨市場行情逐年調整", "經濟部說明現行費率審定機制",
      "業者代表反映投資回收期考量")),
    ("委員針對水庫清淤經費編列提出意見。",
     ("委員質疑清淤預算是否足以因應淤積速度", "水利署說明近年清淤量與成效")),
    ("會議審議都市更新獎勵容積規定。",
     ("委員詢問獎勵上限是否應設定分級制", "內政部說明現行審議程序",
      "委員要求補充實施成效評估")),
    ("委員就偏鄉醫療巡迴服務提出質詢。",
     ("委員質疑巡迴頻率是否應提高至每週一次", "衛福部說明現行人力配置限制")),
    ("會議討論數位身分證換發時程，尚未定案。",
     ("委員詢問換發作業是否應延後至資安疑慮釐清後", "內政部說明現行資安檢測進度",
      "委員要求提出風險評估報告")),
    ("委員針對國家公園門票調整案表達意見。",
     ("委員質疑調漲幅度是否應考量在地居民負擔", "管理處說明現行收費結構")),
    ("會議審查食品標示新制實施期程。",
     ("委員詢問緩衝期是否應延長至一年", "食藥署說明業者準備情形",
      "委員要求加強稽查量能")),
)

_SYS = """你是一個會議記錄助手。以下是目前累積的會議記憶（ARC 與 POINTS）。

請根據這份記憶，寫出一段流暢連貫的繁體中文摘要，不超過 1000 個字：
- 不使用條列式、不使用小標題、不加時間戳記。
- 只寫一段連續的文字，讀起來像一篇完整的會議摘要，而不是重點清單。
- **記憶中若有「是否」這類提問語氣的重點，摘要必須保留其提問／未定調的語氣**，
  例如寫成「委員質疑……是否應……」或「就……是否應……提出質詢」，
  絕對不可以改寫成肯定句或否定句，也不可以省略該項重點。
- 全部使用繁體中文書寫。"""


def render_memory_prompt(arc: str, points: tuple[str, ...]) -> str:
    body = "\n".join(f"- {p}" for p in points)
    return f"MEMORY:\nARC: {arc}\nPOINTS:\n{body}\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8082")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    model = LlamaServer(base_url=args.url, max_tokens=1200, seed=0, temperature=0.3,
                        extra={"chat_template_kwargs": {"enable_thinking": False}})

    rows, refused = [], 0
    for i, (arc, points) in enumerate(CASES):
        prompt = render_memory_prompt(arc, points)
        raw = model(_SYS, prompt)
        text = to_traditional(" ".join(raw.split()))
        hedged = next((pt for pt in points if "是否" in pt), None)
        # The whole point of the row is that the question form survives. A target that
        # dropped or resolved it would teach exactly the behaviour being fixed.
        if "是否" not in text or len(text) < 60:
            refused += 1
            print(f"[hedge] case {i}: refused (是否 not preserved)", file=sys.stderr)
            continue
        rows.append({
            "meeting": f"hedge-{i}",
            "step": 999,  # synthesis rows carry no chunk index
            "prompt_version": PROMPT_VERSION,
            "system": synth_system_prompt(),  # the REAL shipped prompt, not `_SYS`
            "prompt": prompt,
            "completion": text,
            "is_nop": False,
        })
        print(f"[hedge] case {i}: ok ({hedged[:24]}…)", file=sys.stderr)

    out = args.out / "hedge_synth.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[hedge] wrote {len(rows)}, refused {refused} -> {out}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
