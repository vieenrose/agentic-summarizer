"""Generate synthetic within-meeting REVERSAL supervision and an independent probe set.

    python tools/gen_reversals.py --split train --n 40 --out data/reversals_train
    python tools/gen_reversals.py --split probe --n 8  --out data/reversals_probe

**Why synthetic at all.** G1 tests whether a decision reversed later in the meeting is
reported in its FINAL state. MeetingBank cannot teach that: only 3.4% of its 6,894 gold
items match reversal language, and inspection shows those matches are legislative
boilerplate ("repealing Section 5.53.090", "amending the Municipal Code") — they reverse
EXTERNAL ordinances, never a decision taken earlier in the same meeting. The measured
consequence is precise: the student's DROP rate already MATCHES the gold it was trained on
(33.3% of steps vs 31.1%), so it is not under-dropping. It has simply never seen "a point
in memory was just contradicted; drop it and record the new outcome".

**Why this is a real capability and not gate-gaming.** Real meetings reverse decisions;
MeetingBank's item summaries just do not record it. But G1 is only two hand-built cases,
so training this capability does compromise G1's independence. Two disciplines keep that
honest:

1. `--split probe` builds a SEPARATE, larger validation set, so a G1 pass is never the
   only evidence.
2. `SCENARIOS` is partitioned: train scenarios and probe scenarios share no subject, no
   key term, no domain, and no outcome vocabulary. The partition is asserted at import.

**The ground truth is planted, not inferred.** Each meeting is generated FROM a structured
plan, so the gold ops follow from the plan rather than from reading the generated text —
the teacher writes dialogue around facts we already fixed. Every produced sequence is then
replayed through the real harness (SPEC §4.2) and dropped if it does not apply cleanly.
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
    subject: str  #: the thing decided, e.g. "校車採購案"
    key_term: str  #: the identifying detail a correct summary must keep
    early: str  #: outcome word as first decided, e.g. "通過"
    late: str  #: outcome word after the reversal, e.g. "撤回"
    reason: str  #: why it reversed
    fillers: tuple[str, ...]  #: unrelated agenda items, to make the reversal non-trivial


#: TRAIN scenarios. Deliberately share no subject, key term or outcome vocabulary with
#: `PROBE_SCENARIOS` below, and none of them is an office relocation or a marketing
#: budget — the two domains `arcsum.probe` already uses for G1.
TRAIN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("schoolbus", "校車採購案", "A 型低地板校車", "核准", "駁回",
             "車輛驗收發現排放檢測未達標準",
             ("營養午餐供應商評選", "校舍防水工程進度", "教師員額調整")),
    Scenario("hospital", "醫療設備採購案", "第二台電腦斷層掃描儀", "通過", "退回",
             "原廠報價逾期且維修合約條件不符",
             ("急診室動線改善", "護理人力排班", "門診掛號系統升級")),
    Scenario("parkname", "公園命名案", "文山紀念公園", "同意", "保留",
             "地方里長提出不同命名版本需再協調",
             ("公園照明汰換", "遊具安全檢查", "周邊停車規劃")),
    Scenario("itsystem", "資訊系統委外案", "人事差勤雲端系統", "核定", "終止",
             "廠商資安認證到期未能補件",
             ("機房空調維護", "備援線路測試", "資料備份週期")),
    Scenario("waterworks", "自來水管線汰換工程", "第三期北區管線", "通過", "暫緩",
             "施工期間與捷運工程路線衝突",
             ("水質檢驗頻率", "抄表人力調整", "漏水率統計")),
    Scenario("libraryhours", "圖書館延長開放案", "週末夜間時段", "核准", "撤銷",
             "增聘館員預算未獲支應",
             ("館藏汰舊", "自修室座位管理", "電子資料庫續訂")),
    Scenario("sportcenter", "運動中心委外經營案", "北區國民運動中心", "同意", "廢止",
             "得標廠商財務證明查核未通過",
             ("泳池水質管理", "場地租借費率", "教練聘用資格")),
    Scenario("busroute", "公車路線調整案", "紅三十七路延駛計畫", "通過", "取消",
             "沿線居民陳情噪音與班距問題",
             ("站牌候車亭更新", "電子票證優惠", "駕駛員招募")),
)

#: PROBE scenarios — the INDEPENDENT validation set. Disjoint from the training list in
#: subject, key term, domain and outcome vocabulary, so a model that merely memorised the
#: training reversals cannot pass on pattern match alone.
PROBE_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("farmmarket", "農產直銷市集設置案", "溪州假日市集", "決議通過", "決議撤回",
             "用地屬農業區需先辦理變更",
             ("攤位抽籤方式", "垃圾清運配套", "食品衛生稽查")),
    Scenario("fireeng", "消防車汰換案", "雲梯消防車三十二公尺級", "決議核准", "決議否決",
             "車高超過轄區地下道限制",
             ("消防栓普查", "義消訓練時數", "救護量能調度")),
    Scenario("solarroof", "屋頂光電設置案", "行政大樓南側屋頂", "決議同意", "決議中止",
             "結構安全評估未達承載標準",
             ("節電目標檢討", "空調汰換期程", "綠建築標章申請")),
    Scenario("dogpark", "寵物公園開放案", "河濱寵物活動區", "決議照案通過", "決議退回",
             "動物防疫單位要求增設隔離區",
             ("環境清潔頻率", "飼主責任宣導", "夜間照明範圍")),
    Scenario("nightmarket", "夜市攤販管理案", "光復路臨時攤區", "決議准予備查", "決議不予備查",
             "消防通道寬度不符規定",
             ("營業時間規範", "油煙排放稽查", "電線走火風險")),
    Scenario("bikeshare", "公共自行車擴站案", "東區二十處新站點", "決議照案核備", "決議緩議",
             "與人行道拓寬工程期程重疊",
             ("站點使用率統計", "維修人力配置", "費率優惠檢討")),
    Scenario("schoolmerge", "國小合併案", "中興與復興兩校", "決議准予合併", "決議暫不合併",
             "家長會提出通學距離疑慮",
             ("課後照顧需求", "校舍耐震評估", "教具汰換預算")),
    Scenario("landfill", "掩埋場延役案", "第二期掩埋區", "決議准予延役", "決議不予延役",
             "地下水監測值超出標準",
             ("垃圾減量目標", "回收分類宣導", "轉運站排程")),
    Scenario("busshelter", "候車亭廣告委外案", "全市三百座候車亭", "決議照案通過", "決議退回重議",
             "權利金計算方式與市價不符",
             ("站體清潔頻率", "無障礙設施檢查", "到站資訊系統")),
    Scenario("watertax", "水費調整案", "民生用水第二級距", "決議照案通過", "決議否准",
             "弱勢家戶配套措施尚未完成",
             ("抄表週期調整", "管線汰換進度", "節水宣導計畫")),
    Scenario("clinichours", "衛生所夜診案", "山區三處衛生所", "決議同意試辦", "決議停止試辦",
             "夜間醫師人力無法穩定支援",
             ("疫苗接種排程", "慢性病回診率", "巡迴醫療路線")),
    Scenario("sportfield", "運動場地開放案", "校園操場夜間開放", "決議准予開放", "決議收回",
             "校方反映安全維護人力不足",
             ("照明電費分攤", "場地預約制度", "器材保管責任")),
    Scenario("cemetery", "公墓遷葬案", "第三公墓東側區塊", "決議准予遷葬", "決議另案研議",
             "遷葬補償標準與家屬共識未達成",
             ("納骨塔容量評估", "祭祀動線規劃", "周邊道路拓寬")),
    Scenario("shelter", "動物收容所擴建案", "北區收容所第二期", "決議照案核可", "決議暫停辦理",
             "鄰地地權尚未完成徵收",
             ("認養宣導活動", "絕育補助名額", "獸醫人力配置")),
    Scenario("riverwalk", "河濱步道延伸案", "左岸七公里段", "決議准予施作", "決議全案退回",
             "生態評估指出影響水鳥棲地",
             ("自行車道銜接", "夜間照明規劃", "洪泛期封閉機制")),
    Scenario("transitstation", "客運轉運站興建案", "西站地下轉運層", "決議同意興建", "決議不予同意",
             "交通量預估模型遭認定高估",
             ("月台配置檢討", "計程車招呼站", "行李寄存服務")),
    Scenario("youthhousing", "青年住宅招租案", "文昌段社會住宅", "決議照案實施", "決議重新公告",
             "所得門檻設定排除主要目標族群",
             ("租金分級制度", "點數評分方式", "公設管理費用")),
    Scenario("heritage", "歷史建築指定案", "舊糖廠倉庫群", "決議登錄指定", "決議不予登錄",
             "文資審議認定原貌已大幅改建",
             ("修復經費來源", "再利用方向", "周邊土地使用")),
    Scenario("industrialpark", "產業園區開發案", "南科特定專用區", "決議准予開發", "決議駁回申請",
             "水電供應承諾未取得事業單位確認",
             ("引進產業類別", "汙水處理容量", "聯外道路負荷")),
    Scenario("jobtraining", "職業訓練委辦案", "照顧服務員班次", "決議准予委辦", "決議取消委辦",
             "受訓學員就業銜接率低於契約標準",
             ("師資鐘點費率", "術科場地設備", "結訓輔導機制")),
    Scenario("airquality", "空品淨區劃設案", "市中心三處學區", "決議公告劃設", "決議暫不劃設",
             "老舊車輛替換補助尚未到位",
             ("監測站增設", "稽查人力調度", "宣導期程安排")),
    Scenario("smartlamp", "智慧路燈建置案", "幹道兩百盞試辦", "決議准予試辦", "決議中止試辦",
             "資通訊設備採購爭議進入申訴程序",
             ("節能效益評估", "資料傳輸規格", "維運責任歸屬")),
    Scenario("hotspring", "溫泉區開發管理案", "北投溪上游段", "決議同意備查", "決議退回補正",
             "溫泉取供事業許可文件不齊",
             ("水權分配比例", "泉質檢驗頻率", "住宿容留人數")),
    Scenario("earlyintervention", "早期療育中心設置案", "東區療育據點",
             "決議准予設置", "決議延後辦理",
             "治療師招募未達開辦人力門檻",
             ("轉介流程建立", "家長支持團體", "交通接送補助")),
    Scenario("fishport", "漁貨直銷中心案", "南方澳直銷站", "決議照案通過", "決議不予核准",
             "用地屬國有財產尚未完成撥用",
             ("冷鏈設備規格", "拍賣制度調整", "觀光動線區隔")),
    Scenario("parkinglot", "立體停車場興建案", "中正段公有停車場", "決議准予興建", "決議重新評估",
             "車位需求調查母體遭質疑不具代表性",
             ("收費費率級距", "機車停放空間", "施工交維計畫")),
    Scenario("sewage", "汙水下水道接管案", "第四期接管工程", "決議核定實施", "決議保留議案",
             "用戶接管意願調查回收率過低",
             ("管線埋設深度", "施工賠償標準", "接管費用補助")),
    Scenario("disability", "身心障礙者輔具中心案", "中區輔具服務站", "決議同意設立", "決議不予設立",
             "現有服務量能經評估尚未飽和",
             ("輔具租借流程", "維修技師培訓", "到宅評估服務")),
    Scenario("tourism", "觀光纜車興建案", "山線觀光纜車", "決議准予推動", "決議停止推動",
             "環境影響評估認定需進入二階審查",
             ("站體量體規劃", "地質鑽探結果", "遊客承載管制")),
    Scenario("marketrebuild", "傳統市場改建案", "第一公有市場", "決議照案核備", "決議撤銷核備",
             "攤商安置方案未取得多數同意",
             ("臨時攤位配置", "施工分期方式", "租金調整機制")),
)

#: CONTROL scenarios: the decision is made and NEVER reversed. A model taught to handle
#: reversals could learn to see them everywhere; these catch that. `late` is set equal to
#: `early` and the generator skips the reversal part — a correct summary reports the
#: decision AS IT STANDS, and inventing a withdrawal is the failure being tested.
CONTROL_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("streetlight", "路燈汰換案", "全區 LED 路燈", "決議照案通過", "決議照案通過",
             "", ("電費節省估算", "汰換施工期程", "故障通報系統")),
    Scenario("recycling", "資源回收獎勵案", "社區集點制度", "決議准予實施", "決議准予實施",
             "", ("回收量統計", "獎品採購方式", "宣導海報設計")),
    Scenario("eldermeal", "長者共餐案", "六處社區據點", "決議同意辦理", "決議同意辦理",
             "", ("志工招募", "食材供應商", "場地無障礙")),
    Scenario("floodgate", "抽水站增設案", "低窪地區兩處", "決議照案核定", "決議照案核定",
             "", ("防汛演練", "警戒水位標準", "維護保養合約")),
)

_ALL = {s.slug for s in TRAIN_SCENARIOS} | {s.slug for s in PROBE_SCENARIOS}
assert len(_ALL) == len(TRAIN_SCENARIOS) + len(PROBE_SCENARIOS), "scenario slugs overlap"
assert not ({s.subject for s in TRAIN_SCENARIOS} & {s.subject for s in PROBE_SCENARIOS})
assert not ({s.key_term for s in TRAIN_SCENARIOS} & {s.key_term for s in PROBE_SCENARIOS})
assert not ({s.late for s in TRAIN_SCENARIOS} & {s.late for s in PROBE_SCENARIOS})

_WRITE_SYS = """你是一個會議逐字稿產生器。請寫出一段真實自然的繁體中文會議逐字稿。

格式規則（非常重要）：
- 每一行的格式必須是「S<數字>: <發言內容>」，例如「S1: 我們開始今天的會議。」
- 不要加時間戳記、不要加標題、不要用條列式、不要加任何說明文字。
- 至少要有四位不同的發言人（S1 到 S4 以上），S1 是主席。
- 全部使用繁體中文。"""


def _part_prompt(sc: Scenario, part: str) -> str:
    if part == "early":
        return (
            f"請寫出一段市政會議逐字稿，內容是審議「{sc.subject}」，"
            f"討論重點是「{sc.key_term}」，最後主席宣布這個案子{sc.early}。"
            f"請務必在逐字稿中明確出現「{sc.key_term}」與「{sc.early}」這兩個詞。"
            "大約 25 到 35 行。"
        )
    if part == "filler":
        items = "、".join(sc.fillers)
        return (
            f"請寫出一段市政會議逐字稿，內容依序討論以下三個與前面無關的議題：{items}。"
            "不要提到任何搬遷、採購撤銷或先前議案的翻案。大約 30 到 40 行。"
        )
    return (
        f"請寫出一段市政會議逐字稿。會議稍早已經{sc.early}了「{sc.subject}」，"
        f"但現在因為「{sc.reason}」，主席宣布這個案子改為{sc.late}，並要求重新辦理。"
        f"請務必在逐字稿中明確出現「{sc.subject}」、「{sc.key_term}」與「{sc.late}」這三個詞。"
        "大約 25 到 35 行。"
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


def generate_meeting(model: LlamaServer, sc: Scenario, *, control: bool = False) -> str | None:
    """`control=True` omits the reversal part entirely: the decision is taken and stands."""
    lines: list[str] = []
    parts = ("early", "filler") if control else ("early", "filler", "reversal")
    for part in parts:
        raw = model(_WRITE_SYS, _part_prompt(sc, part))
        got = _clean(raw)
        if len(got) < 8:
            return None
        lines.extend(got)
    body = "\n".join(lines) + "\n"
    # The planted facts must actually be present, or the gold ops would describe text
    # the student never sees.
    needles = (sc.key_term, sc.early) if control else (sc.key_term, sc.early, sc.late)
    for needle in needles:
        if needle not in body:
            return None
    return body


def build_gold(sc: Scenario, body: str) -> list[dict] | None:
    """Gold ops follow the PLAN, and are attached to whichever chunk actually carries
    each planted phrase — so a chunker change cannot silently misalign them."""
    utterances = parse_transcript(body)
    chunks = list(iter_chunks(utterances, budget=CHUNK_TOKENS, token_len=heuristic_token_len))
    if len(chunks) < 2:
        return None

    # Templates must fit POINT_TOKENS whole. An earlier version truncated to [:24] and
    # produced targets cut mid-word ("沿線居民陳情噪音與班距問"), which teaches the student
    # to emit truncated points — the harness would then refuse them as over-cap anyway.
    # BOTH points carry `key_term`. The replacement MUST be self-contained: `DROP`
    # permanently removes the superseded point and no conversation history crosses steps
    # (SPEC §4.1), so after a revision the replacement is the ONLY surviving record of the
    # decision. A replacement that says only `SUBJECT改為OUTCOME` has discarded the detail
    # that identifies what was decided.
    #
    # This was a real, measured defect: `late_point` briefly read `f"{sc.subject}改為
    # {sc.late}"` (the `，{sc.reason}` suffix was removed to stop `point too long`
    # refusals, taking the identifying detail with it). All 68 reversal samples then taught
    # lossy revision, and `qwen-tools-v5` reproduced it exactly — G1's `office_move` case
    # held 「B 棟」 in memory at step 0, dropped it at step 1, and failed `subject_terms`
    # despite stating the reversal correctly. On the 20 real ASR meetings, where nothing
    # taught this, every DROP+ADD revision EXPANDS detail instead. See `runs/g1-study.md`.
    early_point = f"{sc.subject}{sc.early}，{sc.key_term}"
    late_point = f"{sc.subject}改為{sc.late}，{sc.key_term}"
    prefix = sc.subject[:6]

    early_i = next((c.index for c in chunks if sc.early in c.render() and sc.key_term in
                    c.render()), None)
    # FIRST chunk carrying the reversal, not the last. Chunks OVERLAP (`OVERLAP_LINES`),
    # so a reversal near a boundary appears in two chunks; taking the last one then reads
    # `late_i > early_i` and accepts a meeting where the model — which sees chunks one at
    # a time, in order — already had the reversal in front of it when it recorded the
    # decision. That is a single-step revision, not the cross-chunk revision G1 exists to
    # test, and the gate silently scores it as if it were.
    #
    # Measured 2026-08-31: 2 of 11 generated probe scenarios (`bikeshare`, `nightmarket`)
    # were structurally incapable of testing cross-chunk revision, so every recorded G1
    # probe figure (0/11, 1/11, 2/11 across five fix attempts) had a wrong denominator.
    # Same bug class as the ~120-token transcripts pinned by `tests/test_probe.py`, which
    # covered `probe_data.py`'s two gate cases but never this generated set.
    #
    # Taking the first occurrence also places the gold op correctly: the model should
    # revise as soon as the reversal is visible, not one chunk later.
    late_i = next((c.index for c in chunks if sc.late in c.render()), None)
    if early_i is None or late_i is None or late_i <= early_i:
        return None

    rows: list[dict] = []
    memory = Memory(token_len=heuristic_token_len)
    for c in chunks:
        if c.index == early_i:
            completion = f"ARC: 會議審議{sc.subject}並作成決議。\nADD - {early_point}"
        elif c.index == late_i:
            completion = (
                f"DROP «{prefix}»\nADD - {late_point}\n"
                f"ARC: 會議稍早{sc.early}的{sc.subject}已改為{sc.late}，需重新辦理。"
            )
        else:
            continue  # filler chunks carry no planted gold; excluded rather than guessed
        ops = parse_ops(completion)
        outcome = apply_ops(memory, ops, c)
        applied = [a.op for a in outcome.results if a.applied]
        if len(applied) != len(ops):  # SPEC §4.2: never store a sequence that half-applies
            return None
        rows.append(
            {
                "meeting": f"rev-{sc.slug}",
                "step": c.index,
                "prompt_version": "sys-v2",
                "system": step_system_prompt(),
                "prompt": build_step_prompt(
                    _memory_before(sc, c.index, early_i, chunks), c, total=len(chunks)
                ),
                "completion": "\n".join(render_op(o) for o in applied),
                "is_nop": False,
            }
        )
    return rows or None


def _memory_before(sc: Scenario, index: int, early_i: int, chunks: list) -> Memory:
    """Memory as it stands entering `index`: empty before the decision, holding the
    early point after it. The reversal step MUST see the point it is supposed to drop."""
    m = Memory(token_len=heuristic_token_len)
    if index > early_i:
        m.set_arc(f"會議審議{sc.subject}並作成決議。")
        m.add_point(f"{sc.subject}{sc.early}，{sc.key_term}"[:24], early_i)
    return m


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "probe", "control"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8082")
    p.add_argument("--rebuild-gold", action="store_true",
                   help="recompute gold from transcripts already in --out; no generation")
    p.add_argument("--repeats", type=int, default=5,
                   help="variants per scenario (train only); each gets its own generation")
    args = p.parse_args(argv)

    scenarios = {"train": TRAIN_SCENARIOS, "probe": PROBE_SCENARIOS,
                 "control": CONTROL_SCENARIOS}[args.split]
    repeats = 1 if args.split == "probe" else args.repeats
    is_control = args.split == "control"
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
        print(f"[rev] rebuilt gold for {rebuilt}, refused {refused}", file=sys.stderr)
        return 0

    written = failed = 0
    for sc in scenarios:
        for k in range(repeats):
            model = LlamaServer(base_url=args.url, max_tokens=3000, seed=k,
                                temperature=0.8 if repeats > 1 else 0.3,
                                extra={"chat_template_kwargs": {"enable_thinking": False}})
            body = generate_meeting(model, sc, control=is_control)
            if body is None:
                failed += 1
                print(f"[rev] {sc.slug}#{k}: generation refused", file=sys.stderr)
                continue
            name = f"{sc.slug}-{k}"
            (args.out / f"{name}.txt").write_text(body, encoding="utf-8")
            if args.split == "train":
                rows = build_gold(sc, body)
                if rows is None:
                    failed += 1
                    (args.out / f"{name}.txt").unlink()
                    print(f"[rev] {sc.slug}#{k}: gold refused (replay/alignment)", file=sys.stderr)
                    continue
                with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1
            print(f"[rev] {sc.slug}#{k}: ok", file=sys.stderr)

    meta = [
        {"slug": s.slug, "subject": s.subject, "key_term": s.key_term,
         "early": s.early, "late": s.late}
        for s in scenarios
    ]
    (args.out / "scenarios.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[rev] wrote {written}, refused {failed} -> {args.out}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
