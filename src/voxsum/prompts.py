"""System protocol and per-step prompt assembly (CLAUDE.md §5.0, PLAN.md §2c).

`PROMPT_VERSION` is recorded in every trace and every eval run. A silent edit here
invalidates train/eval comparability (CLAUDE.md §7.8) — bump the version instead.

Two surfaces, one source:

* **text grammar** — the spec's op syntax (§5.1). Used for the teacher, and for the
  screen, because it measures whether a model *naturally* emits valid ops.
* **FunctionGemma declarations** — the student's post-trained call format, emitted by
  `function_declarations()` for SFT and inference.

`build_step_prompt` is the single builder used by trace generation, training, and
inference, so the three cannot drift (PLAN.md §3).
"""

from __future__ import annotations

from .chunker import Chunk
from .render import render_for_prompt
from .state import CAPS, MIN_PREFIX, NotesState

__all__ = [
    "PROMPT_VERSION",
    "build_step_prompt",
    "build_window_prompt",
    "function_declarations",
    "reduce_prompt",
    "system_prompt",
    "window_system_prompt",
]

PROMPT_VERSION = "sys-v1"

_CAPS_LINE = ", ".join(f"{s} {c}" for s, c in CAPS.items())

_SYS_EN = f"""\
You curate one evolving set of meeting NOTES as a transcript streams past you.

You are shown the current NOTES (STATE) and the next block of transcript lines (CHUNK).
Reply with edit operations only — one per line, no prose, no explanation, no markdown.

Sections: SUMMARY, DECISIONS, ACTIONS, OPEN, TOPICS. Caps: {_CAPS_LINE}.

Operations:
ADD <SECTION> - <bullet> [m:ss]
UPD <SECTION> «<old bullet prefix>» -> <new bullet> [m:ss]
DEL <SECTION> «<bullet prefix>»
CMP <SECTION>            (then up to the cap of rewritten bullets, one `- ` per line)
TITLE: <short title>
NOP

Rules:
- Every ADD and UPD bullet ends with an [m:ss] copied exactly from a line in THIS CHUNK.
- «prefix» is the first {MIN_PREFIX} or more characters of a bullet already in STATE,
  copied exactly.
- When this chunk changes something already in STATE — a decision reversed or approved, a
  deadline moved, an action reassigned — use UPD to revise that bullet. Do not add a second
  bullet that contradicts the first.
- Use DEL only when this chunk shows an existing bullet is wrong.
- Keep bullets short and factual: 20 words or fewer, stating what was decided or agreed.
- NOP alone is a complete, correct answer when this chunk changes nothing.
"""

_SYS_ZH = f"""\
你負責維護一份會議筆記（NOTES），逐段閱讀逐字稿並持續更新。

系統會提供目前的筆記（STATE）與下一段逐字稿（CHUNK）。
只回覆編輯指令，一行一個，不要加任何說明、前言或 markdown。

區段：SUMMARY, DECISIONS, ACTIONS, OPEN, TOPICS。上限：{_CAPS_LINE}。

指令：
ADD <SECTION> - <條目> [m:ss]
UPD <SECTION> «<原條目前綴>» -> <新條目> [m:ss]
DEL <SECTION> «<條目前綴>»
CMP <SECTION>            （接著寫出重寫後的條目，每行以 `- ` 開頭，不超過上限）
TITLE: <簡短標題>
NOP

規則：
- 每個 ADD 與 UPD 的條目結尾都要有 [m:ss]，必須從「本段」的某一行原樣抄錄。
- «前綴» 是 STATE 中既有條目開頭至少 {MIN_PREFIX} 個字元，必須原樣抄錄。
- 當本段推翻或改變 STATE 中已有的內容——決議被否決或通過、期限改變、負責人更換——
  請用 UPD 修改那一條，不要另外新增一條與前一條矛盾的條目。
- 只有在本段證明既有條目有誤時才使用 DEL。
- 條目簡短具體，20 字以內，寫清楚決定或共識了什麼。
- 若本段沒有任何需要更動，單獨回覆 NOP 就是完整且正確的答案。
"""

_SYS = {"en": _SYS_EN, "zh-TW": _SYS_ZH}

# FunctionGemma tool declarations (PLAN.md §0.2). The `<escape>` delimiter and the
# `declaration:NAME{...}` shape are the model's post-trained format — do not restyle them.
_DECLARATIONS = [
    (
        "ADD",
        "Append a new anchored bullet to a NOTES section.",
        [
            ("section", "STRING", "One of SUMMARY, DECISIONS, ACTIONS, OPEN, TOPICS."),
            ("bullet", "STRING", "The bullet text, 20 words or fewer."),
            ("anchor", "STRING", "The [m:ss] timestamp of the chunk line that states it."),
        ],
    ),
    (
        "UPD",
        "Replace an existing bullet, for example when a decision is revised.",
        [
            ("section", "STRING", "The section holding the bullet."),
            ("prefix", "STRING", f"First {MIN_PREFIX}+ characters of the existing bullet."),
            ("bullet", "STRING", "The replacement bullet text."),
            ("anchor", "STRING", "The [m:ss] timestamp supporting the revision."),
        ],
    ),
    (
        "DEL",
        "Remove a bullet this chunk shows to be wrong.",
        [
            ("section", "STRING", "The section holding the bullet."),
            ("prefix", "STRING", f"First {MIN_PREFIX}+ characters of the existing bullet."),
        ],
    ),
    (
        "TITLE",
        "Set the meeting title.",
        [("title", "STRING", "A short title, 8 words or fewer.")],
    ),
    ("NOP", "Report that this chunk changes nothing.", []),
]


def _declaration(name: str, description: str, params: list[tuple[str, str, str]]) -> str:
    if not params:
        return (
            f"declaration:{name}{{description:<escape>{description}<escape>,"
            f"parameters:{{properties:{{}},required:[],type:<escape>OBJECT<escape>}}}}"
        )
    props = ",".join(
        f"{key}:{{description:<escape>{desc}<escape>,type:<escape>{typ}<escape>}}"
        for key, typ, desc in params
    )
    required = ",".join(f"<escape>{key}<escape>" for key, _, _ in params)
    return (
        f"declaration:{name}{{description:<escape>{description}<escape>,"
        f"parameters:{{properties:{{{props}}},required:[{required}],"
        f"type:<escape>OBJECT<escape>}}}}"
    )


def function_declarations() -> str:
    """The FunctionGemma declaration block for the op set.

    The literal phrase below is FunctionGemma's documented prompt trigger — it is load
    bearing, not boilerplate.
    """
    body = "\n".join(_declaration(*d) for d in _DECLARATIONS)
    return (
        "You are a model that can do function calling with the following functions\n"
        f"<start_function_declaration>\n{body}\n<end_function_declaration>"
    )


def system_prompt(lang: str = "en", *, declarations: bool = False) -> str:
    """The SYS block. `declarations=True` appends the FunctionGemma tool declarations."""
    if lang not in _SYS:
        raise ValueError(f"unsupported language: {lang!r} (expected 'en' or 'zh-TW')")
    sys = _SYS[lang]
    return f"{sys}\n{function_declarations()}\n" if declarations else sys


def build_step_prompt(state: NotesState, chunk: Chunk, *, highlight: bool = False, lang: str = "en") -> str:
    """The per-step user content: STATE then CHUNK, in that fixed order.

    Order matters for prompt caching and for learnability — STATE is small and stable in
    shape, CHUNK is the varying part, and the model always reads them in the same places.

    `highlight` prepends a marker to commitment-bearing lines (A2, deterministic) —
    the [m:ss] text stays byte-intact, so the anchor-copy rule is unaffected.
    """
    rendered = chunk.render()
    if highlight:
        from .highlight import highlight_chunk
        rendered = highlight_chunk(rendered, lang)
    return f"STATE:\n{render_for_prompt(state)}\nCHUNK:\n{rendered}"


# --- map-reduce baseline prompts ------------------------------------------------
#
# The baseline must be a *fair* opponent: same NOTES v2 vocabulary, same anchor rule, same
# chunk size. What it must NOT have is STATE — independent per-window digests are the
# defining property of map-reduce, and the thing CURSOR is being measured against.

_MAP_EN = """\
You are summarising ONE block of a meeting transcript, in isolation.

Reply with bullets only, one per line, in this form:
TOPICS - Discussion topic [0:00]

SECTION is one of SUMMARY, DECISIONS, ACTIONS, OPEN, TOPICS.
Every bullet ends with an [m:ss] copied exactly from a line in this block.
Keep bullets short and factual: 20 words or fewer.
If this block contains nothing worth recording, reply NONE.
"""

_MAP_ZH = """\
你正在為會議逐字稿的「其中一段」做摘要，只看這一段。

只回覆條目，一行一個，格式如下：
TOPICS - 討論議題 [0:00]

SECTION 是 SUMMARY, DECISIONS, ACTIONS, OPEN, TOPICS 其中之一。
每個條目結尾都要有 [m:ss]，必須從本段的某一行原樣抄錄。
條目簡短具體，20 字以內。
若本段沒有值得記錄的內容，請回覆 NONE。
"""

_REDUCE_EN = """\
You are shortening one section of a set of meeting notes.

You will be given the section name, a cap, and the bullets collected from the whole
meeting. Reply with at most the cap number of bullets, one per line, each `- ` prefixed and
each keeping an [m:ss] taken from the bullets you were given. Merge duplicates, keep the
decisions and commitments, drop the incidental. Reply with bullets only.
"""

_REDUCE_ZH = """\
你要精簡會議筆記中的一個區段。

系統會給你區段名稱、數量上限，以及整場會議收集到的條目。請回覆不超過上限的條目，
一行一個，每行以 `- ` 開頭，並保留原本條目中的 [m:ss]。合併重複、保留決議與承諾、
刪去枝節。只回覆條目。
"""

_MAP_SYS = {"en": _MAP_EN, "zh-TW": _MAP_ZH}
_REDUCE_SYS = {"en": _REDUCE_EN, "zh-TW": _REDUCE_ZH}


def window_system_prompt(lang: str = "en") -> str:
    """SYS for the baseline's map step (also used by the coverage fallback, §5.3)."""
    if lang not in _MAP_SYS:
        raise ValueError(f"unsupported language: {lang!r}")
    return _MAP_SYS[lang]


def build_window_prompt(chunk: Chunk) -> str:
    """The map step's user content: the chunk alone. No STATE — that is the point."""
    return f"CHUNK:\n{chunk.render()}"


def reduce_prompt(lang: str, section: str, cap: int, bullets: list[str]) -> tuple[str, str]:
    """(system, user) for the baseline's reduce step on one over-cap section."""
    if lang not in _REDUCE_SYS:
        raise ValueError(f"unsupported language: {lang!r}")
    body = "\n".join(bullets)
    return _REDUCE_SYS[lang], f"SECTION: {section}\nCAP: {cap}\nBULLETS:\n{body}\n"
