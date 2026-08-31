"""Model-facing prompt text: the SYS prompts, and the single builders that construct
every user-turn seen by trace generation, training, and inference alike.

**Byte-stability.** `PROMPT_VERSION` is recorded on every trace and eval record. A
silent edit to any prompt below invalidates train/eval comparability — bump the version
instead of editing quietly. Caps and thresholds are INTERPOLATED FROM THE CONSTANTS in
`arcsum.memory`/`arcsum.prose`, not restated as literals, so the prompt can never promise
a cap the harness does not actually enforce.

**One builder per prompt, reused everywhere.** `build_step_prompt` is used by trace
generation, SFT construction, and inference — so the three cannot drift apart. This
replaces the prior project's `build_step_prompt(state, Chunk(idx, ()))` trick (reusing
the step builder with an empty chunk to capture a state-only view) with a dedicated
`build_memory_view`.
"""

from __future__ import annotations

from collections.abc import Sequence

from arcsum.chunker import Chunk
from arcsum.memory import ARC_TOKENS, MIN_PREFIX_TOKENS, POINT_TOKENS, POINTS_CAP, Memory
from arcsum.prose import PROSE_MAX_TOKENS
from arcsum.render import render_memory

#: Bump on ANY change to the text below, or to a constant it interpolates.
PROMPT_VERSION = "sys-v2"

#: SPEC §4.1 v1.0 tool-call protocol. Separate constant so a v1 run is never confused
#: with a v0 one in any trace or eval record.
TOOLCALL_PROMPT_VERSION = "tools-v1"

_CAPS_LINE = (
    f"ARC 上限 {ARC_TOKENS} 個字；POINTS 最多 {POINTS_CAP} 項，每項上限 {POINT_TOKENS} 個字"
)
_PREFIX_RULE = f"前綴至少需要 {MIN_PREFIX_TOKENS} 個字"

_STEP_SYS = f"""你是一個會議記錄助手，正在閱讀一份會議逐字稿的其中一段（CHUNK），並維護一份精簡的記憶（MEMORY）。

記憶由兩部分組成（{_CAPS_LINE}）：
ARC：1 到 3 句話，描述會議目前為止的發展脈絡。
POINTS：關鍵論點、決議或承諾的清單。

閱讀完這段 CHUNK 後，只能用以下四種指令回覆，每行一個指令：

ADD - <重點內容>       新增一項重點
DROP «<前綴>»          移除先前記錄中，開頭符合此前綴的重點（{_PREFIX_RULE}）
ARC: <文字>            以新內容取代目前的 ARC 摘要
NOP                    這段內容沒有值得記錄的新資訊

規則：
- 只能輸出上述四種指令，不要輸出其他文字、標點符號、時間戳記或說明。
- 若這段內容推翻或修正了先前的重點，先 DROP 舊的重點，再 ADD 新的重點——不要同時保留互相矛盾的兩項重點。
- 全部使用繁體中文書寫。"""

_SYNTH_SYS = f"""你是一個會議記錄助手。以下是目前累積的會議記憶（ARC 與 POINTS）。

請根據這份記憶，寫出一段流暢連貫的繁體中文摘要，不超過 {PROSE_MAX_TOKENS} 個字：
- 不使用條列式、不使用小標題、不加時間戳記。
- 只寫一段連續的文字，讀起來像一篇完整的會議摘要，而不是重點清單。
- 全部使用繁體中文書寫。"""

_MAP_SYS = """你是一個會議記錄助手，正在閱讀一份會議逐字稿的其中一段。請將這段內容摘要成一段簡短的繁體中文文字：
- 不使用條列式、不加時間戳記，只描述這段內容本身，不需要考慮其他段落。
- 全部使用繁體中文書寫。"""

_REDUCE_SYS = f"""你是一個會議記錄助手。以下是同一場會議依序產生的多段摘要。

請將這些摘要整合成一段流暢連貫的繁體中文摘要，不超過 {PROSE_MAX_TOKENS} 個字：
- 不使用條列式、不加時間戳記，只寫一段連續的文字。
- 全部使用繁體中文書寫。"""


def step_system_prompt() -> str:
    return _STEP_SYS


def synth_system_prompt() -> str:
    return _SYNTH_SYS


def map_system_prompt() -> str:
    return _MAP_SYS


def reduce_system_prompt() -> str:
    return _REDUCE_SYS


def build_memory_view(memory: Memory) -> str:
    """The state-only view: just the rendered memory, no chunk. Used for the trace
    record's `state_before`/`state_after` and by `build_synth_prompt`, which is
    identical in content but kept as its own name for clarity at call sites."""
    return f"MEMORY:\n{render_memory(memory)}\n"


#: **REFUTED, 2026-08-28 — do not re-add an empty-memory label.** The reading step's
#: recency bias is a STEP-0 effect: over 6 real meetings, step-0 points score 0.134
#: trigram containment against their chunk's HEAD and 0.413 against its TAIL
#: (head-favoured 9/40), while steps 1+ are head-favoured (0.174 vs 0.102, 25/46). Adding
#: "（尚無任何記錄，這是會議的第一段，本段所有重點都是新的）" to the `MEMORY:` header DOES fix
#: that: the probe's chunk 0 then yields `ADD - 辦公室搬遷案決議透過，確定搬遷至B棟`.
#:
#: It still makes G1 WORSE — 1 of 2 probe cases passing under `sys-v2`, 0 of 2 with the
#: label, and all three wordings tried behaved the same. The mechanism: a fuller step-0
#: memory suppresses REVISION at step 1, where the model then emits an ARC-only update
#: with no `DROP` of the superseded point and no `ADD` of the reversal. One probe case
#: went from stating the reversal to asserting the STALE decision as current
#: (`states_earlier_as_current` flipped to True), which is exactly what G1 exists to
#: catch. Same shape as trap 7: fuller memory is not free.


def position_line(index: int, total: int) -> str:
    """The `POSITION:` prefix, defined ONCE. Offline tools that re-render a stored pool
    must emit the byte-identical line inference will send, so they import this rather
    than reproducing the format -- the same single-source-of-truth rule `tokens.py`
    applies to "is this character CJK"."""
    return f"POSITION: 第 {index + 1} 段，共 {total} 段\n"


#: SPEC §4.1 v1.0's step SYS prompt: a COMPACT hand-written tool schema, deliberately not
#: the chat template's `tools=` rendering. Measured on Qwen3.5-0.8B: the template preamble
#: is 313 tokens for one tool and 434 for four, against 81 for this and 266 for the v0
#: edit-line prompt. The rendered preamble is instruction boilerplate aimed at a model that
#: has never seen the schema; a fine-tuned student does not need it, and paying it on every
#: one of ~14 steps is the difference between fitting §7's budget and missing it.
_TOOL_STEP_SYS = f"""你是一個會議記錄助手。逐段閱讀會議逐字稿，維護一份精簡的記憶：
ARC（1到3句會議脈絡，上限 {ARC_TOKENS} 個字）與 POINTS（最多 {POINTS_CAP} 項，每項上限 {POINT_TOKENS} 個字）。

每讀完一段，只回覆一次工具呼叫：
<tool_call>{{"name":"update_memory","arguments":{{"arc":"…","add":["…"],"drop":["…"]}}}}</tool_call>

- add：新增重點。drop：移除開頭符合前綴的舊重點（前綴至少 {MIN_PREFIX_TOKENS} 個字）。arc：取代脈絡摘要。三者皆可省略。
- 若這段推翻了先前的重點，同時給 drop 與 add，不要保留互相矛盾的兩項。
- 這段沒有值得記錄的新資訊時，arguments 留空：{{}}。
- 全部使用繁體中文。"""


def tool_step_system_prompt() -> str:
    return _TOOL_STEP_SYS


def build_step_prompt(memory: Memory, chunk: Chunk, *, total: int | None = None) -> str:
    """POSITION, then MEMORY, then CHUNK, in that fixed order — memory is small and
    stable in shape, the chunk is the varying part, and the model always reads them in
    the same places.

    **The order is now MEASURED, not just argued.** Swapping to CHUNK-then-MEMORY was
    tested on the hypothesis that the model re-ADDs points already visible in memory
    because memory sits at the under-attended head of the prompt (the same positional
    weakness that drops chunk heads). It is much worse, on 4 meetings at the production
    budget: applied-op rate 79.0% -> 44.2%, duplicate points 12.2% -> 29.5%, unchanged
    ARCs 5.5% -> 11.4%, and total emitted ops 271 -> 509. Do not reorder these.

    **`POSITION` exists because late-step behaviour was otherwise unlearnable.** Until
    `sys-v2` the prompt carried no step index and no chunk count, so the model could not
    tell step 3 of 5 from step 44 of 55 except indirectly, through how saturated the
    memory happened to look. That made position-dependent behaviour impossible to
    *condition* and possible only to absorb globally, which is exactly what two measured
    builds did: adding genuine long-meeting supervision moved long meetings 4/9 -> 8/9
    -> 9/9 while pushing short ones 10/11 -> 6/11 -> 5/11, with
    `corr(meeting length, change) = +0.671`. Holding the NOP share fixed did not
    separate them (`runs/sft-dropv5/RESULT.md`), because the mix was never the cause.

    `total` is optional only so the state-only and map views stay buildable; every
    reading step passes it. Omitting it reproduces the `sys-v1` body exactly.
    """
    head = "" if total is None else position_line(chunk.index, total)
    return f"{head}MEMORY:\n{render_memory(memory)}\nCHUNK:\n{chunk.render()}\n"


def build_synth_prompt(memory: Memory) -> str:
    """The SYNTHESIZE call's user turn: just the final memory, no chunk."""
    return build_memory_view(memory)


def build_map_prompt(chunk: Chunk) -> str:
    """The baseline's per-window map call: a chunk with NO carried state — this is the
    defining property that makes it a fair, structurally different opponent."""
    return f"CHUNK:\n{chunk.render()}\n"


def build_reduce_prompt(summaries: Sequence[str]) -> str:
    """The baseline's single final compress call, over all window summaries."""
    body = "\n".join(f"- {s}" for s in summaries)
    return f"SUMMARIES:\n{body}\n"
