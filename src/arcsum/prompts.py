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
PROMPT_VERSION = "sys-v1"

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


def build_step_prompt(memory: Memory, chunk: Chunk) -> str:
    """MEMORY then CHUNK, fixed order — memory is small and stable in shape, the chunk
    is the varying part, and the model always reads them in the same places."""
    return f"MEMORY:\n{render_memory(memory)}\nCHUNK:\n{chunk.render()}\n"


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
