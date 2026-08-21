"""Pins the byte-stability discipline (SPEC §4.1): `PROMPT_VERSION`, fixed segment
order, and caps that are interpolated from the enforced constants rather than restated
as literals a code change could silently drift away from.
"""

from __future__ import annotations

from arcsum.chunker import Chunk
from arcsum.memory import ARC_TOKENS, MIN_PREFIX_TOKENS, POINT_TOKENS, POINTS_CAP, Memory
from arcsum.prompts import (
    PROMPT_VERSION,
    build_map_prompt,
    build_memory_view,
    build_reduce_prompt,
    build_step_prompt,
    build_synth_prompt,
    map_system_prompt,
    reduce_system_prompt,
    step_system_prompt,
    synth_system_prompt,
)
from arcsum.prose import PROSE_MAX_TOKENS
from arcsum.render import render_memory
from arcsum.transcript import Utterance


def test_prompt_version_is_a_stable_string() -> None:
    assert PROMPT_VERSION == "sys-v1"


def test_step_system_prompt_is_zh_tw() -> None:
    sys = step_system_prompt()
    assert "ADD" in sys and "DROP" in sys and "ARC" in sys and "NOP" in sys
    assert "繁體中文" in sys


def test_step_system_prompt_names_only_the_four_ops() -> None:
    """SPEC §4.1's grammar is exactly ADD/DROP/ARC/NOP -- the removed UPD/CMP must
    never reappear in the prompt text."""
    sys = step_system_prompt()
    assert "UPD" not in sys
    assert "CMP" not in sys


def test_caps_line_is_interpolated_from_the_enforced_constants() -> None:
    """The prompt must never promise a cap the harness doesn't enforce."""
    sys = step_system_prompt()
    assert str(ARC_TOKENS) in sys
    assert str(POINT_TOKENS) in sys
    assert str(POINTS_CAP) in sys


def test_prefix_rule_is_interpolated_from_min_prefix_tokens() -> None:
    sys = step_system_prompt()
    assert str(MIN_PREFIX_TOKENS) in sys


def test_synth_system_prompt_states_the_prose_budget() -> None:
    sys = synth_system_prompt()
    assert str(PROSE_MAX_TOKENS) in sys
    assert "條列" in sys  # instructs against bullets


def test_synth_system_prompt_is_zh_tw() -> None:
    assert "繁體中文" in synth_system_prompt()


def test_map_system_prompt_carries_no_state_language() -> None:
    """The baseline's map step must not be told about any memory/ARC/POINTS concept --
    that would make it a disguised copy of the agent rather than a fair opponent."""
    sys = map_system_prompt()
    assert "ARC" not in sys
    assert "POINTS" not in sys


def test_reduce_system_prompt_states_the_prose_budget() -> None:
    assert str(PROSE_MAX_TOKENS) in reduce_system_prompt()


def test_build_step_prompt_orders_memory_before_chunk() -> None:
    """Fixed segment order for prompt-cache stability and learnability."""
    memory = Memory()
    memory.set_arc("摘要")
    chunk = Chunk(0, (Utterance("S1", "a"),), tokens=4)
    prompt = build_step_prompt(memory, chunk)
    assert prompt.index("MEMORY:") < prompt.index("CHUNK:")


def test_build_step_prompt_contains_the_rendered_memory_and_chunk() -> None:
    memory = Memory()
    memory.add_point("一項決議", chunk=0)
    chunk = Chunk(0, (Utterance("S1", "討論內容"),), tokens=8)
    prompt = build_step_prompt(memory, chunk)
    assert "一項決議" in prompt
    assert "S1: 討論內容" in prompt


def test_build_memory_view_matches_render_memory() -> None:
    memory = Memory()
    memory.set_arc("摘要")
    memory.add_point("一項決議", chunk=0)
    assert render_memory(memory) in build_memory_view(memory)


def test_build_memory_view_contains_no_chunk_content() -> None:
    memory = Memory()
    memory.add_point("一項決議", chunk=0)
    assert "CHUNK:" not in build_memory_view(memory)


def test_build_synth_prompt_matches_build_memory_view() -> None:
    """Same content by design -- SYNTHESIZE reads the final memory alone, no chunk."""
    memory = Memory()
    memory.set_arc("摘要")
    memory.add_point("一項決議", chunk=0)
    assert build_synth_prompt(memory) == build_memory_view(memory)


def test_build_map_prompt_contains_no_memory_concept() -> None:
    chunk = Chunk(0, (Utterance("S1", "討論內容"),), tokens=8)
    prompt = build_map_prompt(chunk)
    assert "MEMORY" not in prompt
    assert "S1: 討論內容" in prompt


def test_build_reduce_prompt_lists_all_summaries() -> None:
    prompt = build_reduce_prompt(["第一段摘要", "第二段摘要"])
    assert "第一段摘要" in prompt
    assert "第二段摘要" in prompt


def test_build_reduce_prompt_handles_a_single_summary() -> None:
    prompt = build_reduce_prompt(["唯一的摘要"])
    assert "唯一的摘要" in prompt


def test_build_reduce_prompt_handles_no_summaries() -> None:
    prompt = build_reduce_prompt([])
    assert "SUMMARIES:" in prompt
