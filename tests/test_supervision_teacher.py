"""Pins `supervision.teacher` (SPEC §4.2 steps 1-3): grounded per-step gold-edit
generation that never leaks its grounding into the stored (prompt, completion) pair,
and correctly walks a whole meeting's chunks carrying memory forward via the real
harness (`ops.parse_ops`, `guards.apply_ops`).
"""

from __future__ import annotations

from tests.conftest import Scripted

from arcsum.memory import Memory
from arcsum.prompts import build_step_prompt, step_system_prompt
from arcsum.supervision.align import Item
from arcsum.supervision.teacher import (
    TEACHER_PROMPT_VERSION,
    build_teacher_step_prompt,
    generate_meeting_supervision,
    generate_step,
    replay_step_cleanly,
    teacher_step_system_prompt,
)
from arcsum.transcript import Utterance

ITEM_A = Item(
    item_id="2020-001",
    type="Ordinance",
    summary="市議會核准搬遷案，預算編列兩百萬元。",
    start_sec=0.0,
    end_sec=60.0,
)
ITEM_B = Item(
    item_id="2020-002",
    type="Resolution",
    summary="市議會通過決議，支持社區公園整修計畫。",
    start_sec=60.0,
    end_sec=120.0,
)


def _chunk(text: str = "S1: 市長已核准搬遷案，預算編列兩百萬元。"):
    from arcsum.chunker import iter_chunks

    utterances = [Utterance(*line.split(": ", 1)) for line in text.splitlines()]
    return next(iter_chunks(utterances))


# --- teacher_step_system_prompt / build_teacher_step_prompt --------------------------


def test_teacher_prompt_version_is_pinned() -> None:
    assert TEACHER_PROMPT_VERSION == "teacher-v1"


def test_covered_system_prompt_has_no_uncovered_suffix() -> None:
    covered = teacher_step_system_prompt(covered=True)
    uncovered = teacher_step_system_prompt(covered=False)
    assert covered != uncovered
    assert uncovered.startswith(covered)


def test_covered_system_prompt_contains_the_deployed_grammar() -> None:
    """The teacher's grammar must be the SAME ADD/DROP/ARC/NOP instructions the
    deployed model is trained on, not a bespoke re-description."""
    sys = teacher_step_system_prompt(covered=True)
    assert "ADD -" in sys
    assert "DROP «" in sys
    assert "ARC:" in sys
    assert "NOP" in sys


def test_build_teacher_step_prompt_shows_grounding_when_covered() -> None:
    memory = Memory()
    chunk = _chunk()
    prompt = build_teacher_step_prompt(memory, chunk, grounding_items=[ITEM_A])
    assert "對應議程摘要" in prompt
    assert ITEM_A.summary in prompt
    assert "[Ordinance]" in prompt


def test_build_teacher_step_prompt_shows_neighbours_when_uncovered() -> None:
    memory = Memory()
    chunk = _chunk()
    prompt = build_teacher_step_prompt(memory, chunk, concluded_items=[ITEM_A], next_item=ITEM_B)
    assert "鄰近議程項目" in prompt
    assert ITEM_A.summary in prompt
    assert ITEM_B.summary in prompt
    assert "對應議程摘要" not in prompt  # not covered -- no grounding section


def test_build_teacher_step_prompt_includes_concluded_items_for_arc() -> None:
    memory = Memory()
    chunk = _chunk()
    prompt = build_teacher_step_prompt(
        memory, chunk, grounding_items=[ITEM_B], concluded_items=[ITEM_A]
    )
    assert "已結束議程項目" in prompt
    assert ITEM_A.summary in prompt


def test_build_teacher_step_prompt_shows_next_item_foresight_only_when_covered() -> None:
    memory = Memory()
    chunk = _chunk()
    covered_prompt = build_teacher_step_prompt(
        memory, chunk, grounding_items=[ITEM_A], next_item=ITEM_B
    )
    assert covered_prompt.count(ITEM_B.summary) == 1
    assert "推翻" in covered_prompt


def test_build_teacher_step_prompt_starts_with_memory_then_chunk() -> None:
    memory = Memory()
    chunk = _chunk()
    prompt = build_teacher_step_prompt(memory, chunk)
    assert prompt.startswith("MEMORY:\n")
    assert "CHUNK:" in prompt
    assert prompt.index("MEMORY:") < prompt.index("CHUNK:")


# --- generate_step: grounding must never leak into the stored pair -------------------


def test_generate_step_calls_the_teacher_with_grounding() -> None:
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(default="ADD - 市議會核准搬遷案，預算編列兩百萬元。")

    generate_step(memory, chunk, teacher, grounding_items=[ITEM_A])

    assert len(teacher.calls) == 1
    _sys, user_sent = teacher.calls[0]
    assert ITEM_A.summary in user_sent


def test_generate_step_stores_the_plain_deployed_prompt_not_the_grounded_one() -> None:
    """The core correctness property: a student trained on this Step's (system, user)
    must never see grounding text it will never receive at real inference time."""
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(default="ADD - 市議會核准搬遷案，預算編列兩百萬元。")

    step, _memory_after = generate_step(memory, chunk, teacher, grounding_items=[ITEM_A])

    assert step.system == step_system_prompt()
    assert step.user == build_step_prompt(Memory(), chunk)
    assert ITEM_A.summary not in step.user
    assert ITEM_A.summary not in step.system
    assert "對應議程摘要" not in step.user


def test_generate_step_never_mutates_the_input_memory() -> None:
    """`generate_step` must apply ops against a CLONE, never the caller's object --
    the caller commits the returned memory explicitly, which is what makes a failed
    retry's partial mutation impossible to leak into the next attempt."""
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(default="ADD - 市議會核准搬遷案，預算編列兩百萬元。")

    _step, memory_after = generate_step(memory, chunk, teacher, grounding_items=[ITEM_A])

    assert len(memory.points) == 0
    assert len(memory_after.points) == 1


def test_generate_step_records_the_raw_teacher_completion() -> None:
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(default="NOP")
    step, _memory_after = generate_step(memory, chunk, teacher)
    assert step.raw == "NOP"
    assert step.is_nop


# --- generate_step retry-on-failed-replay ---------------------------------------------


def test_generate_step_retries_once_on_a_failed_replay() -> None:
    """The first attempt is garbled (a stray unmatched DROP mixed with a real ADD);
    the second attempt is clean. The retried step must be the CLEAN one, and
    memory_after must reflect only the successful attempt's ops."""
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(
        responses=("DROP «不存在的重點»\nADD - 市議會核准搬遷案，預算編列兩百萬元。",),
        default="ADD - 市議會核准搬遷案，預算編列兩百萬元。",
    )

    step, memory_after = generate_step(memory, chunk, teacher)

    assert len(teacher.calls) == 2
    assert step.retried is True
    assert replay_step_cleanly(step) is True
    assert len(memory_after.points) == 1


def test_generate_step_nudges_the_retry_prompt_so_it_differs_from_the_first_attempt() -> None:
    """The teacher is called at temperature 0 in practice: a bare repeat of the
    identical prompt would only ever reproduce the identical (already-failed) output.
    The retry attempt's user turn must differ from the first's."""
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(default="DROP «不存在的重點前綴內容測試»")

    generate_step(memory, chunk, teacher, max_attempts=2)

    assert len(teacher.calls) == 2
    first_user = teacher.calls[0][1]
    second_user = teacher.calls[1][1]
    assert second_user != first_user
    assert second_user.startswith(first_user)


def test_generate_step_drops_the_step_when_every_attempt_fails_replay() -> None:
    """A persistently-bad teacher (same invalid DROP every time) exhausts retries;
    the caller's memory must come back UNCHANGED -- SPEC §4.2: never half-applied."""
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(default="DROP «不存在的重點前綴內容測試»")

    step, memory_after = generate_step(memory, chunk, teacher, max_attempts=2)

    assert len(teacher.calls) == 2
    assert replay_step_cleanly(step) is False
    assert memory_after is memory
    assert len(memory_after.points) == 0


# --- replay_step_cleanly --------------------------------------------------------------


def test_replay_step_cleanly_true_when_all_ops_applied() -> None:
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(default="ADD - 市議會核准搬遷案，預算編列兩百萬元。")
    step, _memory_after = generate_step(memory, chunk, teacher)
    assert replay_step_cleanly(step) is True


def test_replay_step_cleanly_true_for_a_pure_nop_step() -> None:
    memory = Memory()
    chunk = _chunk()
    teacher = Scripted(default="NOP")
    step, _memory_after = generate_step(memory, chunk, teacher)
    assert replay_step_cleanly(step) is True


def test_replay_step_cleanly_false_when_a_drop_prefix_does_not_match() -> None:
    memory = Memory()  # empty -- no point can match any DROP prefix
    chunk = _chunk()
    teacher = Scripted(default="DROP «不存在的重點前綴內容測試»")
    step, _memory_after = generate_step(memory, chunk, teacher)
    assert replay_step_cleanly(step) is False


# --- generate_meeting_supervision ------------------------------------------------------


def test_generate_meeting_supervision_walks_every_chunk() -> None:
    utterances = [
        Utterance("S1", "市議會核准搬遷案，預算編列兩百萬元。"),
        Utterance("S2", "市議會通過決議，支持社區公園整修計畫。"),
    ]
    offsets = [(0.0, 60.0), (60.0, 120.0)]
    items = [ITEM_A, ITEM_B]
    teacher = Scripted(default="NOP")

    trace = generate_meeting_supervision(utterances, offsets, items, teacher, budget=50)

    assert len(trace.steps) >= 1
    assert trace.synthesis is None


def test_generate_meeting_supervision_carries_memory_forward_between_steps() -> None:
    utterances = [
        Utterance("S1", "市議會核准搬遷案，預算編列兩百萬元。"),
        Utterance("S2", "市議會通過決議，支持社區公園整修計畫。"),
    ]
    offsets = [(0.0, 60.0), (60.0, 120.0)]
    items = [ITEM_A, ITEM_B]
    teacher = Scripted(
        responses=(
            "ADD - 市議會核准搬遷案，預算編列兩百萬元。",
            "ADD - 市議會通過決議，支持社區公園整修計畫。",
        )
    )

    trace = generate_meeting_supervision(utterances, offsets, items, teacher, budget=30)

    # Both steps' points must survive in the FINAL memory -- proof state carried across.
    assert len(trace.memory.points) == len(trace.steps)


def test_generate_meeting_supervision_grounds_each_step_from_its_own_overlapping_item() -> None:
    """A stronger end-to-end check: step i's teacher call is grounded on the item that
    actually overlaps chunk i's time span, not a fixed or misaligned one."""
    utterances = [
        Utterance("S1", "市議會核准搬遷案，預算編列兩百萬元。"),
        Utterance("S2", "市議會通過決議，支持社區公園整修計畫。"),
    ]
    offsets = [(0.0, 60.0), (60.0, 120.0)]
    items = [ITEM_A, ITEM_B]
    teacher = Scripted(default="NOP")

    generate_meeting_supervision(utterances, offsets, items, teacher, budget=30)

    seen_users = [u for _s, u in teacher.calls]
    assert any(ITEM_A.summary in u for u in seen_users)
    assert any(ITEM_B.summary in u for u in seen_users)


def test_generate_meeting_supervision_tracks_coverage_gaps_on_nop_collapse() -> None:
    from arcsum.guards import NOP_COLLAPSE_K

    utterances = [
        Utterance("S1", f"這是一段有內容的發言編號 {i}，內容值得記錄，理應被視為重點。")
        for i in range(NOP_COLLAPSE_K + 1)
    ]
    offsets = [(float(i), float(i + 1)) for i in range(len(utterances))]
    teacher = Scripted(default="NOP")

    trace = generate_meeting_supervision(utterances, offsets, [], teacher, budget=20)

    assert len(trace.coverage_gaps) >= 1
