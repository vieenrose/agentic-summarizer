"""Pins SPEC §4.1's memory render.

Guards against the prior project's incidental byte-level conditionality
(`f"TITLE: {title}".rstrip()` rendered differently depending on emptiness): here every
slot's shape is unconditional, pinned byte-for-byte against the spec's own example.
"""

from __future__ import annotations

from arcsum.memory import Memory
from arcsum.render import EMPTY, render_memory


def test_render_matches_the_spec_example_byte_for_byte() -> None:
    """SPEC §4.1's literal form:

    ARC: <1-3 sentences: how the meeting has developed so far>
    POINTS:
    [1] <key point, decision, or commitment>

    SPEC §4.1 v1.1 renders a stable id per point, because the model addresses points by
    id. v1.0 rendered `- ` and addressed by text prefix, which required the model to
    reproduce a prefix of its own earlier phrasing — measured churn 28.2% of steps.
    """
    m = Memory()
    m.set_arc("會議討論辦公室搬遷，已決定遷至 B 棟。")
    m.add_point("同意搬到 B 棟大樓", chunk=0)
    assert render_memory(m) == (
        "ARC: 會議討論辦公室搬遷，已決定遷至 B 棟。\nPOINTS:\n[1] 同意搬到 B 棟大樓\n"
    )


def test_empty_arc_renders_as_dash() -> None:
    m = Memory()
    m.add_point("一項決議", chunk=0)
    assert render_memory(m).startswith(f"ARC: {EMPTY}\n")


def test_empty_points_renders_as_dash() -> None:
    m = Memory()
    m.set_arc("會議剛開始")
    assert render_memory(m).endswith(f"POINTS:\n{EMPTY}\n")


def test_fully_empty_memory_renders_both_slots_as_dash() -> None:
    assert render_memory(Memory()) == f"ARC: {EMPTY}\nPOINTS:\n{EMPTY}\n"


def test_render_shape_is_unconditional_regardless_of_content() -> None:
    """Whether ARC/POINTS are empty or full, the byte shape (labels, order, newlines)
    never changes — only the payload does."""
    empty = render_memory(Memory())
    m = Memory()
    m.set_arc("摘要")
    m.add_point("一項", chunk=0)
    full = render_memory(m)
    assert empty.startswith("ARC: ") and full.startswith("ARC: ")
    assert "\nPOINTS:\n" in empty and "\nPOINTS:\n" in full
    assert empty.endswith("\n") and full.endswith("\n")


def test_render_applies_caps_by_default() -> None:
    m = Memory()
    for i in range(20):
        m.add_point(f"第{i}項", chunk=i)
    rendered = render_memory(m)
    assert len([ln for ln in rendered.splitlines() if ln.startswith("[")]) == 16  # POINTS_CAP


def test_render_without_enforcing_caps_shows_the_raw_overflow() -> None:
    m = Memory()
    for i in range(20):
        m.add_point(f"第{i}項", chunk=i)
    rendered = render_memory(m, enforce_caps=False)
    assert len([ln for ln in rendered.splitlines() if ln.startswith("[")]) == 20


def test_render_does_not_mutate_the_source_memory() -> None:
    """render_memory(enforce_caps=True) must clone before spreading, or a diagnostic
    render silently truncates the caller's live memory."""
    m = Memory()
    for i in range(20):
        m.add_point(f"第{i}項", chunk=i)
    render_memory(m)
    assert len(m.points) == 20
