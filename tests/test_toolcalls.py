"""Pins SPEC §4.1 v1.0's `update_memory` tool call.

The step grammar changed; the harness did not. These tests assert the parser lands on the
same `Op` list the guards, memory and eviction rule already consume, so the two protocols
stay comparable on one harness instead of forking into two.
"""

from __future__ import annotations

from arcsum.ops import Add, Arc, Drop, Malformed, Nop
from arcsum.toolcalls import parse_tool_calls, render_tool_call


def test_parses_a_batched_call_into_ops() -> None:
    raw = (
        '<tool_call>{"name":"update_memory","arguments":'
        '{"arc":"會議處理搬遷案","add":["搬遷案通過"],"drop":["舊辦公室"]}}</tool_call>'
    )

    ops = parse_tool_calls(raw)

    assert [type(o) for o in ops] == [Drop, Arc, Add]
    assert ops[0].prefix == "舊辦公室"
    assert ops[1].text == "會議處理搬遷案"
    assert ops[2].point == "搬遷案通過"


def test_drop_is_emitted_before_add() -> None:
    """§4.1's inversion guard refuses a contradicting ADD unless the superseded point was
    dropped EARLIER IN THE SAME STEP, so it reads emission order. A batched call has no
    inherent order, which is exactly why this one is fixed and pinned."""
    raw = (
        '<tool_call>{"name":"update_memory","arguments":'
        '{"add":["新決議"],"drop":["舊決議"]}}</tool_call>'
    )

    ops = parse_tool_calls(raw)

    assert isinstance(ops[0], Drop) and isinstance(ops[-1], Add)


def test_empty_arguments_is_the_v1_spelling_of_nop() -> None:
    """The model was asked and answered "nothing here" — which is not the same as
    emitting nothing, and must not be scored as a malformed step."""
    ops = parse_tool_calls('<tool_call>{"name":"update_memory","arguments":{}}</tool_call>')

    assert [type(o) for o in ops] == [Nop]


def test_never_raises_on_garbage_and_records_it() -> None:
    """Same policy as `ops.parse_ops`: a bad step is DATA for `valid_op_rate`, never an
    exception that aborts the meeting."""
    for raw in ("not json at all", "<tool_call>{broken</tool_call>", "<tool_call>[]</tool_call>"):
        ops = parse_tool_calls(raw)
        assert ops and isinstance(ops[0], Malformed), raw


def test_unknown_tool_is_recorded_not_skipped() -> None:
    ops = parse_tool_calls('<tool_call>{"name":"delete_everything","arguments":{}}</tool_call>')

    assert isinstance(ops[0], Malformed)


def test_scalar_add_is_accepted_as_a_single_point() -> None:
    """`"add":"點"` expresses the same intent as `"add":["點"]`; scoring that as malformed
    would count a formatting slip as a curation failure."""
    raw = '<tool_call>{"name":"update_memory","arguments":{"add":"單一重點"}}</tool_call>'
    ops = parse_tool_calls(raw)

    assert [type(o) for o in ops] == [Add]
    assert ops[0].point == "單一重點"


def test_render_round_trips_through_the_parser() -> None:
    """Supervision is built with `render_tool_call`, so a drift between it and the parser
    would train the student on text the harness cannot read back."""
    ops = [Drop("舊案"), Arc("會議脈絡"), Add("第一點"), Add("第二點")]

    assert parse_tool_calls(render_tool_call(ops)) == ops
