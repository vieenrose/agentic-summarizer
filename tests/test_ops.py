"""Pins SPEC §4.1's step grammar: ADD / DROP / ARC / NOP.

Guards two decisions the prior project's much larger op set doesn't need: parsing is
LINE-LOCAL (no multi-line accumulation state, since the rewrite op that needed it is
gone), and it NEVER RAISES — a malformed line becomes a `Malformed` record, logged, never
fatal, so nothing the model emits can crash the harness.
"""

from __future__ import annotations

import pytest

from arcsum.ops import Add, Arc, Drop, Malformed, Nop, Op, parse_ops, render_op


def test_add_parses_the_spec_syntax() -> None:
    assert parse_ops("ADD - 同意搬到 B 棟") == [Add("同意搬到 B 棟")]


def test_drop_parses_the_spec_syntax() -> None:
    assert parse_ops("DROP «同意搬到 B 棟»") == [Drop("同意搬到 B 棟")]


def test_arc_parses_the_spec_syntax() -> None:
    assert parse_ops("ARC: 會議討論辦公室搬遷") == [Arc("會議討論辦公室搬遷")]


def test_nop_parses() -> None:
    assert parse_ops("NOP") == [Nop()]


@pytest.mark.parametrize(
    "line",
    [
        "DROP «搬遷案»",
        "DROP <<搬遷案>>",
        "DROP「搬遷案」",
        'DROP "搬遷案"',
    ],
)
def test_drop_accepts_all_four_prefix_delimiter_variants(line: str) -> None:
    """A small model will emit whichever delimiter it has seen most in pretraining."""
    assert parse_ops(line) == [Drop("搬遷案")]


def test_arc_accepts_fullwidth_colon() -> None:
    assert parse_ops("ARC：會議討論辦公室搬遷") == [Arc("會議討論辦公室搬遷")]


def test_add_accepts_various_dash_styles() -> None:
    for dash in ("-", "–", "—"):
        assert parse_ops(f"ADD {dash} 同意搬到 B 棟") == [Add("同意搬到 B 棟")]


def test_add_accepts_no_dash_at_all() -> None:
    assert parse_ops("ADD 同意搬到 B 棟") == [Add("同意搬到 B 棟")]


def test_nop_accepts_trailing_punctuation() -> None:
    assert parse_ops("NOP。") == [Nop()]
    assert parse_ops("NOP.") == [Nop()]


def test_only_four_ops_parse_everything_else_is_malformed() -> None:
    """SPEC §4.1 lists exactly ADD/DROP/ARC/NOP. UPD and CMP are gone."""
    for line in ("UPD - something -> other", "CMP SUMMARY", "TITLE: x"):
        ops = parse_ops(line)
        assert len(ops) == 1
        assert isinstance(ops[0], Malformed)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "just some random prose",
        "ADD",
        "ADD -",
        "ADD - ",
        "ARC:",
        "ARC: ",
        "DROP",
        "DROP «»",
        "DROP «  »",
        "NOP extra garbage",
        "the model rambled: ADD something",
        "「」",
        "\x00\x01\x02",
        "ADD " + "很" * 3000,
    ],
)
def test_parse_ops_never_raises(line: str) -> None:
    """Nothing the model emits can crash the harness."""
    ops = parse_ops(line)
    assert isinstance(ops, list)
    for op in ops:
        assert isinstance(op, Op)  # type: ignore[misc] — runtime isinstance against a union alias


def test_empty_add_body_is_malformed() -> None:
    ops = parse_ops("ADD -")
    assert len(ops) == 1
    assert isinstance(ops[0], Malformed)
    assert "empty" in ops[0].reason


def test_empty_drop_prefix_is_malformed() -> None:
    ops = parse_ops("DROP «»")
    assert len(ops) == 1
    assert isinstance(ops[0], Malformed)
    assert "empty" in ops[0].reason


def test_empty_arc_is_malformed() -> None:
    ops = parse_ops("ARC:")
    assert len(ops) == 1
    assert isinstance(ops[0], Malformed)


def test_unmatched_line_is_malformed_with_a_reason() -> None:
    ops = parse_ops("the model said something unstructured")
    assert ops == [
        Malformed("the model said something unstructured", "does not match the op grammar")
    ]


def test_blank_lines_are_skipped_not_recorded() -> None:
    ops = parse_ops("ADD - a\n\n   \nNOP")
    assert ops == [Add("a"), Nop()]


def test_ops_are_returned_in_emission_order() -> None:
    text = "ADD - a\nDROP «b»\nARC: c\nNOP"
    assert parse_ops(text) == [Add("a"), Drop("b"), Arc("c"), Nop()]


def test_match_order_arc_before_add() -> None:
    """An ARC line must never be misread as an ADD, or vice versa."""
    assert parse_ops("ARC: 摘要") == [Arc("摘要")]
    assert isinstance(parse_ops("ARCHIVE the plan")[0], Malformed)  # 'ARC' as a prefix, not the op


def test_hallucinated_timestamp_is_stripped_from_add() -> None:
    """v2 has no timestamps (SPEC §2); a hallucinated [m:ss] must not enter memory."""
    assert parse_ops("ADD - 同意搬到 B 棟 [3:35]") == [Add("同意搬到 B 棟")]
    assert parse_ops("ADD - 同意搬到 B 棟 [1:02:07]") == [Add("同意搬到 B 棟")]


def test_hallucinated_timestamp_is_stripped_from_arc() -> None:
    assert parse_ops("ARC: 會議摘要 [3:35]") == [Arc("會議摘要")]


def test_multiline_response_parses_each_line_independently() -> None:
    text = "ADD - 第一項\nADD - 第二項\nDROP «第一項»\nARC: 更新後摘要\nNOP"
    ops = parse_ops(text)
    assert len(ops) == 5
    assert [type(op).__name__ for op in ops] == ["Add", "Add", "Drop", "Arc", "Nop"]


def test_render_op_round_trips() -> None:
    for op in (Add("a"), Drop("b"), Arc("c"), Nop()):
        assert parse_ops(render_op(op)) == [op]


def test_render_op_uses_the_spec_delimiters() -> None:
    assert render_op(Add("x")) == "ADD - x"
    assert render_op(Drop("y")) == "DROP «y»"
    assert render_op(Arc("z")) == "ARC: z"
    assert render_op(Nop()) == "NOP"


def test_render_malformed_returns_the_raw_text() -> None:
    assert render_op(Malformed("garbled input", "does not match the op grammar")) == "garbled input"


def test_empty_text_yields_no_ops() -> None:
    assert parse_ops("") == []


def test_render_op_handles_every_op_type_including_revise():
    """`Revise` landed with SPEC §4.1 v1.1 and `render_op` was not updated, so it raised
    "unhandled op type" the first time a checkpoint emitted one — during the G1 REVISION
    probe, the very gate the op exists to serve. This asserts EXHAUSTIVENESS rather than
    spot-checking one case, so the next op added cannot repeat it."""
    from arcsum.ops import Add, Arc, Drop, Malformed, Nop, Revise, render_op

    every = [
        Add("同意搬到 B 棟大樓"),
        Drop(prefix="同意搬到"),
        Drop(pid=3),
        Revise(1, "改為撤回"),
        Arc("會議討論搬遷案"),
        Nop(),
        Malformed("garbage", "unparseable"),
    ]
    for op in every:
        assert render_op(op), f"render_op produced nothing for {op!r}"

    assert render_op(Drop(pid=3)) == "DROP #3"
    assert render_op(Revise(1, "改為撤回")) == "REVISE #1 - 改為撤回"
