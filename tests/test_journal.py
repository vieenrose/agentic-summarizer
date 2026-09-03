"""Pins SPEC §4.1 v1.1: the working set / journal split, ids, and `revise`.

Each test names the v1.0 measurement that forced the change. The numbers come from
`runs/PROJECT-REVIEW.md` and are the reason this redesign exists rather than another
round of fine-tuning.
"""

from __future__ import annotations

from arcsum.chunker import Chunk
from arcsum.guards import apply_ops
from arcsum.memory import POINTS_CAP, Memory
from arcsum.ops import Add, Drop, Revise
from arcsum.prompts import build_synth_prompt
from arcsum.render import render_memory
from arcsum.toolcalls import parse_tool_calls, render_tool_call
from arcsum.transcript import Utterance


def _chunk(index: int = 0) -> Chunk:
    return Chunk(index=index, utterances=[Utterance("S1", "測試內容。")], tokens=10)


def _mem() -> Memory:
    return Memory(token_len=len)


# --- the working set / journal split ------------------------------------------------

def test_eviction_retires_to_the_journal_instead_of_destroying():
    """v1.0's central defect. On the three longest held-out meetings the model correctly
    recorded 41, 23 and 27 points and 80%, 65% and 48% were evicted before synthesis ran.
    Nothing the model recorded may be lost."""
    m = _mem()
    for i in range(POINTS_CAP + 6):
        m.add_point(f"第{i}項決議", chunk=i)
    m.enforce_caps()

    assert len(m.points) == POINTS_CAP, "working set must stay bounded"
    assert len(m.journal) == 6, "everything evicted must be journalled"
    assert all(e.reason == "evicted" for e in m.journal)
    # The invariant that matters: total recorded content is conserved.
    assert len(m.points) + len(m.journal) == POINTS_CAP + 6


def test_synthesis_sees_everything_recorded_not_just_survivors():
    m = _mem()
    for i in range(POINTS_CAP + 4):
        m.add_point(f"第{i}項決議", chunk=i)
    m.enforce_caps()
    prompt = build_synth_prompt(m)
    for i in range(POINTS_CAP + 4):
        assert f"第{i}項決議" in prompt, f"point {i} lost before synthesis"


def test_the_model_never_sees_the_journal():
    """The journal is free precisely because it costs no per-step prefill. If it leaked
    into the step view, v1.1 would reintroduce the cost v1.0 could not afford (~19% of the
    transcript re-read over 37 chunks)."""
    m = _mem()
    for i in range(POINTS_CAP + 3):
        m.add_point(f"第{i}項決議", chunk=i)
    m.enforce_caps()
    step_view = render_memory(m)
    # Which points get evicted is `spread`'s business — it keeps the endpoints and thins
    # the middle, so the test asks the journal rather than assuming an order.
    assert m.journal, "precondition: something was evicted"
    for e in m.journal:
        assert e.point.text not in step_view, "an evicted point must not appear in the step prompt"
    assert len([ln for ln in step_view.splitlines() if ln.startswith("[")]) == POINTS_CAP


# --- id addressing ------------------------------------------------------------------

def test_points_render_with_stable_ids():
    m = _mem()
    m.add_point("同意搬到 B 棟大樓", chunk=0)
    assert "[1] 同意搬到 B 棟大樓" in render_memory(m)


def test_ids_are_never_reused_after_a_point_leaves():
    """A pid in the journal must stay traceable, so the counter only ever moves forward."""
    m = _mem()
    m.add_point("甲案通過", chunk=0)
    m.drop_id(1)
    m.add_point("乙案通過", chunk=1)
    assert [p.pid for p in m.points] == [2]
    assert m.journal[0].point.pid == 1


def test_drop_by_id_refuses_an_unknown_id_by_name():
    m = _mem()
    m.add_point("甲案通過", chunk=0)
    assert m.drop_id(99) == "no point with id 99"
    assert len(m.points) == 1, "a refused op must leave memory untouched"


# --- revise -------------------------------------------------------------------------

def test_revise_supersedes_atomically_and_records_the_link():
    """G1's requirement. Under v1.0 a reversal needed the model to hold both the decision
    and its overturning in a 480-token window at synthesis time; it managed 3/27. Here the
    harness guarantees the pairing however many chunks separate them."""
    m = _mem()
    m.add_point("公車路線調整案通過", chunk=0)
    assert m.revise_id(1, "公車路線調整案改為取消") is None
    assert [p.text for p in m.points] == ["公車路線調整案改為取消"]
    (entry,) = m.journal
    assert entry.reason == "superseded"
    assert entry.point.text == "公車路線調整案通過"
    assert entry.superseded_by == m.points[0].pid


def test_synthesis_shows_a_reversal_with_its_replacement():
    m = _mem()
    m.add_point("公車路線調整案通過", chunk=0)
    m.revise_id(1, "公車路線調整案改為取消")
    prompt = build_synth_prompt(m)
    assert "公車路線調整案通過" in prompt
    assert "公車路線調整案改為取消" in prompt
    assert "後改為" in prompt, "the supersession must be legible, not just both texts present"


def test_revise_refuses_an_unchanged_rewrite():
    m = _mem()
    m.add_point("甲案通過", chunk=0)
    assert m.revise_id(1, "甲案通過") == "revision unchanged"
    assert m.journal == [], "a refused revision must not journal anything"


def test_revise_is_not_counted_as_churn():
    """`restates_dropped` fires on DROP + near-identical ADD, which is exactly what a
    revision looked like in v1.0 — it cannot distinguish the two. `revise` is the
    sanctioned form, so it must not trip the churn detector that measured 28.2%."""
    m = _mem()
    m.add_point("公車路線調整案通過", chunk=0)
    out = apply_ops(m, [Revise(1, "公車路線調整案改為取消")], _chunk(1), lang_check=False)
    assert out.results[0].applied
    assert out.churn_points == [], "a revise must never register as churn"


# --- codec round-trip ---------------------------------------------------------------

def test_tool_call_round_trips_revise_and_id_drops():
    ops = [Drop(pid=3), Revise(4, "改為撤回"), Add("新增重點")]
    parsed = parse_tool_calls(render_tool_call(ops))
    assert Drop(pid=3) in parsed
    assert Revise(4, "改為撤回") in parsed
    assert Add("新增重點") in parsed


def test_drop_accepts_a_numeric_string_as_an_id():
    """A model emitting `"drop": ["3"]` has expressed the same intent as `[3]`; refusing
    on a quoting detail would be a parser tantrum, not a guard."""
    raw = '<tool_call>{"name":"update_memory","arguments":{"drop":["3"]}}</tool_call>'
    assert parse_tool_calls(raw) == [Drop(pid=3)]


def test_text_prefix_drops_still_parse():
    """v0.9's edit protocol and the entire existing supervision pool address by text.
    Dropping that support would invalidate every stored gold trace at once."""
    raw = '<tool_call>{"name":"update_memory","arguments":{"drop":["公車路線"]}}</tool_call>'
    assert parse_tool_calls(raw) == [Drop(prefix="公車路線")]


def test_malformed_revise_is_reported_not_silently_dropped():
    raw = '<tool_call>{"name":"update_memory","arguments":{"revise":[{"id":"x"}]}}</tool_call>'
    (op,) = parse_tool_calls(raw)
    assert type(op).__name__ == "Malformed"


def test_apply_ops_journals_a_dropped_point():
    m = _mem()
    m.add_point("甲案通過", chunk=0)
    apply_ops(m, [Drop(pid=1)], _chunk(1), lang_check=False)
    assert [e.reason for e in m.journal] == ["dropped"]


def test_clone_does_not_leak_speculative_journal_entries():
    """`apply_ops` mutates a clone and keeps it only if the step succeeds."""
    m = _mem()
    for i in range(POINTS_CAP + 2):
        m.add_point(f"第{i}項", chunk=i)
    c = m.clone()
    c.enforce_caps()
    assert c.journal and not m.journal
