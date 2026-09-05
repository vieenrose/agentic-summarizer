"""Pins `arcsum.rl.step_reward`, the per-step RAFT ranking (SPEC §4.1 reading step).

The reward is a RANKING, so every test here asserts an ORDER between two candidates for the
same state, never an absolute score. Absolute values are free to move; what must not move is
which candidate a rejection sampler keeps, because that is the only thing training sees.

Each test names the measured failure it guards. All four penalties in this module were added
after a sampling run selected for the behaviour they price.
"""

from __future__ import annotations

from arcsum.chunker import Chunk
from arcsum.guards import apply_ops
from arcsum.memory import Memory
from arcsum.ops import Add, Arc, Drop, Nop, Revise
from arcsum.rl.step_reward import score_step
from arcsum.tokens import heuristic_token_len
from arcsum.transcript import Utterance

CHUNK_TEXT = (
    "S1: 市議會今日通過搬遷案，預算編列三十萬元，並要求於三月底前完成。\n"
    "S2: 另有委員提出交通改善計畫，將於下次會議討論。"
)


def _chunk() -> Chunk:
    lines = [ln.split(": ", 1) for ln in CHUNK_TEXT.splitlines()]
    return Chunk(
        index=0,
        utterances=[Utterance(a, b) for a, b in lines],
        tokens=heuristic_token_len(CHUNK_TEXT),
    )


def _score(ops, *, memory=None, raw=None):
    mem = memory if memory is not None else Memory(token_len=heuristic_token_len)
    chunk = _chunk()
    outcome = apply_ops(mem.clone(), ops, chunk, lang_check=False)
    text = raw if raw is not None else "\n".join(str(o) for o in ops)
    return score_step(outcome, ops, chunk_text=CHUNK_TEXT, raw=text)


def test_recording_content_beats_abstaining() -> None:
    """The whole point of RAFT here. `rl-v3` NOPs 46.2% of chunks on the held-out 40 and
    starves 17 of them; a reward that let NOP tie a real ADD could not move that."""
    add = _score([Add("市議會通過搬遷案，預算三十萬元")])
    nop = _score([Nop()])
    assert add.score > nop.score


def test_a_verbose_candidate_loses_to_a_concise_one_recording_the_same_thing() -> None:
    """G4 is a wall-clock gate and decode is 19.3 s of a 77.7 s step, so tokens are a priced
    resource. Before `DECODE_TOKEN_COST` the score was additive in applied ops, so these two
    tied and the sampler kept whichever came first -- measured, the kept rows ran 1.45x gold's
    completion length, enough on its own to move a meeting from 20.3 to 22.5 minutes."""
    op = [Add("市議會通過搬遷案，預算三十萬元")]
    concise = _score(op, raw='{"add":["市議會通過搬遷案，預算三十萬元"]}')
    padded = _score(op, raw='{"add":["市議會通過搬遷案，預算三十萬元"]}' + "，" * 300)
    assert concise.score > padded.score
    assert padded.decode_tokens > concise.decode_tokens


def test_brevity_never_outranks_recording_real_content() -> None:
    """The counterweight to the test above, and the reason `DECODE_TOKEN_COST` is calibrated
    against the op credit rather than picked for feel: an applied op is worth ~100 tokens, and
    a typical ADD costs ~25. A length penalty strong enough to make silence win would
    re-introduce exactly the abstention RAFT exists to remove."""
    recorded = _score([Add("市議會通過搬遷案，預算三十萬元")])
    silent = _score([Nop()], raw="NOP")
    assert recorded.score > silent.score


def test_saying_the_same_thing_twice_does_not_earn_twice() -> None:
    """One sampled step scored 6.75 while emitting the same 30-acre-negotiation point three
    times. The harness ACCEPTS near-duplicates -- they clear the exact-match refusal -- so they
    collected full applied credit for content `synthesis_view` was always going to collapse."""
    once = _score([Add("市議會通過搬遷案，預算三十萬元")])
    twice = _score([Add("市議會通過搬遷案，預算三十萬元"), Add("市議會通過搬遷案，預算為三十萬元")])
    assert twice.near_duplicates >= 1
    assert twice.score <= once.score


def test_two_genuinely_different_points_are_not_penalised_as_duplicates() -> None:
    """The negative control for the test above. A duplicate penalty that fired on distinct
    points would suppress recording and recreate starvation by another route."""
    both = _score([Add("市議會通過搬遷案，預算三十萬元"), Add("委員提出交通改善計畫")])
    one = _score([Add("市議會通過搬遷案，預算三十萬元")])
    assert both.near_duplicates == 0
    assert both.score > one.score


def test_an_ungrounded_claim_costs_more_than_the_op_earns() -> None:
    """Without this the reward credits anything the harness accepts, and the harness accepts
    anything well-formed -- so rejection sampling selects the most productive liar. Caught on
    the first on-policy run, where sampling against a stub chunk still produced 80 confident
    ADDs that the reward ranked highly."""
    grounded = _score([Add("市議會通過搬遷案，預算三十萬元")])
    invented = _score([Add("市議會通過搬遷案，預算八千九百萬元")])
    assert invented.ungrounded >= 1
    assert invented.score < grounded.score


def test_omitting_raw_disables_the_length_term_rather_than_crashing() -> None:
    """`raw` is optional so existing callers and tests keep working, but the docstring warns
    that omitting it re-opens the verbosity bias. Pinned so the default stays inert, not
    accidentally punitive."""
    s = score_step(
        apply_ops(
            Memory(token_len=heuristic_token_len).clone(),
            [Add("市議會通過搬遷案")],
            _chunk(),
            lang_check=False,
        ),
        [Add("市議會通過搬遷案")],
        chunk_text=CHUNK_TEXT,
    )
    assert s.decode_tokens == 0


def test_the_grounding_term_catches_invented_SPECIFICS_and_not_invented_TOPICS() -> None:
    """A LIMIT of the reward, pinned so it is not mistaken for coverage it does not have.

    `grounding.check` compares SPECIFICS -- figures, names, the things a deterministic checker
    can locate in the source. An ARC that invents a whole subject asserts nothing specific, so
    it passes with `ungrounded == 0`. That is not hypothetical: the first RAFT sampling run
    produced an ARC about 海豹保護 (seal protection) and 美食週 (a food week) for a closed
    session on labour negotiation and real estate, and the reward gave it a free pass.

    So `UNGROUNDED_PENALTY` bounds fabricated DETAIL, not fabricated SUBJECT, and the pool
    audit has to keep reading `specifics` beside `ungrounded` -- a candidate asserting nothing
    checkable scores a perfect fabrication rate. Closing this needs an instrument that can
    check topical relevance, which `evalkit.grounding` deliberately is not.
    """
    invented_topic = _score([Arc("會議討論海豹保護與美食週活動")])
    invented_figure = _score([Add("市議會通過搬遷案，預算八千九百萬元")])
    assert invented_topic.ungrounded == 0, "documented limit: topics are not checkable"
    assert invented_figure.ungrounded >= 1, "specifics ARE checked"


def test_a_bare_DROP_is_not_paid_for() -> None:
    """The defect that produced `runs/raft-s0-e1`.

    Crediting every applied op at +1 made DISCARDING a point count as work, so the cheapest
    way to score well was to edit at high volume. Trained, that fixed starvation exactly as
    designed (17/40 -> 5/40 starved, NOP 46.2% -> 7.9%) and took churn from 2.9% to **44.7%**,
    four times over G7's ceiling. The memory shape is the tell, not the churn counter:
    recorded points rose 366 -> 604 while SURVIVING points stayed flat at ~345, so
    retirements went 18 -> 259. It recorded more and threw almost all of it away.
    """
    mem = Memory(token_len=heuristic_token_len)
    mem.add_point("市議會通過搬遷案，預算三十萬元", chunk=0)
    pid = mem.points[0].pid
    dropped = _score([Drop(pid=pid)], memory=mem)
    idle = _score([Nop()], memory=mem)
    assert dropped.applied >= 1, "the harness still APPLIES it — this is about credit, not refusal"
    assert dropped.score <= idle.score, "dropping must not out-earn doing nothing"


def test_dropping_is_free_rather_than_punished() -> None:
    """The counterweight. Retiring a point that turned out irrelevant is legitimate, so a
    drop must stay FREE — penalising it would teach hoarding, which is how the working set
    fills with stale points and `revise` never fires."""
    mem = Memory(token_len=heuristic_token_len)
    mem.add_point("市議會通過搬遷案，預算三十萬元", chunk=0)
    pid = mem.points[0].pid
    drop_and_record = _score([Drop(pid=pid), Add("委員提出交通改善計畫")], memory=mem)
    record_only = _score([Add("委員提出交通改善計畫")], memory=mem)
    # The drop costs only its decode tokens, never a credit penalty.
    assert drop_and_record.score < record_only.score
    assert record_only.score - drop_and_record.score < 1.0


def test_revise_is_still_credited_because_it_carries_replacement_content() -> None:
    """`revise` is the sanctioned form of what DROP+ADD does badly (SPEC §4.1 v1.1). Making
    DROP free must not accidentally make revision free too, or G1's capability loses its
    incentive."""
    mem = Memory(token_len=heuristic_token_len)
    mem.add_point("市議會擬通過搬遷案", chunk=0)
    pid = mem.points[0].pid
    revised = _score([Revise(pid=pid, text="市議會通過搬遷案，預算三十萬元")], memory=mem)
    dropped = _score([Drop(pid=pid)], memory=mem)
    assert revised.score > dropped.score


def test_rewriting_the_ARC_does_not_outrank_recording_a_POINT() -> None:
    """The third instance of one pattern, after DROP.

    `ARC` replaces a single slot; `ADD` accumulates. Credited equally, rewriting the arc is
    the cheapest guaranteed credit on the board -- restate the gist, build no memory. The
    pressure is measured: the first RAFT pool carries 1.56x gold's ARC ops, and its trained
    checkpoints emit an ARC on nearly every step with **~48%** refused as `arc unchanged`,
    against `rl-v3`'s 22.9%.
    """
    arc = _score([Arc("市議會今日審議搬遷案與交通改善計畫")])
    point = _score([Add("市議會通過搬遷案，預算三十萬元")])
    assert arc.score < point.score


def test_the_arc_is_still_worth_setting() -> None:
    """Counterweight: halved, not zeroed. The arc is real work when the meeting's through-line
    moves -- SPEC §4.1 calls it the design's differentiator -- so an accepted rewrite must
    still beat doing nothing, or the slot goes permanently unused."""
    arc = _score([Arc("市議會今日審議搬遷案與交通改善計畫")])
    nothing = _score([Nop()])
    assert arc.score > nothing.score
