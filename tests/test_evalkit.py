"""Pins `arcsum.evalkit`: the provenance, behaviour and grounding instruments.

Each test names the incident it guards. All of them were real, and all of them cost either
an experiment or — in the churn case — a public release that had to be rolled back.

No GPU, no weights, no network: `serving_identity` is exercised through an injected opener,
exactly as `test_judge.py` stubs `urllib.request.urlopen`.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from arcsum.evalkit import behaviour, grounding
from arcsum.evalkit.provenance import CorpusFingerprint, Provenance, serving_identity
from arcsum.evalkit.scorecard import (
    Check,
    IncomparableScorecards,
    Scorecard,
    compare,
)


class _Resp:
    def __init__(self, payload: dict):
        self._b = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._b


def _prov(**over) -> Provenance:
    base = dict(
        model_path="/m/a.gguf", model_sha256="aaa", protocol="tool",
        prompt_version="sys-v2", tokenize_version="chartok-v1",
        generation={"cache_prompt": False, "temperature": 0.0},
        corpora={"probe": CorpusFingerprint("data/probe", 27, "deadbeef")},
        label="a", checkpoint="626", epoch="best", code_revision="abc123",
    )
    base.update(over)
    return Provenance(**base)


# --- provenance -------------------------------------------------------------------

def test_serving_identity_reports_the_model_that_actually_answered():
    """Guards the 2026-09-02 incident where a server failed to bind on a port conflict and
    the PREVIOUS model answered, producing byte-identical 'new' numbers."""
    path, digest = serving_identity(
        "http://x:8081", opener=lambda _u: _Resp({"model_path": "/served/actual.gguf"})
    )
    assert path == "/served/actual.gguf"
    assert digest == ""  # not readable from this host; never fabricated


def test_serving_identity_refuses_rather_than_guessing():
    def boom(_u):
        raise OSError("connection refused")

    with pytest.raises(RuntimeError, match="Refusing to record"):
        serving_identity("http://x:8081", opener=boom)


def test_comparison_key_ignores_the_model_but_not_the_conditions():
    """The model is what an A/B varies; everything else must be held fixed."""
    a = _prov()
    assert _prov(model_path="/m/b.gguf", model_sha256="bbb", label="b",
                 checkpoint="939", epoch="last").comparison_key() == a.comparison_key()
    assert _prov(protocol="edit").comparison_key() != a.comparison_key()
    assert _prov(generation={"cache_prompt": True}).comparison_key() != a.comparison_key()


def test_a_regenerated_corpus_breaks_comparability():
    """The model card recorded `v5` at 5/27 while the same checkpoint measured 3/27 here;
    the probe corpus had been regenerated at the same path with the same file count."""
    a = _prov()
    b = _prov(corpora={"probe": CorpusFingerprint("data/probe", 27, "0ther")})
    assert a.comparison_key() != b.comparison_key()


def test_corpus_fingerprint_is_content_not_path(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "m1.txt").write_text("S1: 甲\n", encoding="utf-8")
    first = CorpusFingerprint.of(d)
    (d / "m1.txt").write_text("S1: 乙\n", encoding="utf-8")
    assert CorpusFingerprint.of(d).content_sha256 != first.content_sha256
    assert CorpusFingerprint.of(d).n_files == 1


def test_corpus_fingerprint_covers_filenames():
    """Paired statistics join on meeting id, so a renamed document is a different corpus."""
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as t:
        d = pathlib.Path(t)
        (d / "a.txt").write_text("x", encoding="utf-8")
        one = CorpusFingerprint.of(d)
        (d / "a.txt").rename(d / "b.txt")
        assert CorpusFingerprint.of(d).content_sha256 != one.content_sha256


# --- scorecard --------------------------------------------------------------------

def test_compare_refuses_incomparable_scorecards_with_an_actionable_message():
    a = Scorecard(_prov()).add(Check("G3_rouge1", True, score=0.05))
    b = Scorecard(_prov(protocol="edit")).add(Check("G3_rouge1", False, score=0.01))
    with pytest.raises(IncomparableScorecards, match="protocol"):
        compare(a, b)


def test_compare_allows_a_pure_model_difference():
    a = Scorecard(_prov()).add(Check("probe", False, score=3.0, n=27))
    b = Scorecard(_prov(model_path="/m/b.gguf", model_sha256="bbb", label="b"))
    b.add(Check("probe", False, score=8.0, n=27))
    (d,) = compare(a, b)
    assert d.change == pytest.approx(5.0)


def test_a_check_missing_from_one_side_is_reported_not_dropped():
    """A checkpoint measured on fewer instruments than its predecessor is a difference
    worth seeing; silently dropping the row is how a regression hides."""
    a = Scorecard(_prov()).add(Check("churn_rate", None, score=0.0))
    b = Scorecard(_prov(model_path="/m/b.gguf", model_sha256="b", label="b"))
    (d,) = compare(a, b)
    assert d.b is None and d.verdict_b == "absent"


def test_withheld_is_not_a_pass():
    card = Scorecard(_prov())
    card.add(Check("G4_budget", None, "no device measurement", n=0))
    card.add(Check("G3_rouge1", True, score=0.05, n=40))
    assert card.get("G4_budget").verdict == "—"
    assert card.withheld and not card.failed


def test_deployment_mismatch_flags_the_cache_setting_that_hid_the_churn():
    """Every gate ran `cache_prompt: false`; the shipped demo ran the cache live. The
    checkpoint that churned in deployment passed everything measured without it."""
    card = Scorecard(_prov())
    assert card.deployment_mismatch({"cache_prompt": True}) == {"cache_prompt": (False, True)}
    assert card.deployment_mismatch({"cache_prompt": False}) == {}


def test_scorecard_roundtrips_to_json(tmp_path):
    card = Scorecard(_prov()).add(Check("probe", False, "8/27", 8.0, 27, "runs/x.json"))
    blob = json.loads(card.write(tmp_path / "s.json").read_text(encoding="utf-8"))
    assert blob["comparison_key"] == card.provenance.comparison_key()
    assert blob["checks"][0]["score"] == 8.0


# --- behaviour --------------------------------------------------------------------

class _Op:
    pass


class _AppliedOp:
    def __init__(self, applied=True, reason=None, note=None):
        self.op, self.applied, self.reason, self.note = _Op(), applied, reason, note


class _Outcome:
    def __init__(self, results):
        self.results = results

    @property
    def churn_points(self):
        return [r for r in self.results if r.note and "restates dropped" in r.note]

    @property
    def hedge_points(self):
        return [r for r in self.results if r.note and "unresolved polarity" in r.note]


class _Step:
    def __init__(self, outcome, is_nop=False):
        self.outcome, self.is_nop = outcome, is_nop


class _Prose:
    def __init__(self, chars, text=""):
        self.chars = chars
        #: Retention needs the actual prose, not just its length, to ask whether each
        #: recorded point was rendered. Defaults to empty so length-only tests are unchanged.
        self.text = text


class _Syn:
    skipped_empty_memory = False

    def __init__(self, chars, ungrounded=(), prose=""):
        self.prose = _Prose(chars or len(prose), prose)
        self.ungrounded_numbers = ungrounded


class _Point:
    def __init__(self, text=""):
        self.text = text


class _Entry:
    def __init__(self, point):
        self.point, self.reason, self.superseded_by = point, "kept", None


class _Memory:
    """Mirrors `Memory`'s v1.1 surface: a working set plus `synthesis_view()`.

    `journalled` lets a test place points OUTSIDE the working set, which is the case the
    behaviour metrics were blind to — `points` stopped meaning "everything recorded" when
    the journal landed.
    """

    def __init__(self, n, arc="", journalled=0, texts=None):
        texts = texts or []
        self.points = [_Point(texts[i] if i < len(texts) else "") for i in range(n)]
        self._retired = [_Point("") for _ in range(journalled)]
        self.arc = arc

    def synthesis_view(self):
        return [_Entry(p) for p in self._retired + self.points]


class _Trace:
    def __init__(self, steps, points, chars, ungrounded=(), arc="",
                 journalled=0, texts=None, prose=""):
        self.steps, self.failed_steps = steps, []
        self.memory = _Memory(points, arc, journalled=journalled, texts=texts)
        self.synthesis = _Syn(chars, ungrounded, prose=prose)


def _churning_trace() -> _Trace:
    """The shipped failure: 6 chunks, 1 surviving point, 4 churn events, frozen ARC."""
    steps = [_Step(_Outcome([_AppliedOp()]))]
    for i in range(5):
        steps.append(_Step(_Outcome([
            _AppliedOp(),
            _AppliedOp(applied=False, reason="arc unchanged"),
            _AppliedOp(note="restates dropped «…»" if i >= 1 else None),
        ])))
    return _Trace(steps, points=1, chars=553)


def test_behaviour_catches_the_churn_that_every_gate_missed():
    r = behaviour.from_trace("dram", _churning_trace())
    assert r.churn_events == 4
    assert r.arc_frozen_steps == 5
    assert r.points == 1 and r.chunks == 6
    assert r.starved and r.confabulating
    assert r.chars_per_point == pytest.approx(553.0)


def test_behaviour_passes_the_healthy_run_on_the_same_meeting():
    """`v5` on that transcript: 4 points, 0 churn, 304 chars — must produce NO flags, or
    the instrument is just a length detector with extra steps."""
    steps = [_Step(_Outcome([_AppliedOp(), _AppliedOp()])) for _ in range(6)]
    r = behaviour.from_trace("dram", _Trace(steps, points=4, chars=304))
    assert r.flags == ()
    assert not r.starved and not r.confabulating


def test_empty_memory_with_prose_is_infinite_not_zero():
    """Synthesising from nothing is the extreme of confabulation, not a clean run."""
    r = behaviour.from_trace("x", _Trace([_Step(_Outcome([]))], points=0, chars=400))
    assert r.chars_per_point == float("inf") and r.confabulating


def test_summary_clean_meetings_cannot_be_satisfied_by_writing_more():
    good = behaviour.from_trace("g", _Trace(
        [_Step(_Outcome([_AppliedOp()])) for _ in range(6)], points=4, chars=304))
    bad = behaviour.from_trace("b", _churning_trace())
    s = behaviour.summarise([good, bad])
    assert s.clean_meetings == 1
    assert s.meetings_with_churn == 1 and s.total_churn == 4


# --- grounding --------------------------------------------------------------------

def test_grounding_catches_cjk_numerals_that_ungrounded_numbers_cannot():
    """`prose.ungrounded_numbers` states it is blind to CJK numerals, and the same
    investigation found fabricated 二零一六年六月三十日 and 十五萬美元.

    The flagged token is the NUMERAL (`十五萬`), not the numeral-plus-unit: units are
    deliberately outside the claim token, matching the Arabic branch. See
    `test_grounding_unit_change_alone_is_a_known_false_negative` for what that costs."""
    r = grounding.check("m", "決議金額為十五萬美元。", "S1: 會議討論預算。")
    assert r.ungrounded == ("十五萬",)


def test_grounding_unit_change_alone_is_a_known_false_negative():
    """Stated rather than hidden: a numeral present in the source under a DIFFERENT unit
    grounds successfully. Widening the token to include units would trade this for a
    larger class of false positives, and this instrument is deliberately the cheap floor
    under the judge, not a replacement for it."""
    r = grounding.check("m", "共十五萬美元", "S1: 出席十五萬人")
    assert r.ungrounded == ()


def test_grounding_accepts_a_figure_present_in_the_source():
    r = grounding.check("m", "總預算 12 億元。", "S1: 通過總預算 12 億元的案子。")
    assert r.ungrounded == ()
    assert r.n_checked >= 1


def test_grounding_folds_width_and_separators_rather_than_crying_wolf():
    r = grounding.check("m", "共 12000 元", "S1: 共 １２,０００ 元")
    assert r.ungrounded == ()


def test_grounding_rate_needs_its_denominator():
    """An empty summary scores a perfect 0.0 and is not thereby faithful. `n_checked`
    is what stops the rate rewarding abstention, the same failure a length-based
    curation metric had in the other direction."""
    r = grounding.check("m", "本次會議沒有具體決議。", "S1: 任何內容。")
    assert r.ungrounded_rate == 0.0 and r.n_checked == 0


def test_grounding_deduplicates_so_verbosity_cannot_dominate():
    once = grounding.check("m", "金額 999 元。", "S1: 無。")
    thrice = grounding.check("m", "金額 999 元，999 元，999 元。", "S1: 無。")
    assert once.n_ungrounded == thrice.n_ungrounded == 1


def test_grounding_summary_aggregates_meetings_not_claims():
    a = grounding.check("a", "共 25 件", "S1: 共 25 件")
    b = grounding.check("b", "共 37 件", "S1: 無數字")
    s = grounding.summarise([a, b])
    assert s.n_meetings == 2 and s.meetings_with_any == 1 and s.total_ungrounded == 1


def test_provenance_differences_names_the_field_not_just_the_hash():
    d = _prov().differences(_prov(protocol="edit"))
    assert "protocol" in d and d["protocol"] == ("tool", "edit")
    assert replace(_prov(), label="z").comparison_key() == _prov().comparison_key()


def test_abstention_is_not_confabulation():
    """`ivod-17673` is genuinely noisy ASR where NOP is the correct answer, and the harness
    sets `skipped_empty_memory` rather than calling the model. The first version of this
    module scored that as infinite confabulation — penalising the empty-memory guard for
    working. `Synthesis` carries the flag precisely so scoring can tell the two apart."""
    class _AbstainedSyn(_Syn):
        skipped_empty_memory = True

    t = _Trace([_Step(_Outcome([]))], points=0, chars=20)
    t.synthesis = _AbstainedSyn(20)
    t.synthesis.skipped_empty_memory = True
    r = behaviour.from_trace("ivod-17673", t)
    assert r.abstained
    assert not r.confabulating
    assert r.chars_per_point == 0.0
    assert "abstained (empty memory)" in r.flags


def test_empty_memory_WITHOUT_abstention_is_still_confabulation():
    """The guard must not become a blanket excuse: prose generated from an empty memory,
    where the model WAS called, is the extreme of the failure."""
    t = _Trace([_Step(_Outcome([]))], points=0, chars=400)
    r = behaviour.from_trace("x", t)
    assert not r.abstained and r.confabulating


def test_arc_counts_as_a_memory_unit():
    """A run can legitimately set a real ARC and zero POINTS (`ivod-17704`: a genuine
    173-character summary from the arc alone). Dividing prose by POINTS alone reported that
    as infinite confabulation."""
    t = _Trace([_Step(_Outcome([]))] * 2, points=0, chars=173, arc="會議審議年度預算案。")
    r = behaviour.from_trace("ivod-17704", t)
    assert r.has_arc and r.memory_units == 1
    assert r.chars_per_point == pytest.approx(173.0)
    assert not r.confabulating


def test_prose_from_a_truly_empty_memory_is_still_infinite():
    """No arc, no points, prose generated anyway — the guard must not swallow this."""
    r = behaviour.from_trace("x", _Trace([_Step(_Outcome([]))], points=0, chars=400))
    assert r.memory_units == 0 and r.chars_per_point == float("inf")


def test_single_digits_are_not_claims():
    """Ordinals and enumeration markers (`第 2 項`, `1.`) dominated the flagged set when
    single digits counted, and a literal containment test on them says nothing about
    faithfulness. Measured on the training pool before the rule was tightened."""
    r = grounding.check("m", "第 2 項與第 3 項均通過。", "S1: 全部通過。")
    assert r.n_checked == 0 and r.ungrounded == ()


def test_two_digit_fabrication_is_still_caught():
    r = grounding.check("m", "共 47 件", "S1: 沒有提到數量。")
    assert r.ungrounded == ("47",)


# --- numeral-system folding ---------------------------------------------------------
# Pins the fix for a false-positive class this module previously documented as accepted.
# It fired systematically, not occasionally: the corpus writes figures in Arabic and fluent
# zh-TW output writes them in CJK, so it penalised exactly the well-written summaries.
# Measured cost when found: 3 of 6 faithful teacher outputs rejected while building journal
# synthesis supervision.

def test_cjk_to_int_reads_multiplicative_and_positional_forms():
    assert grounding.cjk_to_int("六十") == 60
    assert grounding.cjk_to_int("十二") == 12
    assert grounding.cjk_to_int("九十萬") == 900000
    assert grounding.cjk_to_int("兩百萬") == 2000000
    assert grounding.cjk_to_int("一百二十") == 120
    # Years are written positionally, and this is how the corpus writes them.
    assert grounding.cjk_to_int("二零一六") == 2016
    assert grounding.cjk_to_int("台北") is None


def test_a_cjk_figure_grounds_against_an_arabic_source():
    r = grounding.check("m", "租約在到期前六十天生效，十二個月內不得調漲。",
                        "現約到期前60天生效；限制12個月內無理由驅逐次數。")
    assert r.ungrounded == (), f"faithful CJK rendering flagged: {r.ungrounded}"
    assert r.n_checked >= 2, "the claims must still be CHECKED, not skipped"


def test_the_fold_works_in_the_other_direction_too():
    """An Arabic claim against a CJK source — the same defect mirrored."""
    r = grounding.check("m", "補助金額為 2000000 元。", "補助金額為兩百萬元。")
    assert r.ungrounded == ()


def test_mixed_myriad_form_grounds():
    """`900000` is written `90萬` in practice, and containment does not see through that."""
    r = grounding.check("m", "預算增加九十萬美元。", "預算增加90萬美元。")
    assert r.ungrounded == ()


def test_the_fold_does_not_ground_a_genuinely_absent_figure():
    """Negative control. A more permissive test is only safe if it still fails."""
    r = grounding.check("m", "預算增加九十萬美元。", "預算增加13萬美元。")
    assert r.n_ungrounded == 1


def test_every_character_cjk_to_int_reads_is_also_a_claim_character():
    """The detector and the parser must agree on where a numeral starts.

    They did not: `兩` was absent from `CJK_NUMBER`, so `兩百萬` matched as `百萬` and was
    valued at 1,000,000 — half its real value, compared silently against the wrong figure.
    """
    known = set(grounding._CJK_DIGIT) | set(grounding._CJK_UNIT) | set(grounding._CJK_BIG)
    missing = {c for c in known if not grounding.CJK_NUMBER.fullmatch(c * 2)}
    assert not missing, f"parseable but undetectable: {sorted(missing)}"


# --- the journal must be visible to the behaviour metrics --------------------------
# `points` stopped meaning "everything recorded" when SPEC §4.1 v1.1 landed, and this module
# was not updated with it. Three metrics were wrong at once, all in the direction of flattering
# the model: starvation over-fired, under-rendering under-fired, and G5 had no numerator.

def test_recorded_points_include_the_journal_not_just_survivors():
    tr = _Trace(steps=[], points=16, chars=400, journalled=24)
    rep = behaviour.from_trace("m", tr)
    assert rep.points == 16, "the working set is still reported as itself"
    assert rep.recorded_points == 40, "everything recorded must be counted"


def test_a_long_meeting_that_retired_points_is_not_called_starved():
    """40 points over 40 chunks is healthy accumulation, and the journal holds 24 of them.

    Counting only the 16 survivors gives 0.4/chunk — BELOW the 0.5 floor — so the meeting
    was reported as "memory did not accumulate" when it had in fact recorded the most of
    any meeting in the run. That is the false flag, in the direction that hides good work."""
    steps = [_Step(_Outcome([])) for _ in range(40)]
    survivors_only = _Trace(steps=steps, points=16, chars=900)
    assert survivors_only.memory.synthesis_view().__len__() == 16
    assert behaviour.from_trace("m", survivors_only).starved, "precondition: the old reading"

    with_journal = _Trace(steps=steps, points=16, chars=900, journalled=24)
    rep = behaviour.from_trace("m", with_journal)
    assert rep.recorded_points == 40
    assert rep.points_per_chunk == 1.0
    assert not rep.starved


def test_under_rendering_is_measured_against_everything_recorded():
    """The v1.1 deficit in one assertion: ~40 entries recorded, 346 characters emitted.
    Against the 16 survivors that is 21.6 ch/pt and passes; against all 40 it is 8.7 and
    fails, which is the honest reading."""
    tr = _Trace(steps=[], points=16, chars=346, journalled=24)
    rep = behaviour.from_trace("m", tr)
    assert rep.chars_per_point < behaviour.UNDER_RENDERING_CHARS_PER_POINT
    assert rep.under_rendering


def test_retention_counts_points_the_prose_actually_renders():
    """G5. The journal makes SURVIVAL to synthesis automatic, which moves the failure one
    step later: reaching the prompt is not reaching the prose."""
    texts = ["同意搬到 B 棟大樓", "核准增設休閒助理職位", "延後表決公車路線調整案"]
    tr = _Trace(steps=[], points=3, chars=0, texts=texts,
                prose="會議同意搬到 B 棟大樓，並核准增設休閒助理職位。")
    rep = behaviour.from_trace("m", tr)
    assert rep.recorded_points == 3
    assert rep.rendered_points == 2, "the third point never reaches the prose"
    assert abs(rep.retention - 2 / 3) < 1e-9


# --- Simplified internals (arcsum/simplified.py) ------------------------------------

def test_conversion_degrades_to_identity_without_the_extra(monkeypatch):
    """The suite must run with NO optional extra installed, so importing the harness can
    never depend on opencc being present."""
    import builtins

    from arcsum import simplified
    real = builtins.__import__

    def no_opencc(name, *a, **k):
        if name == "opencc":
            raise ImportError("not installed")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_opencc)
    assert simplified.converter(simplified.TO_SIMPLIFIED)("會議") == "會議"
    assert simplified.available() is False


# --- RL reward (SPEC 5.2.2): hard constraints x a bounded objective ------------------

def test_reward_is_retention_when_every_constraint_holds():
    from arcsum.rl import score
    entries = ["同意搬到 B 棟大樓", "核准增設休閒助理職位", "延後表決公車路線調整案"]
    view = "MEMORY:\nARC: -\nPOINTS:\n" + "\n".join("- " + e for e in entries) + "\n\n"
    good = "會議同意搬到 B 棟大樓，核准增設休閒助理職位，並延後表決公車路線調整案。"
    r = score(good, entries, view, max_tokens=1000)
    assert r.ok and r.rendered == 3 and r.reward == 1.0


def test_an_ungrounded_specific_zeroes_the_reward_rather_than_costing_a_penalty():
    """Faithfulness is a CONSTRAINT, not a weighted term. A penalty invites the policy to buy
    fabrications with coverage — the exact trade the `九十萬` incident shows a model will take
    when coverage is rewarded."""
    from arcsum.rl import score
    entries = ["撥30萬美元採納資本改善計畫"]
    view = "MEMORY:\nARC: -\nPOINTS:\n- 撥30萬美元採納資本改善計畫\n\n"
    r = score("會議撥九十萬美元採納資本改善計畫。", entries, view, max_tokens=1000)
    assert r.reward == 0.0 and r.refused == "ungrounded specific"


def test_leaked_reasoning_markup_zeroes_the_reward():
    """`<think>` reached the shipped demo and was 54-77% of v16's flagged tokens."""
    from arcsum.rl import score
    entries = ["同意搬到 B 棟大樓，並核准增設休閒助理職位"]
    view = "MEMORY:\nARC: -\nPOINTS:\n- 同意搬到 B 棟大樓，並核准增設休閒助理職位\n\n"
    leaked = "<think>盤點</think>會議同意搬到 B 棟大樓，並核准增設休閒助理職位。"
    assert score(leaked, entries, view, max_tokens=1000).refused == "markup leak"


def test_padding_cannot_buy_reward_but_over_budget_is_refused():
    """Length only GATES. Rewarding it directly would produce padding, so a long summary
    scores exactly what it renders — and one over SPEC §3's cap scores nothing."""
    from arcsum.rl import score
    entries = ["同意搬到 B 棟大樓", "核准增設休閒助理職位"]
    view = "MEMORY:\nARC: -\nPOINTS:\n" + "\n".join("- " + e for e in entries) + "\n\n"
    padded = "會議同意搬到 B 棟大樓，核准增設休閒助理職位。" + "此外會議持續進行相關討論。" * 6
    assert score(padded, entries, view, max_tokens=1000).reward == 1.0
    assert score(padded, entries, view, max_tokens=40).refused == "over budget"


def test_a_collapsed_summary_is_refused_not_scored_near_zero():
    from arcsum.rl import score
    entries = [f"第{i}項決議通過" for i in range(12)]
    view = "MEMORY:\nARC: -\nPOINTS:\n" + "\n".join("- " + e for e in entries) + "\n\n"
    assert score("會議結束。", entries, view, max_tokens=1000).refused == "collapsed"


# --- RAFT step reward: a RANKING over candidate edits, not a gate ---------------------

def _outcome(ops, memory):
    from arcsum.chunker import Chunk
    from arcsum.guards import apply_ops
    from arcsum.transcript import Utterance
    return apply_ops(memory, ops, Chunk(index=1, utterances=[Utterance("S1", "x")], tokens=10),
                     lang_check=False)


def test_churn_is_priced_worse_than_an_equivalent_refusal():
    """Churn SUCCEEDS — it enters memory, takes a slot, and inflates retention (SPEC §5.2.2).
    A defect that improves the headline metric must cost more than one that merely wastes
    decode tokens, or rejection sampling will happily select for it."""
    from arcsum.memory import Memory
    from arcsum.ops import Add, Drop
    from arcsum.rl.step_reward import score_step

    m = Memory(token_len=len)
    m.add_point("同意搬到 B 棟大樓", chunk=0)
    churny = [Drop(pid=1), Add("同意搬到 B 棟大樓")]
    out = _outcome(churny, m.clone())
    s = score_step(out, churny)
    assert s.churn >= 1
    assert s.score < 0, "a churning step must not out-score doing nothing useful"


def test_a_clean_add_outscores_a_churning_one():
    from arcsum.memory import Memory
    from arcsum.ops import Add, Drop
    from arcsum.rl.step_reward import score_step

    m = Memory(token_len=len)
    m.add_point("同意搬到 B 棟大樓", chunk=0)
    clean = [Add("核准增設休閒助理職位")]
    churny = [Drop(pid=1), Add("同意搬到 B 棟大樓")]
    a = score_step(_outcome(clean, m.clone()), clean)
    b = score_step(_outcome(churny, m.clone()), churny)
    assert a.score > b.score


def test_revise_is_not_penalised_as_churn():
    """`revise` is the sanctioned form of DROP+ADD (SPEC §4.1 v1.1); pricing it as churn
    would train the model away from the op that exists to replace churn."""
    from arcsum.memory import Memory
    from arcsum.ops import Revise
    from arcsum.rl.step_reward import score_step

    m = Memory(token_len=len)
    m.add_point("公車路線調整案通過", chunk=0)
    ops = [Revise(1, "公車路線調整案改為取消")]
    s = score_step(_outcome(ops, m.clone()), ops)
    assert s.churn == 0 and s.revised == 1 and s.score > 0
