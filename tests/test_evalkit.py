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
    def __init__(self, chars):
        self.chars = chars


class _Syn:
    skipped_empty_memory = False

    def __init__(self, chars, ungrounded=()):
        self.prose, self.ungrounded_numbers = _Prose(chars), ungrounded


class _Memory:
    def __init__(self, n, arc=""):
        self.points = [object()] * n
        self.arc = arc


class _Trace:
    def __init__(self, steps, points, chars, ungrounded=(), arc=""):
        self.steps, self.failed_steps = steps, []
        self.memory, self.synthesis = _Memory(points, arc), _Syn(chars, ungrounded)


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
