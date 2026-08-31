"""Pins the paired-comparison statistical protocol and ship gates (SPEC §5.2).

Every test here corresponds to a specific decision the module's docstring justifies —
paired-only, sign test not t-test, per-metric tie bands, withhold below min_n, the
self-glob guard, inversions counted separately, ship-the-baseline-by-default.
"""

from __future__ import annotations

import pytest

from arcsum.metrics.stats import (
    WALL_CLOCK_CEILING_MINUTES,
    Comparison,
    compare,
    count_inversions,
    filter_scoreable_records,
    gate_g2_faithfulness,
    gate_g3_quality,
    gate_g4_budget,
    load_scores,
    ship_decision,
    sign_test,
)

# --- sign_test --------------------------------------------------------------------------


def test_sign_test_matches_a_known_figure() -> None:
    """14 wins / 2 losses -- the prior project's own quoted FAITH-claim result,
    reported there as p=0.004."""
    assert sign_test(14, 2) == pytest.approx(0.0041809, abs=1e-6)


def test_sign_test_is_one_at_zero_wins_and_losses() -> None:
    assert sign_test(0, 0) == 1.0


def test_sign_test_is_one_when_wins_equal_losses() -> None:
    assert sign_test(10, 10) == 1.0


def test_sign_test_approaches_zero_for_a_lopsided_result() -> None:
    assert sign_test(20, 0) < 0.001


def test_sign_test_is_symmetric_in_wins_and_losses() -> None:
    assert sign_test(3, 17) == sign_test(17, 3)


def test_sign_test_never_exceeds_one() -> None:
    for w in range(0, 5):
        for losses in range(0, 5):
            assert 0.0 <= sign_test(w, losses) <= 1.0


# --- Comparison ------------------------------------------------------------------------


def test_comparison_n_is_the_number_of_paired_deltas() -> None:
    c = Comparison("rouge1", wins=3, losses=1, ties=0, deltas=(0.1, 0.2, -0.05, 0.15))
    assert c.n == 4


def test_comparison_mean_delta() -> None:
    c = Comparison("rouge1", wins=2, losses=0, ties=0, deltas=(0.1, 0.3))
    assert c.mean_delta == pytest.approx(0.2)


def test_comparison_mean_delta_is_zero_with_no_data() -> None:
    c = Comparison("rouge1", wins=0, losses=0, ties=0, deltas=())
    assert c.mean_delta == 0.0
    assert c.stdev == 0.0
    assert c.stderr == 0.0


def test_comparison_stdev_requires_at_least_two_points() -> None:
    c = Comparison("rouge1", wins=1, losses=0, ties=0, deltas=(0.5,))
    assert c.stdev == 0.0  # statistics.stdev would raise on n=1; must not propagate


def test_comparison_stderr_is_stdev_over_sqrt_n() -> None:
    c = Comparison("rouge1", wins=2, losses=0, ties=0, deltas=(0.0, 0.2))
    assert c.stderr == pytest.approx(c.stdev / (2**0.5))


def test_comparison_p_value_uses_sign_test() -> None:
    c = Comparison("rouge1", wins=14, losses=2, ties=4, deltas=())
    assert c.p_value == pytest.approx(sign_test(14, 2))


def test_comparison_line_is_a_readable_summary() -> None:
    c = Comparison("rouge1", wins=2, losses=0, ties=0, deltas=(0.1, 0.2))
    line = c.line()
    assert "rouge1" in line
    assert "n=2" in line
    assert "wins=2" in line


# --- the self-glob guard ---------------------------------------------------------------


def test_filter_scoreable_records_keeps_records_with_both_identity_fields() -> None:
    records = [{"meeting_id": "m1", "system": "agent", "rouge1": 0.5}]
    assert filter_scoreable_records(records) == records


def test_filter_scoreable_records_drops_records_missing_meeting_id() -> None:
    records = [{"system": "agent", "rouge1": 0.5}]
    assert filter_scoreable_records(records) == []


def test_filter_scoreable_records_drops_records_missing_system() -> None:
    records = [{"meeting_id": "m1", "rouge1": 0.5}]
    assert filter_scoreable_records(records) == []


def test_filter_scoreable_records_rejects_the_tools_own_report_output() -> None:
    """report.json's own output (comparisons/gates/ship decision) has neither field --
    this is exactly what must not be globbed back in as input on the next run."""
    own_output = {"comparisons": [], "gates": [], "ship_decision": "ship the baseline"}
    assert filter_scoreable_records([own_output]) == []


def test_filter_scoreable_records_ignores_non_dict_entries() -> None:
    assert filter_scoreable_records(["not a dict", 42, None]) == []


def test_load_scores_indexes_by_meeting_then_system() -> None:
    records = [
        {"meeting_id": "m1", "system": "agent", "rouge1": 0.6},
        {"meeting_id": "m1", "system": "baseline", "rouge1": 0.5},
        {"meeting_id": "m2", "system": "agent", "rouge1": 0.7},
    ]
    scores = load_scores(records)
    assert scores["m1"]["agent"]["rouge1"] == 0.6
    assert scores["m1"]["baseline"]["rouge1"] == 0.5
    assert scores["m2"]["agent"]["rouge1"] == 0.7
    assert "baseline" not in scores["m2"]


def test_load_scores_drops_the_tools_own_output_from_the_pool() -> None:
    records = [
        {"meeting_id": "m1", "system": "agent", "rouge1": 0.6},
        {"comparisons": [], "ship_decision": "ship the agent"},
    ]
    scores = load_scores(records)
    assert list(scores.keys()) == ["m1"]


# --- compare: paired-only, per-metric tie bands ---------------------------------------


def _scores(pairs: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    return pairs


def test_compare_only_uses_meetings_present_in_both_systems() -> None:
    scores = _scores(
        {
            "m1": {"agent": {"rouge1": 0.6}, "baseline": {"rouge1": 0.5}},
            "m2": {"agent": {"rouge1": 0.9}},  # unpaired -- baseline missing entirely
        }
    )
    comparisons, paired = compare(
        scores, "agent", "baseline", tie_thresholds={}, metrics=("rouge1",)
    )
    assert paired == 1
    assert comparisons[0].n == 1
    assert comparisons[0].deltas == (pytest.approx(0.1),)


def test_compare_unpaired_meetings_are_dropped_not_averaged_in() -> None:
    """An unpaired meeting must not silently contribute a delta of 0 or be skipped in
    a way that still inflates n -- it must not appear in the comparison at all."""
    scores = _scores(
        {
            "m1": {"agent": {"rouge1": 0.6}, "baseline": {"rouge1": 0.5}},
            "m2": {"agent": {"rouge1": 1.0}},
            "m3": {"baseline": {"rouge1": 1.0}},
        }
    )
    comparisons, paired = compare(
        scores, "agent", "baseline", tie_thresholds={}, metrics=("rouge1",)
    )
    assert paired == 1
    assert comparisons[0].n == 1


def test_compare_drops_a_pair_missing_the_specific_metric() -> None:
    scores = _scores(
        {
            "m1": {"agent": {"rouge1": 0.6}, "baseline": {"rouge1": 0.5}},
            "m2": {"agent": {"rouge2": 0.4}, "baseline": {"rouge2": 0.3}},  # no rouge1 here
        }
    )
    comparisons, _ = compare(scores, "agent", "baseline", tie_thresholds={}, metrics=("rouge1",))
    assert comparisons[0].n == 1  # only m1 has rouge1 under both systems


def test_compare_classifies_wins_losses_and_ties_by_threshold() -> None:
    scores = _scores(
        {
            "m1": {"agent": {"m": 0.60}, "baseline": {"m": 0.50}},  # delta=+0.10 -> win
            "m2": {"agent": {"m": 0.40}, "baseline": {"m": 0.50}},  # delta=-0.10 -> loss
            "m3": {"agent": {"m": 0.501}, "baseline": {"m": 0.50}},  # delta=+0.001 -> tie
        }
    )
    comparisons, _ = compare(
        scores, "agent", "baseline", tie_thresholds={"m": 0.05}, metrics=("m",)
    )
    c = comparisons[0]
    assert (c.wins, c.losses, c.ties) == (1, 1, 1)


def test_compare_tie_threshold_is_per_metric() -> None:
    """A tie band meaningful on a 1-5 scale is meaningless on a 0-1 ROUGE-L -- the
    threshold must be looked up per metric, not applied as one global float."""
    scores = _scores(
        {
            "m1": {
                "agent": {"tight": 0.02, "loose": 0.02},
                "baseline": {"tight": 0.0, "loose": 0.0},
            },
        }
    )
    comparisons, _ = compare(
        scores,
        "agent",
        "baseline",
        tie_thresholds={"tight": 0.01, "loose": 0.5},
        metrics=("tight", "loose"),
    )
    by_metric = {c.metric: c for c in comparisons}
    assert by_metric["tight"].wins == 1  # 0.02 > 0.01 threshold -> a win
    assert by_metric["loose"].ties == 1  # 0.02 < 0.5 threshold -> a tie


def test_compare_missing_tie_threshold_defaults_to_zero() -> None:
    scores = _scores({"m1": {"agent": {"m": 0.01}, "baseline": {"m": 0.0}}})
    comparisons, _ = compare(scores, "agent", "baseline", tie_thresholds={}, metrics=("m",))
    assert comparisons[0].wins == 1  # any positive delta wins with an unset (0.0) threshold


# --- count_inversions ------------------------------------------------------------------


def test_count_inversions_sums_across_meetings() -> None:
    scores = _scores(
        {
            "m1": {"agent": {"inversions": 2}},
            "m2": {"agent": {"inversions": 0}},
            "m3": {"agent": {"inversions": 1}},
        }
    )
    assert count_inversions(scores, "agent") == 3


def test_count_inversions_ignores_meetings_missing_the_system() -> None:
    scores = _scores({"m1": {"baseline": {"inversions": 5}}})
    assert count_inversions(scores, "agent") == 0


def test_count_inversions_is_not_folded_into_a_mean() -> None:
    """A separate, plain count -- SPEC §5.1: one inverted decision is a product
    defect, not a fractional penalty."""
    scores = _scores({"m1": {"agent": {"inversions": 1}}, "m2": {"agent": {"inversions": 1}}})
    assert count_inversions(scores, "agent") == 2  # not 1.0 (an averaged "0.5 each")


# --- gates ---------------------------------------------------------------------------


def test_gate_g2_passes_when_treatment_has_no_more_inversions() -> None:
    assert gate_g2_faithfulness(2, 3).passed is True
    assert gate_g2_faithfulness(3, 3).passed is True
    assert gate_g2_faithfulness(4, 3).passed is False


def test_gate_g3_withholds_below_min_n() -> None:
    c = Comparison("rouge1", wins=5, losses=0, ties=0, deltas=(0.1,) * 5)
    results = gate_g3_quality([c], min_n=20)
    assert results[0].passed is None
    assert "withheld" in results[0].detail


def test_gate_g3_uses_the_lower_one_se_bound_not_the_raw_mean() -> None:
    """A mean_delta that is barely positive but with a wide SE must not pass --
    the LOWER bound of mean_delta - SE must clear zero."""
    small_positive_wide_se = Comparison(
        "rouge1", wins=11, losses=9, ties=0, deltas=(0.5, -0.5) * 10 + (0.01,)
    )
    result = gate_g3_quality([small_positive_wide_se], min_n=20)[0]
    assert result.passed is False


def test_gate_g3_passes_when_the_lower_bound_clears_zero() -> None:
    consistently_positive = Comparison("rouge1", wins=20, losses=0, ties=0, deltas=(0.1,) * 20)
    result = gate_g3_quality([consistently_positive], min_n=20)[0]
    assert result.passed is True


def test_gate_g4_passes_under_the_ceiling() -> None:
    assert gate_g4_budget(12.9).passed is True


def test_gate_g4_fails_over_the_ceiling() -> None:
    assert gate_g4_budget(22.5).passed is False


def test_gate_g4_withholds_with_no_measurement() -> None:
    assert gate_g4_budget(None).passed is None


def test_gate_g4_ceiling_matches_spec() -> None:
    assert WALL_CLOCK_CEILING_MINUTES == 20.0


# --- ship_decision -----------------------------------------------------------------------


def test_ship_decision_defaults_to_the_baseline_on_any_failure() -> None:
    gates = [gate_g2_faithfulness(5, 3)]  # fails: more inversions than baseline
    assert ship_decision(gates, g1_passed=True) == "ship the baseline"


def test_ship_decision_defaults_to_the_baseline_when_g1_fails() -> None:
    gates = [gate_g2_faithfulness(2, 3)]  # G2 passes, but G1 (revision probe) failed
    assert ship_decision(gates, g1_passed=False) == "ship the baseline"


def test_ship_decision_defaults_to_the_baseline_on_a_withheld_gate() -> None:
    """A withheld gate must not be treated as a pass -- insufficient evidence ships
    the safe default, not the untested treatment."""
    gates = [gate_g4_budget(None)]
    assert ship_decision(gates, g1_passed=True) == "ship the baseline"


def test_ship_decision_ships_the_agent_only_when_everything_clears() -> None:
    gates = [gate_g2_faithfulness(2, 3), gate_g4_budget(12.9)]
    assert ship_decision(gates, g1_passed=True) == "ship the agent"


# --- gates must not pass on absent or noisy evidence -----------------------------------


def test_g2_withholds_when_no_judge_records_exist() -> None:
    """The bug this pins: with no judge run, both inversion sums are 0 and `0 <= 0`
    reported a clean PASS for a gate nothing had measured -- observed in a real ship
    report 2026-08-27. A gate that passes on absent evidence manufactures confidence.
    """
    gate = gate_g2_faithfulness(0, 0, judged_records=0)
    assert gate.passed is None
    assert "withheld" in gate.detail


def test_g2_still_passes_when_records_exist_and_inversions_are_fewer() -> None:
    assert gate_g2_faithfulness(1, 3, judged_records=20).passed is True


def test_g2_fails_when_treatment_has_more_inversions() -> None:
    assert gate_g2_faithfulness(5, 3, judged_records=20).passed is False


def test_g2_without_a_record_count_keeps_the_old_behaviour() -> None:
    """`judged_records=None` is the documented opt-out for callers that know their
    denominator is non-empty."""
    assert gate_g2_faithfulness(0, 0).passed is True


def test_g3_fails_when_effect_size_clears_but_sign_test_does_not() -> None:
    """Measured 2026-08-27: rouge1 posted a positive lower bound on a 12/8 split with
    p=0.50 -- a coin flip that the SE bound alone marked PASS. Tight, consistent
    deltas keep SE small (lower bound clears) while wins/losses stay near even.
    """
    c = Comparison("rouge1", wins=12, losses=8, ties=0, deltas=(0.039,) * 20)
    (gate,) = gate_g3_quality([c], min_n=20)
    assert c.p_value > 0.05, "fixture must be sign-test-insignificant"
    assert c.mean_delta - c.stderr > 0, "fixture must clear the SE bound"
    assert gate.passed is False
    assert "sign test" in gate.detail


def test_g3_passes_when_both_conditions_clear() -> None:
    c = Comparison("rougeL", wins=18, losses=2, ties=0, deltas=(0.05,) * 20)
    (gate,) = gate_g3_quality([c], min_n=20)
    assert c.p_value <= 0.05
    assert gate.passed is True


def test_g3_fails_when_sign_test_clears_but_effect_size_does_not() -> None:
    """The mirror case: significant sign test, but the mean delta is negative."""
    c = Comparison("rouge1", wins=15, losses=5, ties=0, deltas=(-0.004,) * 20)
    (gate,) = gate_g3_quality([c], min_n=20)
    assert gate.passed is False


def test_g3_does_not_gate_on_coverage_or_density() -> None:
    """SPEC §5.2 defines G3 as "beats baseline on ROUGE/BERTScore"; SPEC §5's metric
    table classes Coverage/Density as token-overlap DIAGNOSTICS. The code gated them
    anyway, and the consequence was not cosmetic: both measure EXTRACTIVENESS, so
    requiring agent > baseline demanded the agent copy MORE verbatim than map-reduce --
    the opposite of SPEC §3's abstractive prose, and unreachable by construction.
    """
    comparisons = [
        Comparison("rouge1", wins=18, losses=2, ties=0, deltas=(0.05,) * 20),
        Comparison("coverage", wins=2, losses=18, ties=0, deltas=(-0.011,) * 20),
        Comparison("density", wins=3, losses=17, ties=0, deltas=(-0.79,) * 20),
    ]
    gates = gate_g3_quality(comparisons, min_n=20)
    assert [g.gate for g in gates] == ["G3_rouge1"]


def test_a_failing_diagnostic_cannot_block_the_ship_decision() -> None:
    """The whole point of the demotion: coverage/density are still computed, reported
    and compared, but a bad one must not be able to withhold or fail the ship decision.
    """
    comparisons = [
        Comparison("rouge1", wins=18, losses=2, ties=0, deltas=(0.05,) * 20),
        Comparison("rouge2", wins=19, losses=1, ties=0, deltas=(0.05,) * 20),
        Comparison("rougeL", wins=19, losses=1, ties=0, deltas=(0.06,) * 20),
        Comparison("coverage", wins=0, losses=20, ties=0, deltas=(-0.9,) * 20),
        Comparison("density", wins=0, losses=20, ties=0, deltas=(-9.0,) * 20),
    ]
    gates = gate_g3_quality(comparisons, min_n=20)
    assert all(g.passed is True for g in gates)
    assert ship_decision(gates, g1_passed=True) == "ship the agent"


def test_count_inversions_paired_ignores_meetings_the_other_arm_lacks() -> None:
    """Measured 2026-08-27: a judge run completed 20/20 agent cases but only 16/20
    baseline cases (baseline summaries are ~3x longer -> more claims -> more chances to
    exhaust the judge budget). An UNPAIRED sum read "8 vs 8, PASS" while the baseline
    was simply missing four meetings. `compare` has always been paired-only; this
    brings G2's input in line with G3's.
    """
    scores = {
        "m1": {"agent": {"inversions": 1}, "baseline": {"inversions": 5}},
        "m2": {"agent": {"inversions": 2}, "baseline": {"inversions": 6}},
        # judged for the agent only -- must contribute to NEITHER sum
        "m3": {"agent": {"inversions": 3}},
    }
    assert count_inversions(scores, "agent") == 6  # unpaired, unchanged default
    assert count_inversions(scores, "agent", paired_with="baseline") == 3
    assert count_inversions(scores, "baseline", paired_with="agent") == 11


def test_count_inversions_paired_is_symmetric_on_full_coverage() -> None:
    scores = {
        "m1": {"agent": {"inversions": 1}, "baseline": {"inversions": 2}},
        "m2": {"agent": {"inversions": 3}, "baseline": {"inversions": 4}},
    }
    assert count_inversions(scores, "agent", paired_with="baseline") == count_inversions(
        scores, "agent"
    )


def test_score_and_judge_records_for_one_pair_are_merged_not_overwritten() -> None:
    """Measured 2026-08-27: passing score files and judge files in one `arcsum-report`
    invocation -- the tool's own documented usage -- made the judge records replace the
    score records wholesale, so G2 printed PASS while every G3 metric silently went to
    n=0/WITHHELD. They are complementary views of one measurement.
    """
    records = [
        {"meeting_id": "m1", "system": "agent", "rouge1": 0.5, "rougeL": 0.4},
        {"meeting_id": "m1", "system": "agent", "inversions": 2, "claims": 7},
    ]
    scores = load_scores(records)

    merged = scores["m1"]["agent"]
    assert merged["rouge1"] == 0.5, "score fields must survive the judge record"
    assert merged["inversions"] == 2, "judge fields must be present too"
    assert merged["claims"] == 7


def test_later_record_still_wins_on_a_genuine_key_conflict() -> None:
    records = [
        {"meeting_id": "m1", "system": "agent", "rouge1": 0.1},
        {"meeting_id": "m1", "system": "agent", "rouge1": 0.9},
    ]
    assert load_scores(records)["m1"]["agent"]["rouge1"] == 0.9


def test_merging_does_not_mutate_the_caller_s_records() -> None:
    original = {"meeting_id": "m1", "system": "agent", "rouge1": 0.5}
    records = [original, {"meeting_id": "m1", "system": "agent", "inversions": 3}]
    load_scores(records)
    assert "inversions" not in original, "input records must not be mutated in place"


def test_count_inversions_pairing_requires_the_field_not_just_a_record() -> None:
    """A ROUGE-scored record for the other arm must NOT make a meeting look paired.

    Score records and judge records are merged into one index; the ROUGE pass emits a
    record for every meeting in both arms, so pairing on record EXISTENCE silently
    admits meetings the judge never scored for the opposing arm. Measured 2026-08-29:
    35 judged agent meetings vs 19 judged baseline meetings summed to "14 vs 11" and
    failed G2, where the 19 genuinely paired meetings give 8 vs 11.
    """
    scores = {
        "m1": {"agent": {"inversions": 2}, "baseline": {"inversions": 5}},
        # baseline present (from the ROUGE pass) but NEVER judged:
        "m2": {"agent": {"inversions": 9}, "baseline": {"rouge1": 0.4}},
    }

    assert count_inversions(scores, "agent", paired_with="baseline") == 2
    assert count_inversions(scores, "baseline", paired_with="agent") == 5
