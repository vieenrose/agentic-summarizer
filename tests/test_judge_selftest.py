"""Pins the judge selftest (SPEC §5.1, §5.2 G3): planted-inversion recall, false-alarm
rate, and measured per-input noise — the mechanism that PRODUCES `tie_thresholds` for
`metrics.stats`, rather than an inherited constant.
"""

from __future__ import annotations

import pytest

from arcsum.judge.selftest import (
    FLIPS,
    FlipCase,
    build_flip_cases,
    measure_noise,
    run_selftest,
)


def test_flips_carries_the_zh_tw_pairs() -> None:
    """The exact pairs guards.py's polarity doctrine and the prior project's own
    judge_selftest.FLIPS both confirm."""
    assert ("通過", "否決") in FLIPS
    assert ("核准", "駁回") in FLIPS
    assert ("同意", "拒絕") in FLIPS


def test_build_flip_cases_produces_one_case_per_flip_pair() -> None:
    cases = build_flip_cases()
    assert len(cases) == len(FLIPS)
    assert len({c.name for c in cases}) == len(cases)  # names are distinct


def test_flip_case_true_and_flipped_claims_share_the_same_subject() -> None:
    """The claim/evidence relationship must be unambiguous by construction: only the
    polarity word differs between true_claim and flipped_claim."""
    for case, (positive, _negative) in zip(build_flip_cases(), FLIPS, strict=False):
        suffix = f"已{positive}。"
        assert case.true_claim.endswith(suffix)
        subject = case.true_claim[: -len(suffix)]
        assert case.evidence_text.startswith(subject)
        assert case.flipped_claim.startswith(subject)


# --- run_selftest ------------------------------------------------------------------------


class _PerfectJudge:
    """Always correctly distinguishes SUPPORTED from CONTRADICTED by comparing the
    trailing polarity word of the claim against the evidence's own polarity word."""

    def __call__(self, system: str, user: str) -> str:
        # user = "逐字稿片段：\n- <evidence>\n\n摘要陳述：<claim>"
        evidence_part, claim_part = user.split("摘要陳述：")
        evidence_text = evidence_part.split("- ", 1)[1].strip()
        claim = claim_part.strip()
        evidence_word = evidence_text.rstrip("。")[-2:]
        claim_word = claim.rstrip("。")[-2:]
        return "SUPPORTED" if evidence_word == claim_word else "CONTRADICTED"


def test_run_selftest_a_perfect_judge_passes() -> None:
    report = run_selftest(_PerfectJudge())
    assert report.recall == 1.0
    assert report.false_alarm_rate == 0.0
    assert report.passed is True


class _AlwaysSupportedJudge:
    """The exact failure mode that disqualified gemma-3n-E4B-it in the prior
    project: answers SUPPORTED to every probe case regardless of content."""

    def __call__(self, system: str, user: str) -> str:
        return "SUPPORTED"


def test_run_selftest_an_always_supported_judge_fails_on_zero_recall() -> None:
    report = run_selftest(_AlwaysSupportedJudge())
    assert report.recall == 0.0
    assert report.passed is False


class _AlwaysContradictedJudge:
    """The opposite failure: a judge that cries wolf on every claim."""

    def __call__(self, system: str, user: str) -> str:
        return "CONTRADICTED"


def test_always_contradicted_judge_recall_ok_but_false_alarms_fail() -> None:
    report = run_selftest(_AlwaysContradictedJudge())
    assert report.recall == 1.0
    assert report.false_alarm_rate == 1.0
    assert report.passed is False  # recall alone is not enough to pass


def test_run_selftest_with_no_cases_does_not_pass() -> None:
    report = run_selftest(_PerfectJudge(), cases=())
    assert report.passed is False


def test_run_selftest_accepts_custom_cases() -> None:
    case = FlipCase(
        name="custom",
        evidence_text="議案通過。",
        true_claim="議案已通過。",
        flipped_claim="議案已否決。",
    )
    report = run_selftest(_PerfectJudge(), cases=(case,))
    assert report.recall == 1.0


# --- measure_noise -----------------------------------------------------------------------


class _DeterministicJudge:
    def __call__(self, system: str, user: str) -> str:
        return "SUPPORTED"


def test_measure_noise_is_zero_for_a_deterministic_judge() -> None:
    noise = measure_noise(_DeterministicJudge(), "議案已通過。", "議案通過。", repeats=5)
    assert noise == 0.0


class _FlakyJudge:
    """Deterministically alternates verdicts -- simulates the prior project's
    measured finding of a local judge flipping on identical input."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        return "SUPPORTED" if self.calls % 2 == 0 else "UNSUPPORTED"


def test_measure_noise_detects_a_flaky_judge() -> None:
    """5 calls alternating S/U/S/U/S: majority SUPPORTED=3, disagreements=2 -> 0.4."""
    noise = measure_noise(_FlakyJudge(), "議案已通過。", "議案通過。", repeats=5)
    assert noise == pytest.approx(0.4)


def test_measure_noise_uses_the_identical_prompt_every_repeat() -> None:
    prompts_seen: list[str] = []

    def recording_call(system: str, user: str) -> str:
        prompts_seen.append(user)
        return "SUPPORTED"

    measure_noise(recording_call, "議案已通過。", "議案通過。", repeats=4)
    assert len(set(prompts_seen)) == 1  # every call saw the exact same prompt


def test_measure_noise_treats_a_missing_verdict_as_unsupported() -> None:
    """A judge response with no parseable verdict keyword must not crash the count --
    it is treated as UNSUPPORTED, consistent with judge_meeting's own fallback."""

    def garbled_call(system: str, user: str) -> str:
        return "I cannot determine this."

    noise = measure_noise(garbled_call, "議案已通過。", "議案通過。", repeats=3)
    assert noise == 0.0  # all 3 calls agree (all UNSUPPORTED-by-fallback) -> no noise
