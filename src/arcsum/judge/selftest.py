"""Judge selftest (SPEC §5.1, §5.2 G3): validates the INSTRUMENT before trusting its
numbers — planted-inversion recall, false-alarm rate, and measured per-input noise.

Two independent questions this answers:

1. **Is this judge good enough to use at all?** Plant a known polarity flip in a
   (claim, evidence) pair and check the judge correctly reports `CONTRADICTED`
   (recall) — and separately check it does NOT report `CONTRADICTED` for the
   unflipped, genuinely-true claim (false-alarm rate). A judge failing either belongs
   in `judge.client.DISQUALIFIED_EMPIRICAL`, not in the panel — this is the same
   mechanism the prior project used to disqualify `gemma-3n-E4B-it`, which answered
   SUPPORTED to every probe case regardless of content.
2. **How noisy is this judge's verdict, run to run, on IDENTICAL input?** This
   MEASURED value — not an inherited constant — is what SPEC G3's "more than
   run-to-run noise" tie threshold in `metrics.stats` should be set from. Even a
   `temperature=0.0` local judge was measured flipping its own verdict on identical
   input in the prior project (SUPPORTED/UNSUPPORTED/SUPPORTED on the same prompt) —
   `measure_noise` exists to catch exactly that, not to assume it away.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from arcsum.judge.evidence import Evidence
from arcsum.judge.faith import FAITH_SYS, faith_prompt, parse_verdict

#: The exact zh-TW polarity pairs `guards.py`'s `_POSITIVE`/`_NEGATIVE` doctrine
#: already confirms as a subject/polarity vocabulary; also the prior project's
#: `judge_selftest.FLIPS` set: 通過/否決, 核准/駁回, 同意/拒絕.
FLIPS: tuple[tuple[str, str], ...] = (
    ("通過", "否決"),
    ("核准", "駁回"),
    ("同意", "拒絕"),
)


@dataclass(frozen=True, slots=True)
class FlipCase:
    name: str
    evidence_text: str
    #: Restates the evidence's own polarity word — should be judged SUPPORTED.
    true_claim: str
    #: The identical claim with the polarity word flipped — should be judged
    #: CONTRADICTED against the same evidence.
    flipped_claim: str


def build_flip_cases() -> tuple[FlipCase, ...]:
    """One case per `FLIPS` pair, hand-built rather than generated so the
    evidence/claim relationship is unambiguous by construction."""
    subjects = ("搬遷案", "預算案", "提案")
    return tuple(
        FlipCase(
            name=f"{subject}_{positive}_vs_{negative}",
            evidence_text=f"{subject}{positive}。",
            true_claim=f"{subject}已{positive}。",
            flipped_claim=f"{subject}已{negative}。",
        )
        for (positive, negative), subject in zip(FLIPS, subjects, strict=False)
    )


@dataclass(frozen=True, slots=True)
class SelftestReport:
    #: Fraction of flip cases where the FLIPPED claim was correctly CONTRADICTED.
    recall: float
    #: Fraction of flip cases where the TRUE claim was WRONGLY judged CONTRADICTED —
    #: a judge that cries wolf on real decisions.
    false_alarm_rate: float
    #: 100% recall and 0% false alarms — the bar the prior project used to disqualify
    #: a judge on measured behaviour, not lineage.
    passed: bool


def run_selftest(
    call: Callable[[str, str], str], *, cases: Sequence[FlipCase] | None = None
) -> SelftestReport:
    """`call` is `(system, user) -> raw_response` — the same shape as every other
    model-calling interface in this package, so a `JudgeClient` bound to one model
    can be passed here directly via `functools.partial(client, model)`."""
    resolved = tuple(cases) if cases is not None else build_flip_cases()
    if not resolved:
        return SelftestReport(recall=0.0, false_alarm_rate=0.0, passed=False)

    hits = 0
    false_alarms = 0
    for case in resolved:
        evidence = [Evidence(line=0, text=case.evidence_text, score=1.0)]
        flipped = parse_verdict(call(FAITH_SYS, faith_prompt(case.flipped_claim, evidence)))
        true = parse_verdict(call(FAITH_SYS, faith_prompt(case.true_claim, evidence)))
        if flipped == "CONTRADICTED":
            hits += 1
        if true == "CONTRADICTED":
            false_alarms += 1

    recall = hits / len(resolved)
    false_alarm_rate = false_alarms / len(resolved)
    return SelftestReport(
        recall=recall,
        false_alarm_rate=false_alarm_rate,
        passed=recall == 1.0 and false_alarm_rate == 0.0,
    )


def measure_noise(
    call: Callable[[str, str], str], claim: str, evidence_text: str, *, repeats: int = 5
) -> float:
    """Run the IDENTICAL `(claim, evidence)` prompt `repeats` times and return the
    fraction of calls whose verdict disagrees with the majority verdict — the
    measured per-input noise a tie threshold should be set from. `0.0` for a
    genuinely deterministic judge; anything above that is real evidence of
    instability, not an assumption.
    """
    evidence = [Evidence(line=0, text=evidence_text, score=1.0)]
    prompt = faith_prompt(claim, evidence)
    verdicts = [parse_verdict(call(FAITH_SYS, prompt)) or "UNSUPPORTED" for _ in range(repeats)]
    counts = Counter(verdicts)
    majority_count = max(counts.values())
    return (repeats - majority_count) / repeats
