"""The paired-comparison statistical protocol and ship gates (SPEC §5.2).

Every decision below is deliberate, not incidental — each is a decision the prior
project's own report tool made and recorded a reason for, and porting the reasoning
matters as much as porting the arithmetic:

- **Paired per meeting, unpaired dropped, never averaged in.** An unpaired mean would
  let a favourable meeting split decide the result.
- **Sign test, not a t-test.** An ordinal or noisy metric from n≈20 meetings does not
  justify assuming normality; the sign test only asks "how often did treatment win",
  which needs no distributional assumption.
- **A delta below a per-metric tie threshold is a tie, not a win or a loss.** The
  threshold should come from *measured* run-to-run noise (SPEC G3: "beats baseline...
  by more than run-to-run noise"), never an inherited constant — which is why
  `tie_thresholds` is a per-metric mapping the caller supplies, not a single float. A
  0.5 tie band is meaningful on a 1-5 judge scale and meaningless on a 0-1 ROUGE-L.
- **Below `min_n`, a gate's verdict is withheld, not dressed up as directional.**
- **SE-of-the-mean, not the per-meeting noise band, is the right yardstick for a gate
  stated as a mean difference.** Per-meeting noise might be ±0.5, but the mean over
  n=20 has SE = stdev/sqrt(n) ~= stdev/4.5 — comparing a mean against the per-meeting
  band would make an achievable gate look unreachable.
- **Inversions are a separate count, with their own condition, never folded into a
  mean** (SPEC §5.1: "a single inverted decision is a product defect, not a fractional
  score penalty").
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import comb, sqrt

#: SPEC §5's metric suite this module compares by default. Callers may pass their own
#: subset via `compare(..., metrics=...)`.
DEFAULT_METRICS: tuple[str, ...] = ("rouge1", "rouge2", "rougeL", "coverage", "density")

#: SPEC §7's kill criterion.
WALL_CLOCK_CEILING_MINUTES = 20.0


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign-test p-value under H0: P(win) = 0.5.

    Ties are excluded before calling this — they carry no directional information for
    a sign test. `wins == losses == 0` returns `1.0` (no evidence either way).
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    cdf = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * cdf)


@dataclass(frozen=True, slots=True)
class Comparison:
    """One metric's paired comparison across every meeting present in both systems."""

    metric: str
    wins: int
    losses: int
    ties: int
    #: Per-meeting (treatment - control) delta, PAIRED meetings only.
    deltas: tuple[float, ...] = field(default_factory=tuple)

    @property
    def n(self) -> int:
        return len(self.deltas)

    @property
    def mean_delta(self) -> float:
        return statistics.mean(self.deltas) if self.deltas else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.deltas) if len(self.deltas) > 1 else 0.0

    @property
    def stderr(self) -> float:
        return self.stdev / sqrt(self.n) if self.n > 0 else 0.0

    @property
    def p_value(self) -> float:
        return sign_test(self.wins, self.losses)

    def line(self) -> str:
        return (
            f"{self.metric}: n={self.n} wins={self.wins} losses={self.losses} "
            f"ties={self.ties} mean_delta={self.mean_delta:+.3f} "
            f"(SE={self.stderr:.3f}) p={self.p_value:.3f}"
        )


def _has_identity_fields(record: object) -> bool:
    """The self-glob guard: a record lacking `meeting_id`/`system` cannot be scored —
    notably this tool's own `report.json`, which lands in the same output directory
    and would otherwise be globbed back in as input on a subsequent run."""
    return (
        isinstance(record, dict)
        and record.get("meeting_id") is not None
        and record.get("system") is not None
    )


def filter_scoreable_records(records: Sequence[object]) -> list[dict]:
    """The self-glob guard, decoupled from file I/O so it is directly testable."""
    return [r for r in records if _has_identity_fields(r)]


def index_by_meeting_and_system(records: Sequence[dict]) -> dict[str, dict[str, dict]]:
    """`{meeting_id: {system: record}}`, built from already-filtered records. A later
    record for the same `(meeting_id, system)` pair overwrites an earlier one."""
    scores: dict[str, dict[str, dict]] = {}
    for record in records:
        scores.setdefault(record["meeting_id"], {})[record["system"]] = record
    return scores


def load_scores(raw_records: Sequence[object]) -> dict[str, dict[str, dict]]:
    """Filter and index a batch of raw JSON-decoded records in one step. Callers doing
    their own file I/O (globbing a directory of judged-score JSON files) decode each
    file first, concatenate the records, and pass the combined list here.
    """
    return index_by_meeting_and_system(filter_scoreable_records(raw_records))


def compare(
    scores: Mapping[str, Mapping[str, Mapping]],
    treatment: str,
    control: str,
    *,
    tie_thresholds: Mapping[str, float],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[list[Comparison], int]:
    """Paired comparison of `treatment` against `control`, one `Comparison` per metric.

    Returns `(comparisons, paired_count)`. A meeting missing either system, or missing
    a given metric's value under either system, is dropped from THAT metric's
    comparison — never imputed, never averaged in as a default.
    """
    paired_count = sum(
        1 for systems in scores.values() if treatment in systems and control in systems
    )

    comparisons: list[Comparison] = []
    for metric in metrics:
        threshold = tie_thresholds.get(metric, 0.0)
        deltas: list[float] = []
        wins = losses = ties = 0
        for systems in scores.values():
            if treatment not in systems or control not in systems:
                continue
            t_val = systems[treatment].get(metric)
            c_val = systems[control].get(metric)
            if t_val is None or c_val is None:
                continue
            delta = t_val - c_val
            deltas.append(delta)
            if delta > threshold:
                wins += 1
            elif delta < -threshold:
                losses += 1
            else:
                ties += 1
        comparisons.append(Comparison(metric, wins, losses, ties, tuple(deltas)))
    return comparisons, paired_count


def count_inversions(
    scores: Mapping[str, Mapping[str, Mapping]], system: str, *, field_name: str = "inversions"
) -> int:
    """Total inversion count for `system` across every meeting that has one. Kept as a
    plain sum — SPEC §5.1: inversions are counted, never folded into an average."""
    total = 0
    for systems in scores.values():
        record = systems.get(system)
        if record is None:
            continue
        total += record.get(field_name, 0) or 0
    return total


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    #: `True`/`False` = decided; `None` = withheld (below `min_n`, or no data to judge).
    passed: bool | None
    detail: str


def gate_g2_faithfulness(treatment_inversions: int, control_inversions: int) -> GateResult:
    """SPEC §5.2 G2: `inversions <= baseline`."""
    passed = treatment_inversions <= control_inversions
    detail = f"treatment={treatment_inversions} baseline={control_inversions}"
    return GateResult("G2_faithfulness", passed, detail)


def gate_g3_quality(comparisons: Sequence[Comparison], *, min_n: int = 20) -> list[GateResult]:
    """SPEC §5.2 G3: beats baseline "by more than run-to-run noise" — operationalised
    as the mean delta's lower 1-SE bound still being positive, per the architecture
    doctrine above. One result per metric in `comparisons`.
    """
    results = []
    for c in comparisons:
        if c.n < min_n:
            results.append(GateResult(f"G3_{c.metric}", None, f"withheld: n={c.n} < min_n={min_n}"))
            continue
        lower_bound = c.mean_delta - c.stderr
        results.append(
            GateResult(
                f"G3_{c.metric}",
                lower_bound > 0,
                f"mean_delta={c.mean_delta:+.3f} SE={c.stderr:.3f} lower_bound={lower_bound:+.3f}",
            )
        )
    return results


def gate_g4_budget(wall_clock_minutes: float | None) -> GateResult:
    """SPEC §7's kill criterion, as a gate: measured wall-clock per meeting must not
    exceed `WALL_CLOCK_CEILING_MINUTES`. `None` (no device measurement yet) withholds."""
    if wall_clock_minutes is None:
        return GateResult("G4_budget", None, "withheld: no device measurement provided")
    passed = wall_clock_minutes <= WALL_CLOCK_CEILING_MINUTES
    detail = f"{wall_clock_minutes:.1f} min (ceiling {WALL_CLOCK_CEILING_MINUTES:.0f})"
    return GateResult("G4_budget", passed, detail)


def ship_decision(gates: Sequence[GateResult], *, g1_passed: bool) -> str:
    """SPEC §5.2: "Ship the agent only if G1-G4 clear. Otherwise ship the map-reduce
    baseline and record the negative result" — a legitimate outcome, not a failure to
    report. A withheld gate (`passed is None`) does not clear — the default on
    insufficient evidence is to ship the baseline, not to assume success.
    """
    if not g1_passed:
        return "ship the baseline"
    if any(g.passed is not True for g in gates):
        return "ship the baseline"
    return "ship the agent"
