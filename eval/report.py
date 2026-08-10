#!/usr/bin/env python3
"""Paired comparison and ship gates (CLAUDE.md §7.3, §7.7).

The statistical protocol is the point of this file, and it is deliberately conservative:

* **Paired per meeting.** Same meetings, same judge, same prompts across systems. An unpaired
  mean would let a favourable meeting split decide the result.
* **Δ < tie_threshold is a tie.** §7.3 asserts judge noise of ±0.4–0.5 and a 0.5 tie band.
  `--tie-threshold` defaults to that but should be set from `judge_selftest.py`'s *measured*
  noise — an inherited constant is not evidence.
* **Sign test, not a t-test.** n = 20 on a 1–5 ordinal scale from a stochastic judge does not
  justify assuming normality. Win/loss/tie counts and an exact binomial p are honest here.
* **A reduced cell is directional only.** Below `--min-n` the verdict is withheld rather than
  dressed up.

    python eval/report.py runs/judged/*.json --usage runs/arms/usage.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Ship gates (§7.7). GT3 is the agency bet; GT2 also requires 0% inversions.
GATES = {"GT2_faith_claim": 0.3, "GT3_synth": 0.5}
GT4_PREFILL_MAX = 1.25
JUDGE_METRICS = ("faith_claim", "faith_anchor", "cover", "synth")


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial p under H0: P(win) = P(loss) = 1/2. Ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(2 * tail, 1.0)


@dataclass
class Comparison:
    metric: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    deltas: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.deltas is None:
            self.deltas = []

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def mean_delta(self) -> float | None:
        return sum(self.deltas) / len(self.deltas) if self.deltas else None

    @property
    def p_value(self) -> float:
        return sign_test(self.wins, self.losses)

    def line(self, tie_threshold: float, min_n: int) -> str:
        mean = self.mean_delta
        mean_s = "n/a" if mean is None else f"{mean:+.2f}"
        note = "" if self.n >= min_n else f"  DIRECTIONAL ONLY (n={self.n} < {min_n})"
        return (
            f"  {self.metric:<13} Δ {mean_s}  "
            f"W/L/T {self.wins}/{self.losses}/{self.ties}  "
            f"p={self.p_value:.3f}{note}"
        )


def load_scores(paths: list[Path]) -> dict[str, dict[str, dict]]:
    """{meeting_id: {system: score}} from judge.py output files."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else [data]
        for record in records:
            out[record["meeting_id"]][record["system"]] = record
    return out


def compare(
    scores: dict[str, dict[str, dict]],
    treatment: str,
    control: str,
    *,
    tie_threshold: float,
) -> tuple[dict[str, Comparison], list[str], dict[str, list[str]]]:
    comparisons = {m: Comparison(m) for m in JUDGE_METRICS}
    paired: list[str] = []
    inversions: dict[str, list[str]] = {treatment: [], control: []}

    for meeting_id, systems in sorted(scores.items()):
        if treatment not in systems or control not in systems:
            continue  # unpaired meetings are dropped, not averaged in
        paired.append(meeting_id)
        for system in (treatment, control):
            if systems[system].get("inverted"):
                inversions[system].append(meeting_id)
        for metric, comparison in comparisons.items():
            a, b = systems[treatment].get(metric), systems[control].get(metric)
            if a is None or b is None:
                continue
            delta = a - b
            comparison.deltas.append(delta)
            if delta > tie_threshold:
                comparison.wins += 1
            elif delta < -tie_threshold:
                comparison.losses += 1
            else:
                comparison.ties += 1
    return comparisons, paired, inversions


def gate_verdicts(
    comparisons: dict[str, Comparison],
    inversions: dict[str, list[str]],
    treatment: str,
    prefill_ratio: float | None,
    *,
    min_n: int,
) -> list[str]:
    out: list[str] = []

    faith = comparisons["faith_claim"]
    clean = not inversions.get(treatment)
    if faith.n < min_n:
        out.append(f"GT2 faith        : WITHHELD (n={faith.n} < {min_n})")
    else:
        delta = faith.mean_delta or 0.0
        ok = delta >= GATES["GT2_faith_claim"] and clean
        out.append(
            f"GT2 faith        : {'PASS' if ok else 'FAIL'} "
            f"(Δ {delta:+.2f} vs +{GATES['GT2_faith_claim']}, "
            f"inversions {len(inversions.get(treatment, []))})"
        )

    synth = comparisons["synth"]
    if synth.n < min_n:
        out.append(f"GT3 synthesis    : WITHHELD (n={synth.n} < {min_n})")
    else:
        delta = synth.mean_delta or 0.0
        out.append(
            f"GT3 synthesis    : {'PASS' if delta >= GATES['GT3_synth'] else 'FAIL'} "
            f"(Δ {delta:+.2f} vs +{GATES['GT3_synth']})"
        )

    if prefill_ratio is None:
        out.append("GT4 efficiency   : WITHHELD (no usage.json)")
    else:
        ok = prefill_ratio <= GT4_PREFILL_MAX
        out.append(
            f"GT4 efficiency   : {'PASS' if ok else 'FAIL'} "
            f"(prefill {prefill_ratio:.2f}x vs {GT4_PREFILL_MAX}x)"
        )

    # §7.7: ship CURSOR only if GT2 or GT3 clears at equal inversions.
    if any("WITHHELD" in v for v in out[:2]):
        out.append("SHIP DECISION    : WITHHELD — gates not yet evaluable")
    else:
        equal_inversions = len(inversions.get(treatment, [])) <= min(
            len(v) for k, v in inversions.items() if k != treatment
        )
        passes = any(v.startswith(("GT2", "GT3")) and "PASS" in v for v in out)
        decision = "ship CURSOR" if passes and equal_inversions else "ship the baseline"
        out.append(f"SHIP DECISION    : {decision} (§7.7)")
    return out


def prefill_ratio_from(usage_path: Path | None, treatment: str, control: str) -> float | None:
    if usage_path is None or not usage_path.exists():
        return None
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    a = sum(u["prefill_tokens"] for u in usage if u["arm"] == treatment)
    b = sum(u["prefill_tokens"] for u in usage if u["arm"] == control)
    return a / b if a and b else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("scores", nargs="+", type=Path, help="judge.py JSON outputs")
    p.add_argument("--treatment", default="cursor")
    p.add_argument("--control", default="baseline")
    p.add_argument("--usage", type=Path, default=None, help="runs/arms/usage.json for GT4")
    p.add_argument(
        "--tie-threshold",
        type=float,
        default=0.5,
        help="Δ below this is a tie. Set from judge_selftest.py's MEASURED noise, not this default",
    )
    p.add_argument("--min-n", type=int, default=20, help="below this, verdicts are withheld")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    scores = load_scores(args.scores)
    comparisons, paired, inversions = compare(
        scores, args.treatment, args.control, tie_threshold=args.tie_threshold
    )
    if not paired:
        print(
            f"no meeting has both {args.treatment!r} and {args.control!r} — nothing to pair",
            file=sys.stderr,
        )
        return 2

    ratio = prefill_ratio_from(args.usage, args.treatment, args.control)

    print(f"{args.treatment} vs {args.control}  (paired on {len(paired)} meetings)")
    print(f"tie threshold Δ < {args.tie_threshold}")
    for comparison in comparisons.values():
        print(comparison.line(args.tie_threshold, args.min_n))
    print()
    for system, meetings in inversions.items():
        state = "none" if not meetings else ", ".join(meetings)
        print(f"  INVERT {system:<10}: {state}")
    print()
    for verdict in gate_verdicts(
        comparisons, inversions, args.treatment, ratio, min_n=args.min_n
    ):
        print(verdict)

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "treatment": args.treatment,
                    "control": args.control,
                    "paired_meetings": paired,
                    "tie_threshold": args.tie_threshold,
                    "prefill_ratio": ratio,
                    "metrics": {
                        m: {
                            "mean_delta": c.mean_delta,
                            "wins": c.wins,
                            "losses": c.losses,
                            "ties": c.ties,
                            "p_value": c.p_value,
                        }
                        for m, c in comparisons.items()
                    },
                    "inversions": inversions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
