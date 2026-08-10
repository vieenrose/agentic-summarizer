#!/usr/bin/env python3
"""Validate the judges before trusting any judged number (CLAUDE.md §7.3).

Two questions, both about the *instrument* rather than the systems:

**1. Does the judge detect a planted inversion?** For each meeting we have correct notes and
a polarity-flipped copy. A judge that scores the flipped copy as SUPPORTED cannot certify
the 0%-inversion product requirement, no matter how large it is. This is how
`gemma-3n-E4B-it` was disqualified — it answered SUPPORTED to everything in 4 tokens.

**2. How noisy is it?** §7.3 asserts ±0.4-0.5 and declares Δ < 0.5 a tie. That number is
inherited, and every GT2/GT3 threshold rests on it. Here it is measured: score the same
notes `--repeats` times and report the spread per judge.

    TOGETHER_API_KEY=... python eval/judge_selftest.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import (  # noqa: E402
    _COVER_SYS,
    _FAITH_SYS,
    _VERDICT,
    PANEL,
    JudgeBudgetExceeded,
    TogetherJudge,
    _score_from,
    cover_prompt,
    faith_prompt,
    parse_bullets,
)

from voxsum.index import TranscriptIndex  # noqa: E402
from voxsum.transcript import parse_transcript, sec_to_clock  # noqa: E402

# Polarity pairs used to plant an inversion in otherwise-correct notes.
FLIPS = [
    ("approved", "rejected"),
    ("approve", "reject"),
    ("agreed", "refused"),
    ("accepted", "declined"),
    ("通過", "否決"),
    ("核准", "駁回"),
    ("同意", "拒絕"),
]


def invert_notes(notes: str) -> tuple[str, list[str]]:
    """Flip decision polarity. Returns (flipped notes, terms that were flipped)."""
    out, flipped = notes, []
    for positive, negative in FLIPS:
        if positive in out:
            out = out.replace(positive, negative)
            flipped.append(f"{positive}->{negative}")
    return out, flipped


@dataclass
class JudgeReport:
    model: str
    inversion_caught: int = 0
    inversion_total: int = 0
    false_alarms: int = 0
    correct_total: int = 0
    scores: dict[str, list[int]] = field(default_factory=dict)

    @property
    def recall(self) -> float | None:
        return self.inversion_caught / self.inversion_total if self.inversion_total else None

    @property
    def false_alarm_rate(self) -> float | None:
        return self.false_alarms / self.correct_total if self.correct_total else None

    def noise(self) -> dict[str, float]:
        """Half-range and stdev of repeated scores, per metric AND meeting.

        Keys are `METRIC@meeting`: pooling repeats across meetings would add genuine
        between-meeting differences to the judge's own variance and overstate the noise.
        """
        out: dict[str, float] = {}
        for metric, values in self.scores.items():
            if len(values) > 1:
                out[f"{metric}_halfrange"] = (max(values) - min(values)) / 2
                out[f"{metric}_stdev"] = statistics.stdev(values)
        return out

    def summary(self) -> str:
        def pct(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.0%}"

        lines = [
            f"{self.model}",
            f"  planted inversions caught : {self.inversion_caught}/{self.inversion_total} "
            f"({pct(self.recall)})",
            f"  false alarms on correct   : {self.false_alarms}/{self.correct_total} "
            f"({pct(self.false_alarm_rate)})",
        ]
        for key, value in sorted(self.noise().items()):
            lines.append(f"  {key:<25} : {value:.2f}")
        for metric, values in sorted(self.scores.items()):
            lines.append(f"  {metric} runs                : {values}")
        return "\n".join(lines)


def _verdicts_for(
    notes: str, index: TranscriptIndex, client: TogetherJudge, model: str
) -> list[str]:
    """One verdict per bullet in claim mode."""
    out: list[str] = []
    for _section, bullet, anchor in parse_bullets(notes):
        evidence = index.evidence_for(bullet, anchor, mode="claim")
        text = bullet + (f" [{sec_to_clock(anchor)}]" if anchor is not None else "")
        raw = client(model, _FAITH_SYS, faith_prompt(text, evidence))
        hits = _VERDICT.findall(raw)
        out.append(hits[-1].upper() if hits else "MISSING")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("NOTES", "TRANSCRIPT"),
        required=True,
        help="correct notes + its transcript; repeatable",
    )
    p.add_argument("--repeats", type=int, default=5, help="repeated COVER/SYNTH scorings")
    p.add_argument("--budget-usd", type=float, default=0.20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    client = TogetherJudge(
        api_key=os.environ.get("TOGETHER_API_KEY", ""), budget_usd=args.budget_usd
    )
    verdict_models = [PANEL["faith"], PANEL["second"]]
    reports = {m: JudgeReport(m) for m in {*verdict_models, PANEL["cover"]}}

    try:
        for notes_path, transcript_path in args.pair:
            notes = Path(notes_path).read_text(encoding="utf-8")
            utterances = parse_transcript(Path(transcript_path).read_text(encoding="utf-8"))
            index = TranscriptIndex(utterances)
            inverted, flips = invert_notes(notes)
            if not flips:
                print(
                    f"[selftest] {notes_path}: no polarity term to flip; skipped",
                    file=sys.stderr,
                )
                continue
            print(f"[selftest] {Path(notes_path).name}: flipped {', '.join(flips)}", flush=True)

            # Only bullets whose text actually CHANGED are planted inversions. Counting
            # every bullet of the flipped notes understates recall badly: a TOPICS bullet
            # with no polarity term is unchanged and is still true, so SUPPORTED is the
            # correct verdict for it, not a miss.
            original = parse_bullets(notes)
            flipped_bullets = parse_bullets(inverted)
            changed = [a[1] != b[1] for a, b in zip(original, flipped_bullets, strict=False)]
            print(
                f"[selftest]   {sum(changed)}/{len(changed)} bullets carry the inversion",
                flush=True,
            )

            for model in verdict_models:
                report = reports[model]
                for verdict in _verdicts_for(notes, index, client, model):
                    report.correct_total += 1
                    if verdict == "CONTRADICTED":
                        report.false_alarms += 1
                verdicts = _verdicts_for(inverted, index, client, model)
                for verdict, is_inverted in zip(verdicts, changed, strict=False):
                    if not is_inverted:
                        continue
                    report.inversion_total += 1
                    if verdict == "CONTRADICTED":
                        report.inversion_caught += 1

            # Noise: repeat COVER/SYNTH on the *correct* notes only.
            transcript = "".join(u.render() + "\n" for u in utterances)
            prompt = cover_prompt(notes, transcript)
            model = PANEL["cover"]
            meeting = Path(notes_path).stem
            for _ in range(args.repeats):
                cover, synth = _score_from(client(model, _COVER_SYS, prompt))
                if cover is not None:
                    reports[model].scores.setdefault(f"COVER@{meeting}", []).append(cover)
                if synth is not None:
                    reports[model].scores.setdefault(f"SYNTH@{meeting}", []).append(synth)
    except JudgeBudgetExceeded as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)

    print()
    for report in reports.values():
        print(report.summary())
        print()
    print(client.spend.report())

    worst = max(
        (v for r in reports.values() for k, v in r.noise().items() if k.endswith("halfrange")),
        default=None,
    )
    if worst is not None:
        print(
            f"\nmeasured judge noise (worst per-meeting half-range): {worst:.2f}"
        )
        print(
            "  This is PER-MEETING noise on an integer 1-5 scale, where 0.50 half-range is\n"
            "  the granularity floor (an occasional one-point flip). Use it as the sign-test\n"
            "  tie band, NOT as a band on the mean: the mean over n meetings has standard\n"
            f"  error sigma/sqrt(n) — about {worst * 1.1 / (20 ** 0.5):.2f} at n=20 — so a gate\n"
            "  stated as a mean difference (GT3: +0.5) is several standard errors, not noise."
        )
    for report in reports.values():
        if report.recall is not None and report.recall < 1.0:
            print(
                f"WARNING: {report.model} missed {report.inversion_total - report.inversion_caught}"
                f" of {report.inversion_total} planted inversions — it cannot certify the "
                "0%-inversion requirement",
                file=sys.stderr,
            )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    m: {
                        "model": r.model,
                        "inversion_recall": r.recall,
                        "false_alarm_rate": r.false_alarm_rate,
                        "noise": r.noise(),
                        "scores": r.scores,
                    }
                    for m, r in reports.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
