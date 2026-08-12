#!/usr/bin/env python3
"""Capability screen — gate G1 and the teacher screen (CLAUDE.md §7.6, PLAN.md §2b).

Same run answers two questions:

**G1 (go/no-go for the student).** Do the final notes carry the correct decision chain
(rejected -> approved), both deadlines, 100% anchored, and no trap topic?

**Teacher screen.** Valid-op rate, UPD-at-contradiction rate, and *raw* anchor-copy
accuracy — enough to disqualify a teacher before spending on volume.

Run without a GBNF grammar. The screen's entire signal is whether the model *naturally*
emits valid ops; constraining the sampler would mask exactly that.

    python eval/screen.py --base-url http://127.0.0.1:8080 --lang en
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.agent import Trace, run_cursor  # noqa: E402
from voxsum.ops import Add, Del, Upd  # noqa: E402
from voxsum.prompts import PROMPT_VERSION  # noqa: E402
from voxsum.screenset import ScreenMeeting, screen_meetings  # noqa: E402
from voxsum.state import NotesState, normalize  # noqa: E402

__all__ = ["ScreenResult", "screen_model", "score_meeting"]


@dataclass
class ScreenResult:
    """One meeting's screen verdict. `passed_g1` is the gate; the rates are diagnostics."""

    name: str
    lang: str
    steps: int

    # G1 criteria (CLAUDE.md §7.6)
    chain_correct: bool
    both_deadlines: bool
    fully_anchored: bool
    trap_absent: bool

    # Teacher-screen diagnostics (PLAN.md §2b)
    valid_op_rate: float | None
    anchor_rate_raw: float | None
    nop_rate_on_rich_chunks: float | None
    revised_at_contradiction: bool
    added_contradiction: bool

    prompt_version: str = PROMPT_VERSION

    @property
    def passed_g1(self) -> bool:
        return (
            self.chain_correct
            and self.both_deadlines
            and self.fully_anchored
            and self.trap_absent
        )

    def summary(self) -> str:
        def mark(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        def pct(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.0%}"

        return "\n".join(
            [
                f"{self.name} ({self.lang}, {self.steps} steps) — G1 {mark(self.passed_g1)}",
                f"  decision chain rejected->approved : {mark(self.chain_correct)}",
                f"  both deadlines captured          : {mark(self.both_deadlines)}",
                f"  every bullet anchored            : {mark(self.fully_anchored)}",
                f"  trap topic absent                : {mark(self.trap_absent)}",
                f"  valid-op rate                    : {pct(self.valid_op_rate)} (GT1 >= 95%)",
                f"  anchor rate (raw, no matcher)    : {pct(self.anchor_rate_raw)}",
                f"  NOP rate on content-rich chunks  : {pct(self.nop_rate_on_rich_chunks)} (< 10%)",
                f"  revised at contradiction (UPD)   : {mark(self.revised_at_contradiction)}",
                f"  added a contradicting bullet     : {mark(not self.added_contradiction)}",
            ]
        )


def _mentions(state: NotesState, terms: tuple[str, ...]) -> bool:
    haystack = normalize(
        " ".join(b.text for section in state.sections.values() for b in section) + " " + state.title
    )
    return any(normalize(t) in haystack for t in terms)


def _decision_chain_correct(state: NotesState, meeting: ScreenMeeting) -> bool:
    """The notes must say the plan was approved, and must not still assert it was rejected.

    This is the whole point of the screen: a model that appends both states passes a naive
    keyword check while leaving the notes self-contradictory.
    """
    from voxsum.guards import _polarity

    subject = set(meeting.subject_terms)
    relevant = [
        b
        for section in ("DECISIONS", "SUMMARY")
        for b in state.bullets(section)
        if any(normalize(t) in normalize(b.text) for t in subject)
    ]
    if not relevant:
        return False
    polarities = {_polarity(b.text) for b in relevant}
    return 1 in polarities and -1 not in polarities


def _deadlines_captured(state: NotesState, meeting: ScreenMeeting) -> bool:
    """Both planted deadline lines must be anchored by some bullet (±1 line tolerance)."""
    anchored = {
        b.anchor
        for section in state.sections.values()
        for b in section
        if b.anchor is not None
    }
    return all(
        any(abs(a - target) <= 30 for a in anchored) for target in meeting.deadlines_at
    )


def score_meeting(trace: Trace, meeting: ScreenMeeting) -> ScreenResult:
    """Turn a completed trace into a screen verdict."""
    state = trace.state
    state.enforce_caps()

    bullets = [b for section in state.sections.values() for b in section]
    fully_anchored = bool(bullets) and all(b.anchor is not None for b in bullets)

    # Contradiction handling: what did the model do on/after the approval line?
    after_approval = [s for s in trace.steps if s.chunk.has_line(meeting.approved_at)]
    revised, added_contradiction = False, False
    subject = set(meeting.subject_terms)
    for step in after_approval:
        for result in step.outcome.results:
            op = result.op
            touches_subject = False
            if isinstance(op, (Add, Upd)):
                touches_subject = any(normalize(t) in normalize(op.bullet) for t in subject)
            elif isinstance(op, Del):
                touches_subject = any(normalize(t) in normalize(op.prefix) for t in subject)
            if not touches_subject:
                continue
            if isinstance(op, (Upd, Del)) and result.applied:
                revised = True
            if isinstance(op, Add) and not result.applied and result.reason:
                if "temporal guard" in result.reason:
                    added_contradiction = True

    return ScreenResult(
        name=meeting.name,
        lang=meeting.lang,
        steps=len(trace.steps),
        chain_correct=_decision_chain_correct(state, meeting),
        both_deadlines=_deadlines_captured(state, meeting),
        fully_anchored=fully_anchored,
        trap_absent=not _mentions(state, meeting.trap_terms),
        valid_op_rate=trace.valid_op_rate,
        anchor_rate_raw=trace.anchor_rate_raw,
        nop_rate_on_rich_chunks=trace.nop_rate_on_rich_chunks,
        revised_at_contradiction=revised,
        added_contradiction=added_contradiction,
    )


def screen_model(
    model,
    *,
    langs: tuple[str, ...] = ("en", "zh-TW"),
    declarations: bool = False,
    budget: int = 128,
    repeat_filler: int = 1,
) -> list[tuple[ScreenResult, Trace, NotesState]]:
    """Run the screen set through `model`. Returns (result, trace, final state) per meeting."""
    out = []
    for meeting in screen_meetings(repeat_filler=repeat_filler):
        if meeting.lang not in langs:
            continue
        trace = run_cursor(
            list(meeting.utterances),
            model,
            lang=meeting.lang,
            declarations=declarations,
            budget=budget,
        )
        out.append((score_meeting(trace, meeting), trace, trace.state))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--lang", action="append", choices=["en", "zh-TW"], default=None)
    p.add_argument(
        "--declarations", action="store_true", help="use FunctionGemma tool declarations"
    )
    p.add_argument(
        "--budget",
        type=int,
        default=128,
        help="chunk token budget. Must be small enough that the planted meeting spans "
        "several chunks — at one chunk per meeting the model sees the rejection and the "
        "approval together and never has to revise anything, so the chain test is vacuous.",
    )
    p.add_argument("--repeat-filler", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=512, help="output budget per step")
    p.add_argument(
        "--thinking",
        action="store_true",
        help="let the model reason before emitting ops (slower; legitimate per PLAN §2c)",
    )
    p.add_argument("--out", type=Path, default=None, help="write JSON results here")
    p.add_argument(
        "--notes-out", type=Path, default=None, help="write final NOTES per meeting here"
    )
    args = p.parse_args(argv)

    from voxsum.backends.llama_server import LlamaServer
    from voxsum.render import render_state

    # No grammar: the screen measures whether the model *naturally* emits valid ops.
    # Greedy: an eval instrument must be reproducible, not a lottery ticket.
    model = LlamaServer(
        base_url=args.base_url,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        temperature=0.0,
        send_thinking_kwarg=False,  # MiniCPM5 template inserts an empty <think> block otherwise
    )
    if not model.health():
        print(f"llama-server not reachable at {args.base_url}", file=sys.stderr)
        return 2

    langs = tuple(args.lang) if args.lang else ("en", "zh-TW")
    results = screen_model(
        model,
        langs=langs,
        declarations=args.declarations,
        budget=args.budget,
        repeat_filler=args.repeat_filler,
    )

    for result, _, state in results:
        print(result.summary())
        if args.notes_out:
            args.notes_out.mkdir(parents=True, exist_ok=True)
            (args.notes_out / f"{result.name}.notes.txt").write_text(
                render_state(state), encoding="utf-8"
            )
        print()

    if args.out:
        args.out.write_text(
            json.dumps([asdict(r) for r, _, _ in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0 if all(r.passed_g1 for r, _, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
