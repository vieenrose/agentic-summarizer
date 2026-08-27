"""The G1 revision probe (SPEC §5.2) — the one check that must pass before any
corpus-scale evaluation runs.

Aggregate scores cannot show the one thing external memory buys that map-reduce
structurally cannot do: letting a later chunk overturn an earlier conclusion. Each
`ProbeMeeting` is a hand-built transcript with a planted decision that reverses late in
the meeting (approved -> rescinded), plus a distractor topic that must not appear — see
`ProbeMeeting.distractor_terms`'s docstring for why the distractor must be genuinely
non-decision-bearing filler, not a closed decision on an unrelated topic.

**Pass = the final summary states the LATER decision, does not state the earlier one as
current, and omits the distractor.** `score_probe` operates on the finished prose text
alone — deliberately decoupled from how that prose was produced, so the SAME probe
meetings can score an agent run, a baseline run, or (SPEC §9 Phase 0a) a raw model
response with no harness involved at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from arcsum.probe_data import BUDGET_APPROVAL, OFFICE_MOVE
from arcsum.transcript import Utterance


@dataclass(frozen=True, slots=True)
class ProbeMeeting:
    name: str
    utterances: tuple[Utterance, ...]
    #: The polarity word for the decision as it stood BEFORE the reversal.
    early_decision: str
    #: The polarity word for the decision as it stands AFTER the reversal — the one
    #: state a correct summary must report as current.
    late_decision: str
    #: One entry per subject concept the summary must identify. Each entry is a tuple
    #: of ACCEPTABLE SURFACE FORMS for that one concept — satisfied when ANY form
    #: appears. Measured 2026-08-27: real Phase-2 outputs referred to the planted
    #: "B 棟" as "B 樓" and "行銷預算" as "預算案", scoring false FAILs while stating
    #: the reversal itself perfectly correctly. This is the false-negative case
    #: `score_probe`'s docstring anticipated. Widening surface forms cannot weaken the
    #: probe: a stale summary still fails on `states_earlier_as_current`, which is
    #: independent of this field (pinned by
    #: `tests/test_probe.py::test_stale_summary_fails_even_with_subject_variants`).
    subject_terms: tuple[tuple[str, ...], ...]
    #: Terms belonging to an unrelated topic planted in the transcript. A correct
    #: summary, prioritising the meeting's actual reversal within SPEC §3's <1,000-token
    #: budget, omits these entirely.
    #:
    #: **Must be genuinely non-decision-bearing (SPEC §4.2's "self-contained procedure"
    #: bucket), never a closed decision on an unrelated topic.** §4.2 normatively
    #: instructs the teacher to emit edit lines for EVERY official item overlapping a
    #: chunk, with no relevance filter — only procedural filler (roll call, motions,
    #: recess announcements) is trained to `NOP`. An earlier version of this probe used
    #: a closed decision ("...合約已經簽署") as the distractor, which no model trained
    #: per §4.2 could ever omit without contradicting that same training — measured
    #: directly: the (unfine-tuned) teacher model reproduced the exact same "failure"
    #: as the fine-tuned student, which is what exposed this as a probe-content bug
    #: rather than a model deficiency. Swapping to procedural filler (this version) was
    #: verified to be correctly dropped by the trained model in 3/3 trials.
    distractor_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    states_later: bool
    states_earlier_as_current: bool
    distractor_absent: bool

    @property
    def passed(self) -> bool:
        return self.states_later and not self.states_earlier_as_current and self.distractor_absent


def score_probe(prose: str, meeting: ProbeMeeting) -> ProbeResult:
    """Score a finished summary against `meeting`'s planted reversal.

    `states_later` requires both the subject and the later decision's own word to
    appear — a summary that mentions the subject but never states its outcome does not
    pass. `states_earlier_as_current` is true only when the early decision's word
    appears WITHOUT the late one also appearing: a summary that narrates the reversal
    ("originally approved, but later rescinded") legitimately contains both words and
    must not be penalised for that — the failure mode this guards against is the early
    word appearing ALONE, meaning the reversal never made it into the summary at all and
    the reader is left with a stale reading.

    **Known limitation**: literal substring matching against one word per polarity, not
    a paraphrase-tolerant check. A summary that reports the reversal using a synonym
    the probe doesn't name (e.g. "不通過" when `late_decision` is "撤回") would score as
    a false failure. Acceptable for a cheap, synthetic, hand-graded probe (SPEC §5.2
    calls it exactly that); revisit if Phase 2's real model outputs show this producing
    false negatives in practice.

    **Matching is internal-whitespace-insensitive.** Measured 2026-08-27: a correct,
    semantically faithful summary scored a false FAIL because it rendered a planted
    subject term ("B 棟") without the space ("B棟") — inconsistent spacing around a
    Latin-letter-plus-CJK term is generation noise, not a faithfulness difference, and
    penalising it would have masked a real G1 pass. All matching strips internal
    whitespace from both the prose and the terms before comparing.
    """
    squashed_prose = _squash(prose)
    states_later = _squash(meeting.late_decision) in squashed_prose and all(
        any(_squash(form) in squashed_prose for form in forms) for forms in meeting.subject_terms
    )
    states_earlier_as_current = (
        _squash(meeting.early_decision) in squashed_prose
        and _squash(meeting.late_decision) not in squashed_prose
    )
    distractor_absent = not any(_squash(t) in squashed_prose for t in meeting.distractor_terms)
    return ProbeResult(meeting.name, states_later, states_earlier_as_current, distractor_absent)


def _squash(text: str) -> str:
    """Remove ALL whitespace, not just collapse it — matching must be blind to whether
    a Latin-letter-plus-CJK boundary picked up a space, since that is a rendering
    choice, not a faithfulness signal (see `score_probe`'s docstring)."""
    return "".join(text.split())


def probe_meetings() -> tuple[ProbeMeeting, ...]:
    """Hand-built, not generated — a generated transcript would let the generator's own
    assumptions leak into what "passing" means. Two independent scenarios (different
    subject, different polarity vocabulary, different distractor) so a pass is not an
    artifact of one particular pair of words.

    **The transcripts MUST span more than one chunk at the production budget**, or the
    probe silently stops testing anything: with a single chunk no memory crosses a step,
    `DROP` is never exercised, and the agent arm becomes a one-shot summariser. See
    `probe_data`'s module docstring for the measured history, and
    `tests/test_probe.py::test_probe_transcripts_span_multiple_chunks` for the pin.
    """
    return (
        ProbeMeeting(
            name="office_move_reversal",
            utterances=OFFICE_MOVE,
            early_decision="通過",
            late_decision="撤回",
            subject_terms=(("搬遷",), ("B 棟", "B 樓")),
            distractor_terms=("休息", "十分鐘"),
        ),
        ProbeMeeting(
            name="budget_approval_reversal",
            utterances=BUDGET_APPROVAL,
            early_decision="核准",
            late_decision="駁回",
            subject_terms=(("行銷預算", "預算案"),),
            distractor_terms=("麥克風", "音量"),
        ),
    )
