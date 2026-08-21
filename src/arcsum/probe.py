"""The G1 revision probe (SPEC §5.2) — the one check that must pass before any
corpus-scale evaluation runs.

Aggregate scores cannot show the one thing external memory buys that map-reduce
structurally cannot do: letting a later chunk overturn an earlier conclusion. Each
`ProbeMeeting` is a hand-built transcript with a planted decision that reverses late in
the meeting (approved -> rescinded), plus a distractor topic that must not appear.

**Pass = the final summary states the LATER decision, does not state the earlier one as
current, and omits the distractor.** `score_probe` operates on the finished prose text
alone — deliberately decoupled from how that prose was produced, so the SAME probe
meetings can score an agent run, a baseline run, or (SPEC §9 Phase 0a) a raw model
response with no harness involved at all.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    subject_terms: tuple[str, ...]
    #: Terms belonging to an unrelated topic planted in the transcript. A correct
    #: summary, prioritising the meeting's actual reversal within SPEC §3's <1,000-token
    #: budget, omits these entirely.
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
    """
    states_later = meeting.late_decision in prose and all(t in prose for t in meeting.subject_terms)
    states_earlier_as_current = (
        meeting.early_decision in prose and meeting.late_decision not in prose
    )
    distractor_absent = not any(t in prose for t in meeting.distractor_terms)
    return ProbeResult(meeting.name, states_later, states_earlier_as_current, distractor_absent)


def probe_meetings() -> tuple[ProbeMeeting, ...]:
    """Hand-built, not generated — a generated transcript would let the generator's own
    assumptions leak into what "passing" means. Two independent scenarios (different
    subject, different polarity vocabulary, different distractor) so a pass is not an
    artifact of one particular pair of words."""
    return (
        ProbeMeeting(
            name="office_move_reversal",
            utterances=(
                Utterance("S1", "我們來討論辦公室搬遷案。"),
                Utterance("S2", "建議搬到 B 棟大樓，預算已經核准。"),
                Utterance("S1", "議案通過，確定搬遷至 B 棟。"),
                Utterance("S3", "順便提一下，員工餐廳菜單下個月會更換供應商。"),
                Utterance("S4", "新供應商報價比較便宜，員工餐廳合約已經簽署。"),
                Utterance("S1", "回到搬遷案，因為 B 棟消防檢查未通過，我們必須撤回搬遷決議。"),
                Utterance("S2", "同意撤回，搬遷案不通過，維持原辦公室。"),
            ),
            early_decision="通過",
            late_decision="撤回",
            subject_terms=("搬遷", "B 棟"),
            distractor_terms=("員工餐廳", "供應商"),
        ),
        ProbeMeeting(
            name="budget_approval_reversal",
            utterances=(
                Utterance("S1", "討論下一季行銷預算案。"),
                Utterance("S2", "建議編列兩百萬預算，用於新產品宣傳。"),
                Utterance("S1", "預算案核准，兩百萬行銷預算通過。"),
                Utterance("S3", "另外，公司園遊會的攤位租借合約已經確認。"),
                Utterance("S4", "園遊會攤位租借費用比去年略高，但已經簽約完成。"),
                Utterance("S1", "回到行銷預算，財務部發現數字有誤，決議駁回原預算案。"),
                Utterance("S2", "同意駁回，兩百萬預算不通過，需要重新提案。"),
            ),
            early_decision="核准",
            late_decision="駁回",
            subject_terms=("行銷預算",),
            distractor_terms=("園遊會", "攤位"),
        ),
    )
