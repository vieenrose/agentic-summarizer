"""Planted-fact screen meetings — gate G1 and the teacher screen (CLAUDE.md §7.6).

Each meeting plants, at known timestamps:

* a plan **rejected** early and the *same* plan **approved** later (the decision chain);
* **two deadlines**, both stated explicitly;
* one **trap topic** — raised and explicitly dropped, which must not appear in the notes.

The planted positions are returned alongside the transcript, so a screen can ask precise
questions ("did the model revise the rejected decision after the approval line?") instead
of eyeballing output. Filler is deliberately low-content so NOP is the correct answer on
those chunks — that is what makes the NOP-collapse metric meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .transcript import Utterance

__all__ = ["ScreenMeeting", "screen_meetings"]


@dataclass(frozen=True, slots=True)
class ScreenMeeting:
    """A synthetic meeting with known planted facts."""

    name: str
    lang: str
    utterances: tuple[Utterance, ...]
    rejected_at: int
    approved_at: int
    deadlines_at: tuple[int, ...]
    trap_at: int
    subject_terms: tuple[str, ...] = field(default_factory=tuple)
    trap_terms: tuple[str, ...] = field(default_factory=tuple)

    def line_at(self, sec: int) -> Utterance | None:
        return next((u for u in self.utterances if u.start == sec), None)


_FILLER_EN = [
    "Can everyone hear me?",
    "Yes, loud and clear.",
    "Let me share my screen.",
    "One moment, the slides are loading.",
    "Sorry, I was on mute.",
    "Right, where were we.",
    "I think that is everything from me.",
    "Anything else before we wrap up?",
]

_FILLER_ZH = [
    "大家聽得到嗎？",
    "聽得很清楚。",
    "我把畫面分享出來。",
    "稍等一下，簡報還在載入。",
    "抱歉，我剛才靜音了。",
    "好，我們剛講到哪裡。",
    "我這邊差不多就這樣。",
    "還有其他事情要討論嗎？",
]


def _build(
    name: str,
    lang: str,
    beats: list[tuple[str, str, str]],
    filler: list[str],
    *,
    subject_terms: tuple[str, ...],
    trap_terms: tuple[str, ...],
) -> ScreenMeeting:
    """Interleave planted beats with filler, one line per 30 s.

    `beats` entries are `(kind, speaker, text)`; `kind` tags the planted positions.
    """
    utterances: list[Utterance] = []
    marks: dict[str, list[int]] = {}
    t, f = 0, 0

    for kind, speaker, text in beats:
        if kind == "filler":
            utterances.append(Utterance(t, speaker, filler[f % len(filler)]))
            f += 1
        else:
            utterances.append(Utterance(t, speaker, text))
            marks.setdefault(kind, []).append(t)
        t += 30

    return ScreenMeeting(
        name=name,
        lang=lang,
        utterances=tuple(utterances),
        rejected_at=marks["rejected"][0],
        approved_at=marks["approved"][0],
        deadlines_at=tuple(marks["deadline"]),
        trap_at=marks["trap"][0],
        subject_terms=subject_terms,
        trap_terms=trap_terms,
    )


_BEATS_EN: list[tuple[str, str, str]] = [
    ("topic", "S1", "Today we decide on the warehouse consolidation plan."),
    ("filler", "S2", ""),
    ("context", "S2", "The plan merges the north and south warehouses into one site."),
    ("context", "S3", "My concern is the transition cost, it looks understated."),
    ("rejected", "S1", "Then we reject the warehouse consolidation plan as it stands."),
    ("filler", "S3", ""),
    ("context", "S2", "I can revise the costing with the updated freight numbers."),
    ("trap", "S3", "Should we also discuss the office coffee machine budget?"),
    ("trap_drop", "S1", "No, that is out of scope for today, we will skip it entirely."),
    ("filler", "S2", ""),
    ("context", "S2", "Here is the revised costing, transition cost drops by 40 percent."),
    ("approved", "S1", "On that basis the warehouse consolidation plan is approved."),
    ("deadline", "S2", "I will submit the revised plan by 14 March."),
    ("filler", "S3", ""),
    ("deadline", "S3", "Site surveys will be finished before the end of April."),
    ("context", "S1", "Good, we will review progress at the next meeting."),
    ("filler", "S2", ""),
]

_BEATS_ZH: list[tuple[str, str, str]] = [
    ("topic", "S1", "今天要決定倉庫整併方案。"),
    ("filler", "S2", ""),
    ("context", "S2", "這個方案要把北倉和南倉合併到同一個場地。"),
    ("context", "S3", "我擔心轉換成本，看起來被低估了。"),
    ("rejected", "S1", "那我們否決目前這個版本的倉庫整併方案。"),
    ("filler", "S3", ""),
    ("context", "S2", "我可以用最新的運費數字重算成本。"),
    ("trap", "S3", "我們要不要順便討論辦公室咖啡機的預算？"),
    ("trap_drop", "S1", "不用，那個今天不在範圍內，完全先跳過。"),
    ("filler", "S2", ""),
    ("context", "S2", "重算後的成本在這裡，轉換成本下降了四成。"),
    ("approved", "S1", "既然如此，倉庫整併方案通過。"),
    ("deadline", "S2", "我會在三月十四號前送出修正後的方案。"),
    ("filler", "S3", ""),
    ("deadline", "S3", "場地勘查會在四月底之前完成。"),
    ("context", "S1", "好，下次會議再檢視進度。"),
    ("filler", "S2", ""),
]


def screen_meetings(*, repeat_filler: int = 1) -> list[ScreenMeeting]:
    """The screen set: one en and one zh-TW planted meeting.

    `repeat_filler` pads the filler beats to stretch the meeting across more chunks —
    useful for exercising NOP-collapse without changing any planted fact.
    """

    def expand(beats: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        if repeat_filler <= 1:
            return beats
        out: list[tuple[str, str, str]] = []
        for beat in beats:
            out.append(beat)
            if beat[0] == "filler":
                out.extend([beat] * (repeat_filler - 1))
        return out

    return [
        _build(
            "screen-en",
            "en",
            expand(_BEATS_EN),
            _FILLER_EN,
            subject_terms=("warehouse", "consolidation"),
            trap_terms=("coffee", "machine"),
        ),
        _build(
            "screen-zh",
            "zh-TW",
            expand(_BEATS_ZH),
            _FILLER_ZH,
            subject_terms=("倉庫", "整併"),
            trap_terms=("咖啡機", "咖啡"),
        ),
    ]
