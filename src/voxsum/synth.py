"""Revision-dense synthetic meetings (PLAN.md §2, RESULTS.md conclusion 1).

Why this exists: natural meetings yield mostly ADD. `UPD`/`DEL` are rare and load-bearing —
they *are* the decision-chain behaviour G1 tests and the agency GT3 pays for. A trace set
sampled from real transcripts alone teaches a student that can only append.

Why zh-TW gets more of them: the teacher screen found the revise-don't-append behaviour is
markedly weaker in zh-TW than en at equal capability (RESULTS.md). The weaker side needs
*more* demonstrations, not the same number, so `build_set` oversamples zh by default.

Four revision kinds, each producing a different op shape:

| kind | transcript beat | expected op |
|---|---|---|
| `reversal` | rejected, then approved | UPD on DECISIONS |
| `deadline` | date stated, then moved | UPD on ACTIONS |
| `reassign` | owner named, then changed | UPD on ACTIONS |
| `withdraw` | item raised, then dropped entirely | DEL |

Every meeting also carries a trap topic that is raised and explicitly ruled out of scope,
so "did it stay quiet about the thing that was dropped" is always measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle

from .transcript import Utterance

__all__ = ["REVISION_KINDS", "SynthMeeting", "build_set", "build_meeting"]

REVISION_KINDS = ("reversal", "deadline", "reassign", "withdraw")

LINE_GAP = 30  # seconds between lines


@dataclass(frozen=True, slots=True)
class SynthMeeting:
    """A synthetic meeting with the revision it plants made explicit."""

    meeting_id: str
    lang: str
    kind: str
    utterances: tuple[Utterance, ...]
    #: Line where the original claim is stated.
    setup_at: int
    #: Line that revises it — the step where UPD/DEL is the correct answer.
    revision_at: int
    #: Line raising the trap topic.
    trap_at: int
    subject_terms: tuple[str, ...]
    trap_terms: tuple[str, ...]
    expected_op: str

    def line_at(self, sec: int) -> Utterance | None:
        return next((u for u in self.utterances if u.start == sec), None)

    def render(self) -> str:
        return "".join(u.render() + "\n" for u in self.utterances)


# Subjects vary so a student cannot key on one noun phrase. (en term, zh term)
_SUBJECTS = [
    ("the vendor contract", "廠商合約"),
    ("the hiring freeze", "人事凍結"),
    ("the office relocation", "辦公室搬遷"),
    ("the pricing change", "價格調整"),
    ("the security audit", "資安稽核"),
    ("the training budget", "訓練預算"),
    ("the warehouse merger", "倉庫合併"),
    ("the support rota", "值班輪替"),
]

_TRAPS = [
    ("the coffee machine", "咖啡機"),
    ("the parking spaces", "停車位"),
    ("the team offsite", "團隊旅遊"),
    ("the printer contract", "印表機合約"),
]

_OWNERS = [("Mei", "美玲"), ("Jordan", "家豪"), ("Priya", "淑芬")]

_FILLER = {
    "en": [
        "Can everyone hear me?",
        "Yes, go ahead.",
        "Let me pull up the document.",
        "One moment please.",
        "Sorry, I was on mute.",
        "Right, where were we.",
    ],
    "zh-TW": [
        "大家聽得到嗎？",
        "聽得到，請說。",
        "我把文件打開。",
        "請稍等一下。",
        "抱歉，我剛才靜音。",
        "好，我們剛講到哪裡。",
    ],
}


def _beats(
    lang: str, kind: str, subject: str, trap: str, owners: tuple[str, str]
) -> list[tuple[str, str, str]]:
    """(tag, speaker, text) beats for one meeting. `tag` marks the planted positions."""
    o1, o2 = owners
    if lang == "en":
        opening = [
            ("topic", "S1", f"Next item is {subject}."),
            ("filler", "S2", ""),
            ("context", "S2", f"I have circulated the details on {subject}."),
        ]
        trap_beats = [
            ("trap", "S3", f"Can we also cover {trap} today?"),
            ("context", "S1", "No, that is out of scope for this meeting, we will skip it."),
        ]
        closing = [("filler", "S2", ""), ("context", "S1", "Good, that covers the agenda.")]
        bodies = {
            "reversal": [
                ("setup", "S1", f"For now we reject {subject}."),
                ("context", "S2", "I can rework the numbers if that helps."),
                ("revision", "S1", f"With the reworked numbers, {subject} is approved."),
            ],
            "deadline": [
                ("setup", "S2", f"{o1} will finish {subject} by 14 March."),
                ("context", "S3", "That clashes with the audit week."),
                ("revision", "S2", f"Then {o1} will finish {subject} by 28 March instead."),
            ],
            "reassign": [
                ("setup", "S1", f"{o1} will own {subject}."),
                ("context", "S2", f"{o1} is on leave for three weeks."),
                ("revision", "S1", f"{o2} will own {subject} instead."),
            ],
            "withdraw": [
                ("setup", "S2", f"We should add {subject} to this quarter's plan."),
                ("context", "S3", "There is no budget line for it."),
                ("revision", "S1", f"Then drop {subject} from the plan entirely."),
            ],
        }
    else:
        opening = [
            ("topic", "S1", f"下一個議題是{subject}。"),
            ("filler", "S2", ""),
            ("context", "S2", f"我已經把{subject}的資料發給大家。"),
        ]
        trap_beats = [
            ("trap", "S3", f"今天要不要也討論{trap}？"),
            ("context", "S1", "不用，那個不在今天範圍內，先跳過。"),
        ]
        closing = [("filler", "S2", ""), ("context", "S1", "好，議程就到這裡。")]
        bodies = {
            "reversal": [
                ("setup", "S1", f"目前先否決{subject}。"),
                ("context", "S2", "如果需要，我可以重新計算數字。"),
                ("revision", "S1", f"依照重算後的數字，{subject}通過。"),
            ],
            "deadline": [
                ("setup", "S2", f"{o1}會在三月十四號前完成{subject}。"),
                ("context", "S3", "那一週跟稽核撞期。"),
                ("revision", "S2", f"那改成{o1}在三月二十八號前完成{subject}。"),
            ],
            "reassign": [
                ("setup", "S1", f"{subject}由{o1}負責。"),
                ("context", "S2", f"{o1}要請假三週。"),
                ("revision", "S1", f"{subject}改由{o2}負責。"),
            ],
            "withdraw": [
                ("setup", "S2", f"我們應該把{subject}排進本季計畫。"),
                ("context", "S3", "這個沒有預算科目。"),
                ("revision", "S1", f"那就把{subject}從計畫中整個移除。"),
            ],
        }
    # Trap sits between setup and revision so the model must hold the revision across it.
    body = bodies[kind]
    return [*opening, *body[:2], *trap_beats, ("filler", "S3", ""), *body[2:], *closing]


def build_meeting(
    meeting_id: str, lang: str, kind: str, *, variant: int = 0, padding: int = 0
) -> SynthMeeting:
    """One revision-dense meeting. `variant` rotates subject, trap and owners.

    `padding` inserts filler lines between the setup and the revision so the two land in
    DIFFERENT chunks at a given chunk budget. This is the whole point of these meetings: if
    the model sees the setup and its later contradiction together, it can simply ADD the
    final state and UPD is never the correct answer — the meeting teaches nothing.

    Sizing: an unpadded meeting is ~137 tokens, so it fits whole inside any production chunk
    budget. `padding_for(budget)` computes the filler needed to force a split.
    """
    if kind not in REVISION_KINDS:
        raise ValueError(f"unknown revision kind: {kind!r}")
    if lang not in _FILLER:
        raise ValueError(f"unsupported language: {lang!r}")

    idx = 0 if lang == "en" else 1
    subject_pair = _SUBJECTS[variant % len(_SUBJECTS)]
    trap_pair = _TRAPS[variant % len(_TRAPS)]
    owner_a = _OWNERS[variant % len(_OWNERS)]
    owner_b = _OWNERS[(variant + 1) % len(_OWNERS)]

    subject, trap = subject_pair[idx], trap_pair[idx]
    beats = _beats(lang, kind, subject, trap, (owner_a[idx], owner_b[idx]))

    # Insert padding immediately before the revision beat, so setup and revision are pushed
    # into different chunks while the trap still sits between them.
    if padding > 0:
        revision_at = next(i for i, b in enumerate(beats) if b[0] == "revision")
        beats = [
            *beats[:revision_at],
            *[("filler", "S2", "")] * padding,
            *beats[revision_at:],
        ]

    filler = cycle(_FILLER[lang])
    utterances: list[Utterance] = []
    marks: dict[str, int] = {}
    for i, (tag, speaker, text) in enumerate(beats):
        start = i * LINE_GAP
        utterances.append(Utterance(start, speaker, next(filler) if tag == "filler" else text))
        if tag in ("setup", "revision", "trap"):
            marks.setdefault(tag, start)

    # Strip the article from the en subject so terms match bullet text either way.
    words = tuple(t for t in subject.replace("the ", "").split() if len(t) > 2)
    subject_terms = words or (subject,)
    return SynthMeeting(
        meeting_id=meeting_id,
        lang=lang,
        kind=kind,
        utterances=tuple(utterances),
        setup_at=marks["setup"],
        revision_at=marks["revision"],
        trap_at=marks["trap"],
        subject_terms=subject_terms if lang == "en" else (subject,),
        trap_terms=(trap,),
        expected_op="DEL" if kind == "withdraw" else "UPD",
    )


def padding_for(budget: int) -> int:
    """Filler lines needed so setup and revision fall in different chunks at `budget`.

    A filler line costs roughly 12 tokens rendered. Two full chunks of filler are inserted so
    the split survives the chunker's 2-line overlap, which is what makes a naive one-chunk
    estimate fail (at budget 128 the revision leaks back into chunk 0).
    """
    return max(2 * budget // 12, 0)


def build_set(
    *, en_per_kind: int = 2, zh_per_kind: int = 4, chunk_budget: int = 0
) -> list[SynthMeeting]:
    """The revision-dense set. zh-TW is oversampled by default — see the module docstring.

    Defaults give 8 en + 16 zh = 24 meetings covering all four kinds. Pass `chunk_budget` to
    pad each meeting so its setup and revision land in different chunks at that budget;
    without it the meetings are ~137 tokens and fit whole inside any production chunk, which
    makes them teach no cross-chunk revision at all.
    """
    padding = padding_for(chunk_budget) if chunk_budget else 0
    out: list[SynthMeeting] = []
    for lang, per_kind in (("en", en_per_kind), ("zh-TW", zh_per_kind)):
        for kind in REVISION_KINDS:
            for v in range(per_kind):
                tag = "en" if lang == "en" else "zh"
                out.append(
                    build_meeting(
                        f"synth-{tag}-{kind}-{v}", lang, kind, variant=v, padding=padding
                    )
                )
    return out
