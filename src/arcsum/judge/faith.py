"""Faithfulness scoring: prose -> claims -> per-claim verdicts -> `MeetingScore`
(SPEC §5.1).

Per claim in the summary: SUPPORTED / CONTRADICTED / UNSUPPORTED against retrieved
transcript spans. **Inversions are reported as a count, not folded into an average**
(SPEC §5.1: "a single inverted decision is a product defect, not a fractional score
penalty") — `MeetingScore.inverted` is a plain integer, never blended into
`faith_claim`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from arcsum.judge.client import JudgeClient
from arcsum.judge.evidence import Evidence, TranscriptIndex
from arcsum.tokens import char_tokens
from arcsum.transcript import Utterance

#: CJK sentence enders, both half- and full-width. Recalibrated from the prior
#: project's Latin `len > 15 chars` floor, which is wrong for zh — a 15-character zh
#: sentence is already substantial — to a floor measured in the normative tokenizer.
_SENTENCE_END = re.compile(r"(?<=[。！？；.!?;])")
MIN_CLAIM_TOKENS = 8


def split_claims(text: str) -> list[str]:
    """Split prose into claim-sized units for per-claim judging. A fragment shorter
    than `MIN_CLAIM_TOKENS` normative tokens is dropped — too short to be an
    independently checkable claim (a stray connective, a truncated fragment)."""
    raw = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    return [s for s in raw if len(char_tokens(s)) >= MIN_CLAIM_TOKENS]


_VERDICT_RE = re.compile(r"\b(SUPPORTED|CONTRADICTED|UNSUPPORTED)\b", re.IGNORECASE)


def parse_verdict(text: str) -> str | None:
    """The LAST matching verdict keyword in `text` — a judge that restates the claim
    and then decides ends right, so the final mention is the one that counts."""
    matches = _VERDICT_RE.findall(text)
    return matches[-1].upper() if matches else None


_SEVERITY = {"CONTRADICTED": 3, "UNSUPPORTED": 2, "SUPPORTED": 1}


@dataclass(frozen=True, slots=True)
class BulletVerdict:
    text: str
    votes: tuple[str, ...]

    @property
    def majority(self) -> str:
        """The plurality vote; on a genuine tie, the MOST SEVERE tied verdict wins —
        "a 0%-inversion requirement must not average away a dissent." A single
        CONTRADICTED vote in an otherwise-tied split must not be outvoted into
        invisibility."""
        counts = Counter(self.votes)
        top = max(counts.values())
        tied = [v for v, c in counts.items() if c == top]
        if len(tied) == 1:
            return tied[0]
        return max(tied, key=lambda v: _SEVERITY.get(v, 0))


_FAITH_SYS = (
    "你是一個事實查核助手。給定一段會議逐字稿的相關片段與一句摘要陳述，"
    "請判斷這句陳述是否受逐字稿內容支持。只回答以下三個詞之一："
    "SUPPORTED（有明確支持）、CONTRADICTED（與逐字稿矛盾）、"
    "UNSUPPORTED（逐字稿中找不到依據）。不要輸出其他文字。"
)


def faith_prompt(claim: str, evidence: Sequence[Evidence]) -> str:
    ev_text = "\n".join(f"- {e.render()}" for e in evidence) if evidence else "（無相關片段）"
    return f"逐字稿片段：\n{ev_text}\n\n摘要陳述：{claim}"


@dataclass(frozen=True, slots=True)
class MeetingScore:
    #: 1-5 scale: `1 + 4 * supported_rate`, matching the prior project's convention.
    faith_claim: float
    inverted: int
    unsupported: int
    bullets: tuple[BulletVerdict, ...]


def judge_meeting(
    prose: str,
    utterances: Sequence[Utterance],
    client: JudgeClient,
    *,
    model: str,
    votes: int = 3,
    top_k: int = 6,
) -> MeetingScore:
    """Score one finished summary's faithfulness against its source transcript.

    `votes=3`: the prior project measured a local judge flipping verdicts on
    IDENTICAL input (SUPPORTED/UNSUPPORTED/SUPPORTED on the same prompt) — a 0%
    inversion gate cannot rest on a single stochastic call, so each claim is judged
    `votes` times and resolved via `BulletVerdict.majority`.
    """
    index = TranscriptIndex(utterances)
    claims = split_claims(prose)

    bullets: list[BulletVerdict] = []
    for claim in claims:
        evidence = index.search(claim, top_k=top_k)
        prompt = faith_prompt(claim, evidence)
        raw_votes = tuple(
            parse_verdict(client(model, _FAITH_SYS, prompt)) or "UNSUPPORTED" for _ in range(votes)
        )
        bullets.append(BulletVerdict(claim, raw_votes))

    majorities = [b.majority for b in bullets]
    supported_rate = (
        sum(1 for m in majorities if m == "SUPPORTED") / len(majorities) if majorities else 0.0
    )
    return MeetingScore(
        faith_claim=1 + 4 * supported_rate,
        inverted=sum(1 for m in majorities if m == "CONTRADICTED"),
        unsupported=sum(1 for m in majorities if m == "UNSUPPORTED"),
        bullets=tuple(bullets),
    )
