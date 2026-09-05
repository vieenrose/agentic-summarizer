"""Reward for GRPO on the `SYNTHESIZE` step (SPEC §4.1, §5.2.2, §5.2.5).

**Why RL here at all, when six SFT pools preceded it.** SFT imitates targets, so the student
inherits the teacher's dispositions — and the supervision teacher's length prior is measured
IMMOVABLE: a 28-entry journal produced 664 characters and a 34-entry one 680, and raising
`max_tokens` from 1400 to 3000 returned **byte-identical output**. It is not truncated; it
stops. An explicit "35-45 characters per item" instruction changed nothing. **The teacher is
therefore a ceiling on rendering density**, which is exactly the deficit SPEC §5.2.5 identifies
as the binding constraint on G5 retention. RL optimises against the environment instead of
against a demonstration, so it is not bounded by what the teacher would have written.

**Why the SYNTHESIZE step alone, and not whole trajectories.** A full-meeting rollout is ~15
model calls; a synthesis rollout is one. The deficit being attacked lives entirely in
synthesis — the journal already guarantees every recorded point REACHES the prompt, and the
open question is whether the model USES it. Reading-step RL (for churn) needs credit assignment
across steps and costs ~15x more per rollout; it is a separate, later experiment.

**The reward is jointly gated, and that is not a stylistic choice — it is SPEC §5.2.2.**
Retention is INFLATABLE by the very defect the project is trying to remove: churned re-`ADD`s
become points, so they raise `recorded_points`, raise retention (duplicated content is trivially
easy to render) and lower `starved`. Measured on `v12`: retention 0.837 -> 0.921 *because* it
churned. A naive `reward = retention` would be reward-hacked on the first iteration, and this
repo has the receipts for it.

So the shape here is **hard constraints x a bounded objective**, never a weighted sum of
everything:

* **Grounding is a CONSTRAINT, not a term.** Any ungrounded specific zeroes the reward. A
  weighted penalty invites the policy to buy fabrications with coverage, which is precisely the
  trade `runs/clean-e3` and the `九十萬` incident show a model will take. Faithfulness is not
  purchasable.
* **Language and markup are constraints.** Off-language output, bullet drift, or leaked
  `<think>` / `（後改為：` markup zero the reward. These are product defects, not quality
  gradations.
* **Length is a CONSTRAINT with a floor and a ceiling.** SPEC §3 caps the summary; §5.2.5
  measures that the model uses only ~1/3 of that budget. Rewarding length directly would
  produce padding, so length only gates: too long is refused, absurdly short is refused, and
  everything between is scored on what it actually renders.
* **Retention is the OBJECTIVE**, measured exactly as `evalkit.behaviour` measures it, at the
  same containment threshold, so the training signal and the gate cannot drift apart.

**Everything here is deterministic and model-free.** No judge, no network, no second model in
the loop — the same property that makes the test suite runnable with no GPU makes this reward
cheap enough to compute for every sample of every group.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from arcsum.evalkit import grounding
from arcsum.evalkit.behaviour import RENDERED_CONTAINMENT, containment
from arcsum.lang import MIN_CJK_RATIO_PROSE, cjk_ratio, simplified_hits
from arcsum.tokens import heuristic_token_len

#: Markup that must never reach a user-visible summary. `<think>` is not hypothetical: the
#: base model emits it by habit and it reached the demo, accounting for 54% and 77% of the
#: tokens the grounding instrument flagged on `v16` before `prose.finalize` was fixed.
_BANNED = (re.compile(r"</?think>", re.I), re.compile(r"（後改為："))
_BULLET = re.compile(r"^\s*(?:[-*•▪]|\d+[.)、])\s*", re.M)

#: Collapse floor, in TOKENS per entry — not per budget.
#:
#: Tying it to the output budget was wrong and the tests caught it: a faithful three-entry
#: summary of 37 characters was refused as "collapsed" against a 50-character floor derived
#: from a 1,000-unit cap. How short is too short depends on how much there is to say, not on
#: how much the model is allowed to say. Eight tokens per entry is roughly a bare mention in
#: zh-TW, so anything under it cannot be addressing the entries at all.
#:
#: **Everything here is TOKENS because SPEC §3's budget is tokens.** Writing this in characters
#: was a live bug: at ~1.577 zh-TW characters per token on this tokenizer, a 1,000-CHARACTER cap
#: is a ~634-token budget — 37% tighter than the spec allows. Measured against it, 9 of 24
#: baseline generations were refused "over budget" while their retention averaged 0.956, so the
#: reward would have taught the model to TRUNCATE a summary that was within contract.
MIN_TOKENS_PER_ENTRY = 8

#: Never demand more than half the budget as a floor, so the constraint stays satisfiable at
#: the largest journals (49 entries x 8 = 392 tokens, which fits, but the cap protects
#: against a future `POINTS_CAP` change making the floor unreachable).
MAX_FLOOR_FRACTION = 0.5


@dataclass(frozen=True)
class RewardBreakdown:
    """Every component, kept separate so a reward-hacked run is diagnosable after the fact.

    A scalar alone cannot answer "did it earn this by rendering more, or by padding?" — and
    that question has already been asked once in this project, about `v12`.
    """

    reward: float
    retention: float
    rendered: int
    recorded: int
    chars: int
    tokens: int = 0
    refused: str = ""

    @property
    def ok(self) -> bool:
        return not self.refused


def score(
    summary: str,
    entries: list[str],
    memory_view: str,
    *,
    max_tokens: int = 1000,
    token_len: Callable[[str], int] = heuristic_token_len,
    containment_floor: float = RENDERED_CONTAINMENT,
) -> RewardBreakdown:
    """Reward one candidate summary against the journal it was written from.

    `entries` are the journal texts (`Memory.synthesis_view()`), `memory_view` the rendered
    prompt used as the grounding source — grounding is checked against the MEMORY, not the
    transcript, because at this step the memory is the model's whole world; a detail invented
    during reading is the reading step's defect and is not what this reward is shaping.
    """
    text = (summary or "").strip()
    n = len(entries)
    n_tokens = token_len(text)

    def refuse(why: str) -> RewardBreakdown:
        return RewardBreakdown(reward=0.0, retention=0.0, rendered=0, recorded=n,
                               chars=len(text), tokens=n_tokens, refused=why)

    if not text:
        return refuse("empty")
    if n_tokens > max_tokens:
        return refuse("over budget")
    if n and n_tokens < min(MIN_TOKENS_PER_ENTRY * n, MAX_FLOOR_FRACTION * max_tokens):
        return refuse("collapsed")
    for pattern in _BANNED:
        if pattern.search(text):
            return refuse("markup leak")
    if _BULLET.search(text):
        return refuse("bulleted")
    if cjk_ratio(text) < MIN_CJK_RATIO_PROSE:
        return refuse("off language")
    if simplified_hits(text):
        return refuse("simplified characters")
    if grounding.check("", text, memory_view).n_ungrounded:
        return refuse("ungrounded specific")

    if not n:
        return RewardBreakdown(reward=0.0, retention=0.0, rendered=0, recorded=0,
                               chars=len(text), tokens=n_tokens, refused="no entries")
    rendered = sum(1 for e in entries if containment(e, text) >= containment_floor)
    retention = rendered / n
    return RewardBreakdown(reward=retention, retention=retention, rendered=rendered,
                           recorded=n, chars=len(text), tokens=n_tokens)
