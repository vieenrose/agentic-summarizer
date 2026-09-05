"""Per-step reward for the READING step, for rejection-sampling fine-tuning (RAFT).

**Why a step reward exists at all.** GRPO on `SYNTHESIZE` improved retention and grounding
(`runs/rl-v3`: retention 0.929, ungrounded 2.5%) and left churn untouched at ~3%, because an
`ADD` emitted at step 3 only pays off at synthesis on step 20 — a single terminal reward
carries no signal about which of fifteen reading steps was responsible. The literature calls
this the agentic credit-assignment problem and answers it with turn-level rewards at tool-call
boundaries (TRACE, Turn-PPO). This harness is unusually well placed for that: **a reading step
IS a tool-call boundary, and `guards.Outcome` already reports its consequences
deterministically** — which ops applied, which the harness refused and why, and whether an
applied `ADD` merely restates a point dropped in the same step.

**Why RAFT rather than more GRPO.** Rejection-sampling fine-tuning — sample N, keep the best,
supervised-train on those — is reported competitive with GRPO and PPO, and the stated reason is
that GRPO's advantage comes mostly from DISCARDING all-wrong prompts rather than from its
reward normalisation (arXiv 2504.11343). Two consequences matter here:

* **It needs only the policy network.** No frozen reference, no KL term, no full-vocabulary
  logits for a loss. GRPO on this model OOMed at 4 generations because the 248k-vocab logits
  tensor is the binding constraint; RAFT sidesteps that entirely and reuses the existing SFT
  trainer unchanged.
* **The kept samples are just SFT rows**, so every guarantee the pool already has — completion
  masking, prompt-version checking, replay-clean filtering — applies without new machinery.

**The reward is a RANKING, not a gate.** Unlike `rl.reward`, which scores a finished product
against hard constraints, this scores one edit against the state it edited. There is no
"correct" op sequence to compare against, so the reward orders candidates for the same prompt
and nothing more; only relative order within a group is ever used.

**Churn is the thing being priced.** `restates_dropped` is the harness's own detector, the same
function `evalkit.behaviour` uses to report the metric — so what RAFT optimises and what the
gate measures cannot drift apart. That identity is deliberate and is the reason this module
imports from `guards` rather than reimplementing the test.
"""

from __future__ import annotations

from dataclasses import dataclass

from arcsum.evalkit import grounding
from arcsum.guards import Outcome
from arcsum.memory import _near_duplicate
from arcsum.ops import Add, Arc, Drop, Malformed, Nop, Op, Revise
from arcsum.tokens import heuristic_token_len

#: A refused op is worse than an absent one: it spent decode tokens and taught nothing. The
#: harness refuses for stated reasons (`duplicate point`, `arc unchanged`, `no point with id`),
#: and measured on the production budget 24.5% of attempted ops are refused — so this is a
#: large, real share of the step's output, not an edge case.
REFUSED_PENALTY = 0.5

#: Churn — an applied `ADD` restating a point `DROP`ped in the same step — is the single
#: behaviour this reward exists to suppress, and it is penalised harder than a refusal because
#: it SUCCEEDS: it enters memory, displaces a slot, and inflates retention (SPEC §5.2.2). A
#: defect that improves the headline metric has to cost more than one that merely wastes.
CHURN_PENALTY = 2.0

#: A `Nop` on a chunk with real content is an abstention, and the real-ASR history shows the
#: model over-abstains: 69% NOP on genuinely substantive meetings at one point. Priced mildly
#: negative rather than zero so an all-NOP candidate never ties a candidate that recorded
#: something, while a NOP on an empty chunk (which the caller marks) stays free.
IDLE_NOP_PENALTY = 0.25

#: **An op asserting something the chunk does not contain is fabrication, and it must cost more
#: than the op earns.** Without this the reward credits any op the harness ACCEPTS, and the
#: harness accepts anything well-formed — so rejection sampling selects the most productive
#: liar. Caught on the first on-policy run: sampling against a stub chunk, the policy still
#: emitted 80 confident `ADD`s and the reward ranked them highly, because nothing checked them
#: against the source. This is the same failure `rl.reward` exists to prevent at synthesis, and
#: it has to be enforced at the step too — the memory a faithful synthesis renders is only as
#: honest as the reading step that filled it.
UNGROUNDED_PENALTY = 3.0

#: **Decode tokens are a GATED resource, so the reward has to price them.** SPEC §5.2's G4 is
#: a wall-clock ceiling, and on the reference device a reading step spends 19.3 s of its 77.7 s
#: decoding (190 tokens at 9.87 t/s measured at depth 3400). Without this term the reward is
#: purely additive in applied ops, so among candidates recording the SAME content the most
#: verbose one wins -- measured on the first pass: the kept rows average 78.9 completion tokens
#: against gold's 54.4, a **1.45x** decode cost, which alone moves a meeting from 20.3 to 22.5
#: minutes. Fixing starvation and passing G4 pull against each other, and a reward blind to
#: one of them will silently trade it away.
#:
#: Calibrated against the op credit, not chosen for feel: at 0.01, an applied op (+1) is worth
#: ~100 decode tokens, and a typical `ADD` costs ~25. So recording real content is still
#: strongly favoured -- this breaks ties toward brevity without re-introducing the abstention
#: it took RAFT to remove.
DECODE_TOKEN_COST = 0.01

#: An `ADD` that near-duplicates another op in the same step. The harness ACCEPTS these -- they
#: differ enough to clear the exact-duplicate refusal -- so they earn full applied credit while
#: adding nothing, which is how one sampled step scored 6.75 while emitting the same
#: 30-acre-negotiation point three times. `memory.synthesis_view` already collapses them at
#: synthesis, so the op's content was never going to survive; paying for its decode is pure
#: loss. Priced at one op's credit so a duplicate nets to zero rather than negative: it is
#: waste, not fabrication, and should not outrank an ungrounded claim.
NEAR_DUPLICATE_PENALTY = 1.0


@dataclass(frozen=True)
class StepScore:
    """One candidate step's score, with the components kept separate.

    A scalar cannot answer "did it win by recording more, or by churning less?", and this
    project has already had to reconstruct that answer once, for `v12`.
    """

    score: float
    applied: int
    refused: int
    churn: int
    malformed: int
    revised: int
    ungrounded: int = 0
    decode_tokens: int = 0
    near_duplicates: int = 0


def score_step(
    outcome: Outcome,
    ops: list[Op],
    *,
    chunk_text: str = "",
    chunk_has_content: bool = True,
    raw: str = "",
) -> StepScore:
    """Rank one candidate reading step by what the harness actually did with it.

    `chunk_text` is the SOURCE the step read. Passing it enables the grounding check; omitting
    it disables the single constraint that stops this reward selecting for fabrication, so
    callers scoring real candidates must always supply it.

    `raw` is the candidate's literal completion, whose length is what G4 actually pays for.
    Omitting it disables `DECODE_TOKEN_COST`, which leaves the reward additive in applied ops
    and therefore biased toward verbosity -- so callers scoring real candidates should supply
    it too.
    """
    applied = outcome.applied
    refused = sum(1 for r in outcome.results if not r.applied and not isinstance(r.op, Nop))
    churn = len(outcome.churn_points)
    malformed = len(outcome.malformed)
    revised = sum(1 for o in ops if isinstance(o, Revise))
    # Only content-bearing ops can fabricate; a DROP or NOP asserts nothing.
    asserted = [o.point for o in ops if isinstance(o, Add)]
    asserted += [o.text for o in ops if isinstance(o, Revise)]
    asserted += [o.text for o in ops if isinstance(o, Arc)]
    ungrounded = 0
    if chunk_text:
        for text in asserted:
            ungrounded += grounding.check("", text, chunk_text).n_ungrounded
    # `credited` is what the step CONTRIBUTED: applied ops carrying content. A `Drop` removes
    # information and a `Nop` adds none, so neither counts. Computed before `idle`, which is
    # defined in terms of it.
    credited = sum(1 for r in outcome.results if r.applied and not isinstance(r.op, Drop | Nop))
    # **Idle means the step RECORDED NOTHING, not that it emitted `Nop`.** Keying the penalty
    # on the op kind left a hole big enough to drive the whole `raft-s0-e1` regression through:
    # once a bare `Drop` stopped being credited it was still strictly better than `Nop`,
    # because `Nop` was penalised and dropping was free. A step that only discards a point is
    # an abstention with extra decode tokens, and it has to be scored as one.
    idle = 1 if (chunk_has_content and credited == 0) else 0

    # **A churn pair earns NO applied credit.** Both of its ops "apply" successfully — the
    # DROP removes a point and the ADD puts an equivalent one back — so counting them as work
    # exactly cancels the churn penalty: measured, a drop+restate scored 0.0, meaning a model
    # could churn indefinitely at zero cost and rejection sampling would never deselect it.
    # Neither op advanced the meeting, so neither is credited, and the penalty then bites.
    #
    # **A bare DROP earns no credit either, and that omission cost a whole training round.**
    # Crediting every applied op at +1 makes DISCARDING a point count as work, so the cheapest
    # way to score is to edit at high volume. `runs/raft-s0-e1` is what that trains: against
    # `rl-v3` it fixed starvation exactly as intended (17/40 -> 5/40 starved, NOP 46.2% ->
    # 7.9%, specifics +31%) and took churn from **2.9% to 44.7%**, four times over G7's
    # ceiling. The tell is the memory shape, not the churn counter: recorded points rose
    # 366 -> 604 while SURVIVING points stayed flat at ~345, so retirements went **18 -> 259**.
    # The model recorded more and threw almost all of it away.
    #
    # A drop is neutral, not penalised: retiring a point that turned out irrelevant is
    # legitimate and must stay free. It simply must not be PAID for. `revise` remains the
    # sanctioned way to say "this is now wrong, here is the correction", and it is credited,
    # because it carries replacement content.
    score = float(credited - 2 * churn)
    score -= REFUSED_PENALTY * refused
    score -= CHURN_PENALTY * churn
    # A malformed op is a protocol failure, not a judgement call: it cannot be applied and it
    # is the one error a fine-tune should never reproduce.
    score -= CHURN_PENALTY * malformed
    score -= IDLE_NOP_PENALTY * idle
    score -= UNGROUNDED_PENALTY * ungrounded
    # Wall clock is a gate (G4), so the reward pays for the tokens the step emits. Uses the
    # non-normative heuristic counter deliberately: this ranks candidates against each other,
    # it never produces a reported number, which is exactly the split `tokens.py` defines.
    decode_tokens = heuristic_token_len(raw) if raw else 0
    score -= DECODE_TOKEN_COST * decode_tokens
    # Counted WITHIN the step, among its own additions. Scoring against prior memory as well
    # would be better, but `Outcome` does not carry the pre-step memory and inventing a second
    # duplicate test here is exactly the drift `memory._near_duplicate` is imported to avoid --
    # so this stays the narrower, honest check. The harness's exact-match refusal already
    # covers re-adding an identical existing point.
    added = [o.point for o in ops if isinstance(o, Add)]
    near_duplicates = sum(
        1
        for i in range(len(added))
        for j in range(i + 1, len(added))
        if _near_duplicate(added[i], added[j])
    )
    score -= NEAR_DUPLICATE_PENALTY * near_duplicates
    # `revise` is the sanctioned form of what DROP+ADD does badly (SPEC §4.1 v1.1). It is NOT
    # rewarded as a bonus — that would invite gratuitous revision of correct points, the same
    # mistake as laundering churn into a sanctioned op. It simply avoids the churn penalty by
    # construction, which is the whole reason the op exists.
    return StepScore(
        score=score,
        applied=applied,
        refused=refused,
        churn=churn,
        malformed=malformed,
        revised=revised,
        ungrounded=ungrounded,
        decode_tokens=decode_tokens,
        near_duplicates=near_duplicates,
    )


def op_shape(ops: list[Op]) -> dict[str, int]:
    """Op counts by kind, for reporting which behaviours a kept set actually teaches."""
    kinds = {
        "add": Add,
        "drop": Drop,
        "revise": Revise,
        "arc": Arc,
        "nop": Nop,
        "malformed": Malformed,
    }
    return {name: sum(1 for o in ops if isinstance(o, cls)) for name, cls in kinds.items()}
