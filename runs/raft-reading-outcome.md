# RAFT on the reading step, round 1 — starvation FIXED, churn 15x WORSE — 2026-09-05

**Do not adopt any `raft-s*` checkpoint.** Round 1 is a reward-design failure, kept because
the failure is more useful than the result: it is the third instance of one pattern, and the
pattern now has a rule.

## What it was for

`rl-v3` (GRPO on `SYNTHESIZE`) is precision-strong and thin. On the 40 held-out meetings it
NOPs **46.2%** of chunks and starves **17 of 40**, concentrated in long meetings — 8 of the
10 meetings over 20 chunks, at a 60% median NOP rate against 35% under 20 chunks. GRPO could
not touch this because it only ever trained synthesis: an `ADD` emitted at step 3 pays off at
step 20, and a terminal reward carries no signal about which reading step was responsible.

`tools/raft_reading.py` samples 6 candidates per step ON-POLICY, scores each with
`rl.step_reward` against the harness's own `Outcome`, keeps the winner, and emits ordinary
SFT rows. Gold competes as a candidate and wins ties.

## The pool looked excellent

Audited against gold on the 882 steps where both exist, both scored against the chunk the
POLICY saw:

| | specifics | ungrounded | empty/NOP | adds/step |
|---|---|---|---|---|
| RAFT-kept | 938 | **6.4%** | **22.4%** | 1.40 |
| gold | 949 | **45.1%** | 48.2% | 0.90 |

That 45.1% is a finding in its own right and is NOT a RAFT artifact: §4.2 has the teacher
convert aligned segment MINUTES into ops, so a gold `ADD` may carry a figure the chunk does
not contain — but at serving time the model has only the chunk. **The reading-step gold has
the same "target is not a function of its input" defect found and repaired for `SYNTHESIZE`
and never checked on the reading half.**

## What training on it produced

Four checkpoints, two seeds x two epochs, `arcsum-eval` on `data/heldout_zh`, cache on:

| build | clean | churn | starved | NOP | retention | ungrounded | specifics | retired |
|---|---|---|---|---|---|---|---|---|
| `rl-v3` | 13/40 | **2.9%** | 17/40 | 46.2% | 0.929 | 2.5% | 204 | 18 |
| `raft-s0-e1` | 7/40 | 44.7% | **5/40** | 7.9% | 0.871 | 1.5% | 268 | 259 |
| `raft-s0-e2` | 4/40 | 56.2% | 4/40 | 12.4% | 0.856 | 4.1% | 294 | 302 |
| `raft-s1-e1` | 3/40 | 64.7% | 7/40 | 7.7% | 0.832 | 2.8% | 247 | 268 |

**Starvation is fixed and it replicates** — 5, 4, 7 starved against 17; NOP ~8% against
46.2%; specifics +21% to +44%. Three previous checkpoints never moved this.

**Churn fails G7 by 4-6x and that replicates too** (44.7 / 56.2 / 64.7 against a 10%
ceiling), so it is a property of the pool, not seed noise. Per SPEC §5.2.1 two seeds is what
makes that statement admissible.

## The diagnosis is in the memory shape, not the churn counter

Recorded points rose **366 → 604** while SURVIVING points stayed flat at **~345**, so
retirements went **18 → 259**. The model records far more and discards nearly all of the
surplus. `arc_frozen` — ARC ops refused as `arc unchanged` — went 22.9% → ~48%, meaning it
also emits an ARC almost every step and half are rejected.

Cause, in the reward: `score = applied - …` credited **every applied op at +1**, including a
bare `DROP`. Discarding a point counted as work, so the cheapest way to score was to edit at
high volume. **It was visible in the pool before it was visible in the checkpoint** — the
kept rows carry 0.65 drops per add against gold's 0.37, a 76% higher ratio — and the pool
audit missed it, having checked grounding and NOP but never the drop/add balance.

## Three fixes, and the rule they share

1. **A bare `DROP` is not credited.** Free, not penalised: retiring an irrelevant point is
   legitimate, and punishing it teaches hoarding, which is how the working set fills with
   stale points and `revise` never fires.
2. **`idle` means RECORDED NOTHING, not "emitted `Nop`".** Found because fix 1's test failed:
   with `DROP` merely uncredited it was still strictly better than `Nop` (−0.06 vs −0.27),
   since `Nop` was penalised and dropping was free. A step that only discards is an
   abstention with extra decode tokens.
3. **`ARC` is credited at 0.5, not 1.0.** It REPLACES one slot; `ADD` ACCUMULATES. With
   `DROP` uncredited, ARC becomes the next cheapest guaranteed credit — and the pool already
   carries 1.56x gold's ARC ops. Halved rather than zeroed because the arc is real work when
   the through-line moves; at 0.5 an accepted rewrite nets ~+0.2 while an unchanged one loses
   0.5 to refusal, so it pays only when the arc actually moved.

**The rule: whatever op is cheapest to emit for full credit becomes the policy. Price an op
by what it CONTRIBUTES, not by whether the harness accepted it.** The harness accepts
anything well-formed; acceptance is not achievement.

## Two process notes

* `--save-candidates` now exists. Sampling is the expensive half of RAFT (hours of served
  inference) and the reward is the cheap half; discarding losers made all three fixes above
  cost a full re-sample. They will not next time.
* `arcsum-eval` was serializing a hand-picked subset of `BehaviourReport`, so `decode_tokens`
  (G4's input) and `hedge_points` (the polarity-inversion guard that must be checked before
  shipping) never reached disk. **The four scorecards above report `decode_tokens: 0` and
  cannot support a G4 claim.** Fixed and pinned against the dataclass.

## What round 2 changes

Same policy (`rl-v3`), same 88 long meetings, same 6 candidates — only the reward differs, so
the comparison isolates it. The open question is whether the starvation fix survives pricing
the ops honestly, or whether the two were the same phenomenon all along.
