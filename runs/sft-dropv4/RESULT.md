# sft-dropv4: Phase-4 long-meeting supervision fixes long meetings and costs short ones

**Verdict: 6 of 7 gates, same count as `sft-dropv2`, but for a different and more
useful reason. Not yet a ship; the first positive evidence that the long-meeting
deficit is data-bound and fixable.**

## What changed versus dropv2

One variable: **455 new gold steps, every one at chunk index >= 44**, taken from 50
newly built long meetings (`data/p4_zh`, median 55 chunks against the pilot's 16). Base
pool is dropv2's own `train.jsonl`, unchanged. Step index >= 40 goes 55 -> 510 (9.3x).

Two data defects were fixed before training, both measured rather than assumed:

- **51% of the teacher's gold steps did not replay cleanly** (SPEC §4.2 requires that
  they do). Every completion was replayed through the real harness and rewritten to the
  ops that actually applied: 2,904 ops removed, 341 steps dropped entirely. Dominated by
  "point too long", then "duplicate point", then "arc unchanged". A step left with no
  surviving op is **dropped, not rewritten to NOP** — converting a refused edit into a
  NOP would teach "nothing worth recording" about a chunk the teacher judged worth
  recording.
- **The new supervision is NOP-poor (10.1% after cleaning, 9% before).** Admitting all
  of it lands the pool at 23.6% NOP, below the 25.7% that caused dropv1's churn. Merging
  is therefore capped by arithmetic at a 32% floor. `downsample_nop` cannot rescue this;
  it only ever lowers the share.

## Result (n=20, same eval set and pinned config as dropv2's gate run)

| metric | dropv2 | dropv4 |
|---|---|---|
| rouge1 wins | 14/20 (p=0.115) | 14/20 (p=0.115) |
| rouge1 mean delta | +0.056 | **+0.097** |
| rouge2 wins | 19/20 (p=0.000) | 17/20 (p=0.003) |
| rougeL wins | 19/20 (p=0.000) | 17/20 (p=0.003) |

G3 rouge1 still fails: the effect size clears comfortably (lower bound +0.058) but the
sign test does not move.

## The mechanism is clean, and it is about meeting length

| slice | dropv2 wins | dropv4 wins | mean delta |
|---|---|---|---|
| **long, >= 400 lines** (n=9) | 4/9 | **8/9** | +0.012 -> **+0.217** |
| **short, < 400 lines** (n=11) | 10/11 | 6/11 | +0.091 -> -0.002 |

`corr(meeting length, dropv4 - dropv2 change) = +0.671`.

`LongBeachCC_05232017` — the 53-chunk fixation meeting that dropv3 could not fix by
reweighting — went **-0.164 -> +0.278**. `SeattleCityCouncil_06072021` +0.380,
`LongBeachCC_02012022` +0.375, `AlamedaCC_04042017` +0.304.

The sign test is unmoved because the build traded roughly six short-meeting wins for six
long-meeting wins.

## What this establishes that dropv3 did not

dropv3 tried to fix the same regime by **oversampling the existing 200 meetings** and
regressed (14/20 -> 12/20), concluding that "reweighting the existing pool is not the
lever". dropv4 confirms the other half of that claim: with **genuinely new** long-meeting
supervision the regime moves decisively (4/9 -> 8/9). The deficit is real, it is
data-bound, and it is fixable. Phase 4's premise holds.

## Leading hypothesis for the short-meeting cost

The pool's NOP share drifted **34.9% -> 32.0%** under the floor-based merge. A lower NOP
share makes the model readier to edit, which is the wrong trade on a meeting with few
chunks — the same compounding mechanism as dropv1's churn, at smaller amplitude.

`sft-dropv5` tests exactly this as a single variable: `mix_phase4.py --hold-nop` keeps
the share at 34.9% by admitting every new NOP sample alongside the late non-NOP ones,
which admits **more** long-meeting data (688 samples, 546 at index >= 40), not less.

**This hypothesis may be wrong.** dropv3's transferable lesson was that stable shares do
not imply stable behaviour; holding NOP constant makes dropv5 a fair test, not a
guaranteed fix. If short meetings stay depressed, the cause is the late-step data
shifting global behaviour, and no mixing ratio will address it.

## Caveats

- G2 was not run for this checkpoint (no judge server up); G4 remains dropv2's
  authorized projection, never measured on the phone.
- The eval set is the same 20 held-out meetings throughout. Repeatedly reading it while
  choosing between builds is a real multiple-comparisons exposure — it is the only
  held-out data there is, which is why no hyperparameter search was run against it.
- Confirmed zero overlap between the training pool and the eval meetings; the 20 appear
  only as the validation split, as they did for dropv2, so they produce no gradient.
