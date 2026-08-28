# sft-dropv3: negative result — late-step oversampling regressed the model

**Verdict: rejected. `sft-dropv2` remains the checkpoint.**

## Hypothesis

Deep in long meetings the student fixates — re-emitting a byte-identical `ARC` while the
transcript moves on. The gold pool barely covers that regime (1.6% of steps at index
40+, 6.5% at 30-39) while the correct behaviour there is *more* common than early on
(teacher NOP rate rises 32% → 51%). Raising late-step representation should fix it, and
fixing it should flip one ROUGE-1 meeting — all G3_rouge1 needed (14/20 p=0.115 →
15/20 p=0.021).

Built with `oversample_late_steps`: late share 14.2% → 22.0%, with NOP (0.353 → 0.359)
and DROP-bearing (0.320 → 0.325) held nearly constant and verified by measurement.

## Result (n=20, identical pinned config to dropv2's gate run)

| metric | dropv2 | dropv3 |
|---|---|---|
| rouge1 wins | **14/20** (p=0.115) | 12/20 (p=0.503) |
| rouge2 wins | **19/20** (p=0.000) | 17/20 (p=0.003) |
| rougeL wins | **19/20** (p=0.000) | 18/20 (p=0.000) |
| rouge1 mean delta | **+0.056** | +0.040 |

It moved AWAY from the gate on every ROUGE metric.

## The intervention worked; it just wasn't worth it

The mechanism did what it was designed to do:

- `LongBeachCC_05232017`, the 53-chunk fixation meeting: **-0.164 → -0.073**
- `AlamedaCC_11162021` -0.006 → +0.085 (flipped to a win)
- `LongBeachCC_02012022` -0.030 → +0.012 (flipped to a win)

And it broke four meetings dropv2 won:

- `DenverCityCouncil_10232017` +0.129 → -0.035
- `SeattleCityCouncil_06272016` +0.116 → -0.034
- `DenverCityCouncil_10172016` +0.118 → -0.007
- `SeattleCityCouncil_04022018` +0.033 → -0.008

Net -2 on the sign test.

## Why

Duplicating late steps duplicates NOP-heavy examples: late steps are where the teacher
NOPs most. That buys restraint deep in long meetings and pays for it with content
capture everywhere else — the same compounding trade that produced dropv1's churn,
running in the opposite direction.

**The lesson worth carrying: stable SHARES did not imply stable BEHAVIOUR.** The build
held NOP and DROP shares nearly constant and was still a behavioural regression, because
what changed was *which* samples carried those labels. Checking the reported shares is
necessary and not sufficient.

## What this rules out

Reweighting the existing 200 meetings is not the lever for the long-meeting regime. The
signal is real but too thin to amplify by duplication without trading away the rest.
Closing it needs genuinely more long-meeting supervision — SPEC Phase 4's corpus
expansion — not a different mix of what we have.

Deliberately NOT attempted: a search over `--target-late-frac` / `--late-min-step` to
find a variant that scores better on these same 20 meetings. That is fitting the eval
set, and the eval set is the only held-out data there is.
