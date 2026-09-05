# Journal-shaped SYNTHESIZE supervision — outcome, 2026-09-03

Three pools, two seeds each, all six checkpoints evaluated on `data/heldout_zh` (40 meetings,
the corpus where the journal actually engages — 10 meetings exceed the 16-point working set).

| pool | churn s0 | churn s1 | mean | **spread** | clean s0 | clean s1 | retention s0/s1 |
|---|---|---|---|---|---|---|---|
| `v11` pre-journal synthesis | 3.5% | 13.3% | 8.4% | 9.8 pp | 13/40 | 5/40 | 0.837 / 0.836 |
| `v12` journal synthesis | 29.8% | 17.0% | 23.4% | 12.9 pp | 4/40 | 6/40 | 0.921 / 0.937 |
| `v13` journal + dedup | 36.7% | 9.2% | 23.0% | **27.4 pp** | 1/40 | 6/40 | 0.936 / 0.936 |

## What is established

**The supervision achieved exactly what it was built for.** G5 retention — the fraction of
everything recorded that reaches the summary — goes **0.837 → 0.936**, and it is the one
metric here that is *stable*: the within-pool spread across seeds is 0.001 (v11), 0.016 (v12)
and **0.000** (v13), against a between-pool difference of +0.10. That is a 6x-to-100x
signal-to-noise ratio, and it reproduces at both seeds in both journal pools.

This was the deficit the work targeted: v1.1 made a recorded point SURVIVE to synthesis, and
the remaining failure was that synthesis did not USE it (~40 entries in, 346 characters out).
That is now fixed and the fix replicates.

## What is NOT established, and the reason is the important part

**The churn difference is not resolvable at n = 2 per arm.** Seed spread reaches **27.4
percentage points** within a single pool (`v13`: 36.7% vs 9.2%). The between-pool differences
are of the same order or smaller. Any statement of the form "pool X churns more than pool Y"
requires more replicates than were run.

Two claims made earlier in this session and now RETRACTED:

1. **"`v12` is a decisive churn regression, p = 2.2e-07."** That paired sign test compared one
   run of each pool. It measures whether a difference is consistent ACROSS MEETINGS, and is
   silent on whether a retrained pair reproduces it. The run-to-run term is larger than the
   effect.
2. **"Near-duplicate redundancy in the journal causes the churn."** The mechanism was real in
   the DATA — near-duplicate entries went 5.6% → 11.2% when the view became journal-shaped, and
   the coverage instruction did force the teacher to restate both halves. But the fix that
   removed it (7.2% → 2.1% of entries) did **not** reduce churn: `v13` means 23.0% against
   `v12`'s 23.4%. A measured mechanism in the data is not a demonstrated cause in the model.

The dedup change is retained anyway — it makes the synthesis input strictly more honest, and
`v13`'s retention (0.936/0.936) is the most stable of the three — but it must not be described
as a churn fix.

## Recommendation

**Keep serving `v11-e3`.** It has the lowest mean churn (8.4%) and its worse seed (13.3%) is
still below both journal pools' means. But note this is a weak preference: `v11-e3`'s
much-quoted 3.5% is its LUCKY seed, and its own replicate is 13.3%.

**Carry the journal supervision forward** — it is verifiably better data (0 ungrounded across
2,068 specifics against the old pool's 39.9%; 113 rows above the old 16-entry ceiling; 69 rows
carrying supersession markup that previously had none) and it delivers the retention gain
robustly. What it needs is a churn-neutral formulation, not abandonment.

## The standing methodological consequence

**Run-to-run variance on this setup is up to 27 percentage points of churn at fixed data.**
Every single-run A/B in this project's history is subject to it, including the ones that
motivated retrains. Training costs ~35 minutes, so replicates are affordable; treat n = 1 as
a pilot and n = 2 as the minimum for any behavioural claim, and expect n = 2 to be
insufficient for effects under ~15 points.

**Read `retention` beside `churn`, always.** Churned re-`ADD`s become points, so they inflate
`recorded_points`, inflate retention, and reduce `starved`. In `v12` every metric that improved
improved partly because of the defect. The two must be reported together or the scorecard
rewards the failure.
