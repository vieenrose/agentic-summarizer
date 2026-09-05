# G2's inversion count is instrument-dominated — measured 2026-09-05

Five-judge panel on `rl-v3` vs the map-reduce baseline, 40 held-out meetings each arm
(`runs/g2-rlv3/`). Two judges complete at the time of writing; three still running.

| judge | agent inversions / claims | baseline inversions / claims |
|---|---|---|
| `deepseek-v4-flash` | 53 / 314 (**16.9%**) | 119 / 724 (**16.4%**) |
| `hy3` | 68 / 314 (21.7%) | 125 / 745 (16.8%) |

**The gate PASSES** — its criterion is `inversions ≤ baseline`, and the agent is well under
on both judges. The direction is trustworthy: both arms are scored by the same instrument
over claim sets that match exactly (0 claim-count mismatches between judges, so both saw
identical text).

**The absolute rate is NOT trustworthy, and should never be quoted as "17% of the agent's
claims are contradicted."** Three reasons, in order of confidence.

## 1. The per-claim rate is flat across arms

Under `deepseek-v4-flash` the two arms land within half a point of each other — 16.9% vs
16.4% — despite failing in completely different ways (the agent starves and under-renders;
the baseline over-asserts from independent per-chunk summaries with no shared memory). Two
systems with different failure modes measuring the same per-claim rate is the signature of a
**noise floor**, not of two independent readings of faithfulness.

It also explains the headline: the agent wins the absolute count because it asserts 314
claims against 724. **It says less.** That is a real property worth reporting — it is the
same caveat recorded for `sft-dropv7`, where the per-claim rate favoured the baseline 4.9%
to 7.3% — but it is not the same thing as being more faithful.

## 2. The panel ran `--votes 1`, against the design

`judge_meeting` defaults to `votes=3` and its docstring says why: the prior project measured
a judge returning SUPPORTED / UNSUPPORTED / SUPPORTED **on identical input**, and concluded
that "a 0% inversion gate cannot rest on a single stochastic call." Every number above rests
on exactly one call per claim.

`cli/judge.py` now warns when `--votes 1` is combined with a real judge, because nothing
did. The four panel commands were launched with it and nothing objected.

## 3. Two independent judges disagree per meeting

Same 79 (system, meeting) pairs, identical claim sets:

| arm | exact per-meeting agreement | mean \|diff\| | max \|diff\| | r |
|---|---|---|---|---|
| agent | 16/40 = **40%** | 0.78 | 2 | 0.76 |
| baseline | 11/39 = **28%** | 1.28 | 6 | 0.61 |

Totals happen to converge for the baseline (119 vs 121) and diverge 28% for the agent
(53 vs 68). Per-meeting, the instrument is close to unreliable; only the aggregate direction
survives.

## The likely mechanism, and the measurement that would confirm it

The judge does not see the transcript. `judge_meeting` retrieves `top_k=6` utterances per
claim and asks for a verdict against those alone. On a 48-chunk meeting that is a very thin
evidence window, and **a retrieval miss is indistinguishable to the judge from an absent
fact** — it can surface as either UNSUPPORTED or CONTRADICTED. Consistent with this,
`unsupported` runs ~3x the inversions (160 vs 53 for the agent under `deepseek`).

**Not yet measured**, and the two things that would settle it:

1. Re-judge a sample at `votes=3` and count how many single-vote CONTRADICTED verdicts
   survive a majority. This quantifies (2) directly.
2. Re-judge at a larger `top_k` (or against the full chunk a claim came from) and see
   whether inversions fall. If they do, the count is measuring retrieval, not faithfulness.

Both need `OPENCODE_API_KEY` in the environment.

**Why this is written down before it is resolved.** SPEC §5.0 requires an instrument to be
validated before its output is used to decide anything, and G2 is the one gate whose
instrument has never been checked against itself. The comparison it supports is still sound;
the number it prints is not yet evidence of a rate.
