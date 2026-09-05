# G2 passes, and the agent is nonetheless LESS faithful per claim — 2026-09-05

Five-judge panel on `rl-v3` vs the map-reduce baseline, 40 held-out meetings per arm
(`runs/g2-rlv3/`). Three judges complete; two still running. **Paired** over meetings both
arms have, since judge coverage differs:

| judge | paired n | absolute inversions (the gate) | per-claim rate | meetings agent fewer / more |
|---|---|---|---|---|
| `deepseek-v4-flash` | 39 | 50 vs 119 — **PASS** | 16.8% vs 16.4% | 28 / 5 |
| `hy3` | 40 | 68 vs 125 — **PASS** | 21.7% vs 16.8% | 27 / 6 |
| `longcat-2.0` | 33 | 57 vs 81 — **PASS** | 23.0% vs 13.9% | 15 / 8 |
| `muse-spark-1.3-contributor` | 40 | 62 vs 79 — **PASS** | 19.7% vs 10.6% | — |

**G2 PASSES: 4 of 4 judges**, already a majority under §5.1's 3-of-5 rule with the fifth
(`mimo-v2.5`) still running. It passes on the absolute count the gate is defined over and on
a paired sign test across meetings.

**And on every judge the agent is WORSE per claim**, by 0.4, 4.9, 9.1 and 9.1 points. The
absolute win is a volume effect: the agent asserts ~2.4x fewer claims (298-314 vs 724-745)
and writes summaries a third as long (350 vs 943 median characters).

*(An earlier draft of this file read the flat `deepseek` rate as a per-claim noise floor
common to both arms. With three judges that reading is wrong: the rates are not flat, they
are ordered, and the ordering is the same every time.)*

## The obvious confound is ruled out

A shorter summary could be made of DENSER claims, each carrying more assertions and so
easier to contradict — which would make the per-claim comparison meaningless. Measured with
the judge's own `split_claims` over both arms' prose:

| | median claim length | mean | claims/meeting | prose chars |
|---|---|---|---|---|
| agent | 43.0 | 45.4 | 7.0 | 350 |
| baseline | 46.0 | 47.7 | 19.5 | 943 |

The agent's claims are marginally **shorter**, not denser. The per-claim rates are
comparable, and the deficit is real.

## It tracks starvation, which is the mechanism SPEC already names

Splitting the agent's own meetings by whether `evalkit.behaviour` flagged them starved:

| judge | starved meetings | healthy meetings |
|---|---|---|
| `deepseek-v4-flash` | 21/116 = **18.1%** (n=17) | 32/198 = 16.2% (n=23) |
| `hy3` | 30/116 = **25.9%** (n=17) | 38/198 = 19.2% (n=23) |
| `longcat-2.0` | 34/109 = **31.2%** (n=16) | 45/198 = 22.7% (n=23) |

Starved meetings invert more per claim on every judge (+1.9, +6.7, +8.5 points). Small n and
a modest effect, so this is directional evidence rather than proof — but it is the mechanism
§4.1 v1.1 already states in its own words: an impoverished input is "the pressure that
produces invention". A model that records 6 points across 37 chunks and is then asked for a
summary has to fill the gap from somewhere.

**Consequence for the work queue: starvation is not only a coverage defect, it is plausibly
a faithfulness defect too**, and the RAFT reading-step work targets the root of both.

## The absolute RATE is still not trustworthy, even though the ORDERING is

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
