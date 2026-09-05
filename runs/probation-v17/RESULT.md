# The probation question: G3 cannot decide it — 2026-09-04

SPEC §5.2 places the agentic architecture on probation and says to **retire it** if it does not
beat its own map-reduce baseline. That verdict rested on a single G3 comparison. Re-run
properly, **G3 gives opposite answers on the same systems and the same meetings depending on
how long the reference happens to be**, so it cannot support a retirement decision.

## The reversal

`v17-e3` against its own baseline (same model, same chunking, SPEC §5.2's fairness rule), all
40 held-out meetings, both reference sets reachable (0% ungrounded) and both containing 25%
meetings above `POINTS_CAP`:

| references | median chars | agent wins | mean Δ rouge1 | verdict |
|---|---|---|---|---|
| **verbose span** | 702 | 7 / 33 | **−0.103** (p = 0.000) | G3 rouge1/2/L all **FAIL** |
| **terse span** | 273 | 29 / 11 | **+0.049** (p = 0.006) | G3 rouge1/2/L all **PASS** |

Both directions are significant. Nothing about the systems changed between the rows.

## Why: F1 reweights two stable facts as reference length moves

| | precision | recall | F1 |
|---|---|---|---|
| verbose refs | agent **38 / 2** (+0.124) | baseline **37 / 3** (−0.259) | baseline 33 / 7 |
| terse refs | agent **38 / 2** (+0.157) | baseline **35 / 5** (−0.217) | agent 29 / 11 |

**Precision and recall are perfectly stable across both sets**, every p = 0.0000. Only F1
flips. The length ratios explain it exactly: the agent goes 0.55x -> 1.41x of the reference and
the baseline 1.30x -> 3.35x, so the harmonic mean swings from punishing the agent's brevity to
punishing the baseline's verbosity.

**The finding under the metric is a design tradeoff, not a quality ranking: the agent is
consistently MORE PRECISE, the baseline consistently MORE COMPLETE.** That is what the
architecture was built to do — curate rather than compress — and no single F1 number can
express it.

## Consequences

1. **G3 is WITHHELD for the ship decision** (SPEC §5.2.4 rule 3: a conclusion that reverses
   between reference sets is a measurement of the references). It is reported as
   precision/recall/F1 with length ratios, never as a scalar verdict.
2. **The retirement clause cannot be executed on G3 evidence.** Retiring the architecture is a
   one-way decision and the gate that would trigger it is not decidable as written.
3. **The decision needs the reference-free instruments** — grounding, retention, churn — which
   do not depend on a reference's length at all. On those, the current picture is: `v18-e3`
   churn 3.6% with retention 0.908, and the GRPO policy `rl-v2` retention **0.948** at 5.3%
   ungrounded, the best faithfulness/coverage pair yet measured.
4. **The original "0 wins in 25 meetings" is now fully explained.** It used references that were
   46.5% unreachable, a grounding instrument biased against fluent zh, AND a meeting set
   containing zero meetings above `POINTS_CAP` — measured, all three, this session. It was
   never evidence about the architecture.

## What would decide it

A metric that does not move with reference length. Candidates, in order of cheapness:

* **The reference-free pair already implemented** — grounded specifics per meeting (coverage of
  real content) against ungrounded rate (fabrication). Both arms, same corpus, no reference.
* **Length-controlled ROUGE** — score both arms truncated to the same budget, so the
  precision/recall balance is not set by an accident of the reference builder.
* **The §5.1 human slice**, which is the only check not downstream of a model in this pipeline
  and has still never been run.
