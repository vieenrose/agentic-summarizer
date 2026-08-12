# Feedback on `lfm2.5-350m-cursor-en`/`-zh` — evaluated for on-device deployment in VoxSumDroid

**Context:** VoxSumDroid is an offline Android/Linux meeting summarizer already running a
CURSOR-shaped harness in production (a different, earlier training contract — the
`voxsum-qwen35-0.8b` fine-tune). We evaluated the new LFM2.5-350M `pi-agent` checkpoints as
a drop-in upgrade path, specifically because the currently shipped harness checkpoint
measured 33.3% inversions and stayed non-default. Verdict: **not integrated**, for one
disqualifying reason and one blocking-but-fixable one.

## 1. The 0%-inversion result is sweep-dependent, and the sweep needs judges no phone can run

`RESULTS.md`'s own numbers (T1, n=20, paired vs 9B map-reduce baseline) draw a clean
before/after around the VERIFY/ANCHOR sweep:

| | INVERT | FAITH-claim Δ | verdict |
|---|---|---|---|
| **Harness guards only, no sweep** (raw model output — this is what an on-device deployment would actually run) | **12/20 (60%)** | **−1.28** | your own doc: "FAIL as measured" |
| **+ VERIFY/ANCHOR sweep** (`gpt-oss-20b` + `qwen3.6-35B`, 3× majority) | 0/20 | +1.05 | ship gate PASS |

The sweep is the thing that fixes faithfulness, not the fine-tune. `gpt-oss-20b` and
`qwen3.6-35B` are 20B–35B-parameter models — there is no version of "run the sweep on a
phone." So for any on-device consumer (us, and presumably anyone else deploying this
outside a workstation with two RTX 5090s), the honest number to plan around is the
**raw 12/20**, not the headline 0/20. That's worse than what we already ship without CURSOR
at all (6.2% inversions, single-pass fine-tune + app-side chunking).

Two things follow from this that would be useful to see measured:

- **A sweep-free number for the phase-2 (final, 2026-08-12) checkpoint specifically.**
  The only raw/no-sweep figure in `RESULTS.md` (12/20) is from the *pre*-phase-2 model.
  Phase-2's real-data adaptation might have moved the raw number too, not just the
  swept one — right now that's unmeasured, so it's not possible to tell whether phase-2
  helped the model or only helped the model-plus-sweep combination.
- **Whether the sweep's corrections are being fed back into training.** You already
  called this out as the real fix in the pre-phase-2 write-up ("the remaining fix is
  model-side: real-transcript training") — the sweep is essentially a judge doing at
  inference time what SFT should eventually do at training time. If the DROP/FIX outcomes
  from the sweep runs are being harvested as additional training signal (contradiction
  cases, fabrication cases), that's the path to a checkpoint whose *raw* inversion rate
  approaches what the swept rate is now. Worth stating explicitly as the target metric
  for the next iteration: **raw INVERT, not swept INVERT.**

## 2. No GGUF is actually published yet

Both `Luigi/lfm2.5-350m-cursor-en` and `-zh` on HF currently contain only
`model.safetensors` (709 MB, bf16). The `-en` README references
`lfm2.5-350m-cursor-en.Q4_K_M.gguf` (~215 MB) and gives a `llama-server` command line
against it, but that file isn't in the repo tree — looks like the README was written ahead
of the artifact actually being pushed. Not a blocker on its own (we'd have converted +
quantized ourselves), but worth closing out since the card currently promises something
that isn't there yet.

## What would change the verdict

A future checkpoint (or this one, re-measured) that reports a **sweep-free INVERT rate
competitive with the swept one today** — i.e. the model itself, not an external judge,
holding the faithfulness gains — would be worth re-evaluating for on-device deployment.
The CURSOR harness design itself (`CLAUDE.md` §2–§6: one evolving NOTES state, typed
ADD/UPD/DEL/CMP/NOP ops, the deterministic guards in `guards.py`) is sound and something
we'd want to port; it's specifically the judge-dependent half of the faithfulness story
that doesn't travel to a phone.

---

## Response (2026-08-12) — both points addressed

**1. Sweep-free numbers for the phase-2 checkpoint — measured.**

| | INVERT | note |
|---|---|---|
| pre-phase-2, raw | 12/20 (60%) | previously the only raw figure |
| **phase-2, raw** | **4/20 (20%)** | measured 2026-08-12, same judges/protocol |
| phase-2, + sweep | 0/20 | headline |

Phase-2's real-data adaptation moved the model's own rate (12 → 4), confirming the
model-side lever works — but 4/20 still fails 0% without the sweep, so your planning
number for a phone (no judges) is 4/20, not 0/20. Agreed on the target metric: **raw
INVERT for the next iteration**.

**Sweep → training feedback: not yet done; now the stated next step.** DROP/FIX outcomes
are currently judge-time corrections only. Harvesting them as SFT signal (contradiction and
fabrication demonstrations) is the concrete path to a checkpoint whose raw rate approaches
the swept rate — that is the plan for the next iteration, with raw INVERT as the metric.

**2. GGUFs — published.** `lfm2.5-350m-cursor-en.Q4_K_M.gguf` (~215 MB) and
`lfm2.5-350m-cursor-zh.Q4_K_M.gguf` (~229 MB) are now in both repos; the `llama-server`
command in the card works as written.

**Verdict change:** not on the numbers above (raw 4/20 still fails 0% on-device), but the
gap to re-evaluation is now precisely scoped: raw INVERT ≈ swept INVERT. We will re-measure
the raw rate after the sweep-feedback training pass and report it here.

---

## Owner's response (2026-08-12) — verdict recorded

> "Verdict is still 'not integrated,' but the picture changed materially: raw inversions went
> 12/20 → 4/20 (60%→20%) on the phase-2 checkpoint specifically, which is real model-side
> progress, not just sweep tuning. It's still worse than the shipped 6.2%, so this doesn't
> flip the decision yet. But they've now committed to the right target metric going forward —
> raw INVERT, not swept INVERT — with a concrete plan (harvest the sweep's corrections as SFT
> signal). Updated the memory record to reflect this as a live, narrowing gap rather than a
> dead end, and left a note for future-me to watch for the next training pass and re-derive
> any future headline number from raw judge output the same way, rather than trusting it at
> face value."

**Status: open, narrowing.** Bar for re-evaluation: raw INVERT < 6.2% (their shipped rate) —
i.e. the next training pass must move raw INVERT from 4/20 toward ≤1/20. Their discipline
(numbers re-derived from raw judge output, not headline claims) is adopted as ours: **all
future headlines in RESULTS.md report the raw figure first, the swept figure second.**
