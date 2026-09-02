# The G1 instrument was broken in four ways — 2026-08-31

**Before any further G1 fix attempt, read this.** `runs/g1-study.md` records five refuted
attempts and concludes G1 is a corpus limitation. That conclusion may still be right, but
**every one of those attempts was compared on an instrument with the defects below**, and
two of the four affect the numbers directly. The strategy question ("is revision the
problem, or is it general point quality?") was never actually measured, because the arm
that would answer it — the control arm — was never usable.

## 1. The reversal could be visible in the decision's own chunk

`tools/gen_reversals.py::build_gold` located the reversal with `reversed(chunks)`, i.e. the
LAST chunk containing it. Chunks OVERLAP (`OVERLAP_LINES`), so a reversal near a boundary
appears in two chunks; `late_i > early_i` then reads as "a later chunk" while the model —
which sees chunks one at a time, in order — already had the reversal in front of it when it
recorded the decision. That is single-step revision, which G1 is not testing.

**Measured: 2 of 11** probe scenarios (`bikeshare`, `nightmarket`). So every recorded
independent-probe figure — 0/11, 1/11, 2/11 across five attempts — had a wrong denominator.
9 valid scenarios, not 11.

This is the same bug class `tests/test_probe.py` already pinned for `probe_data.py`'s two
hand-written gate cases, and the ~120-token transcripts before that. It recurred because the
pins covered `arcsum.probe` and never `data/reversals_probe`.

## 2. The transcript was written even when the gold was refused

`build_gold` gates the `.jsonl`. The probe split carries no gold, so a refusal left a
structurally useless `.txt` on disk. **5 of 30** regenerated scenarios put the reversal
inside chunk 0 even after defect 1 was corrected. The check now runs in the write path.

## 3. Generator and scorer disagreed on what counts as the outcome being stated

`score_reversals` accepts the bare verb and the 決議-prefixed form (trap 5's lesson: the gate
asks whether the FINAL STATE is reported, not whether one surface form was chosen). The
generator demanded the literal planted string and discarded the meeting otherwise.

**This hit the two arms unequally: 7 of 15 control scenarios refused against 3 of 30 probe
scenarios**, because control outcomes are longer and clumsier to say
(「主席宣布這個案子決議照案核定」) so a writer paraphrases them more often. **A control arm
sampled by a stricter rule than the treatment arm is not a control.** One definition now
lives in `gen_reversals.outcome_variants` and the scorer imports it. `key_term` stays
verbatim-required — it is the thing whose survival the probe measures.

## 4. Nothing partitioned the control arm from the probe arm

The import-time asserts covered TRAIN vs PROBE only. `tools/loss_map.py` compares CONTROL
against PROBE, so a shared subject or key term would let one arm's result leak into the
other's. Asserts added.

**And the control arm was never actually built.** `data/reversals_control` held 2 of 4
scenarios (`floodgate`, `streetlight`); `recycling` and `eldermeal` had failed generation
silently and nobody noticed, because no tool consumed the control arm. It is now 15
scenarios with retries.

## What was actually measured, and the one new number

`tools/loss_map.py` traces `key_term` through three stages. Run against the EXISTING
`runs/qwen-tools-v5/revprobe_report.json` (the old 11-scenario set):

```
                        STRICT / PARTIAL (>=60% contiguous run)
reversal  n=11  emitted 4/8 (36.4% / 72.7%)  memory 2/2 (18.2%)  prose 1/2 (9.1% / 18.2%)
```

**`emitted` is new** — `runs/g1-study.md`'s loss map began at "reaches MEMORY" and so never
measured whether the reading step produced the term at all.

**And the strict/partial split matters more than the new stage does. Do not quote strict
alone.** Read strictly, the model never writes the detail down in 7 of 11 and the loss is
upstream of memory. Read partially, it writes a recognisable form **8 times out of 11** and
then loses 6 of those between emission and final memory. The partial reading is the correct
one: emission is mostly fine, **the deficit is retention**. An early draft of this file
claimed the opposite from the strict column alone; that was wrong, and it was wrong in the
flattering direction — "the model can't even write it down" is a tidier story than "we lose
it somewhere in our own memory pipeline".

Retention loss has three candidate mechanisms, and at 11 meetings × ~2 chunks the cap is not
plausibly one of them (`POINTS_CAP=16` is never approached):

1. the op was REFUSED (`point too long`, `duplicate point`, contradiction guard);
2. a later step `DROP`ped it and the replacement did not carry the term — this is exactly
   `qwen-tools-v6`'s lossy-revision finding, and it is revision-specific;
3. eviction by `spread` — ruled out by the cap arithmetic above.

If (2) dominates, the loss IS revision-specific and the corpus reading in `g1-study.md`
stands. **The control arm is what separates them**, which is why it had to be built before
any of this could be concluded — and why the fact that it never existed (defect 3/4 above)
is the most consequential of the four defects.

`score_reversals` does not currently record per-op apply outcomes, so mechanism (1) cannot
be distinguished from (2) in the existing artifacts. That is the next instrument gap.

## Separately: extending the two G1 gate transcripts changed the test

`probe_data.py`'s two cases were extended today so they still span multiple chunks. They now
run 3 chunks with ~2,500 tokens of unrelated business BETWEEN the decision and the reversal;
previously the two were adjacent. `qwen-tools-v5` FAILs both, but this is not the same FAIL —
the mechanism is now visible and different:

- `office_move` step 1 (intervening business) OVERWRITES the ARC, losing the 搬遷 thread.
- step 2 carries the full reversal (`搬遷`, `B 棟`, `撤回` all present) and the model emits
  **only an ARC update — no `DROP`, no `ADD`**. The reversal is never recorded.

A 2-chunk instrument structurally could not have shown this. Intervening business, not the
reversal itself, is what breaks the chain.

**One fabricated point, and do not over-read it.** `office_move` step 1 emitted
`道路用地變更：南松街301號由I-A UO-3變更為I-A UO-5` — character-trigram containment against
its own chunk **0.00**. `UO-3` appears in the training pool only inside Denver zoning
SYNTHESIS targets, never in a prompt, so this is memorized-output regurgitation. But the
other points in the same step measure 0.35 and 0.48 and their subjects (`社會住宅`,
`營養午餐`, `在地食材`) ARE in the chunk. **1 fabricated point of 8** — real, worth watching
under G2, not the systemic ungroundedness it first looked like. Measure containment before
calling something hallucination; this project has recorded that mistake once already.

---

# RESULT: the loss is now localized, and half of it is ours to fix

Measured on the corrected instrument (27 reversal + 15 control scenarios), `--protocol tool`,
`cache_prompt: false`. Columns are STRICT/PARTIAL. `runs/lossmap/`.

| checkpoint | arm | emitted | memory | prose | passed |
|---|---|---|---|---|---|
| `v5` | reversal (27) | 51.9 / 74.1% | 18.5 / **22.2%** | 11.1 / 22.2% | 3 |
| `v5` | control (15) | 60.0 / 93.3% | 53.3 / **86.7%** | 20.0 / 53.3% | 7 |
| `v6` | reversal (27) | 55.6 / 77.8% | 44.4 / **70.4%** | 14.8 / 44.4% | 4 |
| `v6` | control (15) | 73.3 / 93.3% | 60.0 / **86.7%** | 20.0 / 60.0% | 6 |

## 1. The retention loss IS revision-specific — my reframe was wrong

`v5`: control holds the detail in memory **86.7%**, reversal **22.2%** — a 64-point gap, with
**0 harness refusals** and the loss attributed to `DROP`. The hypothesis that this is general
point quality trainable from MeetingBank is REFUTED. `runs/g1-study.md`'s corpus reading
stands.

Recorded because the instrument was built specifically to test that hypothesis and then
killed it. That is the instrument working, not wasted effort — the alternative was training
on MeetingBank point-quality data and discovering it later.

## 2. `qwen-tools-v6` DOES fix it, and was rejected on the broken instrument

`v6` = `v5` + `late_point` carrying `key_term`. Reversal-arm memory retention **22.2% ->
70.4%**; the control-vs-reversal gap collapses **64.5 -> 16.3 points**. The control arm is
unchanged (86.7% both), which is what confirms the fix is targeted rather than a general
shift.

`runs/g1-study.md` records `v6` as "did not move the gate — probe 0/11 -> 1/11 (noise)". At
n=11, with 2 of those 11 structurally unable to test cross-chunk revision, it could not have
seen a 48-point mechanism change. **The fix was real and the instrument could not resolve it.**

This does NOT by itself make `v6` shippable: its ASR regression (17/20 -> 15/20) is
independent of this measurement and unretested.

## 3. The dominant REMAINING loss is synthesis, and it is general

```
v6 reversal:  memory 70.4% -> prose 44.4%   (-26 points)
v6 control:   memory 86.7% -> prose 60.0%   (-27 points)
```

**Identical in both arms.** It appears with no reversal anywhere in the transcript, so it is
not revision-specific, not a corpus gap, and fully measurable on ordinary MeetingBank
meetings. Synthesis is handed a memory point carrying the identifying detail and writes prose
that paraphrases it away.

This is the `tools/gen_hedge_synth.py` shape exactly — memory correct, prose lossy — and that
intervention (12 rows) fixed its target, recovered both failing G3 gates, and improved ASR
curation at once. **It has never been tried for identifying detail.** It is the highest-value
untried move on G1 and it needs no new corpus.

## Why the pass rate barely moved (3/27 -> 4/27)

The gate is decided on PROSE. Fixing memory retention without fixing synthesis moves the
mechanism and not the score. Both halves are needed — which is what
`runs/g1-study.md` predicted ("no single-point fix can carry this gate") and this now
quantifies: revision was ~48 points of the loss, synthesis is ~27 more.

## The `v6` ASR regression is REAL — reproduced in-session, and `v7` inherits the risk

Both checkpoints re-run tonight on `data/ly_phase3_v2` (same 20 real zh-TW meetings, same
session, same harness), so this is not drift against a stale record:

| checkpoint | curated | NOP | mean chars |
|---|---|---|---|
| `v5` | **17/20** | 28% | 223 |
| `v6` | **15/20** | 33% | 248 |

`v6` writes LONGER summaries and curates FEWER meetings — it abstains more. So the
`late_point` fix that buys +48 points of reversal-arm memory retention costs 2 real-ASR
meetings, and the mechanism (why carrying `key_term` through a revision would raise the NOP
rate on unrelated ASR meetings) is NOT understood.

**`qwen-tools-v7` = `v6` pool + 106 detail-preserving synthesis rows, so it inherits this.**
Any v7 evaluation MUST include `tools/asr_gate.py`; a G1 improvement bought at 2 more ASR
meetings is not obviously a win, and this is exactly the trade `runs/qwen-v2-heldout/RESULT.md`
recorded for `v3`->`v4` and then found to be avoidable once the real cause was understood.

## Pool signal after the detail rows

```
detail retention in synthesis targets:  46.9% (v5/v6)  ->  66.6% (v7)
NOP share:                              33.2%          (healthy; trap 1's floor is the risk)
v7 rows: 4,837 = 4,731 (v6) + 106 (detail)
```

The 106 rows reuse REAL memory states already in the pool, so only the target's retention
policy changes; 13 were refused for preserving <80% of details and 68 memory states were
skipped for carrying fewer than 2.

**A guard caught a real mistake here:** the new rows were tagged `sys-v2`
(`PROMPT_VERSION`) while the tool-call pool is `tools-v1` throughout — including its
synthesis rows, whose completions are prose rather than tool calls.
`train_toolcalls.py` refuses a mixed-version pool and stopped the run before it started.

---

# `qwen-tools-v7`: G1 mechanism fixed, G3 LOST. Not a replacement for `v5`.

Full gate set, `data/heldout_zh` (40 meetings), CHUNK_TOKENS=2500, epoch-2 checkpoint.

| | `v5` | **`v7` ep2** |
|---|---|---|
| independent probe (27) | 3 | **11** |
| prose retention (probe) | 22.2% | **59.3%** |
| control−reversal memory gap | +64.5 | **−4.4** |
| real-ASR curated | 17/20 | 17/20 |
| G3 rouge1 | **PASS** 28/12 +0.069 p=0.017 | **FAIL** 19/21 −0.010 p=0.875 |
| G3 rouge2 | **PASS** 29/11 +0.041 p=0.006 | **FAIL** 23/17 +0.016 p=0.430 |
| G3 rougeL | PASS 35/5 +0.057 | PASS 33/7 +0.035 |
| G1 gate (2 cases) | FAIL | FAIL |

**`v5` remains the recommendation.** `v7` wins the mechanism and loses two gates.

## Why — and a prediction of mine that was WRONG

Before building `gen_detail_synth.py` I checked whether detail preservation would fight
G3, found the references carry 9.6 distinct details against the agent's 5.2, and concluded
the two were aligned because preserving detail moves output TOWARD the reference. **The
check was sound and the conclusion did not follow.** Measured after the fact:

```
reference    chars 558  details 8.2
v5 agent     chars 349  details 4.5
v7 agent     chars 317  details 2.6   <- FEWER details, and shorter
```

`v7` preserves the planted `key_term` far better (probe prose 22% -> 59%) while emitting
**fewer details overall**. Both are true: the training improved RETENTION OF WHAT IT KEEPS
and reduced HOW MUCH IT KEEPS. A probe that measures survival of one planted term cannot
distinguish those, and rewards the first while being blind to the second.

**The transferable lesson: a retention metric and a coverage metric can move in opposite
directions, and the probe only sees retention.** I measured density (details per summary)
and reasoned about it as though it constrained quantity. It does not. Any future synthesis
intervention must report BOTH the probe's retention rate AND mean details per summary on
held-out meetings, or it can look like a win while shedding content.

This is the same shape as `v3`->`v4` (a real capability gain paid for in G3 precision) —
which was later shown to be avoidable once the true cause was found. Whether this one is
avoidable is unknown; the `v4` precedent says do not assume it is fundamental.

## WHERE v7 sheds content: the READING step, not synthesis

Measured directly on 6 held-out meetings (16.7 steps each), both servers, same harness:

```
        applied-ops/mtg   final points   details-in-memory   prose chars   details-in-prose
v5           37.3            10.33            9.67              301             5.67
v7           18.2             5.33            2.17              306             1.67
```

**Prose length is unchanged (301 vs 306). Memory is starved before synthesis runs.** The
106 SYNTHESIS-ONLY rows halved the READING step's applied ops on real meetings and cut
details in memory by 78%. Synthesis is not the culprit; it faithfully renders a much
poorer memory.

Cross-call bleed like this is documented — `runs/qwen-v2-heldout/RESULT.md` records the
reading step improving when only synthesis rows were added for `v5`. **The same mechanism
runs in reverse here, and it is much larger.**

**Why the probe was blind to it.** Probe scenarios are 2 chunks; real meetings are ~17
steps. A per-step conservatism compounds ~8x more on a real meeting than on the
instrument, and the probe's pass criterion (does ONE planted term survive) improves under
"fewer, more carefully-chosen points" — the exact behaviour that loses G3.

On the probe alone, `v7` looks better on every axis:

```
        points/mtg  details-in-memory  prose chars  details-in-prose
v5 rev     1.07           0.04             180          0.19
v7 rev     1.11           0.11             186          0.30
```

Both measurements are correct. They disagree because the instrument's meetings are short.
**Any future probe-driven change must be re-measured on full-length meetings before it is
believed.**

## Next attempt, and the specific reason to expect it to work

`gen_hedge_synth` moved its target with **12 rows**. This build used **106**, which is 36%
of all synthesis rows in the pool — a far heavier thumb on the scale, and the reading-step
damage scales with it. The obvious next build is the same intervention at ~24 rows,
measuring BOTH:

- probe prose retention (the thing being bought), and
- **applied-ops/meeting and details-in-memory on held-out meetings** (the thing being
  paid), which nothing measured before tonight.

Do NOT rerun at 106 rows expecting a different outcome.

---

# Training configuration dominates: LoRA + a longer LR horizon nearly doubles G1

All rows below: same pool (`sft_pool_tools_v5.jsonl`), same 4,731 examples, same effective
batch 16, same seed, same probe/control instrument, best checkpoint by eval loss.

| config | 0.8B probe | 2B probe | 0.8B ctl | 2B ctl |
|---|---|---|---|---|
| full FT, 2-epoch cosine | 5/27 | 6/27 | 6/15 | 9/15 |
| **LoRA r32, 6-epoch cosine (best = ep2)** | **9/27** | **12/27** | **7/15** | **10/15** |

`2B LoRA` at **12/27 beats `qwen-tools-v7`'s 11/27** — and v7 needed the deliberation,
hedge AND detail-synthesis rows to get there, while this is the plain v5 pool.

## The LR horizon alone moved the probe 4.5x

Two 0.8B LoRA runs, IDENTICAL except `--epochs`, both taking their epoch-2 checkpoint:

| schedule | best eval loss | probe |
|---|---|---|
| cosine over 592 steps (2 epochs) | 0.7530 | **2/27** |
| cosine over 1,776 steps (6 epochs) | 0.7486 | **9/27** |

Same method, same epoch, same data. The 6-epoch run's epoch-2 checkpoint sits earlier on
the decay curve, i.e. at a higher LR. **A 0.004 difference in eval loss accompanies a 4.5x
difference in probe pass rate** — eval loss is unusable for checkpoint selection here.

Both sizes bottom at epoch 2 and rise monotonically after (0.8B: .7596 .7486 .7545 .7844
.8234 .8415), so this is not an under-training artifact; it is the schedule.

## Consequences for earlier conclusions

1. **"G1 is not capacity-limited (+1/27 for 2.5x params)" is REVISED.** That was measured
   under the full-FT 2-epoch schedule, which suppressed both arms. Under the LoRA 6-epoch
   schedule the same size step gives **+3/27** (9 -> 12). At n=27, p~0.4, binomial SD ~2.5,
   so +3 is ~1.2 sigma — suggestive, not conclusive. The corpus-limit claim is weakened,
   not overturned.
2. **Method and schedule were UNCONTROLLED across every comparison in this project**,
   including the v5 -> v6 -> v7 sequence the current recommendation rests on. Those were
   all full-FT at a fixed schedule, so they are internally consistent — but their absolute
   levels are far below what configuration alone can reach.
3. **Data work may have been the smaller lever.** v7's pool additions bought 3 -> 11/27
   under one configuration; changing configuration alone buys 5 -> 12/27 on the ORIGINAL
   pool. Both matter, but the ordering is the opposite of what was assumed.

**Do not read the absolute numbers as ceilings.** The 6-epoch horizon is better than the
2-epoch one at both sizes; nothing establishes it is optimal. The schedule is now the
largest known unexplored axis.

## Data work and configuration work do NOT compose — they reach the same ceiling

0.8B, LoRA r32, 6-epoch cosine, best checkpoint, same instrument. Only the POOL differs:

| pool | rows | best eval loss | probe | control |
|---|---|---|---|---|
| v5 (plain) | 4,731 | 0.7486 | **9/27** | 7/15 |
| v7 (+detail +deliberation +hedge) | 4,837 | **0.7412** | **8/27** | 7/15 |

8 vs 9 at n=27 is noise: **no detectable G1 gain from the enriched pool once the training
configuration is right.** The v7 additions took the probe 3 -> 11/27 under the OLD
(full-FT, 2-epoch) configuration, so they were a route to the same ceiling rather than an
independent gain.

Fourth instance tonight of eval loss disagreeing with the gate: the v7 pool has the LOWER
loss and the (marginally) worse probe.

**This does NOT make the pool work inert.** Those rows were built for things the reversal
probe does not measure and which remain measured elsewhere: real-ASR curation 9/20 ->
16/20, the deterministic synthesis negation bug, and prose detail retention. The claim is
narrow — they do not add to G1 on top of a good configuration.

**Effort-allocation lesson.** The data interventions were the expensive path to this
number; LR horizon + adaptation method reached it alone, on the original pool. Neither was
searched earlier because every prior run in this project held them fixed, which made them
invisible as variables rather than known-good choices.

## SCALE IS THE LARGEST LEVER — the "not capacity-limited" conclusion is OVERTURNED

Identical configuration across sizes: v5 pool, LoRA r32/alpha64, 1,776-step cosine horizon,
effective batch 16, same seed, best checkpoint by eval loss, same probe/control instrument.

| size | probe (27) | control (15) | best eval loss | best epoch |
|---|---|---|---|---|
| 0.8B | 9 | 7 | 0.7486 | 2 |
| 2B | 12 | 10 | 0.6789 | 2 |
| **4B** | **17** | **13** | **0.6061** | **1** |
| 9B | pending | | | |

9 -> 12 -> 17 monotonic. At n=27 the 0.8B-vs-4B gap is ~3 sigma: real, not noise.

**This overturns the earlier finding in this file that G1 is not capacity-limited.** That
was measured full-FT at a 2-epoch cosine horizon (5/27 vs 6/27 for 0.8B vs 2B) — a
configuration now known to suppress both arms. Under a proper schedule, 5x the parameters
nearly doubles the probe, and 4B on the PLAIN v5 pool (17/27) beats every data intervention
made in this project (v7's 11/27 required detail + deliberation + hedge rows).

Ranking of levers, all measured on the same instrument:

| lever | effect on probe |
|---|---|
| model size 0.8B -> 4B | **9 -> 17** |
| LR horizon 2-epoch -> 6-epoch cosine (0.8B) | **2 -> 9** |
| pool v5 -> v7 enrichment (good config) | 9 -> 8 (none) |
| pool v5 -> v7 enrichment (old config) | 3 -> 11 |

**Product implication, and it is not "use a bigger model".** G4 is measured at 19.0 min
against a 20.00 min ceiling for 0.8B Q8 on the Reno 7. A 4B is ~5x the weights and fails
the latency budget outright. So the DEPLOYABLE model is capacity-limited, which makes this
a SPEC 6 question about device or latency budget — not something more supervision fixes.

Best epoch is NOT constant across scale (0.8B/2B bottom at epoch 2, 4B at epoch 1), so
these are best-of-schedule per size rather than same-epoch comparisons. That is the right
choice but should be stated.

## Two silent export failures, both now guarded

`BASE_SNAP` was empty for the 4B export, so `${BASE_SNAP:-default}` fell back to the 0.8B
snapshot and injected `[1024, 2048]` MTP tensors into a 4B model ([2560, 5120]). **The
export reported success.** Worse, the script writes merged weights back into its INPUT
directory, so the corrected re-run then saw 15 `mtp.*` tensors already present and skipped
the restore — a second failure layered on the first, which is why the artifact had to be
stripped before re-exporting.

`export_gguf.sh` now validates the restored MTP hidden dim against a tensor taken from the
model itself and aborts on mismatch. The in-place modification of the source directory
remains a design wart worth fixing.

## The 4B is NOT usable as a teacher: best probe, failing G3

`runs/4b-heldout/`, 40 held-out meetings, agent vs the fair map-reduce baseline at the
SAME model (SPEC 5.2).

| gate | 4B LoRA e6 |
|---|---|
| G3 rouge1 | **FAIL** 19/21, -0.002, p=0.875 |
| G3 rouge2 | **FAIL** 19/21, -0.006, p=0.875 |
| G3 rougeL | PASS 28/12, +0.021 |
| coverage | against, 12/28, p=0.017 |
| density | against, 10/30, **-0.676**, p=0.002 |

Its content on 5 held-out meetings: **210 chars, 0.6 details** against references at
558 chars / 8.2 details.

**Distilling this model would transfer its sparsity.** Generation of a 200-meeting
distillation pool was started and STOPPED before this was measured; had it run, the student
would have been trained to imitate a teacher that loses to the baseline on 2 of 3 G3 gates.
Check a proposed teacher on the SHIPPING gate, not on the diagnostic probe.

## The central finding: probe score and summarization quality are DISSOCIATED

| checkpoint | probe | prose chars | details | G3 |
|---|---|---|---|---|
| `v5` full-FT 2-ep | 5/27 | 301 | 5.67 | **3/3 PASS** |
| `v7` | 11/27 | 317 | 2.6 | 1/3 pass |
| 0.8B LoRA 6-ep (v5 pool) | 9/27 | **133** | **0.2** | not run |
| 0.8B LoRA 6-ep (v7 pool) | 8/27 | 229 | 2.4 | not run |
| 4B LoRA 6-ep (v5 pool) | **17/27** | 210 | 0.6 | **1/3 pass** |
| reference | — | 558 | 8.2 | — |

The reversal probe rewards terse, decision-focused output and is STRUCTURALLY BLIND to
coverage loss. Every configuration that improved the probe tonight did so while shedding
content, and the one checkpoint that passes all three G3 gates (`v5`) has the worst probe.

**Standing requirement: no configuration or data change is "better" on the probe alone.
Report probe AND mean prose chars/details on held-out meetings, or preferably G3 itself.**
This was violated three times in one session — the v7 pool ("no gain", when it nearly
doubled content), the LR horizon ("largest lever", when it halved content), and the scale
sweep ("capacity is the answer", when the 4B fails G3).

## The probe/G3 trade-off is REAL: four checkpoints, three unrelated routes, one curve

| checkpoint | how it got there | probe | rouge1 | rouge2 | rougeL | density delta |
|---|---|---|---|---|---|---|
| `v5` | full-FT, 2-ep, v5 pool | 5/27 | **PASS** +0.069 | **PASS** +0.041 | PASS +0.057 | — |
| `v7` | + detail/deliberation/hedge rows | 11/27 | FAIL -0.010 | FAIL +0.016 | PASS +0.035 | -0.570 |
| `lr2e-4` | LoRA r32, **lr 2e-4**, v7 pool | 13/27 | FAIL **-0.030** | FAIL +0.011 | PASS +0.037 | -0.282 |
| `4B` | 5x parameters | 17/27 | FAIL -0.002 | FAIL -0.006 | PASS +0.021 | -0.676 |

Everything above ~10/27 on the probe fails rouge1 AND rouge2; the only checkpoint passing
all three G3 gates has the WORST probe. Reached by data enrichment, by parameter scale, and
by LoRA hyperparameters — three mechanisms with nothing in common. **When independent routes
land on the same curve, the trade is a property of the objective, not of any one model.**

`lr2e-4` is the least content-degraded of the high-probe group (density -0.282 vs -0.570 and
-0.676; coverage not significantly down at all) and still has the WORST rouge1 delta. So this
is NOT simply "terser output scores worse" — the summaries are differently SHAPED.

### The hyperparameter finding that produced it

Every LoRA run before this used `r=32, alpha=64, lr=5e-5` — the FULL-FINE-TUNE learning
rate, never varied. Holding the v7 pool and 6-epoch horizon fixed:

| change | best loss | probe | prose chars | details | points |
|---|---|---|---|---|---|
| reference r32 / 5e-5 | 0.7412 | 8/27 | 229 | 2.4 | 5.60 |
| r64 / alpha128 | **0.7367** | 10/27 | 238 | 2.4 | 8.00 |
| **lr 2e-4** | 0.7495 | **13/27** | 233 | 2.6 | **9.00** |

`lr=2e-4` gives the best probe with the WORST loss of the three, and bottoms an epoch
earlier — a fifth instance tonight of eval loss failing to predict the gate.

This also retires the "LR horizon is a mysterious large lever" claim: the 6-epoch cosine was
buying effective learning rate by accident. LoRA was simply under-trained at the FFT LR.

### Consequence for the distillation plan

**There is currently no checkpoint worth distilling from.** A teacher must be good at BOTH;
`lr2e-4` would transfer a rouge1 failure, `v5` would transfer a 5/27 probe. The plan is
blocked on finding a checkpoint off this curve, not on generation capacity.

**Open design question, now the central one:** are G1 and G3-rouge1 genuinely opposed for
this task, or is the reversal probe measuring something narrower than "handles revision
well"? It rewards stating the late outcome crisply, which on a real meeting may amount to
writing less about everything else.

## THE TRAINING PATH ITSELF DEGRADES SUMMARIZATION — and it confounds most of tonight

Same recipe (v5 pool, full FT, 2 epochs, lr 5e-5, effective batch 16), same instrument:

| checkpoint | path | G3 | density delta |
|---|---|---|---|
| `v5` 0.8B | **VL repo + plain Trainer** | **3/3 PASS** | — |
| `qwen08b-diag` 0.8B | text-only repo + unsloth | **0/3 FAIL** | **-1.319** |
| `qwen2b-diag` 2B | text-only repo + unsloth | 0/3 FAIL | -1.105 |

**Same size, same recipe, different path: 3/3 PASS vs 0/3 FAIL.** The 2B was not worse
because it was bigger — the path is the variable. Density collapses by more than a full
point and the agent loses coverage on 36 of 40 meetings.

### This weakens the "probe/G3 trade-off curve" conclusion recorded above

Of the four checkpoints on that curve, THREE (`lr2e-4`, `4B`, and the diag runs) came from
the compromised path. Only `v5` (3/27 probe, 3/3 G3) and `v7` (11/27, 1/3 G3) were trained
the way the shipping checkpoint was. **Two points, not four.** The trade may be real; the
evidence for it is far thinner than claimed, and every LoRA/scale number from tonight is
non-comparable to `v5`.

Also corrected here: `v5`'s own independent-probe score is **3/27**, not the 5/27 quoted
earlier — that figure was `qwen08b-diag`, the reproduction, i.e. a DIFFERENT model on the
bad path.

### Consequence: `--lora` now works without unsloth

`tools/train_toolcalls.py --lora` (no `--unsloth`) uses peft on the plain-Trainer path, so
LoRA results can be compared to `v5`/`v7`. Costs: ~7h45m for 1,818 steps vs ~1h under
unsloth. Gradient checkpointing cannot be traded away to recover it — disabling it OOMs
even at 1.7% trainable parameters, because the memory is dominated by the 248k-vocab logits
and their gradient, which LoRA does not shrink.

unsloth CANNOT load the VL repo at all: it routes to the vision processor and fails with
"Incorrect image source ... Got <|im_start|>system". So CLAUDE.md's original
"unsloth cannot train it" was correct FOR THE VL REPO, and the retraction was correct for
the text-only weights. Both statements are true; they are about different checkpoints.

---

# CORRECTION: it was never the training path — it is UNDER-TRAINING, and best-by-loss selects it

The section above ("THE TRAINING PATH ITSELF DEGRADES SUMMARIZATION") is **WRONG**. Left in
place because the reasoning error is the lesson.

## What was actually different

`v5` and `qwen08b-diag` differ in two ways, not one, and I attributed the gap to the wrong one:

| | `v5` | `qwen08b-diag` |
|---|---|---|
| path | VL repo + plain Trainer | text-only + unsloth |
| **checkpoint shipped** | **epoch 3 (LAST)** | **epoch 1 (BEST by eval loss)** |

`v5` shipped its last epoch only because of the `load_best_model_at_end` bug. The
comparison was a 3-epoch model against a 1-epoch model.

**The weights are exonerated.** The VL repo stores 18 `linear_attn.norm.weight` tensors in
float32 and the text-only repo in bf16 (max diff 0.00390625 = 2^-8, 0.4% relative). But
`train_toolcalls.py` loads with `dtype=torch.bfloat16` on BOTH paths, and after that load
**0 of 24 layers differ**. The stored-precision difference cannot affect training.

## The decisive measurement

Same run, same weights, same path — only the checkpoint differs:

| `qwen08b-diag` checkpoint | prose chars | details | points | G3 |
|---|---|---|---|---|
| epoch 1 (BEST by eval loss) | sparse | — | — | **0/3 FAIL**, density -1.319 |
| **epoch 2 (LAST)** | **387** | **5.0** | **11.20** | (running) |
| `v5` epoch 3 (LAST), for reference | 338 | 5.8 | 10.40 | 3/3 PASS |

The last-epoch checkpoint of the "bad path" run matches `v5` on content. unsloth is not the
problem.

## Best-by-eval-loss selects an UNDER-TRAINED model, every time

| checkpoint | selection | G3 |
|---|---|---|
| `v5` | **last** epoch of 3 | **3/3 PASS** |
| `qwen08b-diag` best | epoch 1 of 2 | 0/3 FAIL |
| `qwen2b-diag` best | epoch 1 | 0/3 FAIL |
| `vl-lora-lr2e4` best | epoch 1 | 0/3 FAIL |
| `lr2e-4` best | epoch 1 | 1/3 |
| `4B` best | epoch 1 | 1/3 |

**Every checkpoint selected by best-eval-loss fails G3. The only one that passes was
selected by a BUG.** The `load_best_model_at_end` defect recorded earlier tonight was
LOAD-BEARING, and "fixing" it made every subsequent checkpoint worse.

## What this does to the trade-off curve

The "probe vs G3" curve may largely be **training amount** in disguise: fewer epochs give
terse, decision-focused output (good probe, bad ROUGE); more epochs give richer content
(good ROUGE, worse probe). Every high-probe/low-content checkpoint tonight was a
best-by-loss epoch-1 model. That is a far more actionable story than an intrinsic conflict.

## Standing change

**Select checkpoints by the GATES, not by eval loss.** Six separate disagreements between
loss and gate were recorded tonight; this is the seventh and the most costly, because loss
was being used as the selection rule rather than merely as a report.

## CONFIRMED, and it yields a checkpoint that beats `v5`

`qwen08b-diag`, LAST epoch (`checkpoint-592`), on the 40 held-out meetings:

| gate | result |
|---|---|
| G3 rouge1 | **PASS** 30/10, +0.030, p=0.002 |
| G3 rouge2 | **PASS** 31/9, +0.037, p=0.001 |
| G3 rougeL | **PASS** 37/3, +0.052, p=0.000 |
| independent probe | **6/27** |
| control | 6/15 |

versus `v5`: probe **3/27**, G3 3/3 PASS. **Double the probe, same three gates, same 0.8B
size (so G4 is unchanged).** The G3 margins are solid; 6 vs 3 at n=27 is ~1 sigma, so the
G1 improvement is suggestive rather than established. **G2 has NOT been measured on it and
is required before it can replace `v5`.**

### The selection rule that falls out

NOT "always take the last epoch" — `lr2e-4`'s loss reached 0.9177 by epoch 4, genuinely
overfit. The rule is that **eval loss bottoms roughly one epoch BEFORE the gates peak**:

| checkpoint | epochs | selection | G3 |
|---|---|---|---|
| `v5` | 3 | last | 3/3 PASS |
| `qwen08b-diag` | 2 | last | **3/3 PASS** |
| everything else tonight | 2-6 | best by loss (ep1) | 0/3 or 1/3 |

Train 2-3 epochs, take the LAST checkpoint, validate on the gates. Do not use eval loss to
select.

### Four wrong theories for one symptom, in order

The sparse-output symptom was explained wrongly three times before this:

1. "the pool's synthesis targets drop detail" — real (46.9%) but not the cause
2. "training configuration dominates" — the LR horizon was buying effective LR, not the cause
3. "the training path (unsloth/text-only) degrades summarization" — weights bit-identical
   after the bf16 load; not the cause
4. **under-training, selected by best-eval-loss** — confirmed by the same run's two
   checkpoints differing 0/3 vs 3/3

Each earlier theory was supported by a real measurement and was still wrong, because each
compared checkpoints that differed in more than one way. The discipline that would have
caught it sooner: **enumerate every difference between two artifacts before attributing a
gap to one of them.**

---

# THE SHIPPING PICTURE, resolved: `diag`-last replaces `v5`

All 0.8B, so G4 is unchanged. 40 held-out meetings, same instrument throughout.

| gate | `v5` | **`qwen08b-diag` last epoch** |
|---|---|---|
| G1 independent probe | 3/27 | **6/27** |
| G2 faithfulness | PASS — 16 inv / 283 claims = **5.7%**, baseline 4.0% | **PASS — 11 inv / 272 claims = 4.0%, baseline 5.5%** |
| G3 rouge1 | PASS +0.069 | PASS +0.030 |
| G3 rouge2 | PASS +0.041 | PASS +0.037 |
| G3 rougeL | PASS +0.057 | PASS +0.052 |

**G2 is the headline.** Every prior report carried the caveat that the agent's PER-CLAIM
inversion rate was WORSE than the baseline's (5.7% vs 4.0%) and that it won on absolute
count partly by saying less. `diag`-last wins on both axes: fewer inversions (11 vs 16) at
a comparable claim count, and 4.0% against the baseline's 5.5%. That caveat is now retired.

## The 106 detail rows: a clean, isolated trade

`v7` = `v5` + **106 detail rows and nothing else** (the reversal and deliberation rows were
already in `v5`). So the following is a one-variable ablation at identical LR, length,
selection, size and seed:

| detail rows | probe | control | G3 |
|---|---|---|---|
| 0 | 6/27 | 6/15 | **3/3 PASS** |
| 24 | 6/27 | 9/15 | (running) |
| 106 | **12/27** | 10/15 | **1/3** (rouge1 -0.045, rouge2 -0.001) |

**2.2% of the pool doubles G1 and breaks rouge1/rouge2.** The response is threshold-like,
not graded: 24 rows move the control arm but not the probe at all.

This also corrects a correction. I recorded that the earlier "v7 pool adds nothing" result
was an under-training artifact. Half true: under-training WAS hiding the probe gain
(8/27 -> 12/27 once selection is fixed), but the G3 cost was real the whole time.


## CORRECTION: that ablation was CONFOUNDED — FFT vs LoRA, not the pool

The table above compares `diag` (0 rows) against `v7pool-e2` (106) and `det24` (24) as if
the pool were the only variable. It is not:

```
qwen08b-diag : "Using bfloat16 full finetuning"   <- FULL FINE-TUNE
v7pool-e2    : "Switching to 16bit LoRA"          <- LoRA
det24-e2     : "Switching to 16bit LoRA"          <- LoRA
```

WITHIN LoRA the pool effect runs the OTHER way: 106 rows gives probe 12/27 and G3 1/3,
24 rows gives probe 6/27 and G3 **0/3** (rouge1 -0.076, worse than either neighbour). More
detail rows is better on both axes there, which is the opposite of what was claimed.

The G3 3/3 PASS belongs to FULL FINE-TUNING, not to "0 detail rows". Across everything
measured:

| G3 result | adaptation |
|---|---|
| `v5` 3/3 PASS | full FT |
| `diag`-last 3/3 PASS | full FT |
| `lr2e-4`, `vl-lora-lr2e4`, `v7pool-e2`, `det24` all 0-1/3 | **LoRA** |

So the live hypothesis is **LoRA does not reach the content behaviour full fine-tuning
does**, and the pool is a second-order effect on top of that.

**This is the fifth time in one session that a gap was attributed to the wrong variable**
because the compared artifacts differed in more than one way. The recurring fix is the same
and I keep not applying it: enumerate EVERY difference between two runs before attributing
anything, mechanically, from the logs -- not from memory of what was intended.

## The ASR gate catches what the clean-text gates cannot

`diag`-last on `data/ly_phase3_v2` (20 real zh-TW legislative meetings):

| | curated | NOP | mean chars |
|---|---|---|---|
| `v5` | **17/20** | 28% | 223 |
| `diag`-last | **12/20** | **54%** | 161 |

It wins G1 (6/27 vs 3/27), wins G2 (4.0% vs 5.7% per claim) and holds G3 3/3 -- all on
MeetingBank-derived CLEAN text -- and loses five meetings on the only real-world
instrument, abstaining on more than half the chunks.

**This is the regression CLAUDE.md was written about**: "v0 regressed from 17/20 to ~7/20
across three checkpoints and nothing caught it, because every gate since Phase 3 runs on
clean text." It was one report away from being called a new champion on G1/G2/G3 alone.

Likely cause is training amount, and it matches the night's other findings: `v5` trained
**3 epochs**, `diag` trained **2**. Higher NOP and shorter output is the under-training
signature; clean-text gates tolerate it and noisy real ASR does not.

**Standing rule, reaffirmed the hard way: no checkpoint is a candidate until
`tools/asr_gate.py` has run on it.** The clean-text gates cannot see this failure mode.

---

# SYNTHESIS COLLAPSES ABOVE 12 MEMORY POINTS — found from user demo logs

`qwen-tools-v5`, synthesis in isolation, fixed seed, identical ARC, points added one at a
time from a real run's memory:

| points in memory | prose length |
|---|---|
| 2 | 102 chars |
| 4 | 135 |
| 6 | 189 |
| 9 | 361 |
| **12** | **544** |
| **13** | **116** |
| 14 | 77 |
| 15 | 88 |

**A 4.7x collapse from ONE additional point.** Reproducible at `temperature=0`. Confirmed
independently by three user demo logs, where prose length tracks POINT COUNT and not
meeting length: 3 points -> 258 chars, 6 points -> 588 chars, but 15 points -> 55 chars,
and the 6-point meeting had MORE chunks than the 15-point one.

**Past the cliff it also fabricates.** At 14 points it invented a surname
(`佩爾克·羅森塔爾`); at 15, in the user's log, it reported a committee member as
`已故` (deceased) when memory said only "reappoint Perk". Memory was correct in both cases;
the invention is entirely at the synthesis step.

## It is NOT a data problem — the pool teaches the opposite

| memory size | share of synthesis training rows | mean TARGET length |
|---|---|---|
| <=12 points | 41.7% | 355 chars |
| **>=13 points** | **58.3%** (86 rows at exactly 16) | **641 chars** |

The pool is well populated at high occupancy and teaches "more points -> LONGER summary".
The model inverts that above 12. So this is a generalisation failure of the 0.8B student at
high memory occupancy, not a gap in supervision.

## Why it matters more than any gate number

`POINTS_CAP` is 16, and a meeting of more than ~4 chunks routinely fills memory past 12. So
**the failure mode is the common case, not an edge case**: the longer and richer the
meeting, the less informative the summary — the exact inverse of what the product promises.
No G1/G2/G3 number exposes it, because ROUGE against a reference does not notice that the
summary used 2 of 15 available facts.

## Mitigation available without retraining

Lowering `POINTS_CAP` from 16 to 12 keeps synthesis on the good side of the cliff.
`spread()` evicts evenly and keeps endpoints, so early and late content both survive.
Measured trade: a 12-point memory yields 544 chars covering 12 facts, against a 15-point
memory yielding 88 chars covering 2. Losing 3 points from memory to gain 10 in the summary
is strongly favourable, but it IS a spec-visible change (`POINTS_CAP` is normative) and
should be measured on G3 before adopting.

**Credit where due: this came from a user reading the debug export, not from any gate.**

## Fourth demo log: the ARC language floor causes topic-level confabulation

Real zh-TW ASR meeting (DRAM supply chain), the demo's own example. Three failures:

**1. 5 of 6 chunks returned NOP.** Dense commercial content -- capacity numbers, pricing,
customer relationships -- recorded as nothing. This is the documented ASR/deliberation gap
("the model requires an explicit STATED OUTCOME"), now visible on the demo's own example.

**2. EVERY ARC was refused by the language guard**, so memory held no ARC for the whole run:

```
insufficient zh-TW content (0.66 < 0.7 CJK ratio)
```

The ARC was checked at `MIN_CJK_RATIO_PROSE` (0.70). The same text is acceptable as a POINT
at 0.35. Real technical meetings carry product names (`Xingrui`, `D4`, `AVL`), so the ARC
was unreachable for this entire domain.

**3. Synthesis invented a different meeting.** From 4 points about DRAM pricing it produced
a summary about 醫療器材 (medical devices) under 《長灘市市政法規》 (Long Beach municipal
code), citing ordinances 125097/125098/125099. Nothing in memory or the transcript. This is
confabulation from MeetingBank training priors, not distortion of the input.

### (2) CAUSES (3) — verified, not inferred

Same four points, synthesis re-run with and without an ARC:

| | length | fabricated-topic markers |
|---|---|---|
| no ARC (what happened) | 316 chars | **醫療, 條例** |
| with ARC | 343 chars | **none** |

Without a narrative anchor the model falls back on training priors. With one it stays on
topic. (It still invents details -- a company name, a year -- so the ARC fixes the TOPIC
failure, not all fabrication.)

### Fix

`MIN_CJK_RATIO_ARC = 0.50`, between the point floor (0.35) and the prose floor (0.70).
The ARC is INTERNAL MEMORY STATE; SPEC §3's zh-TW guarantee is about the SUMMARY and is
still enforced at 0.70 in `prose.finalize`. A majority-Latin ARC is still refused, pinned
by a negative-control test.
