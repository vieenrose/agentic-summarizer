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
