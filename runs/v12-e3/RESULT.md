# `v12-e3` — journal-shaped SYNTHESIZE supervision, 2026-09-03

**VERDICT: do not ship. `v12-e3` is a large, paired, significant CHURN regression, and the
mechanism is identified — the coverage gate that was supposed to prevent under-rendering
taught redundancy instead.**

On `data/heldout_zh` (40 meetings, the corpus where the journal actually engages):

| | `v11-e3` ep2 | `v12-e3` ep2 |
|---|---|---|
| clean meetings | 13/40 | **4/40** |
| churn rate | 3.5% | **29.8%** |
| churn events | 23 | **197** |
| meetings under-rendering | 5 | **15** |
| ungrounded | 3.0% of 197 | 6.4% of 267 |
| meetings starved | 12 | 6 |
| retention (G5) | 0.837 | 0.921 |
| specifics | 197 | 267 |

Paired sign test on per-meeting churn: worse on 27 meetings, better on 1, tied 12,
p = 2.2e-07. 29.8% is the regime that got `mixed-e3` publicly rolled back.

## That p-value overstates the certainty — SEED VARIANCE IS THE DOMINANT TERM

Both pools were retrained at seed 1 and re-measured on the same corpus:

| pool | seed 0 | seed 1 |
|---|---|---|
| `v11` (pre-journal synthesis) | **3.5%** churn, 13/40 clean | **13.3%** churn, 5/40 clean |
| `v12` (journal synthesis, no dedup) | **29.8%** churn, 4/40 clean | **17.0%** churn, 6/40 clean |

**`v11-e3`'s 3.5% was the lucky seed.** Changing nothing but the seed moves v11's churn by a
factor of ~4 and its clean count from 13/40 to 5/40 — a swing as large as the one attributed
to the supervision change.

**The p-value was answering the wrong question.** A paired sign test over meetings measures
whether the difference is consistent ACROSS MEETINGS for one pair of checkpoints. It says
nothing about whether a re-trained pair would reproduce it, and the run-to-run term here is
larger than the effect being tested. Quoting p = 2.2e-07 for "v12 is worse than v11" treated a
single training run as the population.

**What survives.** v12 is higher than v11 on both seeds (17.0 / 29.8 against 3.5 / 13.3) and
the ranges do not overlap, so a real effect is likely and the redundancy mechanism below is
independently measured in the DATA rather than inferred from the checkpoints. But with n = 2
per arm this is suggestive, not decisive.

**Standing methodological consequence: a single-seed comparison of two fine-tunes on this
setup is not evidence.** Churn varies ~10 percentage points across seeds at fixed data. Every
prior single-run A/B in this project's history — including the ones that motivated retrains —
is subject to the same caveat, and training costs only ~35 minutes, so replicates are
affordable and should be standard.

**The apparent gains are artifacts of the churn.** `churn_events` counts applied `ADD`s that
restate a point dropped in the same step — those ADDs still become points, so they inflate
`recorded_points` (460 → 682), inflate `retention` (duplicated content is trivially easy to
render), and reduce `starved` (12 → 6). Every metric that improved improved *because* of the
defect. **Do not read retention without churn beside it.**

## Root cause, measured

1. **The journal preserves near-duplicates that eviction used to hide.** Near-duplicate entry
   rate (character-trigram containment ≥ 0.6 against any earlier entry) in the synthesis
   prompts: old slice **5.6%**, journal slice **11.2%** — a 2x increase. `apply_ops` refuses
   only EXACT duplicates, so near-duplicates accumulate, and under v1.1 nothing ever removes
   them because retiring a point no longer destroys it.
2. **The coverage instruction forced the teacher to render all of them.** `COVERAGE_ADDENDUM`
   says 記憶中的每一項都必須寫進摘要，不可遺漏任何一項 ("every item must be written, omit
   none"), and the coverage gate rejected any target that did not reach 70% of entries. So a
   memory holding `批准水務緊急運輸局六十年的租賃與進出許可` and
   `批准水資源緊急運輸管理局六十年租約及臨時進出許可` produced a target dutifully stating both.
3. **The student generalised "redundancy is correct output" into the reading step.** 8.6x
   churn, from a change that touched only synthesis rows.

**The transferable lesson: a coverage gate defined over a redundant input teaches redundancy.**
Coverage and non-redundancy are not independent objectives, and gating on one without the other
optimises the metric by duplicating content — the same shape as `runs/clean-e3`, where deleting
fabricating rows deleted the content too. Both gates were run in pairs against fabrication
(§"Two defects the generation caught"); the missing third gate was against redundancy.

## The fix this implies, not yet run

Dedupe near-duplicates in `Memory.synthesis_view()` so the journal never PRESENTS redundancy,
then regenerate the slice against the deduped view and retrain. Presentation-level dedup is
preferred to tightening the reading step's duplicate guard, which risks refusing legitimately
distinct points; the journal's job is to preserve what was recorded, not to show it twice.

## What changed, and why

SPEC §4.1 v1.1 rebuilt the reading step around the journal and left `SYNTHESIZE` reading a
v1.0 world. Measured on `sft_pool_v11.jsonl`, all 450 synthesis rows:

* prompts hold a **median of 13 points and never more than 16** — exactly `POINTS_CAP`;
* **0 rows contain `後改為`**, the journal's supersession rendering, so the one behaviour
  `revise` exists to produce had no synthesis supervision anywhere;
* targets sit at 38.8 chars/point against a near-constant ~470-character output.

Replaying the pool's own gold ops through the real harness (88.3% applied) puts **51% of
meetings above 16 entries, up to 57**. More than half the serving distribution was absent from
training, and output length was never conditioned on input size — the mechanism behind the
measured under-rendering (~40 entries in, 346 characters out).

`tools/gen_journal_synth.py` rebuilds the slice by replay; `tools/swap_synth_slice.py` merges
it, replacing a synthesis row only where a journal-shaped row exists.

## The new supervision, measured

| | old slice | new slice |
|---|---|---|
| rows | 450 | 174 new + 306 preserved = 480 |
| ungrounded specifics | **1,347 / 3,376 = 39.9%** | **0 / 2,015 = 0.0%** |
| entries per prompt | median 13, max 16 | median 20, max 57 |
| rows above 16 entries | 0 | 113 |
| rows carrying `後改為` | 0 | 69 |
| mean coverage of memory | not measured | 0.955 |

All 12 `hedge-*` rows were preserved — they are what fixed `qwen-tools-v4`'s polarity
inversion, and a wholesale slice replacement would have deleted them.

**Two defects the generation caught, both of which argue for running gates in PAIRS:**

1. **Asking for coverage makes the teacher fabricate.** It summed three separate `30萬` memory
   items into `九十萬`, and invented `二十五萬六千`. A coverage gate alone would have trained
   that in. An explicit "do not sum, compute, or estimate" clause took the 29+ bucket from
   1/3 kept to 3/3.
2. **The teacher copies the harness's `（後改為：…）` markup into prose ~1 time in 3**, on
   exactly the rows carrying G1's revision capability. Rejected on the literal markup, never
   on the phrase `後改為`, which is ordinary Chinese and correct for a summary to say.

## The checkpoint, on real ASR (`data/asr_eval_v1`, 21 meetings)

Both arms re-measured under the SAME (current) instruments — `v11-e3` was re-run rather than
compared against its stored scorecard, because `evalkit` changed today. The re-run reproduced
6.1% ungrounded exactly, independently confirming the numeral refold.

| | `v11-e3` ep2 | `v12-e3` ep2 |
|---|---|---|
| clean meetings | 13/21 | 9/21 |
| churn rate | 0.0% | 4.4% |
| meetings starved | 3 | 9 |
| points recorded | 85 | 64 |
| specifics asserted | 33 | 15 |
| ungrounded | 6.1% of 33 | 26.7% of 15 |
| retention (G5) | — | 0.9375 |

**Read the per-meeting pairing, not the totals.** The aggregate looks like a clean regression.
The pairing does not: **9 meetings worse, 5 better, 7 unchanged**, with large swings in BOTH
directions (−10, −7, −7, −7, −6 against +6, +6, +5, +4, +3). `ivod-17701` goes 1 → 7 points;
`ivod-17675` goes 10 → 0. A two-sided sign test gives **p = 0.424**.

That is the signature of near-random reallocation, not systematic damage — and it is exactly
why `metrics/report.py` mandates paired sign tests over aggregate means. Reading the totals
alone would have recorded a supervision-caused regression that the data does not support.

## Mechanisms proposed and REFUTED by measurement

Both were plausible and both are wrong; recorded so they are not re-derived.

1. **Loss-share dilution** — longer synthesis targets crowding out reading-step gradient.
   Measured on loss-bearing (completion) tokens: synthesis share moved **34.4% → 36.2%**.
   Far too small to explain a 25% drop in recorded points.
2. **Small-memory supervision thinned** — the ASR corpus is dominated by 1–8 point memories,
   so a slice skewed toward large journals might starve that regime. Measured: the small
   buckets are essentially unchanged (00-04: 40 → 41; 05-08: 57 → 55). The 09-16 bucket fell
   353 → 271 only because replay revealed those meetings' true sizes.

## Variance design

Training turned out to cost ~35 minutes, not the ~4 h the notes implied, so the honest test is
cheap: both pools x seeds {0, 1}. `runs/v12-s1` and `runs/v11-s1` hold the replicates. If
`v12-s1` also lands below `v11`, the supervision is at fault; if it lands near it, the single
comparison was noise and neither conclusion was ever available from one run.

## Caveats that must travel with these numbers

* **Best-epoch selection is not trustworthy for `v12-e3`.** Its `eval_loss` was computed
  against `data/staging/valid_tools.jsonl`, which was still **`tools-v1`** — a superseded
  prompt format whose synthesis rows cap at 16 entries and contain no `後改為`, i.e. they
  actively penalise the behaviour this pool teaches. `train_toolcalls.py` now refuses the
  mismatch; `valid_tools_v2.jsonl` (411 rows) is the migrated set. Its synthesis rows are
  still v1.0-SHAPED, so `eval_loss` remains a weak selector.
* **The teacher changed.** The original synthesis targets were authored by Qwen3.8-27B, whose
  local blobs no longer exist; this slice is gemma-3-27b-it. Every synthesis row was
  regenerated rather than only the oversized ones, so the slice is internally consistent, but
  it is not the same teacher as the reading rows.
## The comparison above was run on a corpus that CANNOT show the effect

Measured after the fact, and it invalidates the ASR table as evidence either way:

| corpus | meetings | median chunks | max chunks | meetings >16 chunks |
|---|---|---|---|---|
| `data/asr_eval_v1` | 21 | **1** | 5 | **0** |
| `data/heldout_zh` | 40 | 12 | 37 | 10 |

The working set holds 16 points. A 1-to-5-chunk meeting never overflows it, so **the journal
never fills, no point is ever retired, and `build_synth_prompt` falls back to the plain
working-set view** — byte-identical to v1.0. Every meeting in `asr_eval_v1` runs the code path
this entire change does not touch.

So the ASR numbers measure what journal-shaped supervision does to the model's behaviour on
SHORT meetings, which is a real question (and the answer is "nothing distinguishable at
p = 0.424") but is not the question the work was built to answer. **The evaluation must run on
`data/heldout_zh`**, where 10 of 40 meetings exceed the cap and the deficit — 41, 23 and 27
points recorded with 80%, 65% and 48% evicted — was originally measured.

This is CLAUDE.md trap 11's class, committed again: the instrument was fine, the corpus made
the effect unobservable, and the aggregate still produced a plausible number.

