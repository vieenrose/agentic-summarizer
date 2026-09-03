# Next fine-tune iteration — clean the supervision, add almost nothing

**Every iteration since `qwen-tools-v3` ADDED data**: synthetic reversals, deliberation
examples, hedge rows, detail rows, self-distilled synthesis. Four of the last five moved
one axis and broke another. This one removes measured pathologies from the pool instead,
because they have now been located and counted.

## Diagnosis, measured 2026-09-03 on `data/staging/sft_pool_mixed.jsonl`

### The synthesis supervision is structurally unfaithful

**39.7% of the specific claims in synthesis targets are absent from the memory those
targets were written from** — 1,347 of 3,396, across 203 of 454 rows (45%).

This is not a labelling slip; it is how the corpus was built. SPEC §2.2 stage 3 composes a
whole-meeting summary from the segment minutes, and the synthesis row pairs
`(student-or-teacher memory) -> that gold summary`. The target therefore contains details
the memory does not hold, because it was written from the full minutes rather than from
the input. **The target is not a function of the input, which is definitionally training
the model to invent.**

Decomposed, so the reformatting caveat cannot swallow it:

| ungrounded claim type | count | share |
|---|---|---|
| Arabic number | 629 | 47% |
| Latin identifier | 280 | 21% |
| CJK numeral (numeral-system reformatting is plausible) | 438 | 33% |

Only the last third has an innocent explanation. **~900 claims are Arabic figures or Latin
identifiers asserted in a target that appear nowhere in its memory** — about two per row.

**And the student reproduces the rate.** `arcsum-eval` on the 20 real zh-TW ASR meetings
measures `qwen-tools-v5` at **44.4% ungrounded**, with examples that are unmistakable
fabrication rather than measurement artifacts: `財政部（Central Bank）` (an invented gloss,
and wrong — 財政部 is the Ministry of Finance), `四十九億七千六百九十七美元`,
`二〇一七年六月三十日`, a bare `2021`, a bare `GDP`.

### Three behaviours the student learned from the pool

Measured with the harness's OWN guards, not a proxy:

| behaviour | in the pool | student's expression |
|---|---|---|
| churn (`DROP` + near-identical re-`ADD`) | **106 / 4,540 reading rows (2.3%)**, verified with `guards.restates_dropped` | 28.2% of steps on real ASR (`v5`); 67% on one meeting (`mixed-e3`) |
| ARC re-emitted unchanged | **339 / 2,006 consecutive ARC emissions (16.9%)** | `arc frozen x5` on the rolled-back demo run |
| under-rendering at high occupancy | 13 / 444 synthesis rows below 20 ch/pt, worst **6.2 ch/pt at 16 points** | 15 points -> 53-character summary (user-reported) |

The ARC row is the sharpest: **the harness refuses every one of those 339 ops as
`arc unchanged`.** The pool spends output budget teaching an op that is rejected 100% of
the time.

A correction that matters for anyone re-deriving this: a crude "does the ADD share a prefix
with the DROP" test reports 43% churn. That is wrong — it counts legitimate revision
(`DROP «公車路線調整»` + `ADD 公車路線調整案改為取消`) as churn. Using the real guard:
**106 churn against 1,265 legitimate revisions.**

### The synthesis cliff is an over-extrapolated real trend, not only exposure bias

Teacher synthesis targets, bucketed by the occupancy of the memory they were written from:

| points in memory | rows | median ch/pt | median chars |
|---|---|---|---|
| 1–5 | 56 | 76.6 | 262 |
| 6–9 | 66 | 54.9 | 404 |
| 10–12 | 55 | 43.3 | 459 |
| 13–16 | 277 | 34.2 | 507 |

Per-point length declines monotonically — reasonable, since a summary should not grow
linearly with points. But total length keeps RISING to 507 characters. `v5` at 13 points
produced 116 characters (~9 ch/pt), far past anywhere the teacher goes. **The trend is in
the data; the collapse is not.** This refines, rather than replaces, the exposure-bias
finding in `runs/selfdistil-e3`.

## The plan: four surgeries, one small addition

Each has a deterministic check that can be run on the pool before any GPU time.

### S1 — make synthesis targets a function of their input (the big one)

Reject or repair every synthesis row whose target asserts specifics absent from its memory,
using `evalkit.grounding.check(target, prompt)`.

Two routes, in order:
1. **Filter first, because it is free.** Drop rows above a per-row ungrounded threshold and
   measure what the pool loses. At a threshold of zero this removes 45% of synthesis rows,
   which is probably too aggressive; sweep it and report the retained count.
2. **Then regenerate, because filtering alone will thin the highest-occupancy rows** — the
   regime that already fails. Re-ask the teacher for a summary **of the memory alone**, with
   the gold summary withheld. This is the real fix: it makes the target derivable from the
   input by construction.

### S2 — delete teacher-taught churn

Drop the 106 rows where `restates_dropped` fires. Do not rewrite them: `mix_phase4.py`'s
established rule is that a step which cannot be repaired is DROPPED, never rewritten to NOP.

### S3 — strip always-refused ARC ops

For the 339 targets whose `arc` equals the previous step's arc, remove the `arc` key and
keep the row's `add`/`drop`. The row still teaches valid edits; it stops teaching an op the
harness rejects every time.

### S4 — carry the corrected reversal rows

Use `data/staging/sft_pool_revfix.jsonl`'s 52 reversal rows (**26/26 preserving `key_term`**)
in place of the current pool's 68 (**0/34 preserving**). Already built and verified.

### A1 — the only addition: outcome-stating synthesis rows

`revfix-e3` proved the detail half is fixable — key_term retention went 14.8% -> 48.1% at
memory and 3.7% -> 33.3% at prose, closing the revision-specific gap from +58.5 to +11.9
points — and the gate still fell 8/27 -> 6/27 because **"states the late outcome" fell 8 ->
6 while "subject present" rose 13 -> 20.** The replacement point has a fixed budget and now
spends more of it on the detail.

So generate synthesis rows from reversal memories whose target must contain the LATE
OUTCOME word, and verify that with a deterministic check rather than trusting the teacher.
Keep it small (tens of rows, as with the 12 hedge rows that fixed the negation inversion).

## What NOT to do — all refuted, do not re-derive

- **More synthetic reversals from the same recipe.** Five refuted attempts; `runs/g1-study.md`.
- **Reweighting / oversampling to fix long meetings.** `sft-dropv3` regressed ROUGE while
  holding shares constant: stable shares did not imply stable behaviour.
- **Raising `POINT_TOKENS`.** Tested 25 -> 32 on 2026-09-02: probe 6/27 -> 3/27, worse.
- **Switching base model.** The 3% G4 margin makes a larger base a non-starter; Granite's
  smallest adapter base is ~3B against a 0.8B student.

## Evaluation protocol — the part that failed last time

The previous checkpoint passed every gate and was rolled back within hours. The protocol
changes accordingly:

1. **Scorecard BEFORE and AFTER on real ASR**, via `arcsum-eval --corpus data/ly_phase3_v2`.
   Compare `clean_meetings`, `churn_rate`, `ungrounded_rate`. The current `v5` baseline is
   `runs/qwen-tools-v5/scorecard_asr.json`: **clean 10/20, churn 28.2%, ungrounded 44.4%**.
2. **Measure the DEPLOYED configuration too.** Every gate to date pinned
   `cache_prompt: false` while the demo runs the KV cache live across calls, and that gap is
   exactly where the rolled-back regression lived. Run the ASR scorecard both ways.
3. **Export BOTH epochs and measure both.** Epoch response is build-specific and eval loss
   is anti-correlated with what is gated — the self-distilled run's LOWEST-loss checkpoint
   abstains on 95% of real meetings.
4. Only after 1–3 look right: G3 on `data/heldout_zh`, then G2.

## RESULT of the first build (`runs/clean-e3`, epoch 2) — S2/S3 confirmed, S1 refuted

Pool: S1 at rate 0.35 + S2 + S3 + S4 -> 4,705 rows. Training was **killed at step 708/885
with no traceback and 231 GB of RAM free**; the cause is unexplained and epoch 3 does not
exist. Both saved checkpoints verified intact (320 tensors, correct shapes), so the
epoch-2 measurement below is real, not a corrupted artifact.

| | `v5` | `clean-e3` ep2 |
|---|---|---|
| churn rate | 28.2% | **2.6%** |
| clean meetings | 10/20 | 7/20 |
| starved meetings | 6 | **11** |
| median ch/pt | 39 | 95 |
| ungrounded | 44.4% **of 27 claims** | 0.0% **of 5 claims** |

**S2 + S3 are confirmed, decisively.** Churn fell by an order of magnitude from deleting
106 churn rows and stripping 323 always-refused ARC ops. The pre-registered falsification
was "if churn does not fall below ~10%, the amplification is architectural rather than
learned" — it fell to 2.6%, so **churn is learned from the pool and removable from the
pool.**

**S1-as-a-filter is refuted, and the pre-registered denominator is what caught it.** 0.0%
ungrounded reads as a triumph until `n_checked` is read beside it: **5 specific claims
across 20 meetings, against `v5`'s 27.** The model did not become faithful; it stopped
asserting specifics at all. Starved meetings rose 6 -> 11. This is the risk written into
`tools/clean_pool.py`'s docstring before the run — filtering thins the high-occupancy
regime that already fails — now measured rather than predicted.

**Revised plan: keep S2, S3, S4; drop S1-as-filter.** The real S1 repair is REGENERATION:
re-ask the teacher for a summary of the memory ALONE, with the gold summary withheld, which
removes the target-is-not-a-function-of-its-input defect without deleting rows. Deleting
rows removes the fabrication by removing the content.

`runs/s234-e3` isolates S2+S3+S4 with all 454 synthesis rows retained.

## RESULT of S1-BY-REGENERATION (`tools/regen_synth.py`) — the repair works

The filter was refuted above; this is the repair it pointed to. The teacher is asked to do
exactly the student's task — `synth_system_prompt()` over the row's own stored memory
prompt — so the target is derivable from the input by construction. Every rewrite is
verified with `evalkit.grounding` before acceptance, and a rejected rewrite keeps the
ORIGINAL target, so the pool cannot silently shrink.

| | before | after |
|---|---|---|
| ungrounded specifics in `SYNTHESIZE` targets | 1,347 / 3,376 (**39.9%**) | 786 / 3,154 (**24.9%**) |
| rows containing >=1 fabrication | 203 / 450 (45%) | 112 / 450 (**25%**) |
| synthesis rows retained | 450 | **450** |

310 of 450 repaired, 0 errors. The 140 unrepaired rows kept their originals because the
teacher's rewrite still asserted something ungrounded — rejected rather than trading one
fabrication for another. **25% is a deliberate floor, not a failure**: it is what survives
verification.

**Contrast with the filter, which is the whole lesson.** Filtering reached 0.0% ungrounded
and did it by removing 37% of the synthesis supervision, leaving a model that asserted 5
specific claims across 20 meetings against `v5`'s 26, with starved meetings rising 6 -> 11.
**Repair keeps the content and fixes the target; deletion removes the fabrication by
removing the substance.**

### Two failures on the way, both worth not repeating

* **The teacher answered as the WRONG MODEL.** Port 8082 was already held by another
  user's `gemma-4-26B` server; the launch failed with `Exit 1` and `/props` answered
  Gemma. `evalkit.provenance`'s identity check is the only reason this was caught — and
  Gemma is specifically excluded from this project's judging roles because it did the
  corpus translation.
* **Trap 10 again, from this repo's own notes.** `--jinja` ALONE does not suppress
  thinking. The first pass repaired **0 of 20** because the teacher returned 1,483
  characters of deliberation (`我們需要回答使用者：…`) as the target. Adding
  `chat_template_kwargs: {enable_thinking: false}` fixed it: 11/20. The grounding check
  correctly rejected all 20 bad rewrites, which is the verification step doing its job.

`data/staging/sft_pool_regen.jsonl` carries all four surgeries: 310 synthesis targets
repaired, 106 churn rows dropped, 323 no-op ARC ops stripped, 52 corrected reversal rows.

### And the trained checkpoint is WORSE. The pool defect is fixed; the model did not improve.

All four builds on ONE corpus (`data/asr_eval_v1`, 21 meetings) in the DEPLOYED cache-on
configuration, `comparison_key` verified identical so these are legitimately comparable:

| build | clean | churn | starved | pts/mtg | specifics | ungrounded |
|---|---|---|---|---|---|---|
| `v5` (shipped) | 8/21 | 26.7% | 7 | 3.52 | 33 | 33.3% |
| **`s234-e3` ep2** | **12/21** | **0.0%** | 6 | 2.76 | 7 | 28.6% |
| `regen-e3` ep2 | 3/21 | 6.7% | **17** | **0.81** | 2 | 0.0% |
| `regen-e3` ep3 | 6/21 | 4.4% | 13 | 1.76 | 13 | 46.2% |

**The pre-registered falsifier fires.** Ungrounded fell below 20% (to 0.0%) and specifics
collapsed to 2 across 21 meetings — the exact failure mode the falsifier was written to
catch, and the same one the FILTER produced. Epoch 3 asserts more (13) but fabricates at
`v5`'s rate (46.2%), so neither epoch is usable.

**What collapsed is the READING step** — 0.81 points/meeting against `v5`'s 3.52, 17 of 21
meetings starved — even though only SYNTHESIS targets changed.

**The regenerated targets are not the obvious culprit, because they are good.** Inspected
directly: grounded, fluent, faithful to their memory, and visibly BETTER supervision than
the originals, which invent details the memory never held (a plan number `819`, a hotel's
`96間客房和62個停車位`). Aggregate richness is unchanged: 7.0 specifics/row vs 7.5, 493 mean
chars vs 485. So this is not "the repair made the targets vague".

**Attribution is UNVERIFIED and must not be recorded as fact.** `regen-e3` differs from
`s234-e3` by 310 of 4,872 rows (6%), and a 3.4x change in reading-step output from a 6%
data change on a SINGLE training run is exactly the shape of the confounded attributions
this project has made five times before. Distinguishing "those 310 rows did it" from
"training variance" needs a repeat run at a different seed, which has not been done.

**Standing recommendation: `s234-e3` ep2.** It beats `v5` on the two axes that describe
deployed behaviour — clean meetings 8/21 -> 12/21 and churn 26.7% -> 0.0% — at the cost of
asserting fewer specifics (7 vs 33), which is its own open question. It has NOT been
through G2/G3, and nothing ships on reference-free numbers alone.

## THE SPECIFICS DEFICIT — located, and it is a supervision ceiling

A summariser that records *that* a budget was discussed but not *what the figure was* has
done a fraction of the job. Measured 2026-09-03, the loss map:

| stage | rate |
|---|---|
| chunks containing at least one specific | **99%** |
| gold `ADD` targets carrying one | **42%** |
| `qwen-tools-v5` memory points carrying one | 33% |
| `s234-e3` ep2 memory points carrying one | 16% |

**The student tracks its supervision** (33% against a 42% ceiling), which is the signature
of a data-limited system. No build has ever moved this, consistent with retraining on the
same pool being unable to.

**And the token cap is NOT the cause.** ADD targets that carry a specific and those that do
not have the SAME length distribution — mean 17.6 vs 17.7 tokens against `POINT_TOKENS`=25,
medians both 18, only 7-8% within a token of the cap. ~7 tokens of headroom. The teacher can
carry the figure for free and simply writes the topic instead.

`tools/enrich_points.py` lifted the ceiling **42% -> 60%** (946 rewrites of 3,050
candidates, 0 rows lost). Rejections are the interesting part: **1,309 "unchanged"** — the
teacher correctly declining when no relevant figure existed — and **33 FABRICATIONS caught**
(`850`, `CSULB`, `300/696`, `120`). Unverified, those 33 would have taught the model to
attach plausible-looking numbers to points, which is strictly worse than vagueness.

**This is also the clearest evidence on "is the teacher good enough".** It is capable and
worth pushing — 946 good rewrites — but it fabricates ~1% of the time when pushed, so
teacher output must be VERIFIED against the source, never trusted.

`runs/spec-e3` tests the prediction: if the student really does track its supervision, a
60% ceiling should put it near 50% against `v5`'s 33%. If it does not move, the
"student tracks teacher" model is wrong and the deficit lies elsewhere.

## Falsification, stated in advance

- **Ungrounded rate on real ASR does not fall below ~20%** -> the fabrication is not learned
  from the targets, and S1 is the wrong theory. Next suspect: decoding, since these are
  greedy at `temperature=0`.
- **Churn does not fall below ~10%** -> 2.3% of rows cannot account for 28% of steps, and
  the amplification is architectural (memory rendering, or the DROP-prefix matcher) rather
  than learned.
- **G3 falls while ASR improves** -> the same trade as `v3`/`v4`; then the honest answer is
  that the MeetingBank-derived references and real deliberation reward different behaviours,
  and the gate set needs splitting rather than the model needing another epoch.
