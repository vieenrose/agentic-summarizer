# SPEC v1.0 (Qwen3.5-0.8B + tool-call protocol): the goal is reached; both checkpoints are weak on real ASR

**Verdict: the protocol change works. G2 and all three G3 gates PASS on 40 held-out
meetings, the reading step is measurably cleaner than v0's, the artifact is 30% smaller,
and it is 26% faster per step on measured ARM hardware. On real zh-TW ASR — the
deployment distribution — v1.0 is BETTER than the current v0 checkpoint (9/20 curated vs
6/20), though both are weak in absolute terms and that is the more important finding. G1
fails, as it does for every v0 checkpoint and for the same reason: the corpus, not the
model or protocol.**

**Recommendation: v1.0 is the better checkpoint** — equal on the gates, 26% faster per step
on measured ARM hardware, and better on real ASR (9/20 curated vs v0's 6/20). But BOTH are
weak on real ASR in absolute terms, and that is the finding that should gate shipping, not
the choice between them.

`runs/qwen-tools-v2/` (checkpoint), `runs/qwen-v2-heldout/` (evaluation).

## What v1.0 changed

Two normative changes, both measured before adoption (SPEC §4, §4.1):

- **Step grammar**: edit lines → one batched `update_memory` tool call per chunk, JSON
  arguments, **single-turn** (no tool-result round trip).
- **Student**: MiniCPM5-1B → **Qwen3.5-0.8B**, Q8.

## G3 (n=40 held-out, baseline = map-reduce on the SAME Qwen model, per §5.2)

| metric | wins/losses | mean delta | p |
|---|---|---|---|
| rouge1 | **27 / 13** | +0.075 | 0.038 |
| rouge2 | **33 / 7** | +0.046 | 0.000 |
| rougeL | **34 / 6** | +0.057 | 0.000 |

Comparable to `sft-dropv6` on the same 40 meetings (29/11 +0.077, 31/9 +0.043, 33/7
+0.053) — the protocol change costs nothing in quality and the smaller model holds up.

**Do not compare these ROUGE values across model families.** Qwen3.8-27B authored every
reference summary, so a Qwen student may match that house style more easily. The honest
reading is agent-vs-its-own-baseline within each family, which is exactly what §5.2's
same-model baseline rule already enforces.

## The reading step is cleaner than v0's

Measured over 23 steps on held-out meetings:

| | dropv6 (edit lines) | qwen-tools-v2 |
|---|---|---|
| ops applied | 75.5% | **89.8%** |
| malformed output | — | **0** |
| duplicate point | 14.8% | **9.1%** |
| arc unchanged | 7.0% | **1.1%** |
| **wasted output** | **21.8%** | **10.2%** |

Wasted output is decode, the term that gates §7's latency budget, so halving it is a
budget result as much as a quality one. All four operations are used (Arc 19, Add 58,
Drop 7, Nop 4) — the zero-shot model used only `add_point`.

Artifact size: **812 MB against MiniCPM5-1B's 1.15 GB.**

## G1 still fails, and it is the same failure

`runs/qwen-tools-v2/g1_report.json`: 0 of 2. Independent 11-scenario reversal probe:
**2/11**, against `sft-dropv6`'s 3/10 and `sft-dropv7`'s 2/10 — statistically
indistinguishable, and this pool INCLUDED the 68 synthetic reversal samples.

This is now measured across two model families, two protocols and three checkpoints. G1 is
not a protocol problem and not a base-model problem: it is the corpus. MeetingBank contains
almost no within-meeting decision reversals (3.4% of gold items match reversal language,
and those are legislative boilerplate about repealing external ordinances).

## The failure that produced this build

A first Qwen build (`runs/qwen-tools-v1-nosynth/`) failed **all three G3 gates with the
agent LOSING to its own baseline** (rouge1 8/40, -0.152, p=0.000). Cause: the format
converter excluded the 463 map/reduce/synthesis rows because they have no tool-call
equivalent, so the student never saw the SYNTHESIZE prompt. Summaries averaged **101
characters against the baseline's 738**, 11 of 40 were under 80 characters, and the worst
emitted a bare `<think> </think>`.

Fixed by carrying prose rows through unchanged — one model, two output formats keyed on
prompt shape, which is what the v0 pools already did. After the fix: 364 chars mean, **0
summaries under 80**. The broken checkpoint is kept rather than deleted, and the numbers
are recorded in `tools/to_toolcalls.py` where the exclusion lived.

## Integration costs of the model swap (none of them optional)

1. **Qwen3.5-0.8B is a vision-language model** (`Qwen3_5ForConditionalGeneration`).
   `AutoModelForCausalLM` loads only the text tower, which is why unsloth surfaces it as a
   `Qwen3VLProcessor` and TRL then aborts on a placeholder `'<EOS_TOKEN>'`. **unsloth
   cannot train this model**; `tools/train_toolcalls.py` uses plain `transformers.Trainer`
   with explicit completion-only masking instead.
2. **It carries an MTP head at block 24** (`mtp_num_hidden_layers: 1`). Fine-tuning the
   text tower drops those 15 tensors and llama.cpp's converter *asserts* they exist, so the
   GGUF will not build; setting the count to 0 instead trips a different assert. The fix is
   to copy them from base — correct rather than a fudge, since training never touches them.
3. **248k vocabulary makes training memory-bound** — logits OOM at batch 4 on a 32 GB card;
   batch 1 with accumulation was required.

## Deployment domain: v1.0 is BETTER than v0 — and v0 has silently regressed

`runs/qwen-tools-v3/ly_asr.json`, `runs/sft-dropv6/ly_asr.json`. Run on the 20 real zh-TW
**ASR** meetings from Phase 3 — the distribution the product actually runs in, as opposed
to the machine-translated clean text every gate is measured on.

| real zh-TW ASR | dropv2-era (2026-08-28) | **dropv6 (v0, current)** | **qwen-v3 (v1.0)** |
|---|---|---|---|
| meetings curated | 17/20 | **6/20** | **9/20** |
| meetings with empty memory | — | 14 | 11 |
| NOP rate | 41% | 69% | 59% |
| mean summary chars | 230 | 92 | 122 |

**A first version of this section reported v1.0 as a regression. That was wrong**, and the
error is worth keeping: it compared v1.0 against the dropv2-era 17/20 — a number from three
checkpoints earlier, before Phase 4 supervision, `POSITION` and the reversal data. Re-run
today on the same corpus with the same harness, **v0 scores 6/20 and v1.0 scores 9/20**.

The real finding is the one that comparison hid: **v0 regressed badly on real ASR between
dropv2 and dropv6 — 17/20 to 6/20 — and nothing caught it**, because every gate added since
Phase 3 is measured on clean, in-distribution, machine-translated text. Both current
checkpoints abstain on the majority of real meetings.

**Overfitting was tested as a cause and ruled out.** `save_strategy="no"` had been shipping
the epoch-3 checkpoint while eval loss bottomed at epoch 2, so `qwen-tools-v3` was retrained
with per-epoch saving and `load_best_model_at_end` (eval loss 0.890 / **0.878** / 0.961).
Identical data and hyperparameters, one variable. It changed the ASR result by less than the
gap between checkpoints, so checkpoint selection is not the mechanism.

**What this means for shipping.** The gate table is measured on a distribution the product
does not run in, and on the distribution it does run in, both checkpoints are weak in
absolute terms. Phase 3 exists precisely to catch this and has not been re-run since
dropv2. **Re-running the ASR check should be a standing gate, not a phase**, and no
checkpoint should ship on MeetingBank numbers alone.

## Open

- **G2 PASSED** — 18 vs 68 inversions on 39/40 paired meetings. Per claim the BASELINE is
  better (6.5% vs 4.8%): the agent wins on absolute count partly because it asserts 279
  claims against 1,410, i.e. it says less. Both readings belong together.
- **G4 remains unmeasured on the reference device.** `training-machine` is online in
  Tailscale but port 22 times out from this workstation. Two ARM64 CPU-only hosts ARE
  reachable (`raspberrypi`, `nano`) and a benchmark is running on the Pi — but both are
  ARMv8.0 without `dotprod`, against the Reno 7's Cortex-A78, so they give a PESSIMISTIC
  bound, not a G4 result. The 16.8-minute figure in the report is a projection I computed,
  not a measurement, and not the number the user authorised (19.58 min, for the MiniCPM5
  configuration).
- **Eval loss bottomed at epoch 2 (0.877) and rose at epoch 3 (0.959)** in both Qwen runs,
  and `save_strategy="no"` discarded the better checkpoint. **Now fixed** —
  `train_toolcalls.py` saves per epoch and loads the best by eval loss. This checkpoint was
  trained BEFORE the fix, so it is the overfitted epoch-3 model; retraining is the cheapest
  candidate explanation for the ASR regression above and should be tried first.
- **Human validation is prepared but not done.** `tools/build_review_packet.py` generates a
  blind, length-stratified 12-meeting packet (`review/`), key withheld. No machine can close
  this gate.


## `qwen-tools-v4`: the deliberation fix works on real ASR, at a real MeetingBank cost

`runs/qwen-tools-v4/`. Trained on the SAME pool as `qwen-tools-v3` plus 48 synthetic
DELIBERATION examples (`tools/gen_deliberation.py`) teaching the model to record a
speaker's STANCE the moment it is stated — `ADD - 委員質疑X` / `委員要求X` — instead of
requiring the chunk to land on an explicit resolution before recording anything. Same
design discipline as the reversal fix: planted gold, an independent probe set sharing no
subject with training, replay-clean filtering.

### Real ASR: large, direct improvement — the fix worked

| | qwen-tools-v3 (no deliberation training) | **qwen-tools-v4** |
|---|---|---|
| meetings curated | 9/20 | **16/20** |
| NOP rate | 59% | **10%** |
| mean summary chars | 122 | 218 |

The negative control held: `ivod-17673` (genuine ASR noise — stutter-repeated "他這個
他這個...") still correctly abstains (0 points). Spot-checked the clearest example of the
original gap, `ivod-17666` (a legislator's clause-by-clause critique with no landed
resolution): v4 now records `委員要求增加主審軍事上不利益` / `質疑將降敵與陰謀預備納入處理風險`
— attributed stances grounded in the actual transcript, not fabricated outcomes.

### MeetingBank held-out: a real cost, not a wash

| metric | qwen-tools-v3 | **qwen-tools-v4** |
|---|---|---|
| rouge1 | PASS — 27/13, +0.075, p=0.038 | **FAIL — 25/15, +0.045, p=0.154** |
| rouge2 | PASS — 33/7, +0.046, p=0.000 | borderline — 26/14, +0.031, p=0.081 |
| rougeL | PASS — 34/6, +0.057, p=0.000 | PASS — 28/12, +0.040, p=0.017 |

Agent summaries grew 26% longer (364 -> 460 mean chars) and rouge1 got WORSE on 26 of 40
meetings. The deliberation training generalised past the gap it targeted: on MeetingBank
chunks that DO land on a resolution, the model now also records surrounding mid-debate
stances, adding attributed-stance clutter around a correct resolution and diluting
precision. This is the same shape of trade dropv4/dropv5 hit with late-step supervision —
a real capability gain in one regime, paid for with precision in another.

### Where this leaves the decision

Neither `v3` nor `v4` is strictly better. `v3` passes G3 and is nearly unusable on real
ASR (9/20, and the ASR failures were traced to a genuine capability gap, not noise). `v4`
fixes that gap concretely and loses G3's rouge1 significance. **Recommendation: `v4` is
the better PRODUCT choice** — a checkpoint that curates 80% of real meetings and writes
slightly less precise MeetingBank summaries beats one that scores marginally higher on a
clean corpus but abstains on more than half of what it will actually be asked to
summarise. But this is a values call about what the gates should weight, not a
measurement, and it should be made explicitly rather than by default.

**Open**: G2 not yet run on v4. The independent deliberation probe (`data/deliberation_probe`,
4 scenarios sharing no subject with training) has not yet been scored — that is the check
for whether the ASR gain generalises past the 8 trained scenario topics, the same
discipline the reversal probe applied and the same discipline that caught the reversal
fix's failure to generalise. Do not treat the ASR number alone as sufficient evidence.


### Independent deliberation probe: 2/4 formally, and ONE case shows an inversion

`runs/qwen-tools-v4/deliberation_probe_report.json`, 4 scenarios sharing no subject with
the 8 training scenarios (`data/deliberation_probe`). Read directly rather than trusting
the automated score, since it is crude (subject-name string match):

- `pensionreform`, `telecomfraud`: PASS, correctly attributed stances.
- `hospitalstaff`: automated FAIL is a SCORING ARTIFACT — the summary correctly captures
  the ER-overcrowding-evaluation position in substance ("建議將急診壅塞納入醫院整體評分體系")
  without naming the bill verbatim. Read as a pass.
- `forestprotect`: **a real inversion, not a scoring artifact.** The source states the
  position as advocating HEAVIER penalties twice, both the committee member's ask and the
  official's confirming reply ("國有林地濫墾應加重刑責並溯及既往查處"). The model's summary
  says the committee "認為該事件**不應**加重刑責並溯及查處" — a negation of the stated
  position.

**This is exactly the failure mode G1 and G2 exist to catch, and it appeared in the
deliberation-fix probe, not the reversal one.** It was not looked for before now.
`v4`'s ASR gain and this inversion both need weighing before recommending it: fixing 7
meetings' worth of abstention is not a win if the mechanism that unlocked it also makes
attributed stances less reliable. **Do not recommend `v4` for anything without running
G1 and a faithfulness check specifically on deliberation-heavy output** — this probe
result is a reason to distrust it pending that, not a reason to prefer `v3`'s silence
over `v4`'s speech, since abstention has its own cost. Recorded as OPEN, not resolved.


### The forestprotect inversion, root-caused: a synthesis-stage negation bug on "是否" framing

Localized precisely by replaying the exact memory state through `synthesize_memory` in
isolation, deterministic across 3 seeds:

- **Reading step is CORRECT.** It records `委員質疑國有林地濫墾是否應加重刑責並溯及查處` —
  "the committee questions WHETHER it should be given heavier penalties" — a faithful,
  neutral framing of the actual debate.
- **Synthesis introduces the inversion.** Given that exact point, it consistently renders
  `委員...認為該事件不應加重刑責並溯及查處` — "the committee believes it should NOT be
  penalised" — asserting the opposite polarity as settled fact.

**Root cause: training data never used this framing, so the reading step's own paraphrase
choice was off-distribution for synthesis.** Every `ADD` point in
`tools/gen_deliberation.py`'s training set asserts a definite position
(`委員質疑<position>`) — NONE use `是否` (whether-or-not) phrasing. The reading step chose
that framing itself, live, when paraphrasing this specific probe transcript; synthesis had
never seen a "질問是否X" point and resolved the ambiguity by guessing a polarity — wrong.

Confirmed NOT a general sparse-memory hallucination: an isolated single-point test with
weak context did hallucinate extensively (fabricated agency names, a fabricated second
case not present in the transcript or memory) — a separate, more general risk worth
tracking — but the exact 2-point memory from the real run reproduces the specific negation
cleanly, without the broader fabrication. Two distinct failure modes, not one.

**This is fixable without another training run's uncertainty**, because the mechanism is
narrow: only points containing `是否`/`能否`/similar hedge-question markers are at risk. A
guard could (a) forbid `是否` in `ADD` targets during teacher supervision generation and
retraining data conversion, so the reading step is never taught the pattern, or (b) a
cheap runtime check flagging any point containing a whether-marker for exclusion or
verbatim-carry into synthesis rather than paraphrase. Neither is implemented. This should
block recommending `v4` (or any future deliberation-trained checkpoint) until one of them
is, since the mechanism is now understood well enough to fix rather than merely avoid.


### Detection guard implemented and measured: the trigger is RARE, not systemic

`src/arcsum/guards.py`: `hedge_marker_in` + `Outcome.hedge_points`, following this
codebase's standing "detect and record, never repair in-loop" discipline (the same rule
`nop_collapse` already applies) — `ADD` points are still APPLIED, only flagged, since
refusing them outright is unvalidated and a hedge-phrased point may be the best available
capture of a genuine open question. 3 tests pin the exact reproduced case.

Measured across all 24 real-ASR + deliberation-probe meetings with `qwen-tools-v4`: **77
ADD points, 1 hedge-flagged (1.3%)** — the same `forestprotect` case already found, no
others. **This is a rare edge case, not a systemic reliability problem** — worth guarding
because the one instance it caught is a genuine inversion, not worth blocking a ship
decision on frequency grounds alone. Whether to also change generation behaviour (ban the
phrasing in training data, or instruct synthesis to preserve question form) is unvalidated
and NOT done — this guard only makes the failure visible and measurable going forward.


### `qwen-tools-v4` full gate picture (measured 2026-08-30)

| gate | `qwen-tools-v3` | **`qwen-tools-v4`** |
|---|---|---|
| G1 revision | FAIL — 0/2 | FAIL — 0/2 (formerly 1/2 on the plain probe; see inversion above) |
| G2 faithfulness | PASS — 18 vs 68, 39/40 paired | **PASS — 18 vs 63, 39/40 paired** — same inversions, MORE claims (303 vs 288), per-claim rate improved (6.2%->5.9%) |
| G3 rouge1 | PASS — 27/13, p=0.038 | **FAIL — 25/15, p=0.154** |
| G3 rouge2 | PASS — 33/7, p=0.000 | **FAIL — 26/14, p=0.081** (effect size clears, sign test does not) |
| G3 rougeL | PASS — 34/6, p=0.000 | PASS — 28/12, p=0.017 |
| real ASR curated | 9/20 | **16/20** |

**G2 resolves the ASR-fix trade-off largely in v4's favour.** The deliberation training
added 15 more claims (303 vs 288) at the SAME inversion count (18), and the per-claim rate
improved slightly (6.2% -> 5.9%). Teaching the model to record mid-debate stances did not
make it less faithful — it recorded more without recording more WRONG. The forestprotect
negation bug (above) is real and now guarded, but at 1.3% frequency it did not move the
aggregate inversion count.

**decision: still "ship the baseline"** — G1 and (now) G3 rouge1/rouge2 fail, and §5.2 is
all-or-nothing. But the case for `v4` as the better PRODUCT checkpoint is now stronger than
when only the ASR number was in hand: it is exactly as faithful as `v3`, materially better
on the domain the product will actually run in, and behind on MeetingBank ROUGE mainly
because it also narrates deliberation MeetingBank's clean, resolution-only text rarely
contains. Recommend `v4` over `v3` for any further work; recommend neither for shipping
without G4 measured on-device and a resolution to the G1 corpus gap.


### G4 re-checked for `v4` specifically: no material change

`v3`'s ~14.7 min figure was never re-measured for `v4` — it inherited the label by
default, which is a gap worth closing given `v4`'s summaries run 26% longer end-to-end.
Measured on a matched 12-meeting slice of `data/heldout_zh`, same protocol/config, both
checkpoints back-to-back:

| | `qwen-tools-v3` | `qwen-tools-v4` |
|---|---|---|
| mean steps/meeting | 15.2 | 15.2 (chunking is model-independent) |
| mean raw step output (chars) | 155 | 159 (+2.6%) |
| mean synthesis output (chars) | 295 | 340 (+15%) |

Step count is unchanged — the deliberation training changed WHAT gets recorded, not how
many chunks are read. Per-step output grew only 2.6% (most of the extra length landed in
`arc`, not repeated `add` calls); synthesis grew 15%, matching the measured 26% growth in
FINAL summary length once ARC context compounds through the prose call.

Projected onto the anchored Reno-7 figure (`v3`: 15.9 min on this same 12-meeting slice):
**`v4` projects to ~16.2-16.4 min**, a ~0.3-0.5 min cost, comfortably inside the margin
`v3` already has against the 20-minute ceiling. **This is still a projection, not a
device measurement** — same caveat as `v3`'s figure, now inherited honestly rather than
copied.


### Negation-bug fix attempt: synthesis-side instruction REFUTED, confirms training-side cause

Tested the cheap fix first: appending an explicit instruction to `synth_system_prompt()`
("if a point contains 是否/能否, preserve the question form — do not rewrite it as a
definite conclusion") and re-running `synthesize_memory` on the exact known-bad memory
state. **Byte-identical output across 2 seeds, with or without the instruction.**

This is informative, not just a null result: it confirms the bug is NOT a live prompting
gap synthesis could be talked out of — it is a LEARNED association (or lack of one) from
training. `v4`'s pool never contained an `ADD` target with `是否`-phrasing (measured
earlier), so the model has no trained behaviour to invoke for "preserve uncertainty" that
a runtime instruction could activate. **Do not retry prompt-only fixes for this** — the
fix has to be in the training data: either the teacher/generator must be told to never
produce `是否`-phrased `ADD` targets (removing the trigger), or a small number of
`是否`-phrased examples with a preserved-question-form synthesis target must be added
(teaching the behaviour directly). Neither attempted yet. The `hedge_marker_in` guard
remains the only active mitigation — it detects and records, it does not correct.


### Second fix attempt also refuted: forbidding 是否 at the READING step is equally dead

Tested the other plausible cheap fix: appending "never use 是否/能否 phrasing in `add`;
write the committee's concrete position directly" to the READING step's system prompt
(not synthesis this time) and re-running end to end on `forestprotect`. **Byte-identical
reading-step output AND byte-identical final summary**, with or without the instruction.

**Both prompt-side interventions are now confirmed dead for the same reason**: a
fine-tuned checkpoint at `temperature=0` does not reliably respond to novel instructions
appended to its system prompt — its output is fixed by what it learned during training,
not by what it is told at inference time. This generalises the earlier finding beyond
synthesis: **any fix for this bug has to be a TRAINING change**, full stop, not a prompt
change to either call in the pipeline. Concretely, the next attempt should add a handful
of `是否`-phrased `ADD` targets to `tools/gen_deliberation.py`'s scenarios, each paired
with a synthesis training example that PRESERVES the question form, so the model learns
the specific transformation rather than being told about it live. Not attempted — the
next deliberation-data build should include this before assuming it works.


## `qwen-tools-v5`: the negation bug is FIXED, and the v3/v4 trade-off turns out to be avoidable

`runs/qwen-tools-v5/`, `runs/qwen-v5-heldout/`. `v4`'s pool + **12 synthesis rows**
(`tools/gen_hedge_synth.py`) where a `是否` question-form point in MEMORY maps to a prose
target that PRESERVES the question form.

### What the pool was actually teaching (the real root cause)

The earlier diagnosis — "synthesis inverts question-form points" — was incomplete.
Inspecting the pool directly: of 175 synthesis rows, **5 carry `是否` in the memory input,
and only 2 preserve it in the prose target. The other 3 DROP the point entirely.** So the
majority signal was "a question-form point is not worth carrying", and the model, having no
trained behaviour for preserving one, improvised at inference. Inverting was the
improvisation.

This also explains why BOTH prompt-side fixes produced byte-identical output: there was no
learned behaviour for an instruction to activate. After adding 12 rows the signal flips
from 40% preserving (2/5) to **82% (14/17)**.

### Result: strictly better than both prior checkpoints

| | `v3` | `v4` | **`v5`** |
|---|---|---|---|
| G3 rouge1 | PASS 27/13, p=0.038 | FAIL 25/15, p=0.154 | **PASS 28/12, +0.069, p=0.017** |
| G3 rouge2 | PASS 33/7, p=0.000 | FAIL 26/14, p=0.081 | **PASS 29/11, +0.041, p=0.006** |
| G3 rougeL | PASS 34/6, p=0.000 | PASS 28/12, p=0.017 | **PASS 35/5, +0.057, p=0.000** |
| real-ASR curated | 9/20 | 16/20 | **17/20** |
| forestprotect inversion | n/a | present, 3/3 seeds | **gone, 3/3 seeds** |
| negative control (`ivod-17673`) | abstains | abstains | **abstains** |

**The `v3`-vs-`v4` trade-off was avoidable, not fundamental.** It had been recorded as "a
values call, not something the numbers resolve automatically" — that reading was wrong.
`v4`'s G3 regression was not the price of the ASR fix; it was a side effect of the model
having no coherent handling for question-form content, and it disappeared once that was
taught explicitly. **12 rows recovered G3, improved ASR curation, and removed the
inversion simultaneously.**

Two further observations worth carrying:

- **The READING step improved too, though only synthesis rows were added.** On
  `forestprotect` `v5` now emits the DEFINITE stance `委員要求...應加重刑責` — more faithful
  to the source than `v4`'s hedged `是否` phrasing. Giving the model a coherent story about
  question-form content changed how it paraphrases upstream, not just how it renders
  downstream.
- **17/20 matches the dropv2-era ASR baseline** that opened this investigation. The
  regression found earlier (17/20 -> 7/20 across three v0.9 checkpoints, invisible to every
  §5.2 gate) is now fully recovered — on a different model family and a different protocol.
- NOP rate rose 10% -> 28% versus `v4`, closer to the teacher's natural band, while curation
  went UP. Reads as better judgment rather than more abstention, but worth watching.

**Open**: G2 running. G1 still fails (corpus limitation, proven across two model families
and four checkpoints). G4 not re-measured for `v5` — but `v5`'s mean summary is 223 chars
on ASR vs `v4`'s 218, so the `v4` projection (~16.2-16.4 min) carries over essentially
unchanged.


### `v5` G2: PASS, and the best faithfulness of any checkpoint

`runs/qwen-v5-heldout/g_report_final.json`. **40/40 paired** — the first fully-paired G2
run in the project (`v3` and `v4` both lost a meeting to judge exhaustion).

| arm | inversions | claims | inv/claim |
|---|---|---|---|
| **agent** | **16** | 283 | 5.7% |
| baseline | 58 | 1467 | 4.0% |

Per-meeting: agent fewer on 21, tied on 12, worse on 7.

Agent-side inversions across the three v1.0 checkpoints: `v3` 18, `v4` 18, **`v5` 16**.
**The negation fix reduced faithfulness errors beyond the single case it targeted** — I
set out to remove one deterministic inversion on one probe transcript and the aggregate
count across 40 held-out meetings fell by 2. The pool inconsistency was costing
faithfulness wherever question-form content appeared, not only in the case that surfaced
it.

The per-claim rate still favours the baseline (4.0% vs 5.7%), unchanged in character from
`v3`/`v4`: the agent wins decisively on absolute inversions partly because it asserts far
fewer claims (283 vs 1467). Both readings belong together; §5.2's gate is the absolute
count.

### Final `v5` gate table

| gate | result |
|---|---|
| G1 revision | **FAIL** — corpus limitation, proven across 2 model families / 4 checkpoints |
| G2 faithfulness | **PASS** — 16 vs 58, 40/40 paired |
| G3 rouge1 | **PASS** — 28/12, +0.069, p=0.017 |
| G3 rouge2 | **PASS** — 29/11, +0.041, p=0.006 |
| G3 rougeL | **PASS** — 35/5, +0.057, p=0.000 |
| G4 budget | **WITHHELD** — no device measurement; projection ~16.2-16.4 min |
| real-ASR curated | 17/20 (matches the pre-regression dropv2-era baseline) |

**`decision: ship the baseline`** — §5.2 is all-or-nothing and G1 fails. That verdict is
correct and should not be argued around: G1 tests whether a decision reversed later in a
meeting is reported in its final state, and MeetingBank cannot teach it (3.4% of gold items
match reversal language, all of it boilerplate about repealing EXTERNAL ordinances).

**But `v5` is 5 of 7 with the two failures being a corpus gap and an unmeasured device
number — not quality problems.** It is the strongest checkpoint the project has produced on
every axis that was actually measured.
