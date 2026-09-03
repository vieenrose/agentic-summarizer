# `mixed-e3` — teacher-memory AND student-memory synthesis rows in one pool

Built to keep `runs/selfdistil-e3`'s gains without its G3 loss. `selfdistil-e3` replaced
every teacher-authored synthesis memory with a student-authored one; this pool keeps BOTH
provenances side by side:

    pool: 4994 = 4544 reading + 187 TEACHER-memory + 263 STUDENT-memory synthesis
      teacher-memory  mean memory-details 3.87 | >=13pts 58% | target 522 chars
      student-memory  mean memory-details 5.30 | >=13pts 64% | target 459 chars

with a quality floor (`MIN_DET=2`) dropping 53 low-detail student memories, and 4x
oversampling of >=13-point rows so occupancy is matched across the two provenances.

Trained `--unsloth --epochs 3 --batch-size 1 --grad-accum 16` on
`principled-intelligence/Qwen3.5-0.8B-text-only`. Eval loss 0.8138 -> 0.7814 -> 0.8691, the
same epoch-3 rise every run in this project shows. **The LAST checkpoint (939) was
exported**, not `final/` (which holds best-by-loss, 626), to keep the comparison against
`v5` and `selfdistil-e3` like-for-like — every v1.0 checkpoint is a last-epoch artifact
(see CLAUDE.md on `load_best_model_at_end`).

## A recorded number that does NOT reproduce — read this before using the old one

The decision to build this pool rested on a measurement that self-distillation had
collapsed the reading step, "memory details 10.0 -> 2.8". **That figure was produced by an
inline script that was never committed, and it does not reproduce.** Re-measured with
`tools/measure_memory.py` (committed, definition fixed in its docstring) on the same 5
held-out meetings, all three checkpoints in one pass:

| | `v5` | `selfdistil-e3` | **`mixed-e3`** |
|---|---|---|---|
| points in memory | 10.4 | 7.8 | 9.4 |
| numerals carried | 15.4 | 8.2 | **12.2** |
| chars per point | 20.2 | 18.3 | 17.5 |
| prose chars | 338 | 334 | **433** |

Self-distillation's memory IS thinner than `v5`'s — points 10.4 -> 7.8, and numerals, the
one column padding cannot inflate, 15.4 -> 8.2. But that is a moderate regression, not the
3.5x collapse recorded. **The premise for this build was directionally right and
quantitatively wrong.** `mixed-e3` does recover most of the numeral loss (12.2) and writes
the longest summaries of the three.

## The cliff does not reproduce on a clean pool either — and that is informative

`tools/cliff_curve.py`'s hand-written pool shows `v5` with NO cliff at all: 237 / 1711 /
454 / 348 / 562 / 632 chars at 2/6/12/13/14/15 points (the 1711 is a repetition blow-up,
trap 2 surfacing despite `repeat_penalty=1.1`).

This is not a refutation of the cliff — it is the exposure-bias diagnosis restating
itself. **A clean, well-formed pool is TEACHER-shaped, which is exactly the distribution
every synthesis training row came from.** Real student memories are not clean. The pool
extracted from a real `run_agent` memory (`ivod-17684`, 13 points) carries a point
truncated mid-phrase (`台灣修憲使十八歲公民權門檻高且未解`) and two near-duplicate pairs.

On that student-authored pool:

| points | `v5` | `selfdistil-e3` | **`mixed-e3`** |
|---|---|---|---|
| 2 | 147 | 321 | 185 |
| 6 | 299 | 249 | 201 |
| 10 | 221 | 233 | **326** |
| 12 | 240 | **502** | 326 |
| 13 | 203 | **396** | 258 |

`v5`'s failure on real memory is better described as a CEILING than a cliff: output is flat
at ~200-300 chars regardless of occupancy, so 13 points still yields 203 characters — a
summary that cannot carry 13 points. At the occupancies that matter the ordering is
`v5` < `mixed` < `selfdistil`: mixing recovers part of the gain, not all of it.

Single seed, one pool — directional, not decisive. **Any future cliff number must say which
pool it came from; curves from different pools are not comparable.**

## Real-ASR standing gate (`tools/asr_gate.py`, 20 real zh-TW meetings)

| checkpoint | curated | NOP rate | mean chars |
|---|---|---|---|
| `v5` | 17/20 | 28% | 223 |
| `selfdistil-e3` | **19/20** | **8%** | **292** |
| **`mixed-e3`** | 17/20 | 23% | 246 |

**Mixing gave up most of self-distillation's ASR gain.** `mixed-e3` lands at `v5`'s curation
level. The NOP rate moves only 28% -> 23%, against self-distillation's 28% -> 8%.

This is the same shape as every other trade this project has measured (dropv4/dropv5's
late-step trade, `v3`/`v4`'s ASR-vs-G3 trade): diluting a signal dilutes its effect
proportionally rather than buying both ends.

## G3 (40 held-out meetings, `runs/mixed-heldout/`) — all three gates recovered

Zero failures on either arm. `--reduce-context-tokens 3000`, `--protocol tool`,
`--skip-failed-steps`; both arms off the same server, per `run_arms`'s fairness guarantee.

| gate | `v5` | `selfdistil-e3` | **`mixed-e3`** |
|---|---|---|---|
| rouge1 | PASS 28/12, +0.069, p=0.017 | FAIL 21/19, +0.011, p=0.875 | **PASS 28/12, +0.047, p=0.017** |
| rouge2 | PASS 29/11, +0.041, p=0.006 | FAIL 23/17, +0.014, p=0.430 | **PASS 29/11, +0.028, p=0.006** |
| rougeL | PASS 35/5, +0.057, p=0.000 | FAIL 26/14, +0.027, p=0.081 | **PASS 31/9, +0.037, p=0.001** |

Win counts on rouge1/rouge2 match `v5` exactly; effect sizes are uniformly SMALLER
(+0.047 vs +0.069 on rouge1). Coverage and density remain negative, as on every checkpoint.

## G1 independent reversal probe — 27 scenarios, all three re-measured with artifacts

**The first run of this was INVALID and looked like a decisive negative: 0/27.**
`tools/score_reversals.py` defaults to `--protocol edit`, and every v1.0 checkpoint is a
tool-call model. Under the wrong protocol every case failed on all three signals at once —
including `subject_present`, which is what gave it away, since a model producing real prose
about the right meeting cannot miss the subject 27 times out of 27. **Pass `--protocol
tool` for any v1.0 checkpoint; the default is wrong for every model on this branch.**

The recorded `3/27` (`v5`) and `12/27` (`selfdistil-e3`) had no artifacts behind them — both
were in-session numbers, exactly the gap `run_probe.py`'s docstring exists to close. Both
were re-run under `--protocol tool` and are now on disk:

| | `v5` | `selfdistil-e3` | **`mixed-e3`** |
|---|---|---|---|
| independent probe | **3/27** (reproduced) | **11/27** (recorded as 12/27) | **8/27** |

artifacts: `runs/qwen-tools-v5/revprobe27_report.json`,
`runs/selfdistil-e3/revprobe27_report.json`, `runs/mixed-e3/revprobe_report.json`.

## BEST-EPOCH vs LAST-EPOCH — the cheapest untested lever, and it is worth real points

Every v1.0 checkpoint ever shipped is a LAST-epoch artifact, because
`load_best_model_at_end` silently did not work (CLAUDE.md). Eval loss rises at epoch 3 on
every run in this project, so **every shipped checkpoint is past its own minimum** — and
nobody had ever measured what that costs, because the comparison needs no training at all.

`train_toolcalls.py` now copies the best checkpoint's files directly. Verified before
measuring, on an mlp weight that actually moves (`layers.12.mlp.gate_proj.weight`; note
`model.norm.weight` is a useless discriminator):

    final vs checkpoint-626 (best, epoch 2, eval 0.7814):  0.0        <- bit-identical
    final vs checkpoint-939 (last, epoch 3, eval 0.8691):  6.56e-04   <- genuinely different

| | last epoch (939) | **best epoch (626)** |
|---|---|---|
| real-ASR curated | 17/20 | **19/20** |
| real-ASR NOP rate | 23% | **15%** |
| real-ASR mean chars | 246 | **279** |
| G1 probe (27) | 8/27 | 8/27 |
| G3 rouge1 | PASS 28/12, +0.047, p=0.017 | **PASS 30/10, +0.053, p=0.002** |
| G3 rouge2 | PASS 29/11, +0.028, p=0.006 | **PASS 27/13, +0.031, p=0.038** |
| G3 rougeL | PASS 31/9, +0.037, p=0.001 | **PASS 31/8/1t, +0.039, p=0.000** |

**Best-epoch selection recovers the entire real-ASR gap to `selfdistil-e3` (19/20) for
free** — no new data, no new training, just exporting the checkpoint that was already on
disk. The revision probe is unchanged, so this is not a general "more training is worse"
effect; it is specifically the abstention/curation behaviour that epoch 3 degrades.

**And it is not a trade: G3 improved at the same time**, on all three effect sizes and on
two of three win counts. `runs/mixedbest-heldout/`, zero failures on either arm. An epoch
of overfitting was costing this project real points on two independent axes at once.

### But "use the best epoch" is NOT a general rule — it REVERSES on `v5`

The obvious generalisation ("every v1.0 checkpoint is past its minimum, so every recorded
number is understated") was written here and is **WRONG**. It was checked, because
comparing a best-epoch build against `v5`'s last-epoch export would have overstated the
win by exactly this effect. `v5`'s checkpoints survive; `final/` is bit-identical to
checkpoint-444 (last) and differs from checkpoint-296 (best) by 5.7e-04.

| `v5` | last epoch (444, shipped) | best epoch (296) |
|---|---|---|
| real-ASR curated | **17/20** | 16/20 |
| real-ASR NOP rate | **28%** | 36% |
| G1 probe (27) | **3/27** | 1/27 |

**`v5` is WORSE at its best-by-loss epoch, on both axes measured.** So epoch selection is
pool-dependent, not a universal lever: on `v5`'s pool the third epoch helped, on
`mixed-e3`'s it hurt. Eval loss does not order these checkpoints on anything this project
gates — which is the same lesson as `sft-dropv3`'s "stable SHARES did not imply stable
BEHAVIOUR", now restated for a stable LOSS.

**Practical rule: export BOTH epochs and measure. Do not infer either way from the loss
curve.** (G3 was not run for `v5`-best; it loses on both cheaper axes, so `v5`'s shipped
last-epoch export remains its best configuration and the fair comparison point below.)

## Standing position — `mixed-e3` BEST-EPOCH dominates `v5`, and is the new recommendation

**Use `runs/mixed-e3/gguf_best/` (checkpoint-626), not `gguf/` (checkpoint-939).**

| | `v5` | `selfdistil-e3` | **`mixed-e3` best-epoch** |
|---|---|---|---|
| G3 (3 gates) | PASS / PASS / PASS | FAIL / FAIL / FAIL | **PASS / PASS / PASS** |
| G3 rouge1 effect | **+0.069** | +0.011 | +0.053 |
| G1 probe (27) | 3/27 | **11/27** | 8/27 |
| real-ASR curated | 17/20, NOP 28% | 19/20, NOP 8% | **19/20, NOP 15%** |
| synthesis at 13 points | 203 ch | **396 ch** | 258 ch (last-epoch) |

`mixed-e3` best-epoch holds every G3 gate `v5` holds while nearly TRIPLING the independent
revision probe AND matching `selfdistil-e3`'s best-ever real-ASR result. It is weaker than
`v5` only on G3 effect sizes (+0.053 vs +0.069 on rouge1, both passing) and on memory
numerals (12.2 vs 15.4, measured on the last-epoch export).

**It does not pass G1** — 8/27 is not a passing gate, and SPEC §5.2 is all-or-nothing, so
the ship decision is unchanged at "ship the baseline". What changed is the best checkpoint
on the axis this project has failed on for four checkpoints across two model families.

**The transferable finding: the teacher/student synthesis mix does NOT trade linearly.** The
prediction going in was that mixing two provenances buys a point on the line between them.
It did not. G3 came back FULLY (`v5`'s win counts, to the meeting), while the probe kept
5 of self-distillation's 8-point gain and ASR kept almost none of its. **G3 and revision are
not on the same axis as ASR curation.** Do not model these as one quality dial.

## The ablation (`runs/ablate-e3`) — NEITHER change alone explains it

`mixed-e3` differs from `selfdistil-e3` by TWO things, so a single-variable pool was built:
`selfdistil-e3`'s exact rows PLUS only the 187 teacher-memory synthesis rows (4 teacher rows
whose memories collided with student ones were dropped — same prompt, different target is
contradictory supervision). Nothing else changed.

| pool | teacher rows | student rows | G3 | probe (27) | real ASR |
|---|---|---|---|---|---|
| `selfdistil-e3` | 0 | 341 unfiltered | **0/3** | **11/27** | 19/20, NOP 8% |
| `ablate-e3` | 187 | 341 unfiltered | **1/3** | 4/27 | 16/20, NOP 20% |
| `mixed-e3` | 187 | 263 filtered + 4x oversampled | **3/3** | 8/27 | 19/20, NOP 15% |

**Teacher rows alone recover only 1 of 3 G3 gates (rougeL), and cost revision and ASR
doing it** — 11/27 -> 4/27, 19/20 -> 16/20. The remaining recovery comes from the
student-row side: the `MIN_DET=2` quality floor that dropped 53 low-detail student memories,
plus the 4x oversampling of >=13-point rows.

**That filter was treated as housekeeping and is doing most of the work.** The mechanism is
consistent with the whole self-distillation story: a student-authored memory is only a good
synthesis INPUT if it actually contains something: training synthesis to produce a summary
from a near-empty memory teaches exactly the abstention and confabulation this project has
been fighting. Filtering the inputs matters more than balancing the provenances.

**Ablation G3 detail** (`runs/ablate-heldout/`, checkpoint-951): rouge1 24/16 +0.033
p=0.268 FAIL, rouge2 25/15 +0.029 p=0.154 FAIL, rougeL 29/11 +0.031 p=0.006 PASS. All three
effects positive; only rougeL is consistent. Same shape as `selfdistil-e3`'s failure.

### A new failure mode found on `ablate-e3`'s best epoch: bare `<think>` from synthesis

`ablate-e3` checkpoint-634 (best-by-loss) scores **0/27** on the probe because
`synthesize_memory` returns the literal string `<think>` and stops. The reading step is
FINE on the same input — it captured the reversal correctly
(`會議稍早公告的空氣品質淨區劃設案已改為暫不劃設`) — and the ASR gate on the SAME server
returned real 208-character summaries on 16/20 meetings. **The defect is synthesis-only and
appears on SMALL memories**, which is why the ASR gate does not catch it.

The base model's thinking behaviour re-emerges when the fine-tune's synthesis distribution
is weak for an input shape. The harness handles it correctly — `prose.finalize` flags
`insufficient zh-TW content (0.00 < 0.7 CJK ratio)` and `synthesize_memory` retries — so
this is a checkpoint defect, not a guard gap. **`lang_flags` is the field to check; an
earlier inspection here read a non-existent `language_flags` and wrongly looked clean.**
