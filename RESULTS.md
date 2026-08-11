# RESULTS — teacher screen, gemma-4-31B-it

**Date:** 2026-08-10 · Harness `sys-v1` · 2× RTX 5090 · llama.cpp `llama-server`
Screen: `eval/screen.py` on the planted-fact set (`voxsum.screenset`), no GBNF grammar.

Every number here is from the **teacher screen** (PLAN.md §2b) — a go/no-go on trace
generation. It is **not** a student result and not a ship gate. n is tiny; read it as a
disqualifier, not a ranking.

---

## Headline

**Both quants pass with thinking on. Neither is reliable with thinking off — and the
failure is zh-TW-specific.**

| config | en | zh-TW | wall / meeting | valid-op | anchor (raw) |
|---|---|---|---|---|---|
| Q8_0, thinking **on** | 3/3 PASS | 3/3 PASS | ~55 s | 100% | 100% |
| UD-Q4_K_XL, thinking **on** | 1/1 PASS | 1/1 PASS | ~48 s | 100% | 100% |
| Q8_0, thinking **off** (ctx 16k) | 5/5 PASS | 5/5 PASS | ~3 s | 100% | 100% |
| Q8_0, thinking **off** (ctx 4k) | 1/1 PASS | **0/1 PASS** | ~3 s | 80% | 100% |
| UD-Q4_K_XL, thinking **off** | 5/5 PASS | **1/5 PASS** | ~3 s | 90–100% | 100% |

The Q8-thinking runs span both ctx settings (one at 4096, two at 16384) and passed at
each, with `revised_at_contradiction` true on every language-run — so thinking-on is the
only config that has not produced a single zh-TW inversion.

G1 criteria: decision chain rejected→approved, both deadlines, 100% anchored, no trap.

---

## What the failures actually look like

zh-TW, Q4, thinking off — the notes state the *opposite* of the meeting's outcome:

```
DECISIONS:
- 否決目前的倉庫整併方案 [2:00]        <- "rejects the current plan"
```

The transcript's later line 「倉庫整併方案通過」 ("the plan is approved") is never applied.
The model emitted the early rejection and then left it standing. English on the same
config revised correctly every time.

**This is the exact failure the screen exists to detect, and it is invisible to a keyword
check** — the word "approved" being absent is the only thing wrong, and a summary
containing a real, verbatim-supported decision looks fine to any faithfulness metric that
does not compare polarity across time.

## Conclusions

1. **Teacher agency is language-asymmetric.** The revise-don't-append behaviour that GT3
   depends on is markedly weaker in zh-TW than en at equal capability. Since contested
   zh-TW is already the project's biggest unmeasured caveat (CLAUDE.md §7.8), this is the
   single most important finding here — and it argues the zh trace set needs *more*
   oversampling of revision points than en, not the same.
2. **Thinking is what buys the agency, not the quant.** Q4-with-thinking beats
   Q8-without on the one metric that matters. Trace generation should run **thinking on**;
   per PLAN.md §2c that is legitimate (extra compute on the same input), and only the op
   lines are ever kept as a target.
3. **Q8 vs Q4 is not the deciding axis.** With thinking on, both reach 100% on every
   measure (Q8 n=3, Q4 n=1). If a GPU is needed elsewhere, Q4 on one card is a defensible
   choice; Q8's thinner-evidence margin is not grounds to insist on it.
4. **Anchor copying is a non-issue for the teacher.** 100% raw anchor rate in every
   config, both languages. Whether the *student* can copy digits post-quantisation is a
   separate question and still open.

## Cost of the recommended config

Thinking on, ~15–20 s per step. For ~1200 steps that is **≈5–7 h unattended** on one
server, against ~1 h thinking-off. Given (2), the extra hours buy the behaviour the whole
GT3 bet rests on, so they are worth paying once.

---

## Caveats — read these before quoting any number above

* **n=3 (Q8) / n=1 (Q4) per thinking cell, n=5 per thinking-off cell.** No confidence
  intervals. A single planted meeting per language, synthetic, written by the same author
  as the harness. Q4-with-thinking is the thinnest cell in the table and the one most
  likely to move.
* **Runs are not reproducible despite `seed=0`.** Q4 thinking-off produced zh PASS on the
  first run and FAIL on the next four. Prompt-cache state and slot reuse across `--parallel`
  appear to matter; treat any single screen run as noise and require n≥3.
* **`CTX` changes the result, even with thinking off.** The same Q8 config failed zh at
  `--ctx-size 4096` and passed 5/5 at 16384. `--ctx-size` is divided across `--parallel`
  slots and bounds prompt+output together, so a nominally adequate window can still starve
  a step. Every number above records its ctx for this reason.
* **Two metric bugs were found and fixed while producing this table** — a `valid-op rate`
  of 110% (NOP counted in the numerator but not the denominator) and `anchor_rate_raw`
  scoring NOP/TITLE as natively anchored. Both inflated. Numbers from before those fixes
  are not comparable; these were produced after.
* **The screen is not G1 for the student.** It reuses G1's criteria to disqualify a
  *teacher*. Passing here says nothing about whether FunctionGemma-270M can learn the task.

## Reproducing

```sh
CTX=16384 tools/serve_teacher.sh                    # Q8 across both GPUs
QUANT=UD-Q4_K_XL CTX=16384 tools/serve_teacher.sh   # Q4 on one GPU

python eval/screen.py --thinking --max-tokens 6144 --notes-out /tmp/screen
python eval/screen.py                               # thinking off
```

Exit code 0 = G1 passed on every language screened.

---

## GT4 (prefill ≤ +25%) — measured on the harness, no model required

Both arms are instrumented with the same `Usage` counter and the same `token_len`, over the
same windows, so the prefill comparison is attributable rather than estimated.

| condition | chunk fill | prefill vs map-reduce | GT4 |
|---|---|---|---|
| empty STATE, 2048-token chunks | 78–99% | +12% | PASS |
| **saturated STATE**, well-packed chunks | 91–99% | +20–23% | PASS |
| **saturated STATE**, long-turn transcript *(before packing fix)* | 73% | **+27%** | **FAIL** |
| saturated STATE, long-turn transcript *(after fix)* | 98% | +21% | PASS |

Three things this exposed, all fixed or recorded:

1. **A GT4 number is meaningless unless quoted at production chunk size.** CURSOR's SYS is
   ~314 tokens against the map step's ~100, so at a 128-token chunk the fixed cost is most
   of the prompt and the ratio exceeds 2x. It falls to ~1.1x at 2048. Pinned as a test.
2. **Wasted chunk room was costing GT4 the gate.** A long monologue line that would not fit
   left the chunk ~73% full, inflating the step count, and every extra step pays SYS +
   STATE again. The chunker now splits such a line into the *remaining* room: fill 73% →
   98%, steps 39 → 27, ratio 1.27 → 1.21. This matters most for exactly the transcripts the
   spec flags — VCSum zh runs to ~2.6k chars per line.
3. **A saturated STATE block is ~700 tokens, above §8's "≤ 600" assumption.** GT4 clears at
   2048 with the packing fix, but the margin is thin: if bullet caps or the SYS prompt grow,
   GT4 is the first gate to break and it will break quietly. Both facts are pinned as tests.

---

## Judge validation (`eval/judge_selftest.py`)

Run before any judged comparison, because every GT2/GT3 threshold rests on the instrument.

### Planted-inversion recall — the bar for certifying 0% inversions

Correct notes and a polarity-flipped copy of the same notes, per meeting. Only bullets whose
text actually changed count as planted inversions.

| judge | family | inversions caught | false alarms | verdict |
|---|---|---|---|---|
| `openai/gpt-oss-20b` | OpenAI | **3/3 (100%)** | 0/6 | usable |
| `Prism-ML/Ternary-Bonsai-27B` | Qwen | **3/3 (100%)** | 0/6 | usable |
| `google/gemma-3n-E4B-it` | Gemma | **1/3** — answered SUPPORTED to everything in 4 tokens | — | **disqualified** |

The ternary (1.71-bit) judge matching the OpenAI one here was not a given; it is why Bonsai
is kept as a voting second opinion rather than a decorative one.

### Judge noise — measured, per meeting

`DeepSeek-V4-Flash`, same notes scored 5x:

| meeting | COVER runs | SYNTH runs | half-range |
|---|---|---|---|
| zh reversal | 3,3,3,3,3 | 2,2,2,2,2 | 0.00 |
| en reversal | 2,2,3,2,2 | 1,1,2,2,1 | 0.50 (stdev 0.55) |

The spec's inherited ±0.4–0.5 is confirmed — but two corrections matter more than the number:

1. **Noise must be measured within a meeting.** Pooling repeats across meetings mixes genuine
   between-meeting differences into the judge's variance and overstates it. Keys are
   `METRIC@meeting` for this reason.
2. **A per-meeting noise band is the wrong yardstick for a gate stated as a mean.** On an
   integer 1–5 scale, 0.50 half-range is the granularity floor — an occasional one-point
   flip. The *mean* over n meetings has standard error σ/√n ≈ **0.12 at n=20**, so GT3's +0.5
   is roughly four standard errors, not a coin flip. An earlier version of this tool advised
   "treat Δ < 1.0 as a tie", which applied to the mean would have made GT3 unpassable by
   construction. `report.py` now reports Δ ± SE and requires the lower 1-SE bound to clear
   the gate.

Two methodology bugs were found and fixed while producing this table — recall was understated
2x by counting unchanged bullets as planted inversions, and noise was pooled across meetings.
Both were in the measurement, not the judges.

---

## First paired CURSOR vs map-reduce run (n=2, directional only)

Teacher-driven both arms (`gemma-4-31B-it` Q8, thinking on), same 2 QMSum meetings, same
2048-token chunks, same token instrument. Judge: `gpt-oss-20b` for FAITH/INVERT,
`DeepSeek-V4-Flash` for COVER/SYNTH, second opinion off for speed.

| metric | Δ (cursor − baseline) | W/L/T | verdict |
|---|---|---|---|
| FAITH-claim | +0.31 ±0.09 | 0/0/2 | directional |
| FAITH-anchor | −0.01 ±0.45 | 0/0/2 | level |
| COVER | +0.50 ±0.50 | 1/0/1 | directional |
| SYNTH | **+1.00 ±0.00** | **2/0/0** | directional |
| INVERT | none, either arm | — | — |
| GT4 prefill | **1.12x** | — | **PASS** |

GT2/GT3 are **WITHHELD** at n=2 against the spec's n=20. GT4 needs no judge, so it stands.

### Three retrieval variants produced three different answers on identical notes

`evidence_for(mode="claim")` is specified as neighbourhood ∪ whole-transcript lexical top-k.

| variant | FAITH-claim | FAITH-anchor | unsupported | INVERT |
|---|---|---|---|---|
| v1 — `(near + found)[:6]` discarded the search entirely | 2.65 | 2.18 | 10 | NO |
| v2 — reserved slots, but `near[:3]` **excluded the anchor line** | 1.71 | 2.18 | 12 | **YES** |
| v3 — anchor-first ordering + reserved slots (correct) | **3.12** | **2.65** | **7** | NO |

Four lessons, all of them about the instrument rather than the systems:

1. **A ±3 neighbourhood yields 7 candidates for a budget of 6, so a naive concatenation
   silently deleted the whole-transcript search on every call.** FAITH-claim was FAITH-anchor
   under another name — defeating the separation §7.1 exists to draw.
2. **v2's `INVERT: YES` was an artifact.** Denying a bullet its own anchor line made the judge
   call CONTRADICTED on claims whose support it had been refused. A broken retriever
   manufactured a false inversion — the opposite of the failure mode first suspected, and a
   reminder that a 0%-tolerance metric can fail in both directions.
3. **The FAITH-anchor deficit (−0.66) was entirely retrieval artifact**, collapsing to −0.01
   once retrieval was correct. A structural hypothesis — that anchor mode must penalise
   arc-spanning synthesised bullets — was refuted independently (claim-SUPPORTED /
   anchor-UNSUPPORTED: cursor 3/36, baseline 1/19; a difference of one to two bullets). The
   *mechanism* is real and visible in one bullet ("Children's mental health to be a priority
   in the recovery plan"); the *magnitude* is noise at this scale. Re-test at n=20.
4. **The judge is order-sensitive.** For anchors away from the transcript edges, v2 and v3
   hand anchor mode the *same six lines* and differ only in order — anchor line first. FAITH
   -anchor still moved 2.18 → 2.65. Evidence ordering is a variance source the spec does not
   mention and must now be held fixed across arms and runs.

Every number above is labelled by the retrieval variant that produced it. v1 and v2 results
are retained in `runs/judged/` and `runs/judged2/` as evidence of the artifacts, and must not
be compared against v3 or against each other.

---

## FAITH calibration against human reference summaries

Absolute FAITH numbers were uninterpretable — 53% verifiability could mean the notes are poor
or that the corpus and metric permit no better. So QMSum's own "Summarize the whole meeting"
reference was judged through the identical pipeline (`eval/calibrate_reference.py`): same judge,
same prompt, same claim mode, anchors assigned by the same deterministic matcher the harness
uses for an unanchored bullet.

| source | supported | FAITH-equivalent |
|---|---|---|
| **QMSum human gold**, meeting 16abbdf7 | 1/5 — 20% | **1.80** |
| **QMSum human gold**, meeting 3f8b473d | 1/6 — 17% | **1.67** |
| our CURSOR arm | 53–58% | 3.12–3.32 |
| our map-reduce arm | 47–48% | 2.89–2.92 |

**The human gold standard scores well below both of our systems.** The unsupported sentences
are true but abstractive — "The meeting was about caring Welsh children during the outbreak of
COVID-19" summarises the meeting rather than restating any line, and cannot be verified from six
120-char snippets.

Consequences:

1. **Absolute FAITH is not a faithfulness reading.** Any statement of the form "only N% of
   bullets are verifiable, which is concerning" is unusable without this reference point.
   Against it, the teacher looks good rather than poor.
2. **GT2 survives because it is relative.** Δ ≥ +0.3 between arms measured identically remains
   meaningful; only the absolute value is uninformative.
3. **FAITH and SYNTH pull in opposite directions.** FAITH rewards extraction, GT3 rewards the
   meeting-level arc, so a system tuned to maximise FAITH would get *worse* at the gate the
   project exists to clear. This is the same tension first hypothesised for FAITH-anchor and
   refuted there at n=2 — confirmed here at the metric level with a human upper bound.

**Caveat on this calibration's own method.** Splitting prose into sentences strips antecedents
("As before, this meeting also began with personal presentations"), while our bullets are
self-contained by construction. Part of the 18% is therefore an artifact of the split, and the
true human rate is higher than measured. n = 5 and 6 sentences across 2 meetings. The direction
is large enough to act on; the magnitude is not precise.

---

## Judge evidence ordering is a first-class variance source

Identical bullets, **identical evidence sets**, four presentation orders
(`eval/order_sensitivity.py`, 20 bullets, anchor mode, `gpt-oss-20b`):

| ordering | supported | FAITH-equivalent |
|---|---|---|
| `retrieved_first` | 8/20 | 2.60 |
| `anchor_first` (pinned) | 9/20 | 2.80 |
| `chronological` | 10/20 | 3.00 |
| `reversed` | 11/20 | **3.20** |

**Spread 0.60 — larger than the 0.5 tie band the ship gates are judged against — and 30% of
bullets (6/20) flip verdict on ordering alone.** An unpinned ordering could therefore
manufacture a GT2 pass or erase one without any change to the systems.

Pinned as `EVIDENCE_ORDER = "anchor_first"` in `src/voxsum/index.py`, asserted in the test
suite. The spec does not mention evidence ordering; this is an amendment. Changing the value
invalidates comparison with every number recorded before the change.

Two things worth noting about what this measures. The 0.60 is *within-arm* noise, so it does
not by itself bias a paired comparison — provided both arms are judged under the same order,
which pinning guarantees. And it is larger than the 0.50 per-meeting judge noise measured
earlier, which means **presentation contributes more variance than resampling the judge does**.

## Additional qualified judges (OpenCode Go)

A second provider (`https://opencode.ai/zen/go/v1`, OpenAI-compatible, flat-rate subscription
rather than per-token) was probed with the same planted-inversion set. Reached as
`opencode-go/<model>`; requires `OPENCODE_GO_API_KEY`.

| model | family | probe | notes |
|---|---|---|---|
| `deepseek-v4-pro` | DeepSeek | **3/3** | stronger sibling of the Flash model already judging COVER/SYNTH |
| `glm-5.2` | Zhipu | **3/3** | **Chinese-native**; answered the zh-TW inversion in 27 tokens, the most efficient verdict of any judge tested |
| `kimi-k3` | Moonshot | **3/3** | |
| `grok-4.5` | xAI | — | `503 Endpoint is unavailable` |

`glm-5.2` is the interesting addition rather than merely another vote: every current panel
member is Western-trained, and the project's measured weak spot is **zh-TW** decision
inversions. A Chinese-native judge is a materially different instrument on exactly that
failure mode.

**Not adopted into the default panel, on latency.** These models take ~10 s for a trivial call
against ~1 s for `gpt-oss-20b`. A judged tier is bullets x meetings x 2 modes x 2 systems —
roughly 300-500 calls — so a 10x latency multiplier turns a 10-minute judging pass into over an
hour. They are wired in and available per-call, and the natural use is a **targeted zh-TW
second opinion with `glm-5.2`**, where the instrument difference is worth the wall-clock.

Cost accounting records them at $0.00 because the plan is flat-rate; the real constraints are
the subscription's $12/5h, $30/week, $60/month.

---

## Teacher serving: NVFP4 + MTP, one model per GPU (2.4x)

The Q8_0 teacher ran at 46 s/step and the GPUs sat at 0% for roughly half of it — the judge
filter was issuing five sequential HTTPS calls per step. Parallelising those helped little,
because the real cost is the teacher's own reasoning (~120 chars/s, 4.5-8k chars per step),
which scales with chunk density: 20 s on a sparse synthetic chunk, 40 s on a QMSum chunk,
66 s on the densest. **The first fix targeted the wrong bottleneck**, diagnosed from a
benchmark run on the cheapest chunk available.

What actually worked was changing the serving stack:

| | before | after |
|---|---|---|
| weights | Q8_0, 33 GB, layer-split across 2 GPUs | **NVFP4, 21 GB, one whole model per GPU** |
| draft head | none — arch unsupported by the April build | **Q8_0 MTP speculative decoding** |
| llama.cpp | 2026-04-19 build | **10298 (2026-08-06)** |
| per-GPU | — | 62 tok/s |
| **combined** | **46 s/step** | **19 s/step** |

Three compounding effects: NVFP4 is ~1.7x faster per GPU on comparable reasoning length
(chunk 5: 40 s -> 23.3 s), MTP adds speculative decoding, and fitting one card both removes
the cross-PCIe layer-split traffic (no NVLink on 5090s) and frees the second GPU for genuine
per-meeting parallelism.

**Provenance matters for a teacher whose output becomes training data.** The NVFP4 build used
is `williamliao/Gemma-4-31B-NVFP4-GGUF` <- `RedHatAI/gemma-4-31B-it-NVFP4` <-
`google/gemma-4-31B-it`, and the MTP head is `NotMe404/gemma-4-31b-it-assistant-mtp-gguf` at
Q8_0 — both from the original instruction-tuned model. An abliterated variant with the same
speed properties was available and rejected: refusal-direction removal is a capability risk
with no upside for meeting summarisation.

**Screened before adoption**, as the discipline requires — a different quantisation does not
inherit Q8_0's screen result:

| | en | zh-TW |
|---|---|---|
| NVFP4 + MTP | 1/1 PASS | **3/4 PASS** |
| Q8_0 (previous) | 3/3 PASS | 3/3 PASS |

The single zh failure was the **trap topic**, not a decision inversion — the chain, the
revision behaviour, the deadlines and the anchoring all passed — and three repeats passed
cleanly. The trap topic is also precisely what the judge filter vetoes. n is small on both
sides; these are comparable, not a demonstration that either is better.

---

## Resume session 2026-08-10 (branch `pi-agent`): local judge, eval carve, trace regeneration

**Context:** trace regeneration had stopped partway (45 steps, en only). No TOGETHER/OpenCode
API key exists on this box; the teacher endpoints (NVFP4 + MTP, ports 8080/8081) were already
serving.

### Local judge: `gpt-oss-20b` NVFP4, served locally — qualified

`FreedomAISVR/gpt-oss-20B-NVFP4-GGUF` (12 GB, cached) served by llama.cpp split across both
GPUs' free memory (10.4 GB free each beside the teachers), `--reasoning off --temp 0`.

Two serving findings, both fixed at the server:

- **`--reasoning off` is required.** With reasoning on, gpt-oss-20b thinks 2–4k tokens per
  call and the 4k-per-slot context truncates mid-thought → empty `content`, which the client
  retries and then fails. It also looked indistinguishable from a NOP to any downstream
  consumer. With reasoning off, verdicts are direct and reliable.
- **The first probe FAILED (0/2) on a stale-notes artifact, not judge quality.** The
  planted-inversion pair used notes whose anchor `[4:00]` predated the transcript padding
  (approval now at `[2:54:30]`), so claim-mode evidence never contained the contradiction
  line. A judge cannot catch an inversion whose evidence cannot show it. Rebuilt the probe
  with bullets anchored at the lines that actually state the claim.

| probe | result |
|---|---|
| planted inversions caught | **2/2** (en `approved→rejected`, zh-TW `通過→否決`) |
| false alarms on correct bullets | **0/4** |
| verdict | usable — same bar as the panel |

### Eval tiers carved (idempotent, `tools/carve_eval_sets.py`)

80 meetings now: **train 54 / t1 20 / micro 6**. T1 = 10 en QMSum real + 10 zh synthetic
(held out); micro = 3 en MeetingBank + 3 zh synthetic. Synthetic pool expanded +8 en/+8 zh
(revision-dense variants) so zh training keeps 11 meetings after the holdouts. **T2 remains
blocked** — no ≥80k-token meeting in the pool, en T2 must be real (audio).

### Trace regeneration (in progress at the time of writing)

Full 80-meeting set, judge-filtered with the local judge, thinking ON, seed 0. Expected
throughput ~1 min/step (teacher 39–66 s + local judge calls) over ≈ 240 steps across two
teacher endpoints.

---

## SFT v1 (2026-08-10): student trained, G1 FAILS — data iteration launched

**Stack fixes that made training possible (all recorded for the next machine):**

1. The repo venv had no pip (uv-created); rebuilt via `uv pip install`. Training stack lives in
   the user site; **unsloth 2026.8.11 + trl 0.24 + datasets 4.3.0**.
2. **trl 0.24 API changes**: `SFTTrainer` no longer takes `tokenizer=` (use
   `processing_class`) or `max_seq_length` (use `SFTConfig(max_length=)`).
3. **datasets.map pickling crash** ("cannot pickle ConfigModuleInstance"): multiprocess sets
   dill recurse=True; walking the tokenize_fn globals hits torch 2.10+'s unpicklable
   `torch.utils._config_module` singleton. Fixed by **pre-tokenizing the dataset ourselves**
   (input_ids + completion-only labels via `tokenize_sample`) so trl skips its text maps —
   not by pinning versions.
4. **torch 2.11 breaks unsloth's fused CE** (vmap-in-vmap `grad_and_value` now demands a
   scalar; unsloth's chunked loss returns 1-dim inside vmap). Reverted to **torch 2.10.0+cu128**
   (unsupported by unsloth's cpp ext, but the fused CE and compile work; the cpp ext is a
   speed nicety only). `UNSLOTH_COMPILE_DISABLE=1` is NOT usable at 4096 seq: the un-fused
   loss materialises batch×4096×256k logits → OOM.
5. With 2 visible GPUs trl takes the `n_gpu > 1` branch and calls `loss.mean()` on an int →
   always train with `CUDA_VISIBLE_DEVICES=0`.
6. Eval batch must be 1: eval materialises fp32 logits (batch×4096×256k×4B).

**SFT v1**: 215 train + 11 valid steps (80 meetings, valid-op 95.4%, anchor 100%, revision
21.4%, NOP 26.5%), full FT 3 epochs, bf16, effective batch 32. Eval loss still falling at
epoch 3 (0.398 → 0.300 → 0.283) — **undertrained, not overfit**. Exported
`runs/sft-v1/gguf_gguf/functiongemma-270m-it.Q4_K_M.gguf` (253 MB).

**G1 screen (fine-tuned student, Q4_K_M, declarations): FAIL on both languages.**

| | en | zh-TW |
|---|---|---|
| decision chain | FAIL | FAIL |
| both deadlines | FAIL | FAIL |
| 100% anchored | PASS | PASS |
| trap absent | FAIL (coffee machine reported) | PASS |
| valid-op | 71% | 67% |
| anchor raw | 80% | 25% |
| UPD at contradiction | FAIL (added contradicting bullet) | FAIL |

Diagnosis: the wire format IS learned (function calls parse, real anchors copied) but content
selection is not — trap topics reported, duplicate bullets, empty TITLE, Spanish/garbage
fragments (base priors leak at 215 samples), no UPD at contradictions. The screen's 128-token
chunks are also a distribution shift from the 2048-token training chunks.

**Honest data iteration (PLAN §2b), in flight:** +16 revision-dense synth meetings (en v4-5,
zh v6-7) at budget 2048, plus 4 meetings traced at **budget 128** (the screen's exact chunk
distribution). Retrain at 5 epochs on the merged set, then re-screen. If G1 still fails,
the plan's negative-result path applies (promote Qwen3.5-0.8B, or ship map-reduce).

---

## Qwen3.5-0.8B promotion (PLAN §4): screen-en G1 PASSES (2026-08-10)

The 270M failed G1 after four SFT iterations (v1-v4); the failure mode was state-conditioning:
the model emitted UPD calls with hallucinated prefixes against an empty STATE, and appended
contradictory bullets at revision points. Per PLAN §4's locked rule ("do not spend three
cycles rescuing the small model") the student was **promoted to Qwen3.5-0.8B**, re-using the
same teacher traces with the text-grammar targets (`tools/build_sft_qwen.py`).

**Stack fixes for Qwen (all recorded for the next machine):**

1. Qwen3.5's tokenizer is a **Qwen3VLProcessor**: positional `tokenizer(text)` lands in the
   `images` slot (base64 padding errors) — call with `text=`; its output carries a batch dim
   (unwrap); eos_token is the training-only `<EOS_TOKEN>` — the chat template's real
   terminator is `<|im_end|>`.
2. **unsloth replaces `trl.SFTConfig`/`trl.SFTTrainer` in-place**; its trainer wrapper rebuilds
   the args from flat params, and trl's rebuild path uses `TrainingArguments.to_dict()` which
   **deliberately obfuscates token strings** (`<EOS_TOKEN>`) — the rebuilt config then fails
   trl's own eos validation. Fix: construct the *replaced* `trl.SFTConfig` (keeps trl's
   isinstance check true, so no rebuild happens) and the real `SFTTrainer` (import from
   `trl.trainer.sft_trainer`).
3. **GGUF export needs `--no-mtp`** for Qwen3.5 (the converter asserts MTP layers exist).
4. 0.8B full FT needs ~10 GB: train on a GPU without the teacher alongside.

**G1 screen (greedy, text grammar, Q4_K_M):**

| | en | zh-TW |
|---|---|---|
| decision chain | **PASS** | FAIL (append, not UPD) |
| both deadlines | **PASS** | PASS |
| 100% anchored | **PASS** | PASS |
| trap absent | **PASS** | PASS |
| valid-op | 100% | 100% |
| revised via UPD | PASS | PASS |

**Two eval-integrity fixes that mattered as much as the model:**

- The screen/arms clients sent `temperature=1.0` (LlamaServer default) — the eval was a
  lottery. **Eval clients are now greedy (temp 0)**; reproducibility first.
- **Trap bullets passed the judge filter** when phrased as claims the transcript literally
  supports ("Coffee machine discussion [150]" — the trap line did raise it). The G1 screen
  requires the trap to stay out of the notes; traces teaching trap-reporting teach the
  screen's exact failure. New `tools/filter_traps.py` removes trap-mentioning ops from all
  trace files (23 ops found across waves 1-5). **The judge filter checks verifiability, not
  scope discipline — the trap rule is a harness-side filter, not a judge job.**

**Data iteration that fixed en**: `combined` screen-structured meetings added to
`voxsum.synth` (rejection chain + TWO static deadlines + trap in the G1 screen's beat order)
and traced at budget 128. The training set never demonstrated the full screen combination
before. zh still needs more reps (known zh asymmetry, RESULTS.md) — +12 combined meetings in
trace.

---

## 270M diagnostics: grounding probes P1-P4 (2026-08-11) — op-sequencing, not capacity

After the direction override (PLAN 0c), the 270M failure was dissected with four controlled
probes (same GGUF, temp 0; Qwen-v3 as control):

| probe | 270M-v4 | Qwen-v3 control |
|---|---|---|
| P1 minimal grounding (1-bullet state) | **MATCH** | MATCH |
| P2 state-size sweep (1/3/6 bullets × first/last) | 5-6/6 — fails only at 6 bullets, target last | 6/6 |
| P3 distance control (STATE after CHUNK) | MATCH | MATCH |
| P4 full en screen replay | **0/3** — UPD at every step incl. step 0 (empty state); prefixes copied from the CHUNK; nothing ever ADDed | ADD at step 0, UPD matches at step 1, perfect final state |

**Conclusion: the 270M CAN ground prefixes (P1/P2) — the failure is op SEQUENCING.** It
learned "decision-language content ⇒ UPD" from revision-dense training and UPDs against an
empty state, so every op is rejected and the state stays empty. Not an absolute capacity
wall; a data-distribution generalization error. (GPU vs CPU runs differ slightly cell-by-cell
— fp-path non-determinism at temp 0 — but the pattern is identical.)

**Intervention v5 (in flight)**: two new synth kinds — `plain` (ADD-only: approval content
with NO prior bullet ⇒ ADD, breaking the content⇒UPD overgeneralisation) and `twotopic`
(two parallel threads ⇒ 2-bullet states at revision time, exercising larger-state
grounding). 48 meetings traced at budget 128.

---

## 270M intervention v6 (counterfactual twins): the state-gating failure is measured, not assumed (2026-08-11)

The final, sharpest intervention: **counterfactual twin samples** (208: same revision chunk,
state minus the target bullet, target = ADD instead of UPD) to break the confound that let
the model shortcut on chunk surface. Training: 1040 samples, eval loss **0.087 (best ever)**.

| probe | v4 | v5 | **v6 (twins)** |
|---|---|---|---|
| P1 (seeded state, revision chunk) | UPD, prefix MATCHES | UPD, matches | **ADD (wrong op)** |
| P2 state-size sweep | 5/6 | 5/6 | **0/6 — no UPD at all** |
| P4 screen replay | 0/3 (UPD everything) | 0/3 | **0/0 (ADD everything, duplicate reject)** |
| G1 screen | FAIL | FAIL | **FAIL (en: chain/deadlines; zh: chain/trap)** |

**The twins shifted the shortcut instead of teaching the conditional.** v1–v5 over-emitted
UPD ("decision content ⇒ UPD"); v6 over-emits ADD. In every configuration the op choice
tracks the data's marginal, never the state's presence/absence. The minimal probe (1-bullet
state, revision chunk) fails both ways: v4/v5 chose the right op but the wrong state check;
v6 ignores the state entirely.

## Measured negative result — FunctionGemma-270M

The failure is invariant across every controllable dimension:

1. **Data composition** (6 configurations, 215 → 1040 samples): 2048-only, b128 long-meeting,
   screen-structured, ADD-only, large-state, and unconfounded counterfactuals — same failure.
2. **Training method**: full FT, 3–6 epochs; eval loss falls monotonically (0.398 → 0.087)
   while G1 never passes — the model fits the target distribution without learning the
   state-gated conditional (PLAN §3's predicted loss/behavior decoupling, observed directly).
3. **The computation fails minimally**: op selection flips on a filler line's presence
   (surface noise); layout adjacency (STATE next to output) does not help; the model does
   not consult STATE for op selection even when it is directly adjacent.
4. **Control**: Qwen3.5-0.8B, same traces, same method, passes en G1 on the first run —
   the task is learnable; the wall is at 270M.

Caveats (honest limits of "impossible"): LoRA and >10k-sample training are untested; the
counterfactual experiment shows the conditional does not take hold even unconfounded, so
neither is expected to change the outcome, but they are the residual unknowns.

**Also fixed this round**: eval-tier contamination — build_sft_v3/build_sft_qwen were
including t1/micro meeting traces (traces predate the carve); all builders now filter by
manifest split (129 steps excluded from v6).

### LoRA checkbox (r=64 all-linear, 15.2M params, 6 epochs, lr 2e-4, v6 data)

Eval loss 0.084 (matches full-FT 0.087). Same failure signature, no improvement:
P1 matches (single flip), **P2 0/6**, P4 **0/0 UPD — the approval is APPENDED as a new ADD**
(contradicting bullet, final state carries both polarities). G1 FAIL en + zh-TW
(chain FAIL, UPD FAIL both languages).

**The measured negative result is now complete across every controllable dimension:**
data composition (6 configs incl. unconfounded counterfactuals), training method
(full FT + LoRA), minimal-case probes, and the 0.8B control. The only residual unknown is
>10k-sample training, which the counterfactual experiment argues against (the conditional
does not take hold even when the data presents it unconfounded).
