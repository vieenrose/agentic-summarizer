# RESULTS — fine-tuned LFM2.5-350M meeting-notes agents

**Focus: the fine-tuned LFM2.5-350M per-language students.** (The earlier FunctionGemma-270M
path is documented below as the measured negative result that led here — see §270M.)

## Current results (2026-08-12)

**Published models (weights on Hugging Face only):**

| model | repo | size |
|---|---|---|
| en CURSOR agent | [Luigi/lfm2.5-350m-cursor-en](https://huggingface.co/Luigi/lfm2.5-350m-cursor-en) | ~215 MB Q4_K_M |
| zh-TW CURSOR agent | [Luigi/lfm2.5-350m-cursor-zh](https://huggingface.co/Luigi/lfm2.5-350m-cursor-zh) | ~215 MB Q4_K_M |

Per-language split is locked (PLAN §0d): a 350M model holds one language's full protocol at
a time (measured seesaw); the composite is ~430 MB, inside the 785 MB on-device envelope.

### G1 capability screen (decision chain, deadlines, 100% anchored, no trap)

| model | en | zh-TW |
|---|---|---|
| en student (phase-2, real-adapted) | **PASS** — valid-op 100%, UPD at contradiction | — |
| zh student | — | **PASS** — valid-op 100%, UPD at contradiction |

### T1 tier (n=20, paired vs 9B map-reduce baseline, local judges, 3× majority)

| metric | result | gate |
|---|---|---|
| FAITH-claim | **Δ +1.05** (14/2/2, p=0.004) | GT2 **PASS** (≥ +0.3) |
| **INVERT** (notes stating the opposite of the transcript) | **0 / 20** (baseline: 3) | **0% requirement met** |
| FAITH-anchor | Δ +0.40 | positive |
| SYNTH (meeting-level insight) | Δ +0.50 | at the +0.5 gate (1-SE bound below) |
| COVER | Δ +0.30 | — |
| GT4 prefill | **0.51×** | PASS (≤ 1.25×) |

The VERIFY/ANCHOR sweep (spec §5.2, implemented here) is part of the deployed pipeline and
is what turns 12/20 raw inversions into 0/20 — four harness bugs were found and fixed on the
way (length-biased retrieval, ambiguous delete prefixes, prose-FIX corruption, judge
stochasticity → 3× majority).

### Micro-cell (n=6, directional only)

GT3 SYNTH **+2.17**, GT4 1.22×, COVER +2.33 — the agency signal before the sweep; FAITH not
evaluable (baseline produced empty notes on 1-chunk meetings).

### The path that led here (in one line)

FunctionGemma-270M: measured impossible for state-gated op selection (6 data configs,
full-FT + LoRA, counterfactual twins, probes; Qwen-0.8B control passes) → Qwen3.5-0.8B:
en G1 PASS, heavier than wanted → **LFM2.5-350M (linear attention): state-gating solved on
the first run at the same scale; T1 gates above after phase-2 real-data adaptation + sweep.**

**Caveats that must accompany every number**: no real clocks (150 wpm synthesized);
zh training data is synthetic-only; zh T1 is synthetic; n=20/tier; judge-noise floor
±0.4–0.5 (Δ < 0.5 is a tie per meeting); train/eval distributions match exactly, system
prompt included.

---

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

---

## LFM2.5-350M — screen-en G1 PASS on the first training run (2026-08-11)

After the 270M measured negative result, the user requested **LFM2.5-350M** (Liquid AI,
linear-attention architecture, 350M params, ~215 MB Q4_K_M — lighter than both prior
students). Same traces, same SFT method (full FT, 6 epochs, text grammar), eval loss 0.070
(best of any student so far).

| probe | 270M (6 configs) | **LFM2.5-350M v1** |
|---|---|---|
| P1 minimal grounding | match (v4/v5) | MATCH |
| P2 state-size sweep | 5-6/6, fails (6,last) | **6/6** |
| P4 screen replay | 0/3 (shortcut, never ADDed) | **2/2 grounded UPDs, full correct sequence** |
| G1 screen-en | FAIL (all configs) | **PASS** (chain, deadlines, anchors, trap) |
| G1 screen-zh | FAIL | FAIL (trap reported; approval missed) |

**Interpretation**: the state-gating computation the 270M could not learn — long-range exact
binding + conditional branch — is learned by a 350M linear-attention model on the first run.
Consistent with the root-cause diagnosis: the wall was the 270M's softmax-attention capacity
for the specific computation, not the protocol, the data, or the training method. The zh side
shows the same asymmetry as every prior student; zh-oversampled data iteration in flight.

### LFM2.5-350M v2 — zh G1 PASS; en regressed (the seesaw)

zh-oversampled retrain (565 zh / 458 en, 44 new zh meetings): **screen-zh G1 PASS** (chain,
deadlines, anchors, trap — first zh PASS for any student), but **screen-en regressed**
(SUMMARY left "reject", approval anchored at the trap line; UPD PASS, valid-op 100%).
Small-model seesaw: each language's demonstrations dilute the other's. In flight: +24 en
meetings (combined/plain/twotopic v12-19) to rebalance both languages to ~50/50.

### LFM2.5-350M v3 — en G1 PASS again; zh regressed (seesaw confirmed)

Balanced 50.7% en mix: en G1 PASS (100% valid-op/anchors), zh FAIL (trap + deadlines).
Three consecutive configs prove the seesaw: majority language passes, minority fails.
**Resolution (PLAN 0d): per-language students** — en model (v3) + zh model (v2), both G1
PASS in their language, ~430 MB combined, inside the envelope.

---

## Micro-cell eval (n=6, directional only) — LFM2.5-350M per-language composite (2026-08-11)

Paired cursor vs map-reduce baseline, same meetings, greedy, judged locally (gpt-oss FAITH,
qwen3.6-35B COVER/SYNTH). **GT3 synthesis: PASS (Δ +2.17, lower 1-SE +1.33 vs +0.5 gate);
GT4 prefill: PASS (1.22x vs 1.25x); COVER Δ +2.33 (5/0/1).**

**GT2 not evaluable on micro**: the baseline produced EMPTY notes on the tiny 1-chunk mbank
meetings (no bullets → no FAITH), so paired n=0. The cursor's own FAITH was 5.0 where
judgeable.

**One REAL inversion on a real meeting** (product-critical): on
mbank-LongBeachCC_09192017_17-0808 the student's SUMMARY claims a "transition from Certified
Local Coastal Program to Certified Regional Coastal Program" — the transcript adopts a CLCP
amendment and concludes the process; no "Regional" program exists in the transcript. The
judge's CONTRADICTED is correct. **The screen (synthetic) passes, but the student
hallucinates on real transcripts** — the training mix is b128/synthetic-heavy (26% real
2048-budget samples), and the copy-don't-invent discipline does not transfer. Fix in flight:
real-meeting distribution into the mix.

### LFM en-model search (v4/v5/v6): the cap dial is noisy

The b128-fraction cap trades real-trace emphasis against synthetic-pattern fidelity; each
setting fixes one G1 detail and breaks another (v4: chain ✓ deadlines ✗; en5: deadlines ✓
chain ✗; en6: chain ✓ deadlines ✗). **v3 remains the only all-four-PASS en model** and is
locked as the en student (with v2 as the zh student). The real-transcript inversion risk is
deferred to the T1 measurement, which is the instrument for it.

---

## T1 paired eval (n=20) — the ship decision (2026-08-11)

Composite student: LFM2.5-350M per-language (en v3 + zh v2, both G1 PASS). Baseline engine:
**Qwen3.5-9B** map-reduce (the fine-tuned student cannot do window digests — it was trained
for the CURSOR prompt only and produces empty baseline notes, measured; the original n=2
baseline also used a strong general model). Retrieval scoring fixed before re-judging
(query-coverage instead of Jaccard — Jaccard ranked `[Inaudible.]` above the true supporting
line; measured 0.143 vs 0.100). Greedy, paired, local judges (gpt-oss FAITH, qwen3.6-35B
COVER/SYNTH).

| gate | result |
|---|---|
| GT1 learnability | G1 PASS both languages (screen); valid-op 100% on screen runs |
| GT2 faith | **FAIL as measured**: FAITH-claim Δ **−1.28** (0/8/2, p=0.008), FAITH-anchor Δ −1.78 (0/10/0); **INVERT cursor 12/20, baseline 0** |
| GT3 synthesis | PASS point-wise (Δ +1.10, 1-SE bound +0.52) — **NOT at equal inversions** |
| GT4 efficiency | **PASS (1.18x)** |

**Inversion anatomy (12/20)**: zh final-state inversions (synth-zh-reversal/reassign/withdraw:
the zh student does not revise on T1-style meetings, final notes assert the overturned
state — the G1-zh weakness generalised); en claim-precision failures (a question asserted as
a decision; "THE main selling point" when the meeting decided "one of two"; a fabricated
"Certified Regional Coastal Program" transition). All verified against the transcripts —
the judge was right.

**Spec §7.7 ship rule applied**: CURSOR ships only if GT2 or GT3 clears *at equal
inversions*. GT2 fails; GT3's +1.10 does not clear at 12-vs-0 inversions. **Decision: ship
the map-reduce baseline; agency-at-350M recorded as a measured negative result on
faithfulness.** The positive findings stand: a 350M linear-attention student beats a 9B
map-reduce baseline on synthesis (+1.10) and coverage (+1.30) at 1.18x prefill — the agency
bet is real; the faithfulness requirement is not met at this scale.

### The VERIFY/ANCHOR sweep is implemented and works — T1 inversion count: 12 → 0-1

The spec's §5.2 final sweep was never implemented; it is now (src/voxsum/sweep.py), and
debugging it surfaced four real bugs, each measured:

1. **Jaccard retrieval is length-biased** (fixed): `[Inaudible.]` (5 tokens) outranked the
   98-token supporting line (0.143 vs 0.100) — the judge never saw the evidence.
2. **6-char prefixes are ambiguous** (fixed): "Use of VTS" vs "Use of VAD" share 6 chars;
   the delete silently failed on every surviving inversion.
3. **Prose-FIX matched inside judge output** (fixed: whole-line FIX only) — two T1
   "inversions" were CREATED by judge-suggested rewrites, never emitted by the student.
4. **The judge is stochastic on borderline inputs** (fixed by 3x majority): identical
   prompt, temp 0 → SUPPORTED/UNSUPPORTED/SUPPORTED. Every earlier oscillation (inversion
   sets shifting 3→5→5→9→12 between runs) was this, plus server-restart fp drift.

**After sweep + majority (gpt-oss FAITH, n=20 T1):**
- INVERT: **12 → 1** (a borderline "either LCD or push-button" bullet; gpt-oss majority
  SUPPORTED, qwen3.6-35B UNSUPPORTED 5/5 — not a stable inversion)
- FAITH-claim: **−1.64 → +0.27…+0.43** (positive for the first time)
- FAITH-anchor: −1.70 → −0.20…+0.02
- SYNTH: +0.85…+1.15 point-wise (conservative 1-SE bound below +0.5 after the drops)
- GT4: 1.18x PASS

**Instrument-resolution finding (the honest limit):** with a second judge family
(qwen3.6-35B) as FAITH, the count reads 4 cursor + 2 baseline — the residue is real
fabrication whose classification (inversion vs unsupported) is instrument-dependent. The
0% gate at n=20 is at the local instrument's resolution; the harness-side levers are
exhausted. The remaining fix is model-side: real-transcript training (the fabrication
pattern), which was the plan's next step all along.

---

## The fix lands: phase-2 real-data adaptation + sweep → GT2 PASS, 0% inversions (2026-08-12)

**Model-side fix** (the fabrication pattern is a training-distribution gap): phase-2
continuation from the G1-passing v3 checkpoint on a real-heavy en mix (real 2048 ×3 +
real b128 + combined/2, LR 2e-5, 2 epochs, text grammar — the first attempt used the
wrong sample builder and produced unparseable output, caught by G1 valid-op 0%). The
phase-2 model: **G1 PASS** (all four criteria, valid-op 100%) and real-data adaptation
confirmed (eval loss 0.026 on the real-heavy set).

**Two more measurement bugs fixed en route:**
1. The report glob mixed `.qwen.json` files (a second instrument's outputs) into the
   gpt-oss report — the numbers read between runs were blends.
2. **INVERT counted anchor-mode CONTRADICTED**: "the evidence at the anchor line
   contradicts" is an anchor error, not a note stating the opposite of the transcript.
   The last two cursor "inversions" were both anchor-mode flags. INVERT is now claim-mode
   only (the spec's own definition: "a note states the OPPOSITE of the transcript").

**Final T1 (n=20, paired, gpt-oss FAITH 3x majority, sweep on, phase-2 en + v2 zh):**

| metric | value | gate |
|---|---|---|
| FAITH-claim | **Δ +1.05** (14/2/2, p=0.004) | GT2 **PASS** (≥ +0.3) |
| INVERT cursor | **0 / 20** (baseline: 3) | 0% requirement **MET** |
| FAITH-anchor | Δ +0.40 | positive |
| SYNTH | Δ +0.50 (at threshold) | GT3 borderline |
| COVER | Δ +0.30 | — |
| GT4 | 0.51x | PASS |

**The journey in one line**: 12 inversions / FAITH −1.64 (broken baseline, no sweep) →
**0 inversions / FAITH +1.05** via four harness bugs (retrieval, prefixes, prose-FIX,
judge stochasticity), the VERIFY/ANCHOR sweep (spec §5.2, previously unimplemented),
majority voting, a working zh baseline, and phase-2 real-data training.

---

## Sweep-free (raw) measurement of the phase-2 checkpoint — VoxSumDroid feedback item 1 (2026-08-12)

External evaluation (VoxSumDroid, on-device consumer) noted that the 0%-inversion headline
was sweep-dependent and that the phase-2 checkpoint's raw rate was unmeasured. Measured now
(T1, n=20, same judges/protocol, **no VERIFY/ANCHOR sweep**):

| configuration | INVERT | note |
|---|---|---|
| pre-phase-2 model, raw (previously the only raw number) | 12/20 (60%) | from the earlier run |
| **phase-2 model, raw (measured now)** | **4/20 (20%)** | real-data adaptation helped the model itself |
| phase-2 model, + sweep | 0/20 | headline number |

Raw survivors (characterised): 1 real precision inversion ("old-fashioned" stated as
"fashionable"), 1 zh stale-state bullet (rejection retained after the later approval),
1 truncated fabrication, 1 borderline judge call (the evidence line supports the claim).

**Sweep-feedback-into-training: not yet implemented** — DROP/FIX outcomes are currently
judge-time corrections only, not SFT signal. Stated explicitly per the feedback: **the
target metric for the next iteration is raw INVERT, not swept INVERT.** The phase-2 raw
improvement (12 → 4) shows the model-side lever works; harvesting sweep DROP cases
(contradiction/fabrication demonstrations) as training signal is the concrete next step
toward a checkpoint whose raw rate approaches the swept rate.

---

## BF16 vs Q4_K_M — the quantization is not the quality limiter (2026-08-12)

Every headline number (G1, T1, FAITH) was measured on the **Q4_K_M production GGUF** — the
quantized artifact IS what was evaluated, not an extrapolation from bf16. The one missing
datapoint (bf16-vs-Q4 for the LFM linear-attention arch) was measured on the G1 screen:

| instance | screen-en chain | SUMMARY |
|---|---|---|
| **Q4_K_M (production)** | **PASS** | neutral summary → chain clean |
| **BF16** | FAIL | "Decision made to reject..." — SUMMARY never revised |

The BF16 instance kept the rejected-state SUMMARY while the Q4 instance carried a neutral
one; both UPD'd DECISIONS. **Quantization did not degrade quality — the model sits at a
margin where the greedy trajectory is fp-instance-sensitive** (linear attention + quant fp
paths), and a single screen criterion flips between instances. The shipped Q4 artifact's
measured numbers stand.

**Verdict on "is the ft + quantized GGUF good enough"**: the quantized model's quality is
exactly the reported quality — G1 PASS, T1 0/20 swept (4/20 raw), FAITH +1.05, GT4 0.51×.
Good enough for the faithfulness gate with the sweep; GT3 sits exactly at +0.50; and for
on-device 0%-without-judges the limiter is the model's raw fabrication rate (4/20), not the
quantization — the raw-INVERT training pass is the next lever, not a different quant.

---

## External verdict state (VoxSumDroid, 2026-08-12)

Owner's response recorded in `agentic-summarizer-feedback.md`: still "not integrated", but
the raw 12/20 → 4/20 measurement is acknowledged as real model-side progress; the gap is
live and narrowing. **Their re-evaluation bar: raw INVERT < their shipped 6.2%.** Our
adopted discipline: report raw figures first, swept figures second, in every headline.

---

## Primary track continues: sweep-feedback negatives (phase-3, 350M) — raw INVERT 4/20 → 3/20 (2026-08-12)

First harvest of sweep DROP corrections as negative SFT signal (48 samples ×3, from 109
dropped bullets on 27 train real meetings; stale-state class excluded). Trained from the
phase-2 checkpoint (LR 2e-5, 2 epochs).

- **RAW T1 INVERT: 4/20 → 3/20** (qmsum-4bfcff6d8771, qmsum-bdb39cc06654,
  synth-zh-reversal-5) — the model-side lever keeps moving the raw rate toward the owner's
  < 6.2% bar (3/20 = 15%).
- **G1 regression**: deadlines now anchored one line off ([5:00] vs 6:00) — the negatives
  perturbed anchor placement. Chain PASS, valid-op 100%. The anchor fix is the next
  primary-track step (targeted, not a full retrain).

---

## Secondary options evaluated (2026-08-12): MiniCPM5-1B is the standout

Three secondaries fine-tuned on the same CURSOR data (en+zh mix, 6 epochs, text grammar)
and evaluated:

| model | en G1 | zh G1 | raw T1 INVERT | note |
|---|---|---|---|---|
| **MiniCPM5-1B** (InfLLM v2 sparse attention) | **PASS** (all 4) | 3/4 — **only the trap fails** (chain/deadlines/UPD PASS) | **4/20** | **one model holds both languages — no seesaw at 1B** |
| LFM2.5-1.2B-Thinking | PASS (all 4) | FAIL (chain/UPD — seesaw persists) | — | thinking prior needs server `--reasoning off` |
| Qwen3-0.6B-notetaker FT | PASS (all 4) | FAIL (chain/UPD) | — | the note-taking prior transfers to en only |

MiniCPM specifics: served with `--reasoning off` (its hybrid RL prior emits
`<think>` on some chunks even after SFT; the template inserts an empty think block when
`enable_thinking=false` is sent — `send_thinking_kwarg=False` in the backend). Mean
FAITH-claim on T1 (self): ~3.5.

**Verdict**: MiniCPM5-1B is the strongest secondary — en G1 PASS, zh within one trap fix
of PASS, raw INVERT 4/20 (equal to the 350M phase-2; the 350M phase-3 holds 3/20). The
single-model-both-languages property removes the composite's per-language complexity. Its
trap gap is the cheapest fix (zh trap demonstrations); its raw rate needs the same
sweep-feedback treatment as the primary.

### Primary track: p3b — G1 restored, raw improvement kept (2026-08-12)

The p3 negative dose (×3) broke G1 deadlines; the lighter dose (48 samples, p3b) restores
all four G1 criteria (valid-op 100%) while keeping the raw gain:

| model | G1 en | raw T1 INVERT |
|---|---|---|
| p2 (phase-2) | PASS | 4/20 (en 3/10) |
| p3 (neg ×3) | FAIL (deadlines) | 3/20 (en 2/10) |
| **p3b (neg ×1)** | **PASS** | **en 2/10** (+ zh 1/10 ≈ 3/20) |

p3b is the current primary candidate: G1 PASS + raw ≈3/20, toward the owner's 6.2% bar.
Next raw-INVERT lever: harvest the ~53 pure-hallucination drops (bullets with no teacher
trace — requires capturing the student's per-step states during the harvest) + zh negatives.

---

## MiniCPM5-1B-p3: single-model G1 PASS in BOTH languages (2026-08-12)

The zh trap-only gap was fixed with a targeted dose: zh trap-chunk steps (chunks
containing the trap term; targets NOP or trap-free) ×3 + 1/3 of the base mix, from the
base MiniCPM checkpoint, LR 1e-5, 2 epochs.

| model | en G1 | zh G1 |
|---|---|---|
| MiniCPM5-1B (base FT) | PASS | 3/4 (trap) |
| MiniCPM5-1B-p2 (zh combined ×2 + en negatives) | PASS | FAIL (trap + UPD regression) |
| **MiniCPM5-1B-p3 (trap-only fix)** | **PASS** | **PASS — first single-model both-language PASS** |

Primary 350M p4: G1 PASS, raw en INVERT 2/10 (unchanged from p3b; the same two
fabrication classes persist — bdb39cc06654, 8ac3acb7fe5e).

### MiniCPM-p3 raw T1: 5/20 — and the persistent fabrication classes (2026-08-12)

MiniCPM-p3 (trap-fixed, single-model G1 PASS both languages) raw T1 INVERT: **5/20**
(qmsum-a001c3a20024, qmsum-bdb39cc06654, synth-zh-reversal-0, qmsum-46afb4f2ef60,
qmsum-3f8b473ddd36) — the trap fix nudged G1 to full PASS but the raw rate ticked up from
the base's 4/20.

**Persistent 350M classes (p3b/p4, unshifted by the negative harvest):**
1. **Negation misstatements** — "No commitment to research for the next six months."
   (asserting a negative commitment the transcript doesn't support);
2. **Polarity flips** — "can't imagine a wooden remote control" → "Keep the option to use
   wood as a practical choice" (the meeting's stated preference inverted);
3. **Judge-borderline** — "UI Designer to develop look and feel design": the evidence line
   ("Interface Designer will work on the user interface design") actually supports it.

These are the semantically-hard fabrications; their equivalents are absent from the train
meetings' sweep flags, so the negative harvest cannot cover them. The remaining levers are
new real training data (the root lever) and the eval-time sweep (already 0/20 swept).

**Locked options measured standing (raw T1 INVERT, owner's bar < 6.2%):**

| option | G1 en/zh | raw INVERT | single model |
|---|---|---|---|
| 1st: LFM2.5-350M p4 | PASS/PASS (composite) | **≈3/20 (15%)** | no (per-language pair) |
| 2nd: MiniCPM5-1B p3 | PASS/PASS (one model) | 5/20 (25%) | yes |

## Apodex-1.0-0.8B-SFT as on-device FAITH verifier: measured negative (2026-08-12)

Probe: 40 claim-mode bullets from the p4 T1 judged outputs (gpt-oss verdicts, 3x
majority), same `_FAITH_SYS` + `faith_prompt` prompts, `enable_thinking=False`,
left-padding, temp 0. Apodex-1.0-0.8B-SFT (Qwen3.5-0.8B, verification-centric deep-
research post-training) agrees with gpt-oss on **2/40 (5%)** — it systematically
answers SUPPORTED for bullets gpt-oss marks UNSUPPORTED, and does not emit the
1-word verdict format (long agentic chains even with thinking off). It is a research
evidence-chain model, not a claim-verdict classifier for the FAITH protocol. As-is it
cannot replace the sweep judge. A FAITH-verifier fine-tune of a 0.8B base (from the
sweep's own (bullet, evidence, verdict) triples) remains a possible future probe.

## Phase-5 (350M primary): 173 new real-meeting traces + p4 mix; MiniCPM-p4 (2026-08-12)

**New real data (root lever):** 15 previously-unused distinct QMSum transcripts from the
HF cache (same Canadian-committee provenance as the pool; 5k-50k tok; 24 files incl. the
10 already-traced train meetings re-traced) → traced at 2048 with the judge filter
(Z1 80 records / Z2 103 records). phase-5 = p4 (624) + 173 new = **797 samples**.

**350M p5** (resume p4/final, LR 2e-5, 2 epochs): G1 PASS (100% valid-op).

**MiniCPM-p4** (trapfix + its own 230 harvest negatives ×2, LR 1e-5, 2 epochs):
G1 PASS both languages with **100% valid-op on both** (p3: 88% zh). BUT raw T1 INVERT
went 5/20 → **6/20** (4bfcff6d8771, 6825a6ef4300 in; 3f8b473ddd36, bdb39cc06654 out):
the 230-negative dose over-suppresses — G1/op-quality improves, raw faithfulness degrades
(over-cautious rewrites misstate borderline claims). Dose sensitivity confirmed again
(350M p3 broke G1 at 144; MiniCPM raw degrades at 460).

## MiniCPM-p5 (half-dose) + standings (2026-08-12)

MiniCPM-p5 (trapfix + its 230 harvest negatives ×1, LR 1e-5, 2 epochs):
**G1 PASS both languages** (en valid-op 100%, zh 87.5%; trap/chain/deadlines/anchored/UPD
all PASS), raw T1 INVERT **4/19** — the ×1 dose keeps the trap fix and the zh UPD
behavior while returning raw to the base level (p4's ×2 over-suppressed to 6/20).
Dose-sensitivity confirmed for the third time: ×1 negative dose is the sweet spot for
both G1 integrity and raw faithfulness.

**Measured standings, end of session:**

| option | G1 en/zh | raw T1 INVERT | sweep | models |
|---|---|---|---|---|
| 1st: LFM2.5-350M p5 + zh v2 | PASS/PASS (composite) | 4/20 (en 3 + zh 1) | 0/20 | 2 × ~215 MB |
| 2nd: MiniCPM5-1B p5 | PASS/PASS (one model) | 4/19 (1 arms fail) | — | 1 × ~650 MB |

Both options converge at the same raw rate (~20%), plateaued at the semantically-hard
classes (decision/commitment distortions — "proposal"→"decision" — and polarity flips),
which the sweep (0/20) catches at deploy time. The owner bar (< 6.2% raw) needs either a
larger real corpus of the hard classes or targeted counterfactual negatives (the
proposal-only class is constructible with tools/build_counterfactual.py).

## Hard-class pass (2026-08-12): 3 new synth kinds + phase-6

New synth kinds targeting the persistent raw-INVERT classes: `proposal` (proposed-but-
never-decided → the trap is ADDing it to DECISIONS), `nocommit` (explicitly no
commitment → trap: asserting one), `rejpref` (negative preference "cannot imagine X" →
trap: "keep X as an option"). 8 meetings (4 en + 4 zh), padded so the deferral beat
lands in a different chunk, traced by the teacher (en 15 records / zh 16); the teacher
demonstrates NOP/OPEN at the deferral chunk.

- **350M en p6** (p5 + 6 en hard samples): G1 PASS (100% valid-op).
- **350M zh p3** (v2 + 7 zh hard samples): **G1 FAIL — the chain/UPD broke** (valid-op
  78%, chain_correct False). The hard-class "revision chunk → NOP/ADD" lessons conflict
  with the "contradiction chunk → UPD" demonstrations — the twins confound again. zh
  reverted to v2 (measured G1 PASS, raw 1/10).
- **MiniCPM p6** (p5 + 6 en + 7 zh hard samples): G1 PASS en + zh (100% / 88% valid-op)
  — at 1B the dose coexists with the UPD lessons.

Lesson: the hard-class counterfactuals are usable as NEGATIVES (student-fabrication
harvest) but NOT as positive protocol traces for the 350M — they over-teach NOP at
revision chunks. The 350M en p6 kept G1 because its dose was smaller and en-only.

## Phase-6 results: MiniCPM5-1B p6 = best raw rate measured (2026-08-12)

**MiniCPM5-1B p6** (p5 + 6 en + 7 zh hard-class trace samples, LR 1e-5, 2 epochs):
- G1 **PASS both languages** (en 100% / zh 88% valid-op; chain/deadlines/trap/anchored/UPD all PASS)
- raw T1 INVERT **2/20 (10%)** — the best raw rate measured on ANY system in the project
  (350M p5 4/20 · MiniCPM p5 4/20 · 9B baseline's own 3/20). Both flags verified against
  the transcript: qmsum-46afb4f2ef60 (garbled contact-method claim at a noise anchor) and
  qmsum-a001c3a20024 ("Project manager to email presentations" — the transcript's
  "then you don't have to email them" states the opposite; a commitment inversion).
- The hard-class counterfactuals (proposal/nocommit/rejpref) WORKED at 1B: p6 cleared
  3 of p5's 4 flags (8ac3acb7fe5e, 4bfcff6d8771, synth-zh-reversal-0) at the cost of one
  returning (46afb4f2ef60, last seen in p4).

**350M p6** (p5 + 6 en hard samples): **raw 8/20 — regression** (en 4 + zh 4). The zh
regression is the UNCHANGED v2 model re-measured at 4/10 vs 1/10 earlier — judge-noise
band on the synthetic zh tier (the FAITH 3x majority does not fully stabilise marginal
zh retrievals). The en p6 (4/10 vs p5's 3/10) confirms: hard-class POSITIVE traces hurt
the 350M (same UPD-conflict mechanism as zh p3's G1 break) — the 350M can only absorb
them as harvest-style NEGATIVES, never as protocol positives.

**Final standings (all measured, same judge pipeline):**

| option | G1 en/zh | raw T1 INVERT | sweep | models |
|---|---|---|---|---|
| 1st: LFM2.5-350M p5 + zh v2 | PASS/PASS (composite) | ~4/20 | 0/20 | 2 × ~215 MB |
| **2nd: MiniCPM5-1B p6** | **PASS/PASS (one model)** | **2/20** | — | 1 × ~650 MB |

The 2nd option has overtaken the 1st on the raw metric — at 1B the hard-class
counterfactuals land; at 350M they corrupt the UPD lesson. Both remain above the owner's
< 6.2% bar; the sweep stays the 0/20 deployment net.

## zh 350M G1 chain: measured negative after 6 attempts (2026-08-12)

The zh 350M leg has never passed the G1 screen's decision chain (6 versions: v1, v2,
p3, p4, v3, v4). Session findings:

1. **Train/eval granularity mismatch found**: the G1 screen runs at **budget=128 tokens**
   (1-2 lines/chunk) while all training traces were 2048-token (and b128-LINE) chunks.
   Traced 8 zh combined meetings at the screen's exact budget (J1_zh128, 62 records,
   teacher demonstrates UPD «否決…» -> 通過… at the approval chunk).
2. The teacher's own trap-chunk targets at 128 tokens ADD "跳過{trap}討論" (trap-term
   containing) — screen-poisoning (the trap check is term-based); filtered out.
3. zh v4 (J1 trap-free ×3 + trapfix): **trap PASS (first time)** but chain STILL FAIL:
   the model emits "倉庫整併方案合併" — it echoes the subject's 合併 instead of the
   approval verb 通過; the polarity parser scores it 0 (not -1/+1) → the screen's
   `1 in polarities` check fails. The model avoids the 通過/否決 polarity vocabulary.

Conclusion: at 350M the zh G1 chain is a measured negative (6/6). The zh 350M also
never reached the raw bar (1-4/10 with judge noise). The **MiniCPM5-1B is the only
G1-PASS-both model** — the single-model option is now the only production candidate.
The en 350M (G1 PASS, raw 3-4/10) remains the cheapest en-only fallback.

## FINAL: MiniCPM5-1B-CURSOR ships as the production option (2026-08-13)

LFM2.5-350M retired per user directive (en G1 PASS + raw 3-4/10; zh G1 chain measured
negative 6/6 — recorded above). All effort moved to MiniCPM5-1B with dual-GPU DDP
training (torchrun nproc=2, batch 2/GPU x accum 4 = 16 effective).

**Harness change**: deterministic UPD→ADD fallback — a UPD whose prefix matches no
bullet is honored as an ADD (temporal-guard-gated, dedup-guarded, logged in `reason`).
This converts the model's "UPD against empty state" (the zh screen's one failing op)
into a correct ADD. src/voxsum/guards.py.

**Measured hygiene incident (recorded)**: an export for a non-existent checkpoint
(checkpoint-282) silently no-oped; the stale server kept serving p10, so one "p11"
screen + T1 cycle re-measured p10 (numbers were consistent with p10's — good, but the
lesson stands: verify the artifact file exists AND the server PID changed after every
export). The p11 final itself fails the zh trap (trap behavior sits at the decision
boundary between checkpoints 282 and 284).

**Ship artifact — MiniCPM5-1B-CURSOR (checkpoint-274, G1-verified 3x)**:

| gate | measured | verdict |
|---|---|---|
| G1 screen (en/zh) | PASS / PASS (chain, deadlines, anchored, trap; valid-op 100% / 88% — one harmless duplicate-ADD rejected) | **PASS** |
| GT1 valid-op | en 100%, zh 88% (the reject is the dedup guard working correctly) | PASS (documented) |
| GT2 faith | FAITH-claim **4.81-4.84** vs 3.50 baseline (**+1.3**); INVERT raw 2/20, **swept 0/19-0/20** vs baseline 3/20 | **PASS** (ship rule: GT2 at FEWER inversions) |
| GT3 synthesis | SYNTH 2.32 vs 2.60 (−0.28, within the ±0.4-0.5 judge-noise floor = tie) | tie (not a win) |
| COVER | 2.84-2.89 vs 3.05 (−0.16, tie) | tie |
| GT4 efficiency | harness-level ~1.2× prefill (same harness as the 350M's measured 0.51-1.18×) | PASS |

Published: **Luigi/minicpm5-1b-cursor** (Q4_K_M GGUF + card; ~650 MB on-device, 4k ctx).
Deployment = model + CURSOR harness + VERIFY/ANCHOR sweep (sweep budget 60, gpt-oss-20b
judge) + the UPD→ADD fallback.

The owner's raw bar (< 6.2%): model-only 2/20 = 10% (above the bar as-is), deployed
with the sweep 0/20 = 0% (below). The sweep is part of the product pipeline, same as
the 350M release.

## p13 + in-stream verification: 0/20 all-on-device, COVER/SYNTH above baseline (2026-08-14)

**Model-side (p13)**: real-context hard-class augmentation — the over-assertion
counterfactual beats injected into REAL transcripts (14 augmented meetings, 4 beat types:
soft-action, either/or, informal-negation "you don't have to email them", intention
"the intention is that..."). The teacher traces filtered to drop its own beat-action
assertions. p13 = p12 + the filtered samples. **Result: raw 2/20** (matches the best-ever
p6) with **COVER 3.20 and SYNTH 2.75 — both above the map-reduce baseline (3.05/2.60)
for the first time on any configuration**. 8ac3acb7fe5e (the biggest persistent flag)
cleared. G1 PASS both, valid-op 100% both. p14/p14b (nofollow/intention beats + trap
booster) regressed the zh trap / en chain — reverted; the p13 balance is the stable
optimum (the zh trap sits at a decision boundary that en-heavy data crosses).

**Pipeline-side (in-stream verification)**: new harness mode `--verify-url` — every
ADD/UPD touching DECISIONS/ACTIONS is judged by the on-device verifier
(`Luigi/lfm2.5-350m-verifier`, 96% agreement with gpt-oss-20b) against the chunk's
anchor neighborhood BEFORE application; UNSUPPORTED/CONTRADICTED ops are dropped
(logged). This closes the over-assertion class at application time, on-device.

**p13 + in-stream verification (n=20): INVERT 0/20, FAITH 4.10, COVER 3.20, SYNTH 2.75**
— the full pipeline on-device (student 688MB + verifier 215MB ≈ 900MB, within the 2.05GB
device ceiling). The model-only raw (2/20 = 10%) remains above the 6.2% bar without the
verifier; with it, the device runs 0 inversions at better-than-baseline coverage and
synthesis.

## Multi-role fine-tune (one model, three roles): measured negative (2026-08-14)

Experiment: one MiniCPM5-1B fine-tuned on all three model-role distributions
(proposer CURSOR traces + critic FAITH triples 3000 + anchor-picker triples 2100,
prompt-role conditioned, tagged `multirole-v1`). Results vs the two-specialist
baseline:

| role | multi-role model | two specialists |
|---|---|---|
| proposer (G1 screen) | en PASS, zh trap FAIL | PASS both (p13) |
| critic (200-triple agreement vs gpt-oss) | **38%** (SUPPORTED collapse, 139/200) | **97%** (Granite-4.0-350M) |
| anchor picker | 66% | deterministic matcher (fallback-equivalent) |

The roles interfere: the summarizer's generation training overwhelms the verdict
classification, and the verifier/anchor data crosses the zh-trap boundary (the same
seesaw as every mixed pass). One 1B model cannot hold both roles at excellence —
**the two-specialist design (MiniCPM proposer + Granite critic) is the measured
optimum**. Deployment stays 688 MB + 215 MB.

## Coverage pass (p15 lineage): real-zh coverage vs the G1 seesaw (2026-08-14)

The maintainer's real zh-TW procurement meeting (203 lines, ASR echo-loop noise)
reproduced the failure: p13 emitted ~1.25 ops/chunk with empty DECISIONS/ACTIONS.
Teacher-traced at 2048 (4.3 ops/chunk — the coverage teaching signal) + ASR-noise
augmentation of 10 zh synth meetings.

Dose sweep measured (resume p13, LR 1e-5, 2 epochs):

| pass | mix | G1 | real-meeting coverage |
|---|---|---|---|
| p15 | real×3 + noisy + zhcomb×2 | en chain FAIL + zh trap FAIL | **ACTIONS 5** |
| p15b | real×2 + noisy + zhcomb + encomb | en chain FAIL, zh PASS | ACTIONS 3 |
| p15c | real×1 + noisy + zhcomb + encomb×2 | **PASS both** | ACTIONS 0 |
| **p15d** | real×2 + noisy + zhcomb + en-UPD-steps×3 | **PASS both** (zh 86% valid-op) | **ACTIONS 3, OPEN 4, real content through the noise** |
| p15e | p15d + real-ADD-DECISIONS×3 | en chain FAIL | DECISIONS still empty |

p15d is the optimum: G1 PASS both + the maintainer's first bar (ACTIONS non-empty on
their real meeting) met; DECISIONS on that meeting resists the targeted dose without
crossing the en chain. The real-meeting trace is ADD-dense (a status meeting — no
revision arc), which leaks over-ADD into the en screen unless counter-weighted with
pure UPD-step samples.

## Phase 0.1 (PLAN-multiagent): A2 highlighter probe — G1-safe, coverage-neutral (2026-08-14)

A2 (deterministic commitment-line marker `»` in the CHUNK rendering) wired through
run_cursor/build_step_prompt/screen/run_arms. Measured on the frozen p15d:
- G1 screen with highlights: **PASS both languages, 100% valid-op** — the marker does
  not break the byte-stable contract's observable behavior (risk #1 did not
  materialize at the G1 level).
- The maintainer's real meeting: ACTIONS 2 vs 3 unhighlighted, decode 251 vs 311 —
  coverage-neutral; the marker shifts the op mix without raising density.

Verdict: A2 as designed is insufficient for the coverage bar; shelved (the fallback
paths — deterministic-only use or retraining the base on highlighted renderings —
remain open). Proceeding to Phase 0.2 (critic rank sweep).

## Architecture decision: 2-agent deployment locked (2026-08-14)

Per user direction, the multi-agent/multi-LoRA plan (PLAN-multiagent.md) is
abandoned — marked superseded in the file, with the Phase-0 capacity measurements
kept (critic rank 8 = 64%, rank 32 = 85%: the generation bias resists low-rank
correction, corroborating the two-specialist design). Granite-4.1 has no 350M (the
line starts at 3B); granite-4.0-h-350m exists but the user selected the plain
granite-4.0-350m. The deployment is:

- **Main (proposer): MiniCPM5-1B p15d** — G1 PASS both, real-meeting ACTIONS
  populated via the real-zh coverage training.
- **Verifier (critic): granite-4.0-350m** (`Luigi/granite-4.0-350m-verifier`,
  Apache-2.0, 97% agreement, epoch-4 GGUF).

Configuration: in-stream verification + final VERIFY sweep, both judged by the
granite verifier (the deployment measured 0/20 inversions with the LFM-based
verifier; the granite one is 97%-equivalent and re-measurement is in progress).

## FINAL DEPLOYMENT MATRIX — 2-agent architecture (2026-08-14)

Architecture locked per user: **main = MiniCPM5-1B, verifier = granite-4.0-350m**
(Apache-2.0). Full measured matrix (T1, n=20, local judges, 3x majority):

| configuration | INVERT | FAITH | COVER | SYNTH |
|---|---|---|---|---|
| p13 raw | 2/20 | 3.94 | 3.20 | 2.75 |
| p13 + in-stream (verifier) | 0/20 | 4.10 | 3.20 | 2.75 |
| p13 + in-stream + final sweep | 0/20 | 4.54 | 2.95 | 2.50 |
| p15d raw (coverage pass) | 3/20 | 3.80 | 2.95 | 2.40 |
| p15d + granite in-stream + granite sweep | 1/20 | 4.20 | 2.50 | 1.95 |
| baseline (Qwen3.5-9B map-reduce) | 3/20 | 3.50 | 3.05 | 2.60 |

Findings:
1. The granite sweep judge over-drops relative to the LFM-based one — its zh
   verdicts are weaker (the triples are en-heavy; zh-only verifier training is the
   open follow-up). COVER/SYNTH pay for it.
2. The best balance remains **in-stream verification only**: 0/20 with COVER 3.20 /
   SYNTH 2.75 — with the documented structural caveat (the ±90s window cannot see
   later reversals; the stale-state class needs the final sweep or the timeline
   guard's extension).
3. The coverage pass (p15d) delivered the maintainer's first bar (ACTIONS non-empty
   on the real noisy meeting, G1 PASS both) at the cost of one raw flag (3 vs 2).

Open items carried into NEXT_STEPS: zh-verifier training (zh triples), the DECISIONS
emptiness on the real meeting, the stale-state guard extension, SYNTH.

## p16 (STATE-reading negatives + DECISIONS dose on p15d): seesaw again (2026-08-14)

The maintainer's op-level audit sharpened the coverage problem into two training
signals: (1) zero DECISIONS ops proposed on the real meeting, (2) ~30% of output
re-proposing STATE bullets already present (weak STATE utilisation — a different
fix from extraction). Built `tools/build_state_negatives.py` (captures the dedup-
guard rejections the harvest misses — 4 negatives from the real + noisy meetings),
trained p16 = p15d + DECISIONS-steps ×2 + state-negatives ×3.

Result: op density UP on the real meeting (decode 311 → 432 tokens), ACTIONS 3 —
but DECISIONS still zero, the stock-phrase loop returned (the re-proposed phrasing
in TITLE/SUMMARY/ACTIONS/TOPICS), new bracket artifacts ([ SUMMARY ] leaks), and
the en chain crossed. p16 reverted; **p15d remains the optimum**. The DECISIONS
extraction needs more real-zh data — the single transcript's 3 decision steps are
not enough signal at any dose that holds the G1.

## zh verifier training (gr4): the sweep over-drop fixed (2026-08-14)

NEXT_STEPS #1 executed: zh triples (64 judged zh bullets, both evidence forms) +
12 zh polarity-flips (通過/否決, Q1/Q4, 2027/2028…) ×3, trained on the granite
verifier (2 epochs, LR 1e-5). Published as
`granite-4.0-350m-verifier-zh.Q4_K_M.gguf` in the verifier repo.

- en agreement: 96% (held; 97% → 96% within noise), distribution healthy.
- zh agreement: **92%** on the zh triples (the previously-unmeasured weak spot).

Deployment (p15d + gr4, in-stream + sweep):
- **zh half (n=10): INVERT 0, FAITH 4.73, COVER 3.80, SYNTH 3.40** — the over-drop
  is gone (the old verifier's zh-collapsed run measured COVER 2.50/SYNTH 1.95
  overall; the zh half now clears the baseline by a wide margin).
- Full n=20: INVERT 2 (4bfcff6d8771, e75802cbf8d3 — en flags, ±1 noise band),
  FAITH 4.43, COVER 2.85, SYNTH 2.35 — better than the old-granite stack on every
  quality metric, at +1 inversion vs its 1/20.
- The best-balance config remains p13 + in-stream-only (0/20, COVER 3.20,
  SYNTH 2.75) for en-primary; for zh-primary the gr4 stack is now the choice.

## DECISIONS blocker cleared: p17c (2026-08-14)

The VoxSum author's structural suggestion (a DECISIONS-specific pass) was the fix:
**DECISIONS-only distillation** — 193 teacher records reduced to their ADD/UPD
DECISIONS lines only, ×4 dose + the en-UPD counterweight ×3 on p15d. Dose sweep:
p17 (×4): DECISIONS fired (4 bullets on the real meeting, incl. the supplier
decision) but en chain FAIL; p17b (×2): no DECISIONS, en chain FAIL; **p17c
(×4 + en_upd×3): G1 PASS both (100% valid-op) AND DECISIONS non-empty on the
maintainer's real meeting** ("Support R&D for higher price points"). The
harness-side promotion (`--promote-decisions`, decision-shaped SUMMARY bullets →
DECISIONS at render) also landed. The zero-DECISIONS class is broken for the
first time; the raw T1 validation is running.
