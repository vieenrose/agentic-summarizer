# PLAN — implementation & fine-tuning

**Status:** proposed · **Date:** 2026-08-10 · Companion to the spec in [`CLAUDE.md`](CLAUDE.md).
Where this file and the spec disagree, the spec wins until an amendment listed in §0 is accepted.

---

## 0. Locked decisions (2026-08-10)

| decision | value |
|---|---|
| **base SLM** | **`google/functiongemma-270m-it` — LOCKED.** No fallback rung. If it cannot clear the gates, the outcome is the spec's own negative-result path (ship map-reduce, publish agency-at-270M as measured), not a model swap |
| **transcript source** | authentic audio → **MOSS-Transcribe-Diarize 0.9B** → transcript v1 (§1a). Weights already cached locally |
| **teacher** | **`gemma-4-31B-it` Q8_0, local, tensor-split across both GPUs** (§2b). Frontier API teacher is contingency only, if the screen fails |
| **judge** | **`Qwen3.5-9B`+ as primary** — non-Gemma, non-teacher (§0.1). `gemma-4-E2B-it` retained as a secondary, reported separately |
| **hardware** | 2× RTX 5090 32 GB (Blackwell, 64 GB total) — bf16 native, so Unsloth's float16-infinity workaround is **not needed**. Verify Unsloth/triton carry `sm_120` kernels; torch 2.10+cu128 is installed |

### Why FunctionGemma-270M fits CURSOR unusually well

| property | consequence for us |
|---|---|
| Gemma 3 270M architecture, **32k context** (verify from `config.json`; Gemma 3 270M/1B are 32k, not 4k) | the 4k budget in spec §8 stops being the binding constraint — chunks can grow, or STATE can |
| Post-trained specifically for **function calling**: emit a structured call given declared tools | our per-step output *is* a tool call (`ADD`/`UPD`/`DEL`/`CMP`/`NOP`). We stop fighting the base model's format prior |
| Explicitly **single-turn / parallel calls; no multi-turn chaining or state** | exactly CURSOR's contract — no conversation history crosses steps, STATE is the only memory. The documented weakness is the one thing we never ask for |
| **Parallel** function calling is supported | multiple ops per chunk is in-distribution, not a stretch |
| 256k vocab optimized for JSON + multilingual | zh-TW token efficiency; structured output is cheap |
| ~270M params → ≈**200 MB** at Q4_K_M | vs the 785 MB envelope. ~3× the decode speed of 0.8B |
| "not intended as a dialogue model… designed to be highly performant after further fine-tuning" | we were always going to fine-tune; zero-shot quality is irrelevant to us |

The efficiency headroom is the real prize: it buys a bigger sweep budget (§5.2), more chunks,
or a second pass — all of which target the metrics we're actually gated on.

### The risk, stated plainly

**270M may not abstract.** The gates that could fail are COVER and especially **SYNTH ≥ +0.5**
(GT3) — the whole agency bet. A 270M model can plausibly learn to *emit valid ops* (GT1) and
*copy an anchor* while producing bullets that are near-verbatim extraction with no
meeting-level arc. Secondary risk: zh-TW generation quality at this scale is unproven for us
(the vocab is multilingual; the *capability* is not established).

With the base locked, the mitigation is **data, not model substitution**: G1 (spec §7.6) stays
the go/no-go, and a G1 failure buys one honest data iteration (denser agency traces, §2b)
before the negative-result path is invoked.

### Amendments to the spec this forces

1. **Judge family — RESOLVED.** Spec §7 mandates a gemma-4 judge *because* Qwen teachers
   distilled the student. A Gemma-3 student makes gemma-4 the **same family as the student** —
   self-preference bias, the exact failure the rule exists to prevent. The rule generalizes to
   *judge family ∉ {student family, teacher family}*. With student = Gemma and teacher =
   Gemma/API, the Qwen ban is void: **primary judge = Qwen3.5-9B or larger**, with
   `gemma-4-E2B-it` kept as a secondary and reported separately (its scores are expected to be
   biased *upward*; that direction must be stated whenever it is quoted).
2. **Op wire format (spec §5.1).** Emit ops as FunctionGemma function calls
   (`<start_function_call>call:ADD{...}<end_function_call>`) rather than the `ADD SECTION - …`
   text grammar, to ride the post-training. The NOTES v2 output contract (§3) is unaffected —
   the harness renders it. Requires rewriting §5.0/§5.1 as tool declarations.
3. **Efficiency baseline (§8, GT4).** Per-step budget is recomputed for 32k context, and the
   ≤ +25% prefill gate is re-derived **270M-CURSOR vs 270M-map-reduce** — same model both
   sides, or the comparison measures the model swap instead of the architecture.

---

## 1a. Transcript collection (directive 1)

**Status of local assets:** MOSS-Transcribe-Diarize 0.9B weights are cached; **no meeting audio
is on this box**, so acquisition is real work, not a download.

**Pipeline (implemented):** `tools/transcribe_moss.py` → `voxsum.ingest_moss` → v1.
MOSS emits `[start][Sxx]text[end]` with float seconds; the converter renumbers `S01…` to
`S1…` **by first appearance** (MOSS labels are relative and need not start at 1 or be dense),
floors seconds to the v1 clock, merges adjacent same-speaker segments across gaps < 2 s
(MOSS segments at phrase granularity; v1 wants utterances), and collapses embedded newlines —
one utterance per line is a hard rule. Raw MOSS output is retained alongside each transcript so
a conversion bug can be re-fixed without re-running the GPU.

**What the public corpora actually turned out to be** (measured 2026-08-10, not assumed):

| corpus | on the Hub | speakers | clock | note |
|---|---|---|---|---|
| **VCSum** (zh) | **no** — not published under any findable name | — | — | the spec's zh pool is unobtainable this way; zh needs audio or a manual source |
| **QMSum** (`pszemraj/qmsum-cleaned`) | yes | **real** (`Professor E:`) | **none** | ~31k tokens/meeting; 272 rows carry only 35 distinct transcripts, so dedupe on the body — the `id` field identifies the *query*, not the meeting |
| **MeetingBank** (`huuuyeah/meetingbank`) | yes | **none** | **none** | ~2k-char agenda-item segments, one flat string; ACTIONS attribution is impossible without speakers |
| **AMI** (`edinburghcstr/ami`) | yes, 29 GB audio | real | real | the only Hub path to an *authentic* clock, via MOSS |

**The consequence is sharper than a missing dataset.** Neither fetchable corpus has
timestamps, so `voxsum.corpora` synthesises a clock at 150 wpm. That clock is monotonic and
internally consistent — an anchor still points at the line that states the claim — but the
wall-clock value is invented. So it is fine for **training** (the student learns "copy the
supporting line's timestamp", and the clock's truth is irrelevant to that skill) and **not**
a basis for reporting FAITH-anchor as real-world accuracy. Only audio through MOSS yields a
real clock. Every meeting carries `authentic_clock` / `authentic_speakers` in
`data/transcripts/manifest.json` so this cannot be forgotten downstream.

**Sourcing, in priority order:**

1. **en, real:** public-record council/committee meetings (the same provenance as MeetingBank,
   which is council-derived) — long, genuinely contested, decisions and actions actually
   happen. Licence: public record; record the source URL per meeting.
2. **zh-TW, real — the gap that matters.** The spec's own caveat (§7.8) is that the zh pool is
   monologic VCSum, so **contested zh-TW is unmeasured**. This is the one place authentic audio
   buys something no dataset gives us: Taiwanese public-body proceedings (legislative/council
   committee sessions, publicly broadcast) are multi-speaker, contested, and zh-TW. Getting
   even 6–10 of these retires the largest caveat in the project.
3. **Own recordings** if any internal meetings can be used — consent required, and they cannot
   be published, so they may be train-only, never in a released eval set.

**Quality gate before a transcript enters any set** (`tools/vet_transcript.py`, P0):
parses as v1; ≥ 2 speakers for the contested sets; speaker-label churn below a threshold
(diarization thrash is the failure mode that would silently poison ACTIONS attribution —
"name: what they will do" is only as good as the label); no line over ~3k chars unsplit;
duration and token count recorded; language tagged. **Spot-check a 5-minute window per meeting
against the audio by hand** — ASR error is invisible to every downstream judge, which will
happily score a faithful summary of a mistranscription as faithful.

Split discipline: every meeting gets an ID and a `split` field at ingest
(`train` / `t1` / `t2` / `micro`), asserted disjoint in CI (§2). Synthetic zh T2 concatenations
stay labelled `synthetic` per §7.8.

---

## 1. Repository layout

```
src/voxsum/
  transcript.py     parse_line, clock_to_sec, sec_to_clock, Utterance      ← §2, §7
  chunker.py        streaming cursor, 2048-tok chunks, 2-line overlap      ← §4
  state.py          NOTES state store, per-section caps, spread()          ← §3, §6.5
  ops.py            op dataclasses, parser (both wire formats), validator  ← §5.1
  guards.py         anchor validation, temporal guard, NOP-collapse        ← §6
  render.py         deterministic NOTES v2 renderer                        ← §3
  index.py          lexical search (word / char-bigram), snippet extractor ← §5.3, §7.2
  sweep.py          VERIFY, ANCHOR                                         ← §5.2
  agent.py          the CURSOR loop; the only model-facing orchestration
  baseline.py       map-reduce baseline + coverage fallback                ← §5.3
  prompts/          SYS + tool declarations, per language, VERSIONED
  backends/         llama.cpp / transformers / vLLM behind one interface
eval/
  judge.py          claim & anchor modes, COVER/SYNTH, last-match parsing  ← §7.2
  metrics.py        FAITH/COVER/SYNTH/INVERT + operational metrics         ← §7.4
  screen_g1.py      capability screen                                      ← §7.6
  report.py         paired comparison, sign test, tie-at-Δ<0.5            ← §7.3
train/
  gen_traces.py     teacher → validated op traces
  build_sft.py      traces → chat-formatted SFT dataset
  sft_unsloth.py    Unsloth LoRA/full-FT
  export_gguf.py    Q4_K_M export + on-device smoke test
```

**Build order (each step ends testable):**

1. `transcript.py` + `render.py` + `state.py` — golden-file tests on the §2.1/§3.1 examples.
   Property test: `sec_to_clock(clock_to_sec(t)) == t` over a padding-edge corpus
   (`0:00`, `9:59`, `59:58`, `1:00:00`, `1:02:07`). This is where the known mm↔ss bug lived.
2. `chunker.py` + `ops.py` + `guards.py` — the harness is fully testable with a **scripted
   fake model** (a list of op strings). No GPU needed to prove §6 holds.
3. `index.py` + `sweep.py` + `baseline.py`.
4. `agent.py` + a backend; run G1 zero-shot to get a floor.
5. `eval/`; run the map-reduce baseline end-to-end to fix the comparison point **before** any
   fine-tuning exists (otherwise gates have nothing to clear).
6. `train/`.

Non-negotiables while building: the harness never trusts the model (§6); `prompts/` files are
versioned and the version is recorded in every eval run — a silent SYS edit invalidates
train/eval comparability (§7.8).

## 2. Fine-tuning data — the actual hard part

We have no gold op traces. Generating them is the project's main risk after G1.

**Teacher-generated, harness-validated traces.** For each training meeting, replay the real
CURSOR loop with a large teacher in the student's seat: feed `SYS + STATE_i + CHUNK_i`, take
the teacher's ops, validate through the *real* harness, apply, advance. Keep a step only if
its ops parse, validate, and survive the guards. The result is on-policy w.r.t. the STATE
distribution the student will actually see — because the STATE it conditioned on was itself
built by accepted ops.

Two things this must not get wrong:

- **UPD/DEL are rare and load-bearing.** Natural meetings yield mostly ADD; the decision-chain
  behaviour G1 tests is exactly UPD. Oversample: synthesize planted-revision meetings
  (rejected→approved, deadline moved, action reassigned) in both languages, held out from G1's
  own screen meeting. Target ≳ 15% of steps carrying UPD/DEL, and include CMP steps by forcing
  sections past cap.
- **NOP must be taught, not just tolerated.** Content-poor chunks (small talk, logistics) are
  a large fraction of real transcripts. If NOP is underrepresented the student hallucinates
  ops; if overrepresented it collapses (GT1's NOP-collapse < 10%). Sample to the observed
  content-rich/poor ratio, and log the ratio.

Splits: training meetings **disjoint from T1, T2, and the micro-cell** (§7.5) — assert this in
CI by meeting ID, not by trust.

### 2b. Teacher choice (directive 3)

The teacher is not being asked to summarize. It is being asked to **demonstrate agency on our
op set** — to look at STATE and a chunk and decide *this earlier decision is now wrong, revise
it*. That capability is what we are distilling, and it is the scarcest thing in the pipeline: a
teacher that only ever emits ADD produces a student that can only ever emit ADD, and GT3 is
lost before training starts.

So the selection criterion is **not** general benchmark strength but: given STATE + chunk,
does it (a) emit valid ops in our declared-tool format, (b) *revise* rather than append when
the chunk contradicts STATE, (c) keep a SUMMARY that reads as an arc rather than a pile, and
(d) do all of it in zh-TW as well as en.

**Chosen: `gemma-4-31B-it` at Q8_0, tensor-split across both 5090s.**

`gemma-4-31B-it` is the only fully **dense** Gemma 4 — all 30.7B parameters active per forward
pass, against the 26B's 8-of-128-expert MoE (~4B active) — and currently ranks #3 open model on
Arena text. Dense is the right trade here because **throughput is irrelevant to this job**:
trace generation is a one-time offline batch of ~1200 steps at ~3k in / ~200 out. Measured
against the MoE that is roughly 2 h instead of 30 min, unattended, once. Spend all of it on
quality per forward pass.

**Q8_0 (33 GB), not Q4.** Teacher output *is* the training target, so quantization error
propagates directly into the student — this is the one place in the pipeline where precision is
not a cost/quality dial but a correctness one. Q8 needs both GPUs (33 GB > 32 GB), which is
what the 64 GB is for. `Q6_K` (~25 GB, single GPU, second card free for the judge) is the
fallback if both cards are ever needed elsewhere.

Being Gemma-family, the teacher's op text tokenizes identically to the student's own
vocabulary, which removes a class of distillation mismatch.

**Screen the teacher before generating volume.** Run ~30 steps drawn from known-revision points
and measure: valid-op rate through the real harness, UPD-vs-ADD rate at contradiction points,
and raw anchor-copy accuracy. Below ~90% valid-op, or appending where it should revise,
disqualifies it. Cheap to check, and it catches the failure that would otherwise surface only
as an unexplained GT3 miss after a full training run.

**Contingency, not plan: the API teacher.** If the screen shows 31B appends instead of revising,
generate a few-hundred-step agency seed with a frontier API teacher (Claude Opus 5 — priced at
~$3–6 for ~300 steps, or ~$25 for the entire trace set; the Batch API halves it and offline
generation fits batch perfectly) and few-shot the local teacher from those audited examples.
Not a budget decision — a quality one, only taken if the screen demands it.

**Never trusted, always validated.** A teacher step is kept only if its ops parse, validate, and
survive the guards (§6). Agency-seed steps additionally get a human pass.

### 2c. The teacher runs under the student's context budget (normative)

**The teacher sees exactly what the student will see at that step, and nothing more.** Its
prompt is the byte-identical `SYS + STATE + CHUNK_i` from §4, within the same per-step budget
(§8). This is not a cost measure — it is what makes a trace learnable:

- **No lookahead.** The teacher must not see `CHUNK_{i+1..n}`, the full transcript, or any
  reference summary. A teacher with hindsight writes a *perfect* arc-bearing UPD — and an
  unlearnable one, because the student at step *i* has no way to derive it. The student's only
  available lesson is "guess confidently", which is hallucination.
- **No enlarged STATE.** STATE is capped identically for teacher and student. A teacher shown
  bullets the student's caps would have evicted produces UPDs whose `«prefix»` cannot match.
- **Same clock.** The teacher's anchors must come from the current chunk, same as the student's
  (§6.1) — enforced by the same validator, not by trust.

**The distinction that matters: extra *input* is cheating; extra *compute* is not.** The teacher
may reason at length before answering — chain-of-thought, drafts, self-correction — because that
is more thinking about the *same information*, which is exactly what distillation is for. Only
the op lines are kept as the training target; any reasoning is stripped and never enters the SFT
sample. What is forbidden is widening the teacher's view of the transcript.

**Enforced in three places, not documented and hoped for:**

1. `llama-server` is started with `--ctx-size` at the student's per-step budget, so an
   over-budget prompt fails loudly instead of silently succeeding on the teacher's 128k window.
2. `train/gen_traces.py` asserts, per step, that the rendered prompt tokenizes within budget
   **using the student's tokenizer** (`functiongemma-270m-it`) — not the teacher's. Both are
   262,144-vocab so the counts should agree; asserting with the student's is the one that
   matters, and a disagreement is itself a bug worth catching.
3. The trace record stores the exact prompt it was generated from, so any later claim that a
   step was on-budget is checkable rather than asserted.

The same rule applies to the sweep tools (§5.2): VERIFY and ANCHOR evidence budgets are
identical for teacher and student.

Teacher = Gemma-family (or API) and judge = Qwen keeps *judge ∉ {student, teacher}* satisfied
(§0.1).

**Serving it (both GPUs):**

```sh
~/llama.cpp/build/bin/llama-server \
  -m ~/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/*/gemma-4-31B-it-Q8_0.gguf \
  -md ~/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/*/MTP/mtp-gemma-4-31B-it-Q8_0.gguf \
  --n-gpu-layers 999 --split-mode layer --tensor-split 1,1 \
  --ctx-size 4096 --parallel 2 --flash-attn on \
  --temp 1.0 --top-k 64 --top-p 0.95 --host 127.0.0.1 --port 8080
```

`--split-mode layer --tensor-split 1,1` spreads layers evenly across both cards (layer split
beats row split here — less inter-GPU traffic, and there is no NVLink on 5090s). The `-md`
module is the repo's MTP (multi-token prediction) head for speculative decode. Sampling
params are Gemma 3/4's recommended values (temp 1.0, top-k 64, top-p 0.95); trace generation
should additionally be run greedy-ish and seeded per step so a re-run is reproducible.

`--ctx-size 4096` is deliberately the **student's** budget, not the teacher's capacity (31B
carries 128k+). Per §2c the teacher must not see more than the student will, and capping the
served window makes an over-budget prompt fail loudly rather than quietly succeed. Raise it only
if §8's per-step budget changes — and then for both models together.

## 3. Fine-tuning with Unsloth

Per the [Unsloth Gemma 3 guide](https://unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune#fine-tuning-gemma-3-in-unsloth):

- **bf16 only.** Gemma 3 activations exceed float16's 65,504 → "gradients and activations
  become infinity" on T4/V100/RTX 20x. Requires RTX 30xx+/A100/H100, or Unsloth's float16
  workaround path. **Blocking hardware question — see §7.**
- **Exactly one `<bos>`.** The template is
  `<bos><start_of_turn>user … <end_of_turn><start_of_turn>model … <end_of_turn>`; llama.cpp
  auto-prepends `<bos>`. A doubled BOS is a silent quality killer.
- FunctionGemma's format differs from stock Gemma 3: a `developer` turn carries
  `<start_function_declaration>…<end_function_declaration>`, calls come back as
  `<start_function_call>call:NAME{param:<escape>v<escape>}<end_function_call>`, and `<escape>`
  delimits every string value. Our SYS + tool declarations must be emitted in *this* format,
  and the same builder must serve training and inference — one function, shared, tested.

Starting config (270M):

| knob | value | note |
|---|---|---|
| regime | **full fine-tune** first | 270M fits comfortably; LoRA r=64 on all-linear as the comparison arm |
| lr | 5e-5 full FT · 2e-4 LoRA | cosine, warmup 5% |
| epochs | 2–3 | early-stop on held-out valid-op rate, not loss |
| `max_seq_length` | 4096 | SYS + STATE + 2048-tok chunk + ops; raise only if chunks grow |
| loss | **completion-only** | mask SYS/STATE/CHUNK; train only on op tokens |
| batch | effective 32 via grad accum | |
| dtype | bf16 | see above |

Export: `save_pretrained_gguf` → Q4_K_M, then the on-device smoke test (§4) — quantization can
break exact special-token emission, so the anchor/valid-op rate is re-measured post-quant, not
assumed.

**Eval-during-training must be the operational metrics, not loss.** Track valid-op rate,
anchor rate (raw), NOP rate, and UPD-recall on a held-out slice each epoch; loss will keep
improving while the model learns to emit fluent invalid ops.

## 4. Sequencing & gates

| phase | deliverable | gate to pass before next phase |
|---|---|---|
| P0 | harness + fake-model tests | §6 guards provably hold; clock round-trip green |
| P1 | map-reduce baseline, evaluated | baseline numbers recorded for T1 + micro-cell |
| P2 | zero-shot G1 on both students | floor measured (expected: fail — that's fine) |
| P3 | traces + SFT v1 (270M) | **G1 (§7.6): correct decision chain, both deadlines, 100% anchored, no trap** |
| P4 | micro-cell eval | GT1: valid-op ≥ 95%, NOP-collapse < 10% |
| P5 | T1 paired eval | GT2 faith + 0% inversions |
| P6 | T2 (≥80k) + on-device timing | GT3 SYNTH ≥ +0.5, GT4 prefill ≤ +25%, envelope |

If 270M fails P3 after one honest data iteration, promote Qwen3.5-0.8B to primary and re-run
P3 — do not spend three cycles rescuing the small model. If **both** fail GT3, the spec's own
answer applies: ship map-reduce and publish agency-at-sub-1B as a measured negative result.

## 5. What I'd expect to break

- **Anchor copying under quantization.** `[m:ss]` must be copied verbatim from the chunk; a
  270M model at Q4 will drift digits. The deterministic matcher + ANCHOR sweep exist for this,
  but *raw* anchor rate is the honest signal — report it.
- **CMP.** Rewriting ≤cap bullets in one shot is the highest-capability op in the set and the
  most likely to be unlearnable at 270M. Fallback: harness-side deterministic compaction
  (oldest-unrevised-first) with CMP as an optional upgrade. Decide from P4 data.
- **SYNTH stagnation.** Bullets that are locally faithful and globally inert. The lever is the
  data, not the model: teacher traces must *demonstrate* arc-bearing UPDs and a bottom-line
  SUMMARY revision, or the student cannot learn what it never saw.
- **zh-TW long lines.** VCSum lines run to ~2.6k chars — a single line can exceed a chunk.
  The chunker must split within a line and the snippet extractor must window inside it (§7.2).

## 6. Measurement integrity

Everything reported carries: student + teacher + judge model IDs, prompt version, quant level,
and the §7.8 caveats. Δ < 0.5 on a judged metric is a **tie** (§7.3). Paired per meeting, sign
test, win/loss/tie counts. The n=6 micro-cell is **directional only** and never a ship gate.

## 7. Open questions for you

1. **Audio sourcing** — no meeting audio is on this box. Do you have recordings (internal, with
   consent, or already-collected public proceedings), or should P1 start by collecting public
   council / legislative sessions? The **zh-TW contested** ones are the high-value item (§1a).
2. **VCSum / MeetingBank / QMSum** — not found locally. Download as the base pool (they still
   provide T1 regression and en volume), or go audio-first for everything?
