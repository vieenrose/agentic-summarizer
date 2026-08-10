# PLAN — implementation & fine-tuning

**Status:** proposed · **Date:** 2026-08-10 · Companion to the spec in [`CLAUDE.md`](CLAUDE.md).
Where this file and the spec disagree, the spec wins until an amendment listed in §0 is accepted.

---

## 0. Model change: FunctionGemma-270M as primary student

Proposal: make **`google/functiongemma-270m-it`** the primary student, with
Qwen3.5-0.8B retained as the fallback rung. Rationale, and what it forces us to amend.

### Why it fits CURSOR unusually well

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

Mitigation: **G1 (spec §7.6) is the go/no-go, run on both students before any long-doc eval.**
Both rungs are built against the same harness, so the swap is a config line. We keep the
0.8B rung alive until 270M clears G1 with the correct decision chain.

### Amendments this forces — need your decision

1. **Judge contamination (blocking for reported numbers).** Spec §7 mandates a gemma-4 judge
   *because* Qwen teachers distilled the student. A Gemma-3 student makes gemma-4 the
   **same family as the student** — self-preference bias, the exact failure the rule exists to
   prevent. The rule should generalize to: *judge family ∉ {student family, teacher family}*.
   Options: (a) judge = Qwen3.5-large + teacher = gemma-4/Claude; (b) keep gemma-4 judge, add a
   third-family cross-check judge on the 6-meeting micro-cell and report both. I recommend (b)
   plus (a) for the ship-gate run — see §6.
2. **Op wire format (spec §5.1).** Emit ops as FunctionGemma function calls
   (`<start_function_call>call:ADD{...}<end_function_call>`) rather than the `ADD SECTION - …`
   text grammar, to ride the post-training. The NOTES v2 output contract (§3) is unaffected —
   the harness renders it. Requires rewriting §5.0/§5.1 as tool declarations.
3. **Efficiency baseline (§8, GT4).** Per-step budget is recomputed for 32k context; the
   ≤ +25% prefill gate must be re-derived against the *same* map-reduce baseline model, or it
   compares two different models and means nothing. Baseline stays 0.8B-vs-0.8B **and** we add
   270M-vs-270B-baseline; report both.

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

Teacher choice interacts with amendment §0.1. If the judge stays gemma-4, the teacher should
**not** be gemma-4.

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

1. **Judge family** — amendment §0.1. My recommendation: keep gemma-4 for continuity, add a
   Qwen3.5-large cross-check judge on the micro-cell + the final T1/T2 run, and report both
   families. Cheap insurance against a self-preference artifact invalidating GT3.
2. **Training hardware** — what GPU is available? bf16 is mandatory (Ampere+). If it's a T4 or
   a Colab free tier, we take Unsloth's float16 workaround path and I'd plan accordingly.
3. **Teacher model** — local (gemma-4-27B / Ternary-Bonsai-27B, already in the eval stack) or
   an API teacher? Trace generation is ~40 calls × N meetings; the cost/quality call is yours.
4. **Data access** — are VCSum / MeetingBank / QMSum already downloaded somewhere on this box,
   or is acquisition part of P1?
