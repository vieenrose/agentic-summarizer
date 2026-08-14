# PLAN — Multi-agent CURSOR with multiple LoRA adapters · v2 (research-revised)

**Status:** design v2 · **Date:** 2026-08-14 · **v1:** single-model multi-role SFT
(measured negative: 38% critic) and the six-role LoRA plan; **v2** incorporates the
external research review (on-device inference economics, multi-LoRA serving cost,
small-model multi-agent evidence, fine-tuning recipes). Deliberate changes from v1 are
marked **[v2]**.

## 0. Research findings that change the design

1. **Prefill is cheap, decode is expensive** (GENIE on OnePlus 15: 1463 t/s prefill
   vs 23 t/s decode; llama.cpp CPU 115 t/s prefill). Re-reading is nearly free;
   generating intermediate text is the expensive currency. → few passes, terse
   outputs, no prose-carrying intermediate roles.
2. **Adapter switching is NOT free on-device** (Qualcomm: unfused multi-LoRA costs up
   to +30% decode latency; Samsung: merged-on-load prevents dynamic switching). →
   **each adapter must run in ONE contiguous phase, never interleaved per chunk.**
3. **Below ~3B, extra agent roles cost more than they return** (Qwen2.5-7B: five-role
   JSON society 75.0→45.0, plaintext restores 82.0, two-call gated refinement 86.2
   at 7.4× less; AWS: sub-2B planner/critic plans hallucinate as much as the
   summaries). → drop planner; critics only as ONE final gated phase; plaintext
   inter-agent messages (ours already are).
4. **The white-box hallucination probe beats an LLM critic at this scale** (TrueBrief:
   logistic regression over LogitLens/LookbackLens, P .93/R .72/F1 .81 at 0.5B,
   ~zero marginal cost, *strongest at small scale*). → the free gate for conditional
   re-summarization; caveat: needs activation access (llama.cpp GGUF does not expose
   it — viable only if we move to a runtime that does, e.g. executorch/ONNX).
5. **Dialogue-to-description normalization is the highest-value cheap role**
   (NexusSum: up to +30 BERTScore; near-mechanical transduction small models learn
   well). This replaces v1's fragile deterministic A2 highlighter (measured
   coverage-neutral) with a *trained* normalizer, gated to fire only where needed.
6. **Fine-tuning recipe** (Apple production LoRA + Samsung S23 on-device + TrueBrief
   ablations): per-role teacher outputs (not end-to-end distillation), rejection
   sampling, SFT then DPO with controlled hallucination injection (single reject per
   chosen, β=0.5, rank 16, dropout 0.05), rank 32/α 16 for adapters, **shared
   quantization scheme across adapters** (NPU hard requirement).
7. **Baseline warning:** FLAN-T5-780M-class fine-tuned encoder-decoder reportedly
   beats 7B–70B zero-shot decoder-only models on meeting summarization — must be in
   the measurement ladder as an honesty check.

## 1. Role decomposition (v2)

### Base role — A1 Proposer (no adapter; the frozen p15d lineage)
CURSOR streaming ops (ADD/UPD/DEL/CMP/NOP/TITLE) — unchanged, spec core, the
maintainer's port. No adapter ever touches it.

### A2 — Normalizer (adapter, gated, decode-expensive — the one expensive role)
- **Job:** rewrite a noisy ASR chunk into clean, speaker-attributed declarative
  statements with inline decision/action/open-question tags (NexusSum-style). The
  proposer then streams over the cleaned rendering.
- **Gated [v2]:** fires ONLY when the deterministic content-rich check says the chunk
  has substance AND the proposer's NOP-collapse counter is high (the measured
  coverage-collapse regime). Refinement passes must be gated, not unconditional.
- **Data:** noisy-zh/real-meeting chunks → teacher's clean renderings (the real
  meeting + the ASR-noise augmentations already exist); per-role teacher outputs.
- **Eval:** ops/chunk + DECISIONS/ACTIONS on the maintainer's real meeting (target:
  ≥3 ops/chunk, both sections non-empty), G1 unchanged.

### A3 — Critic (adapter, ONE contiguous final phase) [v2: A3+A4 merged]
- **Job:** the final VERIFY sweep over every bullet against whole-transcript evidence
  (anchor neighbourhood ∪ lexical top-k) — SUPPORTED/UNSUPPORTED/CONTRADICTED, drop
  on the bad verdicts. The reversal clause lives here (enforceable with full
  evidence). In-stream per-op judging is DROPPED [v2] — the per-chunk switching cost
  and the ±90s structural blindness both argue against it; the final phase keeps the
  measured 0/20 inversion deployment.
- **Data:** verifier triples (class-balanced, claim + in-stream forms) + the
  polarity-flip counterfactuals.
- **Eval:** 200-triple agreement ≥95%, per-class CONTRADICTED recall, swept T1 0/20.

### A4 — Anchor picker (adapter, same final phase as A3)
Unchanged from v1: pick `[m:ss]` among ≤8 candidates or NONE → deterministic matcher
floor. Data: the 2100 anchor triples. (A3+A4 share one contiguous phase — switching
applies once, not per bullet.)

### A5 — Writer/merger (adapter, final phase) [v2: A6 renamed]
- **Job:** hierarchical final pass over SUMMARY+TITLE: the meeting arc, near-duplicate
  suppression (the stock-phrase loop), cap enforcement. Chained: output → anchor
  matcher → A3 verification before landing. Never touches DECISIONS/ACTIONS.
- **Gated [v2]:** fires only when the white-box probe (below) or the repetition
  heuristic flags the SUMMARY as weak; unconditional rewriting is the measured
  self-refinement failure (96.3→66.5 on a mastered task).
- **Data:** teacher SUMMARY/arc steps + CMP targets. Then **DPO** per the recipe
  (single-reject, β=0.5) against verbatim-copy, trigram repetition, entity drift.
- **Eval:** SYNTH ≥ baseline +0.5, repeated-trigram rate on the real meeting.

### W — White-box hallucination probe (free gate, conditional)
TrueBrief-style logistic classifier over logit/lookback features to detect
self-hallucination at ~zero marginal cost. **Conditional [v2]:** requires activation
access — not available in the llama.cpp GGUF serving path; adopt only with a runtime
that exposes hidden states (executorch/ONNX). Until then the A3 critic is the gate.

### R1–R6 — Deterministic guards (unchanged)
anchor validation · temporal guard · UPD→ADD fallback · dedup/caps · NOP-collapse +
coverage fallback.

## 2. Framework (v2)

```
stream:  A1 per chunk (unchanged render). If content-rich AND A1 silent:
         A2 normalizes the chunk (contiguous mini-phase) → A1 re-streams it.
final:   A3+A4 (one adapter phase) → A5 (gated) → A3 re-verifies A5's output
         → deterministic render.
```

- Phase-contiguous adapter use only; no per-chunk interleaving.
- Inter-agent messages stay plaintext (the rendered STATE and chunks).
- Fallbacks unchanged: every role has a deterministic floor.

## 3. Execution (v2)

- **Phase 0 (in progress):** critic rank sweep on the p13 base (r8 measured 64% —
  insufficient, as the capacity risk predicted; r32 running; r16 if r32 clears).
  This answers the one blocking question: can a low-rank adapter overwrite the
  generation bias for the verdict task? If no rank clears ≥95%, the critic stays the
  separate Granite-350M (two-model deployment for the critic only) and the LoRA plan
  proceeds for A2/A4/A5 only.
- **Phase 1:** A2 normalizer training (noisy→clean pairs, gated design), A4 anchor,
  A5 writer SFT + DPO.
- **Phase 2:** harness integration — the phase scheduler, `/lora-adapters` switching,
  the gating conditions (content-rich + NOP-collapse for A2; repetition/probe for A5).
- **Phase 3:** the measurement ladder on 50 real transcripts:
  1. FLAN-T5-780M fine-tuned hierarchical baseline (honesty check)
  2. single-adapter (A2 only) · 3. + normalizer · 4. + gated A5 · 5. full stack
  — measuring wall-clock, joules, MESA/P-MESA error typing (not ROUGE),
  repeated-trigram rate, INVERT, FAITH/COVER/SYNTH, G1.
- **Phase 4:** RLVR fallback (KL-to-base) for any role that fails SFT/DPO.
- **Phase 5:** shared-quantization adapter set, NPU-runtime decision (the white-box
  probe's runtime question), publish, on-device numbers.

## 4. Success criteria

| metric | today | target |
|---|---|---|
| INVERT (deployed) | 0/20 | 0/20 |
| FAITH | 4.54 | ≥ 4.5 |
| critic agreement | 97% (350M separate) | ≥95% (adapter) or Granite stays |
| ops/chunk, real noisy zh | ~1.8 | ≥3, DECISIONS+ACTIONS non-empty |
| SYNTH | 2.75 | ≥ +0.5 over baseline |
| repeated-trigram rate | stock-phrase loops observed | near zero |
| resident memory | 903 MB / 2 processes | base + adapters, 1 process (ranks 8–16) |
| wall-clock, 60-min meeting | — | measure; decode budget ≈ passes × output tokens / 22 t/s |

## 5. Risks (amended)

- **The capacity question (r8 64% measured):** if no rank clears the critic, the
  two-specialist deployment stands for A3 — recorded as the plan's fallback, not a
  failure.
- **A2's decode cost** is the dominant added latency (≈ chunk tokens × 1.5 output at
  22 t/s on-device) — the gating design exists to pay it only when coverage actually
  collapses; measure joules in Phase 3.
- **Switching cost** (~30% unfused): phase-contiguous use bounds it to ~3 switches
  per meeting.
- **The white-box probe needs a runtime change** (activations) — conditional item.
- **A5 overreach** — chained through the anchor matcher + A3, DECISIONS/ACTIONS
  untouched.
- **Zh trap:** the base is frozen; adapters cannot cross it in training.
- **Kotlin port dependency:** the maintainer's binding must support `/lora-adapters`;
  confirm before Phase 2.
