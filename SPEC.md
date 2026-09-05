# SPEC — Agentic Meeting Summarizer

**v1.4 · normative.** Where code and this document disagree, this document wins.
Every rule here was forced by a measurement; the measurements live in
[`SPEC-RATIONALE.md`](SPEC-RATIONALE.md), cited as `[R §x]`. Read that before
changing a rule — most were already tried the other way.

---

## 1. Objective

Fine-tune **Qwen3.5-0.8B** to drive an agent that turns a zh-TW meeting transcript
(~34k tokens) into **one flowing zh-TW prose summary under 1,000 tokens**, on a
phone, offline.

The context window is 4k. The transcript does not fit. **The capability being
trained is external-memory curation, not context length.**

zh-TW is the only product language. English appears only as source material.

---

## 2. Input — transcript format v2

One utterance per line. No timestamps.

```
<speaker>: <text>
```

- **Speaker field is mandatory.** `S1…Sn` by first appearance, a real name/role, or
  the literal `UNK`. Never contains `: `; ≤ 40 chars.
- Parse by splitting on the **first** `: `. `parse_line` must be total. `[R §2]`
- No header, footer, markdown, or escaping. Lines may reach ~2.6k chars.

---

## 3. Output

A single flowing **zh-TW prose** summary. No bullets, no sections, no anchors.
**< 1,000 tokens.**

---

## 4. Protocol

The transcript is split into **~2,500-token chunks**, line-atomic.

**Per chunk, the model emits exactly one tool call** editing a harness-owned memory:

| slot | cap | semantics |
|---|---|---|
| `ARC` | ≤ 80 tokens | running context, 1–3 sentences |
| `POINTS` | ≤ 16 entries × ≤ 25 tokens | working set, addressed by **stable integer id** |
| `JOURNAL` | unbounded | append-only, **model never reads it** |

Operations: `add`, `revise(id)`, `drop(id)`, `arc`, `nop`.

**Rules that are load-bearing:**

- **No conversation history crosses steps.** Each step sees system prompt + memory +
  its chunk. This is what makes per-step cost constant. `[R §4.1]`
- **A point leaving the working set is retired to the journal, never destroyed.**
  Under v1.0 eviction destroyed 48–80% of correctly recorded points. `[R §4.1]`
- **`SYNTHESIZE` reads working set + journal**, not the 16-entry window.
- **The harness applies ops deterministically and never repairs.** Malformed input is
  recorded, never fatal. Overflow spreads evenly; it never head-truncates.
- **Revision must be expressed as `revise`**, not `drop`+`add` — the two are
  indistinguishable from churn otherwise. `[R §4.1]`

**Caps are CONVENTIONS, not constraints.** Cap-overflow refusals are 0.6% of ops;
~22% of output re-emits recorded content. Do not raise a cap to recover content.
`[R §4.1.1]`

---

## 5. Corpus

**MeetingBank, Zenodo record 7989108** — 1,250 annotated meetings. The Hugging Face
mirror is a stripped derivative and must not be used. `[R §2.2]`

Stages, each the next's only input:

1. **Import** word-level diarized JSON to format v2.
2. **Translate** en → zh-TW with TranslateGemma-27B, per-line, line counts asserted.
3. **Compose** the whole-meeting summary with the Qwen teacher from translated item
   minutes in order.
4. **Validate** a random sample of 30 per tranche by human read; reject the tranche
   above a 10% defect rate. `[R §5.3]`

**Per-step supervision** is derived by walking chunks: covered spans convert their
aligned minute to ops; uncovered spans are judged against neighbouring items, never
blanket-`NOP`ed. **Every gold sequence must replay cleanly through the real harness
before use.**

---

## 6. Hardware & budget

| | |
|---|---|
| Device | Oppo Reno 7 5G (CPH2371), CPU only, all 8 cores (`-C 0xFF`) |
| Model | Q8_0, 4k context |
| Ceiling | **20.00 min/meeting**, ≤ 2.5 GB RSS |

Measured: 2,811 prefill + 80.3 decode tokens per step, 16.5 steps/meeting →
**16.24 min**. An unstarved build costs **18.51 min**. `[R §7]`

**G4 is projected by `evalkit.latency` from the run's own token profile.** Prefill is
measured at depth 0; decode at the prompt's depth. Decode length is a property of the
**checkpoint**, not the device. `[R §9]`

---

## 7. Evaluation

**Baseline: map-reduce** — same model, same chunks, summarised independently then
compressed once. Deliberately fair.

### Ship gates

| gate | criterion |
|---|---|
| **G1** revision | independent probe **with control arm**; WITHHELD if it cannot discriminate |
| **G2** faithfulness | inversions ≤ baseline, on a 5-judge third-family panel, majority per claim, **inversions on unanimity** |
| **G3** | **RETIRED** — references are Qwen-authored; it measured teacher imitation |
| **G4** budget | fits §6 |
| **G5** retention | ≥ 90% of recorded points rendered, churn no worse than baseline |
| **G6** grounding | ≤ 10% of asserted specifics absent from the transcript |
| **G7** stability | churn ≤ 10% of steps; no ARC frozen from step 0 |
| **G8** coverage | ≤ 25% meetings starved, ≥ 0.5 points/chunk — **joint with G6** |
| **G9** utility | **the only positive gate.** 12 meetings, ≥ 2 blind reviewers |
| **G10** transfer | G5–G9 re-measured on real zh-TW ASR |

### Rules of evidence

- **Two seeds minimum.** Seed alone moves churn 27 points. `[R §5.2.1]`
- **A gate whose between-seed spread exceeds its margin to the threshold is
  WITHHELD**, never passed. `[R §5.2.9]`
- **Gate a metric together with the defect that can inflate it** — G5 with G7,
  G8 with G6. Every rate here has a denominator the model controls. `[R §5.2.2, §5.2.6]`
- **`eval_loss` does not select checkpoints.** Export candidates and measure
  behaviourally. `[R §5.0]`
- **An instrument must be validated before its output decides anything.** `[R §5.0]`

### Decision

1. **Ship the agent** — all gates pass.
2. **Ship routed** — agent on the slice it is gated on, baseline elsewhere. The
   routing key must be computable **before** inference.
3. **Ship the baseline** — and record the negative result.

**Passing the gates is not sufficient.** The agent must beat map-reduce at something
map-reduce cannot do by construction: **revision**, or **long-meeting coherence**. A
build that does neither has not earned its complexity → outcome 3. `[R §5.2.10]`

---

## 8. Status

Working end to end. G4, G5, G6, G7 pass; G2's ordering is established but the
specified panel aggregation has never been run; G8 fails (46% abstention); G9 has
never been run. `[R §5.3]`
