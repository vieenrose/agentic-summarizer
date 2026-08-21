# SPEC — [project name / goal TBD]

**Version:** 0.1 · **Status:** draft — goal, output format, and architecture not yet defined

---

## 1. Goal

TBD.

---

## 2. Input — transcript format v2 (normative, timestamp-free)

One utterance per line. **One utterance = one line is a hard rule** (no embedded newlines).

```
<speaker>: <text>     diarized, name unknown  → S1, S2, …  (first-appearance order)
<name>: <text>        diarized, name known    → real name / role verbatim
<text>                no diarization          → no speaker field
```

- **No timestamp** — dropped from v1 specifically so timestamp-free public datasets
  (VCSum, §2.3) are valid input without forced alignment. §3's output is prose with no
  anchors, so nothing downstream depends on a `[m:ss]` being present.
- **Speaker field**: `S1…Sn` (order of first appearance), a real name/role, or absent.
  A speaker field never contains `: ` and is ≤ 40 chars.
- **No** header, footer, markdown, or escaping. Text is emitted as-is.
- **Parsing (normative)**: split on the FIRST `: `. `parse_line(line) → (speaker|None, text)`.
- Long monologue lines can occur (up to ~2.6k chars/line in real zh-TW source material
  seen previously) — readers must not assume a max line length.

### 2.1 Example (en)

```
S1: Let us discuss the office move.
S2: I propose we move to Building B.
S1: Agreed, Building B it is.
```

### 2.2 Example (zh-TW)

```
S1: 我們來討論辦公室搬遷。
S2: 我建議搬到 B 棟大樓。
S1: 好，就搬到 B 棟。
```

### 2.3 Content source (normative)

**This is the training data.** Content is drawn **entirely** from two existing public
datasets — no manual sourcing of raw podcast/meeting audio, no other corpus:
- **en: MeetingBank** — 1,366 real US city-council meetings, transcripts with
  word-level speaker diarization + timestamps, paired with professionally-written
  official meeting minutes (segment-aligned).
- **zh: VCSum** — 239 real Chinese meetings, transcripts with per-utterance speaker
  labels, paired with human-written overall + per-segment summaries. No native
  timestamps, which is fine — §2 is timestamp-free specifically to fit this dataset.

**Resolved: VCSum zh-CN → zh-TW conversion (normative preprocessing stage).** VCSum
ships in Simplified Chinese; before any other use, both its transcripts and its
reference summaries are converted to zh-TW. Two candidate methods, **pick whichever
is measurably better, don't assume**:
- **OpenCC, `s2tw` config** — mechanical, cheap, deterministic (character conversion +
  Taiwan-standard lexicon substitution, not bare character mapping).
- **The teacher model (Unsloth Qwen3.8-27B, §4)** — if it produces measurably better
  zh-TW (more natural register/idiom, not just correct characters) than OpenCC on a
  sample, use it instead for this conversion step.

Either way this is a one-time offline preprocessing stage — the corpus stored/used
downstream is already zh-TW.

**Initial corpus (normative)**: the **full union** of both datasets — all 1,366
MeetingBank meetings (en) + all 239 VCSum meetings (zh) — no fixed sample size, no
subsetting.

**Granularity (normative): segment-level summaries are intermediate, not the training
target.** Both datasets' per-segment summaries (MeetingBank's segment minutes, VCSum's
segmentation summaries) are inputs to producing the final artifact, not the artifact
itself. The training target is the **whole-meeting** summary only, per §3. VCSum
already has one natively (231.9 tokens avg, per its own published dataset statistics).
MeetingBank does not publish
a whole-meeting summary directly — its segment minutes are the intermediate material
the teacher model (§4) synthesizes into one final whole-meeting summary.

---

## 3. Output (normative)

A single flowing **prose** summary — no bullets, no sections, no anchors. **< 1,000
tokens** (relaxed from the earlier 500-token cap). Everything else (structure/style
within that prose, language matching input, etc.) is still open.

---

## 4. Architecture — mostly TBD; one piece confirmed

- **Teacher model: Unsloth Qwen3.8-27B, Q8 or BF16 quant** (exact quant TBD) — a
  strong, large model, offline only, never on the reference hardware (§6) — runs once
  per meeting to turn the audio pipeline's transcript (§2) into a very-high-quality
  reference summary. This is
  training-distillation ground truth (and/or eval reference), not a deployed
  component. The actual on-device model(s) that run at inference time are separate,
  small, and CPU-only per §6; everything else about the architecture (how many
  models, what each does, how the teacher's output is distilled down) is still open.
- **Human validation (normative)**: every teacher-generated summary must be manually
  reviewed by a human before it enters the training/eval corpus — no teacher output is
  trusted unvalidated, matching the human-in-the-loop discipline the prior project
  applied to its own judge/verifier stages.

---

## 5. Evaluation & measurement — TBD

---

## 6. Reference hardware (normative)

- **Target/reference inference device: Oppo Reno 7 5G (model CPH2371), CPU-only
  inference.** No GPU/NPU acceleration path is assumed — all on-device inference (any
  model in the eventual architecture, §4) must run acceptably on this device's CPU
  alone.
- **Confirmed specs**:
  - SoC: MediaTek **Dimensity 900** (octa-core; 2× Cortex-A78 @ up to 2.4 GHz +
    6× Cortex-A55 @ up to 2.0 GHz, per Dimensity 900's known big.LITTLE layout)
  - RAM: **8 GB**
- These figures are what the peak-RSS and latency budgets (§5) must be sized against
  once the architecture (§4) is defined — no more guessing across regional variants.
