# SPEC — Agentic meeting summarizer (MiniCPM5-1B + external memory, zh-TW / en)

**Version:** 0.2 · **Status:** draft — goal, I/O formats, architecture and evaluation
defined; training-data construction and operational budget still open (§8)

---

## 1. Goal

**Fine-tune MiniCPM5-1B (Q8, 4k context) to drive a lightweight ("smol") agent
framework — built in this repo — that produces the final meeting summary (§3) from a
transcript (§2) too long to fit in one 4k-token pass.** Whole-meeting transcripts in
the training corpus average 14.1k tokens (VCSum) and 28.4k tokens (MeetingBank), with
individual meetings ranging wider — far beyond a single 4k window, so the student
cannot summarize in one shot: it has to learn to work incrementally, reading the
transcript through bounded steps and carrying what matters forward via **external
memory** across those steps, converging on one final <1,000-token prose summary.
Learning to use that external memory well — not raw context size — is the core
capability being trained.

---

## 2. Input — transcript format v2 (normative, timestamp-free)

One utterance per line. **One utterance = one line is a hard rule** (no embedded newlines).

```
<speaker>: <text>     diarized, name unknown  → S1, S2, …  (first-appearance order)
<name>: <text>        diarized, name known    → real name / role verbatim
UNK: <text>           diarization unavailable → reserved literal label UNK
```

- **No timestamp** — dropped from v1 specifically so timestamp-free public datasets
  (VCSum, §2.3) are valid input without forced alignment. §3's output is prose with no
  anchors, so nothing downstream depends on a `[m:ss]` being present.
- **Speaker field is mandatory.** v1's bare `<text>` form is removed: with the
  timestamp gone there is no prefix to anchor on, so a bare line whose text contains
  `: ` (`Here is the plan: we move to B`) is indistinguishable from a diarized one.
  When no speaker is known the emitter writes the reserved label `UNK`. This keeps
  `parse_line` total and unambiguous. Both training datasets carry speaker labels
  natively, so `UNK` occurs only in deployment-time input where diarization failed.
- **Speaker field**: `S1…Sn` (order of first appearance), a real name/role, or `UNK`.
  Never contains `: `, and is ≤ 40 chars.
- **No** header, footer, markdown, or escaping. Text is emitted as-is.
- **Parsing (normative)**: split on the FIRST `: `. `parse_line(line) → (speaker, text)`.
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

**At deployment time** the transcript is produced on-device by the **VoxSum Android
app** (ASR + speaker diarization), emitting format v2 directly. This system never
consumes raw audio — §2 is the whole input contract.

**For training and evaluation**, content is drawn **entirely** from two existing
public datasets — no manual sourcing of raw podcast/meeting audio, no other corpus:
- **en: MeetingBank** — 1,366 real US city-council meetings, transcripts with
  word-level speaker diarization + timestamps, paired with professionally-written
  official meeting minutes (segment-aligned). Timestamps are discarded on import (§2).
- **zh: VCSum** — 239 real Chinese meetings, transcripts with per-utterance speaker
  labels, paired with human-written overall + per-segment summaries. No native
  timestamps, which is fine — §2 is timestamp-free specifically to fit this dataset.

**Initial corpus (normative)**: the **full union** of both datasets — all 1,366
MeetingBank meetings (en) + all 239 VCSum meetings (zh) — no fixed sample size, no
subsetting.

**Import to format v2 (normative first preprocessing stage).** Neither dataset ships
in §2's format, so both are converted before any other use — this is the first stage
of the pipeline, and everything downstream sees only format v2:
- **MeetingBank**: word-level JSON dicts (`text`/`offset`/`duration`/`confidence`)
  with diarization → group consecutive words by speaker into one line per turn,
  discard all timing.
- **VCSum**: `speaker` + `context` JSON → one line per utterance, `<speaker>: <text>`.
- Speaker labels are renamed to `S1…Sn` in first-appearance order unless the dataset
  supplies a real name/role; `UNK` only where a dataset has no label at all (§2).

**VCSum zh-CN → zh-TW conversion (normative, after import).** VCSum ships in
Simplified Chinese; before any other use, both its transcripts and its reference
summaries are converted to zh-TW. Two candidate methods, **pick whichever is
measurably better, don't assume**:
- **OpenCC, `s2tw` config** — mechanical, cheap, deterministic (character conversion +
  Taiwan-standard lexicon substitution, not bare character mapping).
- **The teacher model (Unsloth Qwen3.8-27B, §4)** — if it produces measurably better
  zh-TW (more natural register/idiom, not just correct characters) than OpenCC on a
  sample, use it instead.

One-time offline stage; the corpus stored downstream is already zh-TW. Note this
breaks direct comparability with VCSum's published ROUGE baselines, which were
computed on the original zh-CN data (§5).

**Granularity (normative): segment-level summaries are intermediate, not the training
target.** Both datasets' per-segment summaries (MeetingBank's segment minutes, VCSum's
segmentation summaries) are input material for producing the final artifact, not the
artifact itself. The training target is the **whole-meeting** summary only, per §3.
- VCSum has one natively (231.9 tokens avg, per its published statistics).
- MeetingBank does not publish one — its ~9.8 segment minutes per meeting are the
  intermediate material the teacher (§4) synthesizes into a single whole-meeting
  summary. Consequence: the en reference is *our teacher's output*, not an independent
  gold standard (§5), which is what makes the human-validation step in §4
  load-bearing rather than optional.

---

## 3. Output (normative)

A single flowing **prose** summary — no bullets, no sections, no anchors. **< 1,000
tokens.** Output language matches the input transcript's language (zh-TW in, zh-TW
out). Structure and style within the prose are still open.

---

## 4. Architecture

- **Student / deployed model: MiniCPM5-1B, Q8, 4k context.** Single on-device model
  (not the prior project's 3-model pipeline) — CPU-only per §6. It drives a
  lightweight agent framework, developed in this repo, that steps through a long
  transcript (§2) in bounded chunks, maintaining an **external memory** (state) across
  steps, and eventually emits the final prose summary (§3). Concrete design still
  open: chunk size, memory representation, step/tool grammar, steps per meeting,
  termination condition (§8).
- **Teacher model: Unsloth Qwen3.8-27B, Q8 or BF16 quant** (exact quant TBD) — a
  strong, large model, offline only, never on the reference hardware (§6). Produces
  training-distillation ground truth (and/or eval reference) from the transcript —
  including synthesizing MeetingBank's whole-meeting summary from its segment minutes
  (§2.3) — never deployed on-device.
- **Human validation (normative)**: every teacher-generated summary must be manually
  reviewed by a human before it enters the training/eval corpus — no teacher output is
  trusted unvalidated. This is the only thing standing between the en reference and
  pure circularity (§2.3), so it is not optional.

---

## 5. Evaluation & measurement (normative)

Reuse each dataset's published metric suite rather than inventing one — but the two
are **not** equally grounded, and reported numbers must say which case they are:

- **zh (VCSum) — native whole-meeting standard.** ROUGE-1/2/L F1, as VCSum reports for
  its own "Abstractive Meeting Summarization" task (§5.3 of that paper), which is the
  same whole-meeting granularity as our target (§3). Caveat: our corpus is
  zh-TW-converted (§2.3), so scores are internally comparable but not directly
  comparable to the paper's published baselines.
- **en (MeetingBank) — borrowed suite, not a published whole-meeting standard.**
  MeetingBank's paper explicitly benchmarks *segments*, not whole meetings ("focusing
  on segments of the meetings rather than entire transcripts due to the length
  constraint imposed by abstractive summarizers"). We therefore reuse its Table 2
  suite — ROUGE-1/2/L (+ROUGE-WE), BLEU, METEOR, BERTScore, MoverScore, QAEval,
  summary length, plus Coverage/Density (Grusky et al. 2018) as extractiveness
  diagnostics — applied at whole-meeting granularity, against a teacher-synthesized,
  human-validated reference (§2.3). Neither the granularity nor the reference matches
  the paper's, so en scores are **not** comparable to published MeetingBank numbers.

**ROUGE on Chinese** requires a stated tokenization (character-level vs. word
segmenter); VCSum's exact choice is unrecorded here and must be matched or explicitly
declared before any zh number is reported (§8).

---

## 6. Reference hardware (normative)

- **Target/reference inference device: Oppo Reno 7 5G (model CPH2371), CPU-only
  inference.** No GPU/NPU acceleration path is assumed — the deployed model (§4) must
  run acceptably on this device's CPU alone.
- **Confirmed specs**:
  - SoC: MediaTek **Dimensity 900** (octa-core; 2× Cortex-A78 @ up to 2.4 GHz +
    6× Cortex-A55 @ up to 2.0 GHz)
  - RAM: **8 GB**
- These figures are what §7's operational budget must be sized against.

---

## 7. Operational budget (TBD)

Nothing here is set yet, and for a multi-step on-device agent these are the
constraints most likely to sink the design. To be specified and measured on §6's
hardware:

| quantity | why it matters |
|---|---|
| steps (model calls) per meeting | multiplies every other cost; a 28k-token transcript at 4k ctx is ≥8 reading passes before any revision |
| wall-clock per meeting | the actual user-facing latency on a Dimensity 900 CPU |
| peak RSS | 8 GB device, shared with the OS and the app |
| tokens in/out per step, and totals | prefill dominates cost in a chunk-streaming design |

---

## 8. Open questions

1. **Training-data construction** — the largest gap. §1 requires the model to learn
   external-memory use across steps, but the corpus only supplies *final* summaries.
   How per-step supervision (state + chunk → action) is generated is unspecified.
2. **Target-length asymmetry by language** — zh targets average 231.9 tokens (VCSum
   native) while en targets land near ~856 tokens (teacher synthesis over ~9.8 segment
   minutes). Both fit §3's cap, but training on them as-is teaches "zh ⇒ short, en ⇒
   long". Normalize target length, or accept and document.
3. **Agent design specifics** — chunk size, memory representation, step/tool grammar,
   termination condition (§4).
4. **zh ROUGE tokenization** — must be declared to match VCSum (§5).
5. **Train/deploy distribution gap** — training transcripts come from dataset text
   (clean, human-corrected or professionally transcribed); deployment transcripts come
   from on-device ASR + diarization (§2.3), which carries recognition and
   speaker-attribution errors the training data does not. Unmeasured for now.
