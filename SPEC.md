# SPEC — Agentic meeting summarizer (MiniCPM5-1B + external memory, zh-TW)

**Version:** 0.3 · **Status:** draft — goal, I/O formats, architecture and evaluation
defined; training-data construction and operational budget still open (§8)

---

## 1. Goal

**Fine-tune MiniCPM5-1B (Q8, 4k context) to drive a lightweight ("smol") agent
framework — built in this repo — that produces a zh-TW meeting summary (§3) from a
zh-TW transcript (§2) too long to fit in one 4k-token pass.** Whole-meeting
transcripts average ~28k tokens in the source material, far beyond a single 4k
window, so the student cannot summarize in one shot: it has to learn to work
incrementally, reading the transcript through bounded steps and carrying what matters
forward via **external memory** across those steps, converging on one final
<1,000-token prose summary. Learning to use that external memory well — not raw
context size — is the core capability being trained.

**Single target language: zh-TW.** English is out of scope as a product language; it
appears only as the source material that gets translated during corpus construction
(§2.2).

---

## 2. Input — transcript format v2 (normative, timestamp-free)

One utterance per line. **One utterance = one line is a hard rule** (no embedded newlines).

```
<speaker>: <text>     diarized, name unknown  → S1, S2, …  (first-appearance order)
<name>: <text>        diarized, name known    → real name / role verbatim
UNK: <text>           diarization unavailable → reserved literal label UNK
```

- **No timestamp.** §3's output is prose with no anchors, so nothing downstream
  depends on a `[m:ss]` being present, and dropping it removes a whole class of
  alignment work from corpus construction.
- **Speaker field is mandatory.** A bare `<text>` form would be ambiguous: with no
  timestamp prefix to anchor on, a line whose text contains `: ` (`本案的重點: 我們搬到
  B 棟`) is indistinguishable from a diarized one. When no speaker is known the
  emitter writes the reserved label `UNK`. This keeps `parse_line` total and
  unambiguous.
- **Speaker field**: `S1…Sn` (order of first appearance), a real name/role, or `UNK`.
  Never contains `: `, and is ≤ 40 chars.
- **No** header, footer, markdown, or escaping. Text is emitted as-is.
- **Parsing (normative)**: split on the FIRST `: `. `parse_line(line) → (speaker, text)`.
- Long monologue lines can occur (up to ~2.6k chars/line in real zh source material) —
  readers must not assume a max line length.

### 2.1 Example

```
S1: 我們來討論辦公室搬遷。
S2: 我建議搬到 B 棟大樓。
S1: 好，就搬到 B 棟。
```

### 2.2 Content source (normative)

**At deployment time** the transcript is produced on-device by the **VoxSum Android
app** (ASR + speaker diarization), emitting format v2 directly. This system never
consumes raw audio — §2 is the whole input contract.

**For training and evaluation** the corpus is **MeetingBank, translated to zh-TW** —
all 1,366 meetings, no subsetting, no other corpus. MeetingBank supplies 1,366 real US
city-council meetings (~28k tokens per transcript, 2.6 h average, 2–19 speakers) with
word-level speaker diarization and professionally-written official minutes aligned to
~9.8 segments per meeting.

**The English original is source material, not training or eval data** — nothing in
the shipped corpus is English, and no en metric is reported. VCSum was considered and
dropped: at 239 meetings (193 train) it yields ~908 training steps against
MeetingBank's ~10,322, an 11.4:1 shortfall that made it unusable as the primary zh
source.

Corpus construction, in order — each stage's output is the next stage's only input:

1. **Import to format v2.** MeetingBank ships word-level JSON dicts
   (`text`/`offset`/`duration`/`confidence`) with diarization → group consecutive
   words by speaker into one line per turn, discard all timing. Speaker labels are
   renamed to `S1…Sn` in first-appearance order unless a real name/role is supplied;
   `UNK` only where no label exists (§2).
2. **Translate en → zh-TW** with **TranslateGemma-27B** (`google/translategemma-27b-it`;
   Traditional Chinese is an explicitly supported target in its tech report). Both the
   transcript *and* the segment minutes are translated, so everything downstream is
   zh-TW. Speaker labels are not translated. Scale: ~38.7M source tokens of transcript
   plus ~1.2M of minutes.
3. **Synthesize the whole-meeting summary** from the translated segment minutes with
   the teacher (§4), generating **in zh-TW**. MeetingBank publishes no whole-meeting
   summary, so this is where the training target comes from. Generating in the target
   language (rather than synthesizing in en and translating the result) keeps the
   target's register that of a generative model working in zh-TW; see §8 for the
   alternative ordering.
4. **Human validation** of every synthesized summary (§4) before it enters the corpus.

**Granularity (normative): segment minutes are intermediate, not the training
target.** The ~9.8 per-segment minutes per meeting are input material for stage 3, not
the artifact. The training target is the **whole-meeting** summary only, per §3.

**Provenance consequence, stated plainly**: the reference summary is
translation-then-synthesis — TranslateGemma output fed to Qwen — with no
human-authored zh-TW text anywhere in the chain. The human-validation stage is
therefore the only thing preventing full circularity, which is why §4 marks it
non-optional. Token counts also shift under translation; the ~28k figure is English,
and the zh-TW token count under MiniCPM5's tokenizer must be measured, not assumed
(§7, §8).

---

## 3. Output (normative)

A single flowing **zh-TW prose** summary — no bullets, no sections, no anchors.
**< 1,000 tokens.** Structure and style within the prose are still open.

---

## 4. Architecture

- **Student / deployed model: MiniCPM5-1B, Q8, 4k context.** Single on-device model
  (not the prior project's 3-model pipeline) — CPU-only per §6. It drives a
  lightweight agent framework, developed in this repo, that steps through a long
  transcript (§2) in bounded chunks, maintaining an **external memory** (state) across
  steps, and eventually emits the final prose summary (§3). Concrete design still
  open: chunk size, memory representation, step/tool grammar, steps per meeting,
  termination condition (§8).
- **Teacher model: Unsloth Qwen3.8-27B, Q8 or BF16 quant** (exact quant TBD) — offline
  only, never on the reference hardware (§6). Synthesizes the zh-TW whole-meeting
  summary from translated segment minutes (§2.2 stage 3) and produces any per-step
  distillation targets.
- **Translation model: TranslateGemma-27B** (`google/translategemma-27b-it`) — offline
  only, corpus construction stage 2 (§2.2). A dedicated translation model rather than
  the teacher, on the assumption it produces better zh-TW; verify on a sample before
  committing the full 38.7M-token run.
- **Human validation (normative)**: every teacher-generated summary must be manually
  reviewed by a human before it enters the training/eval corpus. Given the
  translation-then-synthesis provenance (§2.2), this is the only human-authored signal
  in the corpus and is not optional.

---

## 5. Evaluation & measurement (normative)

The corpus is translated MeetingBank (§2.2), so **no published benchmark applies
directly** and no reported number is comparable to prior work. Two independent reasons:

1. MeetingBank's paper benchmarks *segments*, not whole meetings ("focusing on
   segments of the meetings rather than entire transcripts due to the length
   constraint imposed by abstractive summarizers") — our granularity differs.
2. Our data is zh-TW translation, evaluated against a teacher-synthesized reference —
   neither the language nor the reference matches the paper's.

Metric suite, reusing MeetingBank's Table 2 set where the tooling actually supports
zh-TW:

| metric | zh-TW viable? |
|---|---|
| ROUGE-1/2/L | yes, but **only with a declared tokenization** (character-level vs. word segmenter) — the choice changes the numbers and must be recorded (§8) |
| BLEU / METEOR | yes via SacreBLEU with a zh tokenizer |
| BERTScore | yes with a Chinese or multilingual encoder |
| Coverage / Density | yes — token-overlap diagnostics, language-agnostic once tokenization is fixed |
| summary length | yes |
| MoverScore, QAEval | **English-only tooling in practice** — substitutes or omission needed (§8) |

**Judge contamination note**: if an LLM-judge is added later, it must not be
Gemma-family — TranslateGemma authored the corpus text — nor Qwen-family, which
authored the reference summaries. That rules out both model families the pipeline
already depends on.

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
| steps (model calls) per meeting | multiplies every other cost; a ~28k-token transcript at 4k ctx is ≥8 reading passes before any revision |
| wall-clock per meeting | the actual user-facing latency on a Dimensity 900 CPU |
| peak RSS | 8 GB device, shared with the OS and the app |
| tokens in/out per step, and totals | prefill dominates cost in a chunk-streaming design |

Prerequisite: measure zh-TW transcript length under MiniCPM5's tokenizer. Every figure
above derives from it, and the ~28k source figure is English.

---

## 8. Open questions

1. **Training-data construction** — the largest gap. §1 requires the model to learn
   external-memory use across steps, but the corpus supplies only *final* summaries.
   How per-step supervision (state + chunk → action) is generated is unspecified.
2. **Agent design specifics** — chunk size, memory representation, step/tool grammar,
   termination condition (§4).
3. **Translation quality gate** — verify TranslateGemma's zh-TW on a sample before the
   full ~38.7M-token run. The domain is US municipal government (ordinances, council
   procedure, place names) with no Taiwanese counterpart, so fluent-but-foreign output
   is the expected failure mode, and it would propagate into every training target.
4. **Synthesis ordering** — §2.2 translates then synthesizes in zh-TW. The alternative
   (synthesize the summary in en, then translate it) puts a dedicated translation model
   on the final target instead of a generative one; cheaper to compare than to guess.
5. **zh ROUGE tokenization** — must be declared before any number is reported (§5).
6. **zh substitutes for MoverScore / QAEval** — or an explicit decision to drop them
   (§5).
7. **Train/deploy distribution gap** — training transcripts are translated, clean,
   professionally transcribed text; deployment transcripts come from on-device ASR +
   diarization (§2.2) and carry recognition and speaker-attribution errors the training
   data has none of. Unmeasured.
