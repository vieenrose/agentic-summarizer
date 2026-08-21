# SPEC — Agentic meeting summarizer (MiniCPM5-1B + external memory, zh-TW)

**Version:** 0.4 · **Status:** design complete — corpus, agent protocol, training-data
construction, evaluation and budget all specified; see §8 for risks to measure

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
   target's register that of a generative model working in zh-TW; see §4.3 for the
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
  (not the prior project's 3-model pipeline) — CPU-only per §6. It drives the agent
  protocol in §4.1, doing two jobs: per-step memory curation while reading, and the
  final prose synthesis.
- **Teacher model: Unsloth Qwen3.8-27B, Q8 or BF16 quant** (exact quant TBD) — offline
  only, never on the reference hardware (§6). Synthesizes the zh-TW whole-meeting
  summary from translated segment minutes (§2.2 stage 3) and produces the per-step
  distillation targets (§4.2).
- **Translation model: TranslateGemma-27B** (`google/translategemma-27b-it`) — offline
  only, corpus construction stage 2 (§2.2). A dedicated translation model rather than
  the teacher, on the assumption it produces better zh-TW; gated on a sample (§4.3)
  before committing the full ~40M-token run.
- **Human validation (normative)**: every teacher-generated summary must be manually
  reviewed by a human before it enters the training/eval corpus. Given the
  translation-then-synthesis provenance (§2.2), this is the only human-authored signal
  in the corpus and is not optional.

### 4.1 Agent protocol (normative)

The transcript is read as a stream. The harness owns the memory; the model only emits
edit lines. No conversation history crosses steps — memory is the entire carry-forward,
which is the property that keeps each step's context constant-size and learnable at 1B.

**External memory.** Two slots, harness-rendered, capped:

```
ARC: <1–3 sentences: how the meeting has developed so far>
POINTS:
- <key point, decision, or commitment>
```

`ARC` ≤ 80 tokens; `POINTS` ≤ 16 entries of ≤ 25 tokens. Total ≤ ~600 tokens by
construction. `ARC` exists because a flat point list loses the meeting-level
through-line, and §3's output is connected prose — the prior project needed a separate
synthesizer stage largely because bullets alone read as fragments. Carrying the arc
incrementally gives the final synthesis something to build on beyond a list.

**Step grammar.** One call per chunk; zero or more lines:

| op | syntax | semantics |
|---|---|---|
| ADD | `ADD - <point>` | append a point |
| DROP | `DROP «<prefix>»` | remove a point this chunk supersedes |
| ARC | `ARC: <text>` | replace the arc note |
| NOP | `NOP` | nothing worth recording in this chunk |

Deliberately small. No multi-point rewrite op: the prior project measured that as the
heaviest op in its grammar and never validated it at ≤1B. Cap overflow is handled
**deterministically by the harness** (evenly spread, never head-truncated — dropping
the tail of a time-ordered list drops the end of the meeting, where decisions land),
never by asking the model to rewrite the list.

**Termination.** Transcript exhausted → one final `SYNTHESIZE` call: memory → §3 prose.

**Context budget (4k).**

| | reading step | synthesis step |
|---|---|---|
| SYS | ~250 | ~250 |
| memory | ≤600 | ≤600 |
| chunk | ~2,500 | — |
| output | ~150 (edit lines) | <1,000 (prose) |
| **total** | **~3,500** | **~1,850** |

Chunk size ~2,500 tokens follows from the budget, not preference. Note the fit with
MeetingBank's own structure: its segments average 2,892 en tokens, so **one segment ≈
one chunk**, and chunk boundaries can follow the dataset's human-defined segment
boundaries rather than arbitrary token offsets (long segments split, short ones stay
whole). Step count per meeting is then ~9.8 reading steps + 1 synthesis ≈ **11 calls**,
pending the zh-TW token measurement (§7).

### 4.2 Training-data construction (normative)

The corpus supplies only final summaries, so per-step supervision has to be
constructed. The key asset is that **MeetingBank's segment minutes are already aligned
to transcript segments and professionally authored** — the mapping from "this stretch
of transcript" to "what mattered in it" is human-made, not model-invented, which is a
stronger starting point than the prior project had (its teacher had to invent gold
edits from whole-transcript foresight alone).

Per meeting, walking segments in order:

1. **Reading steps.** For step *i*, the input is (memory after step *i*−1, chunk *i*).
   The gold edit lines are derived by the teacher from **segment *i*'s translated
   minute** — a narrow, grounded conversion task ("express this minute as ADD/ARC lines
   against the current memory"), not open-ended summarization. The teacher additionally
   sees later segments' minutes, and only that foresight is used to emit `DROP` for
   points a later segment supersedes.
2. **Synthesis step.** Input is the final memory; target is the human-validated
   whole-meeting summary (§2.2 stage 3–4).

Both step types train the same model (§4). Supervision volume: ~1,092 train meetings ×
~11 steps ≈ **~12k training steps**, of which ~1.1k are synthesis steps — the synthesis
skill therefore sees roughly a tenth of the data the curation skill does, which is a
risk to watch (§8).

**Validation.** Every gold edit sequence is replayed through the real harness before
use: ops must parse, `DROP` prefixes must match an existing point, and the resulting
memory must respect the caps. A sequence that fails replay is regenerated or dropped —
never half-applied into the corpus.

### 4.3 Corpus-construction gates (normative)

Two sample-first gates, both cheap relative to the full runs they precede:

**Translation gate** (before the ~40M-token run). Translate 20 meetings, then check:
- **Line-count integrity is a hard pass/fail.** Format v2 is one utterance per line, so
  translation must preserve the line mapping exactly — a document-level translation that
  merges or splits utterances corrupts the format silently. Translate per-line (in
  context windows, with line boundaries enforced) and assert input and output line
  counts match.
- Speaker labels pass through untranslated.
- zh-TW register, not zh-CN vocabulary leaking through.
- Terminology consistency for the recurring municipal-procedure vocabulary
  (ordinance/motion/council/committee), which appears in every meeting.
- Human read of 3 full transcripts for fluent-but-foreign output — the expected failure
  mode, since US municipal government has no Taiwanese counterpart.

**Synthesis-ordering gate.** §2.2 translates the minutes and then synthesizes the
summary in zh-TW. The alternative — synthesize in en, then translate the summary — puts
a dedicated translation model on the final target instead of a generative one. Run both
on the same 20 meetings and pick by blind human preference for zh-TW naturalness. Cheap
to measure, not worth guessing.

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

| metric | decision |
|---|---|
| ROUGE-1/2/L | **character-level** (CJK split per character; embedded Latin words and numbers split on whitespace). This is standard practice in Chinese summarization precisely to avoid making scores a function of which word segmenter was chosen. Recorded here as normative — a later switch to segmenter-based ROUGE would invalidate comparison with everything measured before it. |
| BLEU / METEOR | SacreBLEU with `tokenize=zh` |
| BERTScore | keep, with a Chinese or multilingual encoder |
| MoverScore | keep — monolingual-English only by default, but the implementation accepts a multilingual BERT, so it survives an encoder swap |
| Coverage / Density | keep — token-overlap diagnostics, language-agnostic once tokenization is fixed (reuse the character-level tokenization above) |
| summary length | keep, in characters and in MiniCPM5 tokens |
| QAEval | **dropped.** Needs Chinese question-generation and question-answering models; no supported zh path. Its natural substitute is an LLM-judge, which the contamination rule below constrains — so faithfulness is measured by ROUGE/BERTScore proxies plus the human review already required by §4, and is otherwise **unmeasured**. Stated rather than papered over: this is the weakest part of the metric set. |

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

## 7. Operational budget

Derived from §4.1's protocol; **all figures are projections pending measurement on §6's
hardware**, and every one of them scales with the zh-TW token count, which is the
prerequisite measurement (the ~28k transcript figure is English — Chinese tokenizes
differently under MiniCPM5's vocabulary, and the direction is not obvious enough to
assume).

| quantity | projection | basis |
|---|---|---|
| calls per meeting | ~11 (≈9.8 reading + 1 synthesis) | one chunk ≈ one MeetingBank segment (§4.1) |
| prefill per meeting | ~38k tokens | ~3.5k × 10 reading steps + ~0.9k synthesis |
| decode per meeting | ~2.5k tokens | ~150 × 10 edit-line steps + <1,000 prose |
| wall-clock per meeting | **must measure** | the user-facing number on a Dimensity 900 CPU; dominated by prefill |
| peak RSS | **must measure** | 8 GB device shared with OS and app; single ~1B Q8 model resident |

The design's efficiency argument is that memory is capped, so per-step context is
**constant-size regardless of meeting length** — a 3-hour meeting costs more steps, not
bigger steps, and never exceeds 4k.

---

## 8. Known risks

Design is specified end to end; what remains are risks to measure, not gaps to fill.

1. **Two skills, one 1B model, unequal data.** §4 merges memory curation and prose
   synthesis into one model, and §4.2's supervision gives synthesis only ~1/10th the
   steps curation gets. This deliberately contradicts the prior project, which used a
   *separate* synthesizer and carried an explicit exclusion against folding synthesis
   into the note-taker (the stated reason: coupling synthesis capability to the
   note-taker's retrain cycle). Merging is justified here by the single-model,
   single-device constraint (§6) — but it is an untested reversal of a prior decision,
   and the first thing to check if prose quality disappoints.
2. **Train/deploy distribution gap — currently unmeasurable.** Training transcripts are
   translated, professionally-transcribed text; deployment transcripts come from the
   VoxSum app's on-device ASR + diarization, with recognition and speaker-attribution
   errors the training data has none of. The prior project hit exactly this failure
   (its verifier produced 0/11 parseable verdicts on real noisy zh windows while
   scoring 8/9 on clean zh). **The corpus cannot measure it: MeetingBank's audio is
   English, so no zh-TW audio exists anywhere in this pipeline.** Closing it needs
   either simulated ASR-noise injection, or a small held-out zh-TW audio eval slice
   (e.g. ~10 立法院 committee recordings through the VoxSum pipeline) — eval only, not
   training. Undecided.
3. **Translationese in every training target.** Gated at 20 meetings (§4.3), but the
   gate can only reject an obviously-bad translation; a subtly foreign register would
   pass review and propagate into all 1,366 targets. There is no native zh-TW reference
   anywhere in the corpus to catch it.
4. **Faithfulness is unmeasured.** QAEval is dropped with no zh substitute, and the
   contamination rule (§5) blocks the obvious LLM-judge replacement. Nothing in the
   metric set detects a fluent summary that states the opposite of the transcript —
   which was a first-class product requirement in the prior project (0% inversions).
5. **Projections, not measurements.** Every figure in §7, and the ~11-calls and
   one-segment-per-chunk assumptions in §4.1, rest on English token counts. The zh-TW
   measurement under MiniCPM5's tokenizer is the first thing to run.
