# SPEC — Agentic meeting summarizer (MiniCPM5-1B + external memory, zh-TW)

**Version:** 0.7 · **Status:** design + execution plan complete — §9 phases it
cheapest-first with gates; §8 attaches each risk to the phase that tests it

**v0.7 changes** — the corpus claims in §2.2 were audited against the actual Zenodo
release rather than the paper, and two of them were wrong. Word-level diarization and
whole-meeting transcripts are **present** (v0.6 was right, and the Hugging Face mirror
that lacks them is a stripped derivative — now named as non-normative). The **full
minutes document is absent**, so §2.2 stage 3 and §4.2 step 3 lose their intended
grounding; the ARC slot is demoted to a Phase-2 ablation (risk 8). Counts corrected to
**1,250 annotated meetings / 6,894 items**, and coverage measured at **56.8%** rather
than the assumed ~51%. Risk 7 records that dropping the verifier contradicts a prior
measured result.

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
no other corpus. MeetingBank supplies **1,366 transcribed** real US city-council meetings
(~28k tokens per transcript, 2.6 h average, 2–19 speakers) with word-level speaker
diarization, of which **1,250 carry aligned summaries** and are therefore the usable
corpus. All 1,250 are in scope, but they are built in two tranches: a 200-meeting pilot,
then the remainder only if the pilot clears its gates (§9). One exception to "no other
corpus": a small held-out zh-TW audio slice used for evaluation only, never training
(§9 Phase 3).

**Which release (normative).** The authoritative distribution is **Zenodo record
7989108** (`MeetingBank.zip`), which carries the word-level diarized transcripts and
`Metadata/MeetingBank.json`. The Hugging Face `huuuyeah/meetingbank` dataset is a
**stripped derivative** — a flat, speakerless, segment-level text blob — and must not be
used for corpus construction; it satisfies none of stage 1 below. (`lytang/MeetingBank-
transcript` is an intermediate form: speaker-labelled, but segment-level and already
turn-grouped.)

**What MeetingBank does and does not provide as summaries** — this determines
everything about supervision (§4.2), so it is worth stating exactly. All figures below
were measured directly from the Zenodo release, not taken from the paper:
- **It does provide human-authored summaries**, and they are good ones: professional
  city-clerk meeting minutes, split by the dataset's authors into passages and aligned
  to specific transcript spans. **6,894 items over 1,250 annotated meetings = ~5.5
  summarized items per meeting**, every one with a non-empty summary, averaging 87 en
  tokens each (~439 tokens of human-authored summary text per meeting). There is no
  reservoir of additional unreleased minutes.
- **Each item carries its own `startTime`/`endTime`** against the whole-meeting timeline,
  which is what makes the covered/uncovered split of §4.2 computable rather than assumed.
- **The full minutes document is NOT distributed.** The paper's description mentions PDF
  minutes, but no PDF appears anywhere in the Zenodo release — the per-item `Summary`
  fields are the only human-authored summary text available. Anything in this spec that
  previously rested on a whole-meeting minutes document has been re-grounded on the
  concatenated per-item summaries plus their type and ordering (stage 3, §4.2 step 3).
  Obtaining the real PDFs would mean scraping each meeting's `URLs.MeetingDetail`
  Legistar page for 1,250 meetings; cost that before assuming it.
- **Coverage is partial — measured.** Item spans cover **56.8% of meeting duration on
  average (59.4% median)**, and 51% of meetings are under 60% covered. So roughly
  **two-fifths of the meeting has no gold minute** at all. Items were filtered (≥60 s,
  summary ≥10 words), so procedural stretches — roll call, motions to adjourn — are
  simply absent from the summarized set.
- **What is missing is the format, not the humans.** There is no whole-meeting prose
  summary matching §3: the minutes are a procedural record (motion numbers, votes,
  ordinance IDs) at segment granularity, not a connected <1,000-token narrative.

**The English original is source material, not training or eval data** — nothing in
the shipped corpus is English, and no en metric is reported. VCSum was considered and
dropped: at 239 meetings (193 train) it yields ~908 training steps against
MeetingBank's ~10,322, an 11.4:1 shortfall that made it unusable as the primary zh
source.

Corpus construction, in order — each stage's output is the next stage's only input:

1. **Import to format v2.** The Zenodo transcripts ship as
   `segments[].nbest[0].words[]`, each word a dict of
   (`text`/`offset`/`duration`/`confidence`), with `segments[].speaker` as an integer
   diarization label → group consecutive words by speaker into one line per turn,
   discard all timing **from the emitted line** while retaining each line's source
   offset out-of-band, since §4.2 needs it to align items to chunks. Speaker labels are
   renamed to `S1…Sn` in first-appearance order unless a real name/role is supplied;
   `UNK` only where no label exists (§2).
2. **Translate en → zh-TW** with **TranslateGemma-27B** (`google/translategemma-27b-it`;
   Traditional Chinese is an explicitly supported target in its tech report). The
   transcript *and* the per-item summaries are translated, so everything downstream is
   zh-TW. Speaker labels are not translated. Scale: ~28k source tokens per meeting —
   ~5.7M for the Phase-1 pilot, ~35M for the full 1,250-meeting corpus.
3. **Compose the whole-meeting summary** with the teacher (§4), generating **in zh-TW**,
   from **the translated per-item summaries, in meeting order, with their `type` labels**.
   The task is composition, not summarization from scratch: human-authored content
   becomes one connected <1,000-token narrative in §3's form. Since no whole-meeting
   minutes document exists (see above), the meeting-level *structure* is the item
   sequence and item types — which is a genuine human-chosen ordering, but a far weaker
   signal than a drafted narrative would be. **Record this as the weakest link in the
   reference chain** and treat §8 risk 1 as correspondingly worse. Generating in the
   target language (rather than composing in en and translating the result) keeps the
   target's register that of a generative model working in zh-TW; see §4.3 for the
   alternative ordering.
4. **Human validation** of every composed summary (§4) before it enters the corpus.

**Granularity (normative): segment minutes are intermediate, not the training
target.** The ~5.5 per-item minutes per meeting are input material for stage 3, not
the artifact. The training target is the **whole-meeting** summary only, per §3.

**Provenance, stated precisely**: the reference summary is
human-selected → machine-translated → machine-composed. *What* is worth recording
comes from professional city clerks, which is the hard part and is human; the *language*
is TranslateGemma's and the *composition* is Qwen's. So this is not a fully synthetic
reference — but no human has ever read the zh-TW text that ends up as the training
target, which is what §4's human-validation stage exists to fix, and why it is
non-optional. Token counts also shift under translation; the ~28k figure is English,
and the zh-TW count under MiniCPM5's tokenizer must be measured, not assumed (§7, §8).

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
  summary from the translated per-item minutes (§2.2 stage 3) and produces the per-step
  distillation targets (§4.2).
- **Translation model: TranslateGemma-27B** (`google/translategemma-27b-it`) — offline
  only, corpus construction stage 2 (§2.2). A dedicated translation model rather than
  the teacher, on the assumption it produces better zh-TW; gated on a sample (§4.3)
  before committing the full ~35M-token run.
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

Chunk size ~2,500 tokens follows from the budget, not preference. **Chunking is
token-based over the whole transcript, not segment-aligned.** MeetingBank's summarized
segments cover only ~51% of a meeting (§2.2), so aligning chunks to them would leave
the other half of the transcript unread — and at deploy time the agent sees everything,
with no segment annotations at all. Chunk boundaries therefore fall at token offsets
(snapped to the nearest line boundary, since §2 lines are atomic), and item minutes
are attached to whichever chunks overlap them during supervision (§4.2).

Step count per meeting: ~28.4k ÷ 2.5k ≈ **11 reading steps + 1 synthesis ≈ 12 calls**,
pending the zh-TW token measurement (§7).

**4k is a budget choice, not a model limit — and it is falsifiable.** The binding
constraint is CPU-only KV cache and prefill latency on §6's hardware, not MiniCPM5's
supported context. The prior project measured CPU RSS scaling hard with context (an 8B
model: 1.6 GB at ctx=2048 → 4.3 GB at ctx=65536), which is where the conservative 4k
comes from. But step count falls roughly linearly as context grows, and **fewer steps
means less error accumulation across the memory chain** — the dominant failure mode of
this whole architecture. So 8k must be measured against 4k on the real device (§9
Phase 0b) before 4k is treated as settled. If 8k fits the latency and RSS envelope, it
is probably the better design: ~6 steps instead of ~12.

### 4.2 Training-data construction (normative)

Per-step supervision has to be constructed, since the corpus has no per-step targets.
The key asset is that **MeetingBank's item minutes are professionally authored and
already aligned to transcript spans** — the mapping from "this stretch of transcript"
to "what mattered in it" is human-made, not model-invented, which is a stronger
starting point than the prior project had (its teacher invented gold edits from
whole-transcript foresight alone).

Per meeting, walking chunks in order:

1. **Reading steps, chunks overlapping a summarized segment (~5 of ~11).** Input is
   (memory after step *i*−1, chunk *i*). Gold edit lines are derived by the teacher
   from the translated minute(s) of the overlapping segment — a narrow, grounded
   conversion task ("express this minute as ADD/ARC lines against the current memory"),
   not open-ended summarization. The teacher also sees later minutes, and that foresight
   is used *only* to emit `DROP` for points a later segment supersedes.
2. **Reading steps with no overlapping item (~4 of ~11).** These are the **~43%** of the
   transcript MeetingBank's annotators filtered out (measured, §2.2). **Do not
   blanket-`NOP` them.** The filter was `≥60 s` duration and `≥10 words` of summary —
   mechanical thresholds, so a 45-second exchange that settles something real was
   excluded for being short, not for being unimportant. Blanket-`NOP` would actively
   train the model to ignore substantive content. Since no whole-meeting minutes document
   exists to classify against (§2.2), the teacher instead judges each uncovered span
   against **the neighbouring items' summaries and the meeting's item list**: if the span
   is continuous with an adjacent item's business, emit the corresponding edit lines; if
   it is self-contained procedure, `NOP`. Filler (roll call, motions to adjourn,
   procedural boilerplate) genuinely resolves to `NOP` and *should* — "recognize filler
   and record nothing" is a skill the agent needs — but that verdict is derived rather
   than assumed. **This classification is weaker than the original design intended and
   its error rate is unmeasured**; sample and hand-check it during the Phase-1 gate.
3. **`ARC` supervision — degraded, and flagged as such.** The arc slot is the design's
   differentiator (§4.1) and cannot be supervised from item fragments, which have no
   through-line by construction. The intended grounding was the full minutes document,
   which **is not distributed** (§2.2). The available substitute is the ordered list of
   item summaries with their `type` labels: the teacher produces the arc state as it
   should stand after each step, given the items concluded up to that point. That is a
   real human-chosen ordering, but it is a list, not a narrative, so the arc's *prose
   through-line* is model-invented even though its *content* is grounded.
   **Consequence (normative): the ARC slot must be treated as an experiment, not an
   assumption.** Phase 2 must run the ablation — agent with `ARC` versus agent with
   `POINTS` only — and if `ARC` does not earn its context budget against that arm, drop
   the slot rather than shipping an ungrounded one. Record the result either way.
4. **Synthesis step.** Input is the final memory; target is the human-validated
   whole-meeting summary (§2.2 stage 3–4).

Both step types train the same model (§4). Supervision volume at full corpus: ~1,000
train meetings × ~12 steps ≈ **~12k training steps**, of which ~1.0k are synthesis; the
Phase-1 pilot yields ~1.9k steps from 160 train meetings. The edit/`NOP` split
among the ~11k curation steps is not fixed by item coverage — it falls out of
step 2's classification — but it must be **reported and monitored**: if `NOP` exceeds
~35% of curation targets, downsample or loss-weight it (§8). The synthesis skill still
sees only ~1/12th the data curation does (§8). **The ~12-step figure is itself a
projection from English token counts** and moves with the measured zh-TW ratio (§7, §9
Phase 0a); it is not a settled number.

**Validation.** Every gold edit sequence is replayed through the real harness before
use: ops must parse, `DROP` prefixes must match an existing point, and the resulting
memory must respect the caps. A sequence that fails replay is regenerated or dropped —
never half-applied into the corpus.

### 4.3 Corpus-construction gates (normative)

Two sample-first gates, both cheap relative to the full runs they precede:

**Translation gate** (before each translation tranche). Translate 20 meetings, then check:
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
| QAEval | **dropped** — needs Chinese QG+QA models, no supported zh path. Replaced by the faithfulness judge below, not left unmeasured. |

### 5.1 Faithfulness (normative)

A fluent summary that inverts a decision is the failure that matters most, and it is
the one ROUGE cannot see. It is measured, by two means:

- **Third-family LLM judge.** The contamination rule constrains *which* model, not
  whether to use one: the judge must be neither **Qwen-family** (authored the reference
  summaries, §2.2) nor **Gemma-family** (translated all corpus text). Any third family —
  Llama, Mistral, DeepSeek locally, or an API model — is uncontaminated by this
  pipeline and is permitted. Per claim in the summary: SUPPORTED / CONTRADICTED /
  UNSUPPORTED against retrieved transcript spans.
- **Human review on a 30-meeting slice.** Given that judge noise runs ±0.4–0.5 on this
  kind of scale, small-n human evaluation is competitive with a large automated run,
  and it is the only check not downstream of some model in this pipeline.

**Inversions are reported as a count, not folded into an average** — a single inverted
decision is a product defect, not a fractional score penalty.

### 5.2 Baseline and ship gates (normative)

**No result means anything without a baseline.** The agent architecture must earn its
complexity against a strictly simpler system using the same model and the same token
budget:

**Baseline — map-reduce, no learned memory.** Same MiniCPM5-1B, same ~2.5k chunking:
summarize each chunk independently, concatenate the chunk summaries, one final compress
pass to §3's form. No state carried across steps, no training beyond what the same
fine-tune provides. This is deliberately a *fair* opponent — same model, same chunk
size, same output contract — because a strawman baseline makes the gates meaningless.

**Ship gates**, all measured on the same meetings with the same metrics:

| gate | criterion |
|---|---|
| G1 revision | passes the revision probe below |
| G2 faithfulness | inversions ≤ baseline, and not worse than baseline on §5.1's judge |
| G3 quality | beats baseline on ROUGE/BERTScore by more than run-to-run noise |
| G4 budget | fits §7's measured envelope on §6's hardware |

**Ship the agent only if G1–G4 clear. Otherwise ship the map-reduce baseline** and
record agentic-memory-at-1B as a measured negative result — that is a legitimate
outcome, not a failure to report.

**Revision probe (G1).** Aggregate scores cannot show the one thing external memory
buys that map-reduce structurally cannot do: letting a later chunk overturn an earlier
conclusion. Probe it directly with hand-built transcripts containing a planted decision
that reverses late in the meeting (approved → rescinded), plus a distractor topic that
must not appear. Pass = final summary states the *later* decision, does not state the
earlier one as current, and omits the distractor. Cheap, synthetic, and diagnostic in a
way corpus averages are not — run it before any corpus-scale evaluation (§9).

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
| calls per meeting | ~12 (≈11 reading + 1 synthesis) | 28.4k transcript ÷ 2.5k chunk (§4.1) |
| prefill per meeting | ~39k tokens | ~3.5k × 11 reading steps + ~0.9k synthesis |
| decode per meeting | ~2.7k tokens | ~150 × 11 edit-line steps + <1,000 prose |
| wall-clock per meeting | **must measure** | the user-facing number on a Dimensity 900 CPU; dominated by prefill |
| peak RSS | **must measure** | 8 GB device shared with OS and app; single ~1B Q8 model resident |

The design's efficiency argument is that memory is capped, so per-step context is
**constant-size regardless of meeting length** — a 3-hour meeting costs more steps, not
bigger steps, and never exceeds the context window.

**Kill criterion.** If measured wall-clock per meeting exceeds ~20 minutes on §6's
hardware, the design is not shippable as specified and must change shape (larger
context and fewer steps, a smaller quant, or moving synthesis off-device) before any
corpus is built. This is Phase 0b in §9 precisely because it needs no corpus at all.

---

## 8. Known risks

Design is specified end to end; what remains are risks to measure, not gaps to fill.
Each is now attached to the phase that tests it (§9).

1. **The corpus teaches a different task than the goal** (tested: Phase 3). The target
   is a connected narrative of an arbitrary meeting; the supervision is US city-council
   procedure — motions, votes, ordinance IDs — which MeetingBank's own paper describes
   as largely extractive (Coverage 0.7–0.9, high Density, "include discussion points
   verbatim rather than performing abstraction"). Two mismatches stack: **discourse
   form** (procedural record ≠ narrative) and **meeting structure** (roll-call /
   readings / public-comment / votes is not what a Taiwanese work meeting looks like).
   A model trained on 1,250 council meetings may hallucinate motions and votes into a
   project review. The mitigation originally credited here — composing from a full
   minutes document — **is no longer available** (§2.2: no PDF is distributed), so this
   risk is *worse* than v0.6 assessed: the reference's meeting-level structure is now an
   item list rather than a human-drafted narrative. Only the in-domain eval slice (§9
   Phase 3) can detect the second mismatch.
2. **Two skills, one 1B model, unequal data** (tested: Phase 2). §4 merges memory
   curation and prose synthesis into one model, and §4.2 gives synthesis ~1/12th the
   steps curation gets. This deliberately contradicts the prior project, which used a
   *separate* synthesizer and carried an explicit exclusion against folding synthesis
   into the note-taker. Merging is justified by the single-model, single-device
   constraint (§6), but it is an untested reversal and the first thing to check if
   prose quality disappoints.
3. **`NOP` share** (tested: Phase 1). §4.2 now derives the edit/`NOP` verdict rather
   than assuming it, but the share is still an open quantity. The prior project shipped
   a NOP-collapse guard because over-NOPing was a *measured* failure mode. Monitor; if
   `NOP` exceeds ~35% of curation targets, downsample or loss-weight.
4. **Translationese in every training target** (tested: Phase 1 gate, §4.3). The gate
   can reject an obviously-bad translation; a subtly foreign register would pass review
   and propagate. There is no native zh-TW reference anywhere in the corpus to catch it —
   the in-domain slice (Phase 3) is the only independent check.
5. **Train/deploy ASR gap** (tested: Phase 3). Training text is clean and professionally
   transcribed; deployment text comes from on-device ASR + diarization with errors the
   training data has none of. The prior project hit this hard (verifier: 8/9 on clean
   zh, 0/11 on real noisy zh). MeetingBank's audio is English, so **nothing in the
   corpus can measure it** — this is the specific gap the in-domain slice exists to
   close.
6. **Projections, not measurements** (tested: Phase 0a/0b). Every figure in §7 and the
   ~12-calls / ~2.5k-chunk assumptions in §4.1 rest on English token counts. The zh-TW
   measurement under MiniCPM5's tokenizer moves the step count, the `NOP` ratio, and all
   of §7 together.
7. **The verifier was dropped without evidence** (tested: before Phase 1 — it needs no
   corpus). §4 replaces the prior project's 3-model pipeline with a single on-device
   model, and that reversal is not merely untested: the prior project **measured the
   single-model configuration on this exact base model and this exact device and
   rejected it** — *"the model alone measures 4/20 inversions; the verifier gate is what
   the device needs to reach ~0."* §5.2's G2 gate demands `inversions ≤ baseline`, which
   is precisely the bar that needed the verifier. If a deterministic guard (§4.1's
   harness-side contradiction check) cannot replace it, §7's budget must carry a second
   model and **the Phase-0 latency measurement is measuring the wrong system.** Settle
   this with the G1 revision probe against the existing fine-tuned checkpoint before
   booking device time.
8. **`ARC` supervision is degraded** (tested: Phase 2 ablation, §4.2 step 3). The full
   minutes document that was to ground the arc slot is not distributed (§2.2). The slot
   is the design's stated differentiator and now carries a weaker signal than intended,
   so it must earn its context budget against a `POINTS`-only arm or be dropped.

---

## 9. Execution plan (normative)

Ordered cheapest-first, each phase gated. **No phase starts until the previous one
passes** — the point is to spend the ~35M-token translation run and the fine-tune only
after the assumptions they rest on have survived contact with the device.

### Phase 0a — configuration check (no corpus, no device, no training)

Runs *before* device time is booked, because it decides **what system Phase 0b should
measure**. Two questions, both answerable from artifacts already on disk:

1. **Is the verifier necessary (risk 7)?** Run §5.2's G1 revision probe against the
   existing fine-tuned MiniCPM5-1B checkpoint in three arms: model alone; model plus the
   prior project's 350M verifier; model plus §4.1's deterministic contradiction guard.
   If the guard suffices, §4's single-model design stands on evidence. If it does not,
   §7's call budget gains a second model *before* the latency gate is measured.
2. **What is the en→zh token ratio?** Translate a handful of meetings with any available
   model and measure both sides under MiniCPM5's tokenizer. This is a *units*
   experiment, not a quality one, and must not wait for TranslateGemma. Every figure in
   §7 and the step count in §4.1 scale with it.

**Gate:** the shippable configuration is known, and the step count rests on a measured
zh-TW token ratio rather than an English one.

### Phase 0b — device reality check (no corpus required)

Stock MiniCPM5-1B on the actual Reno 7 (§6), in the configuration Phase 0a settled.
Measure prefill and decode throughput **at realistic KV depth, not just at depth 0**,
peak RSS, and per-meeting wall-clock for the §7 projection; measure **4k vs 8k context**
to settle §4.1's chunk size on evidence.

**Quant is a variable here, not a constant.** §4 names Q8 by fiat, but the Reno 7's
Cortex-A78 is Armv8.2-A — it has `dotprod` and **no `i8mm`** — so the fastest int8 GEMM
path is unavailable and Q4 variants may win on the metric that actually gates. Sweep
Q8_0 / Q4_K_M / Q4_0 and let the measurement choose.

**Gate:** per-meeting wall-clock ≤ ~20 min at the chosen context, within RSS budget.
Fail → change shape (larger context/fewer steps, smaller quant, synthesis off-device)
before proceeding.

### Phase 1 — 200-meeting pilot corpus

Run §2.2's construction on **200 meetings, not 1,250**: import, translate (~5.7M tokens
rather than ~35M), compose, human-validate. Apply §4.3's gates. Build per-step
supervision (§4.2) and report the edit/`NOP` split.

**Gate:** translation gates pass (line-count integrity is hard pass/fail); human
validators accept the composed summaries; `NOP` share within bounds.

### Phase 2 — pilot fine-tune, baseline, and probe

Fine-tune on the pilot corpus. Build the map-reduce baseline (§5.2). Run the revision
probe (G1) and the full gate set G1–G4 against the baseline.

**Gate:** G1 passes and the agent beats baseline (G2/G3) at Phase-1 scale. Fail → either
the architecture doesn't earn its complexity (ship the baseline, record the negative
result) or the deficit is diagnosably data-volume-bound, which is the only justification
for Phase 4's spend.

### Phase 3 — in-domain zh-TW reality check

~20 real Taiwanese meetings recorded through the **VoxSum Android app**, held out,
**eval only — never training**. This single slice closes three gaps nothing else can:
the ASR train/deploy distribution gap (risk 5), the domain and discourse mismatch
(risk 1), and the absence of any native-zh-TW reference (risk 4). Source from whatever
is obtainable and genuinely multi-speaker — 立法院 committee sessions are the known-good
fallback, real work meetings are better if consent allows.

**Gate:** no catastrophic degradation versus the clean-text corpus. A large drop here
means the model learned council-procedure text, not meeting summarization, and no amount
of additional MeetingBank data fixes that.

### Phase 4 — full corpus

Only now translate the remaining ~1,050 meetings, retrain, re-run G1–G4 and Phase 3's
slice. Full spend is justified only by a Phase-2 result that was gated on volume.
