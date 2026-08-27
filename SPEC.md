# SPEC — Agentic meeting summarizer (MiniCPM5-1B + external memory, zh-TW)

**Version:** 0.9 · **Status:** design + execution plan complete; Phase 0a fully closed,
Phase 0b measured on the actual reference device — §9 phases the remaining work
cheapest-first with gates; §8 attaches each risk to the phase that tests it

**v0.9 changes** — Phase 0a item 2 (the en→zh-TW token ratio) is now measured, closing
the last open Phase 0a question: **1.215** zh-TW tokens per en token under MiniCPM5's
own tokenizer (5 real MeetingBank segments, 4,010 en → 4,874 zh-TW tokens, translated
by a local Qwen3.8-27B instance per Phase 0a's "any available model, must not wait for
TranslateGemma" rule; see §9 Phase 0a for methodology). This revises the step count
from the English-derived ~11 reading steps to **~14** (§4.1, §7, §8 risk 6) — the
Phase 0b wall-clock/RSS figures were measured per-step and so are NOT invalidated, but
the **11-step reading-phase total (~12.9 min)** was a trapezoidal projection over the
old step count and needs re-projection at ~14 steps; flagged as a new open item rather
than silently rescaled (§9 Phase 0b).

**v0.8 changes** — Phase 0b ran on the actual Reno 7 (device access via a proxy host,
2026-08-21), not projected. Headline result: **the core-mask choice matters far more
than the quant choice**, and the prior project's big-cores-only (`0xC0`) serving
convention **fails** the ~20-minute gate (~22.5 min projected for the reading phase
alone) while an all-8-cores config (`0xFF`) **passes** it comfortably (~12.9 min) — see
§9 Phase 0b for the full measured table. Peak RSS measured at 829 MiB–1.34 GiB across
quants, well within budget. §7's "must measure" rows are updated with these figures.

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

Step count per meeting: measured en→zh-TW token ratio **1.215** (§9 Phase 0a,
2026-08-21) → ~28.4k en × 1.215 ≈ 34.5k zh-TW tokens ÷ 2.5k ≈ **14 reading steps + 1
synthesis ≈ 15 calls** (up from the English-derived ~11 reading / ~12 calls estimate).

**4k is a budget choice, not a model limit — and it is falsifiable.** The binding
constraint is CPU-only KV cache and prefill latency on §6's hardware, not MiniCPM5's
supported context. The prior project measured CPU RSS scaling hard with context (an 8B
model: 1.6 GB at ctx=2048 → 4.3 GB at ctx=65536), which is where the conservative 4k
comes from. But step count falls roughly linearly as context grows, and **fewer steps
means less error accumulation across the memory chain** — the dominant failure mode of
this whole architecture. So 8k must be measured against 4k on the real device (§9
Phase 0b) before 4k is treated as settled. If 8k fits the latency and RSS envelope, it
is probably the better design: ~7 steps instead of ~15.

### 4.2 Training-data construction (normative)

Per-step supervision has to be constructed, since the corpus has no per-step targets.
The key asset is that **MeetingBank's item minutes are professionally authored and
already aligned to transcript spans** — the mapping from "this stretch of transcript"
to "what mattered in it" is human-made, not model-invented, which is a stronger
starting point than the prior project had (its teacher invented gold edits from
whole-transcript foresight alone).

Per meeting, walking chunks in order:

1. **Reading steps, chunks overlapping a summarized segment (~8 of ~14).** Input is
   (memory after step *i*−1, chunk *i*). Gold edit lines are derived by the teacher
   from the translated minute(s) of the overlapping segment — a narrow, grounded
   conversion task ("express this minute as ADD/ARC lines against the current memory"),
   not open-ended summarization. The teacher also sees later minutes, and that foresight
   is used *only* to emit `DROP` for points a later segment supersedes.
2. **Reading steps with no overlapping item (~6 of ~14).** These are the **~43%** of the
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
train meetings × ~15 steps ≈ **~15k training steps**, of which ~1.0k are synthesis; the
Phase-1 pilot yields ~2.4k steps from 160 train meetings. The edit/`NOP` split
among the ~14k curation steps is not fixed by item coverage — it falls out of
step 2's classification — but it must be **reported and monitored**: if `NOP` exceeds
~35% of curation targets, downsample or loss-weight it (§8). The synthesis skill still
sees only ~1/14th the data curation does (§8). **The ~15-step figure rests on the
measured en→zh-TW token ratio (§9 Phase 0a, 2026-08-21)**, not an English projection —
it is a settled number pending the Phase 1 pilot corpus's own measured token counts.

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
| G3 quality | beats baseline on ROUGE/BERTScore by more than run-to-run noise. **Coverage/Density are NOT gated** — see below |
| G4 budget | fits §7's measured envelope on §6's hardware |

**Ship the agent only if G1–G4 clear. Otherwise ship the map-reduce baseline** and
record agentic-memory-at-1B as a measured negative result — that is a legitimate
outcome, not a failure to report.

**Coverage and Density are diagnostics, never gates** (normative, clarified
2026-08-27). §5's metric table already classes them as "token-overlap diagnostics" and
G3 above is already defined over ROUGE/BERTScore, but the implementation gated them
anyway and the consequence was not cosmetic. Coverage is the fraction of the summary
copied verbatim from the source; Density is mean *squared* verbatim-fragment length.
Both measure **extractiveness**. Requiring `agent > baseline` on them demands the agent
copy more verbatim than map-reduce — the direct opposite of §3's flowing abstractive
zh-TW prose, and unreachable by construction rather than through any deficiency of the
model. Map-reduce is structurally extractive: it summarises windows and reduces, staying
close to source wording. Measured at n=20 on 2026-08-27: coverage 0.982 agent vs 0.993
baseline (a gap at a near-saturated ceiling), density 3.26 vs 4.05. With those two
gated, "ship the agent" was unreachable no matter how good the agent became.

They remain computed, reported and compared per meeting. Read them as *shape*
descriptors — a large Density gap says the two systems copy differently, which is
expected and is the point of the design — not as quality scores with a preferred
direction. Pinned by `metrics.stats.G3_GATED_METRICS`.

**Revision probe (G1).** Aggregate scores cannot show the one thing external memory
buys that map-reduce structurally cannot do: letting a later chunk overturn an earlier
conclusion. Probe it directly with hand-built transcripts containing a planted decision
that reverses late in the meeting (approved → rescinded), plus a distractor topic that
must not appear. Pass = final summary states the *later* decision, does not state the
earlier one as current, and omits the distractor. Cheap, synthetic, and diagnostic in a
way corpus averages are not — run it before any corpus-scale evaluation (§9).

**The planted reversal must land in a LATER CHUNK than the decision** (normative), not
merely later in the transcript. The mechanism under test is precisely that external
memory carries a conclusion across a step boundary so a later step can overturn it; a
transcript short enough to fit in one chunk exercises none of that — no memory crosses
a step, `DROP` is never used, and the agent arm degenerates into a one-shot summariser
scored as if it were the agent. **This shipped as a real defect**: the original probe
transcripts were ~120 tokens against §4.1's 2,500-token budget, so every G1 result
reported before 2026-08-27 measured the wrong mechanism. Probe transcripts must
therefore be long enough to chunk, and the planted topic must be the meeting's dominant
business — an arc buried as a fraction of a percent of a long transcript loses its
memory slots on salience and measures salience, not revision (also measured, same date).

**The distractor must be genuinely non-decision-bearing** (§4.2's "self-contained
procedure" bucket — a recess announcement, not a second real decision on an unrelated
topic). §4.2 normatively trains the teacher to emit edit lines for *every* official
item overlapping a chunk with no relevance filter, so a distractor phrased as a closed
decision tests a bar no model trained per §4.2 could clear without contradicting that
same training — confirmed directly (2026-08) by running an early, decision-shaped-
distractor probe against the unfine-tuned teacher itself, which reproduced the same
"failure" as the fine-tuned student.

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

Derived from §4.1's protocol. **Step count and token figures now rest on the measured
en→zh-TW token ratio** (1.215, §9 Phase 0a, 2026-08-21) rather than an English
projection; **wall-clock and peak RSS are measured** on the actual Reno 7 (§9 Phase 0b,
2026-08-21), at the all-8-cores configuration the measurement itself selected — but the
reading-phase wall-clock TOTAL below was integrated over the old 11-step count and is
now an open re-projection item (§9 Phase 0b), not yet recomputed at the corrected ~14.

| quantity | value | basis |
|---|---|---|
| calls per meeting | ~15 (≈14 reading + 1 synthesis) | 28.4k en × 1.215 measured ratio ≈ 34.5k zh-TW tokens ÷ 2.5k chunk (§4.1, §9 Phase 0a) |
| prefill per meeting | ~50k tokens | ~3.5k × 14 reading steps + ~0.9k synthesis |
| decode per meeting | ~3.1k tokens | ~150 × 14 edit-line steps + <1,000 prose |
| wall-clock, reading phase (all cores, `-C 0xFF`) | **~12.9 min, measured at 11 steps — OPEN: needs re-projection at ~14** | 11-step trapezoidal projection from measured per-step depth scaling (§9 Phase 0b); computed before the token-ratio measurement above, so the total (not the per-step figures it is built from) is stale by ~+27% steps |
| wall-clock, reading phase (big-cores-only, `-C 0xC0`) | **~22.5 min, measured at 11 steps — FAILS the gate, and only gets worse at ~14 steps** | same method and same re-projection caveat; this was the prior project's serving convention and must not be reused here regardless |
| peak RSS, `--no-mmap`, 4k ctx | **829 MiB (Q4_0) – 1.34 GiB (Q8_0), measured** | one real 2,500-token completion, `VmHWM` + `smaps_rollup` `Pss` (§9 Phase 0b) — per-completion, not step-count-dependent |

The design's efficiency argument is that memory is capped, so per-step context is
**constant-size regardless of meeting length** — a 3-hour meeting costs more steps, not
bigger steps, and never exceeds the context window.

**Kill criterion — cleared, conditionally.** Measured wall-clock per meeting is ~12.9 min
at the all-cores configuration, under the ~20-minute ceiling; the design is shippable as
specified **only if the on-device serving path is built to use all 8 cores**, not the
big-cores-only convention the prior project used, which independently measured **over**
the ceiling on this same hardware. This is Phase 0b in §9, completed without a corpus.

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
   curation and prose synthesis into one model, and §4.2 gives synthesis ~1/14th the
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
6. **Projections, not measurements** (tested: Phase 0a/0b; **item 2 now measured**).
   The en→zh-TW token ratio (1.215, §9 Phase 0a, 2026-08-21) has replaced the
   English-derived ~12-calls / ~2.5k-chunk assumptions in §4.1 with a measured ~15-call
   figure; §7 and §4.2's volume/`NOP` arithmetic are updated accordingly. **Still open:**
   the ratio was measured on 5 real MeetingBank segments (a "handful," per Phase 0a's own
   scope), not a large sample, and Phase 0b's reading-phase wall-clock TOTAL (~12.9 min)
   was integrated at the old 11-step count and needs re-projection at ~14 before the §7
   gate figure is fully trustworthy.
7. **The verifier was dropped without evidence** (probed: 2026-08-21, pre-Phase-1;
   **partial result, not a settlement — see caveats**). §4 replaces the prior project's
   3-model pipeline with a single on-device model, and that reversal is not merely
   untested: the prior project **measured the single-model configuration on this exact
   base model and this exact device and rejected it** — *"the model alone measures 4/20
   inversions; the verifier gate is what the device needs to reach ~0."*
   - **What was run.** The prior checkpoint (`Luigi/minicpm5-1b-cursor`, its own v1
     protocol — no v2-trained checkpoint exists yet) was served on-GPU and run three
     ways: alone, with the prior project's own `enforce_decision_chain` (a deterministic
     guard structurally analogous to this spec's `contradiction()` guard), and with the
     `Luigi/granite-4.0-350m-verifier` in-stream. On the synthetic G1 screen, all three
     passed 1/1 — the single hand-built planted-reversal case is not discriminating. On
     3 real zh-TW ASR meetings (`asr-transcripts-2026-08-16`) scored with the prior
     project's deterministic inversion detector, **model alone measured 0/3 inversions**,
     unchanged by `enforce_decision_chain`.
   - **A finding that narrows the question, not one that closes it.** Even the prior
     project's own *verifier-enabled* historical run on one of these meetings logged
     `INVERSION polarity-bearing bullets=0: inverted=0` — the verifier's catch there was
     an unsupported-claim drop, not a polarity flip. So `detect_inversions()` (and this
     spec's `contradiction()` guard, built on the same subject+polarity mechanism) target
     a narrower failure mode than the verifier's full scope. The 0/3 result is real
     evidence against needing a verifier specifically for §5.2's **G2 inversions gate**;
     it says nothing about the broader unsupported-claim faithfulness question, which is
     §5.1's LLM-judge gate's job, not G2's.
   - **Caveats, explicit.** n=3 real meetings, not the historical T1 tier's n=20;
     `send_thinking_kwarg` had to be set `True` for this checkpoint's chat template to
     avoid a reasoning-then-empty-content failure — a serving-configuration difference
     from whatever produced the historical 4/20 figure, so the two numbers are not a
     clean apples-to-apples comparison; and the checkpoint is v1-protocol, not v2, so
     this is evidence about the base model's underlying capability, not a validation of
     the actual `guards.contradiction()` code in this repo.
   - **Consequently:** proceed on the assumption that a deterministic guard suffices for
     G2 specifically, but do **not** close this risk — re-run once a v2-trained
     checkpoint exists (Phase 2) at the full T1-scale n, and keep §5.1's judge as the
     independent check on the broader faithfulness question the inversion detector
     cannot see.
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
2. **What is the en→zh token ratio? MEASURED (2026-08-21): 1.215.** Translated 5 real
   `huuuyeah/meetingbank` English segments (1,500–6,000 chars each, real MeetingBank
   content — the stripped HF mirror is fine for this units-only measurement even though
   §2.2 requires the authoritative Zenodo release for the actual corpus) with the
   already-running local Qwen3.8-27B instance (`local:8082`, temperature 0, a completion
   instruction forbidding summarization/omission, `max_tokens=6000` to clear its
   reasoning budget — the first attempt's empty-content stall, noted below, was exactly
   this token budget being too tight) and measured both sides under
   `openbmb/MiniCPM5-1B`'s real tokenizer (`arcsum.tokens.hf_token_len`): **4,010 en
   tokens → 4,874 zh-TW tokens**, all 5 calls finishing clean (`finish_reason=stop`, no
   truncation), per-segment ratios tightly banded (1.12–1.30). zh-TW is **more**
   tokens than the English source under MiniCPM5's tokenizer, not fewer — every figure
   in §7 and the step count in §4.1 scale up accordingly (~11 → ~14 reading steps).
   **Not yet done:** this is 5 segments, a "handful" as scoped, not a large sample, and
   translation used Qwen (the eventual composition teacher) as a stand-in, never
   TranslateGemma — re-measure once TranslateGemma-27B (or the Phase 1 pilot corpus
   itself) is available, and treat 1.215 as a solid but non-final estimate until then.

**Gate: CLEARED.** The shippable configuration is known (Phase 0b measured the
guard-only single-model config); the step count now rests on a measured zh-TW token
ratio (1.215) rather than an English one.

### Phase 0b — device reality check (measured 2026-08-21, on the actual Reno 7)

**Measured, not projected.** Stock MiniCPM5-1B (Q8_0, Q4_0, and the prior project's
Q4_K_M fine-tune as a realistic-decode-length stand-in) ran directly on an OPPO CPH2371
(confirmed `ro.product.model=CPH2371`, SoC `mt6877`/Dimensity 900, `asimddp` present,
no `i8mm` — matching §6 exactly) via a cross-compiled arm64-v8a `llama-bench`/
`llama-server` (NDK r27c, `-march=armv8.2-a+dotprod`, no SVE/i8mm). cpu0–5 confirmed
2.0 GHz (LITTLE, A55), cpu6–7 confirmed 2.4 GHz (big, A78), matching the device's known
core layout.

**Headline finding: core-mask choice matters far more than quant choice, and the
`0xC0` big-cores-only convention the prior project's `serve_student.sh` used is the
WRONG choice here.** At depth 0, combined prefill+decode wall-clock for one step
(2,500 prefill + 150 decode) on **all 8 cores** beat **big-cores-only** by close to 2×:

| quant | big-only (2 threads, `0xC0`) | LITTLE-only (6 threads, `0x3F`) | all cores (8 threads, `0xFF`) |
|---|---|---|---|
| Q8_0 | 112.7 s | 89.6 s | **67.3 s** |
| Q4_0 | 102.6 s | 81.7 s | **58.3 s** |
| Q4_K_M (fine-tuned) | 116.7 s | 91.3 s | **66.4 s** |

Without `i8mm`, the fastest available int8 path gains nothing from the A78's higher
clock alone; a 2,500-token prefill is throughput-bound and the six A55s add real
parallel capacity even at a lower clock. **Quant choice barely matters by comparison**
— Q4_0 beats Q8_0 by only ~13–15%, far less than the ~40–68% swing from core-mask
choice alone. This falsifies §4's "Q8 by fiat" only weakly (Q4_0 is faster but not
dramatically), but it strongly falsifies the prior project's big-cores-only serving
convention for this specific device and workload.

**KV-depth scaling** (big-only, Q4_0, the stable reference curve): depth 0 → 101.6 s,
depth ≈1,446 (≈4k total ctx) → 143.8 s (**1.42×**), depth ≈5,542 (≈8k total ctx) →
270.9 s (**2.67×**). Applying the same relative scaling to the all-cores baseline and
taking a trapezoidal average over SPEC §4.1's ~11-step reading phase (depth ramping
~linearly from 0 toward ~4k):

| core mask | avg step (ramping to ~4k) | 11-step reading phase |
|---|---|---|
| big-only (`0xC0`) | ~122.7 s | **~22.5 min** — already over the ~20 min gate, before synthesis |
| all cores (`0xFF`) | ~70.4 s | **~12.9 min** — comfortably under, with headroom for synthesis |

**Peak RSS at 4k context, `--no-mmap` (the honest private-storage number, not
page-cache-backed)**, one real 2,500-token completion:

| quant | `VmHWM` | `smaps_rollup` `Pss` |
|---|---|---|
| Q8_0 | 1,341 MiB | 1,279 MiB |
| Q4_0 | 829 MiB | 813 MiB |
| Q4_K_M | 851 MiB | 835 MiB |

All three comfortably clear the device's 7.3 GB total RAM (though only ~200 MB was
free at idle when first checked — the device was running its normal OS load; a
dedicated-service deployment would need to budget against whatever headroom the target
app actually gets, not against total RAM).

**Gate verdict: PASS, all-cores config only — margin now needs re-checking.** ~12.9 min
projected for the reading phase at all 8 cores cleared the ~20 min ceiling with room
for a synthesis call, but that total was integrated over the OLD 11-step,
English-derived count; §9 Phase 0a's measured 1.215 en→zh ratio revises this to ~14
steps. A naive linear rescale (12.9 × 14/11 ≈ 16.4 min) would still clear the ceiling,
but this is an illustrative bound, not a re-measurement — the depth-scaling curve is
not perfectly linear (§9 Phase 0b's own table shows cost-per-step growing with depth),
so treat the gate as **provisionally still PASS, pending an actual re-projection**
before spending Phase 1/2 compute on the assumption it definitely holds. The
big-cores-only configuration **fails** the same gate regardless (~22.5 min for reading
alone at 11 steps, and only gets worse at 14). **Corollary for §9's later phases and
for any on-device serving script: use `-C 0xFF` (or no mask restriction), never
`0xC0`.**

**What this measurement does not yet settle** (explicitly deferred, not skipped):
- **4k vs 8k** — the depth-scaling data above answers "does 8k fit" (yes: ~8k context
  costs roughly 2.7× a single step's depth-0 time, well within a per-step budget), but
  not "is 8k the right choice" — that trades off against fewer steps and less error
  accumulation, a Phase-2 question per §4.1.
- **Sustained-run thermal drift** — this sweep ran for several minutes of near-continuous
  load; whether decode throughput degrades further over a full ~13-minute meeting from
  thermal throttling was not isolated from the depth-scaling effect already measured.
  A dedicated back-to-back same-config timing series is the way to isolate it.
- **The en→zh token ratio (Phase 0a item 2) — MEASURED 2026-08-21, see §9 Phase 0a.**
  The first attempt (this session) stalled on an empty-content response from the same
  local translation model; root cause confirmed as a reasoning-mode token budget too
  tight to leave room for an answer after the model's own reasoning trace, fixed by
  raising `max_tokens` well above the reasoning budget. Result: **1.215** zh-TW tokens
  per en token. **Consequence for the table above: the ~11-step reading-phase wall-clock
  total (~12.9 min) was integrated at the OLD English-derived step count and is now a
  stale total** — the per-step depth-scaling figures it is built from remain valid, but
  re-projecting over ~14 steps instead of 11 has not yet been done and is a genuinely
  open follow-up, not a rounding error to wave off before trusting the §7 gate figure.
- **Decode-only throughput** (isolated from prefill) — only the combined `-pg 2500,150`
  figure was measured; a `-n`-only sweep would sharpen the per-token decode-cost estimate
  used in the RSS request and in any future latency budget refinement.

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
