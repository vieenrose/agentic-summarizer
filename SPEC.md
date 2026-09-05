# SPEC — Agentic meeting summarizer (Qwen3.5-0.8B + external memory, zh-TW)

**Version:** 1.4 · **Status:** design + execution plan complete; Phase 0a fully closed,
Phase 0b measured on the actual reference device — §9 phases the remaining work
cheapest-first with gates; §8 attaches each risk to the phase that tests it

**v1.4 changes — the gate set defended against regression and never asked for VALUE.** v1.2
and v1.3 each hardened the gates after an instrument was found wrong, seven times in three
days. That was right, and it had a cumulative side effect nobody chose: **every gate became a
guard-rail, and a set made entirely of guard-rails converges on the most cautious model, not
the best one.** With G3 withheld (§5.2.4), *nothing left in the set asked whether the summary
is any good* — a bland, faithful, well-covered, stable, fast summary passes all of G1–G8.
Since §1's objective is a summary **a person reads**, that is a hole at the centre.

Seven rectifications, each attached to the defect it closes:

1. **§5.2.7 — G9 human utility, GATED.** The only instrument in the whole set that cannot be
   satisfied by saying less. Designed to be cheap enough to actually run (12 meetings, 2
   reviewers, forced-choice), because a gate that is too expensive to run is not a gate.
2. **§5.2.8 — G10 domain transfer, GATED.** Every gate ran on translated MeetingBank; the
   product reads real zh-TW ASR. A 17/20 → 7/20 regression on real ASR went undetected across
   three checkpoints because no gate looked there.
3. **§5.2.4 — G3 is RETIRED, not merely withheld.** Its references are composed by a **Qwen**
   teacher from **Gemma**-translated text. §5.1's contamination rule correctly bars judges of
   those families, but ROUGE and BERTScore are not judges — they compare directly against the
   contaminated reference. Repaired, G3 would measure *teacher imitation*.
4. **§5.2.9 — a threshold gate must state what it can detect.** Seed alone moves churn by 27
   points; a fixed threshold read off one training run is not evidence. A gate whose
   between-seed spread exceeds its margin to the threshold is now WITHHELD, never passed.
5. **§5.2.10 — the ship decision admits ROUTING.** v1.2's all-or-nothing choice forecloses the
   one design the measurements actually support. It also now states, for the first time, what
   the agent must beat the baseline on to earn its complexity.
6. **§5.2.11 — G1 is corpus-limited and says so.** MeetingBank has essentially no
   within-meeting reversals, so G1's evidence must be synthesized — and synthetic probes have
   been pattern-matched repeatedly. A gate whose evidence is fabricated is a memorisation
   test. G1 now requires an independent probe AND a control arm, and WITHHOLDS when it cannot
   discriminate.
7. **§4.1.1 — constants are labelled DERIVED or CONVENTION.** The memory caps are stated
   normatively and are not the binding constraint: cap-overflow refusals are **0.6%** of ops
   while **~22%** of output is repetition. The spec was precise about what does not bite.

**v1.2 changes — the MEASUREMENTS were wrong, and the gates now defend against how.** v1.1's
architecture is unchanged; what changes is what counts as evidence for a gate. Every item is
attached to the measurement that forced it (2026-09-03,
`runs/journal-synthesis-outcome.md`, `runs/grounding-refold.md`, `runs/v12-e3/RESULT.md`):

1. **§5.2.1 replication is part of the gate.** Retraining at a second seed moves churn by up
   to **27 percentage points** at fixed data — larger than any effect this project has
   attributed to a data change. Behavioural gates now require ≥2 seeds and are scored on the
   WORSE one; a paired sign test over meetings explicitly does NOT satisfy this, and one was
   quoted at p = 2.2e-07 for a comparison whose run-to-run term exceeded the effect.
2. **§5.2.2 G5 is gated jointly with the new G7 (churn).** Churned re-`ADD`s become points, so
   they inflate retention, inflate `recorded_points` and reduce `starved` — `v12` improved
   every one of those *because* it churned. A metric a known defect can improve must be gated
   with the detector for that defect.
3. **§5.2.3 a gate corpus must exercise the mechanism it gates.** `data/asr_eval_v1` has a
   median of 1 chunk against a 16-point working set, so the journal never fills and the
   v1.0 code path runs. A full v1.1-vs-v1.2 comparison was completed on it before this was
   noticed, and it produced plausible numbers throughout.
4. **§5.0 instrument validity.** Three measurement defects in one session, all flattering the
   model, none caught by a failing test: an "accepted false positive" that was really a bias
   scaling with output quality; a protocol change whose consumers were never re-validated; and
   a validation set left on a superseded prompt version, invalidating every best-epoch claim.
5. **§5.2 the probation verdict must be RE-DERIVED before it is acted on.** The 0/25 result
   that put the architecture on probation was scored against references that are 46.5%
   unreachable, using a grounding instrument that understated the agent's faithfulness by up
   to 15 points. Retirement is one-way; it needs a comparison that is not.
6. **§4.1 two protocol clarifications**: supervision must express a revision as `revise` (118
   genuine revisions were being taught as drop-then-add, the surface form of churn), and
   `synthesis_view()` collapses near-duplicates — with numerals and superseded entries exempt,
   and with the honest note that this did NOT reduce churn.
7. **§1 and §5.2 corrected**: both still named MiniCPM5-1B as the student and the baseline
   model. v1.0 changed it to Qwen3.5-0.8B.
8. **§5.2.3 the G3 reference set may not correlate with meeting length — currently it does,
   perfectly.** The 25 meetings carrying reachable references contain **zero** above
   `POINTS_CAP` chunks; the 15 excluded contain **10 of them**, because the reference builder
   skips whatever exceeds the teacher's context. So every agent-vs-baseline number to date was
   measured in the one regime where the journal never fills and the agent runs the v1.0 code
   path. A route to references for long meetings (the corpus's own per-item gold minutes,
   reachability-filtered) is specified; hierarchical teacher composition stays forbidden.
9. **§7 the journal's G4 cost is measured and bounded**: +250 tokens / +4.2 s worst case on
   one call per meeting, against a ~36 s margin. Reading-step prefill is unchanged, so §7's
   constant-size argument survives intact.

**v1.1 changes — the memory splits into a working set and a journal, because v1.0 LOST to
its own baseline.** All four changes are normative and each is attached to the measurement
that forced it (2026-09-03, `runs/PROJECT-REVIEW.md`):

1. **§4.1 memory: one bounded slot → WORKING SET (unchanged, ≤600 tok) + JOURNAL
   (append-only, model-invisible).** v1.0 shared 480 tokens across every chunk, so its
   compression ratio grew with meeting length — 21:1 at 4 chunks, **193:1 at 37** — against
   map-reduce's constant 10:1. Measured: **48–80% of the points the model correctly
   recorded were evicted before the end.**
2. **§4.1 `SYNTHESIZE` reads the JOURNAL, not the survivor set.** v1.0 asked the model to
   write a rich summary from an impoverished input, which is the pressure that produces
   invention; the corpus then taught it, with **39.9% of synthesis targets asserting things
   absent from their own memory**.
3. **§4.1 addressing: text prefix → integer id, and a new atomic `revise` op.** Prefix
   addressing required the model to reproduce its own earlier phrasing; the resulting
   `DROP` + near-identical `ADD` churn ran at **28.2% of steps** on real ASR.
4. **§5.2 adds G5 (retention) and G6 (grounding)** — the two properties that were failing
   invisibly because no gate looked at them.

**The falsification this version must survive.** v1.0 lost to its own map-reduce baseline
**0 wins in 25 meetings** (−0.213 rouge1 against reachable references; 3.6 grounded
specifics per meeting against 8.5). v1.1 exists to fix the mechanism behind that. **If the
redesigned agent still does not beat the baseline, the agentic memory should be abandoned
and §5.2's standing "ship the baseline" decision made permanent.** The differentiator is
now curation plus recorded supersession, and it is worth exactly what it measures.

**v1.0 changes — the agent protocol becomes tool-calling, and the student changes with
it.** Both are normative and both rest on measurements taken 2026-08-29, recorded inline
at §4 and §4.1:

1. **§4.1 step grammar: edit lines → one `update_memory` tool call per chunk**, JSON
   arguments, single-turn. This makes the protocol expressible in a standard
   function-calling agent framework while preserving the properties that made v0 work —
   harness-owned memory, no conversation history across steps, one model call per chunk.
2. **§4 student: MiniCPM5-1B → Qwen3.5-0.8B.** The tool-call form costs 1.25x decode
   tokens; on the MiniCPM5-1B basis that projects 20.0–21.4 min against §7's 20-minute
   ceiling, and on a ~20% smaller model 16.2–17.3 min. The swap is what pays for the
   protocol.
3. **What was measured and REJECTED**: a conventional observe-the-result agent loop
   (2 model invocations per chunk, 1.89x prefill → 32–51 min), one native tool call per
   operation (2.72x decode), and the chat template's own `tools=` preamble (313–434
   tokens per step against a 266-token v0 system prompt). §4.1 records each with its
   number so none is retried on intuition.

**Everything in §5.2 is unchanged.** The gates are the goal; this revision changes how
the goal is pursued, never the bar. A v1.0 checkpoint is compared against a map-reduce
baseline built from the SAME model under §5.2's existing fairness rule, so no v0.9
number transfers — the baseline is re-run, not reused.

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

**Fine-tune Qwen3.5-0.8B (Q8, 4k context) to drive a lightweight ("smol") agent
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
and the zh-TW count under the STUDENT's tokenizer must be measured, not assumed (§7, §8).
(Historic note: this ratio was first measured under MiniCPM5-1B, the v0.9 student; the student
is Qwen3.5-0.8B since v1.0, whose 248k vocab tokenises zh-TW differently — 1.577 ch/tok
measured. Any figure derived from the old tokenizer is not comparable.)

---

## 3. Output (normative)

A single flowing **zh-TW prose** summary — no bullets, no sections, no anchors.
**< 1,000 tokens.** Structure and style within the prose are still open.

---

## 4. Architecture

- **Student / deployed model: Qwen3.5-0.8B, Q8, 4k context** (v1.0; was MiniCPM5-1B).
  Single on-device model (not the prior project's 3-model pipeline) — CPU-only per §6.
  It drives the agent protocol in §4.1, doing two jobs: per-step memory curation while
  reading, and the final prose synthesis.

  **Why the change is normative and not preference.** §4.1 v1.0 moves to a tool-call
  protocol, whose output is measurably more expensive per step (1.25x decode tokens,
  §4.1). Against §7's 20-minute ceiling the MiniCPM5-1B basis projects **20.0–21.4 min**
  under that protocol — at or over budget — while a ~20% smaller model projects
  **16.2–17.3 min** across every decode-share assumption. The model swap is what buys
  the protocol its headroom; adopting one without the other fails §5.2 G4.
  Qwen3.5-0.8B was verified to emit well-formed tool calls zero-shot (15 calls over 6
  chunks, 0 format failures) before being adopted.
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

The transcript is read as a stream. The harness owns the memory; the model emits **one
tool call per chunk**. No conversation history crosses steps — memory is the entire
carry-forward, which is the property that keeps each step's context constant-size and
learnable at sub-1B scale.

**Single-turn, by measurement.** The model emits its tool call and the harness applies
it; there is NO tool-result message and NO second invocation. A conventional
observe-the-result agent loop was measured on real chunks and costs **exactly 2 model
invocations per chunk plus 1.89x prefill** (the second turn must re-send the system, the
full ~2,500-token chunk, the assistant's calls, and the tool results). Projected against
§7 that is **32–51 min** depending on decode share, versus a 20-minute ceiling — and it
is a property of the control flow, so no amount of fine-tuning recovers it. The model
still chooses which operations to perform and with what content; what is removed is only
the round trip that tells it what the harness already guarantees.

**External memory — WORKING SET plus JOURNAL (v1.1, normative).**

v1.0 had one memory serving two incompatible jobs: the model's per-step context (wants to
be small, because it is re-prefilled every step) and the sole carrier of everything the
meeting produced (wants to be large). v1.1 splits them.

**1. The WORKING SET — what the model sees. Bounded, unchanged from v1.0.**

```
ARC: <1–3 sentences: how the meeting has developed so far>
POINTS:
[1] <key point, decision, or commitment>
[2] <…>
```

`ARC` ≤ 80 tokens; `POINTS` ≤ 16 entries of ≤ 25 tokens; total ≤ ~600 tokens. This keeps
every property that made v1.0 learnable at sub-1B: constant per-step context, no
conversation history, bounded prefill. Points now carry a **stable integer id** (see the
step grammar).

#### 4.1.1 These caps are CONVENTIONS, not constraints (v1.4, normative)

Stating a number normatively implies it is doing work. **Measured, these are not.** Of 384
attempted ops at the production budget, 24.5% are refused — and `point too long` plus
`arc too long` together are **0.6%**. The caps almost never bite. What consumes the step's
output is repetition: `duplicate point` 14.8% plus `arc unchanged` 7.0%, i.e. **~22% of every
step re-emits already-recorded content**, which on-device is latency spent against G4.

So the spec was precise about what does not matter and silent about what does. Two rules
follow, and they apply to every numeric constant in this document:

- **A constant is labelled DERIVED or CONVENTION.** *Derived* means a measurement fixes it and
  it must be re-derived when its inputs change (`CHUNK_TOKENS` against the device's context
  window; G4's device throughput constants, which carry the depth each was measured at).
  *Convention* means the value is arbitrary within a wide band and is fixed only so numbers
  stay comparable across builds. **`ARC`, `POINTS_CAP` and `POINT_TOKENS` are CONVENTIONS.**
- **A convention may not be cited as an explanation.** "The model lost detail because of
  `POINT_TOKENS`" was checked and is false — every point in the G1 probe fits the cap — and
  raising it 25 → 32 to relieve the pressure made the gate *worse* (8/27 → 3/27). Do not raise
  a cap hoping to recover content; the content is being spent on repetition, not truncated.

**The real constraint is unnamed and ungated**, and this is where a future version should act:
nothing in the protocol prices re-emitting what memory already holds. The harness refuses such
ops after they are decoded, so the tokens are already paid for. `guards` detects it, G7 gates
the churn subset of it, and the remaining ~15% of duplicate-point output is neither priced nor
gated.

**2. The JOURNAL — append-only, harness-owned, the model NEVER reads it.**

Every point ever added is appended to the journal with its chunk index. A point leaving
the working set — whether evicted by cap overflow or superseded by the model — is
**retired to the journal, never destroyed**, carrying a `superseded_by` link when the
model replaced it.

**Why this is normative and not an optimisation.** v1.0's single memory forced a
compression ratio that grows with meeting length, because 480 tokens are shared across
every chunk:

| chunks | agent budget/chunk | agent ratio | map-reduce ratio |
|---|---|---|---|
| 4 | 120 tok | 21:1 | 10:1 |
| 12 | 40 tok | 62:1 | 10:1 |
| 37 | **13 tok** | **193:1** | **10:1** |

Measured consequence on the three longest held-out meetings (2026-09-03): the model
correctly identified 41, 23 and 27 points worth recording and **80%, 65% and 48% of them
were evicted before the end**. Past ~16 chunks the working set is in permanent overflow —
every new point destroys an old one. That is the mechanism behind v1.0 losing to its own
map-reduce baseline **0 wins in 25 meetings** (−0.213 rouge1 against reachable
references) and recording 3.6 grounded specifics per meeting against the baseline's 8.5.

The journal costs nothing at read time — the model never sees it, so per-step prefill is
unchanged — and it makes capacity scale with meeting length instead of against it.

**Step grammar.** One call per chunk; zero or more lines:

One `update_memory` tool call, JSON arguments, all fields optional:

```
<tool_call>{"name":"update_memory","arguments":{
  "arc":"<replacement arc note>","add":["<point>",…],
  "revise":[{"id":<int>,"text":"<replacement>"}],"drop":[<int>,…]}}</tool_call>
```

| field | semantics |
|---|---|
| `add` | append these points, each assigned a fresh id |
| `revise` | **v1.1**: atomically supersede point `id` with `text` — one op, journalled with a `superseded_by` link |
| `drop` | retire these ids from the working set (content this chunk closes out); the points survive in the journal |
| `arc` | replace the arc note |
| *(empty `arguments`)* | nothing worth recording in this chunk — the former `NOP` |

**Addressing is by integer id, not text prefix (v1.1, normative).** v1.0 addressed points
by a ≥4-token text prefix with ambiguity resolved as refusal, which required the model to
reproduce a prefix of its own earlier phrasing. Measured consequence: the churn signature
`DROP «X»` + `ADD «X'»` with X'≈X, at **28.2% of steps** on real ASR — the model
rewriting what it already had instead of reading forward. Ids remove the failure by
construction: you cannot mis-address a point you can see numbered in front of you.

**`revise` exists because DROP-then-ADD is what produced churn.** v1.0 had no way to
express "this point is now wrong, here is the correction" as a single act, so revision was
two ops that the harness could not distinguish from churn — and `guards.restates_dropped`
detects that pattern precisely because it fires on both. `revise` makes supersession
atomic, journalled, and separable from churn in both the training data and the metrics.

**Supervision must express a revision AS `revise` (v1.2, normative).** A gold step that drops
a point and adds a reworded version of it demonstrates the two-op form, which is the surface
shape of churn, on exactly the occasions the atomic op exists to cover. Measured on the v1.3
pool: 118 single-drop-single-add steps are genuine revisions (`第158號宣告支援…` →
`支援…宣告`; `法案將東斯皮爾大道123號列為歷史地標` → `法案將第10選區東斯皮爾大道123號列為歷史
地標`) that the migration left unconverted because its text-prefix relatedness test missed
them. Conversion is by content similarity, and the three-way split is unchanged and still
load-bearing: **near-identical → churn, drop the row; related-but-changed → `revise`;
unrelated → genuinely separate `drop` + `add`.** Converting indiscriminately would launder
churn into a sanctioned op.

**The synthesis view collapses near-duplicates (v1.2, normative).** `apply_ops` refuses only
EXACT duplicate points, and before v1.1 eviction destroyed most near-duplicates; now nothing
removes them, so the journal accumulates the same point said twice. Measured: near-duplicate
entries rose **5.6% → 11.2%** when the synthesis view became journal-shaped. `synthesis_view()`
therefore merges entries above a high character-trigram similarity, keeping the later phrasing.

Two constraints on that merge, both normative because both were found by measurement:

* **Points carrying different numerals are never merged.** `第1項決議` and `第11項決議` score
  0.667 similarity and would have collapsed; this corpus is full of agenda items, ordinance
  numbers and dollar amounts distinguished by exactly one figure, and merging those loses a
  distinct decision.
* **Superseded entries are exempt.** A revision's two halves are near-duplicates by
  construction — that is what a revision is — and collapsing them would delete precisely the
  evidence G1 measures.

**Honest scope: this did not reduce churn.** It was built as a churn fix and, measured at two
seeds, is not one (`v13` mean 23.0% against `v12`'s 23.4%). It is retained because it makes the
synthesis input truthful and produced the most stable retention measured (0.936 at both seeds).
A mechanism measured in the data is not thereby a demonstrated cause in the model.

**One batched call, not one call per operation.** Both forms are valid Qwen tool calls;
the choice is measured. For the same three operations: edit lines 36 tokens, **one
batched call with JSON arguments 45 (1.25x)**, one batched call with XML parameters 71
(1.97x), one native call per operation **98 (2.72x)**. Decode dominates CPU latency, so
the per-operation form alone would put §5.2 G4 out of reach.

The tool schema is declared in a compact hand-written system prompt (**187 tokens**, as
implemented, including the caps and the supersede rule), not via the chat template's
`tools=` rendering (**313 tokens for one tool, 434 for four**).
The rendered preamble is instruction boilerplate for a model that has not been trained
on the schema; a fine-tuned student does not need it, and it would cost more prefill per
step than the entire v0 system prompt (266 tokens).

Deliberately small. No multi-point rewrite op: the prior project measured that as the
heaviest op in its grammar and never validated it at ≤1B. Cap overflow is handled
**deterministically by the harness** (evenly spread, never head-truncated — dropping
the tail of a time-ordered list drops the end of the meeting, where decisions land),
**and in v1.1 the evicted point is retired to the journal rather than deleted**, so
overflow costs working-set attention but not information.

**Termination — `SYNTHESIZE` reads the JOURNAL, not the working set (v1.1, normative).**
Transcript exhausted → one final call over every point the meeting produced, with
superseded points marked as such, → §3 prose.

**Why this is the most important change of v1.1.** In v1.0 synthesis saw only the ≤480-token
survivor set, so anything lost at read time was unrecoverable — and the model was asked to
write a rich summary from an impoverished input, which is exactly the pressure that
produces invention. The corpus then institutionalised it: **39.9% of the specific claims in
`SYNTHESIZE` training targets do not appear in the memory those targets were written from**
(1,347 of 3,376, across 45% of rows), because §2.2 stage 3 composed the target from the
whole-meeting gold summary rather than from the input. The student reproduced the rate
almost exactly: **44% ungrounded on real ASR**. Feeding synthesis the journal removes the
gap between what the target may assert and what the input contains.

**Journal overflow is folded, not truncated.** If the journal exceeds the synthesis
context it is reduced in hierarchical passes — the mechanism `baseline.run_map_reduce`
already implements and which measured **1 fold pass on a 92k-token meeting**, never
overflowing. Superseded points are folded last so a reversal cannot be lost to batching.

**Revision becomes RECORDED rather than destructive, which is what G1 needs.** In v1.0 a
reversal required the model to hold both the original decision and its overturning inside
a 480-token window at synthesis time; G1 measured 3/27 for `qwen-tools-v5` and never
exceeded 8/27 across five refuted fix attempts and two model families. Under v1.1 the
harness guarantees the pairing: the journal carries `X, superseded_by Y` regardless of how
many chunks separate them. **This does not make the model better at noticing reversals —
it removes the requirement that it remember one.**

**Context budget (4k).**

| | reading step | synthesis step (v1.1) |
|---|---|---|
| SYS | ~187 (tool schema) | ~250 |
| memory | ≤600 (working set) | journal, ~25 tok/point |
| chunk | ~2,500 | — |
| output | ~190 (one tool call) | <1,000 (prose) |
| **total** | **~3,477** (unchanged) | **~1,500–2,500 typical** |

**The reading step's budget is deliberately unchanged.** The journal is invisible at read
time, so v1.1 buys its capacity without touching the per-step cost that G4 is measured
against — and re-prefilling the working set every step already costs ~19% of the
transcript again (17,760 tokens over 37 chunks), which is why growing the *visible* memory
was never an option. Synthesis grows instead, once per meeting: ~40 journalled points is
~1,000 tokens, well inside the 8k context, and folds hierarchically beyond that.

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
| summary length | keep, in characters and in `tokens.char_tokens` (normative); the STUDENT's tokenizer for budget estimates only |
| QAEval | **dropped** — needs Chinese QG+QA models, no supported zh path. Replaced by the faithfulness judge below, not left unmeasured. |

### 5.0 Instrument validity (v1.2, normative)

Three measurement defects were found in a single session (2026-09-03), each producing
plausible numbers for weeks, and none caught by a failing test. They share one shape: the
code still ran and still returned something readable. Three rules follow.

**1. A documented false positive must be MEASURED and BOUNDED, never merely acknowledged.**
`evalkit/grounding.py` declared numeral-system reformatting an "accepted false positive". It
was not a constant background rate — it fired hardest on the best-written summaries, because
the corpus writes figures in Arabic and fluent zh-TW writes them in CJK. The correction was
~6 points for the crudest checkpoint and ~15 for the best. **When an instrument documents a
known false positive, measure whether it correlates with the property being measured; if it
does, it is a bias, not noise, and results cannot be compared across it.**

**2. A protocol change requires re-validating every consumer of every term whose meaning
changed.** v1.1 split `Memory.points` (the ≤16 working set) from everything recorded.
`evalkit/behaviour.py` kept reading `points`, so three metrics silently changed meaning —
`starved` began firing on the best-accumulating meetings, `chars_per_point` could no longer
detect under-rendering, and G5 had no numerator at all. **All three shifted in the direction
of flattering the model.** When a version bump changes what a noun denotes, grep for every
consumer of that noun before reporting anything.

**3. Train/serve artifacts must agree on version, and the check must be enforced, not
assumed.** The training pool's `prompt_version` was validated while the VALIDATION set's was
not, so every v1.1 build computed `eval_loss` — the basis of every "best epoch" claim —
against the superseded `tools-v1` format, whose synthesis rows actively penalise v1.1
behaviour. Any artifact participating in a reported number carries its version, and mismatches
are refused loudly rather than tolerated.

**Corollary, normative: `eval_loss` does not select checkpoints here.** Measured across three
builds, the best-by-loss checkpoint has been the best artifact, the worst artifact, and
neither. Export the candidate epochs and measure them behaviourally.

### 5.1 Faithfulness (normative)

A fluent summary that inverts a decision is the failure that matters most, and it is
the one ROUGE cannot see. It is measured, by two means:

- **Third-family LLM judge.** The contamination rule constrains *which* model, not
  whether to use one: the judge must be neither **Qwen-family** (authored the reference
  summaries, §2.2 — and, as of v1.2, the supervision) nor **Gemma-family** (translated all
  corpus text, and now authors the span references). Any third family — Llama, Mistral,
  DeepSeek, GLM, Kimi, locally or via API — is uncontaminated by this pipeline and is
  permitted. Per claim in the summary: SUPPORTED / CONTRADICTED / UNSUPPORTED against
  retrieved transcript spans.

  **TWO independent third-family judges, with agreement reported (v1.2, normative).** One
  judge cannot detect its own failure mode, and this project has already paid for that: the
  single local judge (`gpt-oss-20b`) spent its whole budget in the reasoning channel and
  returned empty `content` on **21 of 40** meetings, systematically on the LONGEST summaries
  (median 5,087 chars vs 562) — so G2 silently compared only the control arm's shortest
  outputs and reported "14 vs 11, FAIL" for what was really 18 vs 109. That is the identical
  argument §5.0 makes for the grounding instrument: a failure correlated with the property
  being measured is invisible from inside.

  **FIVE judges from five different third families**, scored by MAJORITY per claim, with the
  per-claim agreement rate reported.

  **Why an ODD panel.** A verdict needs 3 of 5; there is no tie to break. The panel passed
  through four while it was being assembled, and four is the one size to avoid: an even panel
  deadlocks 2-2, and any tiebreaker (seniority, cost, "the fastest one") smuggles an unmeasured
  preference into the gate. Odd panels resolve on their own.

  **Why five and not two.** Two was the first rule here and is strictly weaker: a disagreement
  can only be escalated, never resolved, so every split becomes manual work and one flaky judge
  vetoes a comparison. A larger panel makes the common case self-resolving, survives a judge
  degrading without collapsing to a bare pair, and — most importantly — makes judge reliability
  MEASURABLE. Unanimity rate is a property of the panel that a pair cannot report about itself,
  and the failure this whole rule exists to prevent was exactly a judge failing invisibly:
  `gpt-oss-20b` returned empty content on 21 of 40 meetings, systematically on the longest
  summaries, and nothing detected it.

  **An INVERSION is claimed only on unanimity.** A contradicted decision is the failure §5.1
  exists to catch and the one that most damages a user, so it is held to the strictest
  standard; any split on an inversion goes to the human slice. Majority is sufficient for
  SUPPORTED/UNSUPPORTED, which are gradations rather than product defects.

  **Panel disagreement is reported as a result, not smoothed away.** A claim that five
  independent families cannot agree on is evidence about the CLAIM — usually that the summary
  is ambiguous rather than wrong — and that is worth surfacing to a human, not averaging into
  a rate.

  **Cost is part of the choice, and picking on quality alone was wrong.** A G2 pass is
  ~2,400 calls (40 meetings x 2 arms x ~10 claims x 3 votes). Measured against the provider's
  published allowances, the first two judges selected here — `glm-5.3` (1,080 calls/month) and
  `kimi-k3` (~490) — would each consume MONTHS of quota in a single run. Verified working,
  correct on planted inversions, and affordable:

  | judge | family | calls/month | note |
  |---|---|---|---|
  | `opencode:muse-spark-1.3-contributor` | Muse | **226,600** | needs the `/responses` endpoint |
  | `opencode:mimo-v2.5` | Xiaomi MiMo | 150,400 | reasoning-heavy; needs the capped retry |
  | `opencode:longcat-2.0` | Meituan | 57,200 | |
  | `opencode:deepseek-v4-flash` | DeepSeek | 37,800 | fastest, clean bare JSON |
  | `opencode:hy3` | Tencent Hunyuan | 21,500 | reasoning-heavy; needs the capped retry |

  `glm-5.3-flash` (Zhipu, 7,900) is verified working and held as a substitute if one degrades.
  Each judge draws on its OWN allowance, so a 5-judge pass costs ~800 calls per judge, not
  4,000 from one budget — panel size is cheap here; picking an expensive model is not.
  **Avoid `glm-5.3` (1,080/month) and `kimi-k3` (~490).** Both were chosen here first, on
  answer quality alone, before their allowances were checked: a single G2 pass is ~2,400 calls,
  so either would consume months of quota in one run. Judge selection is a cost decision as
  well as a contamination one.

  **The provider's models do not share one protocol or one output budget**, and both failures
  masquerade as outages: a `/responses`-only model posted to `/chat/completions` returns an
  opaque `HTTP 500`, and a reasoning-heavy judge given a 600-token ceiling returns EMPTY
  content after spending 828 of 860 tokens on reasoning. Both are handled in
  `judge/client.py`; neither is discoverable from `/v1/models`.

  **A hosted judge is not reproducible** — a provider may change a model behind a stable
  name — so the model id is recorded with every result and hosted numbers are never compared
  across dates. Prefer a clean third-family LOCAL judge when one exists.
- **Human review on a 30-meeting slice.** Given that judge noise runs ±0.4–0.5 on this
  kind of scale, small-n human evaluation is competitive with a large automated run,
  and it is the only check not downstream of some model in this pipeline.

**Inversions are reported as a count, not folded into an average** — a single inverted
decision is a product defect, not a fractional score penalty.

### 5.2 Baseline and ship gates (normative)

**No result means anything without a baseline.** The agent architecture must earn its
complexity against a strictly simpler system using the same model and the same token
budget:

**Baseline — map-reduce, no learned memory.** Same Qwen3.5-0.8B, same ~2.5k chunking:
summarize each chunk independently, concatenate the chunk summaries, one final compress
pass to §3's form. No state carried across steps, no training beyond what the same
fine-tune provides. This is deliberately a *fair* opponent — same model, same chunk
size, same output contract — because a strawman baseline makes the gates meaningless.

**Ship gates.** G1-G9 are measured on the same held-out meetings with the same metrics, so
that arms are comparable; **G10 re-measures G5-G9 on a second, real-ASR corpus** (§5.2.8),
because comparability within one distribution says nothing about the one the product serves.

| gate | criterion |
|---|---|
| G1 revision | passes the independent revision probe **with its control arm**; WITHHELD when the probe cannot discriminate (v1.4 — see 5.2.11) |
| G2 faithfulness | inversions ≤ baseline, and not worse than baseline on §5.1's judge panel |
| ~~G3 quality~~ | **RETIRED v1.4** — its references are Qwen-authored from Gemma-translated text, so it measures teacher imitation, not quality. ROUGE/BERTScore remain DESCRIPTIVE. Quality is gated by G2 + G9. See 5.2.4 |
| G4 budget | fits §7's measured envelope on §6's hardware, projected by `evalkit.latency` from the run's OWN token profile |
| **G5 retention** (v1.1) | **≥90% of recorded points reach `SYNTHESIZE`'s input AND are rendered in the summary, with churn no worse than the comparison arm** (v1.2 — see "G5 is inflatable") |
| **G6 grounding** (v1.1) | **≤10% of the specifics asserted in the summary are absent from the transcript, over ≥20 asserted specifics** |
| **G7 stability** (v1.2) | **churn ≤ 10% of steps and no meeting with the ARC frozen from step 0**, on the worse of two seeds |
| **G8 coverage** (v1.3) | **≤ 25% of meetings flagged `starved`, and ≥ 0.5 recorded points per chunk in aggregate** — gated jointly with G6, see 5.2.6 |
| **G9 human utility** (v1.4) | **the only POSITIVE gate.** ≥ 2 reviewers over a fixed 12-meeting sample: agent preferred-or-tied vs baseline on ≥ 8/12, AND ≥ 9/12 rated *usable without consulting the transcript*. See 5.2.7 |
| **G10 domain transfer** (v1.4) | **G5–G9 re-measured on real zh-TW ASR** (`data/asr_eval_v1`), not only on translated MeetingBank; no gate may degrade by more than its stated allowance. See 5.2.8 |

### 5.2.1 Replication is part of the gate (v1.2, normative)

**A behavioural claim from a single training run is not evidence, and this is measured, not
cautionary.** Three pools were retrained at two seeds each and evaluated on the same 40
meetings (`runs/journal-synthesis-outcome.md`):

| pool | churn seed 0 | churn seed 1 | spread | retention s0/s1 |
|---|---|---|---|---|
| `v11` | 3.5% (13/40 clean) | 13.3% (5/40) | 9.8 pp | 0.837 / 0.836 |
| `v12` | 29.8% (4/40) | 17.0% (6/40) | 12.9 pp | 0.921 / 0.937 |
| `v13` | 36.7% (1/40) | 9.2% (6/40) | **27.4 pp** | 0.936 / 0.936 |

Changing only the seed moves churn by up to **27 percentage points** and the clean-meeting
count by 12 of 40 — larger than any effect this project has ever attributed to a data change.

Therefore, normatively:

1. **Every gate whose criterion is behavioural (G1, G5, G6, G7) is measured at a FIXED
   THREE seeds and scored on the MEDIAN, with the full spread reported.**

   *(Superseded rule, kept because the correction matters: this said "at least two seeds,
   scored on the WORSE seed". Worse-of-n is not a fixed standard — it gets harsher as n
   grows, so a pool is penalised for being replicated more, and two pools measured at
   different n are not comparable at all. It produced a live misreading: `v18` measured
   3.6 / 30.8 / 3.9 churn at three seeds and `v17` measured 6.2 / 5.3 / 8.3. Worse-of-n
   ranks `v17` ahead (8.3 vs 30.8) while the median ranks `v18` ahead (3.9 vs 6.2) — and
   `v18` is below EVERY `v17` seed on two of its three runs. The worse-seed rule was
   answering "how bad can one draw be", which is a real question, but it is the OUTLIER
   RATE question below, not the central-tendency one.)*

   The median is the ship-relevant statistic because it is what a typical training run
   produces; the spread and the outlier rate carry the risk.

2. **An outlier rate is reported alongside, and a pool with any catastrophic seed is
   flagged, not silently averaged.** A run is catastrophic when it exceeds G7's churn
   ceiling by more than 2x. `v18` has one such seed in three on identical data — an
   optimisation defect, not a supervision one, and it must be fixed or bounded rather than
   absorbed into an average. A pool may not ship on a good median if its catastrophic rate
   is non-zero and unexplained.

3. **Three seeds is a floor set by cost, not by statistics.** At ~35 minutes per run it is
   affordable; it is NOT enough to estimate an outlier rate (one event in three bounds it
   only very loosely). Report it as "1 of 3", never as "33%".
4. **Every seed's value is reported, never just the summary**, so a reader can see whether
   an effect exceeds run-to-run noise and whether the distribution is bimodal.
5. **A paired sign test over meetings does NOT satisfy this.** It measures whether a
   difference is consistent across meetings for one pair of checkpoints and is silent on
   whether a retrained pair reproduces it. A p-value of 2.2e-07 was produced for a
   comparison whose run-to-run term was larger than the effect.
6. **Effects smaller than the measured spread are reported as UNRESOLVED, never as a
   result.** Measured spreads on this setup reach 27 points of churn within one pool.
7. **Two pools are compared at the SAME seed count**, and a pool measured at fewer seeds
   than another is reported as such rather than ranked against it.

Training a 0.8B full fine-tune on this pool costs ~35 minutes, so this is affordable.

### 5.2.2 G5 is inflatable by the defect G7 detects (v1.2, normative)

Churned re-`ADD`s become points. They therefore **inflate `recorded_points`, inflate
retention (duplicated content is trivially easy to render), and reduce `starved`** — so a
checkpoint can improve every retention-adjacent number *because* it is churning. Measured on
`v12`: retention 0.837 → 0.921 and starved 12 → 6, alongside churn 3.5% → 29.8%.

**G5 and G7 are therefore gated jointly and reported as a pair.** A retention gain accompanied
by a churn increase is not a pass. The general rule, which applies beyond these two: **a metric
that a known defect can improve must be gated together with the detector for that defect.**

### 5.2.3 A gate corpus must exercise the mechanism it gates (v1.2, normative)

`data/asr_eval_v1` has 21 meetings with a **median of 1 chunk and a maximum of 5**, against a
16-point working set. No meeting in it ever overflows, so the journal never fills, nothing is
ever retired, and `build_synth_prompt` falls back to the plain working-set view — the exact
v1.0 code path. An entire v1.1-vs-v1.2 comparison was run on it before this was noticed, and it
produced plausible aggregate numbers throughout.

**Any gate that depends on the journal (G5, and G1 whenever the reversal spans an eviction)
must be measured on a corpus where at least 25% of meetings exceed `POINTS_CAP` chunks.**
`data/heldout_zh` qualifies (10 of 40 exceed 16 chunks); `data/asr_eval_v1` does not and is
valid only for short-meeting behaviour, which is a different question.

**The G3 reference set must not correlate with meeting length (v1.2, normative).** This rule
exists because the current one does, perfectly:

| reference set | n | median chunks | max | meetings over `POINTS_CAP` |
|---|---|---|---|---|
| `data/heldout_refs_reachable.json` | 25 | 7 | 14 | **0** |
| the 15 meetings it excludes | 15 | 23 | 37 | **10** |

The exclusion is structural, not incidental: `build_reachable_refs.py` composes a reference by
reading the WHOLE transcript in one teacher pass — deliberately, so the reference is not a
map-reduce artifact — and therefore skips every meeting above the teacher's context. Those are
precisely the meetings in which the working set overflows and the journal does any work.

**Consequence: every agent-vs-baseline result to date was measured where the agent's
differentiator is switched off by construction.** At ≤14 chunks nothing is ever retired,
`build_synth_prompt` falls back to the working-set view, and the agent is a more expensive way
to run the v1.0 code path. Map-reduce's constant 10:1 compression is entirely adequate there.
The comparison is not wrong, it is *narrow* — and it has been read as a verdict on the
architecture.

Therefore: a G3 result is reported with the chunk-length distribution of the meetings it was
computed over, and **the ship decision may not rest on a reference set containing no meeting
above `POINTS_CAP` chunks.**

**Composing references for long meetings (v1.2, normative).** Hierarchical teacher composition
stays forbidden — it is the baseline's own algorithm and would tilt the comparison.

**The obvious route is REFUTED by measurement, and is recorded here so it is not retried.**
Composing from the corpus's own per-item gold minutes (`itemInfo[].Summary`) looks ideal —
human-authored, present for every meeting regardless of length, independent of both arms. It
does not work: measured against the transcripts they summarise, the gold item summaries are
**55.6% ungrounded (499 of 898 specifics)**, worse than the references they would replace
(38.3%) and far worse than the one-pass teacher route (2.2%). MeetingBank's item summaries are
written from the minutes documents, so they carry ordinance numbers, dollar figures and
department codes (`（財務部門2410）`) that are never spoken aloud.

**The permitted route is SPAN-LOCAL rewriting.** `itemInfo` aligns every item to a
`startTime`/`endTime` in the transcript, and the spans are small — median 366 s (~900 zh
tokens), 95th percentile ~8k tokens — so each item fits a teacher's context regardless of how
long the meeting is. For each item the teacher rewrites its gold minute using ONLY the
transcript span it is aligned to, dropping whatever is not said there; the rewritten items are
then concatenated in time order and smoothed into §3's prose form.

**Why this is not the baseline's shape**, which is the objection that forbids hierarchical
composition: map-reduce *chooses what is salient* in each window, and that choice is the thing
under test. Here the selection is fixed in advance by the human-authored item list, and the
model is only permitted to REMOVE unreachable detail. The per-window structure is shared; the
editorial judgement, which is what would tilt the comparison, is not.

Reachability is measured and reported per meeting, exactly as `build_reachable_refs.py` does,
and a reference set is only usable if its ungrounded rate is comparable to the one-pass route's.

### 5.2.4 G3 must be decomposed and length-controlled (v1.2, normative)

**ROUGE-F1 against an uncontrolled reference measures length matching, not quality**, and this
project has now published a conclusion in each direction from that artifact alone. Measured on
40 held-out meetings, `v14-s1` against its own map-reduce baseline, span references (median 273
characters) — agent output 310 characters, baseline 874:

| metric | agent wins | baseline wins | mean Δ |
|---|---|---|---|
| rouge1 **precision** | **37** | 3 | **+0.151** |
| rouge1 **recall** | 5 | **35** | **−0.230** |
| rouge1 **F1** | 32 | 8 | +0.083 |

The agent is more PRECISE (what it says is in the reference); the baseline is more COMPLETE
(it covers more of the reference). F1 then reports whichever the reference's length favours:

* against the ~870-character one-pass references, the verbose baseline wins — this produced
  the **"0 wins in 25 meetings"** that put the architecture on probation;
* against the 273-character span references, the terse agent wins **32/8** — the mirror image,
  from the same two systems.

Therefore, normatively:

1. **G3 reports precision, recall and F1 separately.** A single F1 number is not a G3 result.
2. **The candidate/reference length ratio is reported with every G3 result**, per arm. A gate
   claim is void if the two arms' ratios differ by more than 2x, because at that point F1 is
   dominated by length.
3. **A G3 conclusion must be stable across at least two reference sets of materially different
   length.** A result that reverses between them is a measurement of the references.
4. **Reference length should match §3's output contract.** A reference far shorter than the
   product's target length makes terseness look like quality; one far longer does the reverse.

**The reversal is now DEMONSTRATED, not predicted (2026-09-04, `runs/probation-v17/`).** The
same checkpoint against its own baseline, the same 40 meetings, both reference sets reachable
(0% ungrounded) and both containing 25% meetings above `POINTS_CAP`:

| references | median chars | agent wins | mean Δ rouge1 | verdict |
|---|---|---|---|---|
| verbose span | 702 | 7 / 33 | **−0.103** (p = 0.000) | G3 all **FAIL** |
| terse span | 273 | 29 / 11 | **+0.049** (p = 0.006) | G3 all **PASS** |

Decomposed, precision and recall are STABLE across both sets (agent 38/2 on precision, baseline
~36/4 on recall, every p = 0.0000); only F1 flips, because the length ratios move from
0.55x/1.30x to 1.41x/3.35x. **The agent is consistently more precise and the baseline
consistently more complete** — a design tradeoff, not a ranking.

**Therefore G3 was WITHHELD in v1.2** — superseded by the retirement below, which is a
stronger finding: the gate is not merely undecidable as written, it would measure the wrong
thing even if decided. Use the reference-free instruments (G5 retention, G6 grounding, G7
churn, G8 coverage) plus G9's human slice.

**Neither existing reference set satisfies (4).** §3 targets flowing prose under 1,000 tokens;
the span references are ~273 characters and the one-pass references ~870. The span set is
REACHABLE (0.0% ungrounded) but too terse; the one-pass set is better-sized but 2.2% ungrounded
and, more seriously, excludes every long meeting (§5.2.3).

#### G3 is RETIRED as a gate (v1.4, normative) — the references are contaminated at the source

v1.2 withheld G3 pending a reference set that is both reachable and correctly sized. **Building
one would not fix it, because the defect is not the length — it is the authorship.**

§2.2 composes every reference with a **Qwen** teacher from text translated by **Gemma**. §5.1's
contamination rule bars judges from both families for exactly that reason. But ROUGE, BERTScore
and MoverScore **are not judges** — they compare the student's output *directly against the
contaminated reference*, so the contamination rule never applied to them. The student is
Qwen3.5-0.8B, distilled from a Qwen teacher, scored on n-gram overlap with that teacher's own
prose. **A high G3 means the student imitates its teacher's style**, which is not the objective
in §1 and is not what a reader wants.

This was hiding behind the length problem: the length flip made G3 undecidable, so nobody asked
whether a decidable G3 would have measured the right thing.

**Consequences, all normative:**

- **G3 is retired**, not withheld. Retirement is one-way (§5.2), and this is a defect of
  construction rather than of calibration.
- **ROUGE / BERTScore / SacreBLEU / MoverScore remain REQUIRED and DESCRIPTIVE.** They are
  reported for continuity with every prior measurement and are useful for detecting gross
  content loss between builds of the same lineage. They decide nothing.
- **Quality is gated by G2 (is it false?) and G9 (is it worth reading?)**, both reference-free
  and neither Qwen-authored.
- **G3 could return only with a reference set authored independently of the student's family**
  — human-written zh-TW summaries of held-out meetings, sized to §3. That is a corpus purchase,
  not a metric fix, and it is out of scope until G9 shows the product is worth the spend.

**The transferable rule: a contamination policy must cover every path from the contaminated
artifact to the score, not only the paths that look like models.** A reference set is as much a
model output here as a judge's verdict is.

**G5 and G6 exist because v1.0 failed both invisibly.** No gate looked at what the memory
retained or at whether the summary's specifics were real, so a checkpoint could pass
G1–G4 while discarding 80% of what it recorded and fabricating a third of its figures —
which is what shipped. Both are measured by `arcsum-eval` and are reference-free, so
neither depends on the reference set whose defects §5.2's own G3 was found to inherit.

**G6 must be read with its denominator.** A summary asserting nothing specific scores a
perfect 0% and is not thereby faithful — it is empty. The **≥20 specifics** floor is the
gate; a build below it is WITHHELD, never passed. This is not hypothetical: a build that
filtered its way to 0.0% ungrounded did so by dropping from 26 asserted specifics to 5.

**Ship the agent only if G1–G6 clear. Otherwise ship the map-reduce baseline** and
record agentic-memory-at-1B as a measured negative result — that is a legitimate
outcome, not a failure to report.

**As of 2026-09-03 the baseline is winning.** v1.0 lost 0/25 on ROUGE against reachable
references and records 3.6 grounded specifics per meeting against the baseline's 8.5. v1.1 is
a targeted fix to the mechanism behind that; **it is on probation, and if it does not beat the
baseline the architecture should be retired rather than iterated on again.**

**The probation verdict must be RE-DERIVED before it is acted on (v1.2, normative).** The
comparison that put the architecture on probation was measured with two instruments since
found defective, both biased against the agent:

* **G3's references are ~46.5% unreachable** — 211 of 454 specific claims do not appear in the
  transcript being summarised, because §2.2 stage 3 composed them from MeetingBank's minutes
  documents. A faithful agent cannot reach them, and surface overlap with such a reference is
  best achieved by a model that invents in the same style. Only 25 of 40 meetings currently
  have reachable references (`tools/build_reachable_refs.py`).
* **The grounding instrument penalised fluent zh-TW.** It compared numerals literally, so a
  correct `六十` against a source `60` counted as fabricated. The corpus writes Arabic and
  fluent zh-TW writes CJK, so the bias scaled with output quality: correcting it moved v1.1
  from 21.2% ungrounded to **6.1%**, and `spec-e3` from 15.6% to **3.1%**, while moving the
  much cruder `qwen-tools-v5` only 33.3% → 27.3% (`runs/grounding-refold.md`).

* **The 25 meetings it was scored on contain ZERO meetings above `POINTS_CAP` chunks**, while
  the 15 excluded contain 10 of them (§5.2.3). The verdict was therefore measured entirely in
  the regime where the journal never fills and the agent executes the v1.0 code path — the one
  regime in which the architecture is expected to buy nothing.

**A bias that scales with the quality of the thing being measured cannot be subtracted out.**
Retiring the architecture is a one-way decision; it may only be taken on a comparison that is
(a) scored against reachable references, (b) measured with the corrected grounding fold,
(c) replicated per §5.2.1, and (d) computed over a reference set that includes meetings above
`POINTS_CAP` chunks. Until that comparison exists, the standing decision remains "ship the
baseline" — which is a shipping decision, not a verdict on the architecture. **"Not measured"
must not be recorded as "lost."**

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

### 5.2.5 Retention is limited by DENSITY, not by the output budget (v1.2, normative)

**Do not relax §3's output cap hoping to raise G5 retention — the cap is not binding, and this
is measured.** Rendering 90% of everything recorded, at the full `POINT_TOKENS` (25) per point,
costs a median of **292 tokens and a maximum of 698** across the 40 held-out meetings against a
**1,000-token** ceiling. **Zero of 40 meetings are budget-limited.** Meanwhile the model emits
252–432 characters — about one third of what it is allowed — and is terser than its own
supervision (targets ~505 characters, student 378 and 252).

Six pools across two teachers and two composition modes reproduce one pattern: **churn and
retention are two faces of a single disposition — how much the model commits to saying per
recorded point — and every lever tried moves them together in opposite directions.**

| pool | churn (worse seed) | retention (worse seed) | recorded |
|---|---|---|---|
| `v13` gemma-3, full replace | 36.7% | **0.936** | 327 |
| `v14` + `revise` | 17.4% | 0.902 | 398 |
| `v16` Qwen3.8, full replace | 8.6% | 0.848 | 369 |
| `v17` Qwen3.8, **additive** | **6.2%** | 0.801 | **423** |

The two axes separate cleanly: **the TEACHER sets the restraint level** (Qwen3.8 covers more
memory in fewer characters than gemma-3 — 0.991 coverage at 28.9 ch/entry vs 0.955 at 34.6 —
and the student inherits the compactness), while **the COMPOSITION MODE sets stability**
(additive beats full-replace on churn for both teachers: 36.7→21.7 and 8.6→6.2).

Therefore, normatively:

1. **Synthesis supervision specifies a per-entry RENDERING DENSITY, not only coverage.** A
   target that merely mentions a point satisfies coverage and fails retention.
2. **The coverage gate's containment threshold is recorded with every target set, and a
   threshold low enough to accept a bare mention is not a coverage gate.** At 0.30 a passing
   target may say almost nothing about an entry; density gates belong above it.
3. **A retention figure is only comparable to another at the SAME containment threshold.**
   Report the threshold with the number.

### 5.2.6 G8: the gate set rewarded saying almost nothing (v1.3, normative)

§5.2.2 established that a metric a known defect can improve must be gated together with the
detector for that defect, and applied it to churn inflating retention. **The mirror image was
left open, and a shipped-candidate checkpoint is sitting in it.**

Abstention improves *five of seven gates at once*, because every one of them is a rate over
what the model chose to say:

| gate | what abstaining does to it |
|---|---|
| G2 faithfulness | fewer claims → fewer ABSOLUTE inversions → passes more easily |
| G4 budget | fewer ops → less decode → more headroom |
| G5 retention | a handful of points is trivially rendered in full |
| G6 grounding | few specifics asserted → few that can be ungrounded |
| G7 stability | almost nothing recorded → almost nothing to churn |

This is measured, not hypothetical. `rl-v3` NOPs **46.2%** of chunks and starves **17 of 40**
held-out meetings, and passes G2, G4, G5, G6 and G7. Splitting its own meetings by the
`starved` flag:

| | n | recorded points | retention | churn events |
|---|---|---|---|---|
| starved | 17 | 132 | **0.955** | **4** |
| healthy | 23 | 234 | 0.915 | 15 |

**Its starved meetings score BETTER on both gated metrics.** And the cost is not merely
coverage: across four independent judges the agent is less faithful *per claim* than the
baseline (by 0.4 to 9.1 points) while asserting ~2.4x fewer claims of near-identical length,
and on every judge the starved meetings invert more per claim than the healthy ones. A model
with an impoverished memory, still asked for a summary, fills the gap — which is the same
mechanism §4.1 v1.1 cites for why the journal exists.

**G8 gates coverage, jointly with G6.** Neither is meaningful alone: coverage alone is
satisfiable by fabricating, and grounding alone is satisfiable by silence. A build must clear
both, and its `specifics` count must always be reported beside its `ungrounded` rate —
a checkpoint asserting 5 specific claims across 20 meetings has a perfect fabrication rate and
is not thereby faithful.

Thresholds are set from the observed range rather than chosen: `STARVED_POINTS_PER_CHUNK`
(0.5) is already the reporting flag's definition, and 25% is loose enough to admit genuinely
sparse meetings — procedural sessions where little is decided — while excluding `rl-v3`'s
42.5%. **The point of G8 is to close a direction the gate set could not see, not to be tight.**

**Generalisation of §5.2.2's rule, and the reason both are stated normatively:** *every*
quality rate here has a denominator the model controls. Gate the numerator too, or the
degenerate solution is to shrink the denominator.

### 5.2.7 G9: human utility, the only positive gate (v1.4, normative)

**Why this must exist.** With G3 retired (§5.2.4), G1–G8 are entirely composed of
defect-absence, budget, and one isolated capability. **None of them asks whether the summary
is good.** A bland, generic, faithful, well-covered, stable, fast summary passes all eight.
§1's objective is a summary *a person reads*, so the set was gating everything except its own
purpose. G9 is also the only instrument here that **cannot be satisfied by saying less** —
§5.2.6 showed that abstention improves five gates at once, and G8 only forces enough to be
*recorded*, not enough to be *useful*.

**Protocol, fixed so it is cheap enough to actually run.** A gate nobody runs is not a gate,
and this project has a tool (`asr_gate.py`) that was meant to run every time and did not.

- **Sample: 12 meetings**, drawn once from the held-out set, stratified 6 long (≥ 20 chunks)
  / 6 short, and **frozen** — the same 12 for every build, so builds are comparable and the
  sample cannot be re-drawn until a build looks good.
- **Reviewers: ≥ 2**, reading zh-TW natively, who did not author the build.
- **Blind and order-randomised.** Reviewers see two summaries per meeting, unlabelled, with
  the transcript available. Arm order is randomised per meeting.
- **Two questions per meeting**, and both are gated:
  1. *Forced choice with ties allowed*: which summary would you rather receive? →
     **agent preferred-or-tied on ≥ 8/12.**
  2. *Absolute*: could you act on this summary without reading the transcript? →
     **≥ 9/12 yes for the agent arm.**

Question 2 is the one that resists the degenerate solution: a summary can win a pairwise
comparison by being marginally less bad while still being useless, and this project has
already shipped a build on a comparative win (`mixed-e3`, 19/20 "curated") that churned on the
first real meeting a user ran.

**Reviewer disagreement is reported, never averaged away.** Two reviewers disagreeing on a
meeting is evidence about the instrument, exactly as the 26–44% inter-judge agreement in
`runs/g2-panel-instrument.md` is. If reviewers agree on fewer than 8/12 meetings, **G9 is
WITHHELD** and the disagreement is the finding.

**G9 does not replace the judge panel; it bounds it.** §5.1's five LLM judges scale and are
reproducible; they also agree with each other on only 26–44% of meetings per meeting, and
they measure claim-level contradiction, which is not the same as usefulness. G2 answers *is it
false?*; G9 answers *is it worth reading?* Both are required.

### 5.2.8 G10: the gates must run on the DEPLOYMENT distribution (v1.4, normative)

Every gate in v1.3 ran on MeetingBank-derived text: clean, professionally transcribed,
machine-translated, and in-distribution for the training corpus. **The product reads real
zh-TW ASR** from the on-device pipeline of §2 — noisy, disfluent, stutter-repeated, with
diarization errors.

The cost of not gating this is on the record: real-ASR curation fell **17/20 → 7/20 across
three checkpoints and nothing caught it**, because every gate since Phase 3 ran on clean text.
It was found by a tool that had to be remembered, not by a criterion. `tools/asr_gate.py` is
additionally now known to have rewarded the failure it was meant to catch, scoring a
553-character confabulation as "curated" because it cleared a length floor.

**G10: G5, G6, G7, G8 and G9 are re-measured on `data/asr_eval_v1`** — the 20 real zh-TW ASR
meetings plus `dram-supply`, the meeting that failed in production and was in no evaluation
corpus. Allowances, stated rather than left to judgement:

| gate | clean-corpus criterion | allowance on real ASR |
|---|---|---|
| G6 grounding | ≤ 10% ungrounded | ≤ 15% |
| G7 churn | ≤ 10% of steps | ≤ 15% |
| G8 starved | ≤ 25% of meetings | ≤ 35% |
| G5 retention | ≥ 90% | ≥ 85% |
| G9 utility | 8/12 and 9/12 | 6/12 and 7/12, on its own 12-meeting ASR sample |

**The allowances are deliberately generous and are not a licence to degrade.** Their purpose
is to make the *direction* gateable at all: a build may be worse on noisy input without that
being a defect, but it may not be **unusable**, and until v1.4 nothing said where that line
was. A build failing G10 while passing G1–G9 has been trained on the wrong distribution, which
is a finding about the corpus (§8 risk 5), not a rounding error.

**This is domain shift, not noise.** Read directly, the NOP'd real-ASR transcripts show the
model requires an explicit *stated outcome* and treats open-ended debate, interpellation and
in-progress Q&A as NOP-worthy — correct for MeetingBank's resolved-agenda-item format and
wrong for legislative proceedings, where much of the value IS the deliberation. Closing that
needs supervision, not a threshold.

### 5.2.9 A threshold gate must state what it can detect (v1.4, normative)

§5.2.1 made replication part of the gate, which makes a *comparison* admissible. It does not
make a *threshold* reliable, and most of G4–G10 are thresholds.

Measured: seed alone moves churn by up to **27 percentage points** and the clean-meeting count
by 12 of 40 — larger than the effect any A/B in this project has ever attributed to data. A
fixed criterion like "churn ≤ 10%" read off one training run is therefore a coin flip in the
region that matters.

**Three requirements, all cheap:**

1. **Every threshold gate reports the between-seed spread** alongside its value, from the ≥ 2
   seeds §5.2.1 already requires.
2. **A gate whose between-seed spread exceeds its margin to the threshold is WITHHELD, not
   passed.** Example: churn 8% against a 10% ceiling with a 9-point spread is not a pass; it
   is an unresolved measurement. This is the same withholding logic §5.2.4 applies to G3 and
   `min_n` applies to the paired comparisons, applied to thresholds.
3. **Each gate states the effect it can detect at the sample size used.** n = 40 meetings for
   the behavioural gates, n = 12 for G9. A criterion nobody can meet or fail at that n should
   be reported as descriptive rather than dressed as a gate.

**Which metric a claim rests on decides whether n = 2 seeds is enough**, and only the
replicate answers it: `retention` moved +0.10 between pools with a within-pool spread of
0.000–0.016 and reproduced at both seeds — real. `churn` differences of the same nominal size
at n = 1 were pure noise. Both were measured in the same experiment.

### 5.2.10 The ship decision admits ROUTING, and states what the agent must earn (v1.4, normative)

v1.2's decision was binary: ship the agent if all gates pass, otherwise ship the baseline and
record the negative result. **That forecloses by construction the one design the measurements
support**, and it never said what would make the agent worth its complexity.

What the measurements actually show: the agent wins on long meetings and loses on short ones;
it has fewer absolute inversions than the baseline and a *worse* per-claim rate on 4 of 4
judges; it fits G4 with margin while the baseline's cost at scale is unmeasured.

**The decision is now over three outcomes, not two:**

1. **Ship the agent** — all gates pass on the whole eval corpus.
2. **Ship a ROUTED system** — the agent serves the slice it is gated on, the baseline serves
   the rest. Admissible only if: (a) the routing key is computable **before** inference from
   the transcript alone (length in tokens is the only such key currently justified); (b) the
   full gate set passes **on the agent's slice**, measured separately, not on the average; and
   (c) the baseline arm is itself gated on its slice. Routing on anything the model produces
   is forbidden — that selects on the measurement.
3. **Ship the baseline** — and record the negative result.

**The agent's minimum value proposition, stated for the first time.** §1 justifies this
architecture by "learning to use external memory well". The architecture therefore has to
demonstrate something map-reduce cannot do by construction, and only two candidates are
available: **revision** (a later chunk correcting an earlier one — map-reduce cannot, its map
calls see no shared state) and **long-meeting coherence** (a single through-line across ~48
chunks, where map-reduce's compression ratio is constant but its cross-chunk consistency is
not). **A build that passes every gate while beating the baseline on neither has not earned
its complexity, and outcome 3 applies even to a clean sweep.** Absent that, the honest
conclusion is that a simpler system does the job.

### 5.2.11 G1 is corpus-limited, and says so rather than pattern-matching (v1.4, normative)

MeetingBank contains essentially no within-meeting reversals: **3.4%** of gold items match
reversal language and those are legislative boilerplate repealing *external* ordinances, never
a decision reversed within the same meeting. So G1's evidence must be synthesized — and
synthetic reversals have been pattern-matched repeatedly, across two model families, two
protocols and six checkpoints, with the independent probe never moving more than noise.

**A gate whose evidence is fabricated by the same process that trains for it is a
memorisation test.** G1 is therefore constrained:

1. **The probe must be independent**: no subject term, entity or figure shared with any
   training reversal. Enforced when the probe set is built, not asserted afterwards.
2. **A control arm is mandatory**: the same probe over decisions taken and *never* reversed.
   Without it the loss cannot be located. With it, `tools/loss_map.py` showed the control arm
   loses nothing between emission and memory (73.3% → 73.3%) while the reversal arm collapses
   (59.3% → 14.8%) — which is what proved the reading step *can* retain an identifying detail
   and that revision specifically drops it.
3. **Both columns are reported**: whether the subject/key term survives, AND whether the late
   outcome is stated. They trade against each other — a fix moved term retention 13 → 20 while
   "states the late outcome" fell 8 → 6, leaving the gate unmoved. One column alone reads as
   progress.
4. **G1 is WITHHELD, not FAILED, when the probe cannot discriminate** — when the control arm
   does not separate from the reversal arm, or when n is below `min_n`. A corpus that cannot
   pose the question cannot answer it, and recording that honestly is worth more than a
   verdict manufactured from synthetic data.

**Route out, if the capability is judged essential**: obtain a corpus that contains natural
within-meeting reversals — legislative committee proceedings do, which is also the deployment
domain (§5.2.8). That is a corpus decision, not a training one, and §8 risk 8 owns it.

---

### 5.3 Conformance status — which normative requirements have EVIDENCE (v1.4, normative)

A specification that states requirements without recording whether they were met produces
exactly the failure §5.0 exists to prevent, one level up: everything still reads correctly and
nothing is checkable. **This section is the register. It must be updated whenever a normative
requirement is executed, and a build may not cite a requirement listed here as unmet.**

#### The panel-aggregation gap: G2 as SPECIFIED has never been measured

§5.1 requires five judges scored by **MAJORITY per claim**, with an **INVERSION claimed only
on unanimity**, at **3 votes** per claim. What has actually been run (2026-09-05,
`runs/g2-panel-instrument.md`) is five judges scored **independently at 1 vote each**, then
aggregated by counting how many judges pass the gate — a different procedure with a different
error profile, and a weaker one.

The obstacle is structural, not an oversight to argue about: `cli/judge.py::judge_case`
persists per-meeting COUNTS (`inversions`, `unsupported`, `claims`) and discards the per-claim
verdicts. **Panel aggregation is therefore impossible from the stored artifacts**, at any vote
count, because the object it must aggregate is thrown away.

**Consequence: the "4/4 judges PASS" result is evidence of the ORDERING and is not a G2
verdict.** It is a legitimate reading — the agent has fewer absolute inversions than the
baseline under every judge, on identical claim sets — and it is not the gate. To close this,
`judge_case` must persist per-claim verdicts keyed by claim, and the panel aggregation must be
computed over them.

**Note the direction of the error.** Unanimity is STRICTER than any single judge, so the
specified G2 would report FEWER inversions for both arms; whether the comparison survives is
unknown, which is precisely why it must be measured rather than assumed to be conservative.

#### Human-in-the-loop requirements: six required, zero with artifacts

| § | requirement | status |
|---|---|---|
| §2.2 stage 4 / §4 | human validation of **every** composed summary before it enters the corpus, called "non-optional" | **NOT DONE** |
| §4.2 step 2 | sample and hand-check the uncovered-span classification at the Phase-1 gate | **NOT DONE** |
| §4.3 | human read of 3 full transcripts (translation gate) | **NOT DONE** |
| §4.3 | blind human preference on synthesis ordering, 20 meetings | **NOT DONE** |
| §5.1 | human review on a 30-meeting slice | **NOT DONE** |
| §5.2.7 | G9 human utility, 12 meetings, ≥2 reviewers | **NOT DONE** (added v1.4) |

**Every checkpoint to date was therefore trained on a corpus that does not satisfy its own
construction spec**, and every reported quality number is downstream of a model at every stage:
Gemma translated it, Qwen composed it, Qwen distilled it, and LLM judges scored it. §5.1's own
sentence — human review "is the only check not downstream of some model in this pipeline" —
describes something that has never happened.

**Two rules follow, both normative.**

1. **§2.2 stage 4 is rescoped, because as written it is unbounded and that is why it was
   skipped.** Validating 1,250 machine-composed zh-TW summaries is an annotation project that
   was never costed. It is replaced by: **a random sample of 30, validated before each corpus
   tranche, with the defect rate reported.** If the sampled defect rate exceeds 10%, the
   tranche is rejected. A requirement that cannot be met is not a safeguard; it is a comment.
2. **G9 is the one human requirement that gates.** The others inform; G9 blocks. This is
   deliberate: five advisory human requirements produced zero artifacts in the project's
   entire history, so a sixth advisory one would produce zero too. If G9 cannot be resourced,
   **the honest state of the project is that its central quality claim is unverified**, and
   §5.2.10's outcome 3 (ship the baseline) is the default rather than the fallback.

#### Other normative requirements without recorded results

| § | requirement | status |
|---|---|---|
| §4.2 step 3 | the **ARC ablation** — agent with `ARC` vs `POINTS` only — required in Phase 2, "drop the slot rather than shipping an ungrounded one" | `runs/ablate-e3/` exists with **no `RESULT.md`**; the conclusion was never recorded, so the slot ships unvalidated |
| §4.2 | report and monitor the edit/`NOP` split; downsample if `NOP` > ~35% of curation targets | partially — `build_sft` reports shares, but the serving NOP rate (46.2%) was never checked against this threshold until §5.2.6 |

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
2026-08-21), at the all-8-cores configuration the measurement itself selected. The
reading-phase total is no longer a projection at all: it is measured per-meeting from each
run's own token profile (v1.4, and see §9's correction — the old trapezoidal basis is void).

**SUPERSEDED FIGURES REMOVED (v1.4).** Every projected quantity below was replaced by a
measurement on 2026-09-05; the table previously carried the trapezoidal depth-ramp basis that
§9's correction retired, and disagreed with it by up to 2.4x. **A budget table that contradicts
the gate it feeds is worse than no table.** All values are now measured on `rl-v3` over the 40
held-out meetings via `arcsum-eval`'s persisted token profile, projected by `evalkit.latency`:

| quantity | value | basis |
|---|---|---|
| calls per meeting | **16.5 reading + 1 synthesis** (was: ~15) | `iter_chunks` at the production budget over `data/heldout_zh`, measured |
| prefill per reading step | **2,811 tokens** (was: ~3,500) | SYS + MEMORY + CHUNK rendered by `build_step_prompt` against a SATURATED memory |
| decode per reading step | **80.3 tokens** (was: ~150) | `Trace.usage`, measured — **and this is checkpoint-specific, not a device constant** (§9 correction) |
| prefill per meeting | ~46.4k tokens | 2,811 × 16.5 + ~1k synthesis |
| decode per meeting | **~1.3k tokens** (was: ~3.1k) | 80.3 × 16.5 + ~250 prose |
| **wall-clock per meeting, all cores (`-C 0xFF`)** | **16.24 min — MEASURED, PASSES the 20.00 ceiling with +18.8% margin** | prefill at depth 0 (58.15 t/s) + decode at depth 3400 (9.87 t/s); median 14.59, p90 34.54; 30/40 meetings under the ceiling |
| same, for an UNSTARVED checkpoint | **18.51 min — PASSES, +7.4%** | `raft-s0-e1`, NOP 7.9%, decode 154.3 tok/step. Recording properly costs 2.3 min |
| wall-clock, big-cores-only (`-C 0xC0`) | **over the ceiling** | the prior project's serving convention; must not be reused regardless |
| peak RSS, `--no-mmap`, 4k ctx | **829 MiB (Q4_0) – 1.34 GiB (Q8_0), measured** | one real 2,500-token completion, `VmHWM` + `smaps_rollup` `Pss` (§9 Phase 0b) — per-completion, not step-count-dependent |

**`rl-v3` is cheap BECAUSE IT STARVES**, and the budget must be read with §5.2.6 in hand: 80
tokens per step is what NOPing 46.2% of chunks costs in decode. The unstarved row is the honest
figure for a build that passes G8.

The design's efficiency argument is that memory is capped, so per-step context is
**constant-size regardless of meeting length** — a 3-hour meeting costs more steps, not
bigger steps, and never exceeds the context window.

**The v1.1 journal does not threaten this, and the number is recorded so it is not
re-litigated (v1.2).** The journal is invisible to the READING steps, so per-step prefill is
unchanged and the constant-size property holds exactly as stated. It enlarges only the single
`SYNTHESIZE` call, whose input grows from ≤16 working-set entries to everything recorded.
Measured on the 40 held-out meetings: median 10 entries, **maximum 26** — so the synthesis
prompt goes from ~480 to at most ~730 tokens, i.e. **+250 tokens on one call per meeting,
+4.2 s at the measured 59.23 t/s prefill**, against a ceiling whose measured margin is ~36 s.

**But it is bounded only by the meeting, not by the protocol**, which is the part worth
watching: a meeting recording 100 points would produce a ~2.6k-token synthesis prompt. That
still fits 4k context, and the reading steps — not synthesis — remain the binding constraint.
**G4 must be re-measured whenever the synthesis input's growth law changes**, not merely when
the per-step cost does.

**Kill criterion — cleared.** Measured wall-clock per meeting is **16.24 min** for `rl-v3`
and **18.51 min** for an unstarved checkpoint, both under the 20.00-minute ceiling, at the
all-cores configuration. The design is shippable as specified **only if the on-device serving
path uses all 8 cores**, not the big-cores-only convention the prior project used, which
independently measured **over** the ceiling on this same hardware.

**The margin narrows to 7% once G8 is satisfied**, which matters against the transient decode
stalls measured over 29.5 minutes of continuous load (2 of 14 rounds losing 13-37% of decode,
with prefill unaffected). Thermal throttling is NOT the risk at 0.8B/Q8 — process contention
is. A build that records more than `raft-s0-e1` must re-measure rather than interpolate.

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

#### CORRECTION (v1.4, 2026-09-05, normative): there is no depth ramp, and G4 must be computed from a measured profile

**The trapezoidal model above contradicts §4.1 and must not be used.** It averages over a
KV depth "ramping ~linearly from 0 toward ~4k" across the reading phase. That is the cost
shape of a conversational agent whose history accumulates — and §4.1 states the opposite
property as a design invariant: **no conversation history crosses steps.** The harness
re-renders memory into a fresh prompt every step, so depth does not ramp. Every reading
step has the *same* cost: prefill from empty, then decode at the depth its own prompt
created. The correct model is therefore a constant, not an integral, which is both simpler
and less forgiving.

Two measurement rules follow, and getting either wrong has already produced a wrong verdict:

1. **Prefill is measured at depth 0; decode is measured at the PROMPT's depth.** They are
   not the same depth and it matters: on the reference device decode runs 12.57 t/s at depth
   0 and 9.87 t/s at depth 3400 — **26% slower where the system actually runs**. The recorded
   19.0 min figure used the depth-0 rate for decode and that single substitution is the
   difference between passing and failing.
2. **Decode LENGTH is a property of the checkpoint, not of the device**, so it must come from
   the run being scored. It was inherited from `qwen-tools-v5` (~190 tokens/step) and reused
   for every later checkpoint; the RAFT pool's targets run **1.45×** longer, which alone
   costs ~2 minutes. **A checkpoint can therefore fail G4 purely by recording more** — which
   is exactly what fixing starvation does, so G5/G7 and G4 pull against each other and must
   be read together.

**Measured profile of the deployed configuration** (40 held-out meetings, `iter_chunks` at
the production budget, prompt rendered through `build_step_prompt` against a SATURATED
memory — the mid-meeting case, not the empty one the model starts with):

| chunk budget | full prompt (mean) | steps/meeting (mean) |
|---|---|---|
| 2,500 | **3,018 tok** | **16.50** |
| 6,400 | 6,398 tok | 6.08 |

Both differ from the values G4 was previously computed with (3,400 tokens, 15.2 steps).

`arcsum.evalkit.latency` is the normative implementation: it holds the device constants
*with the depth each was measured at* and projects from a run's own measured token profile,
which `BehaviourReport.prefill_tokens` / `decode_tokens` now carry off `Trace.usage`. **A G4
claim not produced by that path is not evidence.**

**G4's ceiling applies to a TYPICAL meeting, not the longest one.** The held-out set's
longest meeting is 48 chunks against a mean of 16.5, and no configuration brings it near 20
minutes — at any chunk budget it exceeds 55 minutes. That is a property of the corpus's
length distribution, not a regression, and a per-meeting universal ceiling would be
unsatisfiable by construction. Report the mean and the distribution; do not quietly gate on
the worst case, and do not quietly gate on the best.

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
probe (G1) and the full gate set G1–G10 against the baseline.

**Gate (v1.4):** G1 is not WITHHELD (§5.2.11) and the agent beats the baseline at Phase-1
scale on **G2 plus at least one of §5.2.10's two value propositions** — revision, or
long-meeting coherence. G3 is retired and cannot be cited here (§5.2.4). Fail → either the
architecture doesn't earn its complexity (§5.2.10 outcome 2 or 3: route, or ship the
baseline and record the negative result) or the deficit is diagnosably data-volume-bound,
which is the only justification for Phase 4's spend.

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

Only now translate the remaining ~1,050 meetings, retrain, re-run G1–G10 and Phase 3's
slice. Full spend is justified only by a Phase-2 result that was gated on volume.
