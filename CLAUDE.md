# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 0. Repository state (2026-08-10)

The repo currently contains **only this spec** — no source, tests, or build config exist yet.
There are therefore no build/lint/test commands to run; add them to this section as soon as
the first tooling lands (package/venv setup, how to run the harness on one transcript, how to
run a single eval meeting, how to run the G1 capability screen).

Referenced but not present in-repo: `TOOLS2.md`, the map-reduce baseline, and the eval sets
(T1/T2/micro-cell). Treat §§1–8 below as the contract when writing the first code:

- The harness is deterministic and owns the final word (§5.3, §6); the model only emits op
  lines. Keep model-facing surfaces (SYS prompt, op grammar, NOTES render) byte-stable —
  fine-tune and eval distributions must match exactly (§7.8).
- `clock_to_sec` / `parse_line` are the two primitives everything else depends on; the
  mm↔ss-inverted clock formula is a known past bug (§7).
- Judges must never be Qwen-family (§7).

---

# SPEC — Agentic meeting-transcript summarizer (sub-1B SLM, zh-TW / en)

**Version:** 1.0 · **Date:** 2026-08-09 · **Status:** normative design specification
Implements the CURSOR-agent tool set (`TOOLS2.md`) against the VoxSum transcript/notes
formats inherited from `Luigi/voxsum-qwen35-0.8b-anchored`. Where this file and code
disagree, this file is the contract.

---

## 1. Goal

A **production-ready agentic summarizer** running on a **sub-1B SLM** (primary student:
official Qwen3.5-0.8B, 4k context) that converts a meeting transcript of **≥80k tokens**
(zh-TW or en) into structured meeting notes that are:

- **faithful** — every claim supported by the transcript (FAITH-claim, judged);
- **anchored** — every bullet linked to the transcript line that states it (FAITH-anchor);
- **complete** — the meeting's decisions/actions/key content captured (COVER);
- **globally insightful** — the notes state the meeting-level arc: how the discussion
  evolved, how decisions changed, the bottom line (SYNTH) — **measurably better than
  classic map-reduce** (the ship gate is SYNTH ≥ +0.5 over the map-reduce baseline);
- **safe** — **0% inversions** (no note may state the opposite of the transcript about a
  decision, approval, outcome, or commitment);
- **on-device** — 4k context, ≈785 MB memory envelope, ≤ +25% prefill tokens over the
  map-reduce baseline (~3.2 h per 80k-token transcript on RPi4-class hardware).

**Hard exclusions:** no map-reduce architecture (no per-window independent digests + merge
+ shrink); no free ReAct tool loops (measured unlearnable at ≤1B — multi-turn state and
temporal integration across tool RESULTs fail, and tool results exceed the context).

---

## 2. Input — transcript format v1 (normative)

One utterance per line. **One utterance = one line is a hard rule** (no embedded newlines).

```
[<start>] <speaker>: <text>     diarized, name unknown  → S1, S2, …  (first-appearance order)
[<start>] <name>: <text>        diarized, name known    → real name / role verbatim
[<start>] <text>                no diarization          → no speaker field
```

- **Timestamp** = utterance start only. `M:SS` under 1 hour, `H:MM:SS` from 1 hour.
  Seconds and minutes-in-hour are zero-padded; the leading unit is unpadded.
  Examples: `[0:00]`, `[3:35]`, `[59:58]`, `[1:02:07]`.
- **Speaker field**: `S1…Sn` (order of first appearance), a real name/role, or absent.
  A speaker field never contains `] ` or `: ` and is ≤ 40 chars.
- **No** header, footer, markdown, or escaping. Text is emitted as-is.
- **Parsing (normative)**: split on the FIRST `] `, then the FIRST `: ` after it.
  `parse_line(line) → (timestamp, speaker|None, text)`.
- Long monologue lines exist (VCSum zh: up to ~2.6k chars/line) — readers must not assume
  a max line length.

### 2.1 Example (en)

```
[0:00] S1: Let us discuss the office move.
[2:30] S2: I propose we move to Building B.
[5:10] S1: Agreed, Building B it is.
```

### 2.2 Example (zh-TW)

```
[0:00] S1: 我們來討論辦公室搬遷。
[2:30] S2: 我建議搬到 B 棟大樓。
[5:10] S1: 好，就搬到 B 棟。
```

---

## 3. Output — NOTES format v2 (normative)

Sections in **fixed order**, **all always present**:

```
TITLE: <one short title, ≤ 8 words>
SUMMARY:
- <3–5 short bullets, each ≤ 20 words>
DECISIONS:
- <key decisions made>
ACTIONS:
- <one bullet per assigned action: "name: what they will do"; append "(due: …)" ONLY when a deadline was actually stated>
OPEN:
- <open questions / follow-ups>
TOPICS:
- <main topics discussed>
```

Rules:
- Section keys exactly `TITLE, SUMMARY, DECISIONS, ACTIONS, OPEN, TOPICS`, in that order.
- An empty section is exactly `-` on one line — never "none", never a placeholder.
- `- ` bullets, plain text only. No markdown, no preamble, no commentary, no thinking.
- **Every bullet MUST end with the `[m:ss]` timestamp of the transcript line it supports**
  (same v1 clock format: `[14:30]`, `[1:02:07]` for ≥1 h). The anchor must resolve to a
  real transcript line.
- Per-section caps (harness-enforced): SUMMARY 5, DECISIONS 5, ACTIONS 6, OPEN 4, TOPICS 6.
  TITLE carries no anchor.

### 3.1 Example

```
TITLE: Office move decision
SUMMARY:
- Move to Building B agreed after discussion [5:10]
DECISIONS:
- Relocate the office to Building B [5:10]
ACTIONS:
- S2: circulate the move checklist (due: Friday) [6:02]
OPEN:
- Parking allocation for Building B [7:40]
TOPICS:
- Office move [0:00]
```

---

## 4. Agent architecture — CURSOR (normative)

The transcript is processed as a **stream**. There is exactly **one evolving NOTES state**,
curated by the model; there is no per-window map, no merge, no reduce.

```
harness holds:   STATE  = the current NOTES (v2), capped per section
                 CURSOR = position in the transcript
per step i:
  harness → model:  SYS (protocol, §5.0) + STATE (≤ ~600 tok) + CHUNK_i (~2048 tok of
                    raw transcript lines, contiguous, 2-line overlap with CHUNK_{i-1})
  model → harness:  zero or more op lines (§5.1) — or NOP
  harness:          validate → apply → dedup/cap → advance cursor
termination:       end of transcript → optional VERIFY/ANCHOR sweep (§5.2) → render
```

**No conversation history crosses steps.** STATE is the entire memory. This is the
load-bearing property for sub-1B learnability: temporal integration is done by revising a
visible earlier bullet (UPD), never by remembering past tool results.

---

## 5. Tool set (normative wire formats)

### 5.0 System protocol (SYS)

Fixed per language (`en` / `zh-TW`). States: the NOTES v2 contract (§3), the op grammar
(§5.1), and these rules:
- Every ADD/UPD bullet ends with a `[m:ss]` copied from a line **in the current chunk**.
- `«prefix»` = the first ≥ 6 characters of an existing STATE bullet.
- Output only op lines, one per line. `NOP` is always a valid complete answer.

### 5.1 Edit ops (the agent's tools — one call per chunk)

| op | syntax | semantics |
|---|---|---|
| **ADD** | `ADD <SECTION> - <bullet> [m:ss]` | append an anchored bullet to SECTION |
| **UPD** | `UPD <SECTION> «<old bullet prefix>» -> <new bullet> [m:ss]` | replace the matched bullet (e.g. decision revised: "rejected" → "approved after revision") |
| **DEL** | `DEL <SECTION> «<bullet prefix>»` | remove a bullet contradicted by this chunk |
| **CMP** | `CMP <SECTION>` followed by ≤cap rewritten bullets for that section | model-curated compaction when SECTION exceeds its cap (against the live state — not a batch shrink) |
| **NOP** | `NOP` | nothing worth changing this chunk |

Example step output:
```
ADD DECISIONS - Budget increase approved at 10% [32:14]
UPD SUMMARY «Budget increase» -> Budget increase approved at 10% after CFO revision [32:14]
NOP
```

### 5.2 Final sweep tools (optional, budget-gated; loop-free single calls)

**VERIFY** — per bullet, faithfulness:
- in: one bullet + ≤ 6 evidence snippets (harness-retrieved: anchor neighborhood ∪ lexical
  top-k across the whole transcript, snippet-extracted, ≤ 120 chars each)
- out, exactly one line:
  - `KEEP` — evidence supports the claim
  - `DROP` — claim absent from or contradicted by the evidence
  - `FIX: <corrected bullet> [m:ss]` — claim misstated; corrected rewrite

**ANCHOR** — per bullet, anchor repair:
- in: one bullet + ≤ 8 candidate transcript lines (lexical top-k, each with its `[m:ss]`)
- out, exactly one line: the `[m:ss]` of the candidate that states the claim, or `NONE`
  (→ bullet falls back to the deterministic matcher)

### 5.3 Harness primitives (deterministic, zero model tokens)

Streaming cursor + chunker; STATE store with per-section caps; op parser; bullet-prefix
matcher; anchor validation (must resolve to a real transcript line — within the current
chunk for ADD/UPD); lexical search index; snippet extractor; decision/action timeline
guard; `spread()` cap enforcement; the classic per-window summarizer as **coverage
fallback only** (never part of the agent protocol).

---

## 6. Guards (normative; the harness owns the final word)

1. **Anchor validation** — an op whose `[m:ss]` does not resolve to a line in the current
   chunk is rejected (logged); the bullet falls to the deterministic matcher.
2. **Temporal guard** — any op touching DECISIONS/ACTIONS is cross-checked against the
   time-sorted decision/action timeline; a contradiction is dropped and logged. This is
   the 0%-inversions backstop.
3. **NOP-collapse guard** — K consecutive NOPs over content-rich chunks → coverage
   fallback fills the gap (logged).
4. **Malformed ops** — ignored and logged, never fatal.
5. **Deterministic render** — the final NOTES are rendered from STATE by the harness;
   caps enforced by `spread()` (never head-truncation).

---

## 7. Evaluation & measurement (normative)

All evaluation is judge-based (no reference summaries). The judge family is **gemma-4
only** (local gemma-4-E4B-it for FAITH/INVERT; Ternary-Bonsai-27B for COVER/SYNTH) —
**never a Qwen-family judge** (contamination: Qwen teachers distilled the student).
All timestamps are parsed with `clock_to_sec` (`M:SS` = M×60+S; `H:MM:SS` = H×3600+M×60+S)
— the mm↔ss-inverted formula is a known past bug that corrupted evidence placement.

### 7.1 Quality metrics (per meeting, judged)

| metric | scale | definition |
|---|---|---|
| **FAITH-claim** | 1–5 | every bullet's claim is supported by evidence from **anywhere** in the transcript. 5 = all supported. |
| **FAITH-anchor** | 1–5 | every bullet's claim is supported by the transcript lines **at its `[m:ss]` anchor** (±3 lines). 5 = all supported. |
| **COVER** | 1–5 | how much of the meeting's important content (decisions, actions, commitments, key topics) appears in the notes. |
| **SYNTH** | 1–5 | meeting-level global insight — the arc, how decisions evolved, cross-part dependencies, the bottom line — vs disconnected local fragments. 5 = strong global insight. |
| **INVERT** | YES/NO | any note states the **opposite** of the transcript about a decision, approval, outcome, or commitment. Product requirement: 0%. |
| **UNSUPPORTED** | count | bullets the judge could not verify (reported alongside FAITH). |

FAITH-claim and FAITH-anchor are reported **separately** (the P1 lesson: a single
anchor-strict FAITH conflates "is it true" with "does this one line prove it" and
underestimates true faithfulness by ~1 point).

### 7.2 Judge protocol

**Claim mode** (FAITH-claim): per bullet, the harness retrieves ≤ 6 evidence snippets —
the anchor neighborhood ∪ lexical top-k matches over the **whole transcript** (word
overlap for en, character-bigram overlap for zh), each snippet extracted at the
best-matching window within the (possibly 2.6k-char) line, ≤ 120 chars. Judge verdicts:
SUPPORTED / CONTRADICTED (→ INVERT) / UNSUPPORTED; score 5 = every bullet supported.

**Anchor mode** (FAITH-anchor): identical, but evidence = the anchor neighborhood only.

**COVER/SYNTH call**: notes + per-part agenda (first line of each deterministic window)
→ both scores. **Full-context mode** (single call, transcript ≤ judge budget) is the
cross-validation reference for shorter meetings.

Parsing: last-match regex per key; any unparsable judge output is re-run once, then
scored missing. Judge output caps: ≤ 64 tokens per score block.

### 7.3 Statistical protocol

- **Judge noise ≈ ±0.4–0.5 FAITH** → **Δ < 0.5 is a tie**; never claim sub-noise wins.
- Comparisons are **paired per meeting** (same meetings, same judge, same prompts across
  systems); report win/loss/tie counts and a sign test per metric pair.
- n per tier is 20 → treat reduced cells (n = 6) as directional only.

### 7.4 Operational metrics (per meeting)

| metric | definition | budget |
|---|---|---|
| valid-op rate | % of op lines that parse AND validate (anchor resolves, prefix matches) | ≥ 95% (GT1) |
| malformed rate | % of unparseable op lines | < 5% |
| NOP-collapse | % of content-rich chunks answered NOP | < 10% |
| anchor rate (raw) | % of model bullets natively ending in a resolvable `[m:ss]` | report |
| anchor rate (final) | after deterministic matcher / ANCHOR sweep | report |
| prefill tokens | total input tokens vs map-reduce baseline | ≤ +25% (GT4) |
| decode tokens | total generated tokens | report |
| calls / meeting | model calls (chunks + sweep) | report |
| wall-time / memory | vs the 3.2 h / 785 MB on-device envelope | within envelope |

### 7.5 Eval sets

| tier | n | composition | role |
|---|---|---|---|
| **T1** | 20 | 10 zh-TW (VCSum) + 10 en (MeetingBank/QMSum), 12k–40k tokens, real | regression tier |
| **T2** | 20 | 10 en real (MeetingBank) + 10 zh-TW **synthetic** (adjacent VCSum concatenations, labelled `synthetic`), ≥ 80k tokens | target tier |
| **micro-cell** | 6 | 3 zh + 3 en, held out from all training data | cheap iteration only |

### 7.6 Capability screen (gate G1, pre-eval)

Synthetic meeting with planted facts: an early REJECTED plan, a later APPROVED plan,
two deadlines, and one trap topic (must not be reported). A configuration passes only if
the final notes carry the correct decision chain (rejected → approved), both deadlines,
100% anchored, no trap. The free tool loop failed this for every sub-1B candidate; the
CURSOR protocol must pass before any long-doc eval is run.

### 7.7 Ship gates (vs the map-reduce baseline, same meetings/judge/prompts)

- **GT1 learnability:** valid-op rate ≥ 95%, NOP-collapse < 10%.
- **GT2 faith:** FAITH-claim ≥ baseline +0.3, 0% inversions.
- **GT3 synthesis (the agency bet):** SYNTH ≥ baseline **+0.5** on T1 and T2.
- **GT4 efficiency:** prefill ≤ +25% over baseline.

Ship the CURSOR rung only if GT2 or GT3 clears at equal inversions; otherwise ship the
map-reduce baseline and record agency-at-0.8B as a measured negative result.

### 7.8 Known caveats (must accompany every reported number)

zh T2 is synthetic; the zh pool is largely monologic (VCSum) — contested-zh is
unmeasured; MeetingBank has no speaker labels; judge-noise floor ±0.4–0.5; n = 20/tier;
the fine-tune/eval distribution must match exactly (system prompt included) or scores
are not comparable.

---

## 8. Efficiency budget (4k context, 80k-token transcript ≈ 40 chunks)

Per step: SYS ~250 + STATE ≤600 + CHUNK 2048 ≈ **2.9k in / ~120 out** — constant size,
fits 4k, no growth across steps. Totals ≈ 116k prefill (+14% vs baseline) and ~4.8k
decode (−60%). Final sweep ≤ `AGENT_BUDGET` (default 20) calls × ~900 tok. Memory ≈ the
785 MB envelope (Q4_K_M, 4k KV).
