# SPEC — [project name / goal TBD]

**Version:** 0.1 · **Status:** draft — goal, output format, and architecture not yet defined

---

## 1. Goal

TBD.

---

## 2. Input — transcript format v1 (normative, carried over unchanged from the prior design)

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
- Long monologue lines can occur (up to ~2.6k chars/line in real zh-TW source material
  seen previously) — readers must not assume a max line length.

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

### 2.3 Content source (normative)

Transcripts come only from the **VoxSum Android audio pipeline** (on-device ASR +
speaker diarization) run on real podcast audio — never hand-authored or synthetic.
This system never consumes raw audio; §2's format v1 is exactly that pipeline's
output, and the only input contract this spec needs.

**Selection criteria**: 2–3h duration, ideally 2+ speakers (not a single-host
monologue — unlike the prior project's Gooaye/股癌 corpus), official summary
preferred but not required (useful for eyeballing quality; not part of the formal
eval, §5).

**en — confirmed**: **Lex Fridman Podcast** (~3h avg, 2 speakers, strong official
transcripts + timestamped outlines).

**zh-TW — real 會議 (meeting) recordings, not podcasts** (mainstream Taiwanese
podcasts run 30–60 min and don't clear 2h; real meetings do, and come with a legally
mandated official record instead of show notes). Confirmed categories:
- **立法院委員會** (Legislative Yuan committee sessions), official @legislativeyuan
  YouTube channel — e.g. 經濟委員會 2h59m, 外交及國防委員會 3h19m, 教育及文化委員會
  3h12m. Official verbatim record: 立法院公報 (ly.gov.tw / ppg.ly.gov.tw), indexed by
  committee + date.
- **市政總質詢** (city council mayor question time, e.g. 台北市議會) — 4–5h+ (longer
  than target; may need trimming to an 80k-token-scale window).
- **股東常會** (listed-company annual shareholder meetings) — continuous recording +
  official 議事錄 (minutes, filed within 20 days on MOPS) are legally mandated for
  every listed company. Confirmed example: TDCC FY2024, 2h23m.

Plan: draw zh-TW corpus meetings from these three categories, verify duration/audio
quality per recording before use, and pair each with its official record (公報/議事錄)
as an independent quality reference alongside the teacher model's summary (§4).

**Initial corpus size (normative)**: **10 en** recordings (Lex Fridman Podcast) + **10
zh-TW** recordings (drawn across the 立法院委員會 / 市政總質詢 / 股東常會 categories
above) = 20 meetings total.

---

## 3. Output — TBD

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
