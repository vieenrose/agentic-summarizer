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

**Selection criteria**: real podcast episodes, 2–3h duration, zh-TW or en, ideally 2+
speakers (not a single-host monologue — unlike the prior project's Gooaye/股癌
corpus), official show notes preferred but not required (useful for eyeballing
quality; not part of the formal eval, §5).

**en — confirmed**: **Lex Fridman Podcast** (~3h avg, 2 speakers, strong official
transcripts + timestamped outlines). Joe Rogan Experience dropped — fits
duration/speakers but has no official summary.

**zh-TW — no single episode clears 2h** (mainstream Taiwanese episodes run 30–60 min;
2–3h marathon interviews aren't a native zh-TW format). Leading candidate: **法客電台
BY 法律白話文運動** (5 rotating hosts, award-winning legal-media outlet). Longest
episodes run ~1–1.5h with genuine structured show notes on some installments:
政治歸政治 #218, 法客話題 #216 (職場霸凌), YO智事務所 #135/#134/#133/#132 — pick
episodes by checking notes individually, quality varies by installment. Plan:
concatenate 2 same-series episodes to reach the 2–3h target (as the prior project did
for its synthetic long-zh eval tier).

---

## 3. Output — TBD

---

## 4. Architecture — TBD

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
