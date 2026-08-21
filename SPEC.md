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

---

## 3. Output — TBD

---

## 4. Architecture — TBD

---

## 5. Evaluation & measurement — TBD

---

## 6. Reference hardware (normative)

- **Target/reference inference device: Oppo Reno 7 smartphone, CPU-only inference.**
  No GPU/NPU acceleration path is assumed — all on-device inference (any model in the
  eventual architecture, §4) must run acceptably on this device's CPU alone.
- **Open — needs confirming before it can drive numeric budgets (RAM/context/latency
  caps in §5)**: "Reno 7" spans several regional variants with different chipsets
  (e.g. Reno7 4G: MediaTek Helio G35; Reno7 5G / Reno7 Z: Snapdragon 695 or Dimensity
  900, depending on market) and RAM configs (6/8 GB, some with virtual-RAM extension).
  Confirm the exact variant/chipset/RAM in use so the peak-RSS and latency budgets in
  this spec are grounded in the real device, not a guess.
