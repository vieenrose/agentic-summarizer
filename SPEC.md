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
