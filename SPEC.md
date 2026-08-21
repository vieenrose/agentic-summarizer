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

The transcript is never hand-authored or synthetic for training/eval corpora — it is
the direct output of the **VoxSum Android audio pipeline** (on-device ASR + speaker
diarization) run against a real recorded audio source. This spec's system never
consumes raw audio itself; §2's format v1 is exactly what that pipeline emits, and is
the only contract this system's input side needs to honor.

**Source material selection criteria** (for building training/eval corpora):
- **Format**: podcast episodes (real recordings, not scripted/synthetic).
- **Duration**: 2–3 hours per episode — long enough to stress the ≥80k-token target
  this system is meant to handle without concatenating multiple episodes.
- **Language**: zh-TW or en.
- **Speakers**: ideally **2 or more** (dialogue/interview format), not a single-host
  monologue — this is a deliberate change from the prior project's Gooaye/股癌 corpus,
  which was monologue-heavy and is no longer considered an ideal source for this
  reason.
- **Preferred but not required**: episodes with an official, high-quality summary
  published by the show itself (show notes, timestamped outline, or full transcript)
  — useful as an independent quality reference when eyeballing this system's output,
  even though the formal evaluation protocol (§5) does not compare against reference
  summaries.

**Candidate channels researched so far**:

| channel | language | typical episode length | speakers | official summary quality |
|---|---|---|---|---|
| **Lex Fridman Podcast** | en | ~3h average (range ~1–6h) | 2 (host + one guest) | **Strong** — official timestamped outline and full transcript published per episode on lexfridman.com |
| **The Joe Rogan Experience** | en | ~2h39m average | 2 (host + one guest) | Weak — episode pages carry only a short description, no official transcript/outline |

**zh-TW: no single-episode match on duration; leading candidate identified for the
other two criteria.** Popular Taiwanese talk podcasts (台灣通勤第一品牌 ~45–55 min,
百靈果 News, 大人的Small Talk) are multi-speaker but run well under 2 hours per
episode — this appears to be a market-wide norm (Taiwanese podcasts mainstream around
30–60 min/episode; 2–3h marathon interviews are not a native zh-TW podcast format the
way they are in the US market). 館長陳之漢's live streams fit the duration/speaker
count but are YouTube livestream recordings, not a conventional podcast feed, with no
official written summary.

**Leading zh-TW candidate: 法客電台 BY 法律白話文運動 (Plain Law Movement)**
- Multi-host legal/political-commentary show, 5 rotating hosts (貴智, 珞亦, Yoyo/鎬佑,
  廷奕, 子鈺) — clears the "2+ speakers" bar comfortably.
- Produced by an award-winning independent legal-media outlet (19th 卓越新聞獎
  winner, 20th nominee) that also publishes written legal explainers — the org has
  real editorial capacity behind it.
- Its longest episodes (checked directly against Apple Podcasts): 法客話題 #216
  (職場霸凌), 1h33m; 政治歸政治 #218, 1h18m; YO智事務所 #135, 1h27m; 政治歸政治 #219,
  1h16m; YO智事務所 #134, 1h12m — still short of the 2h floor on any single episode.
- **Show-notes quality is inconsistent, checked per-episode, not assumed show-wide**:
  政治歸政治 #218 and 法客話題 #216 (職場霸凌) carry genuine structured topic
  outlines / discussion-question lists (high quality); 政治歸政治 #219 and both
  YO智事務所 episodes checked have only a sponsor blurb or bare timestamp labels
  (low quality). Pick specific episodes by checking their show notes individually —
  do not assume the whole show meets the summary-quality bar.
- **Practical use**: since no single episode clears 2h, reach the 2–3h target by
  concatenating 2 same-series episodes with good show notes (e.g. two 政治歸政治 or
  法客話題 installments) — the same concatenation approach the prior project used for
  its synthetic long-zh eval tier (§9 of the old spec).

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
