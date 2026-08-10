# agentic-summarizer

Agentic meeting-transcript summarizer for a **sub-1B** small language model (primary student:
Qwen3.5-0.8B, 4k context), targeting zh-TW and en meetings of **≥80k tokens** and producing
structured, timestamp-anchored meeting notes on-device.

> **Status:** design stage. This repo currently contains the normative specification only —
> no code yet. [`CLAUDE.md`](CLAUDE.md) is the contract; where it and future code disagree,
> the spec wins.

## The idea: CURSOR, not map-reduce

Classic map-reduce summarization (independent per-window digests → merge → shrink) produces
locally-correct but globally-disconnected notes: it cannot say how a decision *evolved*. Free
ReAct-style tool loops are the obvious alternative, and were measured **unlearnable** at ≤1B —
multi-turn state and cross-result temporal integration fail, and tool results overflow context.

CURSOR is the middle path. The transcript is streamed; there is exactly **one evolving NOTES
state**, curated by the model; **no conversation history crosses steps**.

```
per step i:
  harness → model:  SYS (~250 tok) + STATE (≤600 tok) + CHUNK_i (~2048 tok of raw lines)
  model → harness:  edit ops — ADD / UPD / DEL / CMP / NOP
  harness:          validate → apply → dedup/cap → advance cursor
end:                optional VERIFY / ANCHOR sweep → deterministic render
```

Because STATE is the entire memory, temporal integration becomes *revising a visible earlier
bullet* (`UPD`) rather than remembering a past tool result — the property that makes the
protocol learnable at this scale. Per-step input is constant-size (~2.9k in / ~120 out), so it
fits 4k context with no growth across ~40 chunks.

## Output format

Fixed sections, always all present, every bullet ending in the `[m:ss]` of the transcript line
that supports it:

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

## Guarantees the harness enforces

The model proposes; the deterministic harness decides. Anchors must resolve to a real line in
the current chunk; ops touching DECISIONS/ACTIONS are cross-checked against a time-sorted
timeline (the 0%-inversions backstop); K consecutive NOPs over content-rich chunks trigger a
coverage fallback; malformed ops are logged, never fatal; caps are enforced by `spread()`,
never head-truncation.

## Ship gates

Judged evaluation only (no reference summaries), gemma-4 judge family — **never** Qwen-family
(the Qwen teachers distilled the student). Paired per meeting; judge noise is ±0.4–0.5, so
Δ < 0.5 is a tie.

| gate | requirement (vs map-reduce baseline) |
|---|---|
| GT1 learnability | valid-op rate ≥ 95%, NOP-collapse < 10% |
| GT2 faith | FAITH-claim ≥ baseline +0.3, **0% inversions** |
| GT3 synthesis | SYNTH ≥ baseline **+0.5** on T1 and T2 |
| GT4 efficiency | prefill ≤ +25% over baseline |

Ship CURSOR only if GT2 or GT3 clears at equal inversions — otherwise ship the map-reduce
baseline and record agency-at-0.8B as a measured negative result.

On-device envelope: ≈785 MB (Q4_K_M, 4k KV), ~3.2 h per 80k-token transcript on RPi4-class
hardware.

## Caveats

zh T2 is synthetic (adjacent VCSum concatenations) and the zh pool is largely monologic, so
contested-zh is unmeasured; MeetingBank has no speaker labels; n = 20 per tier. These must
accompany every reported number.

## Full specification

See [`CLAUDE.md`](CLAUDE.md) for the normative transcript format (v1), NOTES format (v2), op
wire formats, guards, judge protocol, eval tiers, and efficiency budget.
