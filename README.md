# Agentic Meeting Summarizer

A 0.8B model that reads a long Chinese meeting transcript in pieces, keeps notes as it
goes, and writes one short summary — entirely on a phone, offline.

## The problem

A recorded meeting runs about **34,000 tokens** in zh-TW. The model's context window holds
**4,000**. It cannot read the meeting and summarise it in one pass, so it has to read
incrementally and carry forward what matters.

## The approach

The transcript is split into ~2,500-token chunks. For each chunk the model makes **one tool
call** that edits an external memory the harness owns:

- **ARC** — one to three sentences of running context.
- **POINTS** — up to 16 decisions and commitments, each addressable by a stable id.
- **JOURNAL** — every point ever recorded, append-only and invisible to the model. Points
  retired from the working set are kept here, never destroyed.

No conversation history crosses steps: each step sees the system prompt, the current memory,
and its own chunk. A final call turns memory plus journal into the summary.

**The capability being trained is memory curation, not context length.** Deciding what to
record, what to supersede and what to drop is the whole task.

## Constraints

| | |
|---|---|
| Model | Qwen3.5-0.8B, Q8, 4k context |
| Device | Oppo Reno 7 5G, CPU only, offline |
| Budget | ≤ 20 minutes per meeting |
| Language | zh-TW only |
| Output | flowing prose, < 1,000 tokens, no bullets |

## What counts as success

The control arm is **map-reduce**: the same model, the same chunks, each summarised
independently then compressed once. It is deliberately a fair opponent — a strawman would
make the result meaningless.

Ten gates must clear (`SPEC.md` §5.2), covering faithfulness, coverage, grounding, stability,
latency and human utility. **Passing them is not sufficient.** The agent must also beat
map-reduce at something map-reduce cannot do by construction:

1. **Revision** — a later chunk correcting an earlier one. Independent map calls share no
   state, so they cannot.
2. **Long-meeting coherence** — a single through-line across 48 chunks.

A build that clears every gate while beating the baseline on neither has not earned its
complexity, and the project ships the baseline and records the negative result.

## Status

Working end to end. Latency, faithfulness, retention, grounding and stability gates pass on
the held-out set. Two things are unresolved:

- **Coverage.** The model still records too little on long meetings — it abstains on nearly
  half of all chunks, which also makes it less faithful per claim.
- **Verification.** The central quality claim is unverified: no human review has been run,
  and every number so far is downstream of a model at every stage.

`SPEC.md` is the normative contract. Where code and spec disagree, the spec wins.
