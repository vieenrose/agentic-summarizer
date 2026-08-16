# agentic-summarizer

[![GitHub](https://img.shields.io/badge/source-github-181717?logo=github&logoColor=white)](https://github.com/vieenrose/agentic-summarizer)
[![HF main](https://img.shields.io/badge/HF-main%20model-yellow?logo=huggingface)](https://huggingface.co/Luigi/minicpm5-1b-cursor)
[![HF verifier](https://img.shields.io/badge/HF-verifier-yellow?logo=huggingface)](https://huggingface.co/Luigi/granite-4.0-350m-verifier)

Agentic meeting-transcript summarizer for a **sub-1B** small language model: a streaming
edit-protocol (**CURSOR**) that converts zh-TW and en meeting transcripts into structured,
timestamp-anchored meeting notes, on-device.

> **Status: two-specialist deployment, measured.** The main is a fine-tuned
> **MiniCPM5-1B** (p15d, ~688 MB Q4_K_M) and the critic is an on-device verifier
> **granite-4.0-350m** (Apache-2.0, ~215 MB, 97% agreement with a 20B judge). G1 PASS
> both languages; the verifier-gated deployment reaches ~0 inversions. Weights on HF:
> [`Luigi/minicpm5-1b-cursor`](https://huggingface.co/Luigi/minicpm5-1b-cursor) ·
> [`Luigi/granite-4.0-350m-verifier`](https://huggingface.co/Luigi/granite-4.0-350m-verifier).
> [`CLAUDE.md`](CLAUDE.md) is the normative protocol contract; [`RESULTS.md`](RESULTS.md)
> records every measured number; [`voxsum-integration.md`](voxsum-integration.md) is the
> integration note for the VoxSum maintainer.

## The idea: CURSOR, not map-reduce

Classic map-reduce summarization (independent per-window digests → merge → shrink) produces
locally-correct but globally-disconnected notes: it cannot say how a decision *evolved*. Free
ReAct-style tool loops were measured **unlearnable** at ≤1B (multi-turn state fails, tool
results overflow context). CURSOR is the middle path: the transcript is streamed; there is
exactly **one evolving NOTES state**, curated by the model; **no conversation history crosses
steps**.

```
per step i:
  harness → model:  SYS (~250 tok) + STATE (≤600 tok) + CHUNK_i (~2048 tok of raw lines)
  model → harness:  edit ops — ADD / UPD / DEL / CMP / NOP
  harness:          validate → apply → dedup/cap → advance cursor
end:                VERIFY / ANCHOR sweep (the verifier) → deterministic render
```

Because STATE is the entire memory, temporal integration becomes *revising a visible earlier
bullet* (`UPD`) rather than remembering a past tool result — the property that makes the
protocol learnable at this scale.

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

## The two-specialist architecture (final)

| role | model | size | license |
|---|---|---|---|
| **main (proposer)** | MiniCPM5-1B, fine-tuned (p15d) | ~688 MB Q4_K_M | apache-2.0 |
| **verifier (critic)** | granite-4.0-350m, fine-tuned (zh-augmented) | ~215 MB Q4_K_M | apache-2.0 |

The main streams the edit ops; the verifier gates every DECISIONS/ACTIONS op at application
time (in-stream) and re-verifies every final bullet against whole-transcript evidence (the
final VERIFY sweep). **The verifier is required**: the model alone measures 4/20 inversions
(above the bar); the verifier gate brings the deployment to ~0. The single-model multi-role
alternative was measured and rejected (the critic collapsed to 38% agreement).

Two deterministic guards complete the harness (zero model tokens):
- `promote_decision_summaries` — decision-shaped SUMMARY bullets are **moved** into DECISIONS
  (the measured zero-DECISIONS class on noisy zh meetings).
- `enforce_decision_chain` — opposing-polarity bullets on one subject across DECISIONS+SUMMARY
  keep the LATEST (the over-ADD failure). The harness owns the final word (spec §6).

## Results (T1, n=20, local judges, 3× majority)

| configuration | INVERT | FAITH | COVER | SYNTH |
|---|---|---|---|---|
| main p15d, model-only (raw) | 4/20 | 3.57 | 3.00 | 2.30 |
| main + verifier, in-stream + sweep | **0-2/20** | **4.43** | 2.85-3.80 | 2.35-3.40 |
| map-reduce baseline (Qwen3.5-9B) | 3/20 | 3.50 | 3.05 | 2.60 |

G1 capability screen: **PASS both languages** (with the guards). The verifier's agreement
with gpt-oss-20b: **97% en / 92% zh**. The VoxSum maintainer's real 62-minute ASR-noisy zh
meeting yields populated DECISIONS + ACTIONS sections (the coverage blocker, fixed).

## Measured negatives (kept for the record)

- **LFM2.5-350M** (the earlier student): abandoned for MiniCPM5-1B.
- **Thinking-enabled fine-tuning** (p18-think): measured net-negative at our data scale — the
  think→ops transition is unlearnable from ~55 reasoning traces (the literature needs ~140k),
  and the think block leaks into the no-think mode.
- **Multi-agent / multi-LoRA** (PLAN-multiagent.md, superseded): critic adapters reach 64-85%
  agreement — the base's generation bias resists low-rank correction.

## Known gaps

- **No transcript has a real clock** — clocks are synthesised at 150 wpm; FAITH-anchor
  measures self-consistency, not real-world anchor accuracy.
- **Contested zh-TW is under-measured** — the zh T1 tier is largely synthetic; one real
  ASR-noisy transcript is in the training set.
- **T2 (≥80k-token meetings)** remains blocked on corpus.
- **GT3 SYNTH** is a tie, not a win, at the best config.

## Repo layout

- `src/voxsum/` — the CURSOR harness (chunker, ops, state, guards, sweep, render)
- `eval/` — `run_arms.py` (the two arms + the deployment flags), `judge.py`, `screen.py` (G1)
- `train/` — `sft_unsloth.py` (full-FT and LoRA), `gen_traces.py` (teacher traces)
- `tools/` — data builders, the negative-harvest, the ASR-noise augmenter
- `data/` — transcripts, traces, SFT mixes
- `runs/` — checkpoints and GGUF exports (weights are HF-only — see `.githooks/pre-push`)
