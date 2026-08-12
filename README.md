# agentic-summarizer

[![GitHub](https://img.shields.io/badge/source-github-181717?logo=github&logoColor=white)](https://github.com/vieenrose/agentic-summarizer)
[![HF en](https://img.shields.io/badge/HF-en%20model-yellow?logo=huggingface)](https://huggingface.co/Luigi/lfm2.5-350m-cursor-en)
[![HF zh-TW](https://img.shields.io/badge/HF-zh--TW%20model-yellow?logo=huggingface)](https://huggingface.co/Luigi/lfm2.5-350m-cursor-zh)

Agentic meeting-transcript summarizer for a **sub-1B** small language model, targeting zh-TW and
en meetings of **≥80k tokens** and producing structured, timestamp-anchored meeting notes
on-device.

> Source: **https://github.com/vieenrose/agentic-summarizer**
>
> **Status: student trained and measured.** The fine-tuned **LFM2.5-350M** (per-language en/zh
> composite) passes the G1 capability screen in both languages; on the T1 tier (n=20, paired)
> it reaches **0% inversions**, **FAITH-claim +1.05 over the map-reduce baseline**, SYNTH at
> the +0.5 gate and GT4 at 0.51× prefill. Published on the Hub:
> [`Luigi/lfm2.5-350m-cursor-en`](https://huggingface.co/Luigi/lfm2.5-350m-cursor-en) ·
> [`Luigi/lfm2.5-350m-cursor-zh`](https://huggingface.co/Luigi/lfm2.5-350m-cursor-zh).
> [`CLAUDE.md`](CLAUDE.md) is the normative contract; [`PLAN.md`](PLAN.md) records locked
> decisions; [`RESULTS.md`](RESULTS.md) records every measured number with its caveats.

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
protocol learnable at this scale. Per-step input is constant-size, so it fits the context
budget with no growth across ~40 chunks.

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

---

## Current configuration (locked)

These supersede the spec where noted; see `PLAN.md` §0–§0d.

| | value |
|---|---|
| **student** | **`LiquidAI/LFM2.5-350M` — fine-tuned, per-language pair** (en + zh-TW, ~215 MB each at Q4_K_M; ~430 MB composite). A 350M model holds one language's full protocol at a time (measured seesaw) — per-language students are the locked architecture (PLAN §0d) |
| **teacher** | `gemma-4-31B-it` **NVFP4 + Q8_0 MTP** draft head, one whole model per GPU, **thinking ON** |
| **judge panel (local)** | `gpt-oss-20b` (FAITH/INVERT, **3× majority** — the judge flips verdicts on identical input at temp 0, measured), `qwen3.6-35B` (COVER/SYNTH). Judge family ∉ {student, teacher} |
| **judges disqualified** | every gemma judge, enforced as `DISQUALIFIED` in `eval/judge.py` |
| **hardware** | 2× RTX 5090 32 GB (Blackwell sm_120), bf16 native |

**History that led here (all measured, in RESULTS.md):** FunctionGemma-270M was proven
incapable of the protocol's load-bearing computation — state-gated op selection — across 6
data configurations, full-FT and LoRA, unconfounded counterfactual twins and minimal probes
(measured negative result). Qwen3.5-0.8B passed the en screen but was heavier than wanted.
**LFM2.5-350M's linear attention solves the state-gating on the first run** at the same
scale — the architecture, not just size, was the wall.

## What has been measured

Every number is in `RESULTS.md` with its full caveats.

| finding | status |
|---|---|
| **270M (FunctionGemma): measured impossible** for state-gated op selection — 6 data configs, full-FT + LoRA, counterfactual twins, probes; Qwen 0.8B control passes | recorded negative result (PLAN §0c) |
| **LFM2.5-350M G1 capability screen** (decision chain, deadlines, anchors, trap), en + zh-TW | **PASS** (valid-op 100%) |
| **T1 tier (n=20, paired vs 9B map-reduce baseline)** | **INVERT 0/20** (baseline 3) · **FAITH-claim +1.05** (14/2/2, p=0.004) · FAITH-anchor +0.40 · SYNTH +0.50 · COVER +0.30 · **GT4 0.51×** |
| Trace set (train split) | valid-op 95.4%, raw anchor 100%, revision share 21.4% |
| VERIFY/ANCHOR sweep (spec §5.2, implemented this project) | turns 12/20 raw inversions into 0/20; 4 harness bugs fixed (retrieval, prefixes, prose-FIX, judge stochasticity) |

**FAITH is only interpretable as a relative gate.** Human QMSum reference summaries score
17–20% supported against our arms' 47–58% — the metric penalises abstraction, so absolute
values are meaningless and only the paired delta is usable.

## Known gaps and open risks

- **No transcript anywhere has a real clock.** All clocks are synthesised at 150 wpm and
  labelled `authentic_clock: false`. **FAITH-anchor measures self-consistency, not real-world
  anchor accuracy** — only audio (MOSS transcription) yields a real clock.
- **Contested zh-TW is unmeasured** — all zh training data is synthetic (VCSum unobtainable);
  the zh model's T1 numbers are synthetic-meeting numbers.
- **T2 (≥80k-token meetings) remains blocked on corpus** — the pool has no ≥80k meeting and
  en T2 must be real (needs audio collection).
- **GT3 SYNTH is at the threshold** (+0.50 point-wise; the conservative 1-SE bound sits
  below) — the sweep's strictness trades synthesis content for faithfulness.
- **Sweep costs coverage**: 5–30% of raw bullets are dropped per meeting (verified against
  the transcript — most drops are request-as-decision fabrications, correctly caught).
- **zh-TW evaluation set is synthetic** (labelled per §7.8).

## Ship gates

Judged evaluation only (no reference summaries). Paired per meeting; judge noise is ±0.4–0.5
per meeting, so a **per-meeting** Δ < 0.5 is a tie — this is *not* a band on the mean
(SE ≈ 0.12 at n=20).

| gate | requirement (vs map-reduce baseline) |
|---|---|
| GT1 learnability | valid-op rate ≥ 95%, NOP-collapse < 10% |
| GT2 faith | FAITH-claim ≥ baseline +0.3, **0% inversions** |
| GT3 synthesis | SYNTH ≥ baseline **+0.5** on T1 and T2 |
| GT4 efficiency | prefill ≤ +25% over baseline |

Ship CURSOR only if GT2 or GT3 clears at equal inversions. **Current T1 standing: GT2 clears
(+1.05 FAITH, 0/20 inversions); GT3 is exactly at +0.50; GT4 clears (0.51×).** The final ship
call awaits the T2 tier and a stable judge repeat; the 270M negative result is recorded and
the agency bet is carried by the LFM2.5-350M student.

On-device envelope: ≈785 MB (Q4_K_M, 4k KV). The per-language students are ≈215 MB each at
Q4_K_M (~430 MB composite), leaving headroom for a bigger sweep budget.

## Repository layout

```
src/voxsum/      harness — transcript/ops/state/guards/chunker/index/sweep/agent/baseline/render
eval/            judge panel, planted-inversion selftest, G1 screen, arms runner, reports
train/           gen_traces.py (teacher traces), build_sft.py, sft_unsloth.py
tools/           prepare_data.py, trace_report.py, carve_eval_sets.py, filter_traps.py,
                 build_sft_v3.py / build_sft_qwen.py / build_sft_phase2.py, serve_*.sh
tests/           1270 tests
```

`transcript.py` (`parse_line`, `clock_to_sec`) is the primitive everything depends on; the
mm↔ss-inverted clock formula is a known past bug that corrupted evidence placement.

## Running

```bash
.venv/bin/python -m pytest tests/ -q                    # full suite (1270 tests)
.venv/bin/python eval/judge_selftest.py                 # judge planted-inversion probe
.venv/bin/python eval/screen.py                         # G1 capability screen
.venv/bin/python tools/trace_report.py data/traces_v2/  # trace-set health
# T1 tier end-to-end (arms + VERIFY/ANCHOR sweep + judged report), per-language students:
STUDENT_URL=http://127.0.0.1:8093 STUDENT_URL_ZH=http://127.0.0.1:8094   FAITH=local:8090/gpt-oss-20b COVER=local:8091/qwen3.6-35b bash eval/run_tier.sh t1
```

Trace generation needs teacher endpoints (`tools/serve_teacher_dual.sh`) and a judge for the
filter — the project runs fully local (gpt-oss-20b judge; `TOGETHER_API_KEY` only for the
optional cloud panel).

## Caveats that must accompany every reported number

zh T2 is synthetic and the zh pool is largely monologic, so contested-zh is unmeasured;
MeetingBank has no speaker labels; no clock is authentic; judge-noise floor is ±0.4–0.5;
n = 20 per tier; the fine-tune and eval distributions must match exactly, system prompt
included, or scores are not comparable.

## Full specification

See [`CLAUDE.md`](CLAUDE.md) for the normative transcript format (v1), NOTES format (v2), op
wire formats, guards, judge protocol, eval tiers, and efficiency budget.
