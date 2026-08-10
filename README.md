# agentic-summarizer

Agentic meeting-transcript summarizer for a **sub-1B** small language model, targeting zh-TW and
en meetings of **≥80k tokens** and producing structured, timestamp-anchored meeting notes
on-device.

> **Status: implementation — trace regeneration in progress (full 80-meeting set), student
> not yet fine-tuned.** No ship gate has been decided. [`CLAUDE.md`](CLAUDE.md) is the normative
> contract; where it and the code disagree, the spec wins. [`PLAN.md`](PLAN.md) records locked
> decisions and amendments; [`RESULTS.md`](RESULTS.md) records every measured number with its
> caveats.

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

These supersede the spec where noted; see `PLAN.md` §0.

| | value |
|---|---|
| **student** | **`google/functiongemma-270m-it` — LOCKED**, no fallback rung. Supersedes the spec's Qwen3.5-0.8B. Its per-step output *is* a function call, so we stop fighting the base model's format prior |
| **teacher** | `gemma-4-31B-it` **NVFP4 + Q8_0 MTP** draft head, one whole model per GPU, **thinking ON** (not optional — see below) |
| **judge panel** | `openai/gpt-oss-20b` (FAITH/INVERT), `deepseek-ai/DeepSeek-V4-Flash-0731` (COVER/SYNTH), `Prism-ML/Ternary-Bonsai-27B` (second opinion) |
| **judges disqualified** | **every gemma judge**, enforced in code as `DISQUALIFIED` in `eval/judge.py` |
| **hardware** | 2× RTX 5090 32 GB (Blackwell sm_120), bf16 native |

**The spec's judge rule is inverted by the student swap.** `CLAUDE.md` §7 says "gemma-4 only,
never Qwen-family" — written when the student was Qwen. With a Gemma student *and* a Gemma
teacher, gemma judges are now the contaminated ones. The panel rule that actually holds is
**judge family ∉ {student, teacher}**. `gemma-3n-E4B-it` also answered SUPPORTED to all three
probe cases in 4 tokens, including a planted inversion — so it is disqualified on measurement,
not only on lineage.

## What has been measured

Every number below is in `RESULTS.md` with its full caveats. **None of these are student
results, and no ship gate has been decided.**

| finding | status |
|---|---|
| **Teacher agency is language-asymmetric** — with thinking OFF, zh-TW produced decision inversions in 4 of 5 runs while en revised correctly every time | the single most important finding; drives zh revision oversampling |
| Teacher screen (G1), thinking ON | 100% valid-op, 100% raw anchor, revises-not-appends |
| **GT4 prefill** | **1.12× — PASS** (needs no judge, so it stands) |
| First paired CURSOR vs map-reduce, **n=2** | SYNTH **+1.00** (2/0/0), COVER +0.50, FAITH-claim +0.31, FAITH-anchor −0.01, no inversions |
| GT1 / GT2 / GT3 | **not decided.** GT2/GT3 are **WITHHELD** at n=2 against the spec's n=20 |

**FAITH is only interpretable as a relative gate.** Human QMSum reference summaries score
17–20% supported against our arms' 47–58% — the metric penalises abstraction, so absolute
values are meaningless and only the paired delta is usable.

## Known gaps and open risks

- **No transcript anywhere has a real clock.** QMSum has speakers but no timestamps;
  MeetingBank has neither; VCSum is unobtainable on the Hub. All clocks are synthesised at
  150 wpm and labelled `authentic_clock: false`. **FAITH-anchor on this data measures
  self-consistency, not real-world anchor accuracy.**
- **Contested zh-TW is unmeasured** — the zh pool is largely monologic, and zh T2 is synthetic.
- **270M may not abstract.** GT3 (SYNTH ≥ +0.5) is the whole agency bet; a 270M model can
  plausibly learn to emit valid ops and copy anchors while producing near-verbatim extraction
  with no meeting-level arc.
- **Trace regeneration is complete** (resumed 2026-08-10, branch `pi-agent`). The
  judge-filter bug (evidence limited to the chunk horizon) is fixed; the full set now covers
  all **80** meetings in `data/transcripts/manifest.json`, judge-filtered with a **local**
  `gpt-oss-20b` judge (see RESULTS.md). `data/traces_v2/` is the current set;
  `data/traces_v1_brokenfilter/` is the superseded set, retained for before/after comparison.
  The partial run's 45-step slice (86.3% valid-op) is archived in `data/traces_v2_partial/`.
- **Eval tiers are carved** (`tools/carve_eval_sets.py`): t1 = 10 en QMSum (real) + 10 zh
  synthetic; micro = 3 en MeetingBank + 3 zh synthetic; train = 54. T2 remains blocked on
  corpus: the pool has no ≥80k-token meeting and en T2 must be real (needs audio).
- **zh T1 is synthetic**, not VCSum (unobtainable on the Hub) — labelled per §7.8.
- On the partial regenerated slice, aggregate **valid-op was 86.3%**, below GT1's 95% floor.
  The full-set number is the first thing to check when regeneration finishes.

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

Ship CURSOR only if GT2 or GT3 clears at equal inversions — otherwise ship the map-reduce
baseline and record agency-at-270M as a measured negative result.

On-device envelope: ≈785 MB (Q4_K_M, 4k KV), ~3.2 h per 80k-token transcript on RPi4-class
hardware. FunctionGemma-270M is ≈200 MB at Q4_K_M, leaving real headroom.

## Repository layout

```
src/voxsum/      harness — transcript/ops/state/guards/chunker/index/agent/baseline/render
eval/            judge panel, planted-inversion selftest, G1 screen, arms runner, reports
train/           gen_traces.py (teacher traces), build_sft.py, sft_unsloth.py
tools/           prepare_data.py, trace_report.py, transcribe_moss.py, serve_teacher*.sh
tests/           1261 tests
```

`transcript.py` (`parse_line`, `clock_to_sec`) is the primitive everything depends on; the
mm↔ss-inverted clock formula is a known past bug that corrupted evidence placement.

## Running

```bash
.venv/bin/python -m pytest tests/ -q                    # full suite
.venv/bin/python eval/judge_selftest.py                 # judge planted-inversion probe
.venv/bin/python eval/screen.py                         # G1 capability screen
.venv/bin/python tools/trace_report.py data/traces_v2/  # trace-set health
.venv/bin/python tools/carve_eval_sets.py               # expand synth pool + carve T1/micro
```

Trace generation needs a teacher endpoint (`tools/serve_teacher_dual.sh`) and
`TOGETHER_API_KEY` for the judge filter, which is **required, not optional** — only 53–58% of
the teacher's own bullets are judge-verifiable on real meetings, and the guards check protocol
rather than truth.

## Caveats that must accompany every reported number

zh T2 is synthetic and the zh pool is largely monologic, so contested-zh is unmeasured;
MeetingBank has no speaker labels; no clock is authentic; judge-noise floor is ±0.4–0.5;
n = 20 per tier; the fine-tune and eval distributions must match exactly, system prompt
included, or scores are not comparable.

## Full specification

See [`CLAUDE.md`](CLAUDE.md) for the normative transcript format (v1), NOTES format (v2), op
wire formats, guards, judge protocol, eval tiers, and efficiency budget.
