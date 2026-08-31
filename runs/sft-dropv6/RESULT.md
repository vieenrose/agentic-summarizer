# sft-dropv6: `POSITION` clears all three G3 gates; G1 fails on a subject term

**Verdict: G2 and all three G3 gates PASS. G1 fails; G4 is still an unmeasured
projection. SPEC §5.2 is all-or-nothing, so the decision remains "ship the baseline" —
but dropv6 is one gate away, and that gate's cause is understood.**

| gate | dropv6 |
|---|---|
| G1 revision | **FAIL** — one missing subject term |
| G2 faithfulness | **PASS** — 3 vs 53 inversions (3.9% vs 8.9% per claim) |
| G3 rouge1 | **PASS** — +0.134, 16/20, p=0.012 |
| G3 rouge2 | **PASS** — +0.064, 18/20 |
| G3 rougeL | **PASS** — +0.082, 19/20 |
| G4 budget | **not measured on device** (dropv2's projection: 19.58 min, 2.1% margin) |

## What changed

One variable versus `sft-dropv5`: `PROMPT_VERSION` `sys-v1` -> `sys-v2`, adding a
`POSITION: 第 N 段，共 M 段` line to the reading-step prompt. Same pool (4,604 samples,
NOP 34.9%), same base, same eval set.

The reason is measured, not stylistic. dropv4 and dropv5 each added genuine long-meeting
supervision and each fixed long meetings while breaking short ones, regardless of NOP
mix. `build_step_prompt` carried no step index and no chunk count, so position-dependent
behaviour could not be *conditioned* — only absorbed into the global policy.

## Result (n=20, same eval set and pinned config as every prior gate run)

| metric | dropv2 | dropv5 | **dropv6** |
|---|---|---|---|
| rouge1 | 14/20, p=0.115 | 14/20, p=0.115 | **16/20, p=0.012, +0.134** |
| rouge2 | 19/20 | 18/20 | 18/20, +0.064 |
| rougeL | 19/20 | 19/20 | **19/20, +0.082** |

| slice | dropv2 | dropv5 | dropv6 |
|---|---|---|---|
| long, >= 400 lines (n=9) | 4/9, +0.012 | 9/9, +0.182 | **9/9, +0.236** |
| short, < 400 lines (n=11) | 10/11, +0.091 | 5/11, +0.010 | 7/11, +0.052 |

The position signal did what it was predicted to do: long meetings held at 9/9 while
short meetings recovered 5/11 -> 7/11. Short meetings are still below dropv2's 10/11, so
the trade is reduced, not eliminated.

## G1: FAIL — and NOT for the reason first recorded

`runs/sft-dropv6/g1_report.json`. `office_move_reversal` FAIL, `budget_approval_reversal`
PASS.

**A first diagnosis in this session called the `office_move` output hallucination**, on
the grounds that it discusses library procurement, park renovation, bus routes and refuse
collection while the case is an office relocation. **That was wrong, and the correction
matters.** Those topics are all genuinely in the probe transcript — it is a multi-item
council agenda and the probe plants competing items deliberately. Nothing was fabricated.

`tools/measure_grounding.py` is what settled it: character-trigram containment of each
emitted point in its own chunk.

| domain | n points | mean | median | frac < 0.3 |
|---|---|---|---|---|
| MeetingBank (`data/eval20_zh`) | 380 | 0.416 | 0.394 | 0.329 |
| **probe** | 33 | **0.465** | 0.429 | 0.152 |
| LY zh-TW ASR | 14 | 0.352 | 0.267 | 0.571 |

The probe scores *higher* than real in-domain material. That is not what fabrication
looks like.

**The real failure is salience.** dropv6 states the reversal correctly — the summary ends
「以撤回先前已通過但未完成搬遷的…相關搬遷案」, carrying both 搬遷 and 撤回. It fails one
check: `subject_terms` also requires the building, and neither 「B 棟」 nor 「B 樓」 appears,
even though 「B 棟」 occurs **19 times** in the transcript, tied with 搬遷 as the dominant
topic. The model spends the summary on the minor agenda items and compresses the main one
into a clause stripped of its identifying detail.

This is adjacent to trap 5 (a false FAIL from surface-form matching) but is judged NOT a
false FAIL: the gate asks the summary to identify *what* was reversed, and "some
relocation case" does not. `states_earlier_as_current` is False for every checkpoint, so
no model reports a stale decision as current — the defect is dropped detail, not
incorrect state.

## The prior G1 record was not reproducible

`CLAUDE.md` recorded "G1 revision PASS" for `sft-dropv2` and a 6-of-7 gate count. No probe
artifact exists for that run: `g_report_final.json`'s `g1_passed: true` came from the
`--g1-passed` flag being asserted on the report command line. Re-run today under a
recorded configuration, **no checkpoint passes G1**:

| checkpoint | prompt | office_move | budget_approval | G1 |
|---|---|---|---|---|
| dropv2 | sys-v1 | FAIL | FAIL | FAIL |
| dropv5 | sys-v1 | FAIL | PASS | FAIL |
| dropv6 | sys-v2 | FAIL | PASS | FAIL |

Each graded on the prompt it was trained for. **dropv2 is 5 of 7, not 6 of 7, and is the
worst of the three on this gate.** `tools/run_probe.py` now records prose, per-step raw
ops, and every generation knob, so a G1 claim always has an artifact behind it — trap 4
showed the prompt cache alone changes generation, making an unconfigured verdict
uninterpretable.

## Standing

dropv6 is the strongest checkpoint measured on quality: it is the only build to clear all
three G3 gates, and the only one to fix long meetings without surrendering the sign test.
It does not ship, because G1 fails and §5.2 is all-or-nothing.

## G2 faithfulness: PASS (measured 2026-08-28)

`runs/sft-dropv6/g2_summary.json`, judge `local:8090/gpt-oss-20b` (third family — neither
Qwen, which authored the references, nor Gemma, which translated the corpus), 3 votes.

| arm | inversions | claims | inversions / claim |
|---|---|---|---|
| **agent** | **3** | 77 | **3.9%** |
| baseline | 53 | 598 | 8.9% |

Paired on **14 of 20** meetings; agent has fewer inversions on 7, ties 6, is worse on 1.
Better than dropv2 both absolutely and per claim (8 inversions, 5.7%).

**The 6 unscored baseline cases are the caveat, and it is larger here than for dropv2.**
All 6 failed with "empty content from judge after retry", and their summaries have a
median length of **6,638 characters against 710** for the ones that scored — the judge
exhausts on long inputs, so the exclusions are systematically the baseline's longest
outputs. They are excluded from BOTH arms (the comparison is paired), so this is not a
silent advantage, but the surviving baseline sample is not a random half of it.

Note also the claim-count asymmetry: 598 baseline claims against 77 agent claims on the
same 14 meetings. The baseline asserts ~8x more, which is why the per-claim rate is the
fairer of the two readings and both are given.

**Open, in order of leverage:**

1. G1's `office_move` subject term. The reversal is already stated; the missing element is
   one identifying noun. **Not chunk-size-bound** — see `runs/chunk1500/RESULT.md`; both
   smaller chunks and a coverage instruction were tested and neither fixes it end-to-end.
2. G4 — still dropv2's authorized projection, never measured on the phone.
4. LY ASR grounding is the weakest of the three domains (0.352 mean, 57% of points below
   0.3) on a small sample (n=14). Worth a larger measurement before trusting deployment
   behaviour on real ASR.

## Caveats

- Four builds have now been read against the same 20 held-out meetings. That is real
  multiple-comparisons exposure; `sys-v2` was motivated by a measured mechanism rather
  than a sweep, but the 16/20 should be read with the search history in view.
- `measure_grounding.py` scores each chunk against a FRESH empty memory, so it isolates
  chunk -> point grounding and does not reproduce the accumulating-memory loop.
- Containment cannot prove a point is invented: a correct paraphrase also scores below
  1.0, and a false statement assembled from the chunk's own vocabulary scores high. Read
  the per-domain gap, not the absolute value.
