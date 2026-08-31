# Held-out confirmation: dropv6's quality advantage is real, on 40 unseen meetings

**Verdict: 6 of 7 gates pass. G1 is the sole blocker. SPEC §5.2 is all-or-nothing, so the
decision remains "ship the baseline".**

| gate | result |
|---|---|
| G1 revision | **FAIL** — one missing subject term (`runs/sft-dropv6/g1_report.json`) |
| G2 faithfulness | PASS — 3 vs 53 inversions, 3.9% vs 8.9% per claim (on `eval20`) |
| G3 rouge1 | **PASS** — 29/11, +0.077, LB +0.053, p=0.006 |
| G3 rouge2 | **PASS** — 31/9, +0.043, LB +0.031, p=0.001 |
| G3 rougeL | **PASS** — 33/7, +0.053, LB +0.045, p=0.000 |
| G4 budget | PASS — 19.58 min projected, **user-authorized, never measured on device** |

## Why this run exists

`data/eval20_zh` had been read six times (dropv2, dropv4, dropv5, dropv6, chunk-1500, the
G1 probes) and `PROMPT_VERSION sys-v2` was chosen partly on its evidence. Every number
against it carried that search history, and SPEC §9's Phase-1 split reserved exactly one
eval slice, which the project had spent.

This set is **40 meetings drawn seeded at random from the 1,000 annotated MeetingBank
meetings present in no training pool and no prior measurement** (`tools/build_heldout.py`,
seed 20260828). Not stratified to resemble the old set: meeting length is the variable the
last four builds were tuned around, so anchoring on it would defeat the purpose.

Built tonight through the full SPEC §2.2 pipeline — import → translate (TranslateGemma-27B)
→ translate 232 gold items → compose references (Qwen3.8-27B) → `prose.finalize`.

Corpus integrity: **0 line-count mismatches, 0 UNK speakers, 0 simplified characters**,
CJK ratio median 0.847 / min 0.767, 0 items falling back to English, 40/40 references
within budget.

## Result

The advantage holds out-of-sample, with the gain concentrated exactly where Phase 4 aimed:

| slice | wins/losses | mean delta |
|---|---|---|
| long, >= 400 lines (n=10) | **9 / 1** | **+0.202** |
| short, < 400 (n=30) | 20 / 10 | +0.035 |

Effects are smaller than the reused set's (+0.077 vs +0.134 on rouge1) — the expected cost
of removing multiple-comparisons optimism — but direction and significance hold, and n=40
tightens every bound. **The long-meeting fix generalizes to meetings the model has never
seen**, which is the strongest evidence that Phase 4 worked rather than fitting `eval20`.

The agent also produces summaries **8x shorter** than the baseline: mean 320 chars vs
2,492.

`--skip-failed-steps` was on (the robustness change landed the same night), and **all 40
meetings paired** despite two llama.cpp 500s (trap 3) — the failure mode that previously
cost `n=20 -> 19` and withheld every gate.

## Two bugs caught in this run, both of which would have produced wrong headlines

1. **11 of 40 composed references came back EMPTY.** Qwen3.8 emits reasoning by default and
   returned empty assistant content for those agendas. `build_heldout_refs.py` reported
   "0 flagged" because it checked `over_budget` and `lang_flags` and never emptiness. A
   blank reference scores 0.0 against every candidate, so both arms tied at zero, the pair
   became a silent no-op, and the mean delta was diluted while `n` still counted it. Fixed
   with `enable_thinking: False`, a retry, and a hard refusal to store a blank reference.
2. **A reported rouge1 of +0.156 was an artifact of reading half-written files.** That
   report ran while the baseline was still re-scoring, so new agent scores were compared
   against stale baseline ones. Caught only because the gate line and the comparison line
   disagreed. The true figure is +0.077.

## Caveats that belong with these numbers

- **G4 is a projection the user authorized, not a measurement.** 19.58 min against a 20.00
  ceiling is a 2.1% margin; sustained thermal throttling on a passively cooled phone is
  unmodelled and acts in the failing direction. It is the only gate in the set with no
  measurement behind it.
- **G2 was measured on `eval20`, not here**, and rests on 14 of 20 paired meetings — six
  baseline summaries (median 6,638 chars vs 710) exhausted the judge. Re-running G2 on this
  held-out set is the obvious next check.
- These references share the pilot's provenance (Qwen composing from gold item summaries),
  so this tests generalization to new **meetings**, not to a different reference style. Any
  judge used on this set must stay third-family for the same reason.
- One reference (`LongBeachCC_07052016`) carries a CJK-ratio flag at 0.63; inspection shows
  fluent zh-TW whose ratio is driven down by digits and program names, so it was kept.
- Coverage and density both favour the baseline here, as they consistently have. They are
  diagnostics, not gates (SPEC §5), and they measure length-correlated quantities against a
  system whose summaries are 8x shorter by design.
