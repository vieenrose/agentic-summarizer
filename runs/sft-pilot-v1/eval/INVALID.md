# INVALID — do not cite these numbers

**Every score in this directory was computed against a leaked evaluation set.**

Measured 2026-08-27: all 20 meetings in this eval corpus are present in the training
split that produced the checkpoint they were used to score. The train/valid split is
assigned by meeting (`supervision.sft.split_by_meeting`) and shifted when the SFT pool
was rebuilt with the short-chunk and synthesis-augmentation rows, so a corpus that had
been held out at the time it was created was, by the time these numbers were produced,
entirely inside training. Scoring a model on its own training data measures
memorisation, not generalisation.

These files are kept rather than deleted because the session that produced them is part
of the project's audit trail, and a silently-vanished result is harder to reason about
later than a clearly-marked bad one. They must not be compared against, or averaged
with, any other run.

**Valid replacements**, both built from the current held-out split with overlap against
train explicitly verified as zero before use:

| directory | checkpoint | notes |
|---|---|---|
| `runs/sft-synth-v1/eval2/` | `sft-synth-v1` | first run against a baseline that actually reduces (`eval/`, one level up, predates the hierarchical-reduce fix and is also unusable) |
| `runs/sft-dropv1/eval/` | `sft-dropv1` | current; G1 passes on this checkpoint |

Two further defects invalidated results produced before 2026-08-27, both fixed and
regression-tested; any number predating them is suspect for these reasons too:

- The map-reduce control arm silently degraded into map-*and-concatenate* when the
  reduce prompt overflowed context, emitting 3,695-token mean "summaries" against
  SPEC §3's 1,000-token cap. `baseline.run_map_reduce` now folds hierarchically.
- `metrics.stats` gated G2 as PASS on zero judge records, and gated G3 on effect size
  alone (passing a 12/8 split at p=0.50). Both now withhold or fail respectively.
