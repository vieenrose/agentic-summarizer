# INVALID — do not cite these numbers

Superseded by `../eval2/`. Two defects, both measured 2026-08-27:

1. **The control arm was not map-reduce.** With `--reduce-context-tokens 4096` the
   reduce call was skipped whenever its own prompt overflowed, falling back to
   concatenating the window summaries — 11 of 20 meetings, mean 3,695 tokens, max
   12,540, against SPEC §3's 1,000-token cap. SPEC §5.2 requires a *fair* opponent
   "because a strawman baseline makes the gates meaningless"; this was an accidental
   strawman, and it invalidated G3 in BOTH directions (the agent "won" ROUGE against
   invalid over-length output and "lost" coverage/density purely because raw
   concatenation is more extractive than any real summary).
   Fixed by `baseline.run_map_reduce`'s hierarchical fold.
2. **The gates over-reported.** G2 passed on zero judge records; G3 passed on effect
   size alone. Both fixed in `metrics.stats`.

The eval **corpus and references** in this directory are valid and are reused by
`../eval2/` and by `runs/sft-dropv1/eval/` — it is the scores that are not.
