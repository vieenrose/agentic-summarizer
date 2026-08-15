# Next steps (2026-08-14) — 2-agent deployment locked

Architecture: main = MiniCPM5-1B p15d (published), verifier = granite-4.0-350m
(published, 97%, Apache-2.0). Deployment config: in-stream verification (+ final
sweep where the stale-state class matters; see the tradeoff table in RESULTS.md).

1. **zh verifier training** — the granite verifier's zh verdicts are weak (en-heavy
   triples); build zh triples (the judged zh meetings + zh flips) and re-measure the
   sweep's COVER/SYNTH cost.
2. **DECISIONS on the maintainer's real meeting** — needs more real zh data (their
   next transcript) or a lighter DECISIONS dose that doesn't cross the en chain.
3. **Stale-state guard** — extend the timeline guard (or keep the final sweep) for
   the reversal class the ±90s window cannot see.
4. **SYNTH** — the writer-role work was part of the superseded plan; revisit only if
   a cheap path appears (the in-stream-only config's 2.75 stands).
5. On-device numbers (latency/thermals) — the maintainer's foreground-service
   measurements pending on their side.
