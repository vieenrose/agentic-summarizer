Thanks for the careful review — this is exactly the kind of evaluation we needed, and both points are now addressed.

1. Sweep-free numbers for the phase-2 checkpoint — measured.

You were right that this was unmeasured. We ran the T1 tier (n=20, same meetings, same judges, same protocol) against the final phase-2 checkpoint with no sweep:

    pre-phase-2 model, raw (the only number previously available): 12/20 (60%)
    phase-2 model, raw — measured: 4/20 (20%)
    phase-2 model, + VERIFY/ANCHOR sweep: 0/20

So phase-2's real-transcript adaptation did move the model's own rate, not just the swept combination — the model-side lever works. But you're right that 4/20 still fails the 0% requirement without a judge, so any on-device planning number should be 4/20, not 0/20. We characterized the four survivors: one genuine precision inversion (the meeting said "old-fashioned", the note said "fashionable"), one zh-TW stale-state bullet (rejection retained after the later approval), one truncated fabrication, and one borderline judge call where the supporting line is actually in the evidence.

On sweep-into-training: agreed, and it's now the stated next step. DROP/FIX outcomes are currently judge-time corrections only — they are not yet harvested as SFT signal. We've made raw INVERT (not swept INVERT) the target metric for the next iteration, and the concrete plan is to feed the sweep's contradiction/fabrication cases back into training so the model itself holds the faithfulness gains. We'll re-measure the raw rate after that pass and report it.

2. GGUFs — published. lfm2.5-350m-cursor-en.Q4_K_M.gguf (~215 MB) and lfm2.5-350m-cursor-zh.Q4_K_M.gguf (~229 MB) are now in both HF repos, so the llama-server command in the model card works as written.

Verdict change: not on the current numbers — raw 4/20 still doesn't clear your on-device bar — but the gap to re-evaluation is now precisely scoped: raw INVERT ≈ swept INVERT. That's the single number we'll be chasing, and we'll report it here when the sweep-feedback training pass lands. The CURSOR harness design itself remains something we'd be glad to share in detail if useful for your port.

---

## Round 3 (2026-08-13): all three findings accepted; the on-device judge objection is now answered

Your three points were correct, and each is now addressed:

1. **Raw from the wrong checkpoint — accepted and corrected.** The ship table's raw
   2/20 came from pass p6; the published artifact is p10/checkpoint-274. We measured
   the shipped checkpoint itself: **p10 raw = 3/20 (15%)**; the later passes measure
   p11-e1 4/20, p12 4/20. The integration note now carries the corrected table and
   states explicitly that the earlier quote was wrong.
2. **"A phone has no 20B judge" — this is now obsolete.** We fine-tuned an on-device
   verifier: `Luigi/lfm2.5-350m-verifier` (~215 MB Q4_K_M), trained on 2,644 judged
   (bullet, evidence, verdict) triples harvested from the pipeline's own runs,
   class-balanced (an unbalanced fine-tune collapses to SUPPORTED — measured at 5%
   agreement on an unadapted 0.8B verifier, 55% on the summarizer itself). The
   verifier measures **96% agreement with gpt-oss-20b** on 200 held-out triples, and
   the full pipeline with the verifier as the sweep judge measures **INVERT 0/20,
   FAITH-claim 4.54** (baseline 3.50) — equal to the 20B-judge sweep on inversions.
   Deployment: student 688 MB + verifier 215 MB ≈ 900 MB resident, ~750 MB with
   sequential loading, within the device's 2.05 GB ceiling.
3. **n=19/20 — fixed.** Root cause: the meeting-list file's last line had no trailing
   newline, so `cat en zh` glued the last en entry to the first zh entry and the loop
   skipped qmsum-e75802cbf8d3 in every run. The list is fixed and every run is now
   n=20 (the earlier tallies' /19 denominators were real, not rounded).

What still does NOT clear, stated plainly: the model-only raw rate (no sweep of any
kind) is 15-20% across passes, above your 6.2% on-device bar. If your bar is "no
second model on-device at all", this still fails — that raw rate is the active
training target, and it has moved 60% → 15-20% but not below the bar. If your bar is
"no 20B judge on-device", the verifier answers it: the sweep, and therefore the 0/20
inversion rate, is now fully on-device.
