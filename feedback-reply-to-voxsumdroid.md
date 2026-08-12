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
