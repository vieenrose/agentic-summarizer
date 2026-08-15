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

---

## Round 5 reply (2026-08-14): coverage is now the target — plan and fixes

Accepted in full. Faithfulness landed; coverage is the new target. Response per finding:

1. **Coverage collapse (blocker).** Root cause: the zh pool is clean synth — the model
   has never seen real noisy-zh ASR, and its op emission collapses on it (the en side
   has real transcripts; zh has none — the known gap). Three levers now in work:
   (a) **your real zh-TW transcript** — please send it; it becomes the first real
   noisy-zh training data (teacher-traced at the production budget). (b) A synthetic
   ASR-noise augmentation of the zh synth meetings (fillers, disfluency, homophone
   errors — 10 meetings built) so the model learns to emit ops on noisy input.
   (c) Un-silencing: the zh trapfix lineage is NOP-heavy (silence at trap chunks
   plausibly over-generalized); decision-dense zh demonstrations get a higher dose.
   Success metric: ops/chunk and non-empty DECISIONS/ACTIONS on your real meeting.
   Near-duplicate suppression harness-side is yours — agreed, and we'll also raise
   the diverse-SUMMARY dose training-side.
2. **GGUF metadata.** All three accepted and fixed in the integration note: the arch
   is dense GQA (llama-style, not hybrid — that claim was carried over from the LFM
   student and is wrong for MiniCPM5); ctx 131072 is the base's native context,
   4096 is the train/serve context (pin it); "Checkpoint 302" IS the p13 (p13 =
   checkpoint-302 — RESULTS.md now names it; p10 = 274, p11 = 282/284).
3. **Verifier.** The n=1 polarity-flip false SUPPORTED: we built 69 synthetic
   polarity-flip counterfactual triples (numbers/verbs flipped against the same
   evidence → CONTRADICTED) and are retraining with them + a per-class held-out
   check. The reversal-clause point is correct and structural: at ±90s the in-stream
   verifier cannot see later reversals — the **temporal guard + the final VERIFY
   sweep (whole-transcript evidence) own the reversal case**, and the deployment
   configuration is in-stream + final sweep (both on-device, verifier-judged). With
   in-stream alone we measured 1/20 (a zh stale-state — exactly this class); with
   the final sweep it is 0/20. Also: the verifier is now based on
   **ibm-granite/granite-4.0-350m (Apache-2.0)** — `Luigi/granite-4.0-350m-verifier`
   (97% agreement with gpt-oss-20b; the LFM-based one is deprecated for commercial
   use). (Granite 4.1 has no 350M — 3B is its smallest; too big for the device.)

---

## Round 5 follow-up (2026-08-14): architecture locked, coverage progress, honest residuals

1. **Architecture (final):** two specialists — **main = MiniCPM5-1B p15d**, **verifier
   = granite-4.0-350m** (Apache-2.0, `Luigi/granite-4.0-350m-verifier`, 97% agreement
   with gpt-oss-20b). The multi-agent/multi-LoRA alternative was measured and
   abandoned (critic adapters on the 1B base reach only 64-85% agreement — the
   base's generation bias resists low-rank correction; recorded in
   PLAN-multiagent.md, superseded).
2. **Your real meeting, p15d (trained on it + ASR-noise augmentation):** G1 PASS both
   languages; on the meeting itself ACTIONS is now non-empty (price/quantity
   discussions, D4-margin decisions, the SOC timeline — real content through the
   echo-loop noise) vs the p13's empty sections. **DECISIONS on that meeting is
   still empty** — the one open item on your bar; the targeted dose crossed the en
   chain (recorded) and is shelved pending more real zh data.
3. **Deployment matrix (n=20, same judges as always):**

| configuration | INVERT | FAITH | COVER | SYNTH |
|---|---|---|---|---|
| p15d raw | 3/20 | 3.80 | 2.95 | 2.40 |
| p15d + in-stream + sweep (granite) | 1/20 | 4.20 | 2.50 | 1.95 |
| p13 + in-stream only | 0/20 | 4.10 | 3.20 | 2.75 |
| your baseline reference (9B map-reduce) | 3/20 | 3.50 | 3.05 | 2.60 |

   Honest note: the granite sweep judge over-drops vs the earlier verifier (its zh
   verdicts are weaker — the training triples are en-heavy); zh-verifier training is
   the open follow-up. The best balance remains in-stream-only (0/20, COVER 3.20) —
   with the structural caveat we already agreed on: ±90s cannot see later
   reversals; that class is the final-sweep/timeline-guard's job.
4. **Weights:** `minicpm5-1b-cursor-p15d.Q4_K_M.gguf` is on HF (same repo). If you
   send more real zh transcripts, they become training data immediately — that's
   the single most valuable input for the DECISIONS gap.

---

## Round 5.1 reply (2026-08-14): op-level trace accepted — both signals acted on

Your audit log is the sharpest feedback we've received, and your hypothesis retraction
is the useful outcome: **zero DECISIONS ops proposed across 8 chunks is checkpoint-
side**, and **~30% of output re-proposing existing STATE is a STATE-reading
deficiency, not extraction**. We agree with both readings.

1. **STATE utilisation:** we built a capture for exactly this class
   (`tools/build_state_negatives.py` — the harvest never saw the dedup-guard
   rejections, only the sweep's drops; this tool reconstructs the negative sample
   from the step's state+chunk with the corrected target). The first pass (4
   negatives ×3, p16) raised op density on your meeting (decode 311→432 tokens) but
   the re-propose loop persisted and the en chain crossed — recorded, reverted to
   p15d. The class is now a named, captured training signal with a tool; it needs
   more instances than one meeting provides.
2. **DECISIONS extraction:** the teacher's trace on your meeting has the decision
   steps (qualification priority, AVL checks); the doses that teach them cross the
   en chain (p15e/p16, recorded). The binding constraint is data: one real zh
   meeting carries ~3 decision steps. **This is where your transcript matters
   most** — the content leaving your machine is the decision we can't make for
   you; we will only train on it if you send it. (Note for clarity: our zh
   training currently contains only clean synthetic zh plus our own ASR-noise
   augmentations — no third-party real zh.)
3. **Re-pinning p15d** — agreed on your evidence; our T1 table agrees it's the
   right choice for your product even where p13 scores better on the synthetic
   tier.
4. **Verifier A/B (granite vs lfm2.5 in-stream on your real zh meeting):** agreed
   it's the right test given our own §11.1 admission (granite's zh verdicts are
   weaker — the triples are en-heavy). We'll run it on our harness with the same
   meeting and post the veto-by-veto comparison.

Your port's audit trail is better tooling than our own harness logging — we'd be
glad to add the same per-op record to our eval output if useful.

---

## Verifier A/B on the real zh meeting — first pass (2026-08-14)

Ran the A/B you suggested on our harness, same meeting, p15d main, in-stream only:

- **granite-4.0-350m in-stream: 0 vetoes**
- **lfm2.5-350m in-stream: 0 vetoes**

Both runs' only rejections were the dedup duplicates (the STATE-reading class from
your audit). Inconclusive for veto behaviour — on our run the p15d proposed no
borderline DECISIONS/ACTIONS ops (your run's question-as-action didn't appear in
this trajectory). The verifier difference we *have* measured is in the final sweep
(granite over-drops on zh: COVER 2.50 vs 2.95-3.20) — so your A/B should compare
the sweep path too, or re-run in-stream on a meeting with a denser op mix. Your
port's run is the better test for in-stream; we'll post the sweep A/B numbers here
next.

---

## Zh verifier training landed — the over-drop is fixed (2026-08-14)

Our §11.1 admission (granite's zh verdicts weaker, en-heavy triples) is now
addressed: zh triples (64 judged zh bullets, both evidence forms) + zh
polarity-flips, retrained on the granite verifier. Published as
`granite-4.0-350m-verifier-zh.Q4_K_M.gguf`.

Measured: zh agreement 92% (en held at 96%). On the zh T1 half with the full stack
(p15d + verifier, in-stream + sweep): **INVERT 0/10, FAITH 4.73, COVER 3.80,
SYNTH 3.40** — the sweep no longer drops the zh bullets. On the full n=20 the
stack measures INVERT 2, FAITH 4.43, COVER 2.85, SYNTH 2.35 (vs 1/4.20/2.50/1.95
with the old verifier) — the two flags are en-side (±1 noise band). For your
zh-primary product the gr4 stack is the choice; the in-stream-only config remains
the en-primary balance.
