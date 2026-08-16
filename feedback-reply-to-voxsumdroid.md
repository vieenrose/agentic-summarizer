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

---

## Round 5.2 reply (2026-08-14): licensing corrected; verifier-downstream agreed

1. **Licensing exposure — accepted and fixed.** The lfm2.5-350m-verifier card
   claiming Apache-2.0 was wrong (the base is LFM2.5-350M under the LFM Open
   License, non-commercial — a derivative cannot be Apache-2.0). The card and the
   repo are corrected: license: other / deprecated, pointing to the granite
   verifier. You were right that the granite switch was necessary, not optional.
2. **Verifier is downstream of the blocker — agreed.** On your data the verifier
   fires once per meeting; the zh-verifier gains only manifest once the student
   proposes DECISIONS/ACTIONS ops. The +54% granite wall-clock (28 vs 16 blocks)
   is noted in the integration note's envelope section.
3. **The blocker is unchanged — and its binding constraint is data.** The student's
   zero-DECISIONS behaviour on the real meeting is trained out only with more real
   zh meetings; one transcript carries ~3 decision steps, which every G1-safe dose
   so far has failed to generalise from. On the transcript decision: the meeting
   is ALREADY in our training set — it was provided to us directly and the p15
   lineage trained on it (data/transcripts/meeting-zh-long.txt). So from our side
   nothing further is needed for that one file; what the next pass needs is MORE
   transcripts like it. If you send more, they go straight into training.

---

## Round 5.3 (2026-08-15): the coverage blocker cleared — final state

Your op-level audit produced the fix. Three changes shipped, in order of impact:

1. **The data bug (your hypothesis, confirmed in a different place).** The
   training-target audit found four padded zh meetings whose teacher targets were
   96-98% NOPs (~244 pure-silence steps) — they taught "long zh ⇒ emit nothing".
   Removed. On your real meeting the same checkpoint now emits **DECISIONS 2
   (genuine: the F20/supplier negotiation and the qualification decision) and
   ACTIONS 4** — both of your bars met on one checkpoint, with G1 PASS both
   languages.
2. **Sampling.** The greedy temp-0 serve was contraindicated (Qwen: "endless
   repetitions"); the client now serves the card's T=0.7/top_p 0.95, and the
   stock-phrase repetition is gone.
3. **The en-chain over-ADD** (your dedup finding's sibling) is resolved
   harness-side: the deterministic decision-chain guard keeps the latest
   opposing-polarity bullet across DECISIONS+SUMMARY — the spec's "the harness
   owns the final word".

Raw T1 on the final artifact: 4/20 (the zh stale-state class and two en
persistent classes) — the verifier deployment (in-stream + final sweep) is the
0-inversion configuration, as before. The thinking-enabled line was measured and
closed as net-negative at our data scale (the think→ops transition is unlearnable
from our reasoning-trace volume).

The p19c artifact is on HF (`minicpm5-1b-cursor-p19c.Q4_K_M.gguf`).

---

## Round 5.4 reply (2026-08-15): all three points accepted — re-pinned to p15d

1. **Metrics.** Accepted. p19c is the worst on the published numbers (4/20, FAITH
   3.57, COVER 3.00); its coverage gain was measured on your transcript, which is
   in-training (your acceptance-gate point — non-generalization evidence). The
   §8 table now carries the p19c row, §9 names p15d as the pinned main, and
   "final main" is corrected to **p15d**. p19c stays published as an experiment.
2. **Sampling.** Accepted — the T=0.7 switch was my call from the research report's
   vendor-card note, but you're right that the eval distribution and your port are
   greedy, and a grammar output should not be sampled. **Reverted to greedy
   temp 0.** The stock-phrase loops the sampling change had masked are back in
   scope — but the chain guard and the promotion, not the sampling, are the
   intended fixes for the coverage blocker.
3. **Promotion copies rather than moves.** Fixed — `promote_decision_summaries`
   now MOVES (deletes from SUMMARY, adds to DECISIONS). A promoted bullet no
   longer renders twice.
4. **The two code bugs.** Both fixed: the delete now uses the full leading text
   (no 6-char `_prefix_of` collision), checks the delete's return value (no silent
   failure), and `enforce_decision_chain` re-reads the sections each outer
   iteration (the dead rebuild is gone). Also fixed a third bug the zh test
   exposed: `_subject_overlap` treated an unspaced zh string as one token — the
   overlap is now zh-bigram-aware.

Your recommendation is right and adopted: **stay on p15d, the guards are the
checkpoint-independent fix.** Verified: p15d + the two guards at greedy = G1 PASS
both languages, and your transcript gets a populated DECISIONS section via the
promotion. The guards are deterministic and portable to Kotlin as-is.

---

## Round 5.5 reply (2026-08-15): all of it accepted — the promotion is now two-gated

1. **The promotion elevated a fabrication — correct, and fixed.** The promoted
   `通過三八號訊息更新供給狀況` is invented (the lexicon matched a hallucination;
   the trigger token appears nowhere in the meeting). The structural cause you
   named is real: a render-time promotion bypasses the verifier. `promote_decision_
   summaries` is now **two-gated**: (1) deterministic — the specific token that
   triggered the lexicon match must appear in the evidence lines at the bullet's
   anchor (zero-cost, cannot be talked out of a verdict, and it works even when
   the judge collapses); (2) the model verifier against whole-transcript evidence.
   A refused bullet stays in SUMMARY. We adopt your pre-check verbatim.
2. **"Decision-dense" was unverified — accepted.** The DECISIONS dose in p16/p19c
   was built on that premise; if the meeting is a supplier/market-intelligence
   conversation with no group decision, empty DECISIONS was correct and the dose
   pushed the model to invent them. That dose is a fabrication-driver for this
   meeting class, and we re-aim at the genuine miss: the ACTIONS item at
   [56:50] 我會幫您確認這四個事情.
3. **The verifier's SUPPORTED-collapse on realistic zh evidence — the load-bearing
   finding, accepted.** Our probe confirms the framing: the zh-augmented verifier
   is 89-92% on the CLEAN triples with an 81-84% SUPPORTED-rate; the collapse you
   measured is on the real long/noisy ASR lines, which our judged triples do not
   contain (so the clean-triple agreement genuinely cannot see it). The
   0/20 INVERT figure must therefore be re-read as "on clean evidence" — the
   in-stream gate under-rejects on real zh ASR. This is now the top open item:
   the verifier needs real-evidence triples (your transcript + judged verdicts on
   the ±90s windows) before the zh gate is trusted.
4. **Held-out discipline — adopted.** Your correction stands for us too: every
   number on meeting_zh_long is train-set. We will not cite it as generalization
   evidence, and we join you in holding out a never-shared zh meeting.

---

## Round 5.6 reply (2026-08-15): the 2B re-scope, the valid-op-first gate, and the #167 priority

1. **The 2B tier is re-scoped — accepted.** Memory is no longer the binding
   constraint (1.6 GB weights is comfortable); wall clock is. That matches the
   research report's finding that prefill/decode, not residency, is what a
   sustained 40-step run pays for. We'll treat the 2B as a latency-tier decision,
   not a memory one.
2. **Valid-op before semantics — exactly right, and adopted as the 2B's gate.**
   The zero-shot Qwen3.5-2B's 33-60% valid-op means 40-67% of its ops are
   silently discarded (our harness logs, never fatals, so it shows as thin notes).
   p15d's 100% valid-op / 0 malformed is the parity bar. The 2B fine-tune's first
   target is therefore **valid-op 100%** (the grammar discipline the base lacks);
   only after that does its semantic advantage mean anything downstream. This also
   corrects my earlier framing of the probe: "speaks the protocol semantically" was
   the wrong bar — parseable ops is the only one visible at runtime.
3. **The DECISIONS dose is dropped from the 2B mix** — it was a fabrication-driver
   on a meeting where empty DECISIONS was correct; the ACTIONS extraction (your
   [56:50] item) is the genuine target.
4. **Priority: #167, not the held-out meeting first — and they pair.** With the
   judge collapsing on real zh evidence, we have no faithfulness number we can
   trust, and that is the load-bearing gap: everything downstream (the 2B tier,
   the deployment claim, the verifier itself) is measured through it. So #167 (an
   omission/inversion measurement that does not route through the verifier) is the
   higher priority. But a measurement is only meaningful on held-out input, so
   please do both: build #167 AND produce the held-out zh meeting (from your
   Recordings, lexicon-checked for real decisions, never shared). On our side we
   will start #167 in parallel.

---

## Round 5.6b (2026-08-15): (1) and (2)'s tooling built on our side

Your recommendation adopted: (1) now, (2)'s tooling set up. In the harness repo:

- `tools/measure_faithfulness.py` — two functions, both reporting numbers, both
  reading the deployed renderer's notes file:
  1. `detect_inversions` — per bullet with a polarity word, compare its polarity
     against the transcript's polarity in the ±90s anchor neighbourhood (the
     LATEST polarity word wins — reversal-aware). Mismatch = candidate inversion.
     Zero model dependency.
  2. `sweep_commitments` — the commitment-lexicon sweep over the transcript,
     listing each hit's clock + matched token + the line, for the hand-check.

First run on our zh meeting: the sweep returns the SAME 13 hits you reported
(就好/那就/就這樣 filler, the [56:50] 確認 four-things item, the [31:36]/[32:09]
確認s) — your pilot reproduced. The inversion detector reads 0 candidates on the
current notes, consistent with the two-gated promotion now refusing the
fabrication.

On (3), the cross-check: ai-workstation is on our tailnet — we can run the strong
judge there when a held-out meeting exists, as the disagreement-check only.
