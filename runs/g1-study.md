# How to pass G1 honestly — a diagnosis, 2026-08-31

**Summary: G1's failure is NOT a missing capability. `qwen-tools-v5` performs the reversal
correctly — `DROP` the stale point, `ADD` the revised one, update `ARC` — and still fails
both probe cases, for two separate and independently fixable reasons. One of them is a
defect I introduced into the training data myself.**

## What "honestly" has to mean here

`sft-dropv7` already "passed" G1 once, in the dishonest way: synthetic reversals were added
until the two gate cases went green, while the independent 11-scenario probe went 3/10 →
2/10. The gate moved; the capability did not. Any claim of an honest pass therefore needs:

1. The fix must be **a correction to something demonstrably wrong**, not the addition of
   probe-shaped data.
2. It must be **verified on the independent probe** (`data/reversals_probe`, 11 scenarios
   sharing no subject, key term, or outcome vocabulary with training), not only on G1's two
   cases.
3. It must **not regress** G2/G3 or the real-ASR check.

The disqualifying move would be generating more scenarios that look like `office_move` /
`budget_approval`. That is explicitly not what follows.

## Diagnosis: the reversal mechanics already work

`qwen-tools-v5` on both G1 cases, reading steps in full:

```
office_move   step0: add ["市政大樓搬遷案同意，B 棟為最佳方案"]
              step1: drop ["市政大樓搬遷"], add ["市政大樓搬遷案改為撤回"],
                     arc "會議稍早同意的市政大樓搬遷案已改為撤回，需重新辦理。"
budget_approval step0: add ["下一季行銷預算案核准，兩百萬"]
              step1: drop ["下一季行銷"], add ["下一季行銷預算改為重新提案"], ...
```

Both perform the full revision correctly. `DROP` fires, the superseded point is removed,
a replacement is added, the arc is updated. **This is not a model that cannot revise.**

## Failure mode A: lossy revision — the replacement discards the identifying detail

`office_move`'s prose correctly says 撤回. It fails `subject_terms`, which requires the
building — 「B 棟」 or 「B 樓」. And 「B 棟」 *was* in memory, at step 0
(`市政大樓搬遷案同意，B 棟為最佳方案`). Step 1 dropped that point, and the replacement
(`市政大樓搬遷案改為撤回`) did not carry the building forward. The detail was not missed in
reading; it was **discarded during revision**.

**Root cause — `tools/gen_reversals.py`, and it is mine.** The two gold templates are
asymmetric:

```python
early_point = f"{sc.subject}{sc.early}，{sc.key_term}"   # carries the key term
late_point  = f"{sc.subject}改為{sc.late}"                # does NOT
```

Earlier tonight `late_point` was `f"{sc.subject}改為{sc.late}，{sc.reason}"[:24]`, which was
truncating mid-word and producing targets the harness refused as `point too long`. Removing
the `reason` fixed the truncation and simultaneously left the replacement carrying no
distinguishing content beyond the subject. Every one of the 68 reversal training samples
therefore teaches: *when you revise, write `SUBJECT改為OUTCOME` and drop the detail.*

**Why fixing this is a correction and not test-fitting.** The principle it restores —
*a replacement point must be self-contained* — follows from the architecture, not from the
probe: `DROP` permanently removes the superseded point, no conversation history crosses
steps (SPEC §4.1), so the replacement is the only surviving record. If it is not
self-contained, the information is gone.

And the model already knows this everywhere it was not taught otherwise. Across the 20 real
zh-TW ASR meetings, every `DROP`+`ADD` revision **expands** detail rather than shedding it:

```
ivod-17664: DROP «建議將聽力»     -> ADD «建議將聽力不良與心理健康納入國教健康與體育課程»
ivod-17695: DROP «…修正案已透過»  -> ADD «…修正案經宣讀後透過»
ivod-17701: DROP «委員要求防止»   -> ADD «委員要求防止故意或疏忽導致證據毀滅»
```

Correct, self-contained replacements — on real meetings, with no reversal training
involved. The lossy pattern appears *only* where the template taught it. That asymmetry is
the evidence that this is a supervision defect rather than a capability gap.

Every proposed corrected `late_point` fits `POINT_TOKENS = 25` (largest: `busshelter`,
exactly 25), so the fix costs nothing at the cap.

## Failure mode B: the decision word loses to the follow-up action

`budget_approval` records `下一季行銷預算改為重新提案` — "changed to re-propose". The source
says 駁回 ("rejected") five times, including the formal outcome:

```
S1: 表決結果全數同意。原兩百萬行銷預算案正式駁回，請行銷部依照最新報價重新編列後再提會議審議。
```

駁回 is the decision; 重新提案 / 重新編列 is its consequence. The model recorded the
consequence. Note where each sits: 駁回 is mid-sentence, the re-proposal action is the
trailing clause — consistent with the within-chunk recency bias already measured and
recorded as trap 6 (points favour a chunk's tail, containment 0.221 head vs 0.322 tail).

This one is **less well understood than A** and should not be presented as solved. The gold
templates already use the literal decision word (`sc.late`), so the pool is not teaching the
substitution directly. Two candidate mechanisms, neither yet tested:

- Recency within the reversal sentence, per trap 6.
- The `ARC` template's trailing clause `…已改為{late}，需重新辦理。` primes a "re-do action"
  frame, and the model merges that action into the outcome slot.

## Plan

1. **Fix `late_point` to carry `key_term`** (failure mode A). Regenerate gold from the
   existing transcripts via `--rebuild-gold` — no regeneration needed, so the transcripts
   are untouched and the change is isolated to the targets.
2. **Retrain and measure on the INDEPENDENT probe first**, before looking at G1's two
   cases. If the independent probe does not move, the fix did not generalise and G1 going
   green would be pattern matching again.
3. **Re-run G2/G3 and the ASR gate** for regressions.
4. **Leave failure mode B open and stated** unless the fix happens to move it too. Do not
   claim a mechanism that has not been tested.

## What would falsify this

If the independent probe stays at ~2-3/11 while G1's two cases flip green, the fix is
test-fitting and should be recorded as another refuted attempt — the same verdict `dropv7`
earned. That check is the whole point and comes first.


---

# OUTCOME: the fix was mechanically correct and is REFUTED as a G1 fix

`qwen-tools-v6` = `v5`'s pool with the corrected `late_point` (carries `key_term`).
Everything else identical. **Do not retry this.**

## The fix did exactly what it was designed to do

Verified at the memory level on G1's `office_move` case:

```
v5  step1: add ["市政大樓搬遷案改為撤回"]                    <- detail discarded
v6  step1: add ["市政大樓搬遷案改為撤回，B 棟為最佳方案"]      <- detail preserved
```

The training defect was real and the correction landed. That part of the diagnosis holds.

## It did not move the gate, and it cost ASR curation

| | `v5` | `v6` |
|---|---|---|
| independent reversal probe (11) | 0/11 | 1/11 |
| real-ASR curated (20) | **17/20** | 15/20 |

1/11 vs 0/11 at n=11 is noise, not a fix. The ASR regression is outside noise and in the
wrong direction. **`v5` remains the recommended checkpoint.**

## Why it was insufficient — the loss map, measured

Tracing `key_term` through the pipeline on all 11 probe scenarios:

| stage | scenarios retaining `key_term` |
|---|---|
| reaches MEMORY at all | **5/11** |
| survives to PROSE | **2/11** |

The revision template was the *third* loss point, not the first. Two larger ones remain:

1. **The reading step never captures `key_term` in 6/11 scenarios.** The `early_point`
   template does include it, so training teaches it — the model simply does not reproduce
   it on these transcripts. Consistent with trap 6's within-chunk recency bias. NOT a
   token-cap problem: every probe scenario's point fits `POINT_TOKENS=25` (max 25 exactly),
   checked directly.
2. **Synthesis drops it for 3 of the 5 that do reach memory.** Same shape as the hedge bug
   fixed in `v5`: memory correct, prose lossy. `office_move` in `v6` holds
   `…改為撤回，B 棟為最佳方案` in memory and writes prose that paraphrases 「B 棟」 into a
   generic phrase about comparing options.

A `gen_hedge_synth.py`-style fix for loss point 2 is the obvious next candidate and is
NOT attempted here — with loss point 1 accounting for 6/11 on its own, fixing synthesis
alone cannot carry the gate, and this attempt has already demonstrated the cost of shipping
a partial fix.

## Standing verdict on G1

G1 remains **honestly unpassed**. Four separate attempts are now recorded as refuted:
synthetic reversals at scale (`dropv7`, pattern-matched the 2 gate cases while the
independent probe fell), prompt-side instruction at synthesis, prompt-side instruction at
the reading step, and this self-contained-revision correction. The measured loss map above
is the useful artifact — it says where the remaining work is, and that no single-point fix
will close it.
