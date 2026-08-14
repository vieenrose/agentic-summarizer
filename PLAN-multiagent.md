# PLAN — Multi-agent CURSOR with multiple LoRA adapters (one base, n roles)

**Status:** design · **Date:** 2026-08-14 · **Supersedes:** the single-model multi-role
SFT experiment (measured negative: critic collapsed to 38%, zh trap crossed) and the
two-model deployment (688 MB proposer + 215 MB granite critic).

## 0. Why this design now

Three measured facts motivate it:

1. **The two-specialist deployment works** (0/20 inversions, FAITH 4.54) but pays
   903 MB for two physical models and gives the critic only 350M capacity.
2. **Mixed-weight multi-role SFT fails** (38% critic, zh-trap crossing) — roles
   interfere inside shared weights.
3. **Adapter separation is the literature's answer** (S-LoRA 2311.03285, Punica,
   FastLibra 2505.03756, LoRA-MoE 2311.02684) and **llama.cpp already serves it**:
   `--lora a,b,c` + POST `/lora-adapters` for runtime switching without reloading
   the base (verified in our build).

Goal: **one MiniCPM5-1B base + n small role adapters**, every role excellent at its
own task, ~790 MB resident, one serving process, on-device.

## 1. Protocol review → role decomposition

The CURSOR loop (agent.py: `run_cursor`) is: chunk → render(SYS+STATE+CHUNK) →
model → parse ops → `apply_ops` (guards.py: anchor validation, temporal guard,
UPD→ADD fallback, dedup, caps) → advance. The final sweep (sweep.py) adds VERIFY
(claim-mode, whole-transcript evidence) and ANCHOR (candidate pick). The measured
weaknesses per stage: op-emission collapse on noisy zh (proposer), SUPPORTED-bias
and polarity misses (critic), stale-state pass-through in-stream (critic, structural:
±90s window can't see later reversals), stock-phrase repetition and thin synthesis
(summary), anchor misses (anchor picker).

Six model roles + one deterministic set fall out:

### A1 — Proposer (streamer)
- **Job:** read chunk+STATE, emit ADD/UPD/DEL/CMP/NOP/TITLE per the sys-v1 grammar.
- **Adapter:** none — the frozen p13 base IS this role.
- **Data:** p13 mix (2414+ samples) — frozen, never retrained in this plan.
- **Eval:** G1 screen, valid-op, raw T1, ops/chunk on real meetings.

### A2 — Decision-line highlighter (deterministic, NOT an adapter)
- **Job:** mark decision/commitment-bearing lines in the chunk rendering
  (zh: 通過/同意/決定/否決/將/會/確認/指派…; en: agree/approve/decide/reject/
  will/assign/commit…), so A1's attention is pulled to the content-dense lines on
  noisy input. Attacks the measured coverage collapse with zero model tokens.
- **Eval:** ops/chunk on the maintainer's real meeting (p13 baseline: 1.25 ops/chunk).

### A3 — Fact critic (in-stream)
- **Job:** per ADD/UPD touching DECISIONS/ACTIONS, judge the anchor neighbourhood
  (±90s): SUPPORTED/UNSUPPORTED; drop on UNSUPPORTED.
- **Adapter:** rank to be probed (8/16/32), trained on the verifier triples
  (claim+in-stream forms, class-balanced) + the 69 polarity-flip triples.
- **Eval:** 200-triple agreement vs gpt-oss (target ≥95%); raw T1 INVERT with A3 on.

### A4 — Contradiction critic (sweep)
- **Job:** final VERIFY per bullet against whole-transcript evidence (anchor
  neighbourhood ∪ lexical top-k): SUPPORTED/UNSUPPORTED/CONTRADICTED; drop/fix.
  The reversal clause is enforceable HERE (the maintainer's structural point).
- **Adapter:** rank to be probed; trained on the CONTRADICTED subset (84 judged) +
  the flips ×3 + the UNSUPPORTED/SUPPORTED balance.
- **Eval:** per-class recall on CONTRADICTED (the polarity-flip check), swept T1 INVERT.

### A5 — Anchor picker
- **Job:** given a bullet + ≤8 lexical candidates, pick the [m:ss] that states it,
  or NONE (→ deterministic matcher).
- **Adapter:** trained on the 2100 anchor triples.
- **Eval:** anchor-pick accuracy ≥90% (matcher fallback is the floor).

### A6 — Summary compiler (synthesis)
- **Job:** once at the end: rewrite SUMMARY+TITLE with the meeting arc
  (rejected→approved, the bottom line), deduplicate near-identical bullets
  (the stock-phrase loop), enforce the ≤5 cap. The SYNTH role.
- **Adapter:** trained on the teacher's SUMMARY/arc steps (88 arc samples + the
  real-meeting SUMMARY records + CMP targets).
- **Eval:** SYNTH vs baseline (target ≥ +0.5), repetition count on the real meeting.

### R1–R6 — Deterministic guards (no adapters, unchanged)
anchor validation · temporal guard · UPD→ADD fallback · dedup/caps · NOP-collapse +
coverage fallback. These stay code, exactly as today.

## 2. The multi-agent framework

```
per chunk:  [R6?] → [A2 highlight lines in chunk render] → A1 proposes ops
            → R-guards apply → A3 gates DECISIONS/ACTIONS ops → STATE advances
end:        A4 verifies every bullet (whole-transcript evidence)
            → A5 re-anchors bullets that need it
            → A6 compiles SUMMARY/TITLE arc → deterministic render
```

- One llama.cpp server, one base, adapters switched via POST `/lora-adapters`
  (`--lora-init-without-apply`). A3 runs per substantive op (switches amortized:
  apply A3 for the chunk's gate batch, then back to A1).
- Memory: base 688 MB + adapters (4 × ~20–60 MB rank 8–32) ≈ **770–800 MB**,
  one process, within the 2.05 GB device ceiling.
- Fallbacks: every role has a deterministic floor (A3/A4 off → sweep off; A5 → the
  matcher; A6 → the cap enforcement) so any adapter failure degrades gracefully.

## 3. Plan of execution

### Phase 0 — probes (no training)
1. **A2 highlighter** on the maintainer's real meeting with the frozen p13:
   measure ops/chunk + DECISIONS/ACTIONS non-empty. (This alone may clear their
   first bar.)
2. **Rank sweep for the critic** (A3/A4): LoRA ranks 8/16/32 on the p13 base from
   the verifier triples; 200-triple agreement each. Decides whether a low-rank
   adapter can overwrite the generation bias (the open capacity question).

### Phase 1 — adapter training (all data exist)
3. Train A3, A4, A5, A6 adapters on their own distributions (`--regime lora` in the
   existing SFT stack; each ~30–60 min on 2 GPUs). No mixed training.

### Phase 2 — integration
4. Harness: the role scheduler + `/lora-adapters` switching + the role prompts
   (A3/A4: FAITH protocol; A5: ANCHOR protocol; A6: new COMPILE prompt).
5. Per-role probes (agreement, recall, accuracy) before the end-to-end run.

### Phase 3 — end-to-end evaluation
6. G1 screen (both languages, the zh trap must hold — the base is frozen so it
   should), the real-meeting coverage, raw T1, swept T1, SYNTH/COVER — vs the
   two-model baseline (INVERT 0/20, FAITH 4.54, COVER 3.20, SYNTH 2.75).

### Phase 4 — RLVR fallback (only for roles that fail)
7. If any adapter underperforms (esp. A3/A4 on the generation-biased base): one
   GRPO run per failing role with verifiable rewards (verdict correctness / anchor
   match) and KL to the base — the literature's tool for capability addition
   without drift (2503.06639), with the KL weight as the knob.

### Phase 5 — ship
8. Publish the base + adapters (HF, apache-2.0 lineage), update the integration
   note, measure the on-device numbers.

## 4. Success criteria (vs today's deployment)

| metric | today (two models) | target (multi-agent LoRA) |
|---|---|---|
| resident memory | 903 MB | ≤ 800 MB, one process |
| INVERT (deployed) | 0/20 | 0/20 |
| FAITH-claim | 4.54 | ≥ 4.5 |
| critic agreement | 97% (350M) | ≥ 95% (1B adapter) |
| ops/chunk, real noisy zh | ~1.8 | ≥ 3 |
| DECISIONS + ACTIONS on the real meeting | ACTIONS only | both non-empty |
| SYNTH | 2.75 (tie) | ≥ baseline +0.5 |

## 5. Risks

- **Capacity:** low-rank adapters may not overwrite the base's generation bias for
  A3/A4 (the probe in Phase 0.2 answers this first).
- **Switching overhead:** per-chunk adapter applies on CPU — measure; if slow,
  batch A3 gate calls per chunk (the plan already batches).
- **Zh trap:** the base is frozen, so the trap cannot regress — the adapters only
  touch their role's layers.
- **A6 overreach:** the compiler must only rewrite SUMMARY/TITLE, never DECISIONS
  (faithfulness stays the guards' + A3/A4's job).
