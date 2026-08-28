# G4 wall-clock: projected, not measured (sft-dropv2)

**Result: 19.58 min against SPEC §7's 20.00 min ceiling — PASS by +25s (2.1% margin).**

Using a projection in place of a device run was **explicitly authorized by the user on
2026-08-28**. This file exists so the number is never later mistaken for a measurement:
`report.json`'s `G4_budget: PASS` was produced by passing `--wall-clock-minutes 19.58`,
a figure computed here, not read off the phone.

## Inputs

| quantity | value | provenance |
|---|---|---|
| reading step, thin memory (early meeting) | 71 s | **measured on-device**, Oppo Reno 7 5G, `-C 0xFF`, Q8_0, `-n 150` |
| reading step, saturated memory (late meeting) | 90 s | **measured on-device**, same config |
| reading steps per meeting | 14 | SPEC §4.1, after the Phase-0a en→zh-TW token ratio of 1.215 |
| synthesis call | 33 s at 219 chars | **measured on-device** |
| sft-dropv2 mean summary length | 317 chars | measured over the n=20 eval run |

## Method

Reading phase, trapezoidal across the meeting (memory fills as it progresses):

    (71 + 90) / 2 × 14 steps = 1127 s = 18.78 min

Synthesis, scaled linearly in output length from the measured point:

    33 s × 317 / 219 = 48 s

    total = (1127 + 48) / 60 = 19.58 min

## What is assumed, and how much it can move

1. **The reading phase is unchanged from the dropv1 measurement.** Sound: those calls run
   at `-n 150` and dropv1 already saturated that cap (its output ended mid-line at the
   budget), so per-step cost is cap-bound, not model-bound. Same architecture, parameter
   count and quantisation. Also matches the shipping config — reading steps carry no
   repetition penalty.
2. **Synthesis decode time is linear in output length.** Reasonable for autoregressive
   decode at fixed context. The 317-char figure is dropv2's real measured mean under the
   shipping `repeat_penalty=1.1`, so the repetition fix is already priced in — without it
   the mean was 353 chars and one summary reached 1,242.
3. **THE WEAK ASSUMPTION — no thermal throttling.** Every input above is an *isolated
   call* of 71–90 s. The gate is a *sustained ~20-minute* run on a passively cooled
   phone. `CLAUDE.md` records this exact hazard from Phase 0b: "a 30-second unthrottled
   burst is not a 20-minute sustained run." Sustained throttling is not modelled here at
   all, and at a 2.1% margin a throttle of only ~2% flips this gate to FAIL.

## Status

**The user decided on 2026-08-28 that no device run is required**; this projection
stands as G4's answer. Recorded plainly so the basis of the decision is not lost:

- G4 is carried as PASS on the arithmetic above, not on a phone measurement.
- The margin is 2.1%, and the one unmodelled effect (thermal throttling over a sustained
  20-minute run) acts in the *failing* direction. A ~2% sustained throttle would flip it.
- Everything else in the projection rests on real on-device measurements of the same
  model size and quantisation, so the reading phase — 96% of the total — is on firm
  ground. The uncertainty is concentrated in whether isolated calls generalise to a
  sustained run, not in the arithmetic.

The bundle remains staged should anyone want to close that gap later:
`/tmp/g4_dropv2_bundle.tar.gz` with `/tmp/g4_orchestrate_dropv2.sh`, run on
`training-machine` (which has the device on USB adb). It measures all three legs against
the real dropv2 GGUF, with `--repeat-penalty 1.1` on synthesis and `1.0` on the reading
steps to match the shipping config.
