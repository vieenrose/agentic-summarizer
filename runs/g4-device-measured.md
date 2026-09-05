# G4 measured on the real Oppo Reno 7 — 2026-08-31

**First actual device measurement of the v1.0 architecture.** Every prior G4 number in this
project was a projection. This replaces them.

**Verdict: PASSES in steady state (19.4 min vs a 20.00 min ceiling, 3% margin) but a single
observed transient stall would put a meeting at 21.6 min — OVER budget.** Thermal
throttling, the risk flagged all along, turns out not to be the problem. Variance is.

## How the device was finally reached

`training-machine` (100.122.78.108) is online in Tailscale but port 22 times out from this
workstation — the blocker recorded for weeks. **The Raspberry Pi can reach it**, so the path
is `workstation -> ProxyJump raspberrypi -> training-machine -> adb -> phone`. The SSH user
is `user` (not `luigi`; Tailscale's ACL rejects that name).

Device confirmed as the reference hardware before trusting any number:

```
Features: fp asimd ... asimddp        <- dotprod present, no i8mm, no sve
nproc:    8
```

Note `asimddp` is ARM's name for the dotprod feature. An earlier check in this session
grepped for the literal string `dotprod`, found nothing, and wrongly concluded the reachable
ARM hosts lacked it. The Pi genuinely lacks it; the Reno 7 has it.

The on-device llama.cpp (`15586e2d7`) already lists `qwen35` among its architectures, so no
cross-compile was needed — checked with `strings` rather than assumed.

## Measured: `qwen-tools-v5`, Q8_0, `-C 0xFF` (all 8 cores)

```
pp3400  59.23 t/s      prefill, one step's 3,400-token prompt
tg190   12.40 t/s      decode, one step's ~190-token tool call
```

Against the measured step profile (15.2 steps/meeting, 340-char synthesis):

| | |
|---|---|
| prefill per step | 57.4 s |
| decode per step | 15.3 s |
| **per reading step** | **72.7 s** |
| synthesis | 37.5 s |
| **full meeting** | **19.0 min** (ceiling 20.00) |

## CORRECTION, 2026-09-05: decode was benchmarked at the WRONG DEPTH, and G4 fails

**The 19.0 min above is optimistic. The honest nominal figure is 20.4 min, over the 20.00
ceiling.** The error is entirely in the decode term and it is a measurement-shape error,
not a modelling one.

`tg190` with no `-d` measures decode starting from an EMPTY cache, averaging over depths
0-190. A reading step does not decode there. Its prompt is SYS + MEMORY + CHUNK ≈ 3,400
tokens, so every one of its ~190 decoded tokens attends over 3,400+ tokens of KV. Decode on
this device is strongly depth-sensitive, which the original run never swept:

```
tg190 @ d0     12.57 t/s
tg190 @ d1000  11.83
tg190 @ d2000  10.90
tg190 @ d3000   9.99
tg190 @ d3400   9.87   <- where a reading step actually decodes
```

Decode is **26% slower** at the depth the system runs at than at the depth it was measured
at. Prefill is unaffected: a step's prompt IS built from empty (SPEC §4.1 — no conversation
history crosses steps), so `pp3400 @ d0` was always the right prefill measure.

Re-measured on `rl-v3` Q8_0, same device, same flags:

| | recorded | corrected |
|---|---|---|
| prefill per step (3400 @ d0, 58.15 t/s) | 57.4 s | 58.5 s |
| decode per step (190 @ **d3400**, 9.87 t/s) | 15.3 s | **19.3 s** |
| **per reading step** | 72.7 s | **77.7 s** |
| **full meeting** (15.2 steps + synthesis) | 19.0 min | **20.4 min** |

So the margin was never 3% in the other direction — nominal is already ~2% OVER, and the
steady-state and worst-case rows below inherit the same correction. **G4 FAILS as
configured.**

**This is CLAUDE.md trap 11 again — measuring the wrong thing, caught by an implausible
value rather than by process.** The tell was the `-d` sweep in a routine re-run: a decode
rate that falls 21% across a depth range shorter than one prompt cannot also be the rate
that applies after that prompt. Nothing in the original run was wrong except which number
was multiplied by 15.2.

**Do not "correct" the mitigation table below by the same ratio.** The 8k/6400-token
projection was computed from depth-0 decode too, and its decode happens deeper still
(~6,400 tokens), so its error is LARGER, not equal. It has to be measured, not scaled.

## The Pi-ratio projection was wrong by 17%, and why

Earlier tonight both models were benched on a Raspberry Pi 4, a cross-model ratio of 0.740
derived, and applied to the Reno 7's known MiniCPM5 numbers — projecting 16.2-16.4 min.
**Measured: 19.0 min.**

The method failed because **the Pi is ARMv8.0 without `dotprod` and the Reno 7 has it**. The
two models evidently exploit dotprod to different degrees, so a ratio measured on
non-dotprod silicon does not transfer to dotprod silicon. **Do not extrapolate cross-model
ratios across ISA feature boundaries** — benchmark on hardware sharing the target's feature
set, or measure the target.

## Sustained load: throttling is NOT the problem

29.5 minutes of continuous back-to-back inference, 14 rounds, exceeding a full meeting:

| round | elapsed min | prefill t/s | decode t/s | step s |
|---|---|---|---|---|
| 1 | 0.0 | 58.27 | 13.22 | 72.7 |
| 3 | 4.5 | 58.54 | **8.26** | **81.1** |
| 7 | 13.7 | 58.20 | 13.13 | 72.9 |
| 10 | 20.4 | 58.55 | **11.52** | **74.6** |
| 14 | 29.5 | 58.15 | 12.90 | 73.2 |

**Prefill -0.2%, decode -2.4% across 29.5 minutes.** Essentially flat.

**This corrects a claim made earlier in this session.** A shorter 6-round test showed
prefill 60.25 -> 57.32 and was reported as "-4.9% throttling, curve still descending". The
longer run shows that was measurement noise, not a thermal trend. The passively-cooled-phone
throttling risk that `CLAUDE.md` has flagged since Phase 0b **is not observable at this
model size** — a 0.8B Q8 model does not generate enough sustained heat to throttle this SoC.

## The real risk: transient decode stalls

Two of 14 rounds (~14%) showed decode collapse with prefill unaffected:

- round 3: decode 8.26 t/s (37% below normal), prefill normal at 58.54
- round 10: decode 11.52 t/s (13% below normal), prefill normal at 58.55

Prefill being unaffected rules out thermal and rules out the model — this is contention,
almost certainly another process on the phone competing for cores. The device is a general
purpose phone, not a dedicated appliance, so this is a realistic deployment condition rather
than a test artifact.

| scenario | per step | meeting | vs 20.00 ceiling |
|---|---|---|---|
| steady state (last 5 rounds) | 73.3 s | **19.4 min** | PASS, 3% margin |
| worst observed round | 81.7 s | **21.6 min** | **FAIL, 8% over** |

## What this means for shipping

**G4 passes on the median case and has no thermal risk — but a 3% margin leaves nothing for
contention, and contention was observed in 14% of rounds.** A meeting that happens to
overlap with background activity on the phone will exceed the budget.

This is a materially better position than "unmeasured projection", and materially worse than
"comfortable pass". Options, none yet taken:

1. **Reduce steps.** The budget is ~79% prefill (57.4 s of 72.7 s). Fewer, larger chunks cut
   total prefill roughly linearly. `runs/chunk1500/` measured the opposite direction and
   found chunk size buys ROUGE, not G1 — the reverse trade is unmeasured.
2. **Q4_K_M.** Measured earlier to cost roughly half the ROUGE margin over baseline
   (`runs/dropv6-q4/`) — a real quality cost, but it would buy latency headroom.
3. **Accept and document.** Ship with a stated expectation that a busy phone may exceed
   20 minutes, which may well be acceptable for an overnight/background transcription
   product but is a product decision, not a technical one.


## 8k context measured against 4k — SPEC §4.1's open question, answered

§4.1 states 4k "is a budget choice, not a model limit — and it is falsifiable… 8k must be
measured against 4k on the real device before 4k is treated as settled." That measurement
had never been made. It has now.

**Throughput** (same device, same config):

| | prefill t/s | per-token cost |
|---|---|---|
| pp3400 (4k config) | 58.20 | baseline |
| pp7300 (8k config) | 48.67 | **16% slower per token** — quadratic attention, as expected |

**Peak RSS** (`VmHWM`, llama-server, `-ngl 0`, after warm requests):

| context | VmHWM |
|---|---|
| 4096 | 1,784 MB |
| 8192 | **1,834 MB (+50 MB)** |

**§4.1's RSS argument against 8k does not hold for this model.** That argument was
extrapolated from an 8B model measured in the prior project (1.6 GB at ctx=2048 -> 4.3 GB
at ctx=65536). Qwen3.5-0.8B's hybrid linear-attention stack (only 6 of 24 layers are full
attention, `layer_types` in its config) makes its KV cache nearly context-independent:
**doubling the context costs 50 MB, not gigabytes.**

**Net latency effect — 8k wins despite the slower per-token prefill:**

| config | steps/meeting | per step | meeting |
|---|---|---|---|
| 4k, 2500-token chunks (current) | 15.2 | 73.0 s | **19.1 min** |
| 8k, 6400-token chunks | 5.9 | 164.5 s | **16.9 min** |

Fewer, larger chunks win because each step re-sends ~800 tokens of SYS + memory overhead,
and decode is per-step: going 15.2 -> 5.9 steps cuts repeated overhead ~61% and decode
tokens ~61%. That more than pays for the 16% per-token prefill penalty. **Margin against
the 20-minute ceiling goes from 5% to 16%** — enough to absorb the process-contention
stalls measured above, which is exactly what the 4k configuration cannot do.

### Why this is NOT yet a recommendation

The quality side is unmeasured and there is specific reason to expect a cost:

- **Trap 6**: at 2,500 tokens the reading step already records a chunk's tail and drops its
  head (containment 0.221 head vs 0.322 tail). A 6,400-token chunk makes that worse, not
  better.
- **`runs/chunk1500/`** measured the *opposite* direction and found smaller chunks buy
  ROUGE. Extrapolating that trend backwards predicts larger chunks lose ROUGE.
- The current checkpoint is trained at 2,500-token chunks with `POSITION` counts derived
  from them, so simply serving it at 6,400 is off-distribution — the same confound that
  made the `chunk1500` result hard to read.

**Testing this properly requires retraining at the larger chunk size**, which is a real
experiment, not a config flip. What is now established is that the *latency and memory*
side of §4.1's open question favours 8k, and that the RSS objection specifically is void
for this architecture. Whether quality survives is the open half.
