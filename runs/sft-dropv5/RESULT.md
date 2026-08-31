# sft-dropv5: the NOP hypothesis is refuted; long meetings are solved; short meetings are not

**Verdict: 6 of 7 gates again. Long-meeting behaviour is now perfect (9/9). The blocker
has moved, and its likely cause is structural, not a mixing ratio.**

## The test

`sft-dropv4` fixed long meetings and cost short ones. The leading hypothesis was that its
pool's NOP share had drifted 34.9% -> 32.0%, making the model too ready to edit on
meetings with few chunks. dropv5 tests that as a **single variable**: `mix_phase4.py
--hold-nop` keeps the share at exactly 34.9% by admitting every new NOP sample alongside
the late non-NOP ones — which admits **more** long-meeting data than dropv4, not less
(688 new samples vs 455; 546 vs 510 at step index >= 40).

## Result (n=20, same eval set and pinned config throughout)

| metric | dropv2 | dropv4 | dropv5 |
|---|---|---|---|
| rouge1 wins | 14/20 | 14/20 | **14/20** (p=0.115) |
| rouge1 mean delta | +0.056 | +0.097 | +0.087 |
| rouge2 wins | 19/20 | 17/20 | 18/20 |
| rougeL wins | 19/20 | 17/20 | **19/20** |

G3 rouge1 fails on the sign test in all three builds. It needs 15/20.

## The hypothesis was wrong

| slice | dropv2 | dropv4 | dropv5 |
|---|---|---|---|
| **long, >= 400 lines** (n=9) | 4/9, +0.012 | 8/9, +0.217 | **9/9, +0.182** |
| **short, < 400 lines** (n=11) | 10/11, +0.091 | 6/11, -0.002 | 5/11, +0.010 |

Holding NOP constant did **not** recover the short meetings — they went 6/11 -> 5/11. So
the NOP share was not the cause. This is the outcome the dropv4 write-up flagged as
possible: "if short meetings stay depressed, the cause is the late-step data shifting
global behaviour, and no mixing ratio will address it."

What it did buy: long meetings reached **9/9**, and rougeL returned to 19/20. The
fixation meeting `LongBeachCC_05232017` holds at +0.282 (from dropv2's -0.164).

## Why: the step prompt carries no position signal

`build_step_prompt` renders **`MEMORY:` then `CHUNK:`** and nothing else. There is no step
index and no chunk count anywhere in the prompt. The model therefore cannot distinguish
step 3 of 5 from step 44 of 55 except indirectly, through how saturated the memory looks.

Under that design, late-step supervision cannot be learned *as* late-step behaviour — it
can only shift the model's global policy toward "behave like it is late in a long
meeting". That is precisely the measured signature, and it appeared in both builds
regardless of NOP mix:

- long meetings, where that policy is correct: 4/9 -> 8/9 -> 9/9
- short meetings, where it is wrong: 10/11 -> 6/11 -> 5/11

`corr(meeting length, dropv4 - dropv2 change) = +0.671`.

## What is now established

1. The long-meeting deficit is **real, data-bound, and fixed** by genuine long-meeting
   supervision (dropv3 had shown reweighting the existing 200 meetings cannot do it).
2. The remaining G3 rouge1 gap is **not** a data-volume problem and **not** a mixing
   problem. Two builds with different NOP shares and different sample counts landed on
   the same 14/20 by trading the same slices in opposite directions.
3. The next lever is structural: give the step prompt a position signal so late-step
   behaviour can be conditioned rather than globally absorbed. That is a
   `PROMPT_VERSION` bump (SPEC §4.1) and makes prior traces incomparable, so it is a
   deliberate decision, not an edit.

## Caveats

- G2 not run for this checkpoint; G4 remains dropv2's authorized projection, never
  measured on the phone.
- Three builds have now been read against the same 20 held-out meetings. That is real
  multiple-comparisons exposure and the reason no hyperparameter search was run against
  it. Any claim that a future build "clears 15/20" needs to be weighed against how many
  builds were compared to get there.
- Two llama.cpp 500s occurred during this run (trap 3) and were absorbed by
  `server_error_retries`; all 20 meetings paired.
