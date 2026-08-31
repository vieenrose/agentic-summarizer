# Chunk size 2500 -> 1500: quality improves materially, and G4 breaks under every assumption

**Verdict: the quality gain is real and larger than any training change measured in this
project. It is not affordable: even the most favourable latency decomposition exceeds
SPEC §7's 20-minute ceiling. `CHUNK_TOKENS` stays 2500.**

This run changed no committed default. It exists to price a trade-off.

## Why it was run

The reading step exhibits **within-chunk recency bias**: at 2,500 tokens it records the
tail of a chunk and drops the head.

- On the G1 probe the effect is total. Chunk 0 carries 「B 棟」 15 times and 搬遷 13 times
  at its head, then moves to agenda items 2-4. All eight emitted points cover items 2-4;
  the office-move vote is dropped. At budget 1200 and 800 the model captures it
  (「行政處建議搬遷至B棟大樓」). This is the sole cause of dropv6's G1 failure.
- On real MeetingBank meetings the bias is present and milder: 113 points over 6
  meetings, mean trigram containment **head 0.221 vs tail 0.322**.
- **It is not inherited from the teacher.** 671 gold points from `data/p4_supervision`
  run the other way: head-favoured, 335 vs 265, mean head 0.183 vs tail 0.146. The
  targets lean toward the head while the student leans toward the tail, so this is
  student behaviour at 1B, not a data defect, and regenerating supervision will not fix
  it.

## Quality (paired on the 19 meetings both runs completed)

| metric | chunk 2500 | chunk 1500 | 1500 better on |
|---|---|---|---|
| rouge1 | 15/19, +0.131 | **17/19, +0.220** | 15/19 meetings |
| rouge2 | 17/19, +0.065 | **18/19, +0.089** | 13/19 |
| rougeL | 18/19, +0.082 | **19/19, +0.105** | 11/19 |

Every metric improves, on both the sign test and the effect size. rouge1's mean delta
against the baseline nearly doubles.

**This understates the effect if anything.** dropv6 was trained with `POSITION` lines
carrying chunk counts computed at 2,500 tokens; running at 1,500 changes both the chunk
contents and the totals in that line, so the model is reading position values from a
distribution it never trained on. The gain arrives despite that mismatch. A retrain at
the smaller budget would be needed to measure the ceiling.

## Cost: G4 fails under every decomposition

Step count on the eval corpus rises **18.4 -> 33.3 per meeting (1.81x)**. Applying that
ratio to G4's basis (14 steps, 71 s thin / 90 s saturated, mean 80.5 s, + 48 s synthesis):

| assumption about per-step cost | per-step | projected |
|---|---|---|
| scales fully with chunk size (**most favourable possible**) | 48.3 s | **21.2 min** |
| half fixed, half chunk-scaled | 64.4 s | 28.0 min |
| constant — decode-dominated | 80.5 s | 34.8 min |

The ceiling is 20.00 min. **Even the bound that assumes per-step time is entirely
prefill — no fixed decode cost at all, which is physically impossible since every step
decodes ~150 tokens — lands 6% over.** The conclusion does not depend on knowing the real
prefill/decode split.

The mechanism is that total prefill work is roughly conserved when chunks shrink (the
same transcript is read either way, plus more re-renderings of memory), while decode cost
scales directly with step count. 1.81x the steps is 1.81x the decode.

## CORRECTION: smaller chunks do NOT fix G1

An earlier draft of this file claimed "G1 needs smaller chunks". **That was wrong, and
was inferred from a step-0 reading test rather than an end-to-end run.** Measured
end-to-end on the probe, G1 still FAILS at both smaller budgets:

| budget | office_move | budget_approval | G1 | note |
|---|---|---|---|---|
| 2500 | FAIL | PASS | FAIL | recency bias drops the head |
| 1200 | FAIL | FAIL | FAIL | office_move memory collapses to 3 points |
| 800 | FAIL | FAIL | FAIL | budget_approval memory collapses to **0 points** |

At 800 the failure is the *original* fixation bug, re-triggered: the model emits an
ARC-only update, repeats a byte-identical ARC three times (three `arc unchanged`
refusals), and finishes the meeting with an ARC and no points at all. dropv6 was trained
with `POSITION` at 2,500-token chunks, so 800 and 1200 are off-distribution in both chunk
content and the counts the position line carries.

So the head-capture improvement is real at the step level and does not survive to the
product. Chunk size buys ROUGE (the table above) but not G1.

## Consequence

- **G1 is not chunk-size-bound.** Its cause is understood at the step level but no tested
  intervention fixes it end-to-end.
- **G4 still forbids smaller chunks anyway.** And G4's 19.58 min baseline is itself a
  projection with a 2.1% margin, never measured on the phone, with sustained thermal
  throttling unmodelled and acting in the failing direction.
- A zero-latency alternative was tested and REJECTED: appending "cover the whole chunk,
  including the beginning" to the step SYS prompt does make the reading step capture the
  head — and then synthesis degenerates into repeated sentences (1,358 chars, trap 2,
  despite `repeat_penalty=1.1`) and loses 搬遷 entirely. Both probe cases went from
  1 PASS to 0. Fuller memory is not free.

## Where the output budget actually goes (production budget, dropv6, 6 meetings)

384 attempted ops: **75.5% applied, 24.5% refused.**

| reason | share of all ops |
|---|---|
| duplicate point | 14.8% |
| arc unchanged | 7.0% |
| prefix did not match exactly one point | 1.8% |
| point/arc too long | 0.6% |
| insufficient zh-TW content | 0.3% |

**The length caps are NOT the problem** (0.6% combined) — a suspicion worth recording as
refuted before someone raises `POINT_TOKENS`. Roughly **22% of every step's output is the
model re-emitting content it has already recorded**, which is both wasted capacity and,
on-device, wasted latency against the G4 budget.

Closing this honestly needs a real device measurement before any further chunk-size work:
every projection above inherits the uncertainty of a number nobody has measured.

Cheaper directions that do not spend the latency budget, none yet tested:

1. Retrain at a smaller budget so `POSITION` matches, then re-price — the gain measured
   here is a floor, not a ceiling.
2. Attack the recency bias directly rather than by shrinking the window, since gold
   supervision is already head-favoured and the student is not learning that from it.
3. Reduce decode per step (`-n 150`), the term that actually scales with step count.

## Caveats

- **n=19, not 20.** `AlamedaCC_11162021` lost the agent arm to trap 3 (llama.cpp 500 on
  invalid UTF-8) after exhausting retries. G3 gates are WITHHELD below `min_n=20`, so the
  table above is a paired comparison, not a gate result. More steps means more exposure
  to that failure — itself a small argument against smaller chunks.
- Both arms ran at the same budget, as SPEC §5.2 requires; the baseline is not
  handicapped.
- The step-count ratio is measured on this 20-meeting corpus; G4's 14-step basis comes
  from SPEC §4.1's token-ratio calculation, so the two are combined by ratio, not
  identity.
