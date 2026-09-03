# `revfix-e3` — the G1 loss is now LOCATED and PARTLY FIXED, and the gate still moved down

**Do not adopt this checkpoint.** `mixed-e3` best-epoch remains the recommendation. This
file exists because the measurement is worth far more than the checkpoint: it is the first
time G1's failure has been traced to a specific, correctable cause, and the first time a
fix has verifiably closed the mechanism without closing the gate.

## What the loss map found (`tools/loss_map.py` on `mixed-e3` best-epoch)

The control arm is what makes this readable. `CONTROL_SCENARIOS` are decisions that are
TAKEN and never reversed — same generator, same prompt shape, same `key_term` templates.

| arm | key_term emitted | reaches MEMORY | survives to PROSE |
|---|---|---|---|
| control (no reversal) | 73.3% | **73.3%** | 40.0% |
| reversal | 59.3% | **14.8%** | 3.7% |

**In the control arm NOTHING is lost between emission and memory — 11/12 emitted, 11/12
retained.** In the reversal arm 16 scenarios emit the term and then lose it, with **zero
harness refusals**. So the loss is not the reading step failing to see the detail, and not
a guard rejecting it. It happens during REVISION specifically.

This corrects `runs/g1-study.md`'s earlier reading ("the reading step fails to capture the
term at all in 6/11"), which had no control arm and could not separate a general
point-quality deficit from a revision-specific one.

## The cause: STALE tool-call supervision, not a capability limit

The pool's 68 reversal rows teach exactly the observed failure:

    {"drop": ["公車路線調整"], "add": ["公車路線調整案改為取消"]}      <- key_term GONE

**0 of 34 revision ADD targets preserved the key term.** The on-disk edit-line gold has
been correct since the 2026-08-31 fix
(`ADD - 公車路線調整案改為取消，紅三十七路延駛計畫`) and `tools/to_toolcalls.py` converts
it correctly — verified by running the converter on the real gold line. **The pool rows
were simply never regenerated**, carried forward from `v5`'s pool through `selfdistil-e3`
into `mixed-e3`. Every v1.0 checkpoint has been trained on a demonstration of lossy
revision, on the exact capability G1 measures.

`gen_reversals.py --rebuild-gold` was run first and was a NO-OP: the gold was already
correct. The staleness is one layer up, in the converted pool.

## The fix landed on the mechanism — hard — and the gate went DOWN

`data/staging/sft_pool_revfix.jsonl` = `mixed-e3`'s pool with only the reversal rows
regenerated (26/26 targets now preserving, 0 lossy). Nothing else changed.

| | `mixed-e3` | **`revfix-e3`** |
|---|---|---|
| key_term reaches memory (reversal arm) | 14.8% | **48.1%** |
| survives to prose | 3.7% | **33.3%** |
| scenarios losing it during revision | 16 | **6** |
| control − reversal memory gap | **+58.5%** | **+11.9%** |
| probe passed | **8/27** | 6/27 |

**The revision-specific gap is essentially closed** — 58.5 points to 11.9 — which flips
`loss_map`'s own verdict from "loss is revision-specific: needs real reversal data" to
"the detail is lost generally: fixable from MeetingBank supervision, no reversal corpus
needed". That is a materially different research position from the one this project has
held for five refuted G1 attempts.

**And the gate still fell.** Because passing needs the LATE OUTCOME word, not just the
detail:

| | subject/key_term present | states late outcome | passed |
|---|---|---|---|
| `mixed-e3` | 13/27 | **8/27** | **8/27** |
| `revfix-e3` | **20/27** | 6/27 | 6/27 |

Detail retention up 54%, outcome statement down 25%. **The replacement point has a fixed
budget and now spends more of it on the identifying detail.** Same trade shape as every
other one measured here.

## Refuted: the point cap is NOT the binding constraint

Three inference-time refusals appear (`point too long (26 > 25 tokens)` x2,
`(27 > 25)` x1) while the gold maxes at 22 tokens and fits — probe scenarios carry longer
key terms than training ones. The obvious hypothesis is that `POINT_TOKENS=25` cannot hold
subject + outcome + key_term together.

**Measured and refuted.** Raising the harness cap to 32 gave **3/27**, worse than 6/27 —
recovering those three refusals added no passes and cost others. Caveat: the system prompt
still advertises 25, so this tests the harness cap alone, not a coordinated prompt+harness
bump; but it does refute "those refusals are what holds the gate back".

## Standing position

`mixed-e3` best-epoch stays the recommendation. The transferable results:

1. **G1's loss is revision-specific and was CAUSED BY STALE DATA**, not by a model or
   corpus limit. That data is now fixed; the fix is in
   `data/staging/sft_pool_revfix.jsonl` and should be carried into any future pool.
2. **Fixing it moves detail retention 3.2x at memory and 9x at prose**, and does not move
   the gate, because the bottleneck relocated to stating the late outcome.
3. The next G1 attempt should target **the outcome word**, not the detail — and should
   check both columns, since this build shows they trade against each other.
