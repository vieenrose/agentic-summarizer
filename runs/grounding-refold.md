# The grounding instrument was miscounting zh numerals — re-scored, 2026-09-03

`evalkit/grounding.py` folded numeral SYSTEMS not at all: a claim was grounded only if its
literal characters appeared in the source. The module documented this as an accepted false
positive ("`兩百萬` vs `2000000`"), which made it sound like a rounding error. It was not.

**The corpus writes figures in Arabic; fluent zh-TW output writes them in CJK.** So the
mismatch did not fire at random — it fired on exactly the summaries that were written well.

## How it was found

While building journal-shaped synthesis supervision (`tools/gen_journal_synth.py`), the
grounding gate rejected 3 of the first 6 teacher outputs. Every rejection was faithful:

| flagged | memory actually said |
|---|---|
| `六十` | `60天` |
| `十二` | `12個月` |
| `九十萬` | `90萬` |

A second, worse defect surfaced with it: `兩` and `〇` were missing from `CJK_NUMBER`'s
character class, so `兩百萬` matched as `百萬` and was **valued at 1,000,000** — half its real
value. That silently compared a correct figure against the wrong number rather than merely
failing to detect it. Both are now pinned by tests, including one asserting that every
character `cjk_to_int` parses is also a character `CJK_NUMBER` detects.

## Re-scored, deterministically, from the stored artifacts

Scorecards persist the flagged tokens per meeting, so every past number could be recomputed
against the real transcripts with no model and no re-run. The fold is **strictly more
permissive** — it can only turn a reported fabrication into a pass — so every pre-fix rate is
an upper bound.

| scorecard | reported | corrected |
|---|---|---|
| `qwen-tools-v5` (shipped) `_v1_cacheon` | 33.3% | **27.3%** |
| `qwen-tools-v5` `_asr_cacheon` | 43.5% | 34.8% |
| `mixed-e3` | 18.2% | 9.1% |
| `regen-e3` ep3 | 46.2% | 38.5% |
| `s234-e3` (three scorecards) | 28.6–33.3% | **0.0%** |
| `spec-e3` ep2 | 15.6% | **3.1%** |
| `spec-e3` ep3 | 47.6% | 19.0% |
| **`v11-e3` (SPEC v1.1)** | **21.2%** | **6.1%** |

## What this changes, and what it does not

**Does not change the ordering.** `v5` remains the worst by a wide margin. Every conclusion
that rested on the *comparison* between checkpoints survives.

**Does change the magnitudes, and unevenly — the correction is largest for the better
checkpoints.** `v5` moves 6 points; `spec-e3` moves 12.5 and `v11-e3` moves 15.1. That is
the expected shape: a model writing more fluent zh triggers more of the false positive, so
the broken instrument was penalising fluency. Reading the old table, the newest checkpoints
looked roughly half as faithful as they are.

**Consequence for `runs/PROJECT-REVIEW.md`.** Its §4 comparison (agent 7% ungrounded vs
baseline 5% vs teacher 3%) was measured with the broken fold on all three arms and should be
re-measured, not adjusted — the arms differ in how much CJK they emit, so the bias is not
common-mode and cannot be corrected by subtraction.

**Consequence for the architecture question.** The case for keeping v1.1 rests on
faithfulness, and v1.1's faithfulness was understated: 6.1% ungrounded over 33 specifics, not
21.2%. Nothing here addresses its coverage deficit, which remains the real problem and is what
the journal synthesis work targets.
