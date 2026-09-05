# Journal synthesis supervision — the full build line, 2026-09-04

Six pools, two seeds each, all evaluated on `data/heldout_zh` (40 meetings; 10 exceed the
16-point working set, so the journal actually engages — SPEC §5.2.3). **Worse-seed scoring**
per §5.2.1, and all ungrounded rates corrected for the `<think>` leak (see below).

| pool | synthesis slice | churn | retention | clean | ungrounded |
|---|---|---|---|---|---|
| `v11` | pre-journal (Qwen3.8, ≤16 entries) | 13.3% | 0.836 | 5 | **4.7%** |
| `v12` | journal, full replace, gemma-3 | 29.8% | 0.921 | 4 | 6.4% |
| `v13` | + near-duplicate dedup | 36.7% | **0.936** | 1 | 7.3% |
| `v14` | + `revise` promotion | 17.4% | 0.902 | 3 | 5.3% |
| `v16` | journal, full replace, **Qwen3.8** + revise | **8.6%** | 0.848 | **6** | 9.9% |
| `v17` | journal, **additive**, Qwen3.8 + revise | **6.2%** | 0.801 | **11** | 6.2% |
| `v18` | + group-wise DENSE targets | 30.8% | 0.876 | 2 | 7.9% |

## `v18`: the density fix worked, and seed variance ate it

SPEC §5.2.5 located the deficit as DENSITY — the model writes ~1/3 of its allowed budget and
gives each memory entry ~26 characters, enough to mention a point but not to state it. Two
things were needed to fix that, and the first is the transferable one:

**The teacher has a hard length prior that instructions and budget do not move.** A 28-entry
journal produced 664 characters and a 34-entry one 680 — 23.7 and 20.0 characters per entry —
and raising `max_tokens` from 1400 to 3000 returned **byte-identical output**. It is not
truncated; it stops. An explicit "35-45 characters per item, ~700-900 for 20 items" clause
changed nothing. Same shape as CLAUDE.md's v4->v5 case: entrenched behaviour moves by what a
model is SHOWN, not by asking.

**Group-wise synthesis is the workaround.** Splitting the journal into groups of 12, each
carrying the ARC, gives every group the teacher's full ~660-character budget, so density scales
with the number of calls: **23.7 -> 38.1** and **20.0 -> 29.1** characters per entry, coverage
27/28 and 34/34 at the STRICT 0.45 containment threshold, 0 ungrounded.

Final slice: 142 rows, **32.6 ch/entry** (from 28.9), coverage 0.968 @ 0.45, **0 ungrounded
across 1,741 specifics**, journals up to 49 entries.

**The student inherited it**: `median_chars_per_point` 23 -> 33, and retention rose to **0.876**
worse-seed, the best of any Qwen3.8 build — at **3.6% churn on seed 0**, the lowest single
measurement in the entire line.

**And seed 1 came in at 30.8% churn, 2/40 clean.** A 27-point spread inside one pool. So
`v18` scores worst-since-`v13` on the worse seed while holding the best single result ever
measured. Per §5.2.1 it fails; per §5.2.1 rule 4 its difference from `v17` is UNRESOLVED,
because the spread swamps it.

## The standing finding: a bimodal training failure that n=2 can detect but not characterise

Two pools now show one excellent seed and one unusable one — `v13` (3.5/13.3 on its
predecessor's scale, then 36.7/9.2) and `v18` (3.6/30.8). This is not a small effect on top of
noise; it looks like two distinct outcomes of the same training configuration. Two seeds can
reveal that and cannot measure its rate.

**`v17` remains the recommended candidate** — not because it is best on any single axis, but
because it is the only build STABLE across seeds: churn 6.2/5.3 and clean 11/11. Stability
across seeds is itself the property a ship decision needs.

**Next measurement is seeds, not another pool variant.** Additional seeds on `v17` and `v18`
answer whether `v18-s1` is a rare mode or a coin flip; six single-pool comparisons cannot.

## The finding: the TEACHER selects the trade, and it is a restraint axis

`v13` and `v16` differ in exactly one thing — who wrote the synthesis targets — and the
outcome inverts:

| | targets | student churn | student retention | recorded pts | prose chars |
|---|---|---|---|---|---|
| `v13` | gemma-3-27b: 0.955 coverage @ 34.6 ch/entry | 36.7% | 0.936 | 501 | 432 |
| `v16` | Qwen3.8-27B: **0.991** coverage @ **28.9** ch/entry | **8.6%** | 0.848 | 443 / 369 | 386 / 316 |

Qwen3.8 covers MORE of the memory in FEWER characters, and the student inherits the
compactness as **restraint**: it records less (369 vs 501 points on the quiet seed), writes
less (316 vs 432 chars), churns far less, and consequently renders a smaller share of what it
recorded.

**Rendering density is NOT the explanation** — chars-per-recorded-point is flat across every
build (31.2, 34.4, 35.9, 32.7, 37.4). The axis is how much the model commits to recording at
all. Churn and coverage move together because both are downstream of that one disposition.

**Consequence for the design: `v13` and `v16` bracket the target rather than one being
better.** `v13` has the retention and unusable churn; `v16` has the best stability ever
measured (seed 1 at **0.8%** churn, the lowest of any build including `v11`) and gives back
the retention gain. `v14` sits between on every axis, which is why it remains the balanced
candidate. `v17` tests whether ADDITIVE composition — keep `v11`'s proven small-memory targets,
add journal rows only for the >16-entry regime `v11` structurally cannot reach — gets both.

## The `<think>` leak, and why it nearly became a false finding

`prose.finalize()` stripped bullets, headings, labels, anchors and markdown — but not
reasoning markup. Qwen3.5 opens `<think>` after the assistant turn by habit, and `--no-jinja`
removes the tag from the TEMPLATE, not from the model. The tag therefore reached the
user-visible summary, and the grounding instrument counted the literal token `think` as a
fabricated specific.

Share of flagged tokens that were this artifact: `v11` 0%, `v12` 0%, `v13-e3` 21%, `v14-e3`
9%, **`v16` 54% and 77%**. Uncorrected, `v16` reported 19.3% / 18.1% ungrounded against a real
**9.9% / 4.9%** — and the obvious reading would have been "the Qwen3.8 teacher made
faithfulness three times worse", which is false and would have reversed a correct decision
about the best available teacher.

**It grew silently across four builds because nothing ever looked at WHICH tokens were
flagged** — only at the rate. That is trap 12's shape exactly: an instrument reporting a
plausible number for a reason unrelated to the property it names. Fixed in `prose.py`, pinned
by four tests, and every rate in the table above is recomputed with the artifact removed from
both numerator and denominator.

**It was also a live product defect**: the demo rendered `<think>` to users.

## Teacher provenance, resolved

The supervision teacher is **Qwen3.8-27B (UD-Q4_K_M, local)** — family-matched to the
Qwen3.5-0.8B student, and the pool's original teacher, so the confound introduced when its
blobs were deleted is now closed rather than merely isolated. Measured against the hosted
`qwen3.8-max` on the same meetings: coverage 1.000 vs 0.991, 0 ungrounded in both. **Equal
quality, so reproducibility decides** — frozen local weights cannot change behind a stable
name mid-campaign.
