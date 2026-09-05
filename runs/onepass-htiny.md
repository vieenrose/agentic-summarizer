# `granite-4.0-h-tiny` bf16, one-pass long-document summarisation — 2026-09-03

Measured with `tools/onepass_probe.py` (committed, re-runnable) against
`data/heldout_zh/LongBeachCC_04172018.txt`, a real 91,802-heuristic-token zh-TW meeting.
Served on `~/llama.cpp/build` (CUDA, knows `granitehybrid`) at `-c 131072`, bf16, greedy with
`repeat_penalty=1.1`, `cache_prompt: false`. Artifacts: `runs/onepass-htiny.json`,
`runs/onepass-htiny-long.json`.

**bf16 was used deliberately.** Two earlier Granite candidates were judged from Q4_K_M alone
and that verdict was wrong both times; the protocol is now full precision first, quantise
afterwards.

## Result

| input (heuristic tok) | granite tok | chars | specifics | ungrounded | sec |
|---|---|---|---|---|---|
| 5,000 | ~7,800 | 878 | 5 | 0 | 7 |
| 12,000 | ~18,700 | 231 | 1 | 0 | 4 |
| 20,000 | ~31,200 | 591 | 4 | 2 | 8 |
| 32,000 | ~50,000 | 593 | 8 | 1 | 11 |
| 50,000 | ~78,000 | 586 | 1 | 0 | 16 |
| 80,000 | 125,054 | 551 | 2 | 0 | 25 |

**Coverage does not scale with input.** Across a 16x input range the specifics count wanders
between 1 and 8 with no trend, and output length pins around 550-590 characters. This is the
same failure already measured on `granite-3.1-3b-a800m` (2-7 specifics at every length) and
`granite-4.0-h-350m` — at 7B total / 1B active it is not a capacity problem that scale fixes
within this family.

**It is worse than the 0.8B map-reduce baseline it would replace.** On the reachable-reference
set the baseline measures 741 characters and **8.5** grounded specifics per meeting, and the
27B teacher one-pass measures 873 and **20.1**. h-tiny returns **2** specifics from an 80,000-
token meeting. Unlike `3b-a800m`, it is also not reliably faithful — it fabricated at 20k and
32k.

## Deployment verdict

Not viable, and not marginally so. bf16 weights are **13.9 GB** against SPEC §6's **2.5 GB**
ceiling — 5.6x over. Q4_K_M would be ~4.4 GB, still 1.8x over, and quantising a model that
already fails on quality does not rescue it.

## The finding worth keeping: zh-TW costs Granite 1.56x in tokens

80,000 heuristic tokens rendered as **125,054 granite tokens** — the ratio that made the
first 80k attempt fail against a 98,304 context. This is the same inefficiency measured
directly on the tokenizers (Granite: 0.727 chars/token on zh-TW vs 0.909 on zh-CN, **+20.0%**;
Qwen3.5-0.8B: 1.577 vs 1.761, **+10.5%**).

Two consequences:

1. Any Granite context budget must be quoted in GRANITE tokens. A "128k context" model holds
   roughly **82k heuristic tokens** of zh-TW — it cannot read this corpus's longest meetings
   in one pass at all.
2. It is independent evidence for running the pipeline's internals in Simplified
   (`arcsum/simplified.py`): the saving is largest exactly where the constraint binds.

## Where this leaves the architecture question

`runs/PROJECT-REVIEW.md` §4 left the single-pass option open pending a small-model benchmark.
It is now closed for this family. Nothing measured is simultaneously (a) inside 2.5 GB,
(b) fast enough for 80k in 20 minutes, and (c) able to cover a long meeting. The chunked
agentic architecture is not competing with a viable one-pass alternative — it is competing
with map-reduce, which remains the real control arm.
