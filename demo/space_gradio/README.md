---
title: arcsum — Live Agentic zh-TW Meeting Summarizer
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: apache-2.0
models:
- Luigi/qwen35-0.8b-arcsum
---

# arcsum — live agentic meeting summarizer (zh-TW)

Watch a 1B model read a Traditional-Chinese meeting transcript in ~2,500-token chunks,
curating a small **external memory** as it goes, then write the summary from that memory
alone.

The memory has exactly two slots:

- **`ARC`** — one rolling sentence holding the meeting's throughline (≤80 tokens)
- **`POINTS`** — up to 16 short facts (≤25 tokens each)

The model never writes prose during reading. It emits only edit lines:

```
ARC: <replacement throughline>
ADD - <new point>
DROP «<prefix of an existing point>»
NOP
```

The harness applies them deterministically — enforcing the caps, refusing malformed or
unsafe ops, spreading evictions rather than truncating — and re-renders the memory into
the next step's prompt. **No conversation history crosses steps.** The memory panel is
the only thing carried forward.

After the last chunk, one **SYNTHESIZE** call turns the finished memory into flowing
zh-TW prose. By then everything the summary can say has already survived curation.

## Why bother, versus map-reduce?

Because of the one thing aggregate scores cannot show: **a later chunk can overturn an
earlier conclusion.** Map-reduce summarises each window independently, so a decision
reversed at minute 90 never reaches the summary of minute 10 — structurally, not as a
tuning failure. That capability is tested directly by the project's G1 revision probe,
and it passes.

## This is the real harness

`arcsum/` is vendored here verbatim from the project and has **zero runtime
dependencies** — the chunker, op parser, memory, guards, caps and prose finaliser are the
same code paths used to produce the evaluation numbers. The demo drives the step loop
itself only so it can stream tokens; nothing about the mechanism is re-implemented for
the UI.

## Honest status

The checkpoint clears **6 of 7** ship gates against a *fair* map-reduce baseline (same
model, same chunk size, same output contract):

| gate | result |
|---|---|
| G1 revision probe | **PASS** |
| G2 faithfulness | **PASS** — 8 inversions vs 18 |
| G3 ROUGE-2 | **PASS** — wins 19/20 meetings |
| G3 ROUGE-L | **PASS** — wins 19/20 meetings |
| G4 on-device budget | **PASS** — projected, not measured |
| G3 ROUGE-1 | **FAIL** — 14/20, p=0.115 |

Under the project's all-or-nothing rule the recorded decision is **"ship the baseline"**.
ROUGE-1's effect size clears comfortably; it is the *consistency* test that misses, by
one meeting. The losses concentrate in long meetings, where the model fixates — the
known open weakness.

Full numbers, both caveated figures, and the training details:
**[model card](https://huggingface.co/Luigi/qwen35-0.8b-arcsum)**.

## Notes on this demo

- **Runs Q4_K_M for speed. The evaluated artifact is Q8_0.** No gate number above was
  measured on the quant running here.
- **CPU-only inference, deliberately.** The Space sits on ZeroGPU hardware, but the model
  is loaded with `n_gpu_layers=0` and never touches the GPU — the single `@spaces.GPU`
  function exists only because ZeroGPU Spaces hard-fail at startup without one. It draws
  no GPU quota. Expect a few seconds per chunk.
- Examples are real held-out meetings from the evaluation set, deliberately short ones;
  the model's weakness is long meetings, and a Space CPU cannot chew a 53-chunk
  transcript in a demo.
- Transcript format is v2: `speaker: text`, one utterance per line, no timestamps.

## Previous contents

This Space previously demonstrated **CURSOR**, the prior project's protocol
(`ADD`/`UPD`/`DEL`/`CMP`/`NOP` over a single NOTES state, timestamped transcripts,
anchored bullets). That is a different, superseded architecture — the URL slug still says
`cursor-wasm-demo` for continuity, but nothing of CURSOR runs here now.
