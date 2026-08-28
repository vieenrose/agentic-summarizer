---
title: arcsum — Live Agentic zh-TW Meeting Summarizer
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
- Luigi/minicpm5-1b-arcsum
---

# arcsum — live agentic meeting summarizer (zh-TW)

A 1B model reads a Traditional-Chinese meeting transcript in ~2,500-token chunks,
curating a small **external memory** as it goes, then writes the summary from that memory
alone. **Everything runs in your browser** — the model is downloaded once and executed
locally via [wllama](https://github.com/ngxson/wllama); no inference server, no data
leaves your machine.

The memory has exactly two slots:

- **`ARC`** — one rolling sentence holding the meeting's throughline (≤80 tokens)
- **`POINTS`** — up to 16 short facts (≤25 tokens each)

During reading the model never writes prose. It emits only edit lines:

```
ARC: <replacement throughline>
ADD - <new point>
DROP «<prefix of an existing point>»
NOP
```

The harness applies them deterministically — enforcing caps, refusing malformed or
ambiguous ops, spreading evictions rather than head-truncating — and re-renders the
memory into the next step's prompt. **No conversation history crosses steps.** After the
last chunk, one **SYNTHESIZE** call turns the finished memory into flowing zh-TW prose.

## Why bother, versus map-reduce?

Because of the one thing aggregate scores cannot show: **a later chunk can overturn an
earlier conclusion.** Map-reduce summarises each window independently, so a decision
reversed at minute 90 never reaches the summary of minute 10 — structurally, not as a
tuning failure. The project's G1 revision probe tests exactly that, and it passes.

## This is the real harness, and that claim is tested

`arcsum.js` is a port of the Python package, **differential-tested byte-for-byte against
it** — the system prompts, the rendered `MEMORY:`/`CHUNK:` blocks, the chunk boundaries,
the op grammar, the `spread` eviction and the prose cleanup all produce identical output
on a real held-out meeting. That fidelity is the point: the model was fine-tuned against
these exact strings, so a "close enough" port would feed it out-of-distribution input and
the demo would misrepresent the system.

(The test caught a genuine bug during the port: the Python joins cleaned lines with a
space and the first JS version joined with nothing, silently welding sentences together.)

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
ROUGE-1's effect size clears comfortably; the *consistency* test misses, by one meeting.
The losses concentrate in long meetings, where the model fixates — the known open
weakness. Full numbers and both caveated figures:
**[model card](https://huggingface.co/Luigi/minicpm5-1b-arcsum)**.

## Notes

- The model (**~656 MB, Q4_K_M**) downloads once and is cached by the browser.
- **Q4_K_M is not the evaluated artifact.** Every gate number above was measured on Q8_0;
  nothing shown in the demo is an eval result.
- In-browser CPU inference is slow. Examples are deliberately short real meetings from
  the held-out set — the model's weakness is long meetings, and a 53-chunk transcript is
  not a demo.
- Transcript format is v2: `speaker: text`, one utterance per line, no timestamps.
- Requires a browser with WebAssembly and enough memory for a ~656 MB model; desktop
  Chrome/Firefox/Safari are fine, phones may struggle.

## Previous contents

This Space previously demonstrated **CURSOR**, the prior project's protocol
(`ADD`/`UPD`/`DEL`/`CMP`/`NOP` over a single NOTES state, timestamped transcripts,
anchored bullets) as a Gradio app. That is a different, superseded architecture. The URL
slug still says `cursor-wasm-demo` for continuity, but nothing of CURSOR runs here.
