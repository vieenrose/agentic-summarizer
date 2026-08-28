# Demo Spaces

Two deployable front-ends for the same checkpoint, both running the **real harness**
rather than a re-enactment of it.

| | `space_gradio/` | `space_wasm/` |
|---|---|---|
| deployed as | **Luigi/cursor-wasm-demo** (live) | not deployed (kept as a fallback) |
| runs | Gradio, server-side `llama-cpp-python` | static page, in-browser via wllama |
| hardware | ZeroGPU (`zero-a10g`), **CPU-only inference** | none — free static Space |
| harness | vendors `src/arcsum/` verbatim | `arcsum.js`, a differential-tested port |

`space_wasm/` needs no GPU tier at all, so it is the fallback if the ZeroGPU allowance
is ever unavailable (free accounts get 2 ZeroGPU Spaces).

## Deploying

Neither directory is self-contained: the example transcripts are generated, not
committed (~200 KB of data that already lives in `runs/*/eval/*_pairs.json`; a second
copy would only drift).

```bash
python demo/make_examples.py --out-py demo/space_gradio/examples.py
python demo/make_examples.py --out-js demo/space_wasm/examples.js
cp -r src/arcsum demo/space_gradio/          # gradio build vendors the real package
```

Then upload the directory to the Space.

## Four things that will bite you

Every one of these was hit for real while deploying this.

1. **Use `HfApi.upload_folder`, NOT the `hf upload` CLI.** The CLI calls
   `/api/repos/create` with `"sdk":"gradio"` hardcoded on every upload — so it returns
   **402 Payment Required** even for a one-file push to an existing *static* Space on a
   non-PRO account. The Python API skips the create when the repo exists and works
   first try. This cost a long detour misdiagnosed as a subscription problem;
   `HF_DEBUG=1` shows the literal request body.
2. **ZeroGPU Spaces hard-fail at startup without a `@spaces.GPU` function.** The Gradio
   app keeps `_zerogpu_registration_noop()`, which is never called. All inference is
   plain CPU (`n_gpu_layers=0`), so it draws zero GPU quota — the decorator exists only
   to satisfy the platform check.
3. **Guard `demo.launch()` behind `if __name__ == "__main__"`.** Without it `import app`
   starts a server and blocks forever, which makes the module untestable and silently
   hung the first smoke test.
4. **Stop generation on the chat-template turn markers, not just `token_eos()`.**
   MiniCPM5 ends a turn with `<|im_end|>`, which is not necessarily the EOS token id.
   Breaking only on EOS let the model run past its turn and emit the literal
   `assistant` followed by a *second, near-duplicate summary* — measured, and it reached
   the UI. Fixing it also cut a run from 2,031 to 247 streamed updates.

## The WASM port is tested, not asserted

`space_wasm/arcsum.js` is a port of the Python package, and the claim that it matches is
backed by a differential test: the system prompts, the rendered `MEMORY:`/`CHUNK:`
blocks, chunk boundaries, op grammar, `spread` eviction and prose cleanup were compared
against the real Python harness on a held-out meeting and had to agree exactly.

That mattered — it caught a genuine bug. `prose.finalize` joins cleaned lines with a
**space**; the first JS version joined with `""`, silently welding adjacent sentences
together in every summary.

If `PROMPT_VERSION` or `TOKENIZE_VERSION` changes in Python, the port must be
re-derived and re-tested. The model was fine-tuned against those exact strings, so a
drifted port feeds it out-of-distribution input and the demo stops representing the
system.

## Honesty requirements

Both READMEs state, and should continue to state:

- the checkpoint clears **6 of 7** ship gates, and the recorded decision is
  **"ship the baseline"**;
- **G4 is a projection**, never measured on the phone;
- the demo runs **Q4_K_M**, while every gate number was measured on **Q8_0**.
