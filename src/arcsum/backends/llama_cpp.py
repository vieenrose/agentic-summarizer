"""In-process `llama-cpp-python` student — the DEPLOYED stack, for measuring it.

**Why this exists, and it is not convenience.** Every gate in this project is measured
against `llama-server` over HTTP, while the reference demo runs `llama-cpp-python` in
process on CPU. On 2026-09-02 a checkpoint that passed every gate churned badly in the
demo, and the divergence was initially attributed to `cache_prompt`. It is not — or not
only. Measured 2026-09-03 on the same GGUF file and the same transcript:

    llama-server, cache_prompt=true, GPU   ->  4 points, 0 churn
    llama-cpp-python, CPU (the demo)       ->  1 point,  4 churn

Same weights, same meeting, opposite outcome. **A flag on the HTTP server does not
reproduce the deployed stack**, because two things differ at once: the inference
implementation and the device. `Scorecard.deployment_mismatch` compares SETTINGS and
therefore cannot see this class at all; the only way to measure deployed behaviour is to
run the deployed code.

**AND THAT IS STILL NOT SUFFICIENT — measured, 2026-09-03.** This backend was written to
reproduce the demo failure locally, and it does not. On the same GGUF and the same
transcript, matching four variables in turn:

    llama-server, GPU,  cache on               -> 4 points, 0 churn
    llama-cpp-python, CPU, 8 threads, cache on -> 4 points, 0 churn
    llama-cpp-python, CPU, 2 threads, cache on -> 4 points, 0 churn   (the demo's thread count)
    the HF Space demo itself                   -> 1 point,  4 churn   <- NOT REPRODUCED

The remaining difference is the deployment HOST. Greedy decoding at `temperature=0` is
deterministic only for a FIXED floating-point reduction order, and that order depends on
the CPU, the thread count, and the kernels llama.cpp selects for the hardware it finds. So
two runs of identical weights and identical settings can diverge on different machines, and
a local evaluation cannot rule the divergence out.

**The honest consequence: this backend narrows the gap and does not close it.** Evaluating
on the deployment target is the only thing that would, and short of that, production logs
are the instrument — which is how the failure was actually found. Treat a local scorecard
as necessary and not sufficient before shipping.

**Kept deliberately thin, and mirroring `demo/space_gradio/model_backend.py` exactly.**
The plain-ChatML rendering is copied rather than improved: this class exists to reproduce
the demo, so any divergence from it is a bug here, not an upgrade. In particular the
prompt is built by hand rather than through the GGUF's embedded template, because
Qwen3.5's template injects a `<think>` block the fine-tune never saw (CLAUDE.md trap 10).

`llama-cpp-python` is in the `serve` extra, never a core dependency: the test suite must
keep running with no GPU, no weights, no network and no optional extra installed. The
import is therefore local to `__init__`, with an actionable message when it is missing.
"""

from __future__ import annotations

STOP = "<|im_end|>"


class LlamaCpp:
    """A `(system, user) -> str` callable over an in-process GGUF, as the demo runs it.

    `n_gpu_layers=0` is the default because that is what the reference demo uses on Space
    CPU, and the point of this backend is fidelity to deployment rather than speed.
    """

    def __init__(self, gguf_path: str, *, n_ctx: int = 8192, n_threads: int = 8,
                 n_gpu_layers: int = 0, max_tokens: int = 512,
                 repeat_penalty: float = 1.0, reuse_cache: bool = True):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on an optional extra
            raise ImportError(
                "llama-cpp-python is required for the deployed-stack backend. "
                "Install the extra: pip install 'arcsum-agentic[serve]'"
            ) from exc

        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_batch=n_ctx,
                         n_ubatch=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        #: The demo keeps ONE `Llama` object across every step of a meeting, so its KV
        #: cache carries over between calls. That statefulness is part of what is being
        #: measured; `reuse_cache=False` calls `reset()` before each step to isolate it.
        self.reuse_cache = reuse_cache
        self.stop_token_ids = set(
            self.llm.tokenize(STOP.encode("utf-8"), add_bos=False, special=True))
        self.stop_token_ids.add(self.llm.token_eos())

    def __call__(self, system: str, user: str) -> str:
        prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
                  f"<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        if not self.reuse_cache:
            self.llm.reset()
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        out: list[int] = []
        for tok in self.llm.generate(tokens, temp=0.0, repeat_penalty=self.repeat_penalty):
            if tok in self.stop_token_ids:
                break
            out.append(tok)
            if len(out) >= self.max_tokens:
                break
            # The markers can surface as plain text rather than as their special token
            # ids, depending on how the vocab round-trips — checked in the demo too.
            if len(out) >= 4:
                tail = self.llm.detokenize(out[-8:]).decode("utf-8", errors="replace")
                if any(m in tail for m in ("<|im_end|>", "<|im_start|>")):
                    break
        return self.llm.detokenize(out).decode("utf-8", errors="replace").strip()
