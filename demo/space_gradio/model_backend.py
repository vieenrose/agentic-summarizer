"""CPU-only inference backend for the arcsum demo, via llama-cpp-python.

Deliberately bypasses Llama.create_chat_completion(): that high-level API has no way to
pass extra Jinja template variables through to the model's own embedded chat template,
and Qwen3.5's template needs `enable_thinking=False` explicitly set (undefined -> the
model free-runs into its own reasoning mode; matches llama-server's `--jinja` +
`chat_template_kwargs` behavior, verified to reproduce the exact production non-thinking
prompt: `<think>\\n\\n</think>\\n\\n` immediately closed, no reasoning preamble).
"""

from __future__ import annotations

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

STOP = "<|im_end|>"


class ArcsumModel:
    """Implements the `model(system, user) -> str` interface `arcsum.agent` expects,
    backed by a local, CPU-only llama-cpp-python instance.

    Two sampling rules, both measured rather than assumed (see the model card):

    * **`repeat_penalty` on PROSE calls only.** Greedy decoding on a 1B model
      degenerates on long free-form output -- one synthesis emitted the same sentence
      eight times, cut from 2,053 to 432 characters by a 1.1 penalty. Reading steps must
      NOT carry it: their output is a fixed op vocabulary, so penalising repetition
      penalises the literal ADD/DROP/ARC tokens the format requires.
    * **Separate token budgets.** A reading step emits a few short edit lines; the
      synthesis call writes a whole summary.
    """

    def __init__(self, gguf_path: str, n_ctx: int = 8192, n_threads: int = 2,
                 plain_chatml: bool = True):
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_ubatch=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        # `plain_chatml` renders `<|im_start|>role\n...<|im_end|>` directly instead of
        # running the GGUF's embedded jinja template. This is NOT a shortcut -- it is the
        # configuration `qwen-tools-v5`'s published gate numbers were measured under
        # (llama-server `--no-jinja`). Qwen3.5's own template appends a THINK block whose
        # form depends on a branch: `enable_thinking=False` gives a closed
        # `<think>\n\n</think>\n\n`, the default gives an open `<think>\n`. Both differ
        # from what the evaluated server sent, and a demo that silently picks a third
        # prompt shape is not showing the model that was measured.
        self.plain_chatml = plain_chatml
        self.formatter = None
        if not plain_chatml:
            template = self.llm.metadata.get("tokenizer.chat_template")
            if not template:
                raise RuntimeError("model gguf has no embedded chat_template")
            self.formatter = Jinja2ChatFormatter(
                template=template, eos_token=STOP, bos_token="<s>", add_generation_prompt=True
            )
        # Qwen3.5 ends a turn with <|im_end|>, which is NOT necessarily llm.token_eos().
        # Breaking only on token_eos() lets the model run past its turn and open a new
        # one -- measured: a synthesis emitted its summary, then the literal "assistant",
        # then a second near-duplicate summary, all of which reached the UI. Resolve the
        # marker to real token ids once, here, and stop on them.
        self.stop_token_ids = set(
            self.llm.tokenize(STOP.encode("utf-8"), add_bos=False, special=True)
        )
        self.stop_token_ids.add(self.llm.token_eos())
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def stream(self, system: str, user: str, *, max_tokens: int = 512, repeat_penalty: float = 1.0):
        """Yields the cumulative decoded completion text after every new token.

        Mirrors the wasm demo's per-token cursor_step_next loop, so the UI can show a
        live typing effect rather than waiting for the whole step to finish.
        """
        if self.plain_chatml:
            prompt_text = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt_text = self.formatter(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                enable_thinking=False,
            ).prompt
        prompt_tokens = self.llm.tokenize(prompt_text.encode("utf-8"), add_bos=False, special=True)
        self.last_prompt_tokens = len(prompt_tokens)

        out_tokens: list[int] = []
        for tok in self.llm.generate(prompt_tokens, temp=0.0, repeat_penalty=repeat_penalty):
            if tok in self.stop_token_ids:
                break
            out_tokens.append(tok)
            yield self.llm.detokenize(out_tokens).decode("utf-8", errors="replace")
            if len(out_tokens) >= max_tokens:
                break
            # Belt and braces: the markers can also surface as plain text rather than
            # as their special token ids, depending on how the vocab round-trips.
            if len(out_tokens) >= 4:
                tail = self.llm.detokenize(out_tokens[-8:]).decode("utf-8", errors="replace")
                if any(m in tail for m in ("<|im_end|>", "<|im_start|>")):
                    break

        self.last_completion_tokens = len(out_tokens)

    def __call__(self, system: str, user: str, **kw) -> str:
        text = ""
        for chunk in self.stream(system, user, **kw):
            text = chunk
        return text.split(STOP)[0].strip()
