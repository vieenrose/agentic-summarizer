"""llama.cpp `llama-server` backend (PLAN.md §2b, §4).

We deliberately do **not** use llama.cpp's tool-calling layer. Our ops are plain text: we
own SYS and `voxsum.ops` parses the output. What we want from llama-server is the correct
chat template (`--jinja`) and, optionally, GBNF-constrained decoding.

On the grammar: the screen must run **without** one, because its whole signal is whether
the model naturally emits valid ops. Volume generation may run **with** one to maximise
usable traces. Constraining during the screen would mask what it measures.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = ["LlamaServer", "OP_GRAMMAR", "THINKING_FLOOR"]

# Minimum output budget when thinking is on. Reasoning length varies by quant — 4k chars
# observed on Q8_0, 5k+ on UD-Q4_K_XL for the same chunk — so this is set well clear of
# both. Too low and the step returns reasoning with no ops.
THINKING_FLOOR = 8192

# GBNF for the text op grammar. Anchors are shape-checked only — that a `[m:ss]` resolves
# to a real line is the harness's job (CLAUDE.md §6.1), not the sampler's.
OP_GRAMMAR = r"""
root        ::= line ("\n" line)*
line        ::= add | upd | del | cmp | title | "NOP"
add         ::= "ADD " section " - " text anchor
upd         ::= "UPD " section " «" text "» -> " text anchor
del         ::= "DEL " section " «" text "»"
cmp         ::= "CMP " section ("\n- " text anchor)+
title       ::= "TITLE: " text
section     ::= "SUMMARY" | "DECISIONS" | "ACTIONS" | "OPEN" | "TOPICS"
anchor      ::= " [" [0-9]+ ":" [0-9] [0-9] (":" [0-9] [0-9])? "]"
text        ::= [^\n«»\[]+
"""


@dataclass
class LlamaServer:
    """Minimal `/v1/chat/completions` client. Callable as the agent loop's `model`.

    **Gemma 4 is a thinking model.** Its reasoning arrives in `reasoning_content` and the
    op lines in `content`; if the token budget runs out mid-thought, `content` is an empty
    string and the step silently looks like a NOP. Two consequences baked in here:

    * `thinking=False` (default) sends `chat_template_kwargs.enable_thinking=false`, which
      is ~6x faster and emits ops directly. `reasoning_budget: 0` is *not* equivalent —
      llama.cpp ignores it for this model and you get an empty `content`.
    * with `thinking=True`, `max_tokens` is floored at `THINKING_FLOOR` to cover reasoning
      **and** ops. Lower quants reason *longer*, not shorter — measure before trimming.

    Per PLAN.md §2c thinking is legitimate for the teacher — it is extra compute on the
    same input, not extra input — and only the op lines are ever kept as a target.
    """

    base_url: str = "http://127.0.0.1:8080"
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    max_tokens: int = 512
    seed: int | None = 0
    grammar: str | None = None
    thinking: bool = False
    #: Some templates (MiniCPM5) INSERT an empty <think> block when
    #: enable_thinking=false is sent — a surface our SFT never showed the model
    #: (measured: MiniCPM then fills it with base-RL reasoning, empty content).
    #: Set False to omit the kwarg entirely; the template then omits the block.
    send_thinking_kwarg: bool = True
    timeout: float = 600.0
    #: Raw reasoning from the last call, for logs. Never used as a training target.
    last_reasoning: str = ""

    def __call__(self, system: str, user: str) -> str:
        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            # Thinking needs room for reasoning plus the ops that follow it.
            "max_tokens": (
                max(self.max_tokens, THINKING_FLOOR) if self.thinking else self.max_tokens
            ),
            # Deliberately no "\n\n" stop: it truncates reasoning mid-thought and yields
            # an empty `content` that is indistinguishable from a genuine NOP.
            "stop": ["<end_of_turn>"],
        }
        if not self.thinking and self.send_thinking_kwarg:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.grammar:
            payload["grammar"] = self.grammar

        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:  # surface the server's own message
            raise RuntimeError(f"llama-server {exc.code}: {exc.read().decode()[:400]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach llama-server at {self.base_url} — is it running? ({exc.reason})"
            ) from exc
        message = body["choices"][0]["message"]
        self.last_reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        if not content and self.last_reasoning:
            # Ran out of budget mid-thought. Fail loudly: a silent "" would be scored as a
            # NOP and quietly depress every metric the screen reports.
            raise RuntimeError(
                f"model emitted {len(self.last_reasoning)} chars of reasoning and no ops "
                f"(finish_reason={body['choices'][0].get('finish_reason')}). "
                "Raise max_tokens or set thinking=False."
            )
        return content

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError):
            return False
