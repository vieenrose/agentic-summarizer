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

__all__ = ["LlamaServer", "OP_GRAMMAR"]

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
    """Minimal `/v1/chat/completions` client. Callable as the agent loop's `model`."""

    base_url: str = "http://127.0.0.1:8080"
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    max_tokens: int = 256
    seed: int | None = 0
    grammar: str | None = None
    timeout: float = 600.0

    def __call__(self, system: str, user: str) -> str:
        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            # Ops are line-oriented; a blank line means the model has stopped emitting.
            "stop": ["\n\n", "<end_of_turn>"],
        }
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
        return body["choices"][0]["message"]["content"]

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError):
            return False
