"""Stdlib-only HTTP client for a local `llama-server` instance — the `ModelFn` contract
made concrete against a real backend.

No `requests`, no `openai` SDK: `urllib.request` only, so the core package's
zero-dependency property extends all the way to inference, and the whole test suite can
stub `urllib.request.urlopen` directly rather than mocking a third-party client.

We deliberately do NOT use llama.cpp's tool-calling layer. Ops are plain text: `arcsum`
owns the SYS prompt and `arcsum.ops` parses the output. What is wanted from `llama-server`
is the correct chat template (`--jinja`) and, optionally, grammar-constrained decoding.

**Two configured instances, not one with a per-call override.** Reading steps and the
`SYNTHESIZE` call have different output-length regimes (~150 edit-line tokens vs
<1,000 prose tokens) and may want different sampling. Rather than adding a `max_tokens`
parameter to `__call__` — which would break the `ModelFn = Callable[[str, str], str]`
seam every test and caller relies on — construct two `LlamaServer` instances with
different `max_tokens`.

**MiniCPM5-specific special tokens are NOT hardcoded here.** The prior project's Gemma
backend hardcoded `stop=["<end_of_turn>"]` and a Gemma-specific `enable_thinking`
chat-template kwarg; MiniCPM5 is a different model family, and these need verification
against its real chat template before being assumed (SPEC §9 Phase 0). `stop` and
`extra` are both caller-configurable for exactly this reason — do not hardcode a
Gemma-ism here on the assumption it will transfer.

**Verified (Phase 2 pilot eval, 2026-08-26): MiniCPM5 needs `enable_thinking`
disabled.** Served via `llama-server --jinja` with no override, a MiniCPM5 chat
completion defaults to emitting `<think>...</think>` reasoning content that can
consume the entire `max_tokens` budget before any answer — `__call__` then raises
("empty content ... reasoning_content present"), not silently degrades, so this
surfaces immediately rather than being mistaken for a model-quality problem. Fix:
construct the server with
`extra={"chat_template_kwargs": {"enable_thinking": False}}`, or start `llama-server`
itself with `--reasoning off`. Left as caller-configurable (not hardcoded here) for
the same reason `stop` is — a future model family may not need it, and one that does
may need a different toggle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib import error, request

DEFAULT_TIMEOUT = 600.0

#: GBNF for the 4-op text grammar (arcsum.ops). DRAFT — the character class handling of
#: multi-byte CJK inside llama.cpp's GBNF engine has NOT been verified against a live
#: model (that needs SPEC §9 Phase 0's device session); treat this as a starting point,
#: not a settled artifact. Screening runs WITHOUT a grammar (the model must naturally
#: emit valid ops for that measurement to mean anything); volume trace generation may
#: run WITH one.
OP_GRAMMAR = r"""
root  ::= line ("\n" line)*
line  ::= add | drop | arc | "NOP"
add   ::= "ADD" [ \t]* ("-" [ \t]*)? text
drop  ::= "DROP" [ \t]* prefix
arc   ::= "ARC" [ \t]* [:：] [ \t]* text
prefix ::= "«" text "»" | "<<" text ">>" | "「" text "」" | "\"" text "\""
text  ::= [^\n]+
"""


@dataclass
class LlamaServer:
    base_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 0.95
    max_tokens: int = 512
    #: Greedy by default (`temperature=0.0` + a fixed seed) — reproducibility for eval.
    seed: int | None = 0
    #: `None` for reading steps' screen (naturalness is the measurement); set for
    #: constrained volume generation.
    grammar: str | None = None
    #: Model-specific stop sequences. Empty by default — verify against MiniCPM5's own
    #: chat template before relying on it (SPEC §9 Phase 0).
    stop: tuple[str, ...] = ()
    timeout: float = DEFAULT_TIMEOUT
    #: Extra body fields passed through verbatim, so a MiniCPM5-specific need discovered
    #: in Phase 0 (e.g. a thinking-mode toggle) doesn't require touching this module.
    extra: dict = field(default_factory=dict)

    def __call__(self, system: str, user: str) -> str:
        body: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.grammar is not None:
            body["grammar"] = self.grammar
        if self.stop:
            body["stop"] = list(self.stop)
        body.update(self.extra)

        response = self._post("/v1/chat/completions", body)
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        if not content:
            # Fail loud: a silent "" would be scored as a NOP and quietly depress
            # whichever system happened to hit this. If the server also returned a
            # reasoning trace, surface its length — that usually means generation was
            # truncated mid-thought before an answer, not that the model chose silence.
            reasoning = message.get("reasoning_content") or ""
            hint = (
                f" (reasoning_content present, {len(reasoning)} chars — likely "
                "truncated before an answer; raise max_tokens)"
                if reasoning
                else ""
            )
            raise RuntimeError(f"empty content from llama-server{hint}")
        return content

    def health(self) -> bool:
        try:
            self._get("/health")
        except RuntimeError:
            return False
        return True

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(req)

    def _get(self, path: str) -> dict:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        return self._send(req)

    def _send(self, req: request.Request) -> dict:
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"llama-server {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"cannot reach llama-server at {self.base_url} — is it running?"
            ) from exc
