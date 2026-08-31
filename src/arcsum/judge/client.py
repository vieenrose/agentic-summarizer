"""Stdlib-only judge HTTP client, contamination refusal, and spend tracking (SPEC
§5.1).

**Local judges only, for now.** A hosted API judge needs a concrete provider contract
(auth scheme, pricing table, endpoint shape) that no key is configured for in this
environment — adding an untested integration for a provider nobody has credentials for
would be speculative, not useful. The `local:<port>/<name>` scheme (any local
`llama-server`-compatible endpoint) is what the harness actually needs and can test.

**Contamination is refused BEFORE any spend**, via `check_judge`, called at the top of
every judge call. SPEC §5.1's rule constrains WHICH model, not whether to judge at
all: the judge must be neither Qwen-family (authored the reference summaries, §2.2
stage 3) nor Gemma-family (translated all corpus text, §2.2 stage 2) — two
INDEPENDENT a-priori sources, unlike the prior project's single-family exclusion.
`DISQUALIFIED_EMPIRICAL` is the second, orthogonal source: a judge can be
provenance-clean and still fail on measured behaviour (e.g. answering SUPPORTED to
every probe case regardless of content) — see `judge.selftest`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib import error, request

#: Named constants with the reason recorded, so nobody reintroduces a disqualified
#: family from the spec by mistake.
CONTAMINATED_FAMILIES: dict[str, str] = {
    "qwen": "authored the reference summaries (SPEC §2.2 stage 3)",
    "gemma": "translated all corpus text (SPEC §2.2 stage 2)",
}

#: Populated by running `judge.selftest` and recording a measured failure — never by
#: assumption. Empty by default.
DISQUALIFIED_EMPIRICAL: dict[str, str] = {}


class ContaminatedJudgeError(ValueError):
    """A contaminated or empirically-disqualified judge must never be called at all,
    not merely flagged after the fact."""


def check_judge(model: str) -> None:
    """Raise if `model` is disqualified, a-priori or empirically. Call this before
    every judge request — refusal-before-spend, not refusal-after-the-fact."""
    lower = model.lower()
    for family, reason in CONTAMINATED_FAMILIES.items():
        if family in lower:
            raise ContaminatedJudgeError(f"{model!r} is {family}-family: {reason}")
    if model in DISQUALIFIED_EMPIRICAL:
        raise ContaminatedJudgeError(
            f"{model!r} failed the judge selftest: {DISQUALIFIED_EMPIRICAL[model]}"
        )


class JudgeBudgetExceeded(RuntimeError):
    """A judged eval is a loop over claims x meetings x systems — an unguarded bug is
    an unbounded bill. Raised BEFORE the request that would exceed the budget."""


@dataclass
class Spend:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int, *, usd: float = 0.0) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += usd
        self.by_model[model] = self.by_model.get(model, 0.0) + usd

    def report(self) -> str:
        lines = [
            f"{self.calls} calls, {self.input_tokens}+{self.output_tokens} tokens, ${self.usd:.4f}"
        ]
        for model, usd in sorted(self.by_model.items()):
            lines.append(f"  {model}: ${usd:.4f}")
        return "\n".join(lines)


_LOCAL_PREFIX = "local:"


def resolve_local_url(model: str) -> tuple[str, str] | None:
    """`local:<port>/<name>` -> `(base_url, name)`. `None` if `model` is not in the
    local scheme — frozen local weights are what the judge relies on for
    reproducibility: "a provider can change a hosted model without notice, which
    silently invalidates every prior comparison."""
    if not model.startswith(_LOCAL_PREFIX):
        return None
    rest = model[len(_LOCAL_PREFIX) :]
    port, _, name = rest.partition("/")
    return f"http://127.0.0.1:{port}", (name or model)


@dataclass
class JudgeClient:
    """**Forced greedy** (`temperature=0.0`) on every call: "a stochastic judge is a
    rubber yardstick" — a judge that flips its own verdict on identical input cannot
    back a faithfulness gate, and SPEC §5.1's 3x majority vote (see `judge.faith`)
    exists precisely because this was measured to happen.
    """

    budget_usd: float = 5.0
    max_tokens: int = 3000
    timeout: float = 300.0
    spend: Spend = field(default_factory=Spend)

    def __call__(self, model: str, system: str, user: str) -> str:
        check_judge(model)
        if self.spend.usd >= self.budget_usd:
            raise JudgeBudgetExceeded(
                f"judge budget exhausted: ${self.spend.usd:.2f} >= ${self.budget_usd:.2f}"
            )

        resolved = resolve_local_url(model)
        if resolved is None:
            raise ValueError(
                f"{model!r} is not a local:<port>/<name> judge — no hosted provider is configured"
            )
        base_url, _name = resolved

        content = self._post_with_retry(base_url, system, user)
        # Local judges are free by construction: frozen weights, no metered API.
        self.spend.add(model, 0, 0, usd=0.0)
        return content

    def _post_with_retry(self, base_url: str, system: str, user: str, *, attempt: int = 0) -> str:
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens * (attempt + 1),
        }
        if attempt >= 1:
            # A reasoning judge can spend its ENTIRE budget in the reasoning channel and
            # return empty `content`. Measured 2026-08-29 against gpt-oss-20b: 21 of 40
            # baseline meetings failed this way, and because the baseline's summaries are
            # ~9x longer (median 5,087 chars vs 562) the losses were systematic, not
            # random -- G2 ended up comparing only the control arm's SHORTEST outputs.
            # Retrying with reasoning capped is what makes the retry meaningfully
            # different; simply re-asking a temperature=0 model reproduces the failure.
            body["chat_template_kwargs"] = {"reasoning_effort": "low"}
        req = request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"judge {exc.code}: {body_text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"cannot reach judge at {base_url} — is it running?") from exc

        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        if not content:
            if attempt >= 1:
                # Fail loud: scoring this as "missing" would quietly depress
                # whichever system happened to draw the flaky call.
                raise RuntimeError("empty content from judge after retry")
            return self._post_with_retry(base_url, system, user, attempt=attempt + 1)
        return content
