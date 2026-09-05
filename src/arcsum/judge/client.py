"""Stdlib-only judge HTTP client, contamination refusal, and spend tracking (SPEC
§5.1).

**Two schemes: `local:<port>/<name>` and `opencode:<model>`.** Local (any
`llama-server`-compatible endpoint) is the default and the reproducible one. The hosted
scheme exists because SPEC §5.1 needs a judge from a THIRD family — neither Qwen nor
Gemma — and the local shelf offers only `gpt-oss-20b`, whose measured failure mode
(spending its whole budget in the reasoning channel, returning empty `content`) once cost
21 of 40 meetings and did so systematically on the longest summaries.

**A hosted judge is NOT reproducible and its results carry that caveat**: a provider can
change a hosted model behind a stable name without notice, silently invalidating every
prior comparison. Record the model id with the result, never compare a hosted number across
dates, and prefer local weights whenever a clean third-family local judge exists.

**The hosted transport needs a `User-Agent`.** Measured 2026-09-04: `urllib`'s default UA is
rejected by the provider's edge with an opaque `HTTP 403: error code: 1010` — a Cloudflare
block, not an auth failure, and indistinguishable from a bad key unless you read the body.
`/v1/models` succeeds without one, so the obvious "my key works" check does NOT catch it.

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
_HOSTED_PREFIX = "opencode:"

#: OpenAI-compatible endpoint for the hosted third-family judges.
HOSTED_BASE_URL = "https://opencode.ai/zen/go/v1"

#: Required — see the module docstring. Any value works; its absence does not.
HOSTED_USER_AGENT = "arcsum-judge/1.0"

#: The environment variable carrying the hosted key. Read at call time and never stored,
#: logged, or written into an artifact: a judged run's provenance records the MODEL, not
#: the credential.
HOSTED_KEY_ENV = "OPENCODE_API_KEY"

#: **The provider's models do NOT all speak one protocol**, and the failure is silent-ish:
#: posting a `/responses`-only model to `/chat/completions` returns an opaque
#: `HTTP 500 Internal server error` that looks like an outage. Measured 2026-09-04 —
#: `muse-spark-1.3-contributor` 500s on `/chat/completions` and answers correctly on
#: `/responses`. Anything absent from this map uses the OpenAI-compatible default.
HOSTED_ENDPOINTS: dict[str, str] = {
    "muse-spark-1.3-contributor": "responses",
    "muse-spark-1.2-contributor": "responses",
}

#: Reasoning models spend most of their output allowance before writing an answer:
#: `muse-spark-1.3-contributor` used **828 of 860** output tokens on reasoning for a
#: three-claim judgement, so a 600-token ceiling produced an EMPTY response that reads
#: exactly like a model failure. Reasoning-heavy judges get a raised floor.
HOSTED_MIN_OUTPUT_TOKENS = 2000


def resolve_hosted_model(model: str) -> str | None:
    """`opencode:<model>` -> `<model>`; `None` if not the hosted scheme."""
    if not model.startswith(_HOSTED_PREFIX):
        return None
    return model[len(_HOSTED_PREFIX) :] or None


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

        hosted = resolve_hosted_model(model)
        if hosted is not None:
            content = self._post_hosted(hosted, system, user)
            # Metered by the provider, not priced here: no pricing table is configured, so
            # recording a fabricated cost would be worse than recording none. The budget
            # guard above still bounds the CALL COUNT, which is the runaway risk.
            self.spend.add(model, 0, 0, usd=0.0)
            return content

        resolved = resolve_local_url(model)
        if resolved is None:
            raise ValueError(
                f"{model!r} is neither local:<port>/<name> nor opencode:<model>"
            )
        base_url, _name = resolved

        content = self._post_with_retry(base_url, system, user)
        # Local judges are free by construction: frozen weights, no metered API.
        self.spend.add(model, 0, 0, usd=0.0)
        return content

    def _post_responses(
        self, model: str, system: str, user: str, key: str, budget: int, *, attempt: int = 0
    ) -> str:
        """OpenAI **Responses** shape: `instructions` + `input`, and the answer arrives in
        `output[]` as a `message` entry ALONGSIDE a `reasoning` entry. Reading only
        `output_text` yields an empty string on these models, which is why this walks the
        `output` list rather than trusting the convenience field."""
        body = {
            "model": model,
            "instructions": system,
            "input": user,
            "temperature": 0.0,
            "max_output_tokens": budget,
        }
        req = request.Request(
            f"{HOSTED_BASE_URL}/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": HOSTED_USER_AGENT,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"hosted judge {exc.code}: {body_text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"cannot reach hosted judge: {exc}") from exc

        if response.get("error"):
            raise RuntimeError(f"hosted judge refused: {json.dumps(response['error'])[:300]}")
        parts: list[str] = []
        for entry in response.get("output", []):
            if entry.get("type") != "message":
                continue  # skip the `reasoning` entry
            for chunk in entry.get("content") or []:
                if chunk.get("text"):
                    parts.append(chunk["text"])
        content = "".join(parts).strip() or (response.get("output_text") or "").strip()
        if not content:
            incomplete = response.get("incomplete_details") or {}
            # **A reasoning judge can spend the WHOLE budget thinking and emit nothing.**
            # Measured on this provider: `muse-spark-1.3-contributor` failed 53 of 80 real
            # judging cases with `incomplete: {'reason': 'max_output_tokens'}` at a 2,000-token
            # ceiling — it had used 828 of 860 tokens on reasoning even for a trivial
            # three-claim probe. The `/chat/completions` path already retried with reasoning
            # capped; this path did not, which is why the failure only appeared at scale.
            #
            # Retrying with a much LARGER budget (not a smaller one) is the fix here: the
            # answer is short, the reasoning is not, so the ceiling has to clear the reasoning
            # before the answer can be written at all.
            if attempt < 1 and incomplete.get("reason") == "max_output_tokens":
                return self._post_responses(model, system, user, key, budget * 4,
                                            attempt=attempt + 1)
            raise RuntimeError(
                f"empty content from hosted judge {model!r} (incomplete: {incomplete})"
            )
        return content

    def _post_hosted(self, model: str, system: str, user: str, *, attempt: int = 0) -> str:
        import os

        key = os.environ.get(HOSTED_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"{HOSTED_KEY_ENV} is not set — a hosted judge needs a credential, and this "
                f"harness never embeds one"
            )
        shape = HOSTED_ENDPOINTS.get(model, "chat")
        budget = max(self.max_tokens * (attempt + 1), HOSTED_MIN_OUTPUT_TOKENS)
        if shape == "responses":
            return self._post_responses(model, system, user, key, budget)

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": budget,
        }
        if attempt >= 1:
            # Same failure, same fix as the local path: a REASONING judge can spend its
            # entire budget in the reasoning channel and return empty `content`. Measured
            # 2026-09-04 on this provider: `mimo-v2.5` and `hy3` both returned empty content
            # on the first attempt while `deepseek-v4-flash` and `longcat-2.0` answered
            # normally. Re-asking a temperature=0 model unchanged reproduces the failure by
            # construction, so the retry must actually differ — cap the reasoning and raise
            # the ceiling.
            body["chat_template_kwargs"] = {"reasoning_effort": "low"}
            body["reasoning_effort"] = "low"
        req = request.Request(
            f"{HOSTED_BASE_URL}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                # Absence of this header is a 403/1010 at the edge, not an auth error.
                "User-Agent": HOSTED_USER_AGENT,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"hosted judge {exc.code}: {body_text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"cannot reach hosted judge: {exc}") from exc

        # The provider reports quota and data-policy refusals as a 200 with an error body.
        # Treating that as an empty verdict would silently score the meeting as UNSUPPORTED.
        if "error" in response:
            raise RuntimeError(f"hosted judge refused: {json.dumps(response['error'])[:300]}")
        message = response["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            if attempt >= 1:
                # Fail loud rather than scoring the meeting as an empty verdict: doing the
                # latter silently depresses whichever system drew the flaky call, and that
                # is exactly how G2 once reported "14 vs 11" for a real 18 vs 109.
                raise RuntimeError(
                    f"empty content from hosted judge {model!r} after a reasoning-capped retry"
                )
            return self._post_hosted(model, system, user, attempt=attempt + 1)
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
