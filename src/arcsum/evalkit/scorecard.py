"""One self-describing artifact per evaluated checkpoint, and a comparison that REFUSES
rather than warns.

**The uniform `Check` triple is borrowed from IBM Mellea's `ValidationResult`**
(`result: bool | None`, `reason: str`, `score: float | None`). The framework itself was
evaluated and declined for this project — Mellea validates a single generation against
requirements and has no batch evaluation, aggregate scoring, paired statistics or
regression tracking, which is the entire content of `metrics/stats.py`; and its
model-backed validators are Granite-coupled with a ~3B floor against a 0.8B student with a
3% latency margin. The SHAPE is the good idea and it costs nothing: every instrument
reports the same triple, so the scorecard can treat ROUGE, churn, grounding and a judge
identically, and a new instrument needs no new plumbing.

**Why `result` is tri-state.** `None` means WITHHELD, not "false". `metrics/stats.py`
already refuses to render a verdict below `min_n`, on the argument that a directional
number dressed up as a gate is worse than no number. Collapsing withheld into failure
would throw that away at the aggregation layer.

**Why comparison refuses.** Every scorecard carries the `Provenance` that produced it. Two
scorecards are comparable only when their `comparison_key()` values match — that is,
when they differ ONLY in the model. This is a hard error because a warning is exactly what
was available on 2026-09-02 and did not help: a model card recorded `v5` at 5/27 on "an
independent 27-scenario probe" while the same checkpoint measured 3/27 here, the probe
corpus having been regenerated in between, and nothing flagged the two as incomparable.

**What a scorecard cannot promise.** It records the configuration a measurement ran under;
it cannot know whether that configuration is the one the PRODUCT uses. On 2026-09-02 every
gate ran with `cache_prompt: false` while the shipped demo ran with the KV cache live
across calls, and a checkpoint that churned badly in the deployed configuration passed
everything. `Provenance.generation` therefore records the cache setting, and
`deployment_mismatch()` below exists to make that specific divergence loud — but the
general problem is a discipline, not a data structure: measure the config you ship.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arcsum.evalkit.provenance import Provenance

#: Generation settings whose value in a measurement should match the value the product
#: runs. Divergence here does not invalidate a comparison BETWEEN scorecards — both sides
#: share it — but it does invalidate the inference "this number predicts deployed
#: behaviour", which is the inference that failed on 2026-09-02.
DEPLOYMENT_SENSITIVE = ("cache_prompt", "protocol", "repeat_penalty", "temperature")


@dataclass(frozen=True)
class Check:
    """One instrument's verdict. Shape borrowed from Mellea's `ValidationResult`.

    `score` is the comparable number; `result` is the gate verdict when the instrument is
    a gate, and `None` when it is purely descriptive (a churn rate is a measurement, not a
    pass/fail) or when a gate is withheld. `n` travels with every check because a rate
    without its denominator is how a length-based curation metric scored a confabulation
    as a success.
    """

    name: str
    result: bool | None
    reason: str = ""
    score: float | None = None
    n: int | None = None
    artifact: str = ""

    @property
    def verdict(self) -> str:
        return {True: "PASS", False: "FAIL", None: "—"}[self.result]


@dataclass
class Scorecard:
    provenance: Provenance
    checks: list[Check] = field(default_factory=list)
    #: Per-unit rows behind the aggregates, keyed by instrument. Persisted because the
    #: FIRST version of this class stored aggregates only, and the question that
    #: immediately mattered — is this build more faithful, or merely asserting less? —
    #: cannot be answered from a rate and a denominator. An aggregate that cannot be
    #: drilled into is a number you have to take on trust, which is the thing this whole
    #: package exists to stop.
    details: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def add(self, check: Check) -> Scorecard:
        self.checks.append(check)
        return self

    def attach(self, instrument: str, rows: list[dict[str, Any]]) -> Scorecard:
        """Record the per-unit rows an instrument aggregated over."""
        self.details[instrument] = rows
        return self

    def get(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    @property
    def gates(self) -> list[Check]:
        return [c for c in self.checks if c.result is not None]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.result is False]

    @property
    def withheld(self) -> list[Check]:
        """Gate-shaped checks with no verdict. Reported separately and never counted as
        passes — `ship_decision` treats an unmeasured gate as unmet."""
        return [c for c in self.checks if c.result is None and c.n is not None]

    def deployment_mismatch(self, deployed: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
        """Deployment-sensitive settings where this measurement disagrees with `deployed`.

        Call this with the settings the PRODUCT actually runs. A non-empty result means
        the scorecard does not predict deployed behaviour, however good its numbers are.
        """
        gen = self.provenance.generation
        return {
            k: (gen.get(k), deployed[k])
            for k in DEPLOYMENT_SENSITIVE
            if k in deployed and gen.get(k) != deployed[k]
        }

    def to_json(self) -> str:
        return json.dumps(
            {"provenance": asdict(self.provenance),
             "comparison_key": self.provenance.comparison_key(),
             "generated_at": self.generated_at,
             "checks": [asdict(c) for c in self.checks],
             "details": self.details},
            ensure_ascii=False, indent=1, default=str,
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


class IncomparableScorecards(RuntimeError):
    """Raised when two scorecards differ in something other than the model.

    Deliberately an exception rather than a returned flag: the caller that would ignore a
    flag is the caller this guards against.
    """


def assert_comparable(a: Scorecard, b: Scorecard) -> None:
    if a.provenance.comparison_key() == b.provenance.comparison_key():
        return
    diffs = a.provenance.differences(b.provenance)
    detail = "\n".join(f"    {k}: {va!r} != {vb!r}" for k, (va, vb) in diffs.items())
    raise IncomparableScorecards(
        "These scorecards differ in more than the model under test, so their numbers are "
        "not comparable:\n" + detail + "\n  Re-measure both sides under one configuration."
    )


@dataclass(frozen=True)
class Delta:
    name: str
    a: float | None
    b: float | None
    verdict_a: str
    verdict_b: str

    @property
    def change(self) -> float | None:
        return None if self.a is None or self.b is None else self.b - self.a

    @property
    def regressed_gate(self) -> bool:
        return self.verdict_a == "PASS" and self.verdict_b == "FAIL"


def compare(a: Scorecard, b: Scorecard) -> list[Delta]:
    """Per-check deltas, `a` -> `b`. Refuses when the two are not comparable.

    Checks present in only one scorecard are still returned, with `None` on the missing
    side — a checkpoint measured on fewer instruments than its predecessor is a real and
    important difference, and silently dropping those rows is how a regression hides.
    """
    assert_comparable(a, b)
    names = list(dict.fromkeys([c.name for c in a.checks] + [c.name for c in b.checks]))
    out = []
    for name in names:
        ca, cb = a.get(name), b.get(name)
        out.append(Delta(
            name=name,
            a=ca.score if ca else None,
            b=cb.score if cb else None,
            verdict_a=ca.verdict if ca else "absent",
            verdict_b=cb.verdict if cb else "absent",
        ))
    return out
