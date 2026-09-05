"""Pins `arcsum-eval`. Runs with no GPU, no weights and no network: every HTTP call —
`/props`, `/apply-template` and the completions endpoint — is served by a stub at
`urllib.request.urlopen`, exactly as `test_cli_run_arms.py` does.

The behaviour under test is mostly REFUSAL, because that is what this command adds over
running the instruments by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcsum.cli import eval_all

MEETING = """S1: 市議會今天審議年度預算案。
S2: 總預算為 12 億元，請問委員有沒有意見。
S1: 沒有意見的話，本案通過。
"""


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub(model_path="/served/model.gguf", completion="NOP"):
    """One stub for every endpoint the run touches, dispatching on the URL."""

    def opener(req, *a, **k):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/props"):
            return _Resp({"model_path": model_path})
        if url.endswith("/apply-template"):
            return _Resp({"prompt": "PROMPT"})
        return _Resp({"content": completion, "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    return opener


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "m1.txt").write_text(MEETING, encoding="utf-8")
    return d


def test_refuses_when_the_server_cannot_be_identified(tmp_path, corpus, monkeypatch):
    """The 2026-09-02 incident: a server that failed to bind, with the previous model
    answering. Measuring against an unidentified server is refused outright."""

    def dead(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", dead)
    with pytest.raises(RuntimeError, match="Refusing to record"):
        eval_all.main(
            [
                "--url",
                "http://x:8081",
                "--protocol",
                "tool",
                "--corpus",
                str(corpus),
                "--out",
                str(tmp_path / "s.json"),
            ]
        )


def test_protocol_is_required(tmp_path, corpus):
    """`score_reversals.py` defaulting to 'edit' scored a tool-call checkpoint as 0/27.
    No default is safe, so there is none."""
    with pytest.raises(SystemExit):
        eval_all.main(["--corpus", str(corpus), "--out", str(tmp_path / "s.json")])


def test_refuses_an_empty_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _stub())
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (
        eval_all.main(
            ["--protocol", "tool", "--corpus", str(empty), "--out", str(tmp_path / "s.json")]
        )
        == 1
    )


def test_scorecard_records_the_model_that_answered(tmp_path, corpus, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _stub(model_path="/served/actual.gguf"))
    out = tmp_path / "s.json"
    assert (
        eval_all.main(
            ["--protocol", "tool", "--corpus", str(corpus), "--label", "t", "--out", str(out)]
        )
        == 0
    )
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["provenance"]["model_path"] == "/served/actual.gguf"
    assert blob["comparison_key"]
    names = {c["name"] for c in blob["checks"]}
    assert {"churn_rate", "ungrounded_rate", "clean_meetings"} <= names


def test_deployment_mismatch_is_a_failed_check_not_a_note(tmp_path, corpus, monkeypatch):
    """The measurement pins cache_prompt=false; declaring that the product ships it TRUE
    must surface as a FAIL, because that exact divergence hid a shipped regression."""
    monkeypatch.setattr("urllib.request.urlopen", _stub())
    out = tmp_path / "s.json"
    eval_all.main(
        [
            "--protocol",
            "tool",
            "--corpus",
            str(corpus),
            "--deployed-cache-prompt",
            "true",
            "--out",
            str(out),
        ]
    )
    blob = json.loads(out.read_text(encoding="utf-8"))
    check = next(c for c in blob["checks"] if c["name"] == "deployment_match")
    assert check["result"] is False and "cache_prompt" in check["reason"]


def test_deployment_match_passes_when_configs_agree(tmp_path, corpus, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _stub())
    out = tmp_path / "s.json"
    eval_all.main(
        [
            "--protocol",
            "tool",
            "--corpus",
            str(corpus),
            "--deployed-cache-prompt",
            "false",
            "--out",
            str(out),
        ]
    )
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert next(c for c in blob["checks"] if c["name"] == "deployment_match")["result"] is True


def test_reference_gates_are_read_not_recomputed(tmp_path, corpus, monkeypatch):
    """`metrics/stats.py` owns the paired protocol and the below-min_n withholding; this
    command must not become a second answer to that question."""
    monkeypatch.setattr("urllib.request.urlopen", _stub())
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "gates": [
                    {"gate": "G3_rouge1", "passed": True, "detail": "mean_delta=+0.05"},
                    {"gate": "G4_budget", "passed": None, "detail": "withheld: no device"},
                ],
                "comparisons": [
                    {
                        "metric": "rouge1",
                        "n": 40,
                        "wins": 28,
                        "losses": 12,
                        "mean_delta": 0.047,
                        "p_value": 0.017,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "s.json"
    eval_all.main(
        ["--protocol", "tool", "--corpus", str(corpus), "--report", str(report), "--out", str(out)]
    )
    checks = {c["name"]: c for c in json.loads(out.read_text(encoding="utf-8"))["checks"]}
    assert checks["G3_rouge1"]["result"] is True
    # Withheld stays withheld — never promoted to a pass by passing through this layer.
    assert checks["G4_budget"]["result"] is None
    assert checks["delta_rouge1"]["score"] == pytest.approx(0.047)


def test_every_behaviour_field_a_gate_consumes_is_serialized() -> None:
    """The persisted rows must carry what the gates are computed FROM, not a subset.

    Two things were added to `BehaviourReport` and not to the serializer, and both failed
    silently rather than loudly:

    * `steps` — the denominator every per-step rate is over. Without it consumers substitute
      `chunks`, which differs whenever a step FAILED, so rates are wrong on exactly the runs
      where something went wrong.
    * `prefill_tokens` / `decode_tokens` — G4's inputs. `evalkit.latency` projects wall clock
      from a run's MEASURED token profile precisely because decode length belongs to the
      checkpoint and not the device. The first four RAFT scorecards report 0 for these and
      cannot support a G4 claim.

    Asserted against the dataclass rather than a hand-written list, so a field added to
    `BehaviourReport` in future fails here instead of silently vanishing from disk.
    """
    import dataclasses
    import inspect
    import pathlib

    from arcsum.cli import eval_all
    from arcsum.evalkit.behaviour import BehaviourReport

    # Read the whole module rather than slicing one call: the earlier version partitioned
    # `main`'s source and silently produced an EMPTY haystack when the formatter moved a
    # bracket, which reported every field as missing. A test whose failure mode is "flags
    # everything" is as useless as one that flags nothing.
    persisted = pathlib.Path(inspect.getsourcefile(eval_all)).read_text(encoding="utf-8")
    # Derived properties are recomputed by consumers; raw COUNTS are the ground truth and
    # cannot be recovered once dropped.
    missing = [
        f.name
        for f in dataclasses.fields(BehaviourReport)
        if f.name not in ("meeting",) and f'"{f.name}"' not in persisted
    ]
    assert not missing, f"BehaviourReport fields never written to the scorecard: {missing}"


def test_g8_fails_a_checkpoint_that_says_almost_nothing() -> None:
    """SPEC §5.2.6. The gate set rewarded abstention on five of seven gates.

    Every quality gate here is a RATE over what the model chose to say, so shrinking the
    denominator improves all of them at once. Measured on `rl-v3`, which NOPs 46.2% of chunks
    and starves 17/40 while passing G2/G4/G5/G6/G7: its STARVED meetings score BETTER than its
    healthy ones on both gated metrics — retention 0.955 vs 0.915, 4 churn events vs 15.

    Asserted on the real numbers, so the gate is pinned to the case that motivated it.
    """
    from arcsum.cli.eval_all import (
        MAX_STARVED_FRACTION,
        MAX_UNGROUNDED_RATE,
    )

    # rl-v3, measured: 17/40 starved, 2.5% ungrounded, 0.55 points/chunk.
    starved_frac = 17 / 40
    assert starved_frac > MAX_STARVED_FRACTION, "rl-v3 must FAIL G8's coverage half"
    assert MAX_UNGROUNDED_RATE >= 0.025, "...while passing G6, which is the whole point"

    # The anti-starvation checkpoint, measured: 5/40 starved.
    assert MAX_STARVED_FRACTION >= 5 / 40, "raft-s0-e1 must PASS the coverage half"


def test_g8_is_joint_with_grounding_because_each_half_alone_is_gameable() -> None:
    """Coverage alone is satisfiable by FABRICATING; grounding alone by SILENCE. A build has
    to clear both, which is §5.2.2's rule ("a metric a known defect can improve must be gated
    with the detector for that defect") applied in the opposite direction."""
    import inspect
    import pathlib

    from arcsum.cli import eval_all

    src = pathlib.Path(inspect.getsourcefile(eval_all)).read_text(encoding="utf-8")
    block = src.partition('"g8_coverage"')[2].partition(")\n    )")[0]
    assert "grounding_ok" in block, "G8 must include the grounding term, not just coverage"
    assert "coverage_ok" in block and "density_ok" in block


def test_g8_density_floor_reuses_the_reporting_flags_definition() -> None:
    """The gate and the `starved` flag must not drift apart — restating the threshold is how
    a gate and the instrument it reads stop meaning the same thing."""
    from arcsum.cli.eval_all import MIN_POINTS_PER_CHUNK
    from arcsum.evalkit.behaviour import STARVED_POINTS_PER_CHUNK

    assert MIN_POINTS_PER_CHUNK == STARVED_POINTS_PER_CHUNK
