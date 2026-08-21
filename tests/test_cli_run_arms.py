"""Pins `arcsum.cli.run_arms`: both the agent and the fair map-reduce baseline run
against the SAME model over the SAME corpus, with `urllib.request.urlopen` stubbed
exactly as `test_backends.py` stubs it.
"""

from __future__ import annotations

import json

import pytest

from arcsum.backends.llama_server import LlamaServer
from arcsum.cli.run_arms import build_parser, main, run_both_arms
from arcsum.prompts import (
    map_system_prompt,
    reduce_system_prompt,
    step_system_prompt,
    synth_system_prompt,
)

TRANSCRIPT = "S1: 市長已核准搬遷案。\nS2: 議會將於下週表決預算案。\n"

_SYNTH_PROSE = "市長已核准搬遷案，議會將於下週表決預算案，會議討論了多項市政議題。"
_MAP_PROSE = "市長核准搬遷案。"
_REDUCE_PROSE = "本次會議市長核准搬遷案，議會將表決預算案。"


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _stub_both_arms(monkeypatch: pytest.MonkeyPatch) -> list:
    captured: list = []

    def fake_urlopen(req, timeout=None):
        captured.append(req)
        body = json.loads(req.data.decode("utf-8"))
        system = body["messages"][0]["content"]
        if system == step_system_prompt():
            content = "NOP"
        elif system == synth_system_prompt():
            content = _SYNTH_PROSE
        elif system == map_system_prompt():
            content = _MAP_PROSE
        elif system == reduce_system_prompt():
            content = _REDUCE_PROSE
        else:
            raise AssertionError(f"unexpected system prompt: {system!r}")
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    return captured


def test_run_both_arms_produces_paired_meetings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_both_arms(monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")

    agent_pairs, baseline_pairs, skipped = run_both_arms(
        corpus,
        {"m1": "reference text"},
        step_model=LlamaServer(),
        synth_model=LlamaServer(),
        reduce_model=LlamaServer(),
    )

    assert skipped == []
    assert len(agent_pairs) == 1
    assert len(baseline_pairs) == 1
    assert agent_pairs[0]["meeting_id"] == "m1"
    assert agent_pairs[0]["reference"] == "reference text"
    assert agent_pairs[0]["source"] == TRANSCRIPT
    assert agent_pairs[0]["candidate"] == _SYNTH_PROSE


def test_run_both_arms_uses_structurally_different_candidates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent's candidate comes from SYNTHESIZE; the baseline's comes from the
    single-window map (no reduce needed at 1 window) -- these must not collapse to the
    same call path, or the baseline stops being a structurally different opponent."""
    _stub_both_arms(monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")

    agent_pairs, baseline_pairs, _ = run_both_arms(
        corpus,
        {"m1": "reference text"},
        step_model=LlamaServer(),
        synth_model=LlamaServer(),
        reduce_model=LlamaServer(),
    )

    assert agent_pairs[0]["candidate"] == _SYNTH_PROSE
    assert baseline_pairs[0]["candidate"] == _MAP_PROSE


def test_run_both_arms_skips_meetings_with_no_reference(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_both_arms(monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")
    (corpus / "m2.txt").write_text(TRANSCRIPT, encoding="utf-8")

    agent_pairs, baseline_pairs, skipped = run_both_arms(
        corpus,
        {"m1": "reference text"},
        step_model=LlamaServer(),
        synth_model=LlamaServer(),
        reduce_model=LlamaServer(),
    )

    assert skipped == ["m2"]
    assert len(agent_pairs) == 1
    assert len(baseline_pairs) == 1


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_references_and_out_paths() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["corpus/"])


def test_main_writes_score_ready_pairs_files(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_both_arms(monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")
    refs_path = tmp_path / "refs.json"
    refs_path.write_text(json.dumps({"m1": "reference text"}), encoding="utf-8")
    out_agent = tmp_path / "agent_pairs.json"
    out_baseline = tmp_path / "baseline_pairs.json"

    rc = main(
        [
            str(corpus),
            "--references",
            str(refs_path),
            "--out-agent",
            str(out_agent),
            "--out-baseline",
            str(out_baseline),
        ]
    )

    assert rc == 0
    agent_pairs = json.loads(out_agent.read_text(encoding="utf-8"))
    baseline_pairs = json.loads(out_baseline.read_text(encoding="utf-8"))
    assert agent_pairs[0]["meeting_id"] == "m1"
    assert baseline_pairs[0]["meeting_id"] == "m1"

    # Each pairs file must be directly consumable by arcsum-score.
    from arcsum.cli.score import score_pairs

    scored = score_pairs(agent_pairs, system="agent")
    assert scored[0]["meeting_id"] == "m1"
