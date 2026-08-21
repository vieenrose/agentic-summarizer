"""Pins `arcsum.cli.gen_traces`: end-to-end `run_agent` over a real `LlamaServer`
`ModelFn`, with `urllib.request.urlopen` stubbed exactly as `test_backends.py` stubs
it -- no real network or model needed to exercise the full step loop.
"""

from __future__ import annotations

import json

import pytest

from arcsum.cli.gen_traces import build_parser, gen_trace_for_meeting, main, trace_to_sft_rows

TRANSCRIPT = "S1: 市長已核准搬遷案。\nS2: 議會將於下週表決預算案。\n"

_SYNTH_PROSE = "市長已核准搬遷案，議會將於下週表決預算案，會議討論了多項市政議題。"


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _stub_agent_and_synth(monkeypatch: pytest.MonkeyPatch) -> list:
    """Reading steps see `CHUNK:` in the user turn; the SYNTHESIZE call does not --
    route each to a distinct, always-valid canned response."""
    captured: list = []

    def fake_urlopen(req, timeout=None):
        captured.append(req)
        body = json.loads(req.data.decode("utf-8"))
        user = body["messages"][1]["content"]
        content = "NOP" if "CHUNK:" in user else _SYNTH_PROSE
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    return captured


def test_gen_trace_for_meeting_runs_the_full_agent_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from arcsum.backends.llama_server import LlamaServer

    _stub_agent_and_synth(monkeypatch)
    trace = gen_trace_for_meeting(TRANSCRIPT, LlamaServer())
    assert len(trace.steps) >= 1
    assert trace.synthesis is not None
    assert trace.synthesis.prose.text == _SYNTH_PROSE


def test_trace_to_sft_rows_matches_sft_sample_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from arcsum.backends.llama_server import LlamaServer
    from arcsum.supervision.sft import build_samples

    _stub_agent_and_synth(monkeypatch)
    trace = gen_trace_for_meeting(TRANSCRIPT, LlamaServer())

    rows = trace_to_sft_rows("m1", trace)
    samples = build_samples("m1", trace)

    assert len(rows) == len(samples)
    for row, sample in zip(rows, samples, strict=True):
        assert row == {
            "meeting": sample.meeting,
            "step": sample.step,
            "prompt_version": sample.prompt_version,
            "system": sample.system,
            "prompt": sample.prompt,
            "completion": sample.completion,
            "is_nop": sample.is_nop,
        }


def test_trace_to_sft_rows_last_row_is_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    from arcsum.backends.llama_server import LlamaServer

    _stub_agent_and_synth(monkeypatch)
    trace = gen_trace_for_meeting(TRANSCRIPT, LlamaServer())
    rows = trace_to_sft_rows("m1", trace)
    assert rows[-1]["completion"] == _SYNTH_PROSE
    assert rows[-1]["is_nop"] is False


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_out() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["corpus/"])


def test_main_writes_rows_and_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_agent_and_synth(monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")
    (corpus / "m2.txt").write_text(TRANSCRIPT, encoding="utf-8")
    out_path = tmp_path / "traces.jsonl"
    report_path = tmp_path / "report.json"

    rc = main([str(corpus), "--out", str(out_path), "--report-out", str(report_path)])

    assert rc == 0
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert {r["meeting"] for r in rows} == {"m1", "m2"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total_steps"] >= 2


def test_main_returns_nonzero_with_no_transcripts(tmp_path) -> None:
    corpus = tmp_path / "empty"
    corpus.mkdir()
    rc = main([str(corpus), "--out", str(tmp_path / "traces.jsonl")])
    assert rc == 1


def test_main_uses_separate_synth_url_when_given(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []
    from arcsum.backends import llama_server as llama_server_module

    real_send = llama_server_module.LlamaServer._send

    def fake_urlopen(req, timeout=None):
        captured_urls.append(req.full_url)
        body = json.loads(req.data.decode("utf-8"))
        user = body["messages"][1]["content"]
        content = "NOP" if "CHUNK:" in user else _SYNTH_PROSE
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    assert real_send  # keep the import used

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")

    main(
        [
            str(corpus),
            "--url",
            "http://127.0.0.1:8080",
            "--synth-url",
            "http://127.0.0.1:9090",
            "--out",
            str(tmp_path / "traces.jsonl"),
        ]
    )

    assert any("127.0.0.1:9090" in u for u in captured_urls)
    assert any("127.0.0.1:8080" in u for u in captured_urls)
