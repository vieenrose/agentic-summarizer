"""Pins the `llama-server` HTTP client's wire contract. The network is stubbed, not
mocked at the library level: `urllib.request.urlopen` is patched directly, so
assertions are on the ACTUAL JSON that would go on the wire.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from arcsum.backends.llama_server import LlamaServer


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _capture_payload(monkeypatch: pytest.MonkeyPatch, body: dict) -> list:
    """Stub `urlopen` to return `body` and record every request it was called with."""
    captured: list = []

    def fake_urlopen(req, timeout=None):
        captured.append(req)
        return _FakeResponse(body)

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    return captured


OK_RESPONSE = {"choices": [{"message": {"content": "NOP"}}]}


def test_call_matches_the_modelfn_contract() -> None:
    """(system, user) -> str -- the whole abstraction `arcsum.agent` depends on."""
    assert callable(LlamaServer())


def test_successful_call_returns_the_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_payload(monkeypatch, OK_RESPONSE)
    client = LlamaServer()
    assert client("SYS", "USER") == "NOP"


def test_request_sends_system_and_user_as_chat_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer()("SYS TEXT", "USER TEXT")
    body = json.loads(captured[0].data)
    assert body["messages"] == [
        {"role": "system", "content": "SYS TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]


def test_request_posts_to_the_chat_completions_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer(base_url="http://127.0.0.1:9999")("s", "u")
    assert captured[0].full_url == "http://127.0.0.1:9999/v1/chat/completions"
    assert captured[0].get_method() == "POST"


def test_default_is_greedy_and_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproducibility for eval: a stochastic default would be a rubber yardstick."""
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer()("s", "u")
    body = json.loads(captured[0].data)
    assert body["temperature"] == 0.0
    assert body["seed"] == 0


def test_seed_none_omits_the_seed_field(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer(seed=None)("s", "u")
    body = json.loads(captured[0].data)
    assert "seed" not in body


def test_grammar_omitted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grammar off by default -- the screen's whole signal is whether the model
    naturally emits valid ops; a grammar would hide that."""
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer()("s", "u")
    body = json.loads(captured[0].data)
    assert "grammar" not in body


def test_grammar_included_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer(grammar='root ::= "NOP"')("s", "u")
    body = json.loads(captured[0].data)
    assert body["grammar"] == 'root ::= "NOP"'


def test_stop_omitted_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer()("s", "u")
    body = json.loads(captured[0].data)
    assert "stop" not in body


def test_stop_included_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer(stop=("<end>", "<eos>"))("s", "u")
    body = json.loads(captured[0].data)
    assert body["stop"] == ["<end>", "<eos>"]


def test_extra_fields_are_merged_into_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch for a MiniCPM5-specific need discovered in Phase 0, without
    touching this module."""
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer(extra={"chat_template_kwargs": {"enable_thinking": False}})("s", "u")
    body = json.loads(captured[0].data)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_max_tokens_is_instance_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two configured instances, not a per-call override -- ModelFn stays (str,str)->str."""
    captured = _capture_payload(monkeypatch, OK_RESPONSE)
    LlamaServer(max_tokens=1200)("s", "u")
    body = json.loads(captured[0].data)
    assert body["max_tokens"] == 1200


def test_empty_content_raises_without_a_reasoning_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_payload(monkeypatch, {"choices": [{"message": {"content": ""}}]})
    with pytest.raises(RuntimeError, match="empty content"):
        LlamaServer()("s", "u")


def test_empty_content_with_reasoning_hints_at_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silent '' would be scored as a NOP and quietly depress metrics -- fail loud,
    and say WHY when there's a clue (reasoning present but no answer emitted)."""
    _capture_payload(
        monkeypatch,
        {"choices": [{"message": {"content": "", "reasoning_content": "thinking..."}}]},
    )
    with pytest.raises(RuntimeError, match="reasoning_content present"):
        LlamaServer()("s", "u")


def test_http_error_is_surfaced_with_code_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error", {}, io.BytesIO(b"boom")
        )

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="llama-server 500: boom"):
        LlamaServer()("s", "u")


def test_url_error_is_surfaced_as_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="is it running"):
        LlamaServer(base_url="http://127.0.0.1:8080")("s", "u")


def test_health_true_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_payload(monkeypatch, {"status": "ok"})
    assert LlamaServer().health() is True


def test_health_false_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    assert LlamaServer().health() is False


def test_health_uses_the_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch, {"status": "ok"})
    LlamaServer(base_url="http://127.0.0.1:8080").health()
    assert captured[0].full_url == "http://127.0.0.1:8080/health"
    assert captured[0].get_method() == "GET"


def test_timeout_is_passed_to_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_timeouts: list = []

    def fake_urlopen(req, timeout=None):
        seen_timeouts.append(timeout)
        return _FakeResponse(OK_RESPONSE)

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    LlamaServer(timeout=42.0)("s", "u")
    assert seen_timeouts == [42.0]


def test_op_grammar_is_a_nonempty_string() -> None:
    from arcsum.backends.llama_server import OP_GRAMMAR

    assert isinstance(OP_GRAMMAR, str)
    assert "ADD" in OP_GRAMMAR
    assert "DROP" in OP_GRAMMAR
    assert "ARC" in OP_GRAMMAR
    assert "NOP" in OP_GRAMMAR
