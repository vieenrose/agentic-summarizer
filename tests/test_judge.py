"""Pins SPEC §5.1's faithfulness judge: contamination refusal before spend, the
budget guard, evidence retrieval and its pinned order, claim splitting, and the
severity tie-break on a split vote.

The network is stubbed, not mocked at the library level: `urllib.request.urlopen` is
patched directly, so assertions are on the actual JSON that would go on the wire.
"""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

from arcsum.judge.client import (
    CONTAMINATED_FAMILIES,
    DISQUALIFIED_EMPIRICAL,
    ContaminatedJudgeError,
    JudgeBudgetExceeded,
    JudgeClient,
    Spend,
    check_judge,
    resolve_local_url,
)
from arcsum.judge.evidence import EVIDENCE_ORDER, Evidence, TranscriptIndex
from arcsum.judge.faith import (
    BulletVerdict,
    MeetingScore,
    faith_prompt,
    judge_meeting,
    parse_verdict,
    split_claims,
)
from arcsum.transcript import Utterance


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _stub(monkeypatch: pytest.MonkeyPatch, content: str) -> list:
    captured: list = []

    def fake_urlopen(req, timeout=None):
        captured.append(req)
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("arcsum.judge.client.request.urlopen", fake_urlopen)
    return captured


# --- resolve_local_url ------------------------------------------------------------------


def test_resolve_local_url_parses_port_and_name() -> None:
    assert resolve_local_url("local:8090/gpt-oss-20b") == ("http://127.0.0.1:8090", "gpt-oss-20b")


def test_resolve_local_url_returns_none_for_non_local_scheme() -> None:
    assert resolve_local_url("openai/gpt-4") is None


def test_resolve_local_url_handles_missing_name() -> None:
    base_url, name = resolve_local_url("local:8090")
    assert base_url == "http://127.0.0.1:8090"
    assert name  # falls back to the whole model string, never empty


# --- check_judge / contamination refusal ------------------------------------------------


def test_qwen_family_is_refused() -> None:
    with pytest.raises(ContaminatedJudgeError, match="qwen"):
        check_judge("local:8080/qwen3.8-27b")


def test_gemma_family_is_refused() -> None:
    with pytest.raises(ContaminatedJudgeError, match="gemma"):
        check_judge("local:8080/gemma-4-31b-it")


def test_contamination_check_is_case_insensitive() -> None:
    with pytest.raises(ContaminatedJudgeError):
        check_judge("local:8080/QWEN3.5-2B")


def test_a_clean_model_passes_the_check() -> None:
    check_judge("local:8080/gpt-oss-20b")  # must not raise


def test_two_independent_contamination_sources_are_named() -> None:
    assert "qwen" in CONTAMINATED_FAMILIES
    assert "gemma" in CONTAMINATED_FAMILIES
    assert CONTAMINATED_FAMILIES["qwen"] != CONTAMINATED_FAMILIES["gemma"]


def test_empirically_disqualified_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(DISQUALIFIED_EMPIRICAL, "local:8080/flaky-judge", "failed the selftest")
    with pytest.raises(ContaminatedJudgeError, match="selftest"):
        check_judge("local:8080/flaky-judge")


def test_client_refuses_a_contaminated_model_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _stub(monkeypatch, "SUPPORTED")
    client = JudgeClient()
    with pytest.raises(ContaminatedJudgeError):
        client("local:8080/qwen3.8-27b", "sys", "user")
    assert captured == []  # refusal happened before any spend


# --- Spend / budget guard ----------------------------------------------------------------


def test_spend_add_accumulates() -> None:
    s = Spend()
    s.add("model-a", 100, 20, usd=0.01)
    s.add("model-b", 50, 10, usd=0.02)
    assert s.calls == 2
    assert s.input_tokens == 150
    assert s.output_tokens == 30
    assert s.usd == pytest.approx(0.03)
    assert s.by_model == {"model-a": pytest.approx(0.01), "model-b": pytest.approx(0.02)}


def test_spend_report_is_a_readable_string() -> None:
    s = Spend()
    s.add("model-a", 10, 5, usd=0.5)
    report = s.report()
    assert "model-a" in report
    assert "1 calls" in report


def test_budget_exceeded_raises_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _stub(monkeypatch, "SUPPORTED")
    client = JudgeClient(budget_usd=0.0)  # already exhausted
    with pytest.raises(JudgeBudgetExceeded):
        client("local:8080/gpt-oss-20b", "sys", "user")
    assert captured == []


# --- JudgeClient: greedy, wire contract, empty-content retry -------------------------


def test_local_judge_is_greedy_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _stub(monkeypatch, "SUPPORTED")
    JudgeClient()("local:8080/gpt-oss-20b", "sys", "user")
    body = json.loads(captured[0].data)
    assert body["temperature"] == 0.0


def test_request_sends_system_and_user_as_chat_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _stub(monkeypatch, "SUPPORTED")
    JudgeClient()("local:8080/gpt-oss-20b", "SYS TEXT", "USER TEXT")
    body = json.loads(captured[0].data)
    assert body["messages"] == [
        {"role": "system", "content": "SYS TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]


def test_request_posts_to_the_resolved_local_port(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _stub(monkeypatch, "SUPPORTED")
    JudgeClient()("local:9999/some-judge", "sys", "user")
    assert captured[0].full_url == "http://127.0.0.1:9999/v1/chat/completions"


def test_successful_call_returns_the_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, "SUPPORTED")
    result = JudgeClient()("local:8080/gpt-oss-20b", "sys", "user")
    assert result == "SUPPORTED"


def test_local_judge_records_zero_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local judges are free by construction: frozen weights, no metered API."""
    _stub(monkeypatch, "SUPPORTED")
    client = JudgeClient()
    client("local:8080/gpt-oss-20b", "sys", "user")
    assert client.spend.usd == 0.0
    assert client.spend.calls == 1


def test_empty_content_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silent '' would be scored as 'missing' and quietly depress whichever system
    was unlucky -- fail loud after one retry, never score it."""
    _stub(monkeypatch, "")
    with pytest.raises(RuntimeError, match="empty content"):
        JudgeClient()("local:8080/gpt-oss-20b", "sys", "user")


def test_empty_content_retry_doubles_the_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _stub(monkeypatch, "")
    with pytest.raises(RuntimeError):
        JudgeClient(max_tokens=100)("local:8080/gpt-oss-20b", "sys", "user")
    assert len(captured) == 2
    first = json.loads(captured[0].data)["max_tokens"]
    second = json.loads(captured[1].data)["max_tokens"]
    assert second > first


def test_an_unrecognised_scheme_is_rejected_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare provider/model string names no transport this client implements. The message
    lists BOTH supported schemes, because the previous wording ("no hosted provider is
    configured") became false once the third-family hosted judge landed."""
    with pytest.raises(ValueError, match="neither local:"):
        JudgeClient()("openai/gpt-4", "sys", "user")


# --- evidence retrieval -----------------------------------------------------------------


def meeting() -> list[Utterance]:
    return [
        Utterance("S1", "我們討論辦公室搬遷案。"),
        Utterance("S2", "建議搬到 B 棟大樓。"),
        Utterance("S1", "議案通過，確定搬遷。"),
        Utterance("S3", "另外討論員工餐廳的菜單更新。"),
    ]


def test_search_returns_lexically_relevant_lines() -> None:
    index = TranscriptIndex(meeting())
    results = index.search("搬遷案")
    assert results
    assert any("搬遷" in e.text for e in results)


def test_search_excludes_zero_overlap_lines() -> None:
    index = TranscriptIndex(meeting())
    results = index.search("完全不相關的詞彙組合")
    assert results == []


def test_evidence_order_is_score_desc_then_line() -> None:
    index = TranscriptIndex(meeting())
    results = index.search("搬遷", top_k=10)
    scores = [e.score for e in results]
    assert scores == sorted(scores, reverse=True)
    # ties (equal score) must be broken by ascending line number
    for a, b in pairwise(results):
        if a.score == b.score:
            assert a.line < b.line


def test_evidence_order_constant_is_pinned() -> None:
    assert EVIDENCE_ORDER == "score_desc_then_line"


def test_evidence_render_returns_the_line_text() -> None:
    e = Evidence(line=0, text="S1: 測試", score=1.0)
    assert e.render() == "S1: 測試"


def test_search_respects_top_k() -> None:
    index = TranscriptIndex(meeting() * 3)  # repeat to guarantee enough matches
    assert len(index.search("搬遷", top_k=2)) <= 2


# --- split_claims --------------------------------------------------------------------


def test_split_claims_splits_on_cjk_sentence_enders() -> None:
    text = "會議討論辦公室搬遷案，最終決議遷至B棟大樓。預算已於會中核准通過。"
    claims = split_claims(text)
    assert len(claims) == 2


def test_split_claims_drops_fragments_below_the_token_floor() -> None:
    text = "好。" + "會議討論辦公室搬遷案並決議通過遷至新址。"
    claims = split_claims(text)
    assert "好。" not in claims
    assert any("搬遷" in c for c in claims)


def test_split_claims_empty_text_yields_no_claims() -> None:
    assert split_claims("") == []


def test_split_claims_handles_text_with_no_sentence_enders() -> None:
    text = "會議討論辦公室搬遷案並決議通過遷至新址"
    assert split_claims(text) == [text]


# --- parse_verdict: last-match-per-key ----------------------------------------------


def test_parse_verdict_finds_the_keyword() -> None:
    assert parse_verdict("The claim is SUPPORTED by the transcript.") == "SUPPORTED"


def test_parse_verdict_is_case_insensitive_but_normalises_to_upper() -> None:
    assert parse_verdict("supported") == "SUPPORTED"


def test_parse_verdict_takes_the_last_match() -> None:
    """A judge that restates then decides ends right -- the LAST mention counts."""
    text = "This might seem UNSUPPORTED at first, but on reflection it is SUPPORTED."
    assert parse_verdict(text) == "SUPPORTED"


def test_parse_verdict_returns_none_when_no_keyword_present() -> None:
    assert parse_verdict("I cannot determine this.") is None


# --- BulletVerdict.majority: the severity tie-break -----------------------------------


def test_majority_returns_the_clear_winner() -> None:
    v = BulletVerdict("claim", ("SUPPORTED", "SUPPORTED", "CONTRADICTED"))
    assert v.majority == "SUPPORTED"


def test_majority_breaks_a_three_way_tie_toward_the_most_severe() -> None:
    """A 0%-inversion requirement must not average away a genuine dissent."""
    v = BulletVerdict("claim", ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED"))
    assert v.majority == "CONTRADICTED"


def test_majority_breaks_a_two_way_tie_toward_the_more_severe() -> None:
    v = BulletVerdict("claim", ("SUPPORTED", "UNSUPPORTED"))
    assert v.majority == "UNSUPPORTED"


def test_majority_unanimous() -> None:
    v = BulletVerdict("claim", ("SUPPORTED", "SUPPORTED", "SUPPORTED"))
    assert v.majority == "SUPPORTED"


# --- faith_prompt / judge_meeting orchestration ---------------------------------------


def test_faith_prompt_includes_the_claim_and_evidence() -> None:
    ev = [Evidence(line=0, text="S1: 搬遷案討論", score=1.0)]
    prompt = faith_prompt("搬遷案已通過", ev)
    assert "搬遷案已通過" in prompt
    assert "S1: 搬遷案討論" in prompt


def test_faith_prompt_handles_no_evidence() -> None:
    prompt = faith_prompt("某個陳述", [])
    assert "某個陳述" in prompt


class _ScriptedJudge:
    """A minimal client double: always returns the same verdict, records call count."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0

    def __call__(self, model: str, system: str, user: str) -> str:
        self.calls += 1
        return self.verdict


def test_judge_meeting_all_supported() -> None:
    prose = "會議討論辦公室搬遷案，最終決議遷至B棟大樓。預算已於會中核准通過。"
    judge = _ScriptedJudge("SUPPORTED")
    score = judge_meeting(prose, meeting(), judge, model="local:8080/x", votes=1)
    assert isinstance(score, MeetingScore)
    assert score.faith_claim == pytest.approx(5.0)
    assert score.inverted == 0
    assert score.unsupported == 0


def test_judge_meeting_all_contradicted_counts_inversions() -> None:
    prose = "會議討論辦公室搬遷案，最終決議遷至B棟大樓。預算已於會中核准通過。"
    judge = _ScriptedJudge("CONTRADICTED")
    score = judge_meeting(prose, meeting(), judge, model="local:8080/x", votes=1)
    assert score.inverted == len(score.bullets)
    assert score.faith_claim == pytest.approx(1.0)


def test_judge_meeting_votes_the_configured_number_of_times_per_claim() -> None:
    prose = "會議討論辦公室搬遷案，最終決議遷至B棟大樓。"
    judge = _ScriptedJudge("SUPPORTED")
    judge_meeting(prose, meeting(), judge, model="local:8080/x", votes=3)
    claims = split_claims(prose)
    assert judge.calls == len(claims) * 3


def test_judge_meeting_empty_prose_yields_no_bullets() -> None:
    """With nothing to judge, faith_claim floors at 1.0 (the scale's minimum) rather
    than 0.0, which would be out of the documented 1-5 range."""
    judge = _ScriptedJudge("SUPPORTED")
    score = judge_meeting("", meeting(), judge, model="local:8080/x")
    assert score.bullets == ()
    assert score.faith_claim == 1.0


def test_inversions_are_a_separate_count_never_folded_into_faith_claim() -> None:
    """SPEC §5.1: a single inverted decision is a product defect, not a fractional
    score penalty -- inverted is a plain int, distinct from the averaged faith_claim."""
    prose = "會議討論辦公室搬遷案。預算已於會中核准通過。"
    judge = _ScriptedJudge("CONTRADICTED")
    score = judge_meeting(prose, meeting(), judge, model="local:8080/x", votes=1)
    assert isinstance(score.inverted, int)
    assert score.inverted > 0


def test_retry_caps_reasoning_effort_after_an_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry must differ from the first attempt in a way that can actually change
    the outcome. A reasoning judge can spend its whole budget in the reasoning channel
    and return empty `content`; at temperature=0 an identical re-ask reproduces that
    exactly. Measured 2026-08-29 on gpt-oss-20b: 21 of 40 baseline meetings failed this
    way, systematically the LONGEST summaries (median 5,087 chars vs 562), which left G2
    comparing only the control arm's shortest outputs.
    """
    bodies: list[dict] = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data.decode("utf-8")))
        content = "" if len(bodies) == 1 else "SUPPORTED"
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("arcsum.judge.client.request.urlopen", fake_urlopen)
    client = JudgeClient()

    assert client("local:8090/gpt-oss-20b", "sys", "user") == "SUPPORTED"
    assert len(bodies) == 2
    assert "chat_template_kwargs" not in bodies[0]
    assert bodies[1]["chat_template_kwargs"] == {"reasoning_effort": "low"}


# --- hosted third-family judge (SPEC 5.1) -------------------------------------------

def test_hosted_scheme_resolves_and_local_is_unaffected():
    from arcsum.judge.client import resolve_hosted_model, resolve_local_url
    assert resolve_hosted_model("opencode:glm-5.3") == "glm-5.3"
    assert resolve_hosted_model("local:8081/gpt-oss") is None
    assert resolve_local_url("opencode:glm-5.3") is None


def test_hosted_judge_still_refuses_a_contaminated_family():
    """The provider offers `qwen3.8-flash`. The contamination rule is about the MODEL,
    not the transport, so routing through a hosted endpoint must not launder it."""
    from arcsum.judge.client import ContaminatedJudgeError, JudgeClient
    with pytest.raises(ContaminatedJudgeError, match="qwen"):
        JudgeClient()("opencode:qwen3.8-flash", "s", "u")


def test_hosted_judge_requires_a_credential_from_the_environment(monkeypatch):
    from arcsum.judge.client import HOSTED_KEY_ENV, JudgeClient
    monkeypatch.delenv(HOSTED_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=HOSTED_KEY_ENV):
        JudgeClient()("opencode:glm-5.3", "s", "u")


def test_a_200_carrying_an_error_body_is_raised_not_scored(monkeypatch):
    """The provider returns quota and data-policy refusals as HTTP 200 with an `error`
    key. Parsing that as an empty verdict would silently score every claim UNSUPPORTED
    and depress whichever system happened to be judged during the outage."""
    import json as _json

    from arcsum.judge import client as C

    monkeypatch.setenv(C.HOSTED_KEY_ENV, "sk-test")

    class _R:
        def read(self):
            return _json.dumps({"error": {"type": "GoUsageLimitError",
                                          "message": "Monthly usage limit reached"}}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(C.request, "urlopen", lambda *a, **k: _R())
    with pytest.raises(RuntimeError, match="refused"):
        C.JudgeClient()("opencode:glm-5.3", "s", "u")


def test_hosted_request_carries_a_user_agent(monkeypatch):
    """Measured 2026-09-04: without one the edge returns 403/1010, which reads exactly
    like a bad key. `/v1/models` succeeds without it, so the obvious check misses this."""
    import json as _json

    from arcsum.judge import client as C

    monkeypatch.setenv(C.HOSTED_KEY_ENV, "sk-test")
    seen = {}

    class _R:
        def read(self):
            return _json.dumps({"choices": [{"message": {"content": "SUPPORTED"}}]}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(req, timeout=None):
        seen.update(req.headers)
        return _R()

    monkeypatch.setattr(C.request, "urlopen", fake)
    assert C.JudgeClient()("opencode:glm-5.3", "s", "u") == "SUPPORTED"
    assert any(k.lower() == "user-agent" for k in seen), seen
