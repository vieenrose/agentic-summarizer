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


#: Marker the fake `/apply-template` wraps the system prompt in, so the fake
#: `/completion` can recover which call it is being asked to serve. The real server
#: renders with the model's own chat template; the shape does not matter here, only
#: that the round trip preserves enough to route on.
_RENDER_PREFIX = "<<SYS>>"


def _system_of(body: dict) -> str:
    """The system prompt for either wire shape — chat (`messages`) or raw (`prompt`),
    since `run_arms` now defaults to the raw `/apply-template` + `/completion` route."""
    if "messages" in body:
        return body["messages"][0]["content"]
    prompt = body["prompt"]
    return prompt.split(_RENDER_PREFIX, 1)[1].split("<<END>>", 1)[0]


def _stub_both_arms(monkeypatch: pytest.MonkeyPatch) -> list:
    captured: list = []

    def fake_urlopen(req, timeout=None):
        captured.append(req)
        body = json.loads(req.data.decode("utf-8"))
        if req.full_url.endswith("/apply-template"):
            sysmsg = body["messages"][0]["content"]
            return _FakeResponse({"prompt": f"{_RENDER_PREFIX}{sysmsg}<<END>>"})
        system = _system_of(body)
        if system == step_system_prompt():
            # ADD (not NOP) so memory is non-empty: an all-NOP run leaves memory
            # empty, which synthesize_memory short-circuits without a model call.
            content = "ADD - 同意搬到 B 棟"
        elif system == synth_system_prompt():
            content = _SYNTH_PROSE
        elif system == map_system_prompt():
            content = _MAP_PROSE
        elif system == reduce_system_prompt():
            content = _REDUCE_PROSE
        else:
            raise AssertionError(f"unexpected system prompt: {system!r}")
        if req.full_url.endswith("/completion"):
            return _FakeResponse({"content": content})
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", fake_urlopen)
    return captured


def test_run_both_arms_produces_paired_meetings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_both_arms(monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")

    agent_pairs, baseline_pairs, skipped, failures = run_both_arms(
        corpus,
        {"m1": "reference text"},
        step_model=LlamaServer(),
        synth_model=LlamaServer(),
        reduce_model=LlamaServer(),
    )

    assert skipped == []
    assert failures == {"agent": {}, "baseline": {}}
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

    agent_pairs, baseline_pairs, _, _ = run_both_arms(
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

    agent_pairs, baseline_pairs, skipped, _ = run_both_arms(
        corpus,
        {"m1": "reference text"},
        step_model=LlamaServer(),
        synth_model=LlamaServer(),
        reduce_model=LlamaServer(),
    )

    assert skipped == ["m2"]
    assert len(agent_pairs) == 1
    assert len(baseline_pairs) == 1


# --- failure isolation ----------------------------------------------------------------------


def test_one_meetings_baseline_failure_does_not_lose_other_meetings(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reduce call overflowing on one meeting (measured to happen on 7/20 real
    held-out meetings at a real 4096-token context) must not take the whole pass down
    with it -- confirmed here via a `reduce_context_tokens` too small for ANY reduce
    prompt to fit, so every multi-window meeting's baseline arm hits it."""
    _stub_both_arms(monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # m1: single line -> single window -> no reduce call, unaffected by the guard.
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")
    # m2: enough lines to force >1 window at the default chunk budget.
    many_lines = "\n".join(f"S1: 第{i}項議程討論。" for i in range(400))
    (corpus / "m2.txt").write_text(many_lines, encoding="utf-8")

    agent_pairs, baseline_pairs, skipped, failures = run_both_arms(
        corpus,
        {"m1": "ref1", "m2": "ref2"},
        step_model=LlamaServer(),
        synth_model=LlamaServer(),
        reduce_model=LlamaServer(),
        reduce_context_tokens=1,
    )

    assert skipped == []
    # m2's baseline arm does NOT raise (the overflow guard in baseline.py converts it
    # to a deterministic fallback, not an exception) -- so it is not a "failure" here,
    # it is a successful (if degraded) baseline result. Both meetings pair normally.
    assert failures == {"agent": {}, "baseline": {}}
    assert {p["meeting_id"] for p in agent_pairs} == {"m1", "m2"}
    assert {p["meeting_id"] for p in baseline_pairs} == {"m1", "m2"}


def test_a_meetings_arm_exception_excludes_it_from_both_pairs_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine per-meeting exception (network error, malformed response, anything
    `run_agent`/`run_map_reduce` themselves raise) on EITHER arm must exclude that
    meeting from BOTH pairs files -- an unpaired candidate cannot enter SPEC §5.2's
    paired comparison -- while leaving every other meeting intact. Triggered on
    content unique to one meeting's transcript, not call order/count, so this cannot
    become order-dependent and flaky."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")
    (corpus / "m2.txt").write_text("S1: 這場會議只在第二場才會出現的內容。\n", encoding="utf-8")

    def flaky_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        system = body["messages"][0]["content"]
        user = body["messages"][1]["content"]
        if system == map_system_prompt() and "第二場才會出現" in user:
            raise OSError("simulated network failure")
        if system == step_system_prompt():
            # ADD (not NOP) so memory is non-empty: an all-NOP run leaves memory
            # empty, which synthesize_memory short-circuits without a model call.
            content = "ADD - 同意搬到 B 棟"
        elif system == synth_system_prompt():
            content = _SYNTH_PROSE
        elif system == map_system_prompt():
            content = _MAP_PROSE
        elif system == reduce_system_prompt():
            content = _REDUCE_PROSE
        else:
            raise AssertionError(f"unexpected system prompt: {system!r}")
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", flaky_urlopen)

    agent_pairs, baseline_pairs, skipped, failures = run_both_arms(
        corpus,
        {"m1": "ref1", "m2": "ref2"},
        step_model=LlamaServer(),
        synth_model=LlamaServer(),
        reduce_model=LlamaServer(),
    )

    assert skipped == []
    assert "m2" in failures["baseline"]
    assert "m2" not in failures["agent"]  # m2's agent arm never calls MAP at all
    assert {p["meeting_id"] for p in agent_pairs} == {"m1"}
    assert {p["meeting_id"] for p in baseline_pairs} == {"m1"}


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_references_and_out_paths() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["corpus/"])


def test_main_passes_extra_json_through_to_every_llama_server_body(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--extra` exists specifically because MiniCPM5 needs `enable_thinking` disabled
    or it can burn its whole `max_tokens` budget on reasoning before answering (found
    live during the Phase 2 pilot eval) -- must land in every request body, on all
    three constructed servers (step/synth/reduce), not just one."""
    captured_bodies: list[dict] = []

    def wrapped_urlopen(req, timeout=None):
        captured_bodies.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "NOP"
                                if json.loads(req.data.decode("utf-8"))["messages"][0]["content"]
                                == step_system_prompt()
                                else _SYNTH_PROSE
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", wrapped_urlopen)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")
    refs_path = tmp_path / "refs.json"
    refs_path.write_text(json.dumps({"m1": "reference text"}), encoding="utf-8")

    rc = main(
        [
            str(corpus),
            "--references",
            str(refs_path),
            "--out-agent",
            str(tmp_path / "agent_pairs.json"),
            "--out-baseline",
            str(tmp_path / "baseline_pairs.json"),
            "--extra",
            '{"chat_template_kwargs": {"enable_thinking": false}}',
        ]
    )

    assert rc == 0
    assert captured_bodies  # sanity: requests were actually made
    for body in captured_bodies:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_repeat_penalty_goes_to_prose_calls_but_never_reading_steps(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two invariants at once. A reading step must NOT carry a repetition penalty --
    its output is a fixed op vocabulary, so penalising repetition penalises the literal
    ADD/DROP/ARC tokens the format requires. And BOTH arms' prose calls must carry the
    SAME one: giving it to only the agent would be precisely the unfair-baseline
    comparison SPEC §5.2 forbids.
    """
    step_bodies: list[dict] = []
    prose_bodies: list[dict] = []

    def wrapped_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        if req.full_url.endswith("/apply-template"):
            # The render step carries no sampling knobs; only the generate step does.
            sysmsg = body["messages"][0]["content"]
            return _FakeResponse({"prompt": f"{_RENDER_PREFIX}{sysmsg}<<END>>"})
        is_step = _system_of(body) == step_system_prompt()
        (step_bodies if is_step else prose_bodies).append(body)
        content = "NOP" if is_step else _SYNTH_PROSE
        if req.full_url.endswith("/completion"):
            return _FakeResponse({"content": content})
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("arcsum.backends.llama_server.request.urlopen", wrapped_urlopen)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "m1.txt").write_text(TRANSCRIPT, encoding="utf-8")
    refs_path = tmp_path / "refs.json"
    refs_path.write_text(json.dumps({"m1": "reference text"}), encoding="utf-8")

    rc = main(
        [
            str(corpus),
            "--references",
            str(refs_path),
            "--out-agent",
            str(tmp_path / "agent_pairs.json"),
            "--out-baseline",
            str(tmp_path / "baseline_pairs.json"),
            "--repeat-penalty",
            "1.1",
        ]
    )

    assert rc == 0
    assert step_bodies and prose_bodies  # sanity: both kinds of call happened
    for body in step_bodies:
        assert "repeat_penalty" not in body
    for body in prose_bodies:
        assert body["repeat_penalty"] == 1.1


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
