"""Pins `arcsum.cli.judge` (SPEC §5.1): batch faithfulness scoring over
(transcript, prose) pairs, writing JSONL records `metrics.stats.load_scores` and
`cli.report`'s `inversions` field can consume directly.

Network is stubbed exactly as `test_judge.py` stubs it: `urllib.request.urlopen` is
patched, never the judge client itself, so assertions are on real request handling.
"""

from __future__ import annotations

import json

import pytest

from arcsum.cli.judge import build_parser, judge_case, judge_cases, main
from arcsum.judge.client import JudgeClient

TRANSCRIPT = "S1: 市長已核准搬遷案，預算編列兩百萬。\nS2: 議會將於下週表決預算案。"
PROSE = "市長已核准搬遷案，預算編列兩百萬元整。"


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


def test_judge_case_all_supported_scores_full_faith_and_zero_inversions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, "SUPPORTED")
    case = {
        "meeting_id": "m1",
        "system": "agent",
        "transcript": TRANSCRIPT,
        "prose": PROSE,
    }
    record = judge_case(case, JudgeClient(), model="local:8080/judge")
    assert record["meeting_id"] == "m1"
    assert record["system"] == "agent"
    assert record["faith_claim"] == pytest.approx(5.0)
    assert record["inversions"] == 0
    assert record["unsupported"] == 0


def test_judge_case_contradicted_is_counted_as_an_inversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, "CONTRADICTED")
    case = {"meeting_id": "m1", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE}
    record = judge_case(case, JudgeClient(), model="local:8080/judge")
    assert record["inversions"] == 1


def test_judge_cases_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, "SUPPORTED")
    cases = [
        {"meeting_id": "a", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE},
        {"meeting_id": "b", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE},
    ]
    records, failures = judge_cases(cases, JudgeClient(), model="local:8080/judge")
    assert [r["meeting_id"] for r in records] == ["a", "b"]
    assert failures == {}


def test_judge_case_refuses_a_contaminated_model_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _stub(monkeypatch, "SUPPORTED")
    case = {"meeting_id": "m1", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE}
    from arcsum.judge.client import ContaminatedJudgeError

    with pytest.raises(ContaminatedJudgeError):
        judge_case(case, JudgeClient(), model="local:8080/qwen2.5")
    assert captured == []


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_model_and_out() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["cases.json"])


def test_main_writes_jsonl_loadable_by_load_scores(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub(monkeypatch, "SUPPORTED")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [{"meeting_id": "m1", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE}]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "judged.jsonl"

    rc = main([str(cases_path), "--model", "local:8080/judge", "--out", str(out_path)])

    assert rc == 0
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["meeting_id"] == "m1"

    from arcsum.metrics.stats import load_scores

    assert load_scores([record]) == {"m1": {"agent": record}}


def test_main_passes_votes_through_to_judge_meeting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _stub(monkeypatch, "SUPPORTED")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [{"meeting_id": "m1", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE}]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "judged.jsonl"

    main(
        [
            str(cases_path),
            "--model",
            "local:8080/judge",
            "--votes",
            "5",
            "--out",
            str(out_path),
        ]
    )

    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["claims"] * 5 == len(captured)


# --- per-case failure isolation --------------------------------------------------------


def test_one_failing_case_does_not_sink_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured 2026-08-27: a single hard claim exhausted the judge's output budget on
    reasoning, raised, and aborted all 40 cases after ~16 had already been judged.
    `cli.run_arms` already isolates per meeting per arm; this mirrors it.
    """
    bad_prose = "這一項主張完全無法從逐字稿中得到支持。"

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        # Empty content (the real failure shape: budget spent on reasoning) for the
        # bad meeting's claim only; every other call answers normally.
        if "無法從逐字稿中得到支持" in body["messages"][1]["content"]:
            return _FakeResponse({"choices": [{"message": {"content": ""}}]})
        return _FakeResponse({"choices": [{"message": {"content": "SUPPORTED"}}]})

    monkeypatch.setattr("arcsum.judge.client.request.urlopen", fake_urlopen)

    cases = [
        {"meeting_id": "ok", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE},
        {"meeting_id": "bad", "system": "agent", "transcript": TRANSCRIPT, "prose": bad_prose},
    ]
    records, failures = judge_cases(cases, JudgeClient(), model="local:8080/judge")

    assert [r["meeting_id"] for r in records] == ["ok"], "the good case must survive"
    assert "agent/bad" in failures
    assert failures["agent/bad"]


def test_a_failed_case_is_absent_rather_than_scored_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """A case the judge could not evaluate is MISSING evidence. Scoring it as zero
    inversions would bias G2 toward passing -- the same defect
    `gate_g2_faithfulness` withholds on when no records exist at all."""

    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"choices": [{"message": {"content": ""}}]})

    monkeypatch.setattr("arcsum.judge.client.request.urlopen", fake_urlopen)
    cases = [{"meeting_id": "x", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE}]

    records, failures = judge_cases(cases, JudgeClient(), model="local:8080/judge")

    assert records == []
    assert list(failures) == ["agent/x"]


def test_main_writes_failures_when_asked(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"choices": [{"message": {"content": ""}}]})

    monkeypatch.setattr("arcsum.judge.client.request.urlopen", fake_urlopen)
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [{"meeting_id": "x", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE}]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "judged.jsonl"
    fails = tmp_path / "failures.json"

    rc = main(
        [
            str(cases_path),
            "--model",
            "local:8080/judge",
            "--out",
            str(out),
            "--out-failures",
            str(fails),
        ]
    )

    assert rc == 0
    assert out.read_text(encoding="utf-8") == ""
    assert "agent/x" in json.loads(fails.read_text(encoding="utf-8"))


def test_max_tokens_flag_reaches_the_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured = _stub(monkeypatch, "SUPPORTED")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [{"meeting_id": "x", "system": "agent", "transcript": TRANSCRIPT, "prose": PROSE}]
        ),
        encoding="utf-8",
    )

    main(
        [
            str(cases_path),
            "--model",
            "local:8080/judge",
            "--out",
            str(tmp_path / "o.jsonl"),
            "--max-tokens",
            "9000",
        ]
    )

    assert captured, "no request was made"
    assert json.loads(captured[0].data)["max_tokens"] == 9000


def _case(mid: str, system: str = "agent") -> dict:
    return {"meeting_id": mid, "system": system, "transcript": TRANSCRIPT, "prose": PROSE}


def test_records_reach_disk_as_they_are_produced_not_at_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Durability at the PROCESS boundary, not just the case boundary.

    `judge_cases` already isolates a failing case so one bad claim cannot discard the
    batch. That left the identical hole one level up: the batch lived in memory until
    the run ended, so a kill at case 79 of 80 discarded 79 completed judgements. The
    five-judge panel of 2026-09-05 ran >3 hours per judge, which is long enough that
    this is the expected case, not the unlucky one. Asserted by observing the file
    MID-RUN -- checking it only after `judge_cases` returns cannot distinguish
    streaming from a single write at the end, which is the whole point.
    """
    _stub(monkeypatch, "SUPPORTED")
    out = tmp_path / "scores.jsonl"
    seen: list[int] = []

    with out.open("w", encoding="utf-8", buffering=1) as f:

        def sink(rec: dict) -> None:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.append(len(out.read_text(encoding="utf-8").splitlines()))

        judge_cases(
            [_case("m1"), _case("m2"), _case("m3")],
            JudgeClient(),
            model="local:8080/judge",
            sink=sink,
        )

    assert seen == [1, 2, 3], "each record must be on disk before the next case starts"


def test_resume_skips_completed_cases_and_appends(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """`--resume` must not re-spend budget on work already on disk.

    A hosted reasoning judge costs real money per case, so recomputing 79 of 80 cases
    after an interruption is a budget bug as much as a time one.
    """
    _stub(monkeypatch, "SUPPORTED")
    out = tmp_path / "scores.jsonl"
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([_case("m1"), _case("m2"), _case("m3")], ensure_ascii=False), encoding="utf-8"
    )

    argv = [str(cases), "--model", "local:8080/judge", "--votes", "1", "--out", str(out)]
    main([*argv[:1], *argv[1:]])
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3

    # Truncate to two records, as an interrupted run would leave it.
    lines = out.read_text(encoding="utf-8").splitlines()[:2]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    calls = _stub(monkeypatch, "SUPPORTED")
    main([*argv, "--resume"])
    ids = [json.loads(ln)["meeting_id"] for ln in out.read_text(encoding="utf-8").splitlines()]
    assert ids == ["m1", "m2", "m3"], "resume must append only the missing case"
    # The saving is the point, so assert the SIZE of the work, not just its result:
    # three cases at one call each would mean resume skipped nothing.
    assert len(calls) == 1, f"resume must judge only the 1 missing case, sent {len(calls)}"


def test_completed_keys_tolerates_a_truncated_final_line(tmp_path) -> None:
    """A process killed mid-write leaves a partial JSON line. Refusing to resume on it
    would defeat resuming exactly when it is most needed.
    """
    from arcsum.cli.judge import completed_keys

    out = tmp_path / "scores.jsonl"
    out.write_text(
        json.dumps({"meeting_id": "m1", "system": "agent"}) + "\n" + '{"meeting_id": "m2", "sys',
        encoding="utf-8",
    )
    assert completed_keys(out) == {"agent/m1"}
