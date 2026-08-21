"""Pins `arcsum.cli.probe` (SPEC §5.2 G1): dumping probe transcripts and scoring
already-produced prose, without needing a live model in this repo's test environment.
"""

from __future__ import annotations

import json

import pytest

from arcsum.cli.probe import build_parser, dump_transcripts, main, score_results
from arcsum.probe import probe_meetings

MEETINGS = probe_meetings()

#: A hand-written prose that correctly states the LATER decision for every probe
#: meeting -- built from each meeting's own fields, so it stays in sync if the probe
#: meetings ever change.
_PASSING_RESULTS = {m.name: f"{' '.join(m.subject_terms)} {m.late_decision}。" for m in MEETINGS}


def test_dump_transcripts_writes_one_file_per_probe_meeting(tmp_path) -> None:
    out = tmp_path / "transcripts"
    names = dump_transcripts(out)

    assert set(names) == {m.name for m in MEETINGS}
    for name in names:
        text = (out / f"{name}.txt").read_text(encoding="utf-8")
        assert text.count("\n") == len(next(m for m in MEETINGS if m.name == name).utterances) - 1


def test_dump_transcripts_content_matches_utterance_render(tmp_path) -> None:
    out = tmp_path / "transcripts"
    dump_transcripts(out)
    meeting = MEETINGS[0]
    text = (out / f"{meeting.name}.txt").read_text(encoding="utf-8")
    assert text == "\n".join(u.render() for u in meeting.utterances)


def test_score_results_all_pass_gives_g1_passed_true() -> None:
    scored, g1_passed = score_results(_PASSING_RESULTS)
    assert g1_passed is True
    assert all(r.passed for r in scored)
    assert len(scored) == len(MEETINGS)


def test_score_results_one_failure_gives_g1_passed_false() -> None:
    broken = dict(_PASSING_RESULTS)
    first = MEETINGS[0]
    broken[first.name] = f"{' '.join(first.subject_terms)} {first.early_decision}。"  # stale

    scored, g1_passed = score_results(broken)

    assert g1_passed is False
    failing = next(r for r in scored if r.name == first.name)
    assert failing.passed is False


def test_score_results_raises_on_missing_meeting() -> None:
    partial = dict(_PASSING_RESULTS)
    del partial[MEETINGS[0].name]

    with pytest.raises(KeyError):
        score_results(partial)


# --- CLI plumbing --------------------------------------------------------------------------


def test_build_parser_requires_a_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_dump_requires_out() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["dump"])


def test_main_dump_writes_transcripts(tmp_path) -> None:
    out = tmp_path / "transcripts"
    rc = main(["dump", "--out", str(out)])
    assert rc == 0
    assert len(list(out.glob("*.txt"))) == len(MEETINGS)


def test_main_score_returns_zero_when_g1_passes(tmp_path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_PASSING_RESULTS), encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = main(["score", str(results_path), "--out", str(out_path)])

    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["g1_passed"] is True
    assert len(report["results"]) == len(MEETINGS)


def test_main_score_returns_nonzero_when_g1_fails(tmp_path) -> None:
    broken = dict(_PASSING_RESULTS)
    first = MEETINGS[0]
    broken[first.name] = f"{' '.join(first.subject_terms)} {first.early_decision}。"
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(broken), encoding="utf-8")

    rc = main(["score", str(results_path)])

    assert rc == 1


def test_main_score_returns_nonzero_on_missing_meeting(tmp_path) -> None:
    partial = dict(_PASSING_RESULTS)
    del partial[MEETINGS[0].name]
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(partial), encoding="utf-8")

    rc = main(["score", str(results_path)])

    assert rc == 1
