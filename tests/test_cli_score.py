"""Pins `arcsum.cli.score` (SPEC §5): a CLI that computes reference metrics for a batch
of meetings and writes JSONL score records `metrics.stats.load_scores` can glob
directly, without transformation.
"""

from __future__ import annotations

import json

import pytest

from arcsum.cli.score import build_parser, main, score_pair, score_pairs
from arcsum.metrics.reference import coverage, density, rouge_l, rouge_n
from arcsum.metrics.stats import DEFAULT_METRICS, filter_scoreable_records

SOURCE = "市長已核准搬遷案。議會將於下週表決預算案。市長也提到人事任命的議題。"
CANDIDATE = "市長已核准搬遷案。"
REFERENCE = "市長核准了搬遷案。"


def test_score_pair_matches_hand_called_metrics() -> None:
    """The record is not a re-implementation -- it must equal calling the library
    functions directly, so a drift between the CLI and the library is caught here."""
    record = score_pair(SOURCE, CANDIDATE, REFERENCE, meeting_id="m1", system="agent")
    assert record["rouge1"] == pytest.approx(rouge_n(CANDIDATE, REFERENCE, n=1).f1)
    assert record["rouge2"] == pytest.approx(rouge_n(CANDIDATE, REFERENCE, n=2).f1)
    assert record["rougeL"] == pytest.approx(rouge_l(CANDIDATE, REFERENCE).f1)
    assert record["coverage"] == pytest.approx(coverage(SOURCE, CANDIDATE))
    assert record["density"] == pytest.approx(density(SOURCE, CANDIDATE))
    assert record["length_chars"] == len(CANDIDATE)


def test_score_pair_coverage_uses_source_not_reference() -> None:
    """The candidate is verbatim-copied from SOURCE but shares little with REFERENCE --
    coverage must reflect the source comparison, not silently fall back to reference."""
    record = score_pair(SOURCE, CANDIDATE, REFERENCE, meeting_id="m1", system="agent")
    assert record["coverage"] == 1.0  # every candidate token is a fragment of SOURCE
    assert record["coverage"] != pytest.approx(coverage(REFERENCE, CANDIDATE))


def test_score_pair_carries_identity_fields_for_load_scores() -> None:
    record = score_pair(SOURCE, CANDIDATE, REFERENCE, meeting_id="m42", system="baseline")
    assert record["meeting_id"] == "m42"
    assert record["system"] == "baseline"
    assert filter_scoreable_records([record]) == [record]


def test_score_pair_record_keys_cover_default_metrics() -> None:
    record = score_pair(SOURCE, CANDIDATE, REFERENCE, meeting_id="m1", system="agent")
    for metric in DEFAULT_METRICS:
        assert metric in record


def test_score_pairs_preserves_input_order_and_stamps_system() -> None:
    pairs = [
        {"meeting_id": "a", "source": SOURCE, "candidate": CANDIDATE, "reference": REFERENCE},
        {
            "meeting_id": "b",
            "source": SOURCE,
            "candidate": "議會將於下週表決預算案。",
            "reference": REFERENCE,
        },
    ]
    records = score_pairs(pairs, system="agent")
    assert [r["meeting_id"] for r in records] == ["a", "b"]
    assert all(r["system"] == "agent" for r in records)


def test_score_pairs_injects_custom_token_len() -> None:
    calls: list[str] = []

    def counting_token_len(text: str) -> int:
        calls.append(text)
        return 7

    pairs = [{"meeting_id": "a", "source": SOURCE, "candidate": CANDIDATE, "reference": REFERENCE}]
    records = score_pairs(pairs, system="agent", token_len=counting_token_len)
    assert records[0]["length_tokens"] == 7
    assert CANDIDATE in calls


# --- main() / CLI plumbing ----------------------------------------------------------------


def test_build_parser_requires_system_and_out() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pairs.json"])


def test_main_writes_jsonl_loadable_by_load_scores(tmp_path) -> None:
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            [{"meeting_id": "a", "source": SOURCE, "candidate": CANDIDATE, "reference": REFERENCE}]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "scored.jsonl"

    rc = main([str(pairs_path), "--system", "agent", "--out", str(out_path)])

    assert rc == 0
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["meeting_id"] == "a"
    assert record["system"] == "agent"
    from arcsum.metrics.stats import load_scores

    assert load_scores([record]) == {"a": {"agent": record}}


def test_main_writes_one_line_per_meeting_in_order(tmp_path) -> None:
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            [
                {
                    "meeting_id": "a",
                    "source": SOURCE,
                    "candidate": CANDIDATE,
                    "reference": REFERENCE,
                },
                {
                    "meeting_id": "b",
                    "source": SOURCE,
                    "candidate": "議會將於下週表決預算案。",
                    "reference": REFERENCE,
                },
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "scored.jsonl"

    main([str(pairs_path), "--system", "agent", "--out", str(out_path)])

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [r["meeting_id"] for r in records] == ["a", "b"]


def test_main_default_tokenizer_is_heuristic_no_extras_needed(tmp_path) -> None:
    """Must not require the [tokenizer] extra when --tokenizer is not passed -- the
    whole suite runs with no optional extra installed (CLAUDE.md)."""
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            [{"meeting_id": "a", "source": SOURCE, "candidate": CANDIDATE, "reference": REFERENCE}]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "scored.jsonl"

    rc = main([str(pairs_path), "--system", "agent", "--out", str(out_path)])

    assert rc == 0
