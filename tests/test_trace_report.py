"""Trace-report aggregation — §7.4 operational metrics from gen_traces JSONL.

The two rules that have burned this repo before are pinned here (CLAUDE.md §7.4, §6):

* valid-op rate excludes NOP from **both** the numerator and the denominator — counting
  NOP as a success has been a real bug twice;
* anchor rate (raw) scores **only** applied ADD/UPD ops — never NOP, never TITLE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import trace_report  # noqa: E402


def _rec(
    raw: str, target: str, *, meeting: str = "qmsum-x", lang: str = "en", vetoed=None
) -> dict:
    return {
        "meeting": meeting,
        "lang": lang,
        "step": 0,
        "raw": raw,
        "target": target,
        "vetoed": vetoed or [],
    }


# --- valid-op rate: NOP excluded from BOTH sides -------------------------------

def test_valid_op_rate_excludes_nop_from_numerator_and_denominator() -> None:
    # One non-NOP op per step. If NOP entered either side the rate would be 200% or 50%.
    records = [
        _rec(raw="ADD SUMMARY - a [0:00]\nNOP", target="ADD SUMMARY - a [0:00]\nNOP"),
        _rec(raw="NOP\nADD TOPICS - t [1:00]", target="NOP\nADD TOPICS - t [1:00]"),
    ]
    assert trace_report.valid_op_rate(records) == pytest.approx(1.0)
    assert trace_report.valid_op_counts(records) == (2, 2)


def test_valid_op_rate_all_nop_steps_are_not_scored() -> None:
    # The old bug: counting NOP as a success made pure-NOP meetings score 100%.
    records = [_rec(raw="NOP", target="NOP"), _rec(raw="NOP", target="NOP")]
    assert trace_report.valid_op_rate(records) is None


def test_valid_op_rate_rejected_ops_still_cost_the_denominator() -> None:
    # BOGUS is not NOP, so it is scored; it is absent from target, so it is not valid.
    records = [_rec(raw="BOGUS\nNOP", target="NOP")]
    assert trace_report.valid_op_rate(records) == 0.0


def test_valid_op_rate_does_not_penalise_judge_vetoes() -> None:
    # A vetoed op never reached the harness, so it is not scored — matches
    # `Trace.valid_op_rate` in agent.py, which scores only ops that reached apply_ops.
    records = [
        _rec(
            raw="ADD TOPICS - t1 [0:00]\nADD TOPICS - t2 [1:00]",
            target="ADD TOPICS - t1 [0:00]",
            vetoed=[{"op": "ADD TOPICS - t2 [1:00]", "reason": "judge: UNSUPPORTED"}],
        )
    ]
    assert trace_report.valid_op_rate(records) == pytest.approx(1.0)


# --- anchor rate (raw): ADD/UPD only, never NOP/TITLE --------------------------

def test_anchor_rate_raw_scores_only_add_and_upd() -> None:
    target = (
        "TITLE: Meeting\n"
        "ADD SUMMARY - anchored [0:00]\n"
        "NOP\n"
        "UPD DECISIONS «old» -> revised [1:00]\n"
        "ADD TOPICS - no timestamp"
    )
    records = [_rec(raw=target, target=target)]
    # Scored 3 (2 ADD + 1 UPD); natively anchored 2. TITLE/NOP must not enter the
    # denominator — if they did the rate would drop to 2/5.
    assert trace_report.anchor_rate_raw(records) == pytest.approx(2 / 3)
    assert trace_report.anchor_counts(records) == (2, 3)


def test_anchor_rate_raw_ignores_cmp_bullets() -> None:
    # CMP bullets carry anchors but are not ADD/UPD and must never be scored.
    records = [
        _rec(
            raw="CMP TOPICS\n- t1 [0:00]\n- t2 [1:00]",
            target="CMP TOPICS\n- t1 [0:00]\n- t2 [1:00]",
        )
    ]
    assert trace_report.anchor_rate_raw(records) is None


# --- shares and veto -----------------------------------------------------------

def test_nop_share_counts_all_nop_steps() -> None:
    records = [
        _rec(raw="NOP", target="NOP"),
        _rec(raw="NOP", target="NOP"),
        _rec(raw="ADD SUMMARY - a [0:00]", target="ADD SUMMARY - a [0:00]"),
    ]
    assert trace_report.nop_share(records) == pytest.approx(2 / 3)


def test_revision_share_counts_upd_and_del_targets() -> None:
    records = [
        _rec(raw="UPD DECISIONS «x» -> y [1:00]", target="UPD DECISIONS «x» -> y [1:00]"),
        _rec(raw="DEL TOPICS «old»", target="DEL TOPICS «old»"),
        _rec(raw="ADD SUMMARY - a [0:00]", target="ADD SUMMARY - a [0:00]"),
    ]
    assert trace_report.revision_share(records) == pytest.approx(2 / 3)


def test_veto_rate_is_vetoes_over_claims() -> None:
    records = [
        _rec(
            raw="ADD TOPICS - t1 [0:00]\nADD TOPICS - t2 [1:00]\nNOP",
            target="ADD TOPICS - t1 [0:00]\nNOP",
            vetoed=[{"op": "ADD TOPICS - t2 [1:00]", "reason": "judge: UNSUPPORTED"}],
        ),
        _rec(raw="ADD SUMMARY - a [0:00]", target="ADD SUMMARY - a [0:00]"),
    ]
    assert trace_report.veto_rate(records) == pytest.approx(1 / 3)


def test_veto_rate_is_none_without_claims() -> None:
    records = [_rec(raw="NOP", target="NOP")]
    assert trace_report.veto_rate(records) is None


# --- CLI / report --------------------------------------------------------------

def test_report_counts_steps_meetings_lang_and_source(tmp_path) -> None:
    rows = [
        {
            "meeting": "qmsum-a",
            "lang": "en",
            "raw": "ADD SUMMARY - a [0:00]",
            "target": "ADD SUMMARY - a [0:00]",
        },
        {
            "meeting": "mbank-b",
            "lang": "en",
            "raw": "ADD TOPICS - t [0:00]",
            "target": "ADD TOPICS - t [0:00]",
        },
        {"meeting": "synth-zh-c", "lang": "zh-TW", "raw": "NOP", "target": "NOP"},
    ]
    (tmp_path / "t.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    out = trace_report.report(rows, files=[tmp_path / "t.jsonl"])
    assert "1 file(s), 3 meetings, 3 steps" in out
    assert "en" in out and "zh-TW" in out
    assert "qmsum" in out and "mbank" in out and "synth" in out


def test_main_reads_a_directory(tmp_path, capsys) -> None:
    (tmp_path / "a.jsonl").write_text(
        json.dumps(
            {"meeting": "synth-en-d0", "lang": "en", "raw": "NOP", "target": "NOP"}
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "b.jsonl").write_text(
        json.dumps(
            {
                "meeting": "qmsum-1",
                "lang": "en",
                "raw": "ADD SUMMARY - x [0:00]",
                "target": "ADD SUMMARY - x [0:00]",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert trace_report.main([str(tmp_path)]) == 0
    assert "2 file(s)" in capsys.readouterr().out
