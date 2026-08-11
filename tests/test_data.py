"""Corpora conversion, synthetic revision set, and SFT sample building."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

from build_sft import build_sample, translate_target  # noqa: E402

from voxsum.corpora import (
    meetingbank_record,
    parse_meetingbank_transcript,
    parse_qmsum_input,
    qmsum_record,
    synthesise_clock,
)
from voxsum.ops import Add, Del, Nop, Upd, parse_ops
from voxsum.synth import REVISION_KINDS, build_meeting, build_set
from voxsum.transcript import parse_transcript

QMSUM_INPUT = (
    "What was agreed upon on sample transcripts?\n"
    "Professor E: So. OK. Doesn't look like it crashed. {disfmarker} That's great.\n"
    "Grad G: I keep starting it and then stopping it.\n"
    "and that causes the crash.\n"
    "Postdoc B: It looks like you found a way {vocalsound} of mapping the location.\n"
)


# --- QMSum ---------------------------------------------------------------------

def test_qmsum_drops_the_query_line() -> None:
    turns = parse_qmsum_input(QMSUM_INPUT)
    assert turns[0][0] == "Professor E"
    assert not any("What was agreed" in t for _, t in turns)


def test_qmsum_strips_annotation_markers() -> None:
    turns = parse_qmsum_input(QMSUM_INPUT)
    joined = " ".join(t for _, t in turns)
    assert "{disfmarker}" not in joined and "{vocalsound}" not in joined


def test_qmsum_continuation_joins_previous_turn() -> None:
    turns = parse_qmsum_input(QMSUM_INPUT)
    assert [s for s, _ in turns] == ["Professor E", "Grad G", "Postdoc B"]
    assert "causes the crash" in dict(turns)["Grad G"]


def test_qmsum_record_is_valid_v1_with_labelled_clock() -> None:
    rec = qmsum_record("va-sq-1", QMSUM_INPUT)
    parse_transcript(rec.render())  # raises if not v1
    assert rec.authentic_speakers is True
    assert rec.authentic_clock is False, "clock is synthesised — must be labelled"
    assert "150 wpm" in " ".join(rec.notes)


# --- MeetingBank ---------------------------------------------------------------

def test_meetingbank_splits_a_flat_blob_into_lines() -> None:
    raw = "Okay. With starting with public comment. So hearing item number one. District three."
    lines = parse_meetingbank_transcript(raw)
    assert len(lines) > 1, "one anchor for a whole segment makes every bullet point at it"


def test_meetingbank_record_has_no_speakers_and_says_so() -> None:
    rec = meetingbank_record("LongBeachCC_1", "Okay. Item one. Motion carries.")
    parse_transcript(rec.render())
    assert rec.authentic_speakers is False
    assert any("no speaker labels" in n for n in rec.notes)
    assert all(u.speaker is None for u in rec.utterances)


def test_meetingbank_splits_overlong_sentences() -> None:
    lines = parse_meetingbank_transcript("word " * 400, max_chars=200)
    assert lines and all(len(line) <= 200 for line in lines)


# --- synthesised clock ---------------------------------------------------------

def test_clock_is_strictly_increasing() -> None:
    # Duplicate timestamps would make `Chunk.has_line` unable to tell two lines apart.
    turns = [("S1", "hi"), ("S2", "ok"), ("S3", "yes")]
    starts = [u.start for u in synthesise_clock(turns)]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_clock_scales_with_turn_length() -> None:
    short = synthesise_clock([("S1", "ok"), ("S2", "next")])
    long = synthesise_clock([("S1", "word " * 300), ("S2", "next")])
    assert long[1].start > short[1].start


# --- synthetic revision set ----------------------------------------------------

@pytest.mark.parametrize("lang", ["en", "zh-TW"])
@pytest.mark.parametrize("kind", REVISION_KINDS)
def test_every_meeting_plants_setup_then_revision(lang: str, kind: str) -> None:
    m = build_meeting(f"t-{lang}-{kind}", lang, kind)
    if kind == "plain":
        # ADD-only kind: decision content with NO prior bullet (the 270M lesson:
        # "no matching state bullet -> ADD", breaking the content->UPD overgeneralisation).
        assert m.line_at(m.revision_at)
    else:
        assert m.setup_at < m.revision_at, "the revision must come after what it revises"
        assert m.line_at(m.setup_at) and m.line_at(m.revision_at)
    parse_transcript(m.render())


@pytest.mark.parametrize("kind", REVISION_KINDS)
def test_trap_sits_between_setup_and_revision(kind: str) -> None:
    m = build_meeting("t", "en", kind)
    if kind == "plain":
        assert m.trap_at is not None  # trap still planted; no setup/revision pair by design
    else:
        assert m.setup_at < m.trap_at < m.revision_at, (
            "the trap must fall between them so the model holds the revision across it"
        )


def test_withdraw_expects_del_others_expect_upd() -> None:
    assert build_meeting("t", "en", "withdraw").expected_op == "DEL"
    assert build_meeting("t", "en", "plain").expected_op == "ADD"
    for kind in ("reversal", "deadline", "reassign", "combined", "twotopic"):
        assert build_meeting("t", "en", kind).expected_op == "UPD"


def test_build_set_oversamples_zh() -> None:
    meetings = build_set()
    zh = [m for m in meetings if m.lang == "zh-TW"]
    en = [m for m in meetings if m.lang == "en"]
    # RESULTS.md: revise-don't-append is weaker in zh, so zh needs more demonstrations.
    assert len(zh) > len(en)
    assert {m.kind for m in meetings} == set(REVISION_KINDS)
    assert len({m.meeting_id for m in meetings}) == len(meetings)


def test_variants_differ_in_subject() -> None:
    a = build_meeting("a", "en", "reversal", variant=0)
    b = build_meeting("b", "en", "reversal", variant=1)
    assert a.subject_terms != b.subject_terms, "a student must not key on one noun phrase"


def test_unknown_kind_and_lang_rejected() -> None:
    with pytest.raises(ValueError, match="unknown revision kind"):
        build_meeting("t", "en", "nope")
    with pytest.raises(ValueError, match="unsupported language"):
        build_meeting("t", "fr", "reversal")


# --- SFT samples ---------------------------------------------------------------

def test_translate_target_emits_functiongemma_calls() -> None:
    out = translate_target("ADD DECISIONS - Plan approved [5:10]")
    assert out.startswith("<start_function_call>call:ADD{")
    assert "section:<escape>DECISIONS<escape>" in out
    assert "anchor:<escape>5:10<escape>" in out
    # Round-trips through the parser the harness uses at inference.
    (op,) = parse_ops(out)
    assert isinstance(op, Add) and op.anchor == 310


def test_translate_upd_del_nop() -> None:
    (upd,) = parse_ops(translate_target("UPD SUMMARY «Budget in» -> Approved [1:00]"))
    assert isinstance(upd, Upd) and upd.prefix == "Budget in"
    (dele,) = parse_ops(translate_target("DEL OPEN «Parking allocation»"))
    assert isinstance(dele, Del)
    (nop,) = parse_ops(translate_target("NOP"))
    assert isinstance(nop, Nop)


def test_translate_cmp_expands_to_adds() -> None:
    ops = parse_ops(translate_target("CMP TOPICS\n- one [0:00]\n- two [1:00]"))
    assert len(ops) == 2 and all(isinstance(o, Add) for o in ops)
    assert [o.anchor for o in ops] == [0, 60]


def test_translate_multiline_target_preserves_order() -> None:
    ops = parse_ops(
        translate_target("ADD TOPICS - First [0:00]\nADD DECISIONS - Second [1:00]")
    )
    assert [o.section for o in ops] == ["TOPICS", "DECISIONS"]


def test_build_sample_splits_prompt_and_completion() -> None:
    record = {
        "meeting": "m1",
        "lang": "en",
        "step": 0,
        "target": "ADD DECISIONS - Plan approved [5:10]",
        "user": "STATE:\nTITLE:\nSUMMARY:\n-\n\nCHUNK:\n[5:10] S1: approved\n",
        "is_nop": False,
        "prompt_tokens": 120,
    }
    sample = build_sample(record)
    # Completion-only masking depends on these staying separate.
    assert "CHUNK:" in sample["prompt"] and "CHUNK:" not in sample["completion"]
    assert sample["completion"].startswith("<start_function_call>")
    assert "<start_function_declaration>" in sample["system"]
    assert sample["has_revision"] is False


def test_build_sample_flags_revisions() -> None:
    record = {
        "meeting": "m1",
        "lang": "zh-TW",
        "target": "UPD DECISIONS «倉庫整併方案» -> 倉庫整併方案通過 [5:10]",
        "user": "STATE:\n...\nCHUNK:\n[5:10] S1: 通過\n",
    }
    assert build_sample(record)["has_revision"] is True


def test_build_sample_carries_prompt_version() -> None:
    from voxsum.prompts import PROMPT_VERSION

    sample = build_sample({"meeting": "m", "lang": "en", "target": "NOP", "user": "x"})
    assert sample["prompt_version"] == PROMPT_VERSION


# --- revision meetings must span chunks -----------------------------------------

@pytest.mark.parametrize("budget", [512, 2048])
def test_setup_and_revision_land_in_different_chunks(budget: int) -> None:
    """The whole point of these meetings, and the easiest thing to get silently wrong.

    Unpadded they are ~137 tokens, so at any production chunk budget the model sees the setup
    and its later contradiction together and can just ADD the final state — UPD is never the
    correct answer and the meeting teaches nothing. Three separate bugs in this project have
    come from short meetings meeting large chunks.
    """
    from voxsum.chunker import iter_chunks
    from voxsum.synth import build_set

    for meeting in build_set(chunk_budget=budget):
        if meeting.kind == "plain":
            continue  # ADD-only kind: no setup by design
        chunks = list(iter_chunks(list(meeting.utterances), budget=budget))
        setup = [i for i, c in enumerate(chunks) if c.has_line(meeting.setup_at)]
        revision = [i for i, c in enumerate(chunks) if c.has_line(meeting.revision_at)]
        assert setup and revision, f"{meeting.meeting_id}: planted lines missing"
        assert max(setup) < min(revision), (
            f"{meeting.meeting_id}: setup and revision share a chunk at budget {budget}"
        )


def test_unpadded_meetings_are_still_compact() -> None:
    """Default (no chunk_budget) stays small — the screen and unit tests rely on it."""
    from voxsum.synth import build_meeting

    assert len(build_meeting("t", "en", "reversal").utterances) < 20


def test_padding_keeps_the_trap_between_setup_and_revision() -> None:
    from voxsum.synth import build_meeting

    m = build_meeting("t", "en", "reversal", padding=40)
    assert m.setup_at < m.trap_at < m.revision_at


def test_meeting_ids_are_filename_safe() -> None:
    """MeetingBank uids contain spaces; an id becomes a filename.

    A space survives every Python path API and then breaks the first unquoted shell
    expansion downstream — two meetings were silently skipped mid-run before this.
    """
    import re

    for uid in ("SeattleCityCouncil_03142016_CB 118618", "a/b:c*d"):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", uid)
        assert " " not in safe and "/" not in safe and ":" not in safe
        assert safe, "sanitising must not empty the id"
