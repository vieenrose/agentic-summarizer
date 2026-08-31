"""Pins the fair map-reduce baseline (SPEC §5.2) — the opponent the agent must beat.

The defining property under test: the map step sees NO STATE. `build_map_prompt` (already
pinned in `test_prompts.py`) contains no ARC/POINTS/MEMORY vocabulary at all, so a
Scripted fake that records its prompts is enough to prove this without any weights.
"""

from __future__ import annotations

from arcsum.baseline import run_map_reduce, summarise_window
from arcsum.chunker import Chunk
from arcsum.prompts import reduce_system_prompt
from arcsum.tokens import heuristic_token_len
from arcsum.transcript import Utterance
from conftest import Scripted


def meeting(n: int, words_per_line: int = 20) -> list[Utterance]:
    return [Utterance(f"S{i % 4 + 1}", "很好 " * words_per_line) for i in range(n)]


# --- summarise_window ------------------------------------------------------------------


def test_summarise_window_sees_no_memory_concept() -> None:
    model = Scripted(("這段內容討論搬遷案。",))
    chunk = Chunk(0, (Utterance("S1", "討論內容"),), tokens=8)
    summarise_window(chunk, model)
    _sys, user = model.calls[0]
    assert "MEMORY" not in user
    assert "ARC" not in user
    assert "POINTS" not in user


def test_summarise_window_cleans_the_output() -> None:
    model = Scripted(("- 這段內容討論搬遷案。",))
    chunk = Chunk(0, (Utterance("S1", "討論內容"),), tokens=8)
    summary = summarise_window(chunk, model)
    assert summary == "這段內容討論搬遷案。"


def test_summarise_window_falls_back_to_window_text_when_the_model_call_fails() -> None:
    """A failed map call must not delete the window's content, and must not take the
    whole meeting down with it — the same principle the reduce step already follows.

    Measured: llama.cpp 500s when the model emits an invalid UTF-8 byte, deterministically
    at temperature 0, and it strikes long map prose far more often than the agent's short
    op lines. Two of twenty meetings died that way, and since SPEC §5.2's comparison is
    PAIRED each loss cost the agent arm a meeting too, withholding G3 for `n < min_n`.
    """

    def exploding_model(system: str, user: str) -> str:
        raise RuntimeError("llama-server 500: Content-only format")

    chunk = Chunk(0, (Utterance("S1", "討論搬遷案的細節"),), tokens=8)
    summary = summarise_window(chunk, exploding_model)
    assert "討論搬遷案的細節" in summary


def test_map_fallback_favours_the_baseline_not_the_agent() -> None:
    """Direction matters: a workaround for a defect on the CONTROL arm must not be one
    that flatters the treatment. Raw transcript text is more extractive than a real
    summary, so this fallback helps the baseline on ROUGE/coverage/density. (Streaming
    was rejected for the opposite reason — it truncates at the bad byte, silently
    shortening only the baseline's output.)"""

    def exploding_model(system: str, user: str) -> str:
        raise RuntimeError("boom")

    utts = tuple(Utterance("S1", f"議程第{i}項的詳細討論內容") for i in range(4))
    chunk = Chunk(0, utts, tokens=40)
    summary = summarise_window(chunk, exploding_model)
    # Every utterance's content survives; nothing is dropped to make the fallback tidy.
    for u in utts:
        assert u.text in summary


def test_summarise_window_records_usage() -> None:
    from arcsum.agent import Usage

    model = Scripted(("摘要文字",))
    chunk = Chunk(0, (Utterance("S1", "討論內容"),), tokens=8)
    usage = Usage()
    summarise_window(chunk, model, usage=usage)
    assert usage.calls == 1
    assert usage.prefill_tokens > 0


# --- run_map_reduce: no state across windows -----------------------------------------


def test_map_step_sees_no_state_across_windows() -> None:
    """Each map call is independent -- the defining property that makes this a
    structurally fair, structurally DIFFERENT opponent."""
    model = Scripted(tuple(f"第{i}段摘要內容" for i in range(30)))
    run_map_reduce(meeting(90, words_per_line=200), model, budget=500)
    map_calls = model.calls[:-1] if len(model.calls) > 1 else model.calls
    for _sys, user in map_calls:
        assert "MEMORY" not in user


def test_windows_are_processed_in_order() -> None:
    model = Scripted(("第一段摘要", "第二段摘要", "第三段摘要"))
    result = run_map_reduce(meeting(90, words_per_line=200), model, budget=500)
    assert result.windows == len(result.window_summaries)
    assert list(result.window_summaries[:3]) == ["第一段摘要", "第二段摘要", "第三段摘要"]


# --- run_map_reduce: exactly the right number of reduce calls -----------------------


def test_single_window_skips_the_reduce_call() -> None:
    model = Scripted(("唯一一段的摘要內容。",))
    result = run_map_reduce(meeting(5), model)  # short meeting, default budget -> 1 chunk
    assert result.windows == 1
    assert result.reduce_calls == 0
    assert result.prose.text == "唯一一段的摘要內容。"


def test_multiple_windows_get_one_reduce_call() -> None:
    map_model = Scripted(tuple(f"第{i}段摘要" for i in range(30)))
    reduce_model = Scripted(("整合後的完整會議摘要內容。",))
    result = run_map_reduce(
        meeting(90, words_per_line=200), map_model, reduce_model=reduce_model, budget=500
    )
    assert result.windows > 1
    assert result.reduce_calls == 1
    assert len(reduce_model.calls) == 1
    assert result.prose.text == "整合後的完整會議摘要內容。"


def test_reduce_uses_a_separate_model_when_given() -> None:
    map_model = Scripted(tuple(f"第{i}段摘要" for i in range(30)))
    reduce_model = Scripted(("整合摘要",))
    run_map_reduce(
        meeting(90, words_per_line=200), map_model, reduce_model=reduce_model, budget=500
    )
    assert len(reduce_model.calls) == 1
    # The reduce model's own calls must never see raw transcript CHUNK: content.
    _sys, user = reduce_model.calls[0]
    assert "CHUNK:" not in user
    assert "SUMMARIES:" in user


def test_reduce_uses_the_same_model_by_default() -> None:
    """A single valid default (rather than a finite canned list) decouples this from
    the exact chunk count -- ANY call, map or reduce, gets the same valid zh-TW text,
    so the reduce call succeeds on its first attempt regardless of how many map calls
    preceded it."""
    model = Scripted(default="很好的內容摘要")
    result = run_map_reduce(meeting(90, words_per_line=200), model, budget=500)
    assert result.windows > 1
    assert result.reduce_calls == 1
    assert len(model.calls) == result.windows + 1


# --- run_map_reduce: reduce retry + deterministic fallback --------------------------


def test_reduce_retries_once_on_over_budget() -> None:
    too_long = "很" * 2000
    ok = "整合後的完整會議摘要內容。"
    map_model = Scripted(("第一段", "第二段"))
    reduce_model = Scripted((too_long, ok))
    result = run_map_reduce(
        meeting(90, words_per_line=200), map_model, reduce_model=reduce_model, budget=500
    )
    assert result.reduce_calls == 2
    assert result.prose.text == ok


def test_reduce_retries_once_on_bad_language() -> None:
    english = "The council approved everything."
    ok = "整合後的完整會議摘要內容。"
    map_model = Scripted(("第一段", "第二段"))
    reduce_model = Scripted((english, ok))
    result = run_map_reduce(
        meeting(90, words_per_line=200), map_model, reduce_model=reduce_model, budget=500
    )
    assert result.reduce_calls == 2
    assert result.prose.lang_flags == ()


def test_failed_reduce_falls_back_to_concatenated_window_summaries() -> None:
    """A failed shrink must not delete the meeting's decisions -- the deterministic
    fallback carries every window summary forward rather than losing content to a
    second failed compress."""
    map_model = Scripted(("第一段重要決議", "第二段重要決議"))
    reduce_model = Scripted(("English forever", "Still English"))
    result = run_map_reduce(
        meeting(90, words_per_line=200), map_model, reduce_model=reduce_model, budget=500
    )
    assert result.reduce_calls == 2
    assert "第一段重要決議" in result.prose.text
    assert "第二段重要決議" in result.prose.text


def test_reduce_context_tokens_none_preserves_old_unbounded_behaviour() -> None:
    """Default (`reduce_context_tokens=None`) must not change any existing caller's
    behaviour -- the guard is opt-in."""
    map_model = Scripted(tuple(f"第{i}段摘要" for i in range(30)))
    reduce_model = Scripted(("整合後的完整會議摘要內容。",))
    result = run_map_reduce(
        meeting(90, words_per_line=200), map_model, reduce_model=reduce_model, budget=500
    )
    assert result.reduce_skipped_overflow is False
    assert result.reduce_calls == 1


def test_reduce_skipped_when_its_own_prompt_would_overflow_the_real_context() -> None:
    """`build_reduce_prompt` concatenates EVERY window summary with no cap -- measured
    to overflow a real 4096-token deploy context on 7/20 real meetings (up to 43
    windows). With a small `reduce_context_tokens`, many short window summaries must
    still trip the guard: the reduce call is skipped BEFORE any request is made
    (never attempted, so `reduce_model.calls` stays empty), never attempted-then-
    failed."""
    map_model = Scripted(tuple(f"第{i}段摘要內容較長一些" for i in range(30)))
    reduce_model = Scripted(("不應該被呼叫",))
    result = run_map_reduce(
        meeting(90, words_per_line=200),
        map_model,
        reduce_model=reduce_model,
        budget=500,
        reduce_context_tokens=10,  # far below what 30 window summaries render to
    )
    assert len(reduce_model.calls) == 0
    assert result.reduce_calls == 0
    assert result.reduce_skipped_overflow is True
    # Same "never delete decisions" fallback as a doubly-failed reduce: every window
    # summary survives in the concatenated output.
    assert "第0段摘要內容較長一些" in result.prose.text
    assert "第29段摘要內容較長一些" in result.prose.text


def test_reduce_runs_normally_when_it_fits_within_reduce_context_tokens() -> None:
    """A generous `reduce_context_tokens` must not spuriously trip the guard -- the
    reduce call still happens exactly as it would with no guard at all."""
    map_model = Scripted(tuple(f"第{i}段摘要" for i in range(30)))
    reduce_model = Scripted(("整合後的完整會議摘要內容。",))
    result = run_map_reduce(
        meeting(90, words_per_line=200),
        map_model,
        reduce_model=reduce_model,
        budget=500,
        reduce_context_tokens=100_000,
    )
    assert len(reduce_model.calls) == 1
    assert result.reduce_calls == 1
    assert result.reduce_skipped_overflow is False
    assert result.prose.text == "整合後的完整會議摘要內容。"


def test_single_window_never_trips_the_overflow_guard() -> None:
    """At <=1 window there is no reduce prompt to overflow at all -- `reduce_calls=0`
    here must stay distinguishable from an overflow skip (`reduce_skipped_overflow`
    stays False) since they mean structurally different things."""
    model = Scripted(("唯一一段摘要",))
    result = run_map_reduce(
        meeting(5, words_per_line=20), model, budget=5000, reduce_context_tokens=1
    )
    assert result.windows <= 1
    assert result.reduce_calls == 0
    assert result.reduce_skipped_overflow is False


def test_reduce_calls_ranges_over_zero_one_or_two_across_scenarios() -> None:
    """reduce_calls in {0,1,2} across scenarios, mirroring agent.synthesize_memory's
    own default retry budget -- never pinned to a fixed count. Exercised together
    (rather than duplicating three near-identical run_map_reduce calls) since each
    branch is already covered individually by the tests above."""
    single_window = run_map_reduce(meeting(5), Scripted(("摘要",)))
    clean_reduce = run_map_reduce(
        meeting(90, words_per_line=200),
        Scripted(("a", "b")),
        reduce_model=Scripted(("整合後的完整會議摘要內容。",)),
        budget=500,
    )
    retried_reduce = run_map_reduce(
        meeting(90, words_per_line=200),
        Scripted(("a", "b")),
        reduce_model=Scripted(("English", "整合後的完整會議摘要內容。")),
        budget=500,
    )
    assert {single_window.reduce_calls, clean_reduce.reduce_calls, retried_reduce.reduce_calls} == {
        0,
        1,
        2,
    }


# --- fairness: same instrument as the agent -------------------------------------------


def test_usage_uses_the_injected_tokenizer() -> None:
    calls: list[str] = []

    def counting(text: str) -> int:
        calls.append(text)
        return heuristic_token_len(text)

    model = Scripted(("摘要",))
    run_map_reduce(meeting(5), model, token_len=counting)
    assert calls, "the injected counter was never called"


def test_token_len_name_is_recorded() -> None:
    model = Scripted(("摘要",))
    result = run_map_reduce(meeting(5), model)
    assert result.token_len_name == "heuristic"


def test_prompt_and_tokenize_version_are_recorded() -> None:
    model = Scripted(("摘要",))
    result = run_map_reduce(meeting(5), model)
    assert result.prompt_version == "sys-v2"
    assert result.tokenize_version == "chartok-v1"


def test_gt4_must_be_measured_at_production_chunk_size() -> None:
    """A GT4-style prefill comparison at a tiny chunk budget is dominated by the fixed
    SYS-prompt cost rather than the architecture -- this is a reminder pinned as a test,
    not a runtime guard, since the fixture itself demonstrates the shape."""
    model = Scripted(("摘要",))
    result = run_map_reduce(meeting(5), model, budget=2500)
    assert result.usage.prefill_tokens > 0


# --- prose contract shared with the agent ----------------------------------------------


def test_baseline_output_shares_the_prose_contract() -> None:
    model = Scripted(("這是一段完整的繁體中文會議摘要。",))
    result = run_map_reduce(meeting(5), model)
    assert result.prose.chars == len(result.prose.text)
    assert isinstance(result.prose.over_budget, bool)


def test_empty_transcript_yields_an_empty_prose_without_crashing() -> None:
    model = Scripted()
    result = run_map_reduce([], model)
    assert result.windows == 0
    assert result.reduce_calls == 0
    assert result.prose.text == ""


# --- hierarchical reduce (SPEC §5.2's "fair opponent") ---------------------------------


def test_hierarchical_reduce_folds_instead_of_concatenating_on_overflow() -> None:
    """The regression this exists to prevent: with a tight context the reduce call used
    to be SKIPPED and the window summaries merely concatenated, so the control arm
    silently stopped being map-REDUCE. Measured 2026-08-27: 11 of 20 held-out meetings
    emitted concatenations averaging 3,695 tokens against SPEC §3's <1,000 cap, which
    made G3 meaningless in both directions.
    """
    model = Scripted(default="這是一段摘要文字。")
    result = run_map_reduce(
        meeting(40), model, budget=60, token_len=heuristic_token_len, reduce_context_tokens=200
    )
    assert result.windows > 1
    assert result.reduce_passes > 0, "should have folded in batches"
    assert result.reduce_calls > 0, "a real reduce must have happened"
    assert result.reduce_skipped_overflow is False


def test_small_meeting_still_uses_a_single_direct_reduce() -> None:
    """Folding must not kick in when the summaries already fit — 0 passes, 1 call."""
    model = Scripted(default="摘要")
    result = run_map_reduce(
        meeting(6), model, budget=200, token_len=heuristic_token_len, reduce_context_tokens=4096
    )
    assert result.reduce_passes == 0
    assert result.reduce_calls == 1


def test_unbounded_context_preserves_the_original_single_reduce_path() -> None:
    """`reduce_context_tokens=None` is the documented opt-out; it must not fold."""
    model = Scripted(default="摘要")
    result = run_map_reduce(meeting(30), model, budget=60, token_len=heuristic_token_len)
    assert result.reduce_passes == 0
    assert result.reduce_calls == 1


def test_partition_to_fit_groups_are_each_within_context() -> None:
    from arcsum.baseline import _partition_to_fit
    from arcsum.prompts import build_reduce_prompt, reduce_system_prompt

    summaries = tuple(f"第{i}段摘要內容。" * 3 for i in range(25))
    context = 300
    groups = _partition_to_fit(summaries, token_len=heuristic_token_len, context=context)

    assert sum(len(g) for g in groups) == len(summaries), "no summary may be lost"
    assert [s for g in groups for s in g] == list(summaries), "order must be preserved"
    for g in groups:
        if len(g) == 1:
            continue  # a lone oversized summary is emitted as-is by contract
        rendered = heuristic_token_len(reduce_system_prompt()) + heuristic_token_len(
            build_reduce_prompt(g)
        )
        assert rendered <= context


class _ReduceScripted:
    """Map calls always succeed; REDUCE calls replay `reduce_responses` in order.
    Dispatching on the system prompt rather than call index keeps the fixture robust to
    how many windows the packer happens to produce."""

    def __init__(self, *reduce_responses: str, map_response: str = "摘要") -> None:
        self.reduce_responses = list(reduce_responses)
        self.map_response = map_response
        self.reduce_calls = 0

    def __call__(self, system: str, user: str) -> str:
        from arcsum.prompts import reduce_system_prompt

        if system != reduce_system_prompt():
            return self.map_response
        idx = self.reduce_calls
        self.reduce_calls += 1
        if idx < len(self.reduce_responses):
            return self.reduce_responses[idx]
        return self.reduce_responses[-1] if self.reduce_responses else self.map_response


def test_over_budget_reduce_output_is_compressed_not_shipped() -> None:
    """Folding bounds the reduce INPUT; this bounds its OUTPUT. Measured 2026-08-27:
    even after hierarchical folding fixed the concatenation bug, 5 of 20 held-out
    meetings still shipped summaries over SPEC §3's <1,000-token cap (max 2,181).
    """
    long_summary = "很好 " * 1500  # comfortably over PROSE_MAX_TOKENS
    short_summary = "這是一段簡短的會議摘要。"
    # Both reduce attempts overflow, so the deterministic concatenation fallback runs --
    # and with long MAP summaries that concatenation is ITSELF over the cap, which is
    # exactly the real-world shape that shipped 2,181-token "summaries".
    model = _ReduceScripted(long_summary, long_summary, short_summary, map_response="很好 " * 400)
    result = run_map_reduce(meeting(12), model, budget=200, token_len=heuristic_token_len)
    assert result.windows > 1, "fixture needs a real reduce"
    assert result.compress_passes >= 1
    assert result.prose.over_budget is False
    assert result.prose.text == short_summary


def test_compress_stops_rather_than_looping_on_a_stubborn_model() -> None:
    """A model that will not shorten must not spin: bounded by MAX_COMPRESS_PASSES."""
    from arcsum.baseline import MAX_COMPRESS_PASSES

    model = _ReduceScripted("很好 " * 1500, map_response="很好 " * 400)
    result = run_map_reduce(meeting(12), model, budget=200, token_len=heuristic_token_len)
    assert result.compress_passes <= MAX_COMPRESS_PASSES


def test_in_budget_reduce_output_is_not_compressed() -> None:
    model = _ReduceScripted("這是一段簡短的會議摘要。")
    result = run_map_reduce(meeting(12), model, budget=200, token_len=heuristic_token_len)
    assert result.compress_passes == 0


def test_reduce_falls_back_to_concatenation_when_the_call_fails() -> None:
    """The map fallback fixed the map leg; llama.cpp's invalid-UTF-8 500 then struck the
    REDUCE leg and still cost a meeting. Paired scoring means that also costs the AGENT
    arm the meeting, withholding G3 for n < min_n. Concatenation preserves every
    window's content and is MORE extractive than a real reduce, so it favours the
    baseline -- the workaround must not flatter the treatment."""
    calls = {"n": 0}

    def map_ok_reduce_explodes(system: str, user: str) -> str:
        if system == reduce_system_prompt():
            calls["n"] += 1
            raise RuntimeError("llama-server 500: Content-only format")
        return "這段討論了搬遷案的細節。"

    utts = [Utterance("S1", f"第{i}項議程的詳細討論內容。" * 20) for i in range(60)]
    result = run_map_reduce(utts, map_ok_reduce_explodes)

    assert calls["n"] > 0  # the reduce call really was attempted and really did fail
    assert result.prose.text  # a meeting survives instead of raising
