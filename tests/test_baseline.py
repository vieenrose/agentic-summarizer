"""Pins the fair map-reduce baseline (SPEC §5.2) — the opponent the agent must beat.

The defining property under test: the map step sees NO STATE. `build_map_prompt` (already
pinned in `test_prompts.py`) contains no ARC/POINTS/MEMORY vocabulary at all, so a
Scripted fake that records its prompts is enough to prove this without any weights.
"""

from __future__ import annotations

from arcsum.baseline import run_map_reduce, summarise_window
from arcsum.chunker import Chunk
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
    assert result.prompt_version == "sys-v1"
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
