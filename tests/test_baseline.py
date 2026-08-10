"""Map-reduce baseline and the token instrument GT4 depends on."""

from __future__ import annotations

import pytest

from voxsum.agent import run_cursor
from voxsum.baseline import run_map_reduce, summarise_window
from voxsum.chunker import Chunk, heuristic_token_len, iter_chunks
from voxsum.state import CAPS
from voxsum.transcript import Utterance, parse_transcript

EXAMPLE = (
    "[0:00] S1: Let us discuss the office move.\n"
    "[2:30] S2: I propose we move to Building B.\n"
    "[5:10] S1: Agreed, Building B it is.\n"
    "[6:02] S2: I will circulate the move checklist by Friday.\n"
)


class MapModel:
    """Emits map-step bullets for whichever lines are in the window it is shown."""

    def __init__(self, *, per_line: int = 1, reduce_reply: str | None = None) -> None:
        self.per_line = per_line
        self.reduce_reply = reduce_reply
        self.map_calls = 0
        self.reduce_calls = 0

    def __call__(self, system: str, user: str) -> str:
        if user.startswith("SECTION:"):
            self.reduce_calls += 1
            if self.reduce_reply is not None:
                return self.reduce_reply
            # Keep the first `cap` bullets it was given, verbatim.
            rows = user.splitlines()
            cap = int(next(r.split(": ")[1] for r in rows if r.startswith("CAP:")))
            bullets = [r for r in rows if r.startswith("- ")]
            return "\n".join(bullets[:cap])
        self.map_calls += 1
        out = []
        for line in user.splitlines():
            if not line.startswith("["):
                continue
            clock = line[1 : line.index("]")]
            for i in range(self.per_line):
                out.append(f"TOPICS - point {clock} number {i} [{clock}]")
        return "\n".join(out) if out else "NONE"


def test_map_step_parses_bullets_and_anchors() -> None:
    chunk = Chunk(0, tuple(parse_transcript(EXAMPLE)))
    digest = summarise_window(chunk, MapModel())
    assert digest["TOPICS"], "map step produced nothing"
    assert all(chunk.has_line(b.anchor) for b in digest["TOPICS"])


def test_map_step_sees_no_state() -> None:
    """The defining property of map-reduce: windows are digested independently."""
    seen: list[str] = []

    def spy(system: str, user: str) -> str:
        seen.append(user)
        return "NONE"

    run_map_reduce(parse_transcript(EXAMPLE), spy, budget=20)
    assert seen and all("STATE:" not in u for u in seen)


def test_none_reply_yields_no_bullets() -> None:
    chunk = Chunk(0, tuple(parse_transcript(EXAMPLE)))
    assert summarise_window(chunk, lambda s, u: "NONE") == {s: [] for s in CAPS}


def test_bad_anchor_falls_back_to_matcher() -> None:
    chunk = Chunk(0, tuple(parse_transcript(EXAMPLE)))
    digest = summarise_window(chunk, lambda s, u: "DECISIONS - Building B agreed [99:99]")
    (bullet,) = digest["DECISIONS"]
    # Held to the same anchor standard as the agent path.
    assert chunk.has_line(bullet.anchor)
    assert "99:99" not in bullet.text


def test_malformed_map_lines_are_ignored() -> None:
    chunk = Chunk(0, tuple(parse_transcript(EXAMPLE)))
    digest = summarise_window(chunk, lambda s, u: "Here is a summary of the meeting.\nblah")
    assert all(not v for v in digest.values())


def test_run_map_reduce_produces_capped_notes() -> None:
    lines = [Utterance(i * 30, "S1", f"discussion point {i} " * 8) for i in range(20)]
    result = run_map_reduce(lines, MapModel(per_line=2), budget=128)
    assert result.windows > 1
    for section, cap in CAPS.items():
        assert len(result.state.bullets(section)) <= cap


def test_reduce_is_called_only_for_over_cap_sections() -> None:
    lines = [Utterance(i * 30, "S1", f"point {i}") for i in range(12)]
    model = MapModel()
    result = run_map_reduce(lines, model, budget=64)
    # Only TOPICS receives bullets from this model, so exactly one reduce call.
    assert result.reduce_calls == 1 == model.reduce_calls


def test_duplicate_bullets_are_merged_across_windows() -> None:
    """Overlapping windows digest the same line twice; caps must not fill with duplicates."""
    lines = [Utterance(i * 30, "S1", f"point {i} " * 10) for i in range(8)]
    result = run_map_reduce(lines, MapModel(), budget=96)
    texts = [b.text for b in result.state.bullets("TOPICS")]
    assert len(texts) == len(set(texts))


def test_failed_reduce_falls_back_to_spread() -> None:
    """A reduce step that returns junk must not delete the meeting's content."""
    lines = [Utterance(i * 30, "S1", f"point {i}") for i in range(12)]
    result = run_map_reduce(lines, MapModel(reduce_reply="I cannot do that."), budget=64)
    topics = result.state.bullets("TOPICS")
    assert len(topics) == CAPS["TOPICS"], "empty section would lose the meeting"
    assert topics[0].anchor == 0 and topics[-1].anchor is not None


def test_reduce_cannot_invent_an_anchor() -> None:
    lines = [Utterance(i * 30, "S1", f"point {i}") for i in range(12)]
    result = run_map_reduce(
        lines, MapModel(reduce_reply="- fabricated decision [59:59]"), budget=64
    )
    anchors = {b.anchor for b in result.state.bullets("TOPICS")}
    assert clock_not_in(anchors, 3599)


def clock_not_in(anchors: set[int | None], sec: int) -> bool:
    return sec not in anchors


def test_title_is_derived_without_an_extra_call() -> None:
    lines = [Utterance(i * 30, "S1", f"point {i}") for i in range(4)]
    model = MapModel()
    result = run_map_reduce(lines, model, budget=64)
    assert result.state.title, "baseline should still produce a TITLE"
    assert model.map_calls + model.reduce_calls == result.usage.calls


# --- the GT4 instrument --------------------------------------------------------

def test_usage_is_recorded_on_both_arms() -> None:
    lines = parse_transcript(EXAMPLE)
    trace = run_cursor(lines, lambda s, u: "NOP", budget=20)
    baseline = run_map_reduce(lines, lambda s, u: "NONE", budget=20)
    for usage in (trace.usage, baseline.usage):
        assert usage.calls > 0
        assert usage.prefill_tokens > 0


def _long_meeting(n: int = 400) -> list[Utterance]:
    return [Utterance(i * 20, "S1", f"discussion point number {i} " * 10) for i in range(n)]


def _prefill_ratio(model, budget: int) -> float:
    lines = _long_meeting()
    trace = run_cursor(lines, model, budget=budget, step_budget=10**6)
    baseline = run_map_reduce(lines, lambda s, u: "NONE", budget=budget)
    assert trace.usage.calls == baseline.usage.calls, "same windows on both arms"
    return trace.usage.prefill_tokens / baseline.usage.prefill_tokens


def test_gt4_must_be_measured_at_production_chunk_size() -> None:
    """The prefill ratio is dominated by SYS at small chunks — a toy budget misreports GT4.

    CURSOR's SYS is ~314 tokens against the map step's ~100, so at a 128-token chunk the
    fixed cost is most of the prompt and the ratio exceeds 2x. It falls to ~1.1x at 2048.
    Any GT4 number quoted at a non-production chunk size is meaningless.
    """
    nop = lambda s, u: "NOP"  # noqa: E731
    assert _prefill_ratio(nop, 128) > 1.8
    assert _prefill_ratio(nop, 2048) < 1.25


class Saturating:
    """Keeps every section at its cap, so each step pays the maximum STATE cost."""

    def __call__(self, system: str, user: str) -> str:
        if "[" not in user.split("CHUNK:")[-1]:
            return "NOP"
        clock = user.split("CHUNK:")[-1].split("[")[1].split("]")[0]
        return "\n".join(
            f"ADD {section} - filler bullet {section} {i} " + "word " * 12 + f"[{clock}]"
            for section in CAPS
            for i in range(2)
        )


def _long_turn_meeting(words_per_line: int, n: int) -> list[Utterance]:
    """Long-monologue transcript — VCSum zh runs to ~2.6k chars per line."""
    return [Utterance(i * 20, "S1", ("word " * words_per_line).strip()) for i in range(n)]


@pytest.mark.parametrize("words_per_line", [200, 600, 1200])
def test_chunks_stay_well_packed_on_long_turn_transcripts(words_per_line: int) -> None:
    """Wasted chunk room is not free — it multiplies the per-step SYS + STATE cost.

    Before the packer split an over-long line into the *remaining* room, long-turn
    transcripts filled chunks to only ~73%: two pieces fit and a third did not. That waste
    inflated the step count enough to push GT4 past +25%.
    """
    lines = _long_turn_meeting(words_per_line, 40)
    chunks = list(iter_chunks(lines, budget=2048))
    fill = sum(heuristic_token_len(c.render()) for c in chunks) / (len(chunks) * 2048)
    assert fill > 0.9, f"chunks only {fill:.0%} full — step count inflated"


def test_gt4_holds_with_a_saturated_state() -> None:
    """GT4 with every section at cap — the worst case the gate has to survive.

    A full STATE block is ~698 tokens, already above §8's "<= 600" assumption, so with SYS
    that is ~1k tokens of fixed overhead per step. It still clears +25% at the specified
    2048-token chunk *provided chunks pack well*; the two facts are linked, which is why
    the packing test above sits next to this one.
    """
    assert _prefill_ratio(Saturating(), 2048) <= 1.25


def test_saturated_state_overhead_exceeds_the_spec_estimate() -> None:
    """§8 budgets STATE at <= 600 tokens; measured at cap it is ~700.

    Pinned because the gate has little headroom: if bullet caps or the SYS prompt grow, GT4
    is the first thing to break, and it will break quietly.
    """
    from voxsum.render import render_for_prompt
    from voxsum.state import NotesState

    full = NotesState()
    full.set_title("A meeting title of eight words exactly here")
    for section, cap in CAPS.items():
        for i in range(cap):
            full.add(section, f"bullet {i} in {section} " + "word " * 15, i * 37)
    assert heuristic_token_len(render_for_prompt(full)) > 600


def test_usage_uses_the_injected_tokenizer() -> None:
    seen: list[str] = []

    def counting(text: str) -> int:
        seen.append(text)
        return heuristic_token_len(text)

    run_map_reduce(parse_transcript(EXAMPLE), lambda s, u: "NONE", budget=20, token_len=counting)
    assert seen


@pytest.mark.parametrize("lang", ["en", "zh-TW"])
def test_baseline_runs_in_both_languages(lang: str) -> None:
    result = run_map_reduce(parse_transcript(EXAMPLE), MapModel(), lang=lang, budget=20)
    assert result.windows > 0
