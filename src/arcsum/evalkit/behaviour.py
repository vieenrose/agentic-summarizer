"""Reading-step behaviour, aggregated from a live `Trace` — the instrument class that was
missing when a churning checkpoint passed every gate and shipped.

**The failure this exists to catch, in full, because it is the specification.** On
2026-09-02 `mixed-e3` was published to the reference demo on the strength of every offline
gate: revision probe 3/27 -> 8/27, real-ASR "curated" 17/20 -> 19/20, all three G3 gates
passing. The first real meeting run through the demo produced this, over six chunks:

    step 0  ADD  南辰封裝測試費用高且明年將由自己辦
    step 1  DROP «南辰封裝»  +  ADD 南辰封裝測試與供給不足，明年將由自己辦
    step 2  DROP «南辰封裝»  +  ADD 南辰封裝測試與供給不足，明年將由自己辦   <- restates dropped
    step 3  DROP «南辰封裝»  +  ADD 天璿與南辰…由自己辦   <- restates dropped
    step 4  DROP ... + ADD ...  <- restates dropped
    step 5  DROP ... + ADD ...  <- restates dropped

Six chunks, ONE surviving point, the ARC frozen at step 0 (every later `ARC:` refused
`arc unchanged`), and then 553 characters of confident competitive-strategy prose
synthesised out of that single point. `qwen-tools-v5` on the same transcript in the same
configuration: 4 points, 0 churn, 304 characters.

**Every signal needed to catch this was already being recorded and none was aggregated.**
`Outcome.churn_points` existed and fired correctly four times. `AppliedOp.reason` carried
`arc unchanged` five times. The gap was not detection; it was that no gate ever reduced
those per-step notes to a per-run number. This module is that reduction, and nothing more.

**Why the length-based curation check could not see it.** `tools/asr_gate.py` counts a
meeting as curated when its summary exceeds a fixed character floor. A 553-character
confabulation clears that floor easily, so the metric that justified shipping was scoring
this run as a SUCCESS. `chars_per_point` is the discriminator that separates them: v5 at
94 ch/pt versus a churning run at 553 ch/pt on the same meeting. A summary is supposed to
be a rendering of the memory; when the ratio explodes, the prose is coming from somewhere
other than what was read.

`chars_per_point` is a RATIO, so read it with `points` beside it — it is undefined on an
empty memory and unstable at one or two points, which is exactly the regime the churn
counters cover.
"""

from __future__ import annotations

from dataclasses import dataclass

from arcsum.agent import Trace

#: A run holding this many points or fewer, after this many chunks, is starved: the memory
#: is not accumulating. Not a gate threshold — a reporting flag, chosen from the observed
#: failure (1 point over 6 chunks) against healthy runs on the same input (4-5 points).
STARVED_POINTS_PER_CHUNK = 0.5

#: Above this, the summary is asserting far more than the memory holds. The failing run
#: measured 553; healthy runs on the same meeting measured 82-94. Deliberately loose: this
#: flags a run for reading, it does not fail it.
CONFABULATION_CHARS_PER_POINT = 250.0

#: Below this, the summary is not RENDERING the memory it built — the opposite failure,
#: and one a user reported before any instrument could see it: a meeting whose reading step
#: recorded 15 points produced a 53-character summary that used almost none of them, i.e.
#: **4 characters per point**. Healthy runs measure 71-98. A point may be up to
#: `POINT_TOKENS` (25) characters, so a summary averaging under ~20 characters per point
#: cannot be saying something about each one; it has silently dropped most of the meeting.
#:
#: This threshold and `CONFABULATION_CHARS_PER_POINT` bracket the same ratio from opposite
#: sides. Reporting only the upper bound would have caught the churn regression and missed
#: the complaint that preceded it.
UNDER_RENDERING_CHARS_PER_POINT = 20.0

#: Character-trigram containment at which a recorded point counts as RENDERED in the prose.
#: Trigrams rather than exact match because a summary legitimately rewrites a point into
#: running prose; requiring the literal string would score good writing as a miss.
RENDERED_CONTAINMENT = 0.30


def trigrams(text: str) -> set[str]:
    s = "".join(text.split())
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else ({s} if s else set())


def containment(needle: str, haystack: str) -> float:
    """Fraction of `needle`'s character trigrams present in `haystack`."""
    a = trigrams(needle)
    if not a:
        return 1.0
    return len(a & trigrams(haystack)) / len(a)


@dataclass(frozen=True)
class BehaviourReport:
    """One meeting's reading-step behaviour. Every field is a count or a ratio over counts
    already present in the trace — this module adds no new judgement, only arithmetic."""

    meeting: str
    chunks: int
    steps: int
    nop_steps: int
    failed_steps: int
    points: int
    #: Applied `ADD`s that restate a point dropped in the SAME step. The direct churn
    #: signal: the model is rewriting what it already had instead of reading forward.
    churn_events: int
    #: Steps whose `ARC:` was refused as `arc unchanged`. A frozen arc means the meeting's
    #: narrative stopped advancing while the transcript kept going.
    arc_frozen_steps: int
    #: Ops refused for any reason, and the attempted total, so a refusal RATE is available
    #: without re-deriving it from per-op records.
    refused_ops: int
    attempted_ops: int
    prose_chars: int
    hedge_points: int
    ungrounded_numbers: int
    #: The memory's ARC slot is non-empty. The ARC is a memory unit, not decoration: a run
    #: can legitimately set a real ARC and zero POINTS (measured: `ivod-17704`, a genuine
    #: 173-character summary built from the arc alone), and dividing prose by POINTS alone
    #: reported that as infinite confabulation. `chars_per_memory_unit` counts it.
    has_arc: bool = False
    #: Everything the meeting RECORDED — the working set plus the journal (SPEC §4.1 v1.1).
    #:
    #: **`points` alone stopped meaning "what was recorded" when v1.1 landed**, and this
    #: module was not updated with it, so three things were quietly wrong: `starved` fired on
    #: long meetings whose points had merely been retired, `chars_per_point` divided by a
    #: denominator too small to reveal under-rendering, and G5 retention could not be computed
    #: at all. Measured on the pool's own replays, eviction moves 9.6% of points out of
    #: `points` and into the journal, concentrated in the 28% of meetings that overflow.
    #:
    #: Defaults to 0 so pre-v1.1 reports deserialize; `recorded_units` falls back to the
    #: working set when it is unset, which is the correct reading for a v1.0 trace where the
    #: two were the same thing.
    recorded_points: int = 0
    #: Recorded points whose text is actually rendered in the prose, by `RENDERED_CONTAINMENT`.
    #: This is G5's numerator: the journal guarantees a point SURVIVES to synthesis, and this
    #: measures whether synthesis then USED it — which is the different, and now binding,
    #: question.
    rendered_points: int = 0
    #: The model was never called for synthesis because memory was empty, and `prose_chars`
    #: counts the harness's fixed `EMPTY_MEMORY_PROSE` string. `Synthesis` carries this flag
    #: precisely so scoring can tell "declined to invent content" from "produced a summary",
    #: and the first version of this module ignored it — which flagged correct abstention on
    #: a genuinely noisy meeting (`ivod-17673`, stutter-repeated ASR where NOP is the right
    #: answer) as infinite confabulation. Abstention is a SUCCESS of the empty-memory guard.
    abstained: bool = False
    #: Tokens the READING steps prefilled and decoded, straight off `Trace.usage`.
    #:
    #: **G4 is the only gate whose inputs were not artifacts.** Its per-meeting wall clock was
    #: reconstructed by hand from a device benchmark plus a decode length inherited from
    #: `qwen-tools-v5` (~190 tokens/step) and then applied to every later checkpoint. Decode
    #: length is NOT a device constant — it is a property of the checkpoint, and the first
    #: RAFT pool's targets run 1.45x longer, enough on its own to move a meeting from 20.3 to
    #: 22.5 minutes against a 20.00 ceiling. So a checkpoint can fail G4 purely by recording
    #: more, which is exactly what the anti-starvation work does. Carrying the profile here
    #: lets `evalkit.latency` project from the run being scored instead of from memory.
    prefill_tokens: int = 0
    decode_tokens: int = 0

    @property
    def churn_rate(self) -> float:
        """Churn events per step. 0.67 in the shipped failure; 0.0 in the healthy run."""
        return self.churn_events / self.steps if self.steps else 0.0

    @property
    def points_per_chunk(self) -> float:
        """Accumulation rate over everything RECORDED, not just what survived in the working
        set — a long meeting that recorded 40 points and retired 24 of them was accumulating
        fine, and counting only survivors reported it as starved."""
        base = self.recorded_points or self.points
        return base / self.chunks if self.chunks else 0.0

    @property
    def recorded_units(self) -> int:
        """Everything recorded, plus the ARC when set — the denominator SYNTHESIZE actually
        faces under v1.1. Falls back to the working set for pre-v1.1 reports, where the
        journal did not exist and the two counts were identical."""
        base = self.recorded_points or self.points
        return base + (1 if self.has_arc else 0)

    @property
    def retention(self) -> float:
        """SPEC G5: the fraction of recorded points the summary actually renders.

        The journal made SURVIVAL to synthesis automatic, which moved the failure one step
        later: a point can reach the prompt and still not reach the prose. That is the
        deficit v1.1 still shows (~40 entries in, 346 characters out), so it needs its own
        number rather than being inferred from a ratio."""
        if not self.recorded_points:
            return 1.0 if self.abstained else 0.0
        return self.rendered_points / self.recorded_points

    @property
    def memory_units(self) -> int:
        """POINTS plus the ARC when set. SPEC §4.1's memory is two slots, and the ratio
        below is only honest if the denominator is the whole memory."""
        return self.recorded_units

    @property
    def chars_per_point(self) -> float:
        """Summary characters per RECORDED memory unit (everything recorded, plus the arc).
        `inf` only when prose was generated from a genuinely empty memory — the extreme of
        the failure, not a division error to be swallowed as 0.0."""
        if self.memory_units:
            return self.prose_chars / self.memory_units
        # Abstention is not division by zero dressed as fabrication: the harness declined
        # to call the model at all, so there is no prose to attribute.
        if self.abstained:
            return 0.0
        return float("inf") if self.prose_chars else 0.0

    @property
    def decode_tokens_per_step(self) -> float:
        """Decode tokens per reading step — the G4 term that varies BY CHECKPOINT."""
        return self.decode_tokens / self.steps if self.steps else 0.0

    @property
    def prefill_tokens_per_step(self) -> float:
        return self.prefill_tokens / self.steps if self.steps else 0.0

    @property
    def starved(self) -> bool:
        """Memory failed to accumulate. Still true when the run abstained — an empty memory
        after several chunks is worth seeing — but see `flags`, which labels it honestly
        rather than calling it fabrication."""
        return self.chunks > 1 and self.points_per_chunk <= STARVED_POINTS_PER_CHUNK

    @property
    def confabulating(self) -> bool:
        return not self.abstained and self.chars_per_point > CONFABULATION_CHARS_PER_POINT

    @property
    def under_rendering(self) -> bool:
        """Memory was built and then largely not used. Requires several points: at one or
        two points a short summary is proportionate, not a failure."""
        return self.memory_units >= 3 and self.chars_per_point < UNDER_RENDERING_CHARS_PER_POINT

    @property
    def flags(self) -> tuple[str, ...]:
        out = []
        if self.abstained:
            out.append("abstained (empty memory)")
        if self.churn_events:
            out.append(f"churn x{self.churn_events}")
        if self.starved:
            out.append(f"starved {self.points}pt/{self.chunks}ch")
        if self.confabulating:
            out.append(f"confabulation {self.chars_per_point:.0f}ch/pt")
        if self.under_rendering:
            out.append(f"under-rendered {self.chars_per_point:.0f}ch/pt x{self.points}pt")
        if self.arc_frozen_steps > 1:
            out.append(f"arc frozen x{self.arc_frozen_steps}")
        if self.hedge_points:
            out.append(f"hedge x{self.hedge_points}")
        if self.ungrounded_numbers:
            out.append(f"ungrounded-numbers x{self.ungrounded_numbers}")
        return tuple(out)


def from_trace(meeting: str, trace: Trace) -> BehaviourReport:
    """Reduce a live `Trace` to its behaviour counts.

    Deliberately takes a `Trace` and not a serialized run. `supervision/report.py` argues
    at length that rates welded to live objects avoid a numerator/denominator split across
    a serialization boundary — a bug that occurred twice in the prior project. The same
    argument applies here, so this reads the objects the agent just produced.
    """
    churn = arc_frozen = refused = attempted = hedge = 0
    for s in trace.steps:
        outcome = getattr(s, "outcome", None)
        if outcome is None:
            continue
        churn += len(outcome.churn_points)
        hedge += len(outcome.hedge_points)
        for r in outcome.results:
            if type(r.op).__name__ == "Nop":
                continue
            attempted += 1
            if not r.applied:
                refused += 1
                if (r.reason or "") == "arc unchanged":
                    arc_frozen += 1

    usage = getattr(trace, "usage", None)
    syn = trace.synthesis
    # `synthesis_view()` is the exact list `build_synth_prompt` renders, so the denominator
    # here and the model's actual input cannot drift apart.
    view = trace.memory.synthesis_view()
    prose = syn.prose.text if syn and not syn.skipped_empty_memory else ""
    rendered = sum(1 for e in view if containment(e.point.text, prose) >= RENDERED_CONTAINMENT)
    return BehaviourReport(
        meeting=meeting,
        chunks=len(trace.steps) + len(trace.failed_steps),
        steps=len(trace.steps),
        nop_steps=sum(1 for s in trace.steps if s.is_nop),
        failed_steps=len(trace.failed_steps),
        points=len(trace.memory.points),
        churn_events=churn,
        arc_frozen_steps=arc_frozen,
        refused_ops=refused,
        attempted_ops=attempted,
        prose_chars=syn.prose.chars if syn else 0,
        hedge_points=hedge,
        ungrounded_numbers=len(syn.ungrounded_numbers) if syn else 0,
        has_arc=bool(trace.memory.arc),
        recorded_points=len(view),
        rendered_points=rendered,
        abstained=bool(syn and syn.skipped_empty_memory),
        # Tolerate a trace without `usage`: this module is deliberately testable against
        # lightweight stubs, and a missing token profile must degrade to "unknown" (0), never
        # to an exception — the behaviour counts do not depend on it.
        prefill_tokens=getattr(usage, "prefill_tokens", 0),
        decode_tokens=getattr(usage, "decode_tokens", 0),
    )


@dataclass(frozen=True)
class BehaviourSummary:
    n_meetings: int
    total_steps: int
    total_churn: int
    meetings_with_churn: int
    meetings_starved: int
    meetings_confabulating: int
    mean_points_per_chunk: float
    median_chars_per_point: float
    per_meeting: tuple[BehaviourReport, ...]
    #: SPEC G5's numerator and denominator, summed over meetings rather than averaged over
    #: rates — a per-meeting mean would weight a 3-point meeting like a 40-point one, and the
    #: long meetings are the entire question.
    total_recorded: int = 0
    total_rendered: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0

    @property
    def mean_steps(self) -> float:
        return self.total_steps / self.n_meetings if self.n_meetings else 0.0

    @property
    def decode_tokens_per_step(self) -> float:
        """Summed, not averaged over per-meeting rates: G4 is a per-MEETING wall clock and
        the long meetings dominate it, so weighting a 3-step meeting like a 48-step one
        would understate exactly the case that fails."""
        return self.total_decode_tokens / self.total_steps if self.total_steps else 0.0

    @property
    def prefill_tokens_per_step(self) -> float:
        return self.total_prefill_tokens / self.total_steps if self.total_steps else 0.0

    @property
    def retention(self) -> float:
        """Fraction of everything recorded that the summaries actually rendered (SPEC G5).

        The journal made survival to synthesis automatic; this measures whether synthesis
        then USED what survived, which is where the remaining coverage deficit lives."""
        return self.total_rendered / self.total_recorded if self.total_recorded else 0.0

    @property
    def meetings_under_rendering(self) -> int:
        return sum(1 for r in self.per_meeting if r.under_rendering)

    @property
    def churn_rate(self) -> float:
        return self.total_churn / self.total_steps if self.total_steps else 0.0

    @property
    def clean_meetings(self) -> int:
        """Meetings with NO behaviour flag at all. This is the number that should have
        replaced the length-based `curated` count: it cannot be satisfied by writing
        more."""
        return sum(1 for r in self.per_meeting if not r.flags)


def summarise(reports: list[BehaviourReport]) -> BehaviourSummary:
    finite = sorted(r.chars_per_point for r in reports if r.chars_per_point != float("inf"))
    median = finite[len(finite) // 2] if finite else 0.0
    n = len(reports)
    return BehaviourSummary(
        n_meetings=n,
        total_steps=sum(r.steps for r in reports),
        total_churn=sum(r.churn_events for r in reports),
        meetings_with_churn=sum(1 for r in reports if r.churn_events),
        meetings_starved=sum(1 for r in reports if r.starved),
        meetings_confabulating=sum(1 for r in reports if r.confabulating),
        mean_points_per_chunk=(sum(r.points_per_chunk for r in reports) / n) if n else 0.0,
        median_chars_per_point=median,
        per_meeting=tuple(reports),
        total_recorded=sum(r.recorded_points for r in reports),
        total_rendered=sum(r.rendered_points for r in reports),
        total_prefill_tokens=sum(r.prefill_tokens for r in reports),
        total_decode_tokens=sum(r.decode_tokens for r in reports),
    )
