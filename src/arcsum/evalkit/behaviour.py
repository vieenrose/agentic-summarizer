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
    #: The model was never called for synthesis because memory was empty, and `prose_chars`
    #: counts the harness's fixed `EMPTY_MEMORY_PROSE` string. `Synthesis` carries this flag
    #: precisely so scoring can tell "declined to invent content" from "produced a summary",
    #: and the first version of this module ignored it — which flagged correct abstention on
    #: a genuinely noisy meeting (`ivod-17673`, stutter-repeated ASR where NOP is the right
    #: answer) as infinite confabulation. Abstention is a SUCCESS of the empty-memory guard.
    abstained: bool = False

    @property
    def churn_rate(self) -> float:
        """Churn events per step. 0.67 in the shipped failure; 0.0 in the healthy run."""
        return self.churn_events / self.steps if self.steps else 0.0

    @property
    def points_per_chunk(self) -> float:
        return self.points / self.chunks if self.chunks else 0.0

    @property
    def memory_units(self) -> int:
        """POINTS plus the ARC when set. SPEC §4.1's memory is two slots, and the ratio
        below is only honest if the denominator is the whole memory."""
        return self.points + (1 if self.has_arc else 0)

    @property
    def chars_per_point(self) -> float:
        """Summary characters per surviving MEMORY UNIT (points, plus the arc when set).
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

    syn = trace.synthesis
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
        abstained=bool(syn and syn.skipped_empty_memory),
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
    )
