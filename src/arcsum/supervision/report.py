"""Aggregate supervision diagnostics over real `agent.Trace` objects (SPEC §4.2,
§8 risk 3): valid-op rate, NOP share on rich chunks, `DROP`/`ARC` share, veto rate.

Built directly on `Trace`/`Step`/`Outcome` rather than a parallel on-disk schema, so
these definitions can never drift from what `agent.run_agent` and `guards.apply_ops`
actually compute — the prior project's `trace_report.py` pinned two metric
definitions specifically because getting them wrong (counting `NOP` into only one
side of a ratio) was a real, twice-repeated bug.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arcsum.agent import Trace
from arcsum.ops import Arc, Drop, Nop


def _non_nop_results(traces: Sequence[Trace]):
    return [
        r for t in traces for s in t.steps for r in s.outcome.results if not isinstance(r.op, Nop)
    ]


@dataclass(frozen=True, slots=True)
class SupervisionReport:
    total_steps: int
    #: `None` when there is no non-NOP op anywhere to compute a rate from.
    valid_op_rate: float | None
    nop_rate_on_rich_chunks: float | None
    drop_share: float | None
    arc_share: float | None
    #: Vetoed / (applied-or-refused + vetoed) — SPEC §4.2's teacher-side filter rate.
    veto_rate: float | None


def report(traces: Sequence[Trace]) -> SupervisionReport:
    non_nop = _non_nop_results(traces)

    valid_op_rate = sum(1 for r in non_nop if r.applied) / len(non_nop) if non_nop else None
    drop_share = (
        sum(1 for r in non_nop if isinstance(r.op, Drop)) / len(non_nop) if non_nop else None
    )
    arc_share = sum(1 for r in non_nop if isinstance(r.op, Arc)) / len(non_nop) if non_nop else None

    rich_steps = [s for t in traces for s in t.steps if s.chunk.is_content_rich(budget=t.budget)]
    nop_rate = sum(1 for s in rich_steps if s.is_nop) / len(rich_steps) if rich_steps else None

    total_vetoed = sum(len(s.vetoed) for t in traces for s in t.steps)
    total_seen = len(non_nop) + total_vetoed
    veto_rate = total_vetoed / total_seen if total_seen else None

    return SupervisionReport(
        total_steps=sum(len(t.steps) for t in traces),
        valid_op_rate=valid_op_rate,
        nop_rate_on_rich_chunks=nop_rate,
        drop_share=drop_share,
        arc_share=arc_share,
        veto_rate=veto_rate,
    )
