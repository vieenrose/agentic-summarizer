"""The applier and deterministic guards. The harness owns the final word — nothing the
model emits is trusted; every op passes through `apply_ops` before it can touch memory.

**Refusal-not-half-application.** `Memory`'s mutations already return a reason string on
refusal rather than partially applying (SPEC §4.2: "never half-applied into the corpus").
This module adds two more refusal sources on top: the language guard (`arcsum.lang`) and
the contradiction guard below.

**The contradiction guard's direction is INVERTED relative to the prior project.** The
prior project's guard compared a candidate bullet's polarity against a *later* bullet's
anchor timestamp — but v2 has no timestamps, and every point already in memory was, by
construction, added from an earlier or same read of the transcript (points only ever
enter memory from the chunk currently being processed). So "the other one is later" can
never be true in a forward read; porting that comparison literally would ship a guard
that looks right but structurally never fires. The v2-correct formulation follows the
same principle — *the meeting's later word wins* — restated for a design with no
timestamps: an `ADD` that contradicts an EXISTING point is refused, unless that point was
already removed by a `DROP` earlier in the same step's emission order. Since `apply_ops`
already applies ops in emission order, `DROP` then `ADD` succeeds for free — the guard
only fires when a bare `ADD` tries to assert the opposite of something still standing.

Points added within the SAME step as the candidate make no ordering claim on each other
(a 2,500-token chunk can span a proposal and its resolution; the model condensing that
into one point is a training/prompt concern, not a harness one) — so the guard only
compares against points from a STRICTLY EARLIER chunk. `Point.chunk` is what makes that
comparison possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arcsum.chunker import Chunk
from arcsum.lang import MIN_CJK_RATIO_POINT, MIN_CJK_RATIO_PROSE, check_zh_tw
from arcsum.memory import Memory
from arcsum.ops import Add, Arc, Drop, Malformed, Nop, Op, render_op
from arcsum.tokens import lexical_tokens

#: K consecutive NOP steps over content-rich chunks flags a coverage gap (SPEC §8 risk 3).
NOP_COLLAPSE_K = 3

#: Same-subject, opposite-polarity overlap threshold for the contradiction guard. Ported
#: from the prior project's `_contradicts_timeline` unchanged.
CONTRADICTION_OVERLAP = 0.34

#: Deliberately small and explicit, following the prior project's stated doctrine for
#: this exact lexicon: "a wrong 'contradiction' verdict silently drops a true decision,
#: so this errs toward not firing." Pairs confirmed by the prior project's judge selftest
#: (`FLIPS`): 通過/否決, 核准/駁回, 同意/拒絕. Negatives win when both appear in one text.
_POSITIVE = ("通過", "核准", "批准", "同意", "確認", "採納", "定案", "附議")
_NEGATIVE = (
    "否決",
    "駁回",
    "拒絕",
    "取消",
    "延後",
    "暫緩",
    "擱置",
    "撤回",
    "撤銷",
    "廢止",
    "反對",
    "不同意",
    "不核准",
    "不通過",
    "未通過",
)


def polarity(text: str) -> int:
    """+1 affirmative, -1 negative, 0 unknown. Negatives win: "not approved" must not
    read as approved just because "approved" appears as a substring."""
    if any(m in text for m in _NEGATIVE):
        return -1
    if any(m in text for m in _POSITIVE):
        return 1
    return 0


def contradiction(memory: Memory, text: str, chunk_index: int) -> str | None:
    """Refuse `text` if it contradicts an existing point about the same subject from a
    STRICTLY EARLIER chunk. `None` if there is nothing to refuse.

    Fires only when both sides have a known polarity and the candidate's polarity
    differs from the existing point's — an unknown-polarity point never triggers or is
    triggered by this guard, by design (see `_POSITIVE`/`_NEGATIVE`'s doctrine).
    """
    pol = polarity(text)
    if pol == 0:
        return None
    subject = lexical_tokens(text)
    if not subject:
        return None
    for existing in memory.points:
        if existing.chunk >= chunk_index:
            continue  # same-step or non-earlier: no ordering claim, do not fire
        other_pol = polarity(existing.text)
        if other_pol == 0 or other_pol == pol:
            continue
        other_tokens = lexical_tokens(existing.text)
        overlap = len(subject & other_tokens) / max(len(subject | other_tokens), 1)
        if overlap >= CONTRADICTION_OVERLAP:
            return (
                f"contradicts existing point «{existing.text}»: DROP it in the same step to revise"
            )
    return None


@dataclass(frozen=True, slots=True)
class AppliedOp:
    """One op's verdict. `reason` explains a refusal; `note` is informational on a
    SUCCESSFUL op — kept as two fields rather than one overloaded field, unlike the
    prior project, whose single `reason` field was sniffed downstream for a substring
    to tell the two meanings apart."""

    op: Op
    applied: bool
    reason: str | None = None
    note: str | None = None

    def log_line(self) -> str:
        verdict = "ok" if self.applied else f"dropped: {self.reason}"
        return f"[{verdict}] {render_op(self.op)}"


@dataclass
class Outcome:
    results: list[AppliedOp] = field(default_factory=list)
    nop_collapse: bool = False

    @property
    def applied(self) -> int:
        """Count of successfully applied ops, EXCLUDING `Nop` (it is not an edit)."""
        return sum(1 for r in self.results if r.applied and not isinstance(r.op, Nop))

    @property
    def valid_op_rate(self) -> float | None:
        """Applied / attempted, with `Nop` excluded from BOTH numerator and
        denominator. `None` on a step with no non-NOP ops. Counting NOP into only one
        side has been a real, twice-repeated bug in the prior project's history —
        exclude it from both, deliberately."""
        non_nop = [r for r in self.results if not isinstance(r.op, Nop)]
        if not non_nop:
            return None
        return sum(1 for r in non_nop if r.applied) / len(non_nop)

    @property
    def malformed(self) -> list[AppliedOp]:
        return [r for r in self.results if isinstance(r.op, Malformed)]


def apply_ops(
    memory: Memory,
    ops: list[Op],
    chunk: Chunk,
    *,
    consecutive_nops: int = 0,
    lang_check: bool = True,
) -> Outcome:
    """Validate and apply a step's ops IN PLACE, in emission order. Returns per-op
    verdicts.

    `consecutive_nops` is the running count BEFORE this step; the caller keeps the
    tally and passes it in, so this function stays stateless. `lang_check=False` exists
    only for testing the other guards in isolation from `arcsum.lang`.
    """
    outcome = Outcome()
    substantive = False

    for op in ops:
        match op:
            case Nop():
                outcome.results.append(AppliedOp(op, True))

            case Malformed(_, reason):
                outcome.results.append(AppliedOp(op, False, reason))

            case Arc(text):
                if lang_check and (bad := check_zh_tw(text, min_cjk_ratio=MIN_CJK_RATIO_PROSE)):
                    outcome.results.append(AppliedOp(op, False, bad))
                    continue
                reason = memory.set_arc(text)
                outcome.results.append(AppliedOp(op, reason is None, reason))
                substantive = substantive or reason is None

            case Add(point):
                if lang_check and (bad := check_zh_tw(point, min_cjk_ratio=MIN_CJK_RATIO_POINT)):
                    outcome.results.append(AppliedOp(op, False, bad))
                    continue
                if contra := contradiction(memory, point, chunk.index):
                    outcome.results.append(AppliedOp(op, False, contra))
                    continue
                reason = memory.add_point(point, chunk.index)
                outcome.results.append(AppliedOp(op, reason is None, reason))
                substantive = substantive or reason is None

            case Drop(prefix):
                reason = memory.drop_point(prefix)
                outcome.results.append(AppliedOp(op, reason is None, reason))
                substantive = substantive or reason is None

            case _:  # pragma: no cover — Op is a closed union; defensive only
                outcome.results.append(AppliedOp(op, False, "unhandled op type"))

    memory.enforce_caps()

    # NOP-collapse: DETECT AND RECORD ONLY, never repair here. SPEC §4.1 names exactly
    # four ops and says the harness applies them deterministically — a hidden fifth
    # "fallback" call (the prior project's map-reduce-window rescue) would run the
    # control arm's summariser inside the treatment arm, contaminating the very
    # comparison SPEC §5.2's gates exist to make. Only a genuinely content-rich chunk
    # counts — a chunk that really has nothing in it deserves a NOP.
    if not substantive and chunk.is_content_rich():
        outcome.nop_collapse = consecutive_nops + 1 >= NOP_COLLAPSE_K

    return outcome
