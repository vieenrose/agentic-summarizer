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

from arcsum.chunker import CHUNK_TOKENS, Chunk
from arcsum.lang import MIN_CJK_RATIO_ARC, MIN_CJK_RATIO_POINT, check_zh_tw
from arcsum.memory import Memory, normalize
from arcsum.ops import Add, Arc, Drop, Malformed, Nop, Op, Revise, render_op
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


#: Markers that leave a point's polarity UNRESOLVED — "questions WHETHER X", not "asserts
#: X". Measured 2026-08-30: a `qwen-tools-v4` reading step recorded
#: `委員質疑國有林地濫墾是否應加重刑責` (faithful — "questions whether it should be
#: strengthened"), and `synthesize_memory` deterministically (3/3 seeds) rewrote it as
#: `認為該事件不應加重刑責` — asserting the OPPOSITE polarity as settled fact. No `ADD`
#: target in `tools/gen_deliberation.py`'s training data ever used this framing, so the
#: reading step's own paraphrase choice put synthesis off-distribution, and it guessed
#: a polarity — wrong. Detected and RECORDED here, never silently rewritten: the fix (a
#: training-side ban on this phrasing, or a synthesis-side instruction to preserve
#: question form) needs to be validated before changing behaviour, per this codebase's
#: standing "detect and record, never repair in-loop" rule for NOP-collapse.
HEDGE_MARKERS = ("是否", "能否", "可否", "是不是", "有無")


def hedge_marker_in(text: str) -> str | None:
    """First hedge marker found in `text`, or `None`. A point carrying one has an
    UNRESOLVED polarity that synthesis has been measured to resolve incorrectly."""
    return next((m for m in HEDGE_MARKERS if m in text), None)


#: Similarity above which a re-`ADD` is judged to restate a point `DROP`ped in the SAME
#: step rather than revise it. Character-trigram Jaccard, so it is script-appropriate and
#: uses the same instrument as the rest of the harness.
CHURN_SIMILARITY = 0.7


def _trigrams(text: str) -> set[str]:
    t = normalize(text)
    return {t[i : i + 3] for i in range(max(len(t) - 2, 0))} or {t}


def restates_dropped(text: str, dropped: list[str]) -> str | None:
    """The text a `DROP` removed this step that `text` merely restates, or `None`.

    **Detected and RECORDED, never refused** — the same discipline as `nop_collapse` and
    `hedge_marker_in`, and for the same reason: the correct repair is genuinely ambiguous.
    A model emitting `DROP «X»` then `ADD «X'»` with X'~=X has said two contradictory
    things, and neither branch is safe to take automatically. Refusing the `ADD` honours
    the `DROP` and loses the point entirely; refusing the `DROP` keeps a point the model
    explicitly retired. Measured 2026-09-01 on G1's `budget_approval`, where
    `qwen-tools-v7` emitted `drop ["行銷預算核准"]` with `arc ...駁回...` and then re-added
    the byte-identical stale point `行銷預算核准，下一季新產品宣傳預算為兩百萬美元`, so the
    ARC recorded the reversal and POINTS recorded the superseded decision.

    Note this is trap 1's churn signature (DROP + near-identical re-ADD), which the record
    attributes to a NOP-starved pool — but `v7`'s pool is at 33.2% NOP, so that explanation
    does NOT fit here and the cause is open. Count it before acting on it.
    """
    cand = _trigrams(text)
    for old in dropped:
        prev = _trigrams(old)
        if len(cand & prev) / max(len(cand | prev), 1) >= CHURN_SIMILARITY:
            return old
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

    @property
    def churn_points(self) -> list[AppliedOp]:
        """Applied `Add`s that merely restate a point dropped in the same step. Exposed
        for measurement; see `restates_dropped` for why nothing is refused."""
        return [r for r in self.results if r.note and "restates dropped" in r.note]

    @property
    def hedge_points(self) -> list[AppliedOp]:
        """Applied `Add`s whose text carries an unresolved polarity marker (measured
        2026-08-30 to be mishandled by synthesis — see `hedge_marker_in`). Exposed so a
        caller can measure how often this fires before deciding whether to act on it."""
        return [r for r in self.results if r.note and "unresolved polarity" in r.note]


def apply_ops(
    memory: Memory,
    ops: list[Op],
    chunk: Chunk,
    *,
    consecutive_nops: int = 0,
    lang_check: bool = True,
    budget: int = CHUNK_TOKENS,
) -> Outcome:
    """Validate and apply a step's ops IN PLACE, in emission order. Returns per-op
    verdicts.

    `consecutive_nops` is the running count BEFORE this step; the caller keeps the
    tally and passes it in, so this function stays stateless. `lang_check=False` exists
    only for testing the other guards in isolation from `arcsum.lang`. `budget` MUST be
    the actual chunking budget the caller used (not left at the default) — it feeds
    `chunk.is_content_rich()`'s threshold, and a caller running at a non-default budget
    that forgets to pass it here would silently measure richness against the wrong
    denominator, which is exactly the class of divergence this design exists to avoid.
    """
    outcome = Outcome()
    substantive = False
    #: Texts removed by a `DROP` earlier in THIS step's emission order, so a later `ADD`
    #: can be checked against them. Emission order is already load-bearing here (the
    #: contradiction guard relies on DROP-then-ADD succeeding), so this rides on it.
    dropped_here: list[str] = []

    for op in ops:
        match op:
            case Nop():
                outcome.results.append(AppliedOp(op, True))

            case Malformed(_, reason):
                outcome.results.append(AppliedOp(op, False, reason))

            case Arc(text):
                if lang_check and (bad := check_zh_tw(text, min_cjk_ratio=MIN_CJK_RATIO_ARC)):
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
                applied = reason is None
                # Recorded, not refused: a hedge-phrased point may still be the best
                # available capture of a genuine open question, and refusing it outright
                # is unvalidated — see `hedge_marker_in`'s docstring.
                hedge = hedge_marker_in(point) if applied else None
                churn = restates_dropped(point, dropped_here) if applied else None
                notes = []
                if hedge:
                    notes.append(f"unresolved polarity ({hedge})")
                if churn:
                    notes.append(f"restates dropped «{churn}»")
                note = "; ".join(notes) or None
                outcome.results.append(AppliedOp(op, applied, reason, note))
                substantive = substantive or applied

            case Revise(pid, text):
                # SPEC §4.1 v1.1. Deliberately NOT fed into `dropped_here`: that list
                # exists to catch DROP + near-identical re-ADD churn, and a `revise` IS
                # the sanctioned way to replace a point. Counting it as churn would make
                # the metric fire on exactly the behaviour the op was added to enable.
                if lang_check and (bad := check_zh_tw(text, min_cjk_ratio=MIN_CJK_RATIO_POINT)):
                    outcome.results.append(AppliedOp(op, False, bad))
                    continue
                reason = memory.revise_id(pid, text)
                applied = reason is None
                hedge = hedge_marker_in(text) if applied else None
                note = f"unresolved polarity ({hedge})" if hedge else None
                outcome.results.append(AppliedOp(op, applied, reason, note))
                substantive = substantive or applied

            case Drop(prefix, pid):
                if pid:
                    idx = memory._index_of(pid)
                    removed = memory.points[idx].text if idx is not None else None
                    reason = memory.drop_id(pid)
                else:
                    idx = memory.find(prefix)
                    removed = memory.points[idx].text if idx is not None else None
                    reason = memory.drop_point(prefix)
                if reason is None and removed is not None:
                    dropped_here.append(removed)
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
    if not substantive and chunk.is_content_rich(budget=budget):
        outcome.nop_collapse = consecutive_nops + 1 >= NOP_COLLAPSE_K

    return outcome
