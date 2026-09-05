"""Step grammar: ADD / DROP / ARC / NOP (SPEC §4.1).

    ADD - <point>
    DROP «<prefix>»
    ARC: <text>
    NOP

Deliberately small. No multi-point rewrite op: SPEC §4.1 explicitly excludes it — "the
prior project measured that as the heaviest op in its grammar and never validated it at
<=1B". That absence is what keeps `parse_ops` genuinely line-local: there is no
multi-line accumulation state to carry between lines, unlike the prior project's `CMP`.

**Parsing never raises.** A malformed line becomes a `Malformed` record and is logged,
never fatal — nothing the model emits can crash the harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: «...» is the spec's own delimiter; ASCII `<<...>>`, corner brackets `「...」`, and
#: plain quotes are all tolerated because a small model will emit whichever it has seen
#: most in pretraining. `.*?` (not `.+?`): an empty `«»` must still MATCH, so it can be
#: refused as "empty prefix" rather than falling through to the generic "does not match
#: the grammar" — the two are different diagnoses and the trace should say which.
_PREFIX = r'(?:«(?P<p1>.*?)»|<<(?P<p2>.*?)>>|「(?P<p3>.*?)」|"(?P<p4>.*?)")'

_NOP_RE = re.compile(r"^NOP[\s。．.]*$", re.IGNORECASE)
_ARC_RE = re.compile(r"^ARC\s*[:：]\s*(?P<text>.*)$", re.IGNORECASE)
_DROP_RE = re.compile(rf"^DROP\s*{_PREFIX}\s*$", re.IGNORECASE)
#: `(?P<sep>[-–—])?` is its own group rather than folded into the point: for `ADD -`
#: (a dash with nothing after it), a combined optional-then-greedy pattern lets the
#: dash itself backtrack into being "the point" instead of a separator with an empty
#: point behind it. Splitting them out makes that case unambiguous.
_ADD_RE = re.compile(r"^ADD\s*(?:(?P<sep>[-–—])\s*)?(?P<point>.*)$", re.IGNORECASE)

#: A hallucinated `[m:ss]`-style anchor, stripped rather than admitted into memory. v2 has
#: no timestamps (SPEC §2), but a 1B model may still emit one from pretraining exposure to
#: anchored formats — silently peeling it beats letting it corrupt the rendered point.
_JUNK_ANCHOR = re.compile(r"\s*[\[［]\s*\d+\s*[:：]\s*\d{2}(?:[:：]\d{2})?\s*[\]］]\s*$")


@dataclass(frozen=True, slots=True)
class Add:
    point: str


@dataclass(frozen=True, slots=True)
class Drop:
    """Retire a point from the working set.

    `prefix` addresses by text (v0.9 / v1.0); `pid` addresses by id (SPEC §4.1 v1.1).
    Exactly one is set. Both forms are kept because the edit protocol and the existing
    supervision pool address by text, and dropping that support would invalidate every
    stored gold trace at once.
    """

    prefix: str = ""
    pid: int = 0


@dataclass(frozen=True, slots=True)
class Revise:
    """Atomically supersede point `pid` with `text` (SPEC §4.1 v1.1).

    Exists because DROP-then-ADD could not be distinguished from churn: v1.0 had no
    single act meaning "this is now wrong, here is the correction", so
    `guards.restates_dropped` fires on genuine revision and on a model rewriting what it
    already had, identically. Measured churn under the two-op form: 28.2% of steps.
    """

    pid: int
    text: str


@dataclass(frozen=True, slots=True)
class Arc:
    text: str


@dataclass(frozen=True, slots=True)
class Nop:
    pass


@dataclass(frozen=True, slots=True)
class Malformed:
    raw: str
    reason: str


Op = Add | Drop | Revise | Arc | Nop | Malformed


def _strip_junk_anchor(text: str) -> str:
    return _JUNK_ANCHOR.sub("", text).strip()


def _prefix_group(m: re.Match[str]) -> str:
    return next(
        g for g in (m.group("p1"), m.group("p2"), m.group("p3"), m.group("p4")) if g is not None
    )


def parse_ops(text: str) -> list[Op]:
    """Parse a step's raw model output into ops. NEVER RAISES.

    Line-local: each line is matched independently, in emission order, against
    NOP -> ARC -> DROP -> ADD -> Malformed (checked in that order; the first match wins).
    """
    if not text:
        return []

    ops: list[Op] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        ops.append(_parse_line(line))
    return ops


def _parse_line(line: str) -> Op:
    if _NOP_RE.match(line):
        return Nop()

    if m := _ARC_RE.match(line):
        text = _strip_junk_anchor(m.group("text").strip())
        if not text:
            return Malformed(line, "empty arc")
        return Arc(text)

    if m := _DROP_RE.match(line):
        prefix = _prefix_group(m).strip()
        if not prefix:
            return Malformed(line, "empty prefix")
        return Drop(prefix)

    if m := _ADD_RE.match(line):
        point = _strip_junk_anchor(m.group("point").strip())
        if not point:
            return Malformed(line, "empty point")
        return Add(point)

    return Malformed(line, "does not match the op grammar")


def render_op(op: Op) -> str:
    """Round-trip an op back to the text grammar. Used for logs and as the SFT target."""
    match op:
        case Add(point):
            return f"ADD - {point}"
        # v1.1 addresses points by integer id; the text-prefix form is v1.0's and still
        # appears in stored traces, so both render.
        case Drop(prefix, pid) if pid:
            return f"DROP #{pid}"
        case Drop(prefix, _):
            return f"DROP «{prefix}»"
        case Revise(pid, text):
            return f"REVISE #{pid} - {text}"
        case Arc(text):
            return f"ARC: {text}"
        case Nop():
            return "NOP"
        case Malformed(raw, _reason):
            return raw
    # NOT unreachable, and it has fired: `Revise` landed with SPEC §4.1 v1.1 and this
    # function was not updated, so `score_reversals.py` crashed with "unhandled op type"
    # the first time a checkpoint actually emitted one — on the G1 REVISION probe, i.e. the
    # exact gate the op exists to serve. Same shape as CLAUDE.md trap 13: a protocol change
    # landed and a consumer was never told.
    raise TypeError(f"unhandled op type: {op!r}")
