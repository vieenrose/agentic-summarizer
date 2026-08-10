"""Edit ops — the agent's tool set (CLAUDE.md §5.1).

Two wire formats parse to the same `Op` objects:

  * **text grammar** (spec §5.1), used for the teacher and for readable logs:
        ADD DECISIONS - Budget increase approved at 10% [32:14]
        UPD SUMMARY «Budget increase» -> ... [32:14]
        DEL OPEN «Parking»
        CMP TOPICS
        - rewritten bullet [0:00]
        NOP

  * **FunctionGemma call format** (PLAN.md §0.2), the student's native post-trained shape:
        <start_function_call>call:ADD{section:<escape>DECISIONS<escape>,
        bullet:<escape>...<escape>,anchor:<escape>32:14<escape>}<end_function_call>

Parsing never raises: a malformed line becomes a `Malformed` record and is logged, never
fatal (CLAUDE.md §6.4). Validation against STATE and the chunk lives in `guards.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .state import CAPS, Bullet
from .transcript import clock_to_sec

__all__ = [
    "Add",
    "Cmp",
    "Del",
    "Malformed",
    "Nop",
    "Op",
    "Upd",
    "parse_ops",
    "render_op",
]

# «...» with the guillemets the spec uses, tolerating ASCII << >> and plain quotes.
_PREFIX = r"(?:«(?P<p1>.+?)»|<<(?P<p2>.+?)>>|\"(?P<p3>.+?)\")"
_SECTION = r"(?P<section>[A-Z]+)"
_ANCHOR_TAIL = re.compile(r"\s*\[(?P<anchor>[0-9:]+)\]\s*$")

_ADD_RE = re.compile(rf"^ADD\s+{_SECTION}\s*(?:-\s*)?(?P<bullet>.+)$", re.I)
_UPD_RE = re.compile(rf"^UPD\s+{_SECTION}\s*{_PREFIX}\s*->\s*(?P<bullet>.+)$", re.I)
_DEL_RE = re.compile(rf"^DEL\s+{_SECTION}\s*{_PREFIX}\s*$", re.I)
_CMP_RE = re.compile(rf"^CMP\s+{_SECTION}\s*$", re.I)
_NOP_RE = re.compile(r"^NOP\s*$", re.I)
_TITLE_RE = re.compile(r"^TITLE:\s*(?P<title>.+)$", re.I)
_BULLET_RE = re.compile(r"^-\s*(?P<bullet>.+)$")

# FunctionGemma: call:NAME{key:<escape>value<escape>,...}
_CALL_RE = re.compile(
    r"<start_function_call>\s*call:(?P<name>\w+)\s*\{(?P<args>.*?)\}\s*<end_function_call>",
    re.S,
)
_ARG_RE = re.compile(r"(?P<key>\w+)\s*:\s*(?:<escape>(?P<esc>.*?)<escape>|(?P<raw>[^,}]*))", re.S)


@dataclass(frozen=True, slots=True)
class Add:
    section: str
    bullet: str
    anchor: int | None


@dataclass(frozen=True, slots=True)
class Upd:
    section: str
    prefix: str
    bullet: str
    anchor: int | None


@dataclass(frozen=True, slots=True)
class Del:
    section: str
    prefix: str


@dataclass(frozen=True, slots=True)
class Cmp:
    section: str
    bullets: tuple[Bullet, ...]


@dataclass(frozen=True, slots=True)
class Title:
    title: str


@dataclass(frozen=True, slots=True)
class Nop:
    pass


@dataclass(frozen=True, slots=True)
class Malformed:
    raw: str
    reason: str


Op = Add | Upd | Del | Cmp | Title | Nop | Malformed


def _split_anchor(text: str) -> tuple[str, int | None]:
    """Peel a trailing `[m:ss]` off a bullet. Returns (bullet, seconds|None)."""
    m = _ANCHOR_TAIL.search(text)
    if not m:
        return text.strip(), None
    try:
        return text[: m.start()].strip(), clock_to_sec(m.group("anchor"))
    except ValueError:
        # A malformed clock (e.g. `[99:99]`) is no anchor at all, but it must still be
        # peeled off: leaving it in the bullet would put it in the rendered notes and
        # skew the deterministic matcher's lexical overlap.
        return text[: m.start()].strip(), None


def _prefix_of(m: re.Match[str]) -> str:
    return (m.group("p1") or m.group("p2") or m.group("p3") or "").strip()


def _known_section(name: str) -> str | None:
    name = name.upper()
    return name if name in CAPS else None


def _parse_call(m: re.Match[str]) -> Op:
    """FunctionGemma call -> Op."""
    args: dict[str, str] = {}
    for a in _ARG_RE.finditer(m.group("args")):
        value = a.group("esc") if a.group("esc") is not None else (a.group("raw") or "")
        args[a.group("key").lower()] = value.strip()

    name = m.group("name").upper()
    raw = m.group(0)
    if name == "NOP":
        return Nop()
    if name == "TITLE":
        title = args.get("title", "")
        return Title(title) if title else Malformed(raw, "TITLE without title")

    section = _known_section(args.get("section", ""))
    if section is None:
        return Malformed(raw, f"unknown section {args.get('section', '')!r}")

    anchor: int | None = None
    if args.get("anchor"):
        try:
            anchor = clock_to_sec(args["anchor"])
        except ValueError:
            return Malformed(raw, f"unparseable anchor {args['anchor']!r}")

    if name == "ADD":
        bullet = args.get("bullet", "")
        return Add(section, bullet, anchor) if bullet else Malformed(raw, "ADD without bullet")
    if name == "UPD":
        prefix, bullet = args.get("prefix", ""), args.get("bullet", "")
        if not prefix or not bullet:
            return Malformed(raw, "UPD needs prefix and bullet")
        return Upd(section, prefix, bullet, anchor)
    if name == "DEL":
        prefix = args.get("prefix", "")
        return Del(section, prefix) if prefix else Malformed(raw, "DEL without prefix")
    return Malformed(raw, f"unknown op {name!r}")


def parse_ops(text: str) -> list[Op]:
    """Parse a step's raw model output into ops. Never raises.

    Both wire formats may appear in one response; ops are returned in emission order.
    Bare bullet lines are consumed by a preceding CMP.
    """
    if not text:
        return []

    # Function calls first — their bodies may span lines and must not be line-parsed.
    calls = list(_CALL_RE.finditer(text))
    if calls:
        ops = [_parse_call(m) for m in calls]
        # Text-format ops outside the call markers still count.
        residue = _CALL_RE.sub("\n", text)
        return ops + [o for o in parse_ops(residue) if not isinstance(o, Malformed)]

    ops: list[Op] = []
    pending_cmp: str | None = None
    cmp_bullets: list[Bullet] = []

    def flush_cmp() -> None:
        nonlocal pending_cmp, cmp_bullets
        if pending_cmp is not None:
            ops.append(Cmp(pending_cmp, tuple(cmp_bullets)))
            pending_cmp, cmp_bullets = None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if pending_cmp is not None:
            b = _BULLET_RE.match(line)
            if b:
                bullet, anchor = _split_anchor(b.group("bullet"))
                cmp_bullets.append(Bullet(bullet, anchor))
                continue
            flush_cmp()

        if _NOP_RE.match(line):
            ops.append(Nop())
            continue

        if m := _TITLE_RE.match(line):
            ops.append(Title(m.group("title").strip()))
            continue

        if m := _CMP_RE.match(line):
            section = _known_section(m.group("section"))
            if section is None:
                ops.append(Malformed(line, f"unknown section {m.group('section')!r}"))
            else:
                pending_cmp = section
            continue

        for regex, kind in ((_UPD_RE, "UPD"), (_DEL_RE, "DEL"), (_ADD_RE, "ADD")):
            m = regex.match(line)
            if not m:
                continue
            section = _known_section(m.group("section"))
            if section is None:
                ops.append(Malformed(line, f"unknown section {m.group('section')!r}"))
                break
            if kind == "DEL":
                prefix = _prefix_of(m)
                ops.append(Del(section, prefix) if prefix else Malformed(line, "empty prefix"))
            elif kind == "UPD":
                prefix = _prefix_of(m)
                bullet, anchor = _split_anchor(m.group("bullet"))
                ops.append(
                    Upd(section, prefix, bullet, anchor)
                    if prefix and bullet
                    else Malformed(line, "UPD needs prefix and bullet")
                )
            else:
                bullet, anchor = _split_anchor(m.group("bullet"))
                ops.append(
                    Add(section, bullet, anchor) if bullet else Malformed(line, "empty bullet")
                )
            break
        else:
            ops.append(Malformed(line, "does not match the op grammar"))

    flush_cmp()
    return ops


def render_op(op: Op) -> str:
    """Render an op back to the text grammar — for logs and for SFT targets."""
    from .transcript import sec_to_clock

    def tail(anchor: int | None) -> str:
        return f" [{sec_to_clock(anchor)}]" if anchor is not None else ""

    match op:
        case Nop():
            return "NOP"
        case Title(title):
            return f"TITLE: {title}"
        case Add(section, bullet, anchor):
            return f"ADD {section} - {bullet}{tail(anchor)}"
        case Upd(section, prefix, bullet, anchor):
            return f"UPD {section} «{prefix}» -> {bullet}{tail(anchor)}"
        case Del(section, prefix):
            return f"DEL {section} «{prefix}»"
        case Cmp(section, bullets):
            return "\n".join([f"CMP {section}", *(b.render() for b in bullets)])
        case Malformed(raw, _):
            return raw
    raise TypeError(f"unrenderable op: {op!r}")
