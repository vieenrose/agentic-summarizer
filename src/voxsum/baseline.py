"""Map-reduce baseline, and the coverage fallback (CLAUDE.md §5.3, §6.3, §7.7).

This is the opponent every ship gate is measured against, so it is built to be **fair, not
weak**: same chunk size, same NOTES v2 vocabulary, same anchor validation, same guards,
same `token_len` instrument. A strawman baseline would make GT2/GT3 meaningless.

What it deliberately lacks is STATE. Each window is digested *independently* — that is the
defining property of map-reduce and precisely the thing CURSOR is claimed to beat:

    map     per window, in isolation:  SYS + CHUNK          -> bullets
    reduce  per over-cap section:      SYS + all bullets    -> <= cap bullets
    render  deterministic NOTES v2 from the surviving bullets

The same per-window summariser doubles as the **coverage fallback** (§6.3): when the agent
NOPs through K content-rich chunks, `summarise_window` fills the gap. It is never part of
the agent protocol — the agent cannot call it — it is the harness's backstop.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .agent import Usage
from .chunker import CHUNK_TOKENS, Chunk, heuristic_token_len, iter_chunks
from .guards import match_anchor
from .prompts import (
    PROMPT_VERSION,
    build_window_prompt,
    reduce_prompt,
    window_system_prompt,
)
from .state import CAPS, Bullet, NotesState, normalize, spread
from .transcript import Utterance, clock_to_sec

__all__ = ["BaselineResult", "run_map_reduce", "summarise_window"]

ModelFn = Callable[[str, str], str]

# "SECTION - bullet [m:ss]" from the map step, tolerating a leading "- ".
_MAP_LINE = re.compile(
    r"^-?\s*(?P<section>SUMMARY|DECISIONS|ACTIONS|OPEN|TOPICS)\s*[-:]\s*(?P<rest>.+)$", re.I
)
_REDUCE_LINE = re.compile(r"^-\s*(?P<rest>.+)$")
_ANCHOR_TAIL = re.compile(r"\s*\[(?P<anchor>[0-9:]+)\]\s*$")
_NONE = re.compile(r"^\s*(NONE|NOP|無|沒有)\s*$", re.I)


@dataclass
class BaselineResult:
    """Final notes plus the accounting GT4 needs."""

    state: NotesState
    usage: Usage = field(default_factory=Usage)
    prompt_version: str = PROMPT_VERSION
    windows: int = 0
    reduce_calls: int = 0
    dropped: int = 0

    @property
    def valid_bullet_rate(self) -> float | None:
        """Share of emitted bullet lines that parsed and validated."""
        total = self._emitted
        return (total - self.dropped) / total if total else None

    _emitted: int = 0


def _split_anchor(text: str, chunk: Chunk | None) -> tuple[str, int | None]:
    """Peel `[m:ss]`, validating it against the window it came from (§6.1)."""
    m = _ANCHOR_TAIL.search(text)
    body = text[: m.start()].strip() if m else text.strip()
    if not m:
        return body, None
    try:
        anchor = clock_to_sec(m.group("anchor"))
    except ValueError:
        return body, None
    if chunk is not None and not chunk.has_line(anchor):
        return body, None
    return body, anchor


def summarise_window(
    chunk: Chunk, model: ModelFn, *, lang: str = "en", usage: Usage | None = None,
    token_len: Callable[[str], int] = heuristic_token_len,
) -> dict[str, list[Bullet]]:
    """Digest one window in isolation. Also the coverage fallback (§6.3).

    Anchors are validated against this window and fall back to the deterministic matcher,
    exactly as in the agent path — the baseline is held to the same anchor standard.
    """
    system, user = window_system_prompt(lang), build_window_prompt(chunk)
    raw = model(system, user)
    if usage is not None:
        usage.record(token_len(system) + token_len(user), token_len(raw))

    out: dict[str, list[Bullet]] = {s: [] for s in CAPS}
    for line in raw.splitlines():
        line = line.strip()
        if not line or _NONE.match(line):
            continue
        m = _MAP_LINE.match(line)
        if not m:
            continue
        text, anchor = _split_anchor(m.group("rest"), chunk)
        if not text:
            continue
        out[m.group("section").upper()].append(
            Bullet(text, anchor if anchor is not None else match_anchor(chunk, text))
        )
    return out


def _reduce_section(
    section: str,
    bullets: list[Bullet],
    model: ModelFn,
    *,
    lang: str,
    usage: Usage,
    token_len: Callable[[str], int],
) -> list[Bullet]:
    """Model-driven shrink of one over-cap section, with a deterministic safety net."""
    cap = CAPS[section]
    rendered = [b.render() for b in bullets]
    system, user = reduce_prompt(lang, section, cap, rendered)
    raw = model(system, user)
    usage.record(token_len(system) + token_len(user), token_len(raw))

    # Anchors must still resolve to a line the meeting actually contains. The reduce step
    # sees no chunk, so validate against the anchors it was given rather than trusting it.
    allowed = {b.anchor for b in bullets if b.anchor is not None}
    kept: list[Bullet] = []
    for line in raw.splitlines():
        m = _REDUCE_LINE.match(line.strip())
        if not m:
            continue
        text, anchor = _split_anchor(m.group("rest"), None)
        if not text or anchor not in allowed:
            continue
        if any(normalize(k.text) == normalize(text) for k in kept):
            continue
        kept.append(Bullet(text, anchor))

    # If the reduce step returned nothing usable, fall back to spread() rather than
    # emitting an empty section — a failed shrink must not delete the meeting's decisions.
    return kept[:cap] if kept else spread(bullets, cap)


def run_map_reduce(
    utterances: list[Utterance],
    model: ModelFn,
    *,
    lang: str = "en",
    budget: int = CHUNK_TOKENS,
    token_len: Callable[[str], int] = heuristic_token_len,
) -> BaselineResult:
    """Classic map-reduce over the same windows the agent would see."""
    result = BaselineResult(state=NotesState())
    collected: dict[str, list[Bullet]] = {s: [] for s in CAPS}

    for chunk in iter_chunks(utterances, budget=budget, token_len=token_len):
        result.windows += 1
        digest = summarise_window(
            chunk, model, lang=lang, usage=result.usage, token_len=token_len
        )
        for section, bullets in digest.items():
            collected[section].extend(bullets)

    # Merge: dedup across windows. Overlapping windows mean the same decision is digested
    # twice, and without this the caps fill with duplicates of the loudest moment.
    for section, bullets in collected.items():
        seen: set[str] = set()
        merged: list[Bullet] = []
        for b in sorted(bullets, key=lambda x: (x.anchor if x.anchor is not None else 0)):
            key = normalize(b.text)
            if key in seen:
                continue
            seen.add(key)
            merged.append(b)
        collected[section] = merged

    result._emitted = sum(len(b) for b in collected.values())

    # Shrink: one model call per over-cap section.
    for section, bullets in collected.items():
        if len(bullets) > CAPS[section]:
            result.reduce_calls += 1
            collected[section] = _reduce_section(
                section, bullets, model, lang=lang, usage=result.usage, token_len=token_len
            )

    for section, bullets in collected.items():
        for b in bullets:
            if result.state.add(section, b.text, b.anchor) is not None:
                result.dropped += 1
    result.state.enforce_caps()

    # TITLE has no map-step equivalent: the baseline derives it from its own top topic,
    # deterministically, rather than spending a call the agent does not spend.
    topics = result.state.bullets("TOPICS")
    if topics:
        result.state.set_title(" ".join(topics[0].text.split()[:8]))
    return result
