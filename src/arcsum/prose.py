"""The §3 output contract — ONE enforcement point for BOTH arms.

SPEC §3: "A single flowing zh-TW prose summary — no bullets, no sections, no anchors.
< 1,000 tokens." `finalize()` is called by both `agent.synthesize_memory` and
`baseline`'s reduce step, so the two arms cannot disagree about what a valid output is.

This is a net-new invariant with no prior-project analogue: the prior project's
`render_for_prompt == render_state` identity guaranteed the model only ever learned one
shape, but that identity dissolves here — the product is prose from `SYNTHESIZE`, while
the memory render (`arcsum.render`) is purely internal. `prose.finalize` is what replaces
it: the memory render and the product output are now two different shapes, and this
module is the single definition of what the product shape must be.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from arcsum.lang import MIN_CJK_RATIO_PROSE, cjk_ratio, simplified_hits

#: SPEC §3.
PROSE_MAX_TOKENS = 1000

#: Leading list markers a model might emit despite the "no bullets" instruction.
_BULLET_LINE = re.compile(r"^\s*(?:[-*•▪]|\d+[.)、])\s*")
#: Markdown headings ("## SUMMARY").
_HEADING_LINE = re.compile(r"^\s*#{1,6}\s*")
#: A hallucinated harness-format label leaking into the prose ("TITLE: ", "ARC: ", ...).
_LABEL_LINE = re.compile(
    r"^\s*(?:TITLE|SUMMARY|ARC|POINTS|DECISIONS|ACTIONS|OPEN|TOPICS)\s*[:：]\s*", re.IGNORECASE
)
#: A hallucinated `[m:ss]`-style anchor — v2 has no timestamps (SPEC §2); this can occur
#: anywhere in the text, not just trailing, so it is not anchored to the line end.
_JUNK_ANCHOR = re.compile(r"\s*[\[［]\s*\d+\s*[:：]\s*\d{2}(?:[:：]\d{2})?\s*[\]］]\s*")
#: Markdown emphasis/code markers, stripped rather than preserved — the product is prose.
_MD_EMPHASIS = re.compile(r"[*_`]{1,3}")


@dataclass(frozen=True, slots=True)
class Prose:
    text: str
    tokens: int
    chars: int
    over_budget: bool
    #: Zero or more independent language-guard failures. Empty means clean.
    lang_flags: tuple[str, ...]
    #: Whether the RAW input (before cleanup) showed bullets/headings/labels — a
    #: diagnostic signal that the model is drifting toward the old bulleted format.
    had_markup: bool


def finalize(raw: str, *, token_len: Callable[[str], int]) -> Prose:
    """Strip bullets/markdown/headings/hallucinated labels and anchors, collapse to one
    flowing block, measure, and language-check. Never raises."""
    had_markup = bool(
        _BULLET_LINE.search(raw) or _HEADING_LINE.search(raw) or _LABEL_LINE.search(raw)
    )

    lines = []
    for line in raw.splitlines():
        line = _LABEL_LINE.sub("", line)
        line = _HEADING_LINE.sub("", line)
        line = _BULLET_LINE.sub("", line)
        lines.append(line)

    text = " ".join(lines)
    text = _JUNK_ANCHOR.sub(" ", text)
    text = _MD_EMPHASIS.sub("", text)
    text = " ".join(text.split())

    tokens = token_len(text)
    flags: list[str] = []
    ratio = cjk_ratio(text)
    if ratio < MIN_CJK_RATIO_PROSE:
        flags.append(f"insufficient zh-TW content ({ratio:.2f} < {MIN_CJK_RATIO_PROSE} CJK ratio)")
    if hits := simplified_hits(text):
        sample = "".join(sorted(hits)[:5])
        flags.append(f"simplified characters detected ({sample})")

    return Prose(
        text=text,
        tokens=tokens,
        chars=len(text),
        over_budget=tokens > PROSE_MAX_TOKENS,
        lang_flags=tuple(flags),
        had_markup=had_markup,
    )
