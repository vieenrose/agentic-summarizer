"""zh-TW output-language enforcement — NET-NEW, does not exist in the prior project.

Justified by measurement, not caution: the prior project shipped an "enforce-lang" guard
after measuring **23.2% English** leakage on real zh ASR input (SPEC §8 risk 5). This
design compounds that risk — a 1B model with an English-heavy pretraining prior, reading
transcript text that was itself machine-translated from English, producing output that
must land as zh-TW. SPEC §4.3 already gates the *input* side against zh-CN vocabulary
leaking in via translation; nothing gated the *output* side. This module is that gate,
consumed at three points with three different thresholds (§4.3's translation gate,
`guards.apply_ops`'s per-point check, and `agent.synthesize_memory`'s prose check).

Two cheap, deterministic, dependency-free checks: a CJK-density floor (catches English
leakage) and a curated simplified-character detector (catches zh-CN leakage). Both
**deliberately err toward not firing** — a false positive here silently drops a true
decision, which is worse than an occasional miss.
"""

from __future__ import annotations

from arcsum.tokens import is_ideograph, normalise

#: Prose output (SPEC §3) must read as connected zh-TW narrative.
MIN_CJK_RATIO_PROSE = 0.70

#: A single point may legitimately be mostly a proper noun, an ordinance ID, or an
#: acronym ("CB 118618"), so the floor is looser than prose's.
MIN_CJK_RATIO_POINT = 0.35


def cjk_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are CJK ideographs.

    `1.0` for empty or whitespace-only text — an empty string has no evidence of
    English leakage, so it should not fail a CJK-ratio check (a separate empty-content
    check is the caller's job).
    """
    chars = [ch for ch in normalise(text) if not ch.isspace()]
    if not chars:
        return 1.0
    return sum(1 for ch in chars if is_ideograph(ch)) / len(chars)


#: Simplified-only codepoints with NO legitimate Traditional usage, seeded from the
#: recurring municipal-procedure vocabulary SPEC §4.3 names (ordinance/motion/council/
#: committee terms translate through 議決/議員/委員會/報告/電腦-adjacent characters).
#: Deliberately small and explicit, following `guards`'s eventual `_NEGATIVE`/`_POSITIVE`
#: doctrine: a wrong "simplified" verdict silently drops a true decision, so this errs
#: toward NOT firing. Ambiguous characters that are valid in BOTH scripts (e.g. `于`,
#: `后`, `划`, which each have legitimate Traditional readings) are deliberately excluded.
_SIMPLIFIED_ONLY = frozenset("议决员会计书报关电脑处号医团国学长讨论")


def simplified_hits(text: str) -> set[str]:
    """Simplified-only characters found in `text`. Empty set means no evidence of zh-CN."""
    return set(text) & _SIMPLIFIED_ONLY


def check_zh_tw(text: str, *, min_cjk_ratio: float) -> str | None:
    """`None` if `text` is acceptable zh-TW; otherwise a refusal reason string.

    Composes with the applier's other refusal reasons ("empty point", "duplicate point")
    for free — callers just need to check for `None`.
    """
    ratio = cjk_ratio(text)
    if ratio < min_cjk_ratio:
        return f"insufficient zh-TW content ({ratio:.2f} < {min_cjk_ratio} CJK ratio)"
    if hits := simplified_hits(text):
        sample = "".join(sorted(hits)[:5])
        return f"simplified characters detected ({sample})"
    return None
