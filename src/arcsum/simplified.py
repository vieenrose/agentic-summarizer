"""Run the agent's INTERNAL representation in Simplified Chinese, emit Traditional.

SPEC §3 fixes the product's output language as zh-TW. This module does not change that: it
changes the script the pipeline THINKS in — the chunk text the model reads, the memory and
journal it writes, and the synthesis it produces — converting back to zh-TW at the boundary.

**Why, measured on this project's own corpora** (16 meetings, 354,995 characters):

| tokenizer | zh-TW | zh-CN | saving |
|---|---|---|---|
| Qwen3.5-0.8B (student, 248k vocab) | 1.577 ch/tok | 1.761 ch/tok | **10.5%** |
| Granite (100k vocab) | 0.727 ch/tok | 0.909 ch/tok | **20.0%** |

Chunking is token-based, so a 10.5% token reduction is ~10.5% fewer reading steps, and
wall-clock per meeting is steps x per-step latency. Against the measured G4 figure that is
19.0 min -> ~17 min under a 20.00 min ceiling whose current margin is 3%
(`runs/g4-device-measured.md`). The saving lands exactly where the budget is tight.

**What the round trip costs, also measured** (`tw2sp` out, `s2tw` back, aligned with
`difflib` rather than positionally): **0.288% of characters**, and most of those are variants
acceptable in zh-TW anyway — `畫/劃`, `裡/里`, `台/臺`, `週/周`.

**Use `s2tw`, NOT `s2twp`, on the way back.** The `p` variant applies a Taiwan PHRASE table
and performs vocabulary localisation, not script conversion: it rewrote `發布` to `釋出` and
`藉` to `借`. Those are real word substitutions in a summary that is supposed to report what
was said, so the phrase table is the wrong tool for the return trip. It is the right tool
going IN (`tw2sp`), where mapping Taiwan vocabulary onto mainland forms is what makes the
text tokenise well.

**The dependency is optional and the seam is injectable**, because the test suite running
with no GPU, no weights, no network and no optional extra is load-bearing for this project.
`converter()` returns identity when `opencc` is absent, so importing this module can never
break a bare install; callers inject a `Callable[[str], str]` exactly as they already inject
`token_len`. A test can therefore exercise the wiring with a fake conversion and no extra.

**This is a version-bumping change.** Every stored prompt, every trace and every reported
metric was produced in zh-TW; a pool converted with this module is a different corpus and
its numbers are not comparable to anything measured before it. Bump `PROMPT_VERSION` and
record `SCRIPT_VERSION` alongside it rather than silently swapping scripts.
"""

from __future__ import annotations

from collections.abc import Callable

#: Bump when the conversion configs change. Stored beside `PROMPT_VERSION` in any artifact
#: produced under Simplified internals, so a reader can never mistake one script for the
#: other when comparing two numbers.
SCRIPT_VERSION = "zhcn-v1"

#: Traditional -> Simplified, WITH the phrase table: mapping Taiwan vocabulary onto the
#: mainland forms the tokenizer was trained on is the point of the exercise.
TO_SIMPLIFIED = "tw2sp"

#: Simplified -> Traditional, WITHOUT the phrase table. See the module docstring: `s2twp`
#: substitutes words, not just characters, and a summary must not have its vocabulary
#: rewritten on the way out.
TO_TRADITIONAL = "s2tw"


def identity(text: str) -> str:
    return text


def converter(config: str) -> Callable[[str], str]:
    """An OpenCC conversion function, or `identity` when the extra is not installed.

    Degrading to identity rather than raising is deliberate: this module is imported by the
    harness, and a bare install must keep working. A caller that REQUIRES conversion should
    check `available()` and refuse loudly itself — silently emitting the wrong script into a
    training pool would be far worse than a missing import.
    """
    try:
        from opencc import OpenCC
    except ImportError:
        return identity
    cc = OpenCC(config)
    return cc.convert


def available() -> bool:
    """Whether real conversion is possible. Any TOOL that writes a pool or reports a metric
    must gate on this; the harness itself may run without it."""
    try:
        import opencc  # noqa: F401
    except ImportError:
        return False
    return True


def to_simplified(text: str) -> str:
    return converter(TO_SIMPLIFIED)(text)


def to_traditional(text: str) -> str:
    return converter(TO_TRADITIONAL)(text)
