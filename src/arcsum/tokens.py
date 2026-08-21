"""Tokenization — the single source of truth (SPEC §5).

SPEC §5 makes the character-level tokenization **normative**: "a later switch to
segmenter-based ROUGE would invalidate comparison with everything measured before it".
This module is therefore the only place that answers "is this character CJK", and
`TOKENIZE_VERSION` stamps every number derived from it.

The prior project (branch `pi-agent`/`master`) had **three** drifted answers to that
question, which is the bug this module exists to prevent recurring:

    chunker.heuristic_token_len   U+3000-U+9FFF  and  U+FF00-U+FFEF
    index.tokenise                U+3400-U+9FFF  and  U+F900-U+FAFF
    guards._tokens                U+4E00-U+9FFF

Three roles are kept deliberately separate, because the prior project blurred the last
two — `Chunk.is_content_rich` silently used the module-level heuristic even when the
caller had injected a real tokenizer, so the guard and the budget disagreed about how
big a chunk was:

    char_tokens()        NORMATIVE. Metrics only (ROUGE, coverage, density, prefix floor).
    heuristic_token_len() NON-NORMATIVE budget estimate. Must never produce a reported number.
    hf_token_len()       The real budget instrument (MiniCPM5's own tokenizer).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence

#: Bump when anything below changes the token stream. Every metrics record carries it,
#: and the golden fixtures assert equality with it, so a tokenizer change cannot land
#: without regenerating the goldens and re-labelling every previously reported number.
TOKENIZE_VERSION = "chartok-v1"

#: Ideographs only. CJK punctuation (U+3000-U+303F) and fullwidth forms (U+FF00-U+FFEF)
#: are handled by normalisation, NOT by the range test — conflating them is exactly what
#: made the prior project's three ranges disagree. NFKC folds fullwidth digits and latin
#: to halfwidth, after which they are correctly treated as latin.
IDEOGRAPH_RANGES: tuple[tuple[str, str], ...] = (
    ("㐀", "䶿"),  # CJK Unified Ideographs Extension A
    ("一", "鿿"),  # CJK Unified Ideographs (URO)
    ("豈", "﫿"),  # CJK Compatibility Ideographs
)

#: Kana are tokenized per character like ideographs. Not expected in a zh-TW corpus, but
#: a total tokenizer must not silently drop characters it did not anticipate.
KANA_RANGES: tuple[tuple[str, str], ...] = (("぀", "ヿ"),)


def is_ideograph(ch: str) -> bool:
    """True if `ch` is a CJK ideograph under `IDEOGRAPH_RANGES`."""
    return any(lo <= ch <= hi for lo, hi in IDEOGRAPH_RANGES)


def is_kana(ch: str) -> bool:
    """True if `ch` is Japanese kana under `KANA_RANGES`."""
    return any(lo <= ch <= hi for lo, hi in KANA_RANGES)


def is_cjk(ch: str) -> bool:
    """The single CJK predicate. Everything that needs one must call this."""
    return is_ideograph(ch) or is_kana(ch)


def normalise(text: str) -> str:
    """NFKC + whitespace collapse.

    NFKC is what makes a fullwidth ordinance number tokenise identically to a halfwidth
    one, so `ＣＢ　１１８６１８` and `CB 118618` produce the same tokens.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split())


def char_tokens(text: str) -> list[str]:
    """NORMATIVE tokenization (SPEC §5).

    One token per CJK character; embedded latin words and numbers split on whitespace
    and other non-alphanumerics. Punctuation is dropped, never emitted as a token.

    This is the tokenizer for ROUGE-1/2/L, coverage, density, and the `DROP` prefix
    floor. It is deliberately NOT a budget estimator — see `heuristic_token_len`.
    """
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            out.append("".join(run))
            run.clear()

    for ch in normalise(text):
        if is_cjk(ch):
            flush()
            out.append(ch)
        elif ch.isalnum():
            run.append(ch.casefold())
        else:
            # Whitespace and punctuation alike terminate a latin run and are dropped.
            flush()
    flush()
    return out


def bigrams(toks: Sequence[str]) -> list[str]:
    """Adjacent token pairs, joined by a space. Empty for fewer than two tokens."""
    return [f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1)]


def lexical_tokens(text: str) -> set[str]:
    """Similarity key: latin word unigrams plus ideograph bigrams.

    The unified replacement for BOTH `guards._tokens` and `index.tokenise` in the prior
    project. Character bigrams are the only sane lexical unit for zh — word-splitting
    Chinese yields one giant token and every overlap score collapses to 0 or 1.

    Used for dedup, the `DROP` prefix fallback, the contradiction guard, and judge
    evidence retrieval. NOT used for metrics — those use `char_tokens`.
    """
    toks = char_tokens(text)
    latin = {t for t in toks if not is_cjk(t[0])}
    cjk = [t for t in toks if is_cjk(t[0])]
    return latin | set(bigrams(cjk))


def heuristic_token_len(text: str) -> int:
    """NON-NORMATIVE budget estimate: ~4 chars/token latin, ~1 token per CJK char.

    Deliberately over-estimates, so a chunk packed against it fits the real tokenizer.
    Exists so the harness stays importable with no `transformers` and testable with no
    weights.

    **Must never produce a reported number.** Every figure in SPEC §7 is defined against
    the student's own tokenizer — use `hf_token_len` for anything that gets written down.
    """
    cjk = sum(1 for ch in text if is_cjk(ch))
    return cjk + (len(text) - cjk + 3) // 4


def hf_token_len(model_id: str = "openbmb/MiniCPM5-1B") -> Callable[[str], int]:
    """The real budget instrument: the student's own tokenizer.

    `transformers` is imported lazily so the core package keeps zero runtime
    dependencies; install the `tokenizer` extra to use this.
    """
    # Imported here, not at module scope: the core package must stay dependency-free.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)

    def _token_len(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    _token_len.arcsum_name = f"hf:{model_id}"  # type: ignore[attr-defined]
    return _token_len


def token_len_name(fn: Callable[[str], int]) -> str:
    """Identify a token counter for the record.

    The budget instrument is allowed to differ between CI and a real run — it just must
    never be silent, because a heuristic-measured cap at trace-generation time and a
    real-tokenizer cap at inference time is a train/deploy divergence.
    """
    name = getattr(fn, "arcsum_name", None)
    if name:
        return str(name)
    if fn is heuristic_token_len:
        return "heuristic"
    return getattr(fn, "__name__", "unknown")
