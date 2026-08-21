"""Reference-based quality metrics, written in-repo against the normative tokenizer
(SPEC §5): character-level ROUGE-1/2/L, extractive Coverage/Density, and length.

**Why in-repo rather than a dependency.** SPEC §5 makes ROUGE's tokenization normative
for CJK — "recorded here as normative — a later switch to segmenter-based ROUGE would
invalidate comparison with everything measured before it." The `rouge-score` package's
tokenizer is Latin-hardcoded; taking it as a dependency would mean either accepting its
tokenization (contradicting §5) or monkey-patching it (fragile, and now two sources of
truth for the same 40 lines of logic). Coverage/Density are computed here for the same
reason SPEC gives explicitly: they must "reuse the exact character-level tokenization"
ROUGE uses, and that tokenization lives in `arcsum.tokens`.

Every metric here operates on `arcsum.tokens.char_tokens` output — never on raw text —
so a candidate/reference pair is only ever compared through the one normative lens.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from arcsum.tokens import TOKENIZE_VERSION, char_tokens


@dataclass(frozen=True, slots=True)
class RougeScore:
    precision: float
    recall: float
    f1: float
    #: Stamped on every score record so a tokenizer change forces a re-label, per
    #: SPEC §5's normative-tokenization discipline.
    tokenize_version: str = TOKENIZE_VERSION


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate: str, reference: str, n: int) -> RougeScore:
    """Character-level ROUGE-N. Multiset (count-clipped) n-gram overlap — NOT set
    intersection, which would under-count a candidate that legitimately repeats an
    n-gram the reference also repeats."""
    cand_tokens = char_tokens(candidate)
    ref_tokens = char_tokens(reference)
    cand_grams = _ngrams(cand_tokens, n)
    ref_grams = _ngrams(ref_tokens, n)
    overlap = sum((cand_grams & ref_grams).values())

    cand_total = sum(cand_grams.values())
    ref_total = sum(ref_grams.values())
    precision = overlap / cand_total if cand_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return RougeScore(precision, recall, f1)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l(candidate: str, reference: str) -> RougeScore:
    """Character-level ROUGE-L: longest common subsequence, F1 at beta=1."""
    cand_tokens = char_tokens(candidate)
    ref_tokens = char_tokens(reference)
    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return RougeScore(precision, recall, f1)


def extractive_fragments(source: Sequence[str], summary: Sequence[str]) -> list[list[str]]:
    """The greedy fragment-matching algorithm behind Coverage/Density (Grusky et al.,
    "Newsroom"): the maximal set of shared token runs between `source` and `summary`,
    read off `summary` left to right. A fragment need not appear in `source` at the
    same relative position as the previous one — the source is rescanned from the
    start for every summary position, which is why summaries can score high Density
    even when their extractive spans are drawn out of source order.
    """
    fragments: list[list[str]] = []
    i = 0
    while i < len(summary):
        best_len = 0
        j = 0
        while j < len(source):
            if summary[i] == source[j]:
                ii, jj = i, j
                while ii < len(summary) and jj < len(source) and summary[ii] == source[jj]:
                    ii += 1
                    jj += 1
                run_len = ii - i
                if run_len > best_len:
                    best_len = run_len
            j += 1
        if best_len > 0:
            fragments.append(list(summary[i : i + best_len]))
            i += best_len
        else:
            i += 1
    return fragments


def coverage(source: str, summary: str) -> float:
    """Fraction of `summary` tokens that belong to some extractive fragment shared
    with `source`. `0.0` for an empty summary (nothing to cover)."""
    src = char_tokens(source)
    summ = char_tokens(summary)
    if not summ:
        return 0.0
    covered = sum(len(f) for f in extractive_fragments(src, summ))
    return covered / len(summ)


def density(source: str, summary: str) -> float:
    """Mean squared extractive-fragment length, normalised by summary length — NOT
    bounded to [0, 1]; a higher value means longer verbatim spans were copied.
    `0.0` for an empty summary."""
    src = char_tokens(source)
    summ = char_tokens(summary)
    if not summ:
        return 0.0
    return sum(len(f) ** 2 for f in extractive_fragments(src, summ)) / len(summ)


@dataclass(frozen=True, slots=True)
class Length:
    chars: int
    tokens: int


def length(text: str, *, token_len: Callable[[str], int]) -> Length:
    """Summary length in BOTH characters and the injected tokenizer's tokens (SPEC §5:
    "keep, in characters and in MiniCPM5 tokens")."""
    return Length(chars=len(text), tokens=token_len(text))
