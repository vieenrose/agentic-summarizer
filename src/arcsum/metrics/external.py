"""Optional reference metrics needing a heavy third-party dependency (SPEC §5):
SacreBLEU (`tokenize=zh`), BERTScore, and MoverScore.

Kept out of the core package on purpose — none of these are importable without
installing the matching extra (`arcsum-agentic[metrics]` / `[metrics-neural]` /
`[metrics-mover]`), and calling one without its extra installed raises a clear error
naming exactly which extra to install, rather than a bare `ModuleNotFoundError` three
frames deep inside someone else's library.

**MoverScore is its own extra, isolated from the rest.** SPEC §5 keeps it anyway
("monolingual-English only by default, but the implementation accepts a multilingual
BERT, so it survives an encoder swap") despite it being effectively unmaintained and
pinning old `pyemd`/BERT versions. Isolating it means an unrelated BLEU/BERTScore run
is never blocked by MoverScore's fragile dependency chain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class MissingExtraError(ImportError):
    """Raised in place of a bare `ModuleNotFoundError`, naming the extra to install.
    Subclasses `ImportError` so an `except ImportError` caller still catches it."""


@dataclass(frozen=True, slots=True)
class BleuScore:
    #: SacreBLEU's native 0-100 scale, not normalised to 0-1.
    score: float


def sacre_bleu(candidate: str, references: Sequence[str]) -> BleuScore:
    """SacreBLEU with `tokenize="zh"` (SPEC §5). Needs the `metrics` extra."""
    try:
        import sacrebleu
    except ImportError as exc:
        raise MissingExtraError(
            "sacre_bleu needs the 'metrics' extra: pip install 'arcsum-agentic[metrics]'"
        ) from exc
    result = sacrebleu.sentence_bleu(candidate, list(references), tokenize="zh")
    return BleuScore(score=result.score)


@dataclass(frozen=True, slots=True)
class BertScoreResult:
    precision: float
    recall: float
    f1: float


def bert_score(
    candidates: Sequence[str], references: Sequence[str], *, lang: str = "zh"
) -> BertScoreResult:
    """BERTScore with a Chinese/multilingual encoder (SPEC §5). Needs the
    `metrics-neural` extra. Scores are averaged across the batch; call with one
    (candidate, reference) pair at a time for a per-meeting score."""
    try:
        import bert_score as _bert_score
    except ImportError as exc:
        raise MissingExtraError(
            "bert_score needs the 'metrics-neural' extra: "
            "pip install 'arcsum-agentic[metrics-neural]'"
        ) from exc
    precision, recall, f1 = _bert_score.score(list(candidates), list(references), lang=lang)
    return BertScoreResult(
        precision=float(precision.mean()), recall=float(recall.mean()), f1=float(f1.mean())
    )


@dataclass(frozen=True, slots=True)
class MoverScoreResult:
    score: float


def mover_score(candidates: Sequence[str], references: Sequence[str]) -> MoverScoreResult:
    """MoverScore. Needs the `metrics-mover` extra, isolated from the others (see
    module docstring)."""
    try:
        from moverscore import get_idf_dict, word_mover_score
    except ImportError as exc:
        raise MissingExtraError(
            "mover_score needs the 'metrics-mover' extra: "
            "pip install 'arcsum-agentic[metrics-mover]'"
        ) from exc
    idf_refs = get_idf_dict(list(references))
    idf_hyps = get_idf_dict(list(candidates))
    scores = word_mover_score(
        list(references),
        list(candidates),
        idf_refs,
        idf_hyps,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
    )
    return MoverScoreResult(score=sum(scores) / len(scores) if scores else 0.0)
