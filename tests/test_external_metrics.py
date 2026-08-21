"""Pins SPEC §5's optional heavy-dependency metrics: each raises a clear,
extra-naming error when its dependency is absent, rather than a bare
`ModuleNotFoundError` — and each stays fully skippable so the core suite runs with
no optional extra installed.
"""

from __future__ import annotations

import importlib.util

import pytest

from arcsum.metrics.external import MissingExtraError, bert_score, mover_score, sacre_bleu

_HAS_SACREBLEU = importlib.util.find_spec("sacrebleu") is not None
_HAS_BERT_SCORE = importlib.util.find_spec("bert_score") is not None
_HAS_MOVERSCORE = importlib.util.find_spec("moverscore") is not None


def test_missing_extra_error_is_an_import_error() -> None:
    """A caller doing `except ImportError` must still catch this."""
    assert issubclass(MissingExtraError, ImportError)


@pytest.mark.skipif(_HAS_SACREBLEU, reason="sacrebleu is installed; this is the missing-extra path")
def test_sacre_bleu_raises_a_clear_missing_extra_error() -> None:
    with pytest.raises(MissingExtraError, match=r"\[metrics\]"):
        sacre_bleu("candidate", ["reference"])


@pytest.mark.skipif(
    _HAS_BERT_SCORE, reason="bert_score is installed; this is the missing-extra path"
)
def test_bert_score_raises_a_clear_missing_extra_error() -> None:
    with pytest.raises(MissingExtraError, match=r"\[metrics-neural\]"):
        bert_score(["candidate"], ["reference"])


@pytest.mark.skipif(
    _HAS_MOVERSCORE, reason="moverscore is installed; this is the missing-extra path"
)
def test_mover_score_raises_a_clear_missing_extra_error() -> None:
    with pytest.raises(MissingExtraError, match=r"\[metrics-mover\]"):
        mover_score(["candidate"], ["reference"])


@pytest.mark.metrics
@pytest.mark.skipif(not _HAS_SACREBLEU, reason="needs the 'metrics' extra")
def test_sacre_bleu_scores_an_identical_pair_near_100() -> None:
    result = sacre_bleu("你好世界", ["你好世界"])
    assert result.score > 90.0


@pytest.mark.metrics
@pytest.mark.skipif(not _HAS_BERT_SCORE, reason="needs the 'metrics-neural' extra")
def test_bert_score_of_an_identical_pair_is_near_one() -> None:
    result = bert_score(["你好世界"], ["你好世界"])
    assert result.f1 > 0.95


@pytest.mark.metrics
@pytest.mark.skipif(not _HAS_MOVERSCORE, reason="needs the 'metrics-mover' extra")
def test_mover_score_runs_without_raising() -> None:
    result = mover_score(["candidate sentence"], ["reference sentence"])
    assert isinstance(result.score, float)
