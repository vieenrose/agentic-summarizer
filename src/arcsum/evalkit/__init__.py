"""Evaluation infrastructure: what configuration produced a number, and what the number
cannot tell you.

The instruments themselves live in `arcsum.metrics` (reference-based) and here
(reference-free). This package exists because the failures that cost this project the most
were not bad metrics — they were good metrics measured under an unrecorded or wrong
configuration, and comparisons between numbers that were never comparable.
"""

from arcsum.evalkit.behaviour import BehaviourReport, BehaviourSummary
from arcsum.evalkit.grounding import GroundingReport, GroundingSummary
from arcsum.evalkit.provenance import CorpusFingerprint, Provenance, capture
from arcsum.evalkit.scorecard import (
    Check,
    IncomparableScorecards,
    Scorecard,
    assert_comparable,
    compare,
)

__all__ = [
    "BehaviourReport", "BehaviourSummary",
    "Check", "CorpusFingerprint", "GroundingReport", "GroundingSummary",
    "IncomparableScorecards", "Provenance", "Scorecard",
    "assert_comparable", "capture", "compare",
]
