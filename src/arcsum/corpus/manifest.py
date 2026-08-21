"""Per-meeting provenance and split assignment (SPEC §2.2, §4.2).

**Provenance matters more here than in the prior project.** That project's
`MeetingRecord.manifest()` tracked `authentic_clock`/`authentic_speakers` — whether a
timestamp or speaker label was real ASR output or invented. Here the reference summary
itself is machine-translated *and* machine-composed (SPEC §2.2: "human-selected ->
machine-translated -> machine-composed"), so provenance is the whole audit trail for
why a training target should be trusted at all.

**Splits must be carved before trace generation, not after.** `pi-agent`'s
`carve_eval_sets.py`/`filter_train_traces.py` exist only because generation ran before
the carve on that project: *"eval meetings have traces too (the generation ran before
the carve)"* — a leak discovered the hard way. `carve_splits` is deterministic and
meant to run first, once, before any supervision is built.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MeetingRecord:
    """One meeting's provenance. `translated_by`/`composed_by` name the model that
    produced that stage's output (SPEC §2.2 stages 2-3); `None` means that stage has
    not run yet for this meeting. `human_validated` is SPEC §4's non-optional gate —
    a meeting's composed summary must not enter the corpus until this is `True`.
    """

    meeting_id: str
    split: str = "train"
    translated_by: str | None = None
    composed_by: str | None = None
    human_validated: bool = False

    def manifest(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "split": self.split,
            "translated_by": self.translated_by,
            "composed_by": self.composed_by,
            "human_validated": self.human_validated,
        }

    def ready_for_corpus(self) -> bool:
        """True once every stage SPEC §2.2/§4 requires has actually run.

        A meeting failing this check must not be used for training OR eval — it is
        not "not yet validated as good enough," it is "not a corpus member yet."
        """
        return (
            self.translated_by is not None and self.composed_by is not None and self.human_validated
        )


#: SPEC §9 Phase 1's eval slice size, held fixed regardless of corpus scale changes.
DEFAULT_EVAL_N = 40


def carve_splits(
    meeting_ids: Sequence[str], *, eval_n: int = DEFAULT_EVAL_N, seed: int = 0
) -> dict[str, str]:
    """Deterministically assign each meeting id to `"train"` or `"eval"`.

    Deterministic in two senses that matter: sorting `meeting_ids` first means the
    result does not depend on the order the caller happened to discover meetings in,
    and a fixed `seed` means re-running the carve (e.g. after adding meetings found
    late) reproduces the same assignment for every id that was already present —
    the split does not silently reshuffle underneath work already done against it.
    """
    ids = sorted(set(meeting_ids))
    if eval_n > len(ids):
        raise ValueError(f"eval_n={eval_n} exceeds the number of meetings ({len(ids)})")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    eval_ids = set(shuffled[:eval_n])
    return {mid: ("eval" if mid in eval_ids else "train") for mid in ids}


@dataclass
class Manifest:
    """The whole corpus's provenance ledger, keyed by meeting id."""

    records: dict[str, MeetingRecord] = field(default_factory=dict)

    def set_split(self, splits: dict[str, str]) -> None:
        """Apply a `carve_splits` result, creating any missing records as `"train"`
        by default before overwriting with the carved split."""
        for meeting_id, split in splits.items():
            existing = self.records.get(meeting_id, MeetingRecord(meeting_id))
            self.records[meeting_id] = MeetingRecord(
                meeting_id=meeting_id,
                split=split,
                translated_by=existing.translated_by,
                composed_by=existing.composed_by,
                human_validated=existing.human_validated,
            )

    def eval_ids(self) -> set[str]:
        return {mid for mid, r in self.records.items() if r.split == "eval"}

    def train_ids(self) -> set[str]:
        return {mid for mid, r in self.records.items() if r.split == "train"}

    def to_list(self) -> list[dict]:
        return [r.manifest() for r in sorted(self.records.values(), key=lambda r: r.meeting_id)]
