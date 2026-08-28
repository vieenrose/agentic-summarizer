"""Traces -> SFT rows, with the discipline the prior project's `build_sft.py` carries
(SPEC §4.2, §8 risk 3).

Built on real `agent.Trace`/`agent.Step` objects rather than a parallel on-disk
schema — a `Trace` produced by replaying a teacher through `agent.run_agent` (its raw
edit-line output *is* the gold completion) already carries everything a sample needs:
the exact prompt, the exact target, and `prompt_version`.

**Raw text target, no function-call translation.** MiniCPM5 is not a function-call
model — unlike the prior project's FunctionGemma path, `SftSample.completion` is the
teacher's raw edit-line (or prose) text verbatim.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from arcsum.agent import Trace

#: SPEC §8 risk 3.
DEFAULT_MAX_NOP_FRAC = 0.35


class MixedPromptVersionError(ValueError):
    """Training across a prompt change makes the run incomparable to any earlier
    trace or eval number — refuse loudly rather than silently mixing versions."""


@dataclass(frozen=True, slots=True)
class SftSample:
    meeting: str
    step: int
    prompt_version: str
    system: str
    #: The exact rendered user turn — completion-only masking needs `prompt` and
    #: `completion` kept separate, since training on the prompt "teaches the model to
    #: reproduce transcripts... which is not the task."
    prompt: str
    completion: str
    is_nop: bool


def build_samples(meeting_id: str, trace: Trace) -> list[SftSample]:
    """One `SftSample` per reading step, plus one more for the synthesis call if the
    trace has one — `is_nop=False` always for synthesis, since it is never a curation
    step subject to the NOP-share cap.

    **A guarded (empty-memory) synthesis yields NO sample.** When every reading step
    NOPs, `agent.synthesize_memory` short-circuits without calling the model and its
    `raw` is `""` — emitting that would put an empty-completion row into the training
    pool. Nor is `prose.text` a valid substitute: it is the fixed
    `agent.EMPTY_MEMORY_PROSE` constant, so training on it would teach the model to
    reproduce a hardcoded string that the deterministic guard already handles at
    inference, and risk biasing it toward "no content" on thin-but-nonempty memories.
    There is simply no model behaviour to learn from this case.
    """
    samples = [
        SftSample(
            meeting=meeting_id,
            step=step.index,
            prompt_version=trace.prompt_version,
            system=step.system,
            prompt=step.user,
            completion=step.raw,
            is_nop=step.is_nop,
        )
        for step in trace.steps
    ]
    if trace.synthesis is not None and not trace.synthesis.skipped_empty_memory:
        samples.append(
            SftSample(
                meeting=meeting_id,
                step=len(trace.steps),
                prompt_version=trace.prompt_version,
                system=trace.synthesis.system,
                prompt=trace.synthesis.user,
                completion=trace.synthesis.raw,
                is_nop=False,
            )
        )
    return samples


def check_single_prompt_version(samples: Sequence[SftSample]) -> str:
    """Refuse (exit-2-style, via exception) rather than silently building across a
    mixed-prompt-version pool. Returns the single shared version on success."""
    versions = {s.prompt_version for s in samples}
    if not versions:
        raise MixedPromptVersionError("no samples to build from")
    if len(versions) > 1:
        raise MixedPromptVersionError(f"mixed prompt versions in one build: {sorted(versions)}")
    return next(iter(versions))


def downsample_nop(
    samples: Sequence[SftSample], *, max_nop_frac: float = DEFAULT_MAX_NOP_FRAC, seed: int = 0
) -> list[SftSample]:
    """Cap the NOP share of the pool at `max_nop_frac` by randomly dropping excess
    NOP samples — SPEC §8 risk 3's ~35% threshold, applied as a knob rather than an
    assumption. A pool already under the cap is returned unchanged.
    """
    if max_nop_frac >= 1.0:
        return list(samples)

    nops = [s for s in samples if s.is_nop]
    non_nops = [s for s in samples if not s.is_nop]
    if not non_nops or not nops:
        return list(samples)

    # Solve max_nops/(max_nops+len(non_nops)) <= max_nop_frac for max_nops.
    max_nops = int(max_nop_frac * len(non_nops) / (1 - max_nop_frac))
    if len(nops) <= max_nops:
        return list(samples)

    kept_nops = set(random.Random(seed).sample(nops, max_nops))
    return [s for s in samples if not s.is_nop or s in kept_nops]


def oversample_drop(
    samples: Sequence[SftSample], *, target_drop_frac: float = 0.0, seed: int = 0
) -> list[SftSample]:
    """Raise the share of DROP-bearing samples to `target_drop_frac` by duplicating
    them, mirroring `downsample_nop` as a knob rather than an assumption.
    `0.0` (the default) is a no-op, so existing builds are unchanged.

    **Motivation, measured 2026-08-27** against `runs/sft-synth-v1` on 40 pool rows
    whose gold completion contains a DROP: the student reproduced one in only 52% of
    them, but the misses were overwhelmingly BEHAVIOURAL rather than comprehension
    failures -- 30% recorded the superseding state via `ADD` and 10% rewrote the `ARC`,
    both of which prove the supersession WAS detected, while only 8% recorded nothing
    at all. The model reliably does the hard part (noticing) and skips the easy part
    (removing), which is the profile that responds to emphasis. Stale points then
    coexist with their own contradictions in memory, which is exactly the G1 symptom.

    Duplication, not loss-weighting, because the trainer takes a flat sample list and
    `build_sft`'s split/caps all operate on counts -- a weight column would have to be
    threaded through three layers that currently have no concept of one.

    **Ordering matters: run this AFTER `downsample_nop`.** DROP-bearing samples are
    never NOPs, so oversampling them first would inflate the non-NOP denominator and
    make the NOP cap admit more NOPs than SPEC §8 risk 3 intends.

    **The two knobs COMPOUND against NOP — check the resulting share, don't assume it.**
    `downsample_nop` solves its cap against the pool as it stands, then every row this
    function appends dilutes NOP further, so the final NOP share lands BELOW
    `max_nop_frac`, not at it. Measured 2026-08-27 on the `sft-dropv1` build: a raw pool
    of 38.2% NOP / 26.4% DROP-bearing became 25.7% NOP / 40.1% DROP-bearing under
    `--max-nop-frac 0.35 --target-drop-frac 0.40` -- NOP fell by a third below its
    natural rate. The resulting checkpoint learned to avoid emitting `NOP` at all and
    instead churned: on low-information chunks it would `DROP` a point and re-`ADD` a
    near-identical rewording, burning up to 45 of a 53-step meeting's steps on one topic
    while later chunks' real content never entered memory. Since the teacher's own NOP
    rate is a genuine signal (long procedural spans SHOULD be NOP), prefer a
    `max_nop_frac` at or above the raw rate when oversampling: `--max-nop-frac 0.40
    --target-drop-frac 0.32` held NOP at 35.3% and measured +0.049/+0.050/+0.060 mean
    ROUGE-1/2/L against the same baseline, versus +0.007/+0.024/+0.039 for `dropv1`.
    """
    if target_drop_frac <= 0.0:
        return list(samples)

    drops = [s for s in samples if "DROP" in s.completion]
    if not drops or len(drops) == len(samples):
        return list(samples)

    # Solve (len(drops)+extra)/(len(samples)+extra) = target for extra.
    total = len(samples)
    extra = int((target_drop_frac * total - len(drops)) / (1 - target_drop_frac))
    if extra <= 0:
        return list(samples)

    rng = random.Random(seed)
    return [*samples, *(rng.choice(drops) for _ in range(extra))]


def late_step_share(samples: Sequence[SftSample], *, min_step: int = 25) -> float | None:
    """Share of samples at step index >= `min_step` — the quantity `oversample_late_steps`
    targets, reported so the knob is never set blind (see `oversample_drop`'s warning
    about compounding)."""
    if not samples:
        return None
    return sum(1 for s in samples if s.step >= min_step) / len(samples)


def oversample_late_steps(
    samples: Sequence[SftSample],
    *,
    min_step: int = 25,
    target_frac: float = 0.0,
    seed: int = 0,
) -> list[SftSample]:
    """Raise the share of LATE reading steps to `target_frac` by duplication.
    `0.0` (the default) is a no-op, so existing builds are unchanged.

    **Motivation, measured 2026-08-27.** Deep in a long meeting the student stops making
    progress: it re-emits a byte-identical `ARC` while the transcript has already moved
    to an unrelated agenda item, so later chunks' content never enters memory. The gold
    pool barely covers that regime — only 1.6% of steps sit at index 40+ and 6.5% at
    30-39 — while the behaviour the model needs there is *more* common than early on,
    not less: the teacher's NOP rate RISES with step index (32% at 0-9 to 51% at 40+).
    The model is least trained exactly where the correct answer is most often "nothing
    changed, do not touch memory".

    This is not a cap artifact: gold `ARC`s stay compact late (mean 50.8 tokens at step
    >= 25 against an 80-token cap, only 2.6% within 10 of it), so the model is not being
    squeezed into restating by an impending overflow.

    **Ordering: run this LAST, after `downsample_nop` and `oversample_drop`, and CHECK
    the resulting shares.** All three knobs compound — see `oversample_drop` — and late
    steps are NOP-heavy, so this one pushes the NOP share back UP, partially undoing the
    NOP cap. That interaction is the reason `late_step_share`, `nop_share` and
    `drop_bearing_share` are all reported by `build_sft`: set this by measuring the
    result, never by assuming the target was hit.
    """
    if target_frac <= 0.0:
        return list(samples)

    late = [s for s in samples if s.step >= min_step]
    if not late or len(late) == len(samples):
        return list(samples)

    total = len(samples)
    extra = int((target_frac * total - len(late)) / (1 - target_frac))
    if extra <= 0:
        return list(samples)

    rng = random.Random(seed)
    return [*samples, *(rng.choice(late) for _ in range(extra))]


def drop_bearing_share(samples: Sequence[SftSample]) -> float | None:
    """Share of ALL samples whose completion contains a `DROP` — the quantity
    `oversample_drop` targets. Distinct from `drop_share`, which is the share of
    NON-NOP samples and is a revision-density proxy rather than a pool-balance knob."""
    if not samples:
        return None
    return sum(1 for s in samples if "DROP" in s.completion) / len(samples)


def nop_share(samples: Sequence[SftSample]) -> float | None:
    """The share this module is capping — reported alongside the build, not just
    enforced silently."""
    if not samples:
        return None
    return sum(1 for s in samples if s.is_nop) / len(samples)


def split_by_meeting(
    samples: Sequence[SftSample], *, valid_frac: float, seed: int = 0
) -> tuple[list[SftSample], list[SftSample]]:
    """Train/valid split assigned by MEETING, never by step — "sibling steps share
    STATE, so a step-level split leaks the answer for a held-out step into training."
    Returns `(train, valid)`.
    """
    meeting_ids = sorted({s.meeting for s in samples})
    if not meeting_ids:
        return [], []
    shuffled = list(meeting_ids)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, round(valid_frac * len(meeting_ids)))
    valid_ids = set(shuffled[:n_valid])
    train = [s for s in samples if s.meeting not in valid_ids]
    valid = [s for s in samples if s.meeting in valid_ids]
    return train, valid


def drop_share(samples: Sequence[SftSample]) -> float | None:
    """Share of non-NOP completions containing at least one `DROP` line — a coarse
    revision-share proxy at the SFT-sample level (text-based, since a sample's
    completion is raw text, not parsed `Op` objects)."""
    non_nop = [s for s in samples if not s.is_nop]
    if not non_nop:
        return None
    return sum(1 for s in non_nop if "DROP" in s.completion) / len(non_nop)
