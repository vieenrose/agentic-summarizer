"""Pins the pure-Python parts of `arcsum.cli.train_sft` -- the pre-tokenization and
completion-only masking logic, and the extras-gating -- without needing the `train`
extra installed (never present on the reference device, SPEC §6) or a GPU. The actual
training path (`FastLanguageModel.from_pretrained`, `trainer.train()`) is exercised
only by a real smoke test against real hardware, not here.
"""

from __future__ import annotations

import pytest

from arcsum.cli.train_sft import (
    RESPONSE_PARTS,
    MissingExtraError,
    assert_prompt_version_consistent,
    build_parser,
    find_subsequence,
    main,
    tokenize_sample,
)

# --- find_subsequence ------------------------------------------------------------------


def test_find_subsequence_returns_last_occurrence() -> None:
    assert find_subsequence([1, 2, 3, 2, 3, 4], [2, 3]) == 3


def test_find_subsequence_returns_none_when_absent() -> None:
    assert find_subsequence([1, 2, 3], [9, 9]) is None


def test_find_subsequence_empty_sub_returns_none() -> None:
    assert find_subsequence([1, 2, 3], []) is None


def test_find_subsequence_sub_longer_than_seq_returns_none() -> None:
    assert find_subsequence([1, 2], [1, 2, 3]) is None


# --- assert_prompt_version_consistent ---------------------------------------------------


def test_assert_prompt_version_consistent_returns_the_shared_version() -> None:
    train = [{"prompt_version": "sys-v1"}, {"prompt_version": "sys-v1"}]
    valid = [{"prompt_version": "sys-v1"}]
    assert assert_prompt_version_consistent(train, valid) == "sys-v1"


def test_assert_prompt_version_consistent_raises_on_mixed_versions() -> None:
    train = [{"prompt_version": "sys-v1"}]
    valid = [{"prompt_version": "sys-v2"}]
    with pytest.raises(SystemExit):
        assert_prompt_version_consistent(train, valid)


def test_assert_prompt_version_consistent_ignores_missing_field() -> None:
    """A record with no `prompt_version` key at all (e.g. hand-written test fixture)
    must not itself count as a second, conflicting version."""
    rows = [{"prompt_version": "sys-v1"}, {}]
    assert assert_prompt_version_consistent(rows) == "sys-v1"


# --- tokenize_sample ---------------------------------------------------------------------


class _CharTokenizer:
    """A trivial tokenizer where every character is its own token, so exact
    input_ids/labels are computable by hand -- no real HF tokenizer needed to pin the
    masking logic itself."""

    bos_token = "<s>"
    eos_token = "</s>"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id(self, ch: str) -> int:
        return self._vocab.setdefault(ch, len(self._vocab))

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [self._id(c) for c in text]}

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False) -> str:
        sys_msg, user_msg, asst_msg = messages
        return (
            f"{sys_msg['content']}\n<|im_start|>user\n{user_msg['content']}\n"
            f"{RESPONSE_PARTS[0]}{asst_msg['content']}"
        )


def test_tokenize_sample_masks_everything_before_the_response_marker() -> None:
    """The mask boundary is the marker's START, not its end -- the response-turn
    marker tokens themselves are part of the trainable target (the model must learn
    to emit its own turn-start token, not just what follows it), matching the
    pi-agent-derived design this was ported from."""
    row = {"system": "SYS", "prompt": "USER TEXT", "completion": "COMPLETION"}
    tok = _CharTokenizer()
    out = tokenize_sample(row, tok, max_length=10_000)

    text = tok.apply_chat_template(
        [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
    )
    response_start = text.index(RESPONSE_PARTS[0])

    assert len(out["input_ids"]) == len(text)
    assert len(out["labels"]) == len(out["input_ids"])
    # Everything before the marker is masked...
    assert all(label == -100 for label in out["labels"][:response_start])
    # ...and the marker onward (marker + completion) is not.
    assert all(label != -100 for label in out["labels"][response_start:])
    # The unmasked labels decode back to exactly "marker + completion".
    marker_and_completion_ids = [tok._id(c) for c in RESPONSE_PARTS[0] + row["completion"]]
    assert out["labels"][response_start:] == marker_and_completion_ids
    assert all(m == 1 for m in out["attention_mask"])


def test_tokenize_sample_front_truncates_over_length_samples() -> None:
    """The completion (target) sits at the end and must survive truncation --
    front-truncating (not back-truncating) is what protects it. Sized so the marker +
    completion comfortably fit within max_length (the realistic case: max_length is
    always far larger than a single turn marker) -- a separate, much smaller
    max_length that cuts INTO the marker itself is covered by the fallback-path test
    below."""
    row = {"system": "S" * 30, "prompt": "U" * 30, "completion": "COMPLETION"}
    tok = _CharTokenizer()
    out = tokenize_sample(row, tok, max_length=40)

    assert len(out["input_ids"]) == 40
    completion_ids = [tok._id(c) for c in row["completion"]]
    assert out["input_ids"][-len(completion_ids) :] == completion_ids
    assert out["labels"][-len(completion_ids) :] == completion_ids


def test_tokenize_sample_fallback_path_still_respects_max_length() -> None:
    """When max_length is small enough to truncate INTO the marker itself, the
    id-level search fails and the string-level fallback re-tokenizes from the
    ORIGINAL untruncated text -- which, before this was fixed, silently discarded
    the truncation and re-expanded the sample back to its full untruncated length
    (measured: 151 tokens instead of the requested 20). The fallback must truncate
    its own rebuilt `head` to still respect max_length; only the marker + completion
    (`tail`, 32 tokens here) is protected from truncation, never the whole sample.

    Below `len(tail)`, there is nothing left to cut without truncating the
    completion itself (which must never happen), so the result floors at exactly
    `len(tail)` rather than hitting max_length precisely -- both max_length values
    below demonstrate that floor.
    """
    row = {"system": "S" * 50, "prompt": "U" * 50, "completion": "COMPLETION"}
    tail_len = len(RESPONSE_PARTS[0]) + len(row["completion"])  # 32

    for max_length in (20, 25):  # both well below tail_len -> floors at tail_len
        tok = _CharTokenizer()
        out = tokenize_sample(row, tok, max_length=max_length)
        assert len(out["input_ids"]) == tail_len
        completion_ids = [tok._id(c) for c in row["completion"]]
        assert out["input_ids"][-len(completion_ids) :] == completion_ids
        assert out["labels"][-len(completion_ids) :] == completion_ids

    # Above tail_len, the fallback's head-truncation hits max_length exactly.
    tok = _CharTokenizer()
    out = tokenize_sample(row, tok, max_length=40)
    assert len(out["input_ids"]) == 40


def test_tokenize_sample_raises_on_two_bos_tokens() -> None:
    class _DoubleBosTokenizer(_CharTokenizer):
        def apply_chat_template(self, messages, tokenize=False):
            return "<s>system<s>user<|im_start|>assistant\ncompletion"

    row = {"system": "s", "prompt": "u", "completion": "c"}
    with pytest.raises(SystemExit):
        tokenize_sample(row, _DoubleBosTokenizer(), max_length=1000)


def test_tokenize_sample_raises_when_no_response_marker_present() -> None:
    class _NoMarkerTokenizer(_CharTokenizer):
        def apply_chat_template(self, messages, tokenize=False):
            return "just plain text with no turn markers at all"

    row = {"system": "s", "prompt": "u", "completion": "c"}
    with pytest.raises(SystemExit):
        tokenize_sample(row, _NoMarkerTokenizer(), max_length=1000)


# --- CLI plumbing ------------------------------------------------------------------------


def test_build_parser_requires_train() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args(["--train", "train.jsonl"])
    assert args.valid is None
    assert args.regime == "full"
    assert args.epochs == 3.0
    assert args.max_steps == -1
    assert args.max_seq_length == 4096


def test_main_raises_missing_extra_error_before_touching_gpu_when_train_extra_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulates the 'train' extra not being installed (its real state on the
    reference device, SPEC §6) -- must fail with a clear, named error before any
    torch/CUDA code runs, not a bare ImportError deep in a traceback."""
    monkeypatch.setattr("arcsum.cli.train_sft._RealSFTTrainer", None)
    with pytest.raises(MissingExtraError, match="train"):
        main(["--train", "nonexistent.jsonl"])
