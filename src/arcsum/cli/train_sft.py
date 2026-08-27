"""`arcsum-train-sft`: fine-tune MiniCPM5-1B with Unsloth on a `build_sft.py` pool
(SPEC §4, Phase 2). Needs the `train` extra (`pip install 'arcsum-agentic[train]'`) --
never installed on the reference device (SPEC §6), so this module must stay
importable (for `pytest` collection, `--help`, argument parsing) without it; only
`main()`'s actual training path requires it.

Structure mirrors `pi-agent:train/sft_unsloth.py` (the prior, superseded project's
FunctionGemma-270M script) — the completion-only-masking and pre-tokenization approach
there is sound and not specific to that model. What's actually new here:

**MiniCPM5-1B's chat template uses ChatML markers** (`<|im_start|>assistant\n`), not
Gemma's `<start_of_turn>model\n`, and confirms (checked directly against the
tokenizer) `<|im_end|>` IS present in vocab, so eos is normalised to it exactly as the
reference script's Qwen-handling branch already does — MiniCPM5 needed no new logic
there.

**Full fine-tune, not LoRA, by default.** At 1B params / hidden_size 1536 / 24 layers
this comfortably fits a single 32GB card the same way 270M did for the prior project;
LoRA stays available via `--regime lora` as the comparison arm SPEC's own ablation asks
for, not because full doesn't fit.

**torch 2.11 vs unsloth's fused CE was an open question this tool's first real run
answered empirically, not by assumption.** CLAUDE.md flags that the prior project
pinned torch 2.10 after 2.11 broke it, and that their `UNSLOTH_COMPILE_DISABLE=1`
workaround was ruled out because the unfused loss materialises batch x 4096 x 256k logits
and OOMs — but MiniCPM5's vocab is 130,560, roughly half, so the unfused path was
suspected to simply work here. It does: a smoke test (`--max-steps 5`) and the full
pilot run both completed cleanly with no workaround. Kept undocumented-as-fact no
further than that — a different model/vocab size should still smoke-test first.

**`torchrun`, not `accelerate launch`, for multi-GPU.** A narrow, over-triggering
guard in `unsloth_zoo.training_utils.get_max_steps` (`if training_args.world_size > 1:
raise RuntimeError(...)`) reads as a hard single-GPU restriction from the source
alone; it is not one in practice when launched correctly (confirmed both against
Unsloth's own DDP docs and empirically end to end: `torchrun --nproc_per_node=N`
against this exact script produced a real 2-GPU run, effective batch size doubled,
final checkpoint intact). `accelerate launch` was not what that guide called for and
was not what was tested.

**Final checkpoint save is rank-guarded.** Under `torchrun` DDP every rank reaches the
end of `main()` with its own full copy of the model; without `is_world_process_zero()`
every rank races to write the SAME output files concurrently. A first DDP smoke test
got lucky (safetensors' write-then-rename meant the last writer simply clobbered the
rest intact, verified via `safetensors.safe_open`) — that is not something to rely on
for a real run, where an unlucky interleaving could truncate or corrupt the
checkpoint.

**Eval signal: don't trust loss alone.** Loss keeps improving while the model learns
to emit fluent *invalid* ops, so the real judgment happens later against the exported
model with `arcsum-run-arms` + `arcsum-score`/`arcsum-judge` on held-out meetings
(SPEC §5), not from the training curve.

    arcsum-train-sft --train data/staging/sft/train.jsonl \\
        --valid data/staging/sft/valid.jsonl --max-steps 5   # smoke test first

    torchrun --nproc_per_node=2 -m arcsum.cli.train_sft \\
        --train data/staging/sft/train.jsonl --valid data/staging/sft/valid.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    # Bind the REAL trl trainer before `import unsloth` replaces the top-level
    # trl.SFTTrainer with its own wrapper (trl's trainer rebuilds args via to_dict()
    # when the isinstance check fails, and to_dict() obfuscates token strings, which
    # then fails the trainer's own eos validation) -- this ordering is why the import
    # can't simply be deferred into main() alongside the rest of the `train` extra.
    from trl.trainer.sft_trainer import SFTTrainer as _RealSFTTrainer
except ImportError:
    _RealSFTTrainer = None


class MissingExtraError(ImportError):
    """Raised in place of a bare `ModuleNotFoundError`, naming the extra to install.
    Subclasses `ImportError` so an `except ImportError` caller still catches it."""


STUDENT = "openbmb/MiniCPM5-1B"

INSTRUCTION_PARTS = ("<|im_start|>user\n", "<start_of_turn>user\n")
RESPONSE_PARTS = (
    "<|im_start|>assistant\n",
    "<start_of_turn>model\n",
    "<|start_of_role|>assistant<|end_of_role|>",
    "Assistant:",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train", type=Path, required=True, help="JSONL from build_sft.py")
    p.add_argument("--valid", type=Path, default=None)
    p.add_argument("--model", default=STUDENT)
    p.add_argument("--resume", default=None, help="continue from a saved checkpoint")
    p.add_argument("--out", type=Path, default=Path("runs/sft-pilot-v1"))
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--regime", choices=["full", "lora"], default="full")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="default: 5e-5 full, 2e-4 LoRA")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = full run; >0 for a smoke test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-gguf", action="store_true")
    return p


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def find_subsequence(seq: list[int], sub: list[int]) -> int | None:
    """Index of the LAST occurrence of `sub` in `seq`, or None."""
    if not sub:
        return None
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i : i + len(sub)] == sub:
            return i
    return None


def tokenize_sample(row: dict, tokenizer, max_length: int) -> dict:
    """One SFT row -> pre-tokenized input_ids + completion-only labels.

    Pre-tokenized because trl 0.24's text-processing `map` pickles its local
    tokenize_fn to a multiprocess pool, and dill's recursive walk of the function
    globals hits torch's unpicklable ConfigModuleInstance. Passing input_ids makes
    trl skip that map entirely.
    """
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["completion"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if tokenizer.bos_token != tokenizer.eos_token and text.count(tokenizer.bos_token or "<s>") > 1:
        raise SystemExit("template produced two bos tokens — fix before training")

    def _ids(s: str) -> list[int]:
        out = tokenizer(text=s, add_special_tokens=False)["input_ids"]
        return out[0] if out and isinstance(out[0], list) else out

    ids = _ids(text)
    if len(ids) > max_length:
        cut = len(ids) - max_length
        print(
            f"[sft] WARNING: sample {len(ids)} tokens > {max_length}; dropping {cut} from the front"
        )
        ids = ids[cut:]

    resp = None
    for marker in RESPONSE_PARTS:
        m_ids = _ids(marker)
        idx = find_subsequence(ids, m_ids)
        if idx is not None and (resp is None or idx > resp[0]):
            resp = (idx, m_ids)
    if resp is None:
        for marker in RESPONSE_PARTS:
            pos = text.rfind(marker)
            if pos < 0:
                continue
            head = _ids(text[:pos])
            tail = _ids(text[pos:])
            # `tail` (the response marker + completion, the actual training target)
            # must survive intact -- so if head+tail still exceeds max_length here,
            # only `head` gets truncated further, exactly like the id-level attempt
            # above. Without this, a marker that only the id-level search failed to
            # find (a tokenizer merge-boundary mismatch, this branch's actual reason
            # to exist) would silently re-expand `ids` back to the FULL untruncated
            # length, discarding the truncation above outright.
            if len(head) + len(tail) > max_length:
                head = head[-(max_length - len(tail)) :] if max_length > len(tail) else []
            idx = len(head)
            if resp is None or idx > resp[0]:
                resp = (idx, None)
                ids = head + tail
    if resp is None:
        raise SystemExit("response marker not found in sample — template mismatch")

    idx = resp[0]
    labels = [-100] * idx + ids[idx:]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def assert_prompt_version_consistent(rows: list[dict], *extra: list[dict]) -> str:
    versions = {r.get("prompt_version") for r in rows}
    for other in extra:
        versions |= {r.get("prompt_version") for r in other}
    versions.discard(None)
    if len(versions) != 1:
        raise SystemExit(
            f"prompt versions in data: {sorted(versions)}. Training across a prompt change "
            "makes the run incomparable to any earlier trace or eval number."
        )
    return versions.pop()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if _RealSFTTrainer is None:
        raise MissingExtraError(
            "arcsum.cli.train_sft needs the 'train' extra "
            "(pip install 'arcsum-agentic[train]') — never installed on the reference "
            "device (SPEC §6); only import this module without it for --help/argument "
            "parsing/testing."
        )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 unsupported on this GPU.")

    from datasets import Dataset
    from trl import SFTConfig  # unsloth-replaced class: keeps trl's isinstance check true
    from unsloth import FastLanguageModel

    train_rows = load_jsonl(args.train)
    valid_rows = load_jsonl(args.valid) if args.valid else []
    version = assert_prompt_version_consistent(train_rows, valid_rows)
    print(f"[sft] {len(train_rows)} train / {len(valid_rows)} valid samples, prompt {version}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.resume or args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=args.regime == "full",
    )
    if hasattr(tokenizer, "tokenizer") and not hasattr(tokenizer, "get_vocab"):
        tokenizer = tokenizer.tokenizer

    _eos = tokenizer.eos_token or "</s>"
    if "<|im_end|>" in tokenizer.get_vocab():
        if tokenizer.eos_token != "<|im_end|>":
            tokenizer.eos_token = "<|im_end|>"
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        _eos = "<|im_end|>"

    if args.regime == "lora":
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    print("[sft] tokenizing train...", flush=True)
    train_ds = Dataset.from_list(
        [tokenize_sample(r, tokenizer, args.max_seq_length) for r in train_rows]
    )
    valid_ds = None
    if valid_rows:
        print("[sft] tokenizing valid...", flush=True)
        valid_ds = Dataset.from_list(
            [tokenize_sample(r, tokenizer, args.max_seq_length) for r in valid_rows]
        )

    lr = args.lr if args.lr is not None else (5e-5 if args.regime == "full" else 2e-4)
    sft_cfg = SFTConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if valid_ds else "no",
        per_device_eval_batch_size=1,
        max_length=args.max_seq_length,
        dataset_text_field=None,
        seed=args.seed,
        report_to="none",
        ddp_find_unused_parameters=True,
        eos_token=_eos,
    )
    print(f"[sft] cfg eos_token={sft_cfg.eos_token!r} regime={args.regime} lr={lr}", flush=True)

    trainer = _RealSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=sft_cfg,
    )

    stats = trainer.train()
    print(f"[sft] final loss {stats.training_loss:.4f}")
    print(
        "[sft] loss is NOT the signal to judge this on — run arcsum-run-arms + "
        "arcsum-score/arcsum-judge against the exported model on held-out meetings."
    )

    # See module docstring: every DDP rank reaches this line with its own full model
    # copy, so the save must be rank-guarded or every rank races to write the same
    # output files concurrently.
    if trainer.is_world_process_zero():
        args.out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.out / "final"))
        tokenizer.save_pretrained(str(args.out / "final"))

        if args.save_gguf:
            model.save_pretrained_gguf(
                str(args.out / "gguf"), tokenizer, quantization_method="q8_0"
            )
            print("[sft] exported GGUF — re-measure everything on it, do not inherit fp16 numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
