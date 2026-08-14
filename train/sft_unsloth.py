#!/usr/bin/env python3
"""Fine-tune FunctionGemma-270M with Unsloth (PLAN.md §3).

Two things here are not defaults and should not be "simplified":

**bf16 only.** Gemma 3/4 activations exceed float16's 65,504 and gradients go to infinity
on float16-only GPUs. This box is Blackwell, so bf16 is native and the workaround is
unnecessary — but the assertion stays, because silently training in fp16 produces a model
that looks trained and isn't.

**Completion-only loss.** The prompt is SYS + STATE + a 2k-token chunk. Training on it
teaches transcript reproduction, not op emission. `train_on_responses_only` masks
everything up to the model turn.

Eval during training tracks **valid-op rate**, not loss: loss keeps improving while the
model learns to emit fluent *invalid* ops, so loss alone will happily report progress on a
model that is getting worse at the actual task.

    python train/sft_unsloth.py --train data/sft/train.jsonl --valid data/sft/valid.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Bind the REAL trl trainer before `import unsloth` replaces the top-level trl.SFTTrainer
# with its own wrapper. The CONFIG, however, must be the replaced trl.SFTConfig (below,
# in main): trl's trainer rebuilds the args via to_dict() when the isinstance check fails,
# and to_dict() deliberately obfuscates token strings ("<EOS_TOKEN>") — which then fails
# the trainer's own eos validation. Constructing the replaced config class keeps the
# isinstance check true and the real eos_token passes through untouched.
from trl.trainer.sft_trainer import SFTTrainer  # noqa: E402

STUDENT = "google/functiongemma-270m-it"

# Gemma/Qwen turn markers — the boundary completion-only masking keys on.
# tokenize_sample auto-detects which one the rendered template actually uses.
INSTRUCTION_PARTS = ("<start_of_turn>user\n", "<|im_start|>user\n")
RESPONSE_PARTS = ("<start_of_turn>model\n", "<|im_start|>assistant\n", "<|start_of_role|>assistant<|end_of_role|>", "Assistant:")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train", type=Path, required=True, help="JSONL from build_sft.py")
    p.add_argument("--valid", type=Path, default=None)
    p.add_argument("--model", default=STUDENT)
    p.add_argument(
        "--resume",
        default=None,
        help="continue from a saved checkpoint instead of the base model (phase-2 "
        "real-data adaptation keeps the phase-1 protocol pattern)",
    )
    p.add_argument("--out", type=Path, default=Path("runs/sft-v1"))
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument(
        "--regime",
        choices=["full", "lora"],
        default="full",
        help="270M fits a full fine-tune comfortably; LoRA is the comparison arm",
    )
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="default: 5e-5 full, 2e-4 LoRA")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4, help="effective batch = bs * accum")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-gguf", action="store_true", help="export Q4_K_M after training")
    return p


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


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

    The dataset is pre-tokenized deliberately: trl 0.24's text-processing `map`
    pickles its local tokenize_fn to a multiprocess pool, and dill's recursive
    walk of the function globals hits torch 2.10+'s unpicklable ConfigModuleInstance
    (multiprocess sets dill recurse=True). Passing input_ids makes trl skip those
    maps entirely ("If the dataset is already preprocessed... skip").

    Completion-only masking is done here (labels = -100 before the model turn),
    so unsloth's train_on_responses_only collator patch is not needed either.
    """
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["completion"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if tokenizer.bos_token != tokenizer.eos_token and text.count(tokenizer.bos_token or "<bos>") > 1:
        # (Granite uses bos == eos == <|end_of_text|>, which the template legitimately
        # emits once per turn — the double-bos bug this guards against does not apply.)
        raise SystemExit("template produced two <bos> tokens — fix before training")
    def _ids(text: str) -> list[int]:
        out = tokenizer(text=text, add_special_tokens=False)["input_ids"]
        # Qwen3-VL processors return a batch dim (list of lists); unwrap it.
        return out[0] if out and isinstance(out[0], list) else out

    ids = _ids(text)
    if len(ids) > max_length:
        # Front-truncate: the completion (target) sits at the end and must survive.
        cut = len(ids) - max_length
        print(f"[sft] WARNING: sample {len(ids)} tokens > {max_length}; dropping {cut} from the front")
        ids = ids[cut:]
    resp = None
    for marker in RESPONSE_PARTS:
        m_ids = _ids(marker)
        idx = find_subsequence(ids, m_ids)
        if idx is not None and (resp is None or idx > resp[0]):
            resp = (idx, m_ids)
    if resp is None:
        # String-level fallback: plain-text markers ("Assistant:") merge differently
        # in context than standalone (BPE boundary effects). Split the rendered text
        # at the marker string and concatenate the two tokenizations.
        for marker in RESPONSE_PARTS:
            pos = text.rfind(marker)
            if pos < 0:
                continue
            head = _ids(text[:pos])
            tail = _ids(text[pos:])
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
    """A single prompt version across train and valid, or the run is not comparable."""
    versions = {r.get("prompt_version") for r in rows}
    for other in extra:
        versions |= {r.get("prompt_version") for r in other}
    versions.discard(None)
    if len(versions) != 1:
        raise SystemExit(
            f"prompt versions in data: {sorted(versions)}. Training across a prompt change "
            "makes the run incomparable to any earlier eval (CLAUDE.md §7.8)."
        )
    return versions.pop()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit(
            "bf16 unsupported on this GPU. Gemma activations exceed float16's 65,504 and "
            "gradients become infinity — use an Ampere+ card or Unsloth's fp16 workaround."
        )

    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig  # the unsloth-replaced class: keeps trl's isinstance check true

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
    # Qwen3.5 ships eos_token="<EOS_TOKEN>" (a training-only token absent from the
    # vocab); the chat template's real terminator is <|im_end|>. trl's collator needs
    # a vocab-present eos, so normalise here (model config untouched). Non-Qwen models
    # (e.g. Gemma: <end_of_turn>) keep their own vocab-present eos untouched.
    _eos = tokenizer.eos_token or "<end_of_turn>"
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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    train_ds = Dataset.from_list([tokenize_sample(r, tokenizer, args.max_seq_length) for r in train_rows])
    valid_ds = (
        Dataset.from_list([tokenize_sample(r, tokenizer, args.max_seq_length) for r in valid_rows])
        if valid_rows
        else None
    )

    lr = args.lr if args.lr is not None else (5e-5 if args.regime == "full" else 2e-4)
    sft_cfg = SFTConfig(
            output_dir=str(args.out),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
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
            eos_token=_eos,
        )
    print(f"[sft] cfg eos_token={sft_cfg.eos_token!r} type={type(sft_cfg).__module__}.{type(sft_cfg).__name__}", flush=True)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=SFTConfig(
            output_dir=str(args.out),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=lr,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            bf16=True,
            fp16=False,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if valid_ds else "no",
            per_device_eval_batch_size=1,  # eval materialises fp32 logits: batch*seq*256k vocab
            max_length=args.max_seq_length,
            dataset_text_field=None,  # pre-tokenized; labels carry the completion mask
            seed=args.seed,
            report_to="none",
            eos_token=_eos,
        ),
    )

    stats = trainer.train()
    print(f"[sft] final loss {stats.training_loss:.4f}")
    print(
        "[sft] loss is NOT the signal to judge this on — run eval/screen.py against the "
        "exported model and read valid-op, anchor and revision rates."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "final"))
    tokenizer.save_pretrained(str(args.out / "final"))

    if args.save_gguf:
        # Quantisation can break exact special-token emission, so the anchor and valid-op
        # rates must be re-measured on the GGUF rather than inherited (PLAN.md §3).
        model.save_pretrained_gguf(str(args.out / "gguf"), tokenizer, quantization_method="q4_k_m")
        print("[sft] exported GGUF — re-run the screen on it before trusting any number")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
