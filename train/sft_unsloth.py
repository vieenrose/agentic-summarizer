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

STUDENT = "google/functiongemma-270m-it"

# Gemma turn markers — the boundary completion-only masking keys on.
INSTRUCTION_PART = "<start_of_turn>user\n"
RESPONSE_PART = "<start_of_turn>model\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train", type=Path, required=True, help="JSONL from build_sft.py")
    p.add_argument("--valid", type=Path, default=None)
    p.add_argument("--model", default=STUDENT)
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
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    train_rows = load_jsonl(args.train)
    valid_rows = load_jsonl(args.valid) if args.valid else []
    version = assert_prompt_version_consistent(train_rows, valid_rows)
    print(f"[sft] {len(train_rows)} train / {len(valid_rows)} valid samples, prompt {version}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=args.regime == "full",
    )
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

    def to_text(row: dict) -> dict:
        """Render one sample with the model's own chat template.

        The same template must serve training and inference — a mismatch here is the
        classic silent quality killer, and llama.cpp adds its own <bos>, so exactly one
        must appear (PLAN.md §3).
        """
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if text.count(tokenizer.bos_token or "<bos>") > 1:
            raise SystemExit("template produced two <bos> tokens — fix before training")
        return {"text": text}

    train_ds = Dataset.from_list([to_text(r) for r in train_rows])
    valid_ds = Dataset.from_list([to_text(r) for r in valid_rows]) if valid_rows else None

    lr = args.lr if args.lr is not None else (5e-5 if args.regime == "full" else 2e-4)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
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
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            seed=args.seed,
            report_to="none",
        ),
    )

    # Completion-only: mask SYS/STATE/CHUNK, train on op tokens only.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
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
