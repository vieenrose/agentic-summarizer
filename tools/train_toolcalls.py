"""Fine-tune a causal LM on the SPEC §4.1 v1.0 tool-call pool, without unsloth.

    python tools/train_toolcalls.py --train data/staging/sft_pool_tools.jsonl \\
        --valid data/staging/valid_tools.jsonl --model Qwen/Qwen3.5-0.8B \\
        --out runs/qwen-tools-v1

`cli/train_sft.py` stays the production path for the v0 protocol. This exists because
unsloth's generated `UnslothSFTTrainer` refuses Qwen3.5-0.8B: it loads the model as a
`Qwen3VLProcessor`, and TRL then reads `args.eos_token` as the literal placeholder
`'<EOS_TOKEN>'` and aborts — while the config we hand it demonstrably holds `'<|im_end|>'`
and the tokenizer resolves that token to id 248046. Clearing the compiled cache did not
help. Rather than keep debugging generated code, this trains through plain
`transformers.Trainer`, which has no such layer.

**Completion-only masking is done here explicitly**, the same property `train_sft` gets
from TRL: labels are -100 across the prompt so loss is taken only on the assistant turn.
Without it the model spends most of its capacity learning to reproduce the transcript,
which is not the task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

IGNORE = -100


def build_examples(rows: list[dict], tokenizer, max_len: int) -> list[dict]:
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": r["system"]},
                {"role": "user", "content": r["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + r["completion"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if len(full_ids) > max_len:
            continue  # a truncated target teaches a truncated op; drop the row instead
        labels = list(full_ids)
        labels[: len(prompt_ids)] = [IGNORE] * len(prompt_ids)
        out.append({"input_ids": full_ids, "labels": labels,
                    "attention_mask": [1] * len(full_ids)})
    return out


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    def pad(key, fill):
        return torch.tensor([b[key] + [fill] * (n - len(b[key])) for b in batch])
    return {"input_ids": pad("input_ids", pad_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", IGNORE)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--valid", type=Path, default=None)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(args.model)
    def load(f: Path) -> list[dict]:
        return [
            json.loads(ln)
            for ln in f.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    train_rows = load(args.train)
    versions = {r.get("prompt_version") for r in train_rows}
    if len(versions) != 1:
        print(f"[train] REFUSED: mixed prompt versions {versions}", file=sys.stderr)
        return 1
    print(f"[train] {len(train_rows)} rows, prompt_version={versions.pop()}", file=sys.stderr)

    train_ds = build_examples(train_rows, tok, args.max_len)
    valid_ds = build_examples(load(args.valid), tok, args.max_len) if args.valid else None
    print(f"[train] {len(train_ds)} usable train examples"
          f"{f', {len(valid_ds)} valid' if valid_ds else ''}", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=10, bf16=True, seed=args.seed,
        report_to=[], eval_strategy="epoch" if valid_ds else "no",
        per_device_eval_batch_size=args.batch_size,
        # Keep the BEST epoch, not the last. Measured on both Qwen runs: eval loss
        # bottomed at epoch 2 (0.877 / 0.829) and ROSE at epoch 3 (0.959 / 0.913), so
        # `save_strategy="no"` was silently shipping the overfitted checkpoint and
        # discarding the better one. Any tuning conclusion drawn from the epoch-3 model
        # was measuring overfitting as much as the change under test.
        save_strategy="epoch" if valid_ds else "no",
        save_total_limit=2,
        load_best_model_at_end=bool(valid_ds),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=valid_ds,
                      data_collator=lambda b: collate(b, pad_id))
    trainer.train()
    final = args.out / "final"
    model.config.use_cache = True
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    print(f"[train] saved -> {final}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
