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
import shutil
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
    # Default matches the 0.8B v1.0 builds exactly, so their results stay reproducible.
    # `adamw_8bit` exists for larger students: AdamW keeps two fp32 moments per parameter,
    # so a 2B model spends 16 GB on optimizer state alone (2e9 x 8 B) before a single
    # activation — which is what OOMs a 32 GB card, not the sequence length. 8-bit moments
    # cut that to ~4 GB. Measured 2026-09-01 while fitting Qwen3.5-2B on two 5090s that
    # were already sharing ~7 GB with another tenant.
    # NOTE: `adamw_8bit` DOES NOT COMPOSE WITH `--fsdp`. Measured 2026-09-01 on
    # Qwen3.5-2B: epoch 1 trains fine, then checkpoint saving dies inside
    # `torch/distributed/fsdp/_optim_utils.py::_convert_all_state_info` with
    # "size of tensor a (254280704) must match tensor b (254278656)" — a 2,048-element
    # gap, exactly the hidden size, because bitsandbytes' 8-bit state does not flatten to
    # the shape FSDP's consolidation expects. Use one or the other. FSDP already shards
    # optimizer state across ranks, which is the memory saving 8-bit was there for.
    p.add_argument("--optim", default="adamw_torch",
                   choices=("adamw_torch", "adamw_8bit"))
    # DDP replicates the whole model per GPU, so it buys throughput and NOT capacity.
    # FSDP shards parameters, gradients and optimizer state across ranks, which is what a
    # model that does not fit on one card needs. Empty by default so the 0.8B builds keep
    # their exact (single-GPU, DDP-free) behaviour.
    p.add_argument("--fsdp", default="", help='e.g. "full_shard auto_wrap"')
    # FSDP's auto-wrap reads the model's `_no_split_modules`, which for this
    # vision-language config is {Qwen3_5VisionBlock, Qwen3_5DecoderLayer}. We load the
    # TEXT TOWER only, so the vision class does not exist and auto-wrap dies with
    # "Could not find the transformer layer class Qwen3_5VisionBlock in the model".
    # Name the text decoder layer explicitly instead.
    p.add_argument("--fsdp-layer-cls", default="Qwen3_5DecoderLayer")
    # Write ONLY model weights in a checkpoint, skipping optimizer/scheduler state.
    # Required to combine `--optim adamw_8bit` with `--fsdp`: 8-bit moments are needed for
    # memory (fp32 Adam OOMs even sharded — measured, step 234 of 592) but FSDP's optimizer
    # consolidation cannot flatten them (crash at epoch 1's save). Skipping that
    # consolidation resolves both, and costs nothing here: this script reloads the best
    # checkpoint by copying its FILES, never by resuming an optimizer.
    p.add_argument("--save-only-model", action="store_true")
    # Load through unsloth's FastLanguageModel instead of AutoModelForCausalLM. Everything
    # downstream -- dataset build, completion-only masking, Trainer, save path -- is
    # unchanged, so an unsloth run stays comparable to the plain-Trainer builds.
    #
    # CLAUDE.md claimed unsloth "cannot train" Qwen3.5. RETRACTED 2026-09-01: that is true
    # only for the VISION-LANGUAGE repo, where the model surfaces as a processor and TRL
    # reads `eos_token` as the literal '<EOS_TOKEN>'. Against
    # `principled-intelligence/Qwen3.5-2B-text-only` (architectures Qwen3_5ForCausalLM),
    # FastLanguageModel loads it with the CORRECT eos_token '<|im_end|>' (248046) and
    # reports "bfloat16 full finetuning which cuts memory usage by 50%" -- which is the
    # constraint that forced FSDP, adamw_8bit and save_only_model onto the plain path.
    p.add_argument("--unsloth", action="store_true")
    # LoRA (bf16, NOT 4-bit -- unsloth's Qwen3.5 docs mark QLoRA "not recommended" for
    # this family). Requires --unsloth. Adapters are merged to 16-bit at save time so the
    # GGUF export path is identical to a full fine-tune's.
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    args = p.parse_args(argv)

    # unsloth patches transformers on import, so it MUST come first when enabled.
    if args.unsloth:
        from unsloth import FastLanguageModel

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    # LoRA WITHOUT unsloth, via peft on the plain-Trainer path. This exists because the
    # training path itself was measured to matter enormously: the same recipe at the same
    # size gives 3/3 G3 PASS on (VL repo + plain Trainer) and 0/3 FAIL with density -1.319
    # on (text-only repo + unsloth). Anything meant to be comparable to `v5`/`v7` must run
    # on the plain-Trainer path. unsloth cannot load the VL repo at all -- it routes to the
    # vision processor and tries to parse the prompt as a base64 image.
    if args.lora and not args.unsloth:
        from peft import LoraConfig, get_peft_model

    if args.unsloth:
        preloaded, tok = FastLanguageModel.from_pretrained(
            args.model, max_seq_length=args.max_len, load_in_4bit=False,
            full_finetuning=not args.lora, dtype=None,
        )
        if args.lora:
            preloaded = FastLanguageModel.get_peft_model(
                preloaded, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                bias="none", use_gradient_checkpointing="unsloth", random_state=args.seed,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            )
    else:
        preloaded = None
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

    model = preloaded if preloaded is not None else AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    if args.lora and not args.unsloth:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
        model.print_trainable_parameters()
    # Always on. Tried disabling it for the peft LoRA path to recover the ~35% it costs
    # (plain Trainer is ~7x slower than unsloth: 7h35m vs 1h for 1,818 steps) -- it OOMs
    # anyway. Trainable-parameter count is irrelevant here: the memory is dominated by the
    # 248k-vocab logits tensor and its gradient, which LoRA does not shrink.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=10, bf16=True, seed=args.seed, optim=args.optim,
        fsdp=args.fsdp,
        save_only_model=args.save_only_model,
        fsdp_config=(
            {"transformer_layer_cls_to_wrap": [args.fsdp_layer_cls]} if args.fsdp else None
        ),
        report_to=[], eval_strategy="epoch" if valid_ds else "no",
        per_device_eval_batch_size=args.batch_size,
        # Keep the BEST epoch, not the last. Measured on both Qwen runs: eval loss
        # bottomed at epoch 2 (0.877 / 0.829) and ROSE at epoch 3 (0.959 / 0.913), so
        # `save_strategy="no"` was silently shipping the overfitted checkpoint and
        # discarding the better one. Any tuning conclusion drawn from the epoch-3 model
        # was measuring overfitting as much as the change under test.
        save_strategy="epoch" if valid_ds else "no",
        save_total_limit=2,
        # Disabled under `--save-only-model`: transformers refuses FSDP +
        # save_only_model + load_best_model_at_end together. No loss — this flag is
        # measured NOT to work on this architecture anyway (see the save block below),
        # and `metric_for_best_model` still populates `state.best_model_checkpoint`,
        # which is what the save block actually reads.
        load_best_model_at_end=bool(valid_ds) and not args.save_only_model,
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

    # DO NOT trust `load_best_model_at_end` here. Measured 2026-09-01 on transformers
    # 5.5.0: it runs, emits "There were missing keys in the checkpoint model loaded"
    # listing EVERY weight, and leaves the last-epoch weights in memory. The cause is a
    # key-prefix mismatch -- `Qwen3_5ForConditionalGeneration` saves
    # `model.language_model.*` while the reload looks for `model.*` -- so nothing matches
    # and the warning is the only symptom.
    #
    # It failed silently for every v1.0 checkpoint: v5, v6 and v7 `final` are each
    # byte-identical to their LAST checkpoint and differ from their best one, despite
    # `trainer_state.json` correctly naming the best. Eval loss rises at epoch 3 on every
    # run (v7: 0.7790 -> 0.7712 -> 0.8590), so every shipped checkpoint was past its
    # minimum -- including the retrain that was recorded as having "ruled out"
    # overfitting as the cause of the real-ASR regression.
    #
    # Copy the best checkpoint's files directly instead. Verified by comparing a tensor
    # that actually moves during training (an mlp weight); `model.norm.weight` is a poor
    # discriminator because it barely changes between epochs.
    # A LoRA run must be MERGED before export: `convert_hf_to_gguf.py` reads full
    # weights, not adapters. Merging here keeps the downstream path (export_gguf.sh ->
    # quantize -> serve) byte-identical to a full fine-tune's, so a LoRA checkpoint and an
    # FFT checkpoint stay comparable through evaluation.
    if args.lora and not args.unsloth:
        # peft path: merge adapters into the base weights so `export_gguf.sh` sees a
        # normal checkpoint, exactly as the unsloth path's save_pretrained_merged does.
        best_ck = getattr(trainer.state, "best_model_checkpoint", None)
        if best_ck and Path(best_ck).is_dir():
            model.load_adapter(best_ck, adapter_name="best")
            model.set_adapter("best")
            print(f"[train] merging BEST adapter ({Path(best_ck).name})", file=sys.stderr)
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(str(final))
        tok.save_pretrained(str(final))
        print(f"[train] saved merged peft LoRA -> {final}", file=sys.stderr)
        return 0

    if args.lora:
        best_ck = getattr(trainer.state, "best_model_checkpoint", None)
        if best_ck and Path(best_ck).is_dir():
            model.load_adapter(best_ck, adapter_name="best")
            model.set_adapter("best")
            print(f"[train] merging BEST adapter ({Path(best_ck).name})", file=sys.stderr)
        model.save_pretrained_merged(str(final), tok, save_method="merged_16bit")
        print(f"[train] saved merged LoRA -> {final}", file=sys.stderr)
        return 0

    best = getattr(trainer.state, "best_model_checkpoint", None)
    if best and Path(best).is_dir():
        final.mkdir(parents=True, exist_ok=True)
        for f in Path(best).iterdir():
            if f.is_file() and f.name not in {"optimizer.pt", "scheduler.pt", "rng_state.pth"}:
                shutil.copy2(f, final / f.name)
        print(f"[train] saved BEST ({Path(best).name}) -> {final}", file=sys.stderr)
    else:
        trainer.save_model(str(final))
        print(f"[train] no best checkpoint recorded; saved LAST -> {final}", file=sys.stderr)
    tok.save_pretrained(str(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
