"""GRPO on the `SYNTHESIZE` step, against the harness's own deterministic reward.

    python tools/grpo_synth.py --init runs/v21-s1/checkpoint-614 \\
        --pool data/staging/sft_pool_v18.jsonl --out runs/rl-v1 \\
        --generations 8 --meetings 160

**Why RL, after six SFT pools.** SFT imitates targets, so the student inherits the teacher's
dispositions — and the teacher's length prior is measured IMMOVABLE (a 28-entry journal gives
664 characters, a 34-entry one 680, and doubling `max_tokens` returns byte-identical output).
SPEC §5.2.5 identifies rendering DENSITY as the binding constraint on G5 retention, and density
is exactly what the teacher caps. RL optimises against the environment, so it is not bounded by
what the teacher would have written.

**Why synthesis only.** One generation per rollout instead of ~15 for a full meeting, and the
deficit lives here: the journal already guarantees every recorded point REACHES the prompt
(SPEC §4.1 v1.1), so the open question is whether the model USES it. Reading-step RL needs
credit assignment across steps and is a separate, later experiment.

**The prompts are REPLAYED, not taken from the pool's stored rows.** A stored synthesis prompt
is whatever the pool happens to contain; replaying each meeting's gold ops through `apply_ops`
and rendering with `build_synth_prompt` reproduces the state the harness really presents at
serving time, including the journal. Same argument as `gen_journal_synth.py`: the input must be
correct by construction, not by whatever was archived.

**Train and eval meetings are disjoint by construction.** Prompts come from the training pool's
meetings; every gate is measured on `data/heldout_zh`, which shares none of them. Optimising a
reward that `arcsum-eval` also reports would otherwise be training on the test set.

**The reward is `arcsum.rl.reward.score`** — hard constraints (grounding, markup, language,
length) times a bounded objective (retention), never a weighted sum. See that module for why a
penalty term instead of a constraint would be reward-hacked immediately; this project has the
receipts.
"""

from __future__ import annotations

# unsloth must be imported before trl/transformers/peft or its patches do not apply.
import unsloth  # noqa: F401  # isort: skip

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# trl 0.24 reads `transformers.utils.hub.TRANSFORMERS_CACHE`, removed in transformers 5.
# Shimmed rather than pinning transformers back: every checkpoint in this project was trained
# on 5.5.0, and downgrading to satisfy an import would invalidate all of them.
import transformers.utils.hub as _hub

if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    from huggingface_hub import constants as _hf_constants

    _hub.TRANSFORMERS_CACHE = _hf_constants.HF_HUB_CACHE

from arcsum.chunker import Chunk
from arcsum.guards import apply_ops
from arcsum.memory import Memory
from arcsum.prompts import build_synth_prompt, synth_system_prompt
from arcsum.prose import finalize
from arcsum.rl import score
from arcsum.tokens import heuristic_token_len
from arcsum.toolcalls import parse_tool_calls
from arcsum.transcript import Utterance


def replay(steps: list[dict]) -> Memory:
    mem = Memory(token_len=heuristic_token_len)
    for r in sorted(steps, key=lambda x: int(x["step"])):
        ops = parse_tool_calls(r["completion"])
        if not ops:
            continue
        apply_ops(mem, ops, Chunk(index=int(r["step"]),
                                  utterances=[Utterance("S1", "x")], tokens=10),
                  lang_check=False)
    return mem


def build_dataset(pool: Path, limit: int, min_entries: int):
    from datasets import Dataset

    rows = [json.loads(ln) for ln in pool.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if "CHUNK:" in r["prompt"] and "tool_call" in r["completion"]:
            by[r["meeting"]].append(r)

    system = synth_system_prompt()
    out: list[dict] = []
    for meeting in sorted(by):
        mem = replay(by[meeting])
        view = build_synth_prompt(mem)
        entries = [ln[2:].strip() for ln in view.splitlines() if ln.startswith("- ")]
        # A meeting with almost nothing recorded gives a reward with no gradient to speak of —
        # retention over 1-2 entries is 0.0 or 1.0 and carries no ranking signal within a group.
        if len(entries) < min_entries:
            continue
        out.append({
            "prompt": [{"role": "system", "content": system},
                       {"role": "user", "content": view}],
            "entries": entries,
            "memory_view": view,
        })
        if limit and len(out) >= limit:
            break
    return Dataset.from_list(out)


def make_reward(max_tokens: int, log_every: int = 10):
    """`reward_funcs` callable in trl's shape: extra dataset columns arrive as kwargs.

    **Reward variance is logged, not assumed.** GRPO normalises advantage WITHIN a group, so
    if every generation in a group scores the same the advantage is exactly zero and training
    is a very expensive no-op — which is what the first smoke run did: loss collapsed to 3e-5
    and grad_norm to 0.02 because the checkpoint's `generation_config.json` carries no
    `do_sample`, so all four "samples" were the same greedy text. Nothing failed; it just
    learned nothing. This project has repeatedly been bitten by instruments that report a
    plausible number for the wrong reason, so the group's spread is printed.
    """
    state = {"calls": 0}

    def reward_retention(completions, entries=None, memory_view=None, **_):
        scores: list[float] = []
        refusals: collections.Counter = collections.Counter()
        for i, completion in enumerate(completions):
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            # Score the FINALIZED prose, which is what the product emits and what every gate
            # measures. Rewarding the raw generation would let the policy earn credit for text
            # `prose.finalize` strips.
            clean = finalize(text, token_len=heuristic_token_len).text
            r = score(clean, entries[i], memory_view[i], max_tokens=max_tokens)
            scores.append(r.reward)
            if r.refused:
                refusals[r.refused] += 1
        state["calls"] += 1
        if state["calls"] % log_every == 1:
            spread = max(scores) - min(scores) if scores else 0.0
            mean = sum(scores) / len(scores) if scores else 0.0
            print(f"[grpo] group {state['calls']}: mean {mean:.3f} spread {spread:.3f} "
                  f"{'ZERO-VARIANCE (no gradient)' if spread == 0 else ''} "
                  f"refusals {dict(refusals)}", file=sys.stderr, flush=True)
        return scores

    return reward_retention


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", type=Path, required=True, help="SFT checkpoint to start from")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--generations", type=int, default=8,
                   help="group size; GRPO's advantage is normalised WITHIN a group, so this "
                        "is what supplies the ranking signal")
    p.add_argument("--meetings", type=int, default=0, help="0 = all eligible")
    p.add_argument("--min-entries", type=int, default=6)
    p.add_argument("--max-tokens-budget", type=int, default=1000, help="SPEC §3 output budget")
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty toward the SFT policy")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for ROLLOUTS; the deployed decode stays greedy. **Not 1.0.** "
             "Measured on the first run: at 1.0, 14 of 35 logged groups scored ZERO VARIANCE "
             "because every sample in them failed a hard constraint (mostly ungrounded "
             "specifics), and GRPO normalises advantage within a group — so 40% of the compute "
             "produced no gradient at all. A lower temperature makes samples differ in QUALITY "
             "rather than in VALIDITY, which is the difference the advantage is meant to rank.",
    )
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scale-rewards",
        default="group",
        choices=("group", "none", "batch"),
        help="GRPO divides the advantage by the GROUP standard deviation; Dr. GRPO shows that "
             "division is what makes the objective length- and difficulty-biased, and drops it "
             "(`none`). That matters here because the reward already contains a length "
             "constraint, so a length bias in the optimiser is not neutral. Default kept at "
             "`group` so `rl-v3` stays reproducible.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="build the dataset and score the reward on it, then stop — verifies "
                        "the prompts and reward before spending GPU time")
    args = p.parse_args(argv)

    ds = build_dataset(args.pool, args.meetings, args.min_entries)
    print(f"[grpo] {len(ds)} synthesis prompts "
          f"(>= {args.min_entries} journal entries)", file=sys.stderr)
    if not len(ds):
        print("[grpo] REFUSED: no eligible meetings", file=sys.stderr)
        return 1

    if args.dry_run:
        import statistics as st

        n = [len(r) for r in ds["entries"]]
        print(f"[grpo] entries per prompt: median {st.median(n):.0f} max {max(n)}",
              file=sys.stderr)
        return 0

    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        str(args.init), max_seq_length=args.max_len, load_in_4bit=False,
        full_finetuning=True, dtype=None,
    )
    # trl 0.24 pokes attributes that transformers 4 put on every model and transformers 5
    # does not: `warnings_issued` (a dict it flips to suppress a token-estimate warning) and
    # `add_model_tags` (a Hub-metadata helper). Neither affects training; both abort __init__.
    # Shimmed rather than pinning transformers back — every checkpoint here was trained on
    # 5.5.0, so downgrading to satisfy an import would invalidate all of them.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if not hasattr(model, "add_model_tags"):
        model.add_model_tags = lambda *a, **k: None

    # **Sampling must be forced on the MODEL's generation_config.** The exported checkpoints
    # carry no `do_sample`, so it defaults to greedy and every member of a GRPO group is the
    # same text — zero spread, zero advantage, zero learning, and no error anywhere. Setting
    # `temperature` in GRPOConfig alone did not override it.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0

    cfg = GRPOConfig(
        output_dir=str(args.out),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.generations,
        gradient_accumulation_steps=1,
        num_generations=args.generations,
        max_completion_length=args.max_new_tokens,
        max_prompt_length=args.max_len - args.max_new_tokens,
        beta=args.beta,
        temperature=args.temperature,
        scale_rewards=args.scale_rewards,
        seed=args.seed,
        # GRPO holds the policy AND a frozen reference model, plus `num_generations` long
        # sequences per step. On a 32 GB card shared with another tenant (~7 GB) that OOMs at
        # 6 generations x 1100 tokens; checkpointing trades compute for the activation memory.
        gradient_checkpointing=True,
        # **The 248k vocab is the memory constraint, not the 0.8B parameters.** GRPO needs a
        # per-token logprob over the full vocabulary for every generation, so the transient
        # logits tensor is `generations x seq_len x 248,044 x 4 B` — ~16 GB at 4 generations
        # of ~4k tokens, which OOMs a 32 GB card shared with another tenant. This is the same
        # constraint CLAUDE.md records for SFT ("248k vocab OOMs at batch 4; use batch 1").
        # Liger's fused loss computes it in chunks and never materialises the full tensor.
        use_liger_loss=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        bf16=True,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward(args.max_tokens_budget)],
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(args.out / "final"))
    print(f"[grpo] wrote {args.out / 'final'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
