#!/usr/bin/env python3
"""Traces -> SFT dataset for FunctionGemma-270M (PLAN.md §3).

Reads the JSONL from `gen_traces.py` and emits training samples in the student's own
format. Three decisions are load-bearing:

**1. Targets are FunctionGemma calls, not the text grammar.** The teacher emits text ops
because that is what it does naturally; the student emits
`<start_function_call>call:ADD{...}<end_function_call>` because that is what it was
post-trained to do. `translate_target` converts, so the teacher's readability costs the
student nothing.

**2. Completion-only masking.** `prompt` and `completion` are kept separate so the loss
covers op tokens only. Training on the prompt teaches the model to reproduce transcripts —
40 chunks x 2k tokens of them — which is not the task.

**3. NOP is kept and its share is reported.** Content-poor chunks are a large fraction of
real meetings; a student that never saw NOP hallucinates ops on them, and one that saw only
NOP collapses (GT1's < 10% bound). The ratio is a number to check, not to assume.

    python train/build_sft.py data/traces/train.jsonl --out data/sft/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.ops import Add, Cmp, Del, Nop, Title, Upd, parse_ops  # noqa: E402
from voxsum.prompts import PROMPT_VERSION, system_prompt  # noqa: E402
from voxsum.transcript import sec_to_clock  # noqa: E402

__all__ = ["build_sample", "translate_target"]


def _call(name: str, **args: str) -> str:
    body = ",".join(f"{k}:<escape>{v}<escape>" for k, v in args.items() if v != "")
    return f"<start_function_call>call:{name}{{{body}}}<end_function_call>"


def translate_target(target: str) -> str:
    """Text-grammar ops -> FunctionGemma calls.

    CMP is emitted as its constituent ADDs: a single call cannot carry a variable-length
    bullet list, and reproducing the section wholesale is the same edit either way.
    """
    calls: list[str] = []
    for op in parse_ops(target):
        match op:
            case Nop():
                calls.append(_call("NOP"))
            case Title(title):
                calls.append(_call("TITLE", title=title))
            case Add(section, bullet, anchor):
                calls.append(
                    _call(
                        "ADD",
                        section=section,
                        bullet=bullet,
                        anchor=sec_to_clock(anchor) if anchor is not None else "",
                    )
                )
            case Upd(section, prefix, bullet, anchor):
                calls.append(
                    _call(
                        "UPD",
                        section=section,
                        prefix=prefix,
                        bullet=bullet,
                        anchor=sec_to_clock(anchor) if anchor is not None else "",
                    )
                )
            case Del(section, prefix):
                calls.append(_call("DEL", section=section, prefix=prefix))
            case Cmp(section, bullets):
                for b in bullets:
                    calls.append(
                        _call(
                            "ADD",
                            section=section,
                            bullet=b.text,
                            anchor=sec_to_clock(b.anchor) if b.anchor is not None else "",
                        )
                    )
    return "\n".join(calls)


def build_sample(record: dict, *, declarations: bool = True) -> dict:
    """One trace step -> one SFT sample with prompt and completion split."""
    lang = record.get("lang", "en")
    system = system_prompt(lang, declarations=declarations)
    completion = translate_target(record["target"])
    return {
        "meeting": record.get("meeting"),
        "lang": lang,
        "step": record.get("step"),
        "prompt_version": PROMPT_VERSION,
        "system": system,
        "prompt": record["user"],
        "completion": completion,
        "is_nop": record.get("is_nop", False),
        "has_revision": any(
            op in completion for op in ("call:UPD", "call:DEL")
        ),
        "prompt_tokens": record.get("prompt_tokens"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("traces", nargs="+", type=Path, help="JSONL from gen_traces.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--valid-out", type=Path, default=None, help="held-out slice")
    p.add_argument("--valid-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-nop-frac",
        type=float,
        default=0.4,
        help="downsample NOP steps above this share (0 disables)",
    )
    p.add_argument(
        "--no-declarations",
        action="store_true",
        help="omit FunctionGemma declarations from SYS (text-grammar student variant)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)

    samples: list[dict] = []
    versions: set[str] = set()
    for path in args.traces:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            versions.add(record.get("prompt_version", "?"))
            sample = build_sample(record, declarations=not args.no_declarations)
            if sample["completion"]:
                samples.append(sample)

    if len(versions) > 1:
        print(
            f"REFUSING: traces mix prompt versions {sorted(versions)}. Training across a "
            "prompt change makes the run incomparable (CLAUDE.md §7.8).",
            file=sys.stderr,
        )
        return 2

    # NOP downsampling: keep it represented without letting it dominate.
    if args.max_nop_frac > 0:
        nops = [s for s in samples if s["is_nop"]]
        others = [s for s in samples if not s["is_nop"]]
        cap = int(len(others) * args.max_nop_frac / max(1 - args.max_nop_frac, 1e-6))
        if len(nops) > cap:
            rng.shuffle(nops)
            dropped = len(nops) - cap
            nops = nops[:cap]
            print(f"[sft] downsampled NOP steps: dropped {dropped}")
        samples = others + nops

    rng.shuffle(samples)
    valid: list[dict] = []
    if args.valid_out and args.valid_frac > 0:
        # Split by meeting, not by step: sibling steps share STATE, so a step-level split
        # leaks the answer for a held-out step into training.
        meetings = sorted({s["meeting"] for s in samples})
        rng.shuffle(meetings)
        n_valid = max(1, round(len(meetings) * args.valid_frac))
        held = set(meetings[:n_valid])
        valid = [s for s in samples if s["meeting"] in held]
        samples = [s for s in samples if s["meeting"] not in held]
        print(f"[sft] held out {len(held)} meetings ({len(valid)} steps) for validation")

    for target, rows in ((args.out, samples), (args.valid_out, valid)):
        if target is None or not rows:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(
        ("nop" if s["is_nop"] else "revision" if s["has_revision"] else "add") for s in samples
    )
    total = max(len(samples), 1)
    print(f"[sft] {len(samples)} train samples -> {args.out}")
    for key in ("add", "revision", "nop"):
        print(f"[sft]   {key:9s} {counts[key]:6d}  {counts[key] / total:5.1%}")
    if counts["revision"] / total < 0.15:
        print(
            "[sft] WARNING: revision share below 15%. UPD/DEL is the behaviour GT3 pays "
            "for; add revision-dense synthetic meetings (voxsum.synth) before training.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
