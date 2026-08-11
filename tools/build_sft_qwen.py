#!/usr/bin/env python3
"""SFT dataset for the TEXT-grammar student (Qwen3.5-0.8B promoted path, PLAN §4).

The traces store targets as text ops ("ADD DECISIONS - ... [m:ss]") — FunctionGemma
translate_target converts them to calls; Qwen is not a function-call model, so here the
raw text target is used verbatim with the text-grammar SYS (declarations=False). The
meeting/split logic mirrors tools/build_sft_v3.py (standard traces with a NOP cap, plus
the screen-structured b128 traces with active steps kept and NOPs lightly sampled).

    python tools/build_sft_qwen.py data/traces_v2 --out data/sft/qwen-train.jsonl \
        --valid-out data/sft/qwen-valid.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.prompts import PROMPT_VERSION, system_prompt  # noqa: E402

NOP_PER_B128_MEETING = 2


def build_sample(record: dict) -> dict:
    """Text grammar, no declarations: system + user + raw target."""
    lang = record.get("lang", "en")
    return {
        "meeting": record.get("meeting"),
        "lang": lang,
        "step": record.get("step"),
        "prompt_version": PROMPT_VERSION,
        "system": system_prompt(lang, declarations=False),
        "prompt": record["user"],
        "completion": record["target"],
        "is_nop": record.get("is_nop", False),
        "has_revision": any(op in record["target"] for op in ("UPD ", "DEL ")),
        "prompt_tokens": record.get("prompt_tokens"),
    }


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tracedir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--valid-out", type=Path, default=None)
    p.add_argument("--valid-frac", type=float, default=0.04)
    p.add_argument("--lang", choices=["en", "zh-TW"], default=None,
                  help="restrict to one language (per-language students, PLAN 0d)")
    p.add_argument("--max-nop-frac", type=float, default=0.28)
    p.add_argument(
        "--max-b128-frac",
        type=float,
        default=0.55,
        help="cap the b128 (synthetic screen-distribution) group's share of the final "
        "train set — the real/standard 2048-budget traces must not be diluted (micro "
        "inversion on a real meeting was traced to exactly that dilution)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    import json as _json

    manifest = _json.loads(
        (Path(__file__).resolve().parent.parent / "data/transcripts/manifest.json").read_text()
    )
    split_of = {row["meeting_id"]: row["split"] for row in manifest}

    def _is_train(r: dict) -> bool:
        return split_of.get(r.get("meeting"), "train") == "train" and (
            args.lang is None or r.get("lang") == args.lang
        )

    standard_all: list[dict] = []
    b128_all: list[dict] = []
    for path in sorted(args.tracedir.glob("*.jsonl")):
        if "b128" in path.name:
            b128_all.extend(load(path))
        else:
            standard_all.extend(load(path))
    dropped = sum(1 for r in standard_all + b128_all if not _is_train(r))
    standard = [r for r in standard_all if _is_train(r)]
    b128 = [r for r in b128_all if _is_train(r)]
    print(f"[qwen] dropped {dropped} eval-meeting steps; kept "
          f"{len(standard)} standard + {len(b128)} b128")

    versions = {r.get("prompt_version") for r in standard + b128}
    if len(versions) != 1:
        raise SystemExit(f"mixed prompt versions: {sorted(versions)}")

    nops = [r for r in standard if r["is_nop"]]
    others = [r for r in standard if not r["is_nop"]]
    cap = int(len(others) * args.max_nop_frac / max(1 - args.max_nop_frac, 1e-6))
    if len(nops) > cap:
        rng.shuffle(nops)
        nops = nops[:cap]
    standard = others + nops

    active = [r for r in b128 if not r["is_nop"]]
    nops = [r for r in b128 if r["is_nop"]]
    # Cap the b128 group's share: real-meeting (standard) traces must stay
    # well-represented or copy-don't-invent does not transfer to real transcripts.
    b128_target = max(int(len(standard) * args.max_b128_frac / max(1 - args.max_b128_frac, 1e-6)), 0)
    per_meeting: dict[str, list[dict]] = {}
    for r in nops:
        per_meeting.setdefault(r["meeting"], []).append(r)
    short_nops: list[dict] = []
    for meeting, rows in sorted(per_meeting.items()):
        if len(rows) > 20:
            continue  # long-meeting b128 NOPs collapse the student (v2/v3 lesson)
        rows = sorted(rows, key=lambda r: r["step"])
        n = min(NOP_PER_B128_MEETING, len(rows))
        idx = {round(i * (len(rows) - 1) / max(n - 1, 1)) for i in range(n)}
        short_nops += [rows[i] for i in sorted(idx)]
    b128 = active + short_nops
    if len(b128) > b128_target:
        rng.shuffle(b128)
        b128 = b128[:b128_target]
        print(f"[qwen] capped b128 group to {b128_target} ({args.max_b128_frac:.0%} cap)")

    samples = [build_sample(r) for r in standard + b128]
    samples = [s for s in samples if s["completion"].strip()]
    rng.shuffle(samples)

    valid: list[dict] = []
    if args.valid_out and args.valid_frac > 0:
        meetings = sorted({s["meeting"] for s in samples})
        rng.shuffle(meetings)
        n_valid = max(1, round(len(meetings) * args.valid_frac))
        held = set(meetings[:n_valid])
        valid = [s for s in samples if s["meeting"] in held]
        samples = [s for s in samples if s["meeting"] not in held]

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
    print(f"[qwen] {len(samples)} train samples -> {args.out}")
    for key in ("add", "revision", "nop"):
        print(f"[qwen]   {key:9s} {counts[key]:6d}  {counts[key] / total:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
