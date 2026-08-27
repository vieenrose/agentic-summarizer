"""`arcsum-build-sft`: `arcsum-gen-traces` rows -> a train/valid SFT split (SPEC §4.2,
§8 risk 3).

    arcsum-build-sft traces1.jsonl traces2.jsonl --out-dir sft/ --valid-frac 0.1

Each input file is `arcsum-gen-traces`'s `--out` JSONL — rows already shaped like
`supervision.sft.SftSample`'s fields, so they decode into `SftSample` directly with no
intermediate schema. This tool owns only the pool-level operations `sft.py` defines:
refusing a mixed `prompt_version` pool, capping the NOP share, and splitting by
meeting (never by step, since sibling steps share state).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arcsum.supervision.sft import (
    DEFAULT_MAX_NOP_FRAC,
    SftSample,
    check_single_prompt_version,
    downsample_nop,
    drop_bearing_share,
    drop_share,
    nop_share,
    oversample_drop,
    split_by_meeting,
)


def load_samples(paths: list[Path]) -> list[SftSample]:
    samples = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                SftSample(
                    meeting=row["meeting"],
                    step=row["step"],
                    prompt_version=row["prompt_version"],
                    system=row["system"],
                    prompt=row["prompt"],
                    completion=row["completion"],
                    is_nop=row["is_nop"],
                )
            )
    return samples


def _write_jsonl(path: Path, samples: list[SftSample]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "meeting": s.meeting,
                        "step": s.step,
                        "prompt_version": s.prompt_version,
                        "system": s.system,
                        "prompt": s.prompt,
                        "completion": s.completion,
                        "is_nop": s.is_nop,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("traces", type=Path, nargs="+", help="arcsum-gen-traces JSONL output files")
    p.add_argument("--out-dir", type=Path, required=True, help="write train.jsonl/valid.jsonl here")
    p.add_argument("--valid-frac", type=float, default=0.1, help="fraction of MEETINGS held out")
    p.add_argument("--max-nop-frac", type=float, default=DEFAULT_MAX_NOP_FRAC)
    p.add_argument(
        "--target-drop-frac",
        type=float,
        default=0.0,
        help="duplicate DROP-bearing samples up to this share of the pool (0.0 = off). "
        "Motivation measured 2026-08-27: see supervision.sft.oversample_drop's docstring.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = load_samples(args.traces)
    if not samples:
        print("[build-sft] no samples found in the given trace files", file=sys.stderr)
        return 1

    try:
        prompt_version = check_single_prompt_version(samples)
    except ValueError as exc:
        print(f"[build-sft] {exc}", file=sys.stderr)
        return 1

    before_nop_share = nop_share(samples)
    samples = downsample_nop(samples, max_nop_frac=args.max_nop_frac, seed=args.seed)
    # AFTER downsample_nop: DROP-bearing samples are never NOPs, so oversampling them
    # first would inflate the non-NOP denominator the NOP cap solves against (see
    # oversample_drop's docstring).
    before_drop_frac = drop_bearing_share(samples)
    samples = oversample_drop(samples, target_drop_frac=args.target_drop_frac, seed=args.seed)
    train, valid = split_by_meeting(samples, valid_frac=args.valid_frac, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "train.jsonl", train)
    _write_jsonl(args.out_dir / "valid.jsonl", valid)

    print(
        f"[build-sft] prompt_version={prompt_version} "
        f"total={len(samples)} train={len(train)} valid={len(valid)} "
        f"nop_share before={before_nop_share:.3f} after={nop_share(samples):.3f} "
        f"drop_share={drop_share(samples)} "
        f"drop_bearing_frac before={before_drop_frac:.3f} after={drop_bearing_share(samples):.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
