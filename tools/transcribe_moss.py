#!/usr/bin/env python3
"""Transcribe meeting recordings with MOSS-Transcribe-Diarize 0.9B -> transcript v1.

    python tools/transcribe_moss.py data/audio/*.wav --out data/transcripts

Writes `<stem>.txt` (v1 transcript) plus `<stem>.moss.txt` (raw model output, kept so a
format-conversion bug can be re-fixed without re-running the GPU).

MOSS handles up to ~90 min in a single pass; longer recordings should be split on
silence beforehand and the parts offset (--offset) so timestamps stay absolute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.ingest_moss import moss_to_v1  # noqa: E402

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("audio", nargs="+", type=Path, help="audio/video files")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--hotwords", default=None, help="comma-separated domain terms")
    p.add_argument(
        "--merge-gap", type=float, default=2.0, help="same-speaker merge gap (s); 0 disables"
    )
    p.add_argument("--drop-events", action="store_true", help="strip (laughter)/(applause) markers")
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--dtype", default="bfloat16", help="bf16 on Ampere+; Blackwell is fine")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=getattr(torch, args.dtype),
        device_map="auto",
    ).eval()

    instruction = "Transcribe the audio with speaker labels and timestamps."
    if args.hotwords:
        instruction += f" Hotwords: {args.hotwords}."

    for path in args.audio:
        print(f"[moss] {path}", flush=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(path)},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        raw = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

        (args.out / f"{path.stem}.moss.txt").write_text(raw, encoding="utf-8")
        v1 = moss_to_v1(
            raw,
            merge_gap=None if args.merge_gap == 0 else args.merge_gap,
            drop_events=args.drop_events,
        )
        target = args.out / f"{path.stem}.txt"
        target.write_text(v1, encoding="utf-8")
        print(f"[moss] -> {target} ({len(v1.splitlines())} lines)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
