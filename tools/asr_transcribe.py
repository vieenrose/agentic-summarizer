"""Transcribe zh-TW audio to format v2 with MOSS-Transcribe-Diarize (ASR + diarization).

    python tools/asr_transcribe.py /tmp/ly_audio.wav --out data/ly_asr/op-audit.txt

Why this exists: SPEC §8 risk 5 (the train/deploy ASR gap) is the one gap the corpus
structurally cannot measure — MeetingBank's audio is English, and every zh-TW transcript
in this project so far is either machine-translated or stenographic. Closing it needs
in-domain zh-TW AUDIO put through a real ASR + diarization stack, which is what this
does. The output is format v2 directly, so it drops straight into the same harness the
gate numbers were measured with.

MOSS emits `[S01]`/`[S02]` speaker labels natively, which map onto v2's mandatory
speaker field without inventing attribution. The helper package the model card imports
(`moss_transcribe_diarize`) is not on PyPI, so the documented four-step flow is done
directly here: apply_chat_template -> load audio -> processor -> generate.

**Long audio is chunked with overlap.** The model advertises single-pass inference up to
90 minutes, but that is a memory claim, not a promise about a 2048-token output budget;
a long meeting silently truncates. Chunking with overlap and stitching is the honest way
to keep the tail of a meeting, which is exactly where decisions land (SPEC §4.1's
rationale for never head-truncating).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

#: The model card's default prompt, verbatim. It is Simplified Chinese in the card and
#: is left exactly as published -- it steers the OUTPUT format, and the transcript
#: content follows the audio's own language.
PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
    "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

#: MOSS emits segments INLINE on one line, not one per line:
#:     [0.00][S01]会同意[0.64][1.12][S01]我们知道了[2.84][2.84][S02]同意没有[3.44]
#: i.e. `[start][speaker]text[end]` repeated, with DECIMAL timestamps (`0.64`), not the
#: `m:ss` form. Parsing it line-wise finds nothing at all.
SEG_RE = re.compile(r"\[\d+(?:\.\d+)?\]\[(?P<spk>S\d{2,3})\](?P<text>.*?)(?=\[\d+(?:\.\d+)?\]|$)")


def to_v2(raw: str, *, to_traditional: bool = True) -> tuple[str, dict]:
    """MOSS output -> v2. Consecutive turns by the same speaker are merged, matching how
    the stenographic gazette records a turn (see tools/ly_odt_to_v2.py).

    **Converts Simplified -> Traditional by default.** Measured: MOSS transcribes zh-TW
    audio into SIMPLIFIED characters (会同意 / 我们 / 韩俊). This project is zh-TW only
    (SPEC §2), `lang.simplified_hits` treats simplified characters as a language-guard
    failure, and every training target is Traditional -- so leaving the ASR output
    Simplified would confound a script mismatch with the ASR-noise effect being measured.
    Conversion is a real part of the deploy path, not a cosmetic step.
    """
    turns: list[tuple[str, str]] = []
    for m in SEG_RE.finditer(raw):
        spk = m.group("spk")
        txt = " ".join(m.group("text").split())
        if not txt:
            continue
        if turns and turns[-1][0] == spk:
            turns[-1] = (spk, (turns[-1][1] + " " + txt).strip())
        else:
            turns.append((spk, txt))

    converted = False
    if to_traditional and turns:
        try:
            from opencc import OpenCC

            cc = OpenCC("s2twp")
            turns = [(s, cc.convert(t)) for s, t in turns]
            converted = True
        except ImportError:
            print(
                "[asr] WARNING: opencc not installed -- output stays SIMPLIFIED, which "
                "will trip the zh-TW language guard. Install `opencc` or pass "
                "--keep-simplified deliberately.",
                file=sys.stderr,
            )

    body = "\n".join(f"{s}: {t}" for s, t in turns) + "\n" if turns else ""
    return body, {
        "utterances": len(turns),
        "speakers": sorted({s for s, _ in turns}),
        "converted_to_traditional": converted,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk-seconds", type=int, default=240)
    p.add_argument("--overlap-seconds", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--raw-out", type=Path, default=None, help="also save the model's raw text")
    p.add_argument(
        "--keep-simplified",
        action="store_true",
        help="skip Simplified->Traditional conversion (default converts; MOSS emits "
        "simplified characters, which trip this project's zh-TW language guard)",
    )
    args = p.parse_args(argv)

    import librosa
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[asr] loading {MODEL_ID} on {device}", file=sys.stderr)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="auto")
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    wav, sr = librosa.load(str(args.audio), sr=16000, mono=True)
    total = len(wav) / sr
    step = args.chunk_seconds - args.overlap_seconds
    print(
        f"[asr] {total:.0f}s audio, {args.chunk_seconds}s chunks, {args.overlap_seconds}s overlap",
        file=sys.stderr,
    )

    raw_parts: list[str] = []
    start = 0.0
    idx = 0
    while start < total:
        seg = wav[int(start * sr) : int(min(start + args.chunk_seconds, total) * sr)]
        messages = [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": seg}, {"type": "text", "text": PROMPT}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, audio=[seg], sampling_rate=16000, return_tensors="pt")
        # Cast FLOATING-POINT tensors to the model dtype and leave integer tensors alone.
        # The feature extractor returns float32 while the model runs in bfloat16; feeding
        # mismatched features produced pure scaffold output ("[0.00][S01][2.00]..." with
        # no transcribed text at all) rather than an error. Integer ids/masks must NOT be
        # cast to a float dtype.
        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                moved[k] = (
                    v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                )
            else:
                moved[k] = v
        inputs = moved
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[-1] :]
        part = processor.decode(gen, skip_special_tokens=True)
        raw_parts.append(part)
        idx += 1
        print(f"[asr]   chunk {idx} @{start:.0f}s -> {len(part)} chars", file=sys.stderr)
        start += step

    raw = "\n".join(raw_parts)
    if args.raw_out:
        args.raw_out.write_text(raw, encoding="utf-8")

    body, stats = to_v2(raw, to_traditional=not args.keep_simplified)
    if stats["utterances"] < 3:
        print(f"[asr] REFUSED: only {stats['utterances']} utterances parsed", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(
        f"[asr] {args.out}  utterances={stats['utterances']} speakers={len(stats['speakers'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
