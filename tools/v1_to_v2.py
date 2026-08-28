"""Convert a v1 timestamped transcript to this project's format v2.

    python tools/v1_to_v2.py data/asr_pilot_v1/*.v1.txt --out-dir data/asr_pilot

v1 (the prior project, and the `asr-transcripts-2026-08-16` slice) is
`[m:ss] SPEAKER: text`; v2 (SPEC §2) is `SPEAKER: text` with no timestamps. The only
transformation is dropping the anchor.

**Fails loudly on a line it does not recognise**, rather than passing it through. A
silently mis-parsed line becomes a `UNK:` utterance downstream — `transcript.parse_line`
is deliberately total and will not raise — so the defect would surface as a quietly
worse chunk rather than an error. This is eval input for a gate; it must be exactly
right or not run at all.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: `[m:ss]` or `[h:mm:ss]`, then a mandatory speaker. Anchored at both ends so a line
#: with trailing junk after the text is not silently accepted.
V1_LINE = re.compile(r"^\[\s*\d+:\d{2}(?::\d{2})?\s*\]\s*(?P<speaker>[^:]{1,40}):\s(?P<text>.*)$")


class ConversionError(ValueError):
    """A line that is not valid v1. Refusing beats emitting a UNK: utterance."""


def convert_line(line: str, *, lineno: int, path: str) -> str | None:
    """Returns the v2 line, or None for a blank line. Raises on anything else."""
    if not line.strip():
        return None
    m = V1_LINE.match(line.rstrip("\n"))
    if not m:
        raise ConversionError(f"{path}:{lineno}: not a v1 line: {line.rstrip()[:80]!r}")
    speaker = m.group("speaker").strip()
    text = m.group("text").strip()
    if not speaker:
        raise ConversionError(f"{path}:{lineno}: empty speaker")
    if not text:
        raise ConversionError(f"{path}:{lineno}: empty text")
    return f"{speaker}: {text}"


def convert(text: str, *, path: str = "<input>") -> str:
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        converted = convert_line(line, lineno=i, path=path)
        if converted is not None:
            out.append(converted)
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for src in args.inputs:
        try:
            v2 = convert(src.read_text(encoding="utf-8"), path=str(src))
        except ConversionError as exc:
            print(f"[v1->v2] REFUSED {src}: {exc}", file=sys.stderr)
            return 1
        # "<name>.v1.txt" -> "<name>.txt"
        stem = src.name[: -len(".v1.txt")] if src.name.endswith(".v1.txt") else src.stem
        dst = args.out_dir / f"{stem}.txt"
        dst.write_text(v2, encoding="utf-8")
        print(f"[v1->v2] {src.name} -> {dst} ({len(v2.splitlines())} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
