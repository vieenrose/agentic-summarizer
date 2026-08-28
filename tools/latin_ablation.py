"""Ablation: does CODE-SWITCHING (Latin ratio), not ASR noise, break curation?

    python tools/latin_ablation.py data/ly_pilot/op-msf-19.txt --out-dir data/ablation

Motivation, measured. The Phase 3 slices differ from the training distribution on three
axes, and only one of them moves far:

    slice              punct/100   filler/100   latin/100   result
    MeetingBank zh          9.27         0.59        1.73   (trained on)
    LY committee            7.42         1.59        4.82   25% NOP, curates fine
    zh-TW tech podcasts     5.67         2.57       10.30   100% NOP, total collapse

Latin is 6x the training rate in the podcasts, and one of them was already annotated by
the prior project as "THE FLIP CASE -- 22.3% Latin, both checkpoints answer in English".
That reframes the podcast collapse: it may not be an ASR-noise finding (risk 5) at all,
but a code-switching one -- which is a different risk with a different fix.

This script holds domain, speakers and language fixed by starting from an LY transcript
that the agent DOES curate, and raising only the Latin ratio, by substituting common
Chinese terms with the English a bilingual Taiwanese speaker would actually code-switch
to. That is how the podcasts read; it is not random noise injection.

The substitutions deliberately preserve meaning, so a drop in curation cannot be blamed
on the text becoming less informative.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.transcript import parse_transcript  # noqa: E402

#: Ordered most-to-least common so lower levels substitute the highest-frequency terms
#: first — mimicking how code-switching actually distributes, rather than uniformly.
SUBSTITUTIONS: list[tuple[str, str]] = [
    ("委員會", "committee"),
    ("會議", "meeting"),
    ("資料", "data"),
    ("政府", "government"),
    ("報告", "report"),
    ("平台", "platform"),
    ("計畫", "project"),
    ("系統", "system"),
    ("團隊", "team"),
    ("議題", "issue"),
    ("提案", "proposal"),
    ("預算", "budget"),
    ("國際", "international"),
    ("公開", "open"),
    ("參與", "participate"),
    ("討論", "discuss"),
    ("決議", "resolution"),
    ("政策", "policy"),
]


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(r"[A-Za-z]", text)) / len(text)


def apply_level(text: str, n_terms: int) -> str:
    """Substitute the first `n_terms` entries. More terms -> higher Latin ratio."""
    out = text
    for zh, en in SUBSTITUTIONS[:n_terms]:
        out = out.replace(zh, en)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--levels", type=int, nargs="*", default=[0, 4, 9, 18])
    args = p.parse_args(argv)

    text = args.source.read_text(encoding="utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.source.stem

    for n in args.levels:
        variant = apply_level(text, n)
        # Ratio over utterance text only, matching how the slices above were measured
        # (speaker labels are not content and would dilute it).
        body = " ".join(u.text for u in parse_transcript(variant))
        ratio = latin_ratio(body)
        dst = args.out_dir / f"{stem}-L{n:02d}.txt"
        dst.write_text(variant, encoding="utf-8")
        print(f"[ablation] {dst.name}  terms={n:2d}  latin={ratio * 100:5.2f}/100", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
