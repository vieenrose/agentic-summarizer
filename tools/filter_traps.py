#!/usr/bin/env python3
"""Trap post-filter for trace files (CLAUDE.md §7.6 discipline).

The judge filter approves trap bullets when they are phrased as claims the transcript
literally supports ("Coffee machine discussion [150]" — the trap line did raise it).
But the G1 screen requires the trap topic to stay OUT of the notes: it was raised and
explicitly dropped. A trace that teaches trap-reporting teaches the screen's exact
failure. This filter rewrites trace records for synthetic meetings, dropping accepted
ops whose text mentions the meeting's planted trap terms (deterministic from the
meeting id's variant rotation, same as voxsum.synth).

    python tools/filter_traps.py data/traces_v2/K_en_b128.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.synth import _TRAPS  # noqa: E402


def trap_terms(meeting_id: str) -> tuple[str, ...]:
    if not meeting_id.startswith("synth-"):
        return ()
    variant = int(meeting_id.rsplit("-", 1)[1])
    return _TRAPS[variant % len(_TRAPS)]


def mentions(text: str, terms: tuple[str, ...]) -> bool:
    t = text.casefold()
    return any(term.casefold() in t for term in terms)


def filter_file(path: Path) -> int:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    dropped = 0
    out = []
    for r in rows:
        terms = trap_terms(r["meeting"])
        if not terms:
            out.append(r)
            continue
        # Drop accepted ops (target lines) that mention the trap; keep everything else.
        lines = [l for l in r["target"].splitlines() if l.strip()]
        kept = [l for l in lines if not mentions(l, terms)]
        dropped += len(lines) - len(kept)
        r["target"] = "\n".join(kept)
        r["trap_filtered"] = len(lines) - len(kept)
        out.append(r)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    return dropped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    args = p.parse_args(argv)
    total = 0
    for path in args.files:
        n = filter_file(path)
        total += n
        print(f"[traps] {path.name}: dropped {n} trap-mentioning ops")
    print(f"[traps] total {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
