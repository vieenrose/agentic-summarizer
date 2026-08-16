"""#167 — deterministic faithfulness measurements that do NOT route through the
collapsed verifier (VoxSum round-5.6, their scoping).

Two tools:
1. `detect_inversions` — per bullet with a polarity word, compare its polarity
   against the transcript's polarity in the anchor neighbourhood (production
   evidence length, ±90s; the LATEST polarity word wins — reversal-aware). A
   mismatch is a candidate inversion. Zero model dependency, fails loudly.
2. `sweep_commitments` — sweep the transcript with the commitment lexicon and list
   the candidate lines + the matched token, so the hand-check (genuine commitment
   or filler?) is a 5-minute pass per meeting, then measure the fraction the notes
   captured (the omission number).

Both report NUMBERS, not pass/fail, and read the deployed renderer's notes file.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.guards import _polarity
from voxsum.highlight import commit_token, is_commit_line
from voxsum.transcript import parse_transcript, sec_to_clock

#: pull the polarity words into a finder
from voxsum import guards as _g

_POL_WORDS = re.compile(
    "|".join(re.escape(w) for w in (_g._NEGATIVE + _g._POSITIVE)),
    re.I,
)


def evidence_polarity(utterances, anchor: int, window: int = 90) -> int:
    """The latest polarity in the ±window neighbourhood of `anchor` (0 = none)."""
    lines = [u for u in utterances if abs(u.start - anchor) <= window]
    pol = 0
    for u in sorted(lines, key=lambda u: u.start):
        p = _polarity(u.text)
        if p != 0:
            pol = p  # the latest polarity word wins (reversal-aware)
    return pol


def detect_inversions(notes_text: str, utterances, lang: str = "en") -> list[dict]:
    """Find bullets whose polarity opposes the transcript's at their anchor."""
    out = []
    for line in notes_text.splitlines():
        line = line.strip()
        m = re.match(r"-\s+(.*?)\s+\[(\d+):(\d{2})(?::(\d{2}))?\]\s*$", line)
        if not m:
            continue
        bullet = m.group(1)
        pi = _polarity(bullet)
        if pi == 0:
            continue
        secs = int(m.group(2)) * 60 + int(m.group(3)) + (int(m.group(4) or 0) * 3600)
        ev = evidence_polarity(utterances, secs)
        if ev == -pi:
            out.append({"bullet": bullet, "anchor": secs,
                        "bullet_polarity": pi, "evidence_polarity": ev})
    return out


def sweep_commitments(utterances, lang: str = "en") -> list[dict]:
    """List commitment-lexicon hits with their matched token, for the hand-check."""
    out = []
    for u in utterances:
        if is_commit_line(u.text, lang):
            out.append({"start": u.start, "clock": sec_to_clock(u.start),
                        "token": commit_token(u.text, lang), "text": u.text[:80]})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("transcript", type=Path)
    p.add_argument("--notes", type=Path, default=None,
                   help="notes file (the deployed renderer's output); inversion check")
    p.add_argument("--lang", default="zh-TW")
    args = p.parse_args(argv)

    utt = parse_transcript(args.transcript.read_text(encoding="utf-8"))

    hits = sweep_commitments(utt, args.lang)
    print(f"[commitment sweep] {len(hits)} lexicon hits in {len(utt)} lines:")
    for h in hits:
        print(f"  [{h['clock']}] ({h['token']}) {h['text']}")

    if args.notes:
        notes = args.notes.read_text(encoding="utf-8")
        inv = detect_inversions(notes, utt, args.lang)
        print(f"\n[inversion detector] {len(inv)} candidate inversions:")
        for c in inv:
            print(f"  bullet polarit{c['bullet_polarity']:+d} vs evidence {c['evidence_polarity']:+d} "
                  f"@ {sec_to_clock(c['anchor'])} — {c['bullet'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
