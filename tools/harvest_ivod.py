"""Harvest multi-speaker zh-TW meeting audio from the Legislative Yuan IVOD archive.

    python tools/harvest_ivod.py --want 20 --out-dir data/ly_phase3 --minutes 12

SPEC §9 Phase 3 wants ~20 real multi-speaker zh-TW meetings and names 立法院 committee
sessions as the known-good fallback when VoxSum recordings are not available. This is
that fallback, automated. Source is the g0v mirror `v2.ly.govapi.tw/ivods`
(104k+ records); the official `ivod.ly.gov.tw` host is not reachable from every network.

**Screens for actual audio before spending ASR time.** Measured 2026-08-28: 2 of the
first 3 sessions sampled were effectively SILENT (peak amplitude 0.008, about -42 dBFS)
— the recording exists, the stream is valid AAC, and there is simply no speech on it.
An ASR model fed that correctly emits timestamp/speaker scaffold with no text, which
looks exactly like a broken pipeline. Screening on RMS from a cheap 30-second probe
turns a confusing debugging session into a skipped row.

Downloads only `--minutes` from `--start-offset`, not whole sessions: a 90-minute
plenary is neither needed for a curation probe nor cheap to transcribe, and sessions
routinely open with silence before the gavel.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

API = "https://v2.ly.govapi.tw/ivods"

#: Below this RMS a clip carries no usable speech. Calibrated against the measured
#: split: silent sessions sat at ~0.001-0.002, the one with real speech at 0.40.
MIN_RMS = 0.02


def fetch_candidates(limit: int, page: int = 1) -> list[dict]:
    url = f"{API}?limit={limit}&page={page}"
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.loads(r.read().decode("utf-8"))
    items = d.get("ivods") or d.get("data") or []
    return [i for i in items if i.get("影片種類") == "Full"]


def probe_rms(url: str, offset: int, seconds: int = 30) -> float | None:
    """Cheap loudness probe. Returns None if the clip cannot be fetched at all."""
    import librosa
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(offset),
            "-i", url, "-t", str(seconds), "-vn", "-ac", "1", "-ar", "16000", "-y", tmp.name,
        ]
        if subprocess.run(cmd, capture_output=True, timeout=300).returncode != 0:
            return None
        try:
            w, _ = librosa.load(tmp.name, sr=16000, mono=True)
        except Exception:
            return None
        if not len(w):
            return None
        return float(np.sqrt((w**2).mean()))


def extract(url: str, offset: int, minutes: int, dst: Path) -> bool:
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(offset),
        "-i", url, "-t", str(minutes * 60), "-vn", "-ac", "1", "-ar", "16000", "-y", str(dst),
    ]
    return subprocess.run(cmd, capture_output=True, timeout=1800).returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--want", type=int, default=20)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--minutes", type=int, default=12)
    p.add_argument("--start-offset", type=int, default=300, help="skip the pre-gavel opening")
    p.add_argument("--scan", type=int, default=300, help="how many API records to consider")
    p.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cands = fetch_candidates(args.scan)
    # Longest first: a session with real debate is likelier to have speech throughout.
    cands.sort(key=lambda i: -(i.get("影片長度") or 0))
    print(f"[harvest] {len(cands)} Full sessions to screen, want {args.want}", file=sys.stderr)

    kept: list[dict] = []
    for c in cands:
        if len(kept) >= args.want:
            break
        dur = c.get("影片長度") or 0
        if dur < (args.start_offset + args.minutes * 60):
            continue
        url = c.get("video_url")
        if not url:
            continue
        # Probe SEVERAL points, not just one: a session can be silent at the opening and
        # active later, and rejecting on a single sample throws away usable meetings.
        # Measured: single-offset screening kept 1 of 11; most of the archive really is
        # silent, but not all of it is silent everywhere.
        offsets = [args.start_offset, int(dur * 0.4), int(dur * 0.7)]
        rms = None
        for off in offsets:
            if off + 30 > dur:
                continue
            r = probe_rms(url, off)
            if r is not None and (rms is None or r > rms):
                rms = r
                args_offset_used = off
                if r >= MIN_RMS:
                    break
        name = (c.get("會議名稱") or "")[:48].replace("\n", " ")
        if rms is None:
            print(f"[harvest]  skip (unfetchable)      {c['IVOD_ID']}  {name}", file=sys.stderr)
            continue
        if rms < MIN_RMS:
            print(f"[harvest]  skip (silent rms={rms:.4f}) {c['IVOD_ID']}  {name}", file=sys.stderr)
            continue

        use_off = locals().get("args_offset_used", args.start_offset)
        dst = args.out_dir / f"ivod-{c['IVOD_ID']}.wav"
        if not extract(url, use_off, args.minutes, dst):
            print(f"[harvest]  skip (extract failed)   {c['IVOD_ID']}", file=sys.stderr)
            continue
        kept.append(
            {
                "ivod_id": c["IVOD_ID"], "date": c.get("日期"), "meeting": c.get("會議名稱"),
                "duration_s": dur, "rms": round(rms, 4), "audio": str(dst),
                "offset_s": use_off, "minutes": args.minutes,
            }
        )
        print(f"[harvest]  KEEP rms={rms:.3f}  {c['IVOD_ID']}  {name}", file=sys.stderr)

    manifest = args.manifest or (args.out_dir / "manifest.json")
    manifest.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[harvest] kept {len(kept)}/{args.want} -> {manifest}", file=sys.stderr)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
