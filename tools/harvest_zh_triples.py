"""Harvest labeled zh triples for verifier retraining.

The verifier (granite-4.0-350m-zh) was NEVER trained on zh triples (verifier/2/3 are
100% en), which is the root cause of its 0/11 real-zh collapse. This tool harvests
(bullet, evidence, verdict) triples in the verifier's exact training format:

  system:    the verify system prompt (mirrors the existing verifier data)
  prompt:    EVIDENCE:\n<real 6-snippet claim-mode window>\nCLAIM: <bullet> [m:ss]
  completion: SUPPORTED / CONTRADICTED / UNSUPPORTED  (judge label)

Call it twice: once on real-noisy zh (dose-01 + tier), once on clean zh (T1 synth).

Usage:
  python tools/harvest_zh_triples.py --notes-dir DIR --transcript-dir DIR \
      --lang zh-TW --out data/sft/verifier4-zh-real.jsonl --judge-url :8090
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.backends.llama_server import LlamaServer
from voxsum.index import TranscriptIndex
from voxsum.sweep import _VERIFY_SYS
from voxsum.transcript import parse_transcript, sec_to_clock

_CLAIM_SYS = (
    "You judge one claim from meeting notes against transcript evidence.\n"
    "Reply with exactly one line:\n"
    "SUPPORTED      - the evidence supports the claim.\n"
    "CONTRADICTED   - the evidence states the opposite of the claim.\n"
    "UNSUPPORTED    - the claim is absent from the evidence.\n"
)

_BULLET_RE = re.compile(r"^- (.*?)\s+\[(\d+):(\d{2})(?::(\d{2}))?\]\s*$")
_VERDICT_RE = re.compile(r"^(SUPPORTED|CONTRADICTED|UNSUPPORTED|KEEP|DROP)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes-dir", type=Path, required=True)
    ap.add_argument("--transcript-dir", type=Path, required=True)
    ap.add_argument("--lang", default="zh-TW")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8090")
    ap.add_argument("--notes-suffix", default=".cursor.notes.txt")
    args = ap.parse_args()

    judge = LlamaServer(base_url=args.judge_url, max_tokens=512, temperature=0.0,
                        send_thinking_kwarg=True)
    out_rows = []
    pulled = 0
    for notes_f in sorted(args.notes_dir.glob(f"*{args.notes_suffix}")):
        mid = notes_f.name[: -len(args.notes_suffix)]
        tr = args.transcript_dir / f"{mid}.txt"
        if not tr.exists():
            print(f"skip {mid}: no transcript {tr}")
            continue
        utt = parse_transcript(tr.read_text(encoding="utf-8"))
        index = TranscriptIndex(utt)
        for line in notes_f.read_text(encoding="utf-8").splitlines():
            m = _BULLET_RE.match(line.strip())
            if not m:
                continue
            bullet, sec = m.group(1), int(m.group(2)) * 60 + int(m.group(3)) + (int(m.group(4) or 0) * 3600)
            ev = index.evidence_for(bullet, sec, mode="claim", limit=6)
            ev_text = "\n".join(e.render() for e in ev)
            if not ev_text.strip():
                continue
            try:
                jv_raw = judge(_CLAIM_SYS,
                               f"CLAIM: {bullet} [{sec_to_clock(sec)}]\nEVIDENCE:\n{ev_text}").strip()
            except Exception as e:
                print(f"  judge ERR {mid}: {str(e)[:50]}"); continue
            mm = _VERDICT_RE.match(jv_raw.upper())
            if not mm:
                pulled = pulled  # self-evident no-op
                continue
            verdict = "CONTRADICTED" if mm.group(1) in ("KEEP", "DROP") else mm.group(1)
            # DROP -> CONTRADICTED/UNSUPPORTED is the DROP-family; map to the 3-way set
            if mm.group(1) == "DROP":
                verdict = "UNSUPPORTED"
            out_rows.append({
                "system": _VERIFY_SYS,
                "prompt": f"EVIDENCE:\n{ev_text}\nCLAIM: {bullet} [{sec_to_clock(sec)}]",
                "completion": verdict,
                "lang": args.lang,
                "meeting": mid,
                "prompt_version": "verifier-zh-v1",
                "real": "dose" if "dose" in str(args.transcript_dir) else
                        ("tier" if "real-asr" in str(args.transcript_dir) else "clean"),
            })
    # dedup identical (prompt) rows
    seen = set()
    dedup = []
    for r in out_rows:
        k = r["prompt"]
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    with open(args.out, "w") as f:
        for r in dedup:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(dedup)} zh triples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
