"""Real-zh evidence-window probe #167-adjacent: does the granite-zh verifier collapse
to KEEP/SUPPORTED on fabricated bullets when the evidence is the harness's real
noisy 6-snippet zh window (the author's round-5.5 finding)?

For every bullet in a run's notes (all three real-ASR tier episodes), build the exact
claim-mode evidence the harness hands the verifier, then:
  1. the gpt-oss FAITH judge labels the triple (SUPPORTED / CONTRADICTED / UNSUPPORTED)
  2. granite-zh verifier labels the SAME triple (KEEP / DROP / FIX)

The collapse = the rate at which granite-zh says KEEP on bullets the judge does NOT
support. Produces the real-zh retraining triples.

Usage:
  python tools/probe_verifier_zh.py --notes-dir runs/real-asr-tier/arms-v4 \
      --out runs/verifier-zh-collapse.jsonl
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


def parse_notes(text: str) -> list[tuple[str, str, int]]:
    out = []
    section = ""
    for line in text.splitlines():
        if line.strip() in ("SUMMARY", "DECISIONS", "ACTIONS", "OPEN", "TOPICS"):
            section = line.strip()
        elif line.startswith("- "):
            m = _BULLET_RE.match(line.strip())
            if not m:
                continue
            secs = int(m.group(2)) * 60 + int(m.group(3)) + (int(m.group(4) or 0) * 3600)
            out.append((section, m.group(1), secs))
    return out


def verdict(response: str, choices: tuple[str, ...]) -> str | None:
    up = response.upper()
    for c in choices:
        if c in up:
            # FIX contains no bare KEEP/DROP; prefer the verb
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes-dir", type=Path, default=Path("runs/real-asr-tier/arms-v4"))
    ap.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts/real-asr"))
    ap.add_argument("--out", type=Path, default=Path("runs/verifier-zh-collapse.jsonl"))
    ap.add_argument("--judge-url", default="http://127.0.0.1:8090")
    ap.add_argument("--verifier-url", default="http://127.0.0.1:8096")
    args = ap.parse_args()

    judge = LlamaServer(base_url=args.judge_url, max_tokens=1500, temperature=0.0,
                        send_thinking_kwarg=True)  # send enable_thinking=false
    verifier = LlamaServer(base_url=args.verifier_url, max_tokens=256, temperature=0.0,
                           send_thinking_kwarg=True)

    rows = []
    for notes_f in sorted(args.notes_dir.glob("*.notes.txt")):
        mid = notes_f.stem.replace(".cursor.notes", "")
        tr = args.transcript_dir / f"{mid}.txt"
        if not tr.exists():
            print(f"skip {mid}: no transcript"); continue
        utt = parse_transcript(tr.read_text(encoding="utf-8"))
        index = TranscriptIndex(utt)
        for section, bullet, anchor in parse_notes(notes_f.read_text(encoding="utf-8")):
            ev = index.evidence_for(bullet, anchor, mode="claim", limit=6)
            ev_text = "\n".join(e.render() for e in ev)
            if not ev_text:
                continue
            bullet_w = f"{bullet} [{sec_to_clock(anchor)}]"
            jp = f"CLAIM: {bullet_w}\nEVIDENCE:\n{ev_text}"
            vp = f"BULLET: {bullet_w}\nEVIDENCE:\n{ev_text}"
            try:
                jv_raw = judge(_CLAIM_SYS, jp).strip()
            except Exception as e:
                jv_raw = f"ERR {e}"
            try:
                vv_raw = verifier(_VERIFY_SYS, vp).strip()
            except Exception as e:
                vv_raw = f"ERR {e}"
            jv = verdict(jv_raw, ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED"))
            vv = verdict(vv_raw, ("KEEP", "DROP", "FIX"))
            rows.append({
                "meeting": mid, "section": section, "bullet": bullet, "anchor": anchor,
                "judge": jv_raw[:30], "verifier": vv_raw[:60],
                "judge_v": jv, "verifier_v": vv,
                "evidence": ev_text,
            })

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # collapse = verifier KEEPs a bullet the judge does NOT support
    coll = [r for r in rows if r["verifier_v"] == "KEEP" and r["judge_v"] != "SUPPORTED"]
    tot = len(rows)
    sup = [r for r in rows if r["judge_v"] == "SUPPORTED"]
    keep_all = [r for r in rows if r["verifier_v"] == "KEEP"]
    print(f"triples: {tot} | judge-SUPPORTED: {len(sup)} | verifier-KEEP: {len(keep_all)}")
    print(f"COLLAPSE: verifier KEEP on judge-not-supported: {len(coll)}/{tot-len(sup)} ({len(coll)/max(tot-len(sup),1)*100:.0f}%)")
    for r in coll:
        print(f"  [{r['meeting']}:{r['section']}] {r['bullet'][:50]} | judge={r['judge_v']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
