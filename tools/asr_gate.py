"""The real-ASR curation check (SPEC §9 Phase 3), as a STANDING gate, not a one-off script.

    python tools/asr_gate.py --url http://127.0.0.1:8081 --protocol edit --label sft-dropv6
    python tools/asr_gate.py --url http://127.0.0.1:8081 --protocol tool --label qwen-tools-v3

**Why this needs to be a tool at all.** Every prior run of this check (dropv2-era,
dropv6, qwen-tools-v2/v3) was an inline script written fresh in the terminal and never
committed. That is exactly how the regression this tool exists to catch went unnoticed:
dropv2-era scored 17/20 curated on real zh-TW ASR; three checkpoints later, with every
gate since measured on clean machine-translated MeetingBank text, dropv6 had fallen to
6/20 and nobody ran the check again to see it happen.

**No gate in SPEC §5.2 currently protects against this**, because G1-G4 are all measured
on MeetingBank-derived text — translated, clean, in-distribution for the training corpus.
Real ASR is neither. A checkpoint can pass every §5.2 gate and still curate a minority of
real meetings, which is what the numbers above say happened.

This is deliberately reference-free (Phase 3's own bar: "no catastrophic degradation",
not a quality score) and reports the numbers that matter for that bar: NOP rate, the
fraction of meetings that end with a non-empty memory, and mean summary length. It does
not replace G1-G4; it is the check that stops a MeetingBank-only ship decision from being
silently wrong on deployment input.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.agent import run_agent  # noqa: E402
from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.transcript import parse_transcript  # noqa: E402

#: Phase 3's own corpus: 20 real Legislative Yuan committee sessions, MOSS-transcribed.
#: Not MeetingBank-derived, not machine-translated -- genuine zh-TW ASR output.
DEFAULT_CORPUS = REPO / "data/ly_phase3_v2"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--protocol", choices=("edit", "tool"), default="edit")
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    common = {"base_url": args.url, "seed": 0, "raw_completion": True,
              "extra": {"cache_prompt": False}}
    step = LlamaServer(max_tokens=256, **common)
    prose = LlamaServer(max_tokens=1200, repeat_penalty=1.1, **common)

    files = sorted(args.corpus.glob("*.txt"))
    if not files:
        print(f"[asr-gate] REFUSED: no transcripts in {args.corpus}", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        trace = run_agent(
            parse_transcript(f.read_text(encoding="utf-8")), step, synth_model=prose,
            protocol=args.protocol, on_step_error="skip",
        )
        rows.append({
            "meeting": f.stem, "steps": len(trace.steps),
            "nop": sum(1 for s in trace.steps if s.is_nop),
            "points": len(trace.memory.points), "chars": trace.synthesis.prose.chars,
            "failed_steps": len(trace.failed_steps),
            "valid_op_rate": trace.valid_op_rate,
        })
        print(f"[asr-gate] {f.stem}: steps={rows[-1]['steps']} nop={rows[-1]['nop']} "
              f"points={rows[-1]['points']} chars={rows[-1]['chars']}", file=sys.stderr)

    total_steps = sum(r["steps"] for r in rows)
    total_nop = sum(r["nop"] for r in rows)
    # "curated" by non-trivial SYNTHESIS output, not `points > 0`. A meeting can set only
    # ARC and no POINTS (measured: `ivod-17704`, a real 173-char summary built from ARC
    # alone) and would otherwise be miscounted as empty. EMPTY_MEMORY_PROSE (agent.py) is
    # the fixed string emitted when memory is genuinely empty, so anything longer is real.
    curated = sum(1 for r in rows if r["chars"] > len("本次會議沒有記錄到具體的決議或討論重點。"))
    summary = {
        "label": args.label, "protocol": args.protocol, "url": args.url,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": str(args.corpus), "n_meetings": len(rows),
        "curated": curated, "empty": len(rows) - curated,
        "nop_rate": round(total_nop / total_steps, 3) if total_steps else None,
        "mean_summary_chars": round(st.mean(r["chars"] for r in rows), 1),
        "total_failed_steps": sum(r["failed_steps"] for r in rows),
        "rows": rows,
    }

    nop_str = f"{summary['nop_rate']:.0%}" if summary["nop_rate"] is not None else "n/a"
    print(f"\n[asr-gate] {args.label or args.protocol}: {curated}/{len(rows)} curated, "
          f"NOP {nop_str}, mean {summary['mean_summary_chars']:.0f} chars",
          file=sys.stderr)
    if curated < len(rows) * 0.5:
        print("[asr-gate] WARNING: fewer than half the real-ASR meetings were curated. "
              "This does not fail any §5.2 gate but is the check that would have caught "
              "the dropv2->dropv6 regression (17/20 -> 6/20).", file=sys.stderr)

    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"[asr-gate] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
