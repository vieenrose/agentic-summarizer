"""Re-render an existing SFT pool's prompts under `PROMPT_VERSION = sys-v2`.

    python tools/repro_pool_v2.py --pool data/staging/sft_pool_p5.jsonl \\
        --out data/staging/sft_pool_p6.jsonl --corpus data/pilot_zh data/p4_zh

The stored `prompt` on every sample was built by `sys-v1`'s `build_step_prompt`, which
emitted `MEMORY:` then `CHUNK:` and nothing else. `sys-v2` prepends a `POSITION:` line
carrying the step index and the meeting's CHUNK COUNT, so every existing row's prompt is
now stale in a way that matters: a student trained on the old text would never see the
signal inference sends.

Re-rendering only needs the count, because the rest of the prompt is already correct and
the position line is a pure prefix. The count is recomputed by chunking the meeting's
real transcript with the same budget and the same injected counter the pool was built
with -- NOT inferred from the maximum step index present, which would be wrong for every
meeting whose trailing steps were dropped by the replay-clean filter (341 of them) and
would silently teach the model that meetings end sooner than they do.

A meeting whose transcript cannot be found is REFUSED rather than passed through with a
`sys-v1` prompt: a pool that mixes both prompt versions trains the model to ignore the
position line exactly where it is least reliable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.chunker import CHUNK_TOKENS, iter_chunks  # noqa: E402
from arcsum.prompts import PROMPT_VERSION, position_line  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402
from arcsum.transcript import parse_transcript  # noqa: E402


def chunk_count(path: Path) -> int:
    utts = parse_transcript(path.read_text(encoding="utf-8"))
    return len(list(iter_chunks(utts, budget=CHUNK_TOKENS, token_len=heuristic_token_len)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpus", type=Path, nargs="+", required=True)
    args = p.parse_args(argv)

    rows = [
        json.loads(ln)
        for ln in args.pool.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    meetings = sorted({r["meeting"] for r in rows})

    totals: dict[str, int] = {}
    missing: list[str] = []
    for m in meetings:
        for d in args.corpus:
            f = d / f"{m}.txt"
            if f.exists():
                totals[m] = chunk_count(f)
                break
        else:
            missing.append(m)
    if missing:
        print(
            f"[repro] REFUSED: {len(missing)} meetings have no transcript, e.g. {missing[:3]}",
            file=sys.stderr,
        )
        return 1

    # The pool holds FOUR prompt shapes, and only ONE of them is a reading step:
    #   "MEMORY:...\nCHUNK:..."  agent reading step   <- sys-v2 adds POSITION here
    #   "CHUNK:..."              baseline MAP row
    #   "SUMMARIES:..."          baseline REDUCE row
    #   "MEMORY:..." (no CHUNK)  SYNTHESIZE row
    # `sys-v2` changed `build_step_prompt` ONLY. Prefixing POSITION onto the map and
    # reduce rows would corrupt the very rows that let one checkpoint serve the baseline
    # arm honestly (SPEC §5.2's "same model" requirement), and the synthesis row has no
    # chunk to be positioned against. Their text is unchanged; only the version stamp
    # moves, because the stamp names the module constant, not the individual builder.
    out, rewritten = [], 0
    for r in rows:
        body = r["prompt"]
        if body.startswith("POSITION:"):  # already v2; do not double-prefix
            body = body.split("\n", 1)[1]
        is_step = body.startswith("MEMORY:") and "\nCHUNK:\n" in body
        if is_step:
            body = position_line(r["step"], totals[r["meeting"]]) + body
            rewritten += 1
        out.append({**r, "prompt": body, "prompt_version": PROMPT_VERSION})

    with args.out.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[repro] {len(out)} rows over {len(meetings)} meetings -> {args.out}", file=sys.stderr)
    print(
        f"[repro] prompt_version={PROMPT_VERSION}, {rewritten} reading steps re-rendered, "
        f"{len(out) - rewritten} map/reduce/synth rows left as-is",
        file=sys.stderr,
    )
    # A reading step whose index is past its own chunk count would mean the count was
    # recomputed with a different budget or counter than the pool was built with, and
    # every POSITION line would then be quietly wrong. Refuse rather than train on it.
    over = [
        r for r in out
        if r["prompt"].startswith("POSITION:") and r["step"] >= totals[r["meeting"]]
    ]
    if over:
        print(
            f"[repro] REFUSED: {len(over)} reading steps have step >= chunk count, "
            f"e.g. {[(r['meeting'], r['step']) for r in over[:3]]}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
