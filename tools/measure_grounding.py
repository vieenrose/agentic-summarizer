"""Measure how GROUNDED a checkpoint's reading steps are, per domain (SPEC §8 risk 1).

    python tools/measure_grounding.py --url http://127.0.0.1:8081 \\
        --corpus meetingbank=data/eval20_zh probe=PROBE ly=data/ly_asr --limit 6

G1 fails on `office_move_reversal` because the model ignores a chunk about an office
relocation and emits memorised council boilerplate — library procurement, park
renovation, bus routes. The summary is fluent, in-format, and about a different meeting.
That is not a summarisation error; it is the reading step inventing content.

**The metric is character-trigram containment of each emitted point in ITS OWN chunk.**
A point that genuinely came from the chunk shares long character runs with it; a
memorised one does not. Trigrams (not tokens) because §5's tokenisation is character
level for zh, and containment (not F1) because a point is a compression of the chunk —
it should be a near-subset of it, and being much shorter than the chunk must not be
penalised.

This cannot prove a point is hallucinated: a correct abstractive paraphrase also scores
below 1.0, and a point assembled from the chunk's own vocabulary in a false arrangement
scores high. It measures the FLOOR — text with no lexical support in its own chunk had
to come from somewhere else. Read the per-domain gap, not the absolute value.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.chunker import CHUNK_TOKENS, iter_chunks  # noqa: E402
from arcsum.memory import Memory  # noqa: E402
from arcsum.ops import Add, Arc, parse_ops  # noqa: E402
from arcsum.probe import probe_meetings  # noqa: E402
from arcsum.prompts import build_step_prompt, step_system_prompt  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402
from arcsum.transcript import parse_transcript  # noqa: E402


def trigrams(text: str) -> set[str]:
    s = "".join(text.split())
    return {s[i : i + 3] for i in range(len(s) - 2)}


def containment(fragment: str, source: str) -> float:
    """Fraction of the fragment's trigrams that appear in the source. 1.0 = every run of
    three characters is present somewhere in the chunk; 0.0 = none is."""
    f = trigrams(fragment)
    if not f:
        return 1.0
    return len(f & trigrams(source)) / len(f)


def run_domain(name: str, transcripts: list[tuple[str, str]], url: str, limit: int) -> dict:
    step = LlamaServer(
        base_url=url, max_tokens=256, seed=0, raw_completion=True,
        extra={"cache_prompt": False},
    )
    sys_prompt = step_system_prompt()
    scores: list[float] = []
    worst: list[tuple[float, str, str]] = []

    for meeting, text in transcripts[:limit]:
        chunks = list(
            iter_chunks(parse_transcript(text), budget=CHUNK_TOKENS, token_len=heuristic_token_len)
        )
        memory = Memory(token_len=heuristic_token_len)
        for chunk in chunks:
            raw = step(sys_prompt, build_step_prompt(memory, chunk, total=len(chunks)))
            body = chunk.render()
            for op in parse_ops(raw):
                if isinstance(op, Add | Arc):
                    frag = op.point if isinstance(op, Add) else op.text
                    c = containment(frag, body)
                    scores.append(c)
                    worst.append((c, meeting, frag))
    worst.sort()
    return {
        "domain": name,
        "n_points": len(scores),
        "mean": round(statistics.mean(scores), 3) if scores else None,
        "median": round(statistics.median(scores), 3) if scores else None,
        "frac_below_0.3": round(sum(1 for s in scores if s < 0.3) / len(scores), 3)
        if scores else None,
        "worst": [{"containment": round(c, 3), "meeting": m, "text": t} for c, m, t in worst[:5]],
    }


def load(spec: str) -> tuple[str, list[tuple[str, str]]]:
    name, _, src = spec.partition("=")
    if src == "PROBE":
        return name, [
            (m.name, "\n".join(u.render() for u in m.utterances)) for m in probe_meetings()
        ]
    d = Path(src)
    return name, [(f.stem, f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.txt"))]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--corpus", nargs="+", required=True, help="name=dir, or name=PROBE")
    p.add_argument("--limit", type=int, default=6, help="meetings per domain")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    out = []
    for spec in args.corpus:
        name, transcripts = load(spec)
        r = run_domain(name, transcripts, args.url, args.limit)
        out.append(r)
        print(
            f"[ground] {r['domain']:14} n={r['n_points']:4} mean={r['mean']} "
            f"median={r['median']} below0.3={r['frac_below_0.3']}",
            file=sys.stderr,
        )
    if args.out:
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
