"""Translate a format-v2 English transcript to zh-TW (SPEC §2.2 stage 2).

    python tools/translate_corpus.py data/p4_en --out-dir data/p4_zh \\
        --url http://127.0.0.1:8200 --shard 0/2

**Line-count integrity is a hard pass/fail** (SPEC §2.2's translation gate). One output
utterance per input utterance, speaker preserved verbatim, in order. So this translates
the TEXT of each utterance independently and never lets the model restructure the
transcript — a model asked to translate a whole meeting will merge turns, drop
back-channels, and silently change the line count, which breaks the chunker's alignment
with the gold item spans downstream.

A refused or empty translation keeps the ENGLISH text for that line rather than dropping
it. Dropping would corrupt the line count; keeping English is visible to the language
gate (`lang.cjk_ratio`) and to the per-file report, so it fails loudly at review time
instead of silently shortening the meeting.

Utterances are batched newline-delimited into one request, because a 2-hour council
meeting runs to ~760 lines and 300 meetings would be ~200k round-trips. Measured:
TranslateGemma preserves line structure exactly. Any batch whose line count does not
come back intact is retried line-by-line, so batching can never cost line integrity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.lang import cjk_ratio  # noqa: E402
from arcsum.transcript import parse_transcript  # noqa: E402

#: TranslateGemma is NOT a chat model. Its chat template rejects ordinary string
#: content (llama.cpp refuses to even start with --jinja), and renders this fixed
#: instruction from a structured {source_lang_code, target_lang_code, text} item. We
#: reproduce it verbatim and drive the raw /completion endpoint instead.
SRC_LANG, SRC_CODE = "English", "en"
TGT_LANG, TGT_CODE = "Chinese", "zh-TW"


def render_prompt(text: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"You are a professional {SRC_LANG} ({SRC_CODE}) to {TGT_LANG} ({TGT_CODE}) "
        "translator. Your goal is to accurately convey the meaning and nuances of the "
        f"original {SRC_LANG} text while adhering to {TGT_LANG} grammar, vocabulary, and "
        "cultural sensitivities.\n"
        f"Produce only the {TGT_LANG} translation, without any additional explanations or "
        f"commentary. Please translate the following {SRC_LANG} text into {TGT_LANG}:\n\n\n"
        + text.strip()
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def post(url: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def translate_one(base_url: str, text: str, *, max_tokens: int) -> str | None:
    """One utterance in, one translation out. TranslateGemma translates a single text
    block; there is no numbered-batch mode, so line-count integrity (SPEC §2.2's hard
    pass/fail) is guaranteed structurally here rather than checked afterwards."""
    if not text.strip():
        return ""
    body = {
        "prompt": render_prompt(text),
        "temperature": 0.0,
        "n_predict": max_tokens,
        "cache_prompt": False,
        "stop": ["<end_of_turn>", "<start_of_turn>"],
    }
    try:
        out = post(f"{base_url}/completion", body)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    return (out.get("content") or "").strip()


#: A single council utterance can run to thousands of characters (measured: one file's
#: 14 longest lines were 7,164 chars on average and 61% of the whole transcript). Those
#: exceed a server slot's context and their translation is rejected, so the line silently
#: falls back to ENGLISH -- which is how a transcript ended up 61% untranslated while
#: reporting only "14 kept_english". Long text is split on sentence boundaries, each
#: piece translated, and the pieces rejoined into ONE line: line count and order are
#: still structurally preserved.
MAX_PIECE_CHARS = 700
_SENT_END = re.compile(r"(?<=[.!?;])\s+")


def split_long_text(text: str, limit: int = MAX_PIECE_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces, cur = [], ""
    for sent in _SENT_END.split(text):
        if cur and len(cur) + len(sent) + 1 > limit:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    # A single sentence longer than the limit still has to be cut somewhere.
    out = []
    for piece in pieces:
        while len(piece) > limit * 2:
            out.append(piece[: limit * 2])
            piece = piece[limit * 2 :]
        out.append(piece)
    return out


def translate_long(base_url: str, text: str, *, max_tokens: int) -> str | None:
    parts = split_long_text(text)
    if len(parts) == 1:
        return translate_one(base_url, text, max_tokens=max_tokens)
    done = []
    for part in parts:
        t = translate_one(base_url, part, max_tokens=max_tokens)
        if not t:
            return None
        done.append(t.strip())
    return "".join(done)


def translate_file(src: Path, base_url: str, *, max_tokens: int, workers: int) -> tuple[str, dict]:
    """ONE CALL PER UTTERANCE. Never batches.

    Batching was tried and abandoned on measured evidence. TranslateGemma MERGES adjacent
    short back-channel turns ("Hmm.", "We have.", "Oh, yeah, sure.") -- ubiquitous in
    council transcripts -- and the output then DRIFTS out of alignment with the input: in
    one 10-line probe, output line 1 carried input line 2's content. A line-count check
    catches the simple case, but a merge plus a split elsewhere would match on count while
    being silently misaligned, which would attach the wrong speaker to the wrong words
    throughout the training data.

    Line count and line ORDER are both structurally guaranteed here: one request per
    utterance, results reassembled by index. Speed comes from concurrency (the server runs
    8 slots) rather than from trusting the model with document structure.
    """
    utts = parse_transcript(src.read_text(encoding="utf-8"))
    out: list[str | None] = [None] * len(utts)
    kept_english = 0

    def work(i: int) -> tuple[int, str | None]:
        return i, translate_long(base_url, utts[i].text, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, t in ex.map(work, range(len(utts))):
            out[i] = t

    final = []
    for t, u in zip(out, utts):
        if t and t.strip():
            final.append(t.strip())
        else:
            final.append(u.text)  # keep ENGLISH; dropping corrupts the line count
            kept_english += 1

    body = "\n".join(f"{u.speaker}: {t}" for u, t in zip(utts, final)) + "\n"
    return body, {
        "lines_in": len(utts), "lines_out": len(final),
        "kept_english": kept_english,
        "cjk_ratio": round(cjk_ratio(" ".join(final)), 3),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src_dir", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--shard", default="0/1", help="i/n -- process only files where idx %% n == i")
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--workers", type=int, default=8, help="match llama-server --parallel")
    p.add_argument("--min-cjk", type=float, default=0.70,
                   help="refuse a file below this CJK ratio -- catches silent English fallback")
    args = p.parse_args(argv)

    i, n = (int(x) for x in args.shard.split("/"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = [f for k, f in enumerate(sorted(args.src_dir.glob("*.txt"))) if k % n == i]
    print(f"[translate] shard {i}/{n}: {len(files)} files -> {args.out_dir}", file=sys.stderr)

    for k, f in enumerate(files, 1):
        dst = args.out_dir / f.name
        if dst.exists():
            print(f"[translate] ({k}/{len(files)}) {f.name} exists, skip", file=sys.stderr)
            continue
        body, st = translate_file(f, args.url, max_tokens=args.max_tokens, workers=args.workers)
        if st["lines_in"] != st["lines_out"]:  # defensive: must never happen
            print(f"[translate] REFUSED {f.name}: line count {st}", file=sys.stderr)
            continue
        # A file can pass every STRUCTURAL check (line count, speakers, no simplified
        # characters, no v2 defects) and still be mostly untranslated: measured, one
        # transcript's 14 longest lines were 61% of its characters and had all fallen
        # back to English, reporting only "kept_english=14". CJK ratio is the check that
        # catches it. The pilot corpus sits around 0.85.
        if st["cjk_ratio"] < args.min_cjk:
            print(
                f"[translate] REFUSED {f.name}: cjk_ratio {st['cjk_ratio']} < {args.min_cjk} "
                f"-- mostly untranslated ({st['kept_english']} lines kept English)",
                file=sys.stderr,
            )
            continue
        dst.write_text(body, encoding="utf-8")
        print(
            f"[translate] ({k}/{len(files)}) {f.name} lines={st['lines_out']} "
            f"cjk={st['cjk_ratio']} kept_en={st['kept_english']}",
            file=sys.stderr, flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
