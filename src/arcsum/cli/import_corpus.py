"""`arcsum-import`: MeetingBank Zenodo transcripts -> format v2 files + a provenance
manifest (SPEC §2.2 stage 1).

    arcsum-import transcripts/ --out corpus/ --eval-n 40

`transcripts/` holds `*.transcript.json` files in the Zenodo record-7989108 shape
(`{"segments": [...]}`, CLAUDE.md "Corpus access"). Each becomes one
`<out>/<safe_id>.txt` format-v2 transcript; `safe_id` is derived from the filename
stem, sanitised the same way as `corpus.meetingbank.safe_id` for meeting ids proper,
since MeetingBank uids and filenames both contain unsafe characters.

This CLI performs only §2.2 stage 1 (import) and split carving. Stages 2-3
(translation, composition) and human validation are separate tools that update the
same manifest by meeting id — this tool only ever creates fresh records with
`translated_by=None, composed_by=None, human_validated=False`; it never overwrites an
existing manifest entry's provenance for a meeting id that is already present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arcsum.corpus.manifest import DEFAULT_EVAL_N, Manifest, carve_splits
from arcsum.corpus.meetingbank import import_meeting, safe_id


def import_transcript_file(path: Path) -> tuple[str, str]:
    """One `*.transcript.json` file -> `(meeting_id, v2_text)`."""
    meeting_id = safe_id(path.stem)
    transcript = json.loads(path.read_text(encoding="utf-8"))
    utterances = import_meeting(transcript)
    v2_text = "\n".join(u.render() for u in utterances)
    return meeting_id, v2_text


def import_directory(
    src_dir: Path, out_dir: Path, *, eval_n: int = DEFAULT_EVAL_N, seed: int = 0
) -> Manifest:
    """Import every `*.transcript.json` under `src_dir`, write one v2 `.txt` per
    meeting under `out_dir`, and return a `Manifest` with splits carved over every
    meeting id imported."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meeting_ids: list[str] = []
    for path in sorted(src_dir.glob("*.transcript.json")):
        meeting_id, v2_text = import_transcript_file(path)
        (out_dir / f"{meeting_id}.txt").write_text(v2_text, encoding="utf-8")
        meeting_ids.append(meeting_id)

    manifest = Manifest()
    splits = carve_splits(meeting_ids, eval_n=eval_n, seed=seed)
    manifest.set_split(splits)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("src", type=Path, help="directory of *.transcript.json files")
    p.add_argument("--out", type=Path, required=True, help="directory to write v2 .txt files to")
    p.add_argument(
        "--manifest", type=Path, default=None, help="manifest path (default: <out>/manifest.json)"
    )
    p.add_argument("--eval-n", type=int, default=DEFAULT_EVAL_N, help="eval split size")
    p.add_argument("--seed", type=int, default=0, help="split-carving seed")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest or (args.out / "manifest.json")

    manifest = import_directory(args.src, args.out, eval_n=args.eval_n, seed=args.seed)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_list(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[import] {len(manifest.records)} meetings -> {args.out} "
        f"({len(manifest.eval_ids())} eval, {len(manifest.train_ids())} train); "
        f"manifest -> {manifest_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
