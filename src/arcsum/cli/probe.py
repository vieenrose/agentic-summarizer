"""`arcsum-probe`: the SPEC §5.2 G1 revision-probe workflow, split into the two halves
that don't require a live model in this repo's test environment.

    arcsum-probe dump --out probe_transcripts/
    # ... run the harness/baseline against each probe_transcripts/<name>.txt
    # externally, saving each finished prose summary as results[name] ...
    arcsum-probe score results.json --out probe_report.json

`results.json` is `{"<probe_meeting_name>": "<prose text>", ...}`. `score` requires
every probe meeting name to be present so a partial run cannot silently pass G1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arcsum.probe import ProbeResult, probe_meetings, score_probe


def dump_transcripts(out_dir: Path) -> list[str]:
    """Write one v2 `.txt` transcript per probe meeting; return the names written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for meeting in probe_meetings():
        text = "\n".join(u.render() for u in meeting.utterances)
        (out_dir / f"{meeting.name}.txt").write_text(text, encoding="utf-8")
        names.append(meeting.name)
    return names


def score_results(results: dict[str, str]) -> tuple[list[ProbeResult], bool]:
    """Score every probe meeting against `results`. Raises `KeyError` naming the first
    missing meeting -- a partial run must fail loudly, not silently pass G1 on however
    many meetings happened to be present."""
    scored = []
    for meeting in probe_meetings():
        if meeting.name not in results:
            raise KeyError(f"missing probe result for {meeting.name!r}")
        scored.append(score_probe(results[meeting.name], meeting))
    g1_passed = all(r.passed for r in scored)
    return scored, g1_passed


def _result_to_dict(r: ProbeResult) -> dict:
    return {
        "name": r.name,
        "states_later": r.states_later,
        "states_earlier_as_current": r.states_earlier_as_current,
        "distractor_absent": r.distractor_absent,
        "passed": r.passed,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump", help="write each probe meeting's v2 transcript to disk")
    dump.add_argument("--out", type=Path, required=True)

    score = sub.add_parser("score", help="score already-produced prose against every probe meeting")
    score.add_argument("results", type=Path, help='JSON file: {"<name>": "<prose>"}')
    score.add_argument("--out", type=Path, default=None, help="write the JSON report here")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "dump":
        names = dump_transcripts(args.out)
        print(f"[probe] wrote {len(names)} transcripts -> {args.out}", file=sys.stderr)
        return 0

    results = json.loads(args.results.read_text(encoding="utf-8"))
    try:
        scored, g1_passed = score_results(results)
    except KeyError as exc:
        print(f"[probe] {exc}", file=sys.stderr)
        return 1

    report = {"g1_passed": g1_passed, "results": [_result_to_dict(r) for r in scored]}
    print(f"G1 (revision probe): {'PASS' if g1_passed else 'FAIL'}")
    for r in scored:
        print(f"  {r.name}: {'PASS' if r.passed else 'FAIL'}")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if g1_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
