"""Trace the identifying detail (`key_term`) through the pipeline, stage by stage.

    python tools/loss_map.py --reversal runs/X/revprobe_report.json \\
                             --control  runs/X/revctl_report.json

**The question this answers.** `runs/g1-study.md` measured `key_term` reaching MEMORY in
5/11 probe scenarios and surviving to PROSE in 2/11, and concluded "no single-point fix
can carry this gate". But that measurement had no CONTROL: it could not distinguish

  (a) the reading step cannot retain an identifying detail AT ALL -- a general
      point-quality deficit that has nothing to do with revision, and is trainable from
      MeetingBank, which is full of decisions-with-details; from
  (b) the detail is captured fine and is lost specifically DURING revision -- which needs
      reversal data and, MeetingBank having none, means new corpus.

The two imply completely different work, and five refuted G1 attempts all assumed (b).
`CONTROL_SCENARIOS` (decision taken, never reversed) is the missing arm: same generator,
same prompt shape, same `key_term` templates, no reversal. Compare the two.

**Reading the output.** `memory` is the share of scenarios where some point in final
memory carries `key_term`; `prose` is the share where the synthesis output does. If
control and reversal lose the term at similar rates, the deficit is general (a). If
control retains it and reversal does not, the loss is revision-specific (b).

Matching is deliberately lenient in one direction only -- whitespace is normalised out,
because trap 5 recorded a false FAIL caused by 「B 棟」 vs 「B棟」 spacing. It is NOT
lenient about the term itself: a paraphrase is a loss, since the point of `key_term` is
that a reader can tell WHICH proposal was decided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from gen_reversals import CONTROL_SCENARIOS, PROBE_SCENARIOS  # noqa: E402


def _norm(text: str) -> str:
    return "".join(text.split())


def _has(term: str, *texts: str) -> bool:
    t = _norm(term)
    return any(t in _norm(x) for x in texts if x)


def analyse(report_path: Path, scenarios) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_slug = {s.slug: s for s in scenarios}
    rows = []
    for r in report["results"]:
        sc = by_slug.get(r["slug"])
        if sc is None:
            continue
        points = r.get("memory_points") or []
        raw_steps = "\n".join(r.get("steps") or [])
        rows.append(
            {
                "slug": r["slug"],
                "key_term": sc.key_term,
                # Emitted at all, even if the op was later refused or dropped: separates
                # "never produced" from "produced then lost".
                "emitted": _has(sc.key_term, raw_steps),
                "memory": _has(sc.key_term, *points),
                "prose": _has(sc.key_term, r.get("prose", "")),
                "passed": r.get("passed"),
            }
        )
    n = len(rows) or 1
    return {
        "path": str(report_path),
        "n": len(rows),
        "emitted": sum(r["emitted"] for r in rows),
        "memory": sum(r["memory"] for r in rows),
        "prose": sum(r["prose"] for r in rows),
        "passed": sum(bool(r["passed"]) for r in rows),
        "pct": {
            k: sum(r[k] for r in rows) / n for k in ("emitted", "memory", "prose")
        },
        "rows": rows,
    }


def _table(name: str, a: dict) -> str:
    p = a["pct"]
    return (
        f"{name:10} n={a['n']:3}  emitted {a['emitted']:3} ({p['emitted']:5.1%})  "
        f"memory {a['memory']:3} ({p['memory']:5.1%})  "
        f"prose {a['prose']:3} ({p['prose']:5.1%})  passed {a['passed']:3}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reversal", type=Path, required=True)
    p.add_argument("--control", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    rev = analyse(args.reversal, PROBE_SCENARIOS)
    ctl = analyse(args.control, CONTROL_SCENARIOS) if args.control else None

    print(_table("reversal", rev))
    if ctl:
        print(_table("control", ctl))
        # The whole point of the control arm: attribute the loss.
        d_mem = ctl["pct"]["memory"] - rev["pct"]["memory"]
        d_pro = ctl["pct"]["prose"] - rev["pct"]["prose"]
        print(
            f"\ncontrol - reversal:  memory {d_mem:+.1%}   prose {d_pro:+.1%}\n"
            "  small gap  -> the detail is lost generally, NOT during revision:\n"
            "               fixable from MeetingBank supervision (point self-containment\n"
            "               + synthesis preservation), no reversal corpus needed.\n"
            "  large gap  -> loss is revision-specific: needs real reversal data."
        )
    print("\nper-scenario (reversal arm):")
    for r in rev["rows"]:
        flags = "".join(c if r[k] else "-" for k, c in
                        (("emitted", "E"), ("memory", "M"), ("prose", "P")))
        print(f"  {r['slug']:18} {flags}  {r['key_term']}")

    if args.out:
        args.out.write_text(
            json.dumps({"reversal": rev, "control": ctl}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\n-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
