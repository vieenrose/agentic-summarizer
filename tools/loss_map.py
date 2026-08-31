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

**Report STRICT and PARTIAL together, and do not headline either alone.** Strict requires
the verbatim `key_term`; partial accepts a >=60% contiguous run, so 「溪州市集」 counts for
「溪州假日市集」 -- a reader can still tell which proposal was decided. The two tell
genuinely different stories and the difference is the finding, not a rounding detail:

    v5, 11-scenario set:  emitted 4/11 STRICT but 8/11 PARTIAL;  memory 2/11 either way.

Read strictly, the model "never writes the detail down" and the loss is upstream of
memory. Read partially, it writes a recognisable form 8 times out of 11 and then loses 6
of those between emission and final memory. **The second reading is the correct one** --
emission is mostly fine, and the deficit is retention. Quoting only the strict column
overstates an emission deficit that largely is not there.

Whitespace is always normalised out, because trap 5 recorded a false FAIL caused by
「B 棟」 vs 「B棟」 spacing.
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


#: A near-miss still identifies the proposal: 「溪州市集」 for 「溪州假日市集」 tells a reader
#: which one was decided. Reporting STRICT alongside PARTIAL keeps the headline honest —
#: a strict-only read would book every paraphrase as a loss and overstate the deficit,
#: which is exactly the direction that would make a "the model never writes it down"
#: story look stronger than the evidence supports.
PARTIAL_RATIO = 0.6


def _has_partial(term: str, *texts: str) -> bool:
    """Does any text contain a contiguous run of at least `PARTIAL_RATIO` of `term`?"""
    t = _norm(term)
    need = max(int(len(t) * PARTIAL_RATIO), 2)
    if not t:
        return False
    subs = {t[i : i + need] for i in range(len(t) - need + 1)}
    return any(any(s in _norm(x) for s in subs) for x in texts if x)



def _attribute(term: str, verdicts: list[dict]) -> dict:
    """Classify what happened to the ops that mentioned `term`.

    `refused` -- the harness rejected an ADD carrying the term (cap, duplicate,
    contradiction guard, language guard). Fixable here: supervision or caps.
    `dropped`  -- an applied ADD carrying the term was later removed by a DROP whose
    replacement did not carry it. That is lossy revision, and revision-specific.
    """
    refused, applied_add, dropped = [], False, False
    for v in verdicts:
        op = v.get("op") or ""
        if not _has_partial(term, op):
            # A DROP names only a PREFIX, so it rarely contains the key term itself;
            # detect the revision case by a drop landing after a term-carrying add.
            if applied_add and op.startswith("DROP") and v.get("applied"):
                dropped = True
            continue
        if v.get("applied"):
            applied_add = True
        elif v.get("reason"):
            refused.append(v["reason"])
    return {
        "refused_reasons": refused,
        "was_refused": bool(refused) and not applied_add,
        "was_dropped": dropped,
    }


def analyse(report_path: Path, scenarios) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_slug = {s.slug: s for s in scenarios}
    rows = []
    for r in report["results"]:
        sc = by_slug.get(r["slug"])
        if sc is None:
            continue
        verdicts = r.get("op_verdicts")
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
                "emitted_p": _has_partial(sc.key_term, raw_steps),
                "memory_p": _has_partial(sc.key_term, *points),
                "prose_p": _has_partial(sc.key_term, r.get("prose", "")),
                "passed": r.get("passed"),
                # Why an emitted term is not in final memory. Refused vs dropped is the
                # whole strategic question: refused is ours to fix, dropped-during-
                # revision is the corpus reading in `runs/g1-study.md`.
                "has_verdicts": verdicts is not None,
                **_attribute(sc.key_term, verdicts or []),
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
        "partial": {k: sum(r[k + "_p"] for r in rows) for k in ("emitted", "memory", "prose")},
        "pct": {
            k: sum(r[k] for r in rows) / n
            for k in ("emitted", "memory", "prose")
        },
        "pct_partial": {
            k: sum(r[k + "_p"] for r in rows) / n
            for k in ("emitted", "memory", "prose")
        },
        "attribution": {
            # A report written before `op_verdicts` existed cannot be attributed. Say so
            # rather than printing 0 refused / 0 dropped, which reads as "neither
            # mechanism fired" instead of "we did not record it".
            "available": sum(1 for r in rows if r["has_verdicts"]),
            "emitted_not_in_memory": sum(
                1 for r in rows if r["emitted_p"] and not r["memory_p"]
            ),
            "refused": sum(1 for r in rows if r["was_refused"]),
            "dropped": sum(1 for r in rows if r["was_dropped"]),
        },
        "rows": rows,
    }


def _table(name: str, a: dict) -> str:
    p, q, w = a["pct"], a["pct_partial"], a["partial"]
    return (
        f"{name:9} n={a['n']:3}  "
        f"emitted {a['emitted']:3}/{w['emitted']:<3} ({p['emitted']:5.1%} / {q['emitted']:5.1%})  "
        f"memory {a['memory']:3}/{w['memory']:<3} ({p['memory']:5.1%} / {q['memory']:5.1%})  "
        f"prose {a['prose']:3}/{w['prose']:<3} ({p['prose']:5.1%} / {q['prose']:5.1%})  "
        f"passed {a['passed']:3}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reversal", type=Path, required=True)
    p.add_argument("--control", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    rev = analyse(args.reversal, PROBE_SCENARIOS)
    ctl = analyse(args.control, CONTROL_SCENARIOS) if args.control else None

    print("columns are STRICT/PARTIAL -- verbatim key_term, then a >=60% contiguous run\n")
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
    a = rev["attribution"]
    if a["emitted_not_in_memory"] and not a["available"]:
        print(
            f"\nretention loss (reversal arm): {a['emitted_not_in_memory']} scenarios "
            "emitted key_term but lost it before final memory.\n"
            "  ATTRIBUTION UNAVAILABLE -- this report predates `op_verdicts`. Re-run "
            "tools/score_reversals.py to tell refusal from lossy revision."
        )
    elif a["emitted_not_in_memory"]:
        print(
            f"\nretention loss (reversal arm): {a['emitted_not_in_memory']} scenarios "
            f"emitted key_term but lost it before final memory\n"
            f"  refused by the harness: {a['refused']}   dropped during revision: "
            f"{a['dropped']}"
        )
        reasons = [x for r in rev["rows"] for x in r["refused_reasons"]]
        for reason in sorted(set(reasons)):
            print(f"    refusal: {reason}  x{reasons.count(reason)}")
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
