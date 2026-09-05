"""Convert gold DROP+ADD steps that are genuine revisions into the atomic `revise` op.

    python tools/promote_revisions.py --pool data/staging/sft_pool_v13.jsonl \\
        --out data/staging/sft_pool_v14.jsonl --report runs/pool-v14-report.json

**The defect.** SPEC §4.1 added `revise` precisely because DROP-then-ADD is the surface form
of churn: the harness cannot distinguish "supersede this point" from "rewrite what I already
had", and `guards.restates_dropped` fires on both by design. But the v1.1 migration decided
relatedness with a shared TEXT PREFIX, so a revision whose replacement rewords the beginning
was left as two ops. Measured on `sft_pool_v13.jsonl`: **94 single-drop-single-add steps are
genuine revisions** the prefix test missed —

    確認收到需求管理試點實施方案       -> 動議確認收到需求管理試點計畫實施方案
    透過大麻零售執照及社會公平計畫條例  -> 透過修改大麻零售執照及社會公平計畫條例
    市律師擬決議支援星巴克公平自由工會選舉 -> 建議市檢察官擬決議支援星巴克公平自由工會選舉

— each adding a qualifier, a modifier, or the responsible actor. Taught as drop+add they
demonstrate the two-op form on exactly the occasions the atomic op exists to cover. Promotion
takes the pool from **247 revises to 341 (+38%)** while removing 94 demonstrations of the
churn-shaped alternative.

**Why similarity and not a looser prefix.** The three-way split from `migrate_pool_v11.py` is
unchanged and still load-bearing — **near-identical → churn (drop the row); related-but-changed
→ `revise`; unrelated → genuinely separate ops.** Converting indiscriminately would launder
churn into a sanctioned op and teach that rewriting what you already had is correct. Only the
RELATEDNESS test changes, from "shares a leading run of characters" to "shares most of its
character trigrams", which is what actually distinguishes a reworded version of a point from a
different point.

**Churn is re-checked with the harness's own `restates_dropped`**, not a proxy, so the training
filter and the runtime detector cannot drift. A crude prefix test reports 43% churn on this
pool by counting legitimate revision; the real rate is 0%.

**The numeral guard is ONE-DIRECTIONAL, and getting this wrong halved the yield.** Importing
`memory._near_duplicate`'s symmetric rule blocked 89 conversions, because it treats any change
of figures as a collision. That is right when MERGING two points (one is lost) and wrong here:
`revise` keeps the replacement whole and journals the original with a `superseded_by` link, so
`…東斯皮爾大道123號…` → `…第10選區東斯皮爾大道123號…` is enrichment, not conflict. A revision
may ADD figures; it may not LOSE them — that is the lossy-revision defect `gen_reversals.py`
shipped for 68 rows, teaching `key_term` loss on the exact capability G1 measures. 139 pairs
are declined on that basis.

**This is a supervision change with no runtime component** — `revise` already exists and is
already parsed. Only what the gold demonstrates changes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.guards import restates_dropped  # noqa: E402
from arcsum.memory import _trigrams, normalize  # noqa: E402
from arcsum.ops import Add, Drop  # noqa: E402
from arcsum.toolcalls import parse_tool_calls  # noqa: E402

TOOL_CALL = re.compile(r'\{"name".*\}', re.S)
NUMERAL = re.compile(r"\d+|[零〇一二兩三四五六七八九十百千萬億兆]{1,}")


def similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return max(inter / len(ta), inter / len(tb))


def prompt_points(prompt: str) -> dict[int, str]:
    """The working set as the stored prompt renders it — `[id] text`."""
    out: dict[int, str] = {}
    for line in prompt.splitlines():
        if line.startswith("[") and "]" in line:
            head, _, text = line.partition("]")
            try:
                out[int(head[1:])] = text.strip()
            except ValueError:
                continue
    return out


def render_call(args: dict) -> str:
    return ('<tool_call>{"name": "update_memory", "arguments": '
            + json.dumps(args, ensure_ascii=False) + "}</tool_call>")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-similarity", type=float, default=0.45,
                   help="trigram similarity at which a drop+add pair is a revision")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    rows = [json.loads(ln) for ln in args.pool.read_text(encoding="utf-8").splitlines()
            if ln.strip()]

    promoted = churn_seen = unrelated = numeral_blocked = 0
    examples: list[dict] = []
    out_rows: list[dict] = []

    for r in rows:
        if "CHUNK:" not in r["prompt"] or "tool_call" not in r["completion"]:
            out_rows.append(r)
            continue
        ops = parse_tool_calls(r["completion"])
        byid = prompt_points(r["prompt"])
        drops = [o for o in ops if isinstance(o, Drop) and o.pid in byid]
        adds = [o for o in ops if isinstance(o, Add)]
        # An `arc` alongside the pair is fine — it is carried through unchanged below. Only
        # the DROP/ADD structure has to be unambiguous, because a step dropping two points
        # and adding one has no single supersession to express.
        others = [o for o in ops if not isinstance(o, Drop | Add)]
        if (len(drops) != 1 or len(adds) != 1
                or len([o for o in ops if isinstance(o, Drop)]) != 1
                or any(type(o).__name__ != "Arc" for o in others)):
            out_rows.append(r)
            continue

        old, new = byid[drops[0].pid], adds[0].point
        # The harness's own detector, never a proxy.
        if restates_dropped(new, [old]):
            churn_seen += 1
            out_rows.append(r)
            continue
        # A revision may ADD figures — that is enrichment, and the replacement text is kept
        # whole while the original is journalled with a `superseded_by` link, so nothing is
        # lost. It may not LOSE them: a replacement that drops a figure the original carried
        # is the lossy-revision defect `gen_reversals.py` shipped for 68 rows, where the
        # replacement `add` discarded `key_term` and taught lossy revision on the exact
        # capability G1 measures. Blocking is therefore one-directional.
        lost = set(NUMERAL.findall(normalize(old))) - set(NUMERAL.findall(normalize(new)))
        if lost:
            numeral_blocked += 1
            out_rows.append(r)
            continue
        s = similarity(old, new)
        if s < args.min_similarity:
            unrelated += 1
            out_rows.append(r)
            continue

        m = TOOL_CALL.search(r["completion"])
        base = json.loads(m.group(0)).get("arguments") or {}
        new_args: dict[str, object] = {}
        if base.get("arc"):
            new_args["arc"] = base["arc"]
        new_args["revise"] = [{"id": drops[0].pid, "text": new}]
        out_rows.append({**r, "completion": render_call(new_args)})
        promoted += 1
        if len(examples) < 12:
            examples.append({"similarity": round(s, 3), "from": old, "to": new})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "pool": str(args.pool), "out": str(args.out),
        "min_similarity": args.min_similarity,
        "rows": len(out_rows),
        "promoted_to_revise": promoted,
        "left_as_drop_add_unrelated": unrelated,
        "left_alone_revision_would_lose_a_figure": numeral_blocked,
        "churn_rows_seen": churn_seen,
        "examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
