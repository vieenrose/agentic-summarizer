"""Migrate a `tools-v1` SFT pool to SPEC §4.1 **v1.1** (`tools-v2`).

    python tools/migrate_pool_v11.py --pool data/staging/sft_pool_spec.jsonl \\
        --out data/staging/sft_pool_v11.jsonl --report runs/migrate-v11-report.json

**Why this cannot be a search-and-replace.** v1.1 renders points as `[id] text` and
addresses them by id, so every stored prompt is stale and every stored `drop` prefix has
to become the id of whatever point that prefix actually matched.

**And why ids come from the STORED PROMPT, not from a replay.** Replaying the pool's own
completions to rebuild memory was tried first and is wrong: the pool has GAPS — rows
removed by the churn filter, the grounding filter, and the replay-clean filter upstream —
so a replay diverges from the state each prompt actually shows, and 738 of 1,872 drop
prefixes (39%) failed to resolve. The prompt is the state the row was trained against and
is therefore authoritative. Ids are assigned by walking a meeting's steps in order and
keeping a text→id map, so a point that persists across steps keeps its id exactly as the
runtime harness would give it.

**Training on unmigrated rows would be the stale-data failure again.** The model would see
`[3] text` in the prompt and be taught to answer with a text prefix. This project has
already paid for that once: 68 reversal rows kept a `tools-v1` shape after the generator
was fixed, taught lossy revision across every checkpoint, and were only found by tracing
`key_term` through a loss map. `TOOLCALL_PROMPT_VERSION` exists to make the mismatch
loud; this tool exists to remove it.

**The valuable part is DROP+ADD -> `revise`, and it is deliberately conservative.**
v1.0 had no way to say "this point is now wrong, here is the correction" in one act, so
genuine revision and mere churn have the same shape in the data. Converting every DROP+ADD
would launder churn into a sanctioned op and teach the model that rewriting what it already
had is correct. So:

* **related but CHANGED** -> `revise` (the reversal case; this is the supervision v1.1 adds)
* **near-identical** -> churn, detected with `guards.restates_dropped`, the same function
  the runtime uses, and the ROW IS DROPPED rather than converted
* **unrelated** -> left as separate `drop` + `add`, because it is a close-out plus a new
  point, not a supersession

A step whose ops do not replay cleanly is DROPPED, never rewritten — `mix_phase4.py`'s
rule, on the same "never invent supervision" principle.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.guards import restates_dropped  # noqa: E402
from arcsum.memory import MIN_PREFIX_TOKENS, normalize  # noqa: E402
from arcsum.prompts import TOOLCALL_PROMPT_VERSION  # noqa: E402

TOOL_CALL = re.compile(r'\{"name".*\}', re.S)


def parse_args_json(completion: str) -> dict | None:
    m = TOOL_CALL.search(completion)
    if not m:
        return None
    try:
        return json.loads(m.group(0)).get("arguments") or {}
    except (json.JSONDecodeError, AttributeError):
        return None


def render_call(a: dict) -> str:
    return ('<tool_call>{"name": "update_memory", "arguments": '
            + json.dumps(a, ensure_ascii=False) + "}</tool_call>")


def related(dropped: str, added: str) -> bool:
    """Same subject? Shared leading run of at least `MIN_PREFIX_TOKENS` characters.

    Uses the same threshold `Memory.find` uses for prefix addressing, so "related enough
    to be a supersession" and "specific enough to address" are one decision, not two.
    """
    a, b = normalize(dropped), normalize(added)
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n >= MIN_PREFIX_TOKENS


def read_state(prompt: str) -> tuple[str, list[str]]:
    """`(arc, points)` exactly as the stored prompt shows them — the authoritative state."""
    head = prompt.partition("CHUNK:")[0]
    arc = ""
    points: list[str] = []
    for ln in head.splitlines():
        if ln.startswith("ARC: "):
            arc = ln[5:].strip()
        elif ln.startswith("- "):
            points.append(ln[2:].strip())
    return ("" if arc == "-" else arc), points


def render_state(prompt: str, arc: str, ids: list[int], points: list[str]) -> str:
    """Rewrite the MEMORY block as v1.1 renders it, leaving POSITION and CHUNK
    byte-identical — the chunk is the bulk of the prompt and re-deriving it would risk
    changing what the model reads."""
    head, sep, chunk = prompt.partition("CHUNK:")
    if not sep:
        return prompt
    pos = head.split("MEMORY:")[0]
    lines = [f"ARC: {arc or '-'}", "POINTS:"]
    lines.extend(f"[{i}] {t}" for i, t in zip(ids, points, strict=True))
    if not points:
        lines.append("-")
    return f"{pos}MEMORY:\n" + "\n".join(lines) + f"\n\n{sep}{chunk}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    raw = args.pool.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(ln) for ln in raw if ln.strip()]
    reading = [r for r in rows if "CHUNK:" in r["prompt"]]
    other = [r for r in rows if "CHUNK:" not in r["prompt"]]

    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for r in reading:
        by_meeting[r["meeting"]].append(r)

    stats = {"reading_in": len(reading), "reading_out": 0, "revise_created": 0,
             "churn_dropped": 0, "map_rows_passed": 0, "id_drops": 0,
             "prefix_unmatched": 0}
    out_rows: list[dict] = []

    for steps in by_meeting.values():
        steps.sort(key=lambda r: int(r["step"]))
        pid_of: dict[str, int] = {}
        next_pid = 0
        for r in steps:
            a = parse_args_json(r["completion"])
            if a is None:
                # A prose completion on a CHUNK prompt is the baseline's MAP row, not a
                # reading step. It has no memory rendering and no ops, so it passes
                # through untouched rather than being discarded.
                out_rows.append({**r, "prompt_version": TOOLCALL_PROMPT_VERSION})
                stats["map_rows_passed"] += 1
                continue

            arc, points = read_state(r["prompt"])
            ids: list[int] = []
            for t in points:
                key = normalize(t)
                if key not in pid_of:
                    next_pid += 1
                    pid_of[key] = next_pid
                ids.append(pid_of[key])

            drops = [d for d in (a.get("drop") or []) if isinstance(d, str)]
            adds = [x for x in (a.get("add") or []) if isinstance(x, str)]

            resolved: list[tuple[int, str]] = []
            for prefix in drops:
                key = normalize(prefix)
                hits = [(i, t) for i, t in zip(ids, points, strict=True)
                        if normalize(t).startswith(key)]
                if len(hits) == 1:
                    resolved.append(hits[0])
                else:
                    stats["prefix_unmatched"] += 1

            new_args: dict[str, object] = {}
            revises: list[dict] = []
            if len(resolved) == 1 and len(adds) == 1:
                pid, old_text = resolved[0]
                if restates_dropped(adds[0], [old_text]):
                    stats["churn_dropped"] += 1
                    continue
                if related(old_text, adds[0]):
                    revises.append({"id": pid, "text": adds[0]})
                    adds, resolved = [], []

            drop_ids = [pid for pid, _ in resolved]
            if drop_ids:
                new_args["drop"] = drop_ids
                stats["id_drops"] += len(drop_ids)
            if revises:
                new_args["revise"] = revises
                stats["revise_created"] += len(revises)
            if a.get("arc"):
                new_args["arc"] = a["arc"]
            if adds:
                new_args["add"] = adds

            out_rows.append({**r,
                             "prompt": render_state(r["prompt"], arc, ids, points),
                             "completion": render_call(new_args),
                             "prompt_version": TOOLCALL_PROMPT_VERSION})
            stats["reading_out"] += 1

    # Synthesis / map / reduce rows carry no memory rendering that changed shape, but the
    # version stamp must move with them or the pool mixes two protocols in one file.
    for r in other:
        out_rows.append({**r, "prompt_version": TOOLCALL_PROMPT_VERSION})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {**stats, "non_reading_rows": len(other), "total_out": len(out_rows),
              "pool": str(args.pool), "out": str(args.out),
              "prompt_version": TOOLCALL_PROMPT_VERSION}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
