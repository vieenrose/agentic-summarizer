#!/usr/bin/env python3
"""Counterfactual twin samples for the 270M (PLAN 0c, intervention v6).

The trace data is confounded: revision chunks always follow setup chunks, so chunk
surface correlates with state content — a model can learn "decision-language chunk
=> UPD" without ever reading the state (measured: op choice flips on filler-line
presence). These twins break the confound:

  original: STATE (has bullet B) + CHUNK C -> target UPD B->B'
  twin:     STATE (WITHOUT B)      + CHUNK C -> target ADD B'

The chunk is byte-identical; only the state (bullet removed) and the op differ. If
the 270M can learn state-gated op selection, this data forces it; if it cannot, that
is the cleanest possible measured-impossibility evidence.

    python tools/build_counterfactual.py data/traces_v2 --out data/sft/270m-twins.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.ops import Add, Upd, parse_ops  # noqa: E402
from voxsum.prompts import PROMPT_VERSION  # noqa: E402
from voxsum.transcript import sec_to_clock  # noqa: E402


def _call(name: str, **args: str) -> str:
    body = ",".join(f"{k}:<escape>{v}<escape>" for k, v in args.items() if v != "")
    return f"<start_function_call>call:{name}{{{body}}}<end_function_call>"


def strip_state_block(user: str, section: str, prefix: str) -> str | None:
    """Remove the bullet matching (section, prefix) from the rendered STATE block.

    The user prompt is `STATE:\n<render>\nCHUNK:\n<chunk>`; the rendered state has
    `SECTION:` headers and `- <text> [m:ss]` bullets. Returns the rebuilt user prompt,
    or None if the bullet cannot be located (twin not constructible).
    """
    m = re.match(r"STATE:\n(.*?)\nCHUNK:\n(.*)$", user, re.S)
    if not m:
        return None
    state_block, chunk = m.group(1), m.group(2)
    lines = state_block.splitlines()
    out: list[str] = []
    in_section = False
    found = False
    for line in lines:
        if line.endswith(":") and line.rstrip(":").strip() in (
            "TITLE", "SUMMARY", "DECISIONS", "ACTIONS", "OPEN", "TOPICS",
        ):
            in_section = line.rstrip(":").strip() == section
            out.append(line)
            continue
        if in_section and line.startswith("- ") and line[2:].startswith(prefix):
            found = True
            continue  # drop the bullet
        out.append(line)
    if not found:
        return None
    return f"STATE:\n{chr(10).join(out)}\nCHUNK:\n{chunk}"


def build_twins(record: dict) -> list[dict]:
    """One twin per accepted UPD in the record (DELs and CMPs are skipped)."""
    twins = []
    for op in parse_ops(record["target"]):
        if not isinstance(op, Upd) or op.prefix is None or op.anchor is None:
            continue
        new_user = strip_state_block(record["user"], op.section, op.prefix)
        if new_user is None or new_user == record["user"]:
            continue
        twin = dict(record)
        twin["meeting"] = f"{record['meeting']}~twin"
        twin["step"] = f"{record['step']}t"
        twin["user"] = new_user
        twin["target"] = _call(
            "ADD",
            section=op.section,
            bullet=op.bullet,
            anchor=sec_to_clock(op.anchor),
        )
        twin["is_nop"] = False
        twin["counterfactual"] = True
        twins.append(twin)
    return twins


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tracedir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    twins: list[dict] = []
    stats: Counter = Counter()
    manifest = json.loads(
        (Path(__file__).resolve().parent.parent / "data/transcripts/manifest.json").read_text()
    )
    split_of = {row["meeting_id"]: row["split"] for row in manifest}
    for path in sorted(args.tracedir.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            if split_of.get(r.get("meeting"), "train") != "train":
                stats["eval_meeting_skipped"] += 1
                continue
            made = build_twins(r)
            twins.extend(made)
            stats["records"] += 1
            stats["twins"] += len(made)
            stats["upd_total"] += sum(1 for o in parse_ops(r["target"]) if isinstance(o, Upd))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for t in twins:
            sink.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[twins] {stats['records']} records, {stats['upd_total']} UPDs, "
          f"{stats['twins']} twins written -> {args.out}")
    langs = Counter(t["lang"] for t in twins)
    print(f"[twins] by lang: {dict(langs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
