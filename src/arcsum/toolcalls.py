"""SPEC §4.1 v1.0: the `update_memory` tool call, parsed into the same `Op` list the
harness has always applied.

The step grammar changed from edit lines to one batched tool call per chunk; the memory,
the guards, the caps and the overflow rule did not. So this module's only job is
`parse_tool_calls(raw) -> list[Op]`, after which everything downstream — `guards.apply_ops`,
`Memory`, the eviction rule — is reached unchanged. Keeping `ops.Op` as the interface is
what makes the two protocols comparable on the same harness rather than two forks.

**Never raises**, exactly like `ops.parse_ops`. A model that emits malformed JSON must
produce a recorded `Malformed` op, not an exception: the whole point of `valid_op_rate` is
to measure how often that happens, which is impossible if it aborts the run.

**Ordering is load-bearing.** `drop` is applied before `add`, so a chunk that supersedes a
point can drop the stale one and add its replacement in a single call. §4.1's inversion
guard refuses a contradicting `ADD` unless the superseded point was dropped earlier in the
same step, and emission order is what it reads.
"""

from __future__ import annotations

import json
import re

from arcsum.ops import Add, Arc, Drop, Malformed, Nop, Op, Revise

#: Both the fenced form the chat template emits and a bare JSON object, because a
#: fine-tuned student is trained on the fenced form but a partially-trained one emits the
#: payload alone often enough that discarding it would understate what it actually did.
_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOOL_NAME = "update_memory"


def _as_list(value: object) -> list[str]:
    """`add`/`drop` accept a list or a bare string. A model that emits `"add": "點"`
    instead of `"add": ["點"]` has expressed the same intent, and treating that as
    malformed would score a formatting slip as a curation failure."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    return []


def parse_tool_calls(text: str) -> list[Op]:
    """Raw model output -> ops, in application order. Never raises."""
    if not text or not text.strip():
        return []

    blocks = _BLOCK.findall(text)
    if not blocks:
        stripped = text.strip()
        # A bare payload with no fence: accept it rather than lose the step.
        if stripped.startswith("{") and stripped.endswith("}"):
            blocks = [stripped]
        else:
            return [Malformed(text.strip()[:120], "no <tool_call> block")]

    ops: list[Op] = []
    for block in blocks:
        try:
            payload = json.loads(block)
        except (ValueError, TypeError):
            ops.append(Malformed(block[:120], "invalid JSON"))
            continue
        if not isinstance(payload, dict):
            ops.append(Malformed(block[:120], "payload is not an object"))
            continue

        name = payload.get("name")
        if name is not None and name != TOOL_NAME:
            # A call to a tool that does not exist is a real failure to record, not
            # something to silently skip.
            ops.append(Malformed(str(name)[:120], "unknown tool"))
            continue

        args = payload.get("arguments", payload if name is None else {})
        if isinstance(args, str):  # some models emit arguments as a JSON *string*
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                ops.append(Malformed(block[:120], "arguments string is not JSON"))
                continue
        if not isinstance(args, dict):
            ops.append(Malformed(block[:120], "arguments is not an object"))
            continue

        # drop BEFORE add: see module docstring — the inversion guard reads emission order.
        # `drop` accepts ids (v1.1) or text prefixes (v0.9/v1.0) in the same field. An int
        # is unambiguous, so the two cannot be confused; a numeric STRING is treated as an
        # id too, because a model that emits "drop": ["3"] has expressed the same intent
        # and refusing it on a quoting detail would be a parser tantrum, not a guard.
        raw_drop = args.get("drop")
        drop_items = raw_drop if isinstance(raw_drop, list) else _as_list(raw_drop)
        for item in drop_items:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                ops.append(Drop(pid=item))
            elif isinstance(item, str) and item.strip().isdigit():
                ops.append(Drop(pid=int(item.strip())))
            elif isinstance(item, str) and item.strip():
                ops.append(Drop(prefix=item.strip()))
        # `revise` is applied with the drops, before adds: it supersedes existing content.
        rev = args.get("revise")
        for item in rev if isinstance(rev, list) else ([rev] if isinstance(rev, dict) else []):
            if not isinstance(item, dict):
                continue
            pid, text = item.get("id"), item.get("text")
            ok_id = isinstance(pid, int) and not isinstance(pid, bool)
            if ok_id and isinstance(text, str) and text.strip():
                ops.append(Revise(pid, text.strip()))
            else:
                ops.append(Malformed(str(item)[:120], "revise needs int id and non-empty text"))
        arc = args.get("arc")
        if isinstance(arc, str) and arc.strip():
            ops.append(Arc(arc.strip()))
        for point in _as_list(args.get("add")):
            ops.append(Add(point.strip()))

    if not ops:
        # An explicit call with empty arguments is the v1.0 spelling of NOP — the model
        # was asked and answered "nothing here", which is NOT the same as saying nothing.
        return [Nop()]
    return ops


def render_tool_call(ops: list[Op]) -> str:
    """Ops -> the exact text a trained student should emit. Used to build supervision, so
    training targets and the parser above cannot drift apart."""
    args: dict[str, object] = {}
    drops = [(o.pid if o.pid else o.prefix) for o in ops if isinstance(o, Drop)]
    revs = [{"id": o.pid, "text": o.text} for o in ops if isinstance(o, Revise)]
    adds = [o.point for o in ops if isinstance(o, Add)]
    arcs = [o.text for o in ops if isinstance(o, Arc)]
    if drops:
        args["drop"] = drops
    if revs:
        args["revise"] = revs
    if arcs:
        args["arc"] = arcs[-1]  # only the last ARC can survive; earlier ones are replaced
    if adds:
        args["add"] = adds
    payload = {"name": TOOL_NAME, "arguments": args}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
