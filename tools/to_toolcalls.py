"""Convert the edit-line SFT pool into SPEC §4.1 v1.0's `update_memory` tool-call format.

    python tools/to_toolcalls.py --pool data/staging/sft_pool_p7.jsonl \\
        --out data/staging/sft_pool_tools.jsonl

Same supervision, same chunks, same memory states — only the OUTPUT SYNTAX changes, so a
difference in the result is attributable to the protocol rather than to the data.

Rendering goes through `toolcalls.render_tool_call`, the same function the parser
round-trips against in `tests/test_toolcalls.py`, and the system prompt comes from
`prompts.tool_step_system_prompt()` — the one the agent actually sends. Neither is
restated here: a second copy of a prompt that drifts from the first is the bug
`prompts.py` exists to prevent.

**Non-step rows are KEPT UNCHANGED, not excluded.** Map, reduce and synthesis rows are
prose and have no tool-call equivalent, but the agent still has to perform synthesis: the
reading loop ends with a SYNTHESIZE call that turns memory into the §3 prose product.

Excluding them was measured to be fatal. A first Qwen build trained on reading steps alone
produced summaries averaging **101 characters against the baseline's 738**, with 11 of 40
under 80 characters and the worst case emitting a bare `<think> </think>` and stopping —
the model had simply never seen the synthesis prompt. All three G3 gates failed with the
agent LOSING to its own baseline (rouge1 8/40 wins, -0.152). One model learning two output
formats keyed on prompt shape is what the v0 pools already did.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.ops import parse_ops  # noqa: E402
from arcsum.prompts import TOOLCALL_PROMPT_VERSION, tool_step_system_prompt  # noqa: E402
from arcsum.toolcalls import render_tool_call  # noqa: E402


def completion_to_tool_call(completion: str) -> str | None:
    """Edit lines -> the ONE batched call §4.1 v1.0 specifies.

    Returns None when nothing parses: a row with an empty target would teach the model to
    say nothing, which is a different lesson from the empty `arguments` that spells NOP.
    """
    ops = parse_ops(completion)
    if not ops:
        return None
    return render_tool_call(ops)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    system = tool_step_system_prompt()
    rows = [
        json.loads(ln)
        for ln in args.pool.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    kept = dropped = passthrough = 0
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            if not (r["prompt"].startswith("POSITION:") or "\nCHUNK:\n" in r["prompt"]):
                # prose row (map / reduce / synthesis): carried through verbatim so the
                # student still learns to synthesise.
                f.write(json.dumps({**r, "prompt_version": TOOLCALL_PROMPT_VERSION},
                                   ensure_ascii=False) + "\n")
                passthrough += 1
                continue
            call = completion_to_tool_call(r["completion"])
            if call is None:
                dropped += 1
                continue
            f.write(json.dumps({**r, "system": system, "completion": call,
                                "prompt_version": TOOLCALL_PROMPT_VERSION},
                               ensure_ascii=False) + "\n")
            kept += 1

    print(f"[tools] {kept} reading steps converted, {dropped} unparseable dropped, "
          f"{passthrough} prose rows carried through -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
