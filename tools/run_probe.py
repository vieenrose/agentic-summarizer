"""Run the G1 revision probe against a served checkpoint and RECORD the artifact.

    python tools/run_probe.py --url http://127.0.0.1:8081 --out runs/sft-dropv6/g1

`cli/probe.py` splits into `dump` (write the probe transcripts) and `score` (grade a
`{name: prose}` JSON), deliberately leaving the middle step — actually running the agent —
to the caller. That gap is why no probe artifact exists for `sft-dropv2` even though its
`g_report_final.json` records `g1_passed: true`: the flag was asserted on the report
command line, and nothing on disk backs it. Re-running the probe today, dropv2 fails both
cases. This tool closes the gap so a G1 claim always has a reproducible artifact behind
it.

**Every knob that changes generation is written into the artifact**, because trap 4
(`CLAUDE.md`) measured that llama.cpp's prompt cache alone can turn a 167-character
answer into a 700-character one at `temperature=0` with the same model and seed. A G1
verdict without its configuration is not reproducible, and a G1 verdict that disagrees
with an older one is uninterpretable unless both configurations are known.

`cache_prompt` is pinned False for the same reason §5.2 numbers are: reproducibility
given a cache state is not reproducibility across runs, since the cache depends on which
meeting ran before.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import arcsum.agent as agent_mod  # noqa: E402
from arcsum.agent import run_agent  # noqa: E402
from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.cli.probe import _result_to_dict, score_results  # noqa: E402
from arcsum.probe import probe_meetings  # noqa: E402
from arcsum.prompts import PROMPT_VERSION, build_step_prompt  # noqa: E402
from arcsum.prompts import step_system_prompt as _base_step_sys  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--out", type=Path, required=True, help="prefix; writes _results/_report")
    p.add_argument("--label", default="", help="checkpoint name, recorded in the artifact")
    p.add_argument("--protocol", choices=("edit", "tool"), default="edit",
                   help="SPEC §4.1 step grammar for the agent arm")
    p.add_argument("--max-tokens-step", type=int, default=256)
    p.add_argument("--max-tokens-synth", type=int, default=1200)
    p.add_argument("--repeat-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--extra-sys-line",
        default="",
        help="append one line to the step SYS prompt (experiment only; a kept change "
        "must become a PROMPT_VERSION bump, not a flag)",
    )
    p.add_argument(
        "--legacy-prompt",
        action="store_true",
        help="build sys-v1 step prompts (no POSITION line) — for grading a pre-sys-v2 "
        "checkpoint on the prompt it was actually trained for",
    )
    args = p.parse_args(argv)

    if args.extra_sys_line:
        line = args.extra_sys_line
        agent_mod.step_system_prompt = lambda: _base_step_sys() + "\n" + line

    if args.legacy_prompt:
        # Patched at the agent's call site, not in `prompts`, so the recorded
        # PROMPT_VERSION still names what the module actually is.
        agent_mod.build_step_prompt = lambda m, c, total=None: build_step_prompt(m, c)

    common = {
        "base_url": args.url,
        "seed": args.seed,
        "raw_completion": True,
        "extra": {"cache_prompt": False},
    }
    step = LlamaServer(max_tokens=args.max_tokens_step, **common)
    # Prose gets the repetition penalty; reading steps must not (trap 2: it would punish
    # the literal ADD/DROP/ARC tokens the op format needs).
    prose = LlamaServer(
        max_tokens=args.max_tokens_synth, repeat_penalty=args.repeat_penalty, **common
    )

    results: dict[str, str] = {}
    traces: dict[str, list[dict]] = {}
    for meeting in probe_meetings():
        trace = run_agent(meeting.utterances, step, synth_model=prose,
                          protocol=args.protocol)
        results[meeting.name] = trace.synthesis.prose.text
        traces[meeting.name] = [
            {"step": i, "raw": st.raw, "prompt_head": st.user.split("\n", 1)[0]}
            for i, st in enumerate(trace.steps)
        ]
        print(f"[probe] {meeting.name}: {len(trace.steps)} steps, "
              f"{trace.synthesis.prose.chars} chars", file=sys.stderr)

    scored, g1_passed = score_results(results)
    report = {"g1_passed": g1_passed, "results": [_result_to_dict(r) for r in scored]}
    config = {
        "label": args.label,
        "protocol": args.protocol,
        "url": args.url,
        "prompt_version": PROMPT_VERSION,
        "legacy_prompt": args.legacy_prompt,
        "extra_sys_line": args.extra_sys_line,
        "max_tokens_step": args.max_tokens_step,
        "max_tokens_synth": args.max_tokens_synth,
        "repeat_penalty": args.repeat_penalty,
        "seed": args.seed,
        "cache_prompt": False,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
    }

    out_results = args.out.with_name(args.out.name + "_results.json")
    out_report = args.out.with_name(args.out.name + "_report.json")
    out_results.write_text(
        json.dumps({"config": config, "prose": results, "steps": traces}, ensure_ascii=False,
                   indent=1),
        encoding="utf-8",
    )
    out_report.write_text(
        json.dumps({"config": config, **report}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[probe] G1: {'PASS' if report['g1_passed'] else 'FAIL'} -> {out_report}",
          file=sys.stderr)
    for r in report["results"]:
        print(f"[probe]   {r['name']}: {'PASS' if r['passed'] else 'FAIL'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
