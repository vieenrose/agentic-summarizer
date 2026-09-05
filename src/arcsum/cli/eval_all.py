"""`arcsum-eval`: run every reference-free instrument against a served checkpoint and emit
ONE scorecard carrying the configuration that produced it.

    arcsum-eval --url http://127.0.0.1:8081 --protocol tool --label mixed-e3-best \\
        --corpus data/ly_phase3_v2 --out runs/mixed-e3/scorecard.json

    # fold in the reference-based gates from an existing paired run
    arcsum-eval ... --report runs/mixedbest-heldout/report_g2.json

**Why one command.** Every instrument in this project has been run as a separate
invocation with its own flags, and the cost is on the record: a probe scored under the
wrong protocol (0/27, read as "cannot revise", really 8/27); a curve measured against a
server that had failed to bind, so the previous checkpoint answered; probe scores quoted
from memory that did not reproduce. A single entry point that captures provenance once and
refuses when it cannot identify the server removes the class.

**What this deliberately does NOT do: repair, retry, or re-run to get a better number.**
It measures once and records. The project's guards follow "detect and record, never repair
in-loop" and the same applies one level up — an evaluator that retries until a metric looks
acceptable is selecting on the measurement.

**Read `deployment_mismatch` before trusting the output.** Pass `--deployed-cache-prompt`
to declare what the PRODUCT runs. On 2026-09-02 every gate ran `cache_prompt: false` while
the shipped demo ran the KV cache live across calls, and a checkpoint that churned badly in
deployment passed everything measured without it. The scorecard cannot detect that on its
own; it can only compare what you tell it against what it measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arcsum.agent import run_agent
from arcsum.backends.llama_server import LlamaServer
from arcsum.evalkit import behaviour, grounding
from arcsum.evalkit.provenance import capture
from arcsum.evalkit.scorecard import Check, Scorecard
from arcsum.transcript import parse_transcript


def _reference_checks(report_path: Path) -> list[Check]:
    """Lift already-computed gate verdicts out of a `cli.report` artifact.

    Read rather than recomputed: `metrics/stats.py` owns the paired protocol, the sign
    test and the below-`min_n` withholding, and duplicating any of that here would create
    a second answer to a question that must have exactly one.
    """
    blob = json.loads(report_path.read_text(encoding="utf-8"))
    out = []
    for g in blob.get("gates", []):
        out.append(
            Check(g["gate"], g.get("passed"), g.get("detail", ""), artifact=str(report_path))
        )
    for c in blob.get("comparisons", []):
        out.append(
            Check(
                f"delta_{c['metric']}",
                None,
                f"{c['wins']}/{c['losses']} p={c['p_value']:.3f}",
                score=c["mean_delta"],
                n=c["n"],
                artifact=str(report_path),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=("llama-server", "llama-cpp"),
        default="llama-server",
        help="'llama-cpp' runs the DEPLOYED stack in process. Measured "
        "2026-09-03: llama-server (cache on, GPU) and llama-cpp-python "
        "(CPU) gave 4 points/0 churn vs 1 point/4 churn on the SAME GGUF "
        "and the same meeting, so a flag on the HTTP server does not "
        "stand in for the deployed stack.",
    )
    p.add_argument("--gguf", default="", help="required for --backend llama-cpp")
    p.add_argument(
        "--n-gpu-layers", type=int, default=0, help="0 = CPU, matching the reference demo"
    )
    p.add_argument(
        "--n-threads",
        type=int,
        default=8,
        help="thread count changes floating-point reduction ORDER, so it can "
        "change greedy token choices at temperature 0. The reference demo "
        "runs 2. Not a performance knob for reproduction purposes.",
    )
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument(
        "--protocol",
        choices=("edit", "tool"),
        required=True,
        help="v0.9 checkpoints are 'edit', v1.0 are 'tool'; no safe default",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="directory of format-v2 .txt meetings, run reference-free",
    )
    p.add_argument("--label", default="")
    p.add_argument("--checkpoint", default="", help="e.g. 'checkpoint-626'")
    p.add_argument("--epoch", default="", help="e.g. 'best (626)' or 'last (939)'")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="an existing cli.report artifact whose gates are folded in",
    )
    p.add_argument(
        "--cache-prompt",
        choices=("true", "false"),
        default="false",
        help="the setting this run MEASURES under. Defaults false for "
        "reproducibility (the cache changes generation: 167 vs 700 chars, "
        "same seed, temperature 0). Set true to measure the configuration "
        "a llama-cpp-python product actually runs — the divergence that "
        "hid a shipped churn regression on 2026-09-02.",
    )
    p.add_argument(
        "--deployed-cache-prompt",
        choices=("true", "false"),
        default=None,
        help="what the PRODUCT runs; mismatch is reported, never silently ok",
    )
    p.add_argument("--max-tokens-step", type=int, default=512)
    p.add_argument("--max-tokens-synth", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if args.backend == "llama-cpp" and not args.gguf:
        print("[eval] REFUSED: --backend llama-cpp needs --gguf", file=sys.stderr)
        return 1
    cache_prompt = args.cache_prompt == "true"
    generation = {
        "backend": args.backend,
        "cache_prompt": cache_prompt,
        "seed": 0,
        "temperature": 0.0,
        "repeat_penalty": 1.1,
        "raw_completion": True,
        "max_tokens_step": args.max_tokens_step,
        "max_tokens_synth": args.max_tokens_synth,
    }
    if args.backend == "llama-cpp":
        generation["n_gpu_layers"] = args.n_gpu_layers
        generation["n_threads"] = args.n_threads
    # Refuses if /props is unreachable — never measures against an unidentified server.
    prov = capture(
        args.url,
        protocol=args.protocol,
        generation=generation,
        corpora=[("corpus", args.corpus)],
        label=args.label,
        checkpoint=args.checkpoint,
        epoch=args.epoch,
        gguf_path=args.gguf if args.backend == "llama-cpp" else "",
    )
    print(f"[eval] serving {prov.model_path}", file=sys.stderr)
    print(f"[eval] comparison_key {prov.comparison_key()[:16]}", file=sys.stderr)

    if args.backend == "llama-cpp":
        from arcsum.backends.llama_cpp import LlamaCpp

        # ONE object shared by both calls, and the cache left intact between steps —
        # that is how the demo runs, and the statefulness is part of what is measured.
        step = LlamaCpp(
            args.gguf,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.n_threads,
            max_tokens=args.max_tokens_step,
            reuse_cache=cache_prompt,
        )
        prose = LlamaCpp(
            args.gguf,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.n_threads,
            max_tokens=args.max_tokens_synth,
            repeat_penalty=1.1,
            reuse_cache=cache_prompt,
        )
    else:
        common = {
            "base_url": args.url,
            "seed": 0,
            "raw_completion": True,
            "extra": {"cache_prompt": cache_prompt},
        }
        step = LlamaServer(max_tokens=args.max_tokens_step, **common)
        prose = LlamaServer(max_tokens=args.max_tokens_synth, repeat_penalty=1.1, **common)

    files = sorted(args.corpus.glob("*.txt"))
    if not files:
        print(f"[eval] REFUSED: no transcripts in {args.corpus}", file=sys.stderr)
        return 1

    behaviours, groundings = [], []
    for f in files:
        source = f.read_text(encoding="utf-8")
        trace = run_agent(
            parse_transcript(source),
            step,
            synth_model=prose,
            protocol=args.protocol,
            on_step_error="skip",
        )
        b = behaviour.from_trace(f.stem, trace)
        behaviours.append(b)
        groundings.append(
            grounding.check(f.stem, trace.synthesis.prose.text if trace.synthesis else "", source)
        )
        print(
            f"[eval] {f.stem}: {b.points}pt {b.chars_per_point:.0f}ch/pt "
            f"{', '.join(b.flags) or 'clean'}",
            file=sys.stderr,
        )

    bs = behaviour.summarise(behaviours)
    gs = grounding.summarise(groundings)
    card = Scorecard(prov)

    # Descriptive (result=None): these are measurements, not gates. Promoting any of them
    # to pass/fail needs a threshold defended on more than one corpus, which does not
    # exist yet — and a threshold invented to make the current checkpoint pass is worse
    # than no threshold.
    card.add(
        Check(
            "clean_meetings",
            None,
            f"{bs.clean_meetings}/{bs.n_meetings} with no behaviour flag",
            score=float(bs.clean_meetings),
            n=bs.n_meetings,
        )
    )
    card.add(
        Check(
            "churn_rate",
            None,
            f"{bs.total_churn} events over {bs.total_steps} steps",
            score=bs.churn_rate,
            n=bs.total_steps,
        )
    )
    card.add(
        Check(
            "meetings_starved",
            None,
            "memory did not accumulate",
            score=float(bs.meetings_starved),
            n=bs.n_meetings,
        )
    )
    card.add(
        Check(
            "meetings_confabulating",
            None,
            "prose asserts far more than memory holds",
            score=float(bs.meetings_confabulating),
            n=bs.n_meetings,
        )
    )
    card.add(
        Check(
            "median_chars_per_point",
            None,
            "summary characters per RECORDED memory unit (working set + journal)",
            score=bs.median_chars_per_point,
            n=bs.n_meetings,
        )
    )
    # SPEC G5. The journal guarantees a recorded point REACHES synthesis; this is the
    # different question of whether synthesis then used it, and it is where the coverage
    # deficit now lives. Reported over summed counts, not a mean of per-meeting rates,
    # because the long meetings are the whole question and rate-averaging would hide them.
    card.add(
        Check(
            "retention",
            None,
            f"{bs.total_rendered} of {bs.total_recorded} recorded points reach the summary",
            score=bs.retention,
            n=bs.total_recorded,
        )
    )
    card.add(
        Check(
            "meetings_under_rendering",
            None,
            "memory was built and then largely not rendered",
            score=float(bs.meetings_under_rendering),
            n=bs.n_meetings,
        )
    )
    card.add(
        Check(
            "ungrounded_rate",
            None,
            f"{gs.total_ungrounded} of {gs.total_checked} specific claims absent "
            f"from their transcript",
            score=gs.ungrounded_rate,
            n=gs.total_checked,
        )
    )
    card.add(
        Check(
            "meetings_with_ungrounded",
            None,
            "at least one fabricated specific",
            score=float(gs.meetings_with_any),
            n=gs.n_meetings,
        )
    )

    # Persist what the aggregates were computed from. "33% ungrounded of 6 claims" and
    # "44% of 27" are not comparable without knowing whether the model asserted less or
    # asserted better, and only the rows answer that.
    card.attach(
        "behaviour",
        [
            {
                "meeting": b.meeting,
                "chunks": b.chunks,
                "points": b.points,
                # `points` is the WORKING SET; `recorded_points` is everything, including what the
                # journal retired. Both are persisted because a row showing 16 survivors out of 40
                # recorded means something entirely different from 16 out of 16.
                "recorded_points": b.recorded_points,
                "rendered_points": b.rendered_points,
                "retention": round(b.retention, 3),
                "memory_units": b.memory_units,
                "prose_chars": b.prose_chars,
                "churn_events": b.churn_events,
                "arc_frozen_steps": b.arc_frozen_steps,
                "nop_steps": b.nop_steps,
                "abstained": b.abstained,
                # `steps` is the denominator every rate here is over, and omitting it forced
                # consumers to substitute `chunks` — which differs whenever a step FAILED, so
                # every rate was quietly computed against the wrong base on exactly the runs
                # where something went wrong.
                "steps": b.steps,
                "failed_steps": b.failed_steps,
                "refused_ops": b.refused_ops,
                "attempted_ops": b.attempted_ops,
                # G4's inputs. `evalkit.latency` projects wall clock from a run's MEASURED token
                # profile because decode length is a property of the CHECKPOINT, not the device —
                # the RAFT pool's targets run 1.45x longer than gold, worth ~2 minutes against a
                # 20.00 min ceiling. These were added to `BehaviourReport` and not to this
                # serializer, so the first four RAFT scorecards report `decode_tokens` as 0 and
                # cannot be used for a G4 claim.
                # `hedge_points` is the polarity-inversion guard: the reading step recorded
                # a faithful question-form point (委員質疑…是否應加重刑責) and synthesis
                # deterministically asserted the OPPOSITE as fact. The standing rule is that
                # no deliberation-trained checkpoint ships without this count being checked
                # -- and it was never written here, so there was nothing to check.
                "hedge_points": b.hedge_points,
                "ungrounded_numbers": b.ungrounded_numbers,
                # `has_arc` is what makes `memory_units` interpretable: a run can legitimately
                # set a real ARC and zero POINTS, and reading the ratio without it reported
                # that as infinite confabulation.
                "has_arc": b.has_arc,
                "prefill_tokens": b.prefill_tokens,
                "decode_tokens": b.decode_tokens,
                "chars_per_point": (
                    None if b.chars_per_point == float("inf") else round(b.chars_per_point, 1)
                ),
                "flags": list(b.flags),
            }
            for b in behaviours
        ],
    )
    card.attach(
        "grounding",
        [
            {"meeting": g.meeting, "n_checked": g.n_checked, "ungrounded": list(g.ungrounded)}
            for g in groundings
        ],
    )

    if args.report:
        for c in _reference_checks(args.report):
            card.add(c)

    if args.deployed_cache_prompt is not None:
        deployed = {"cache_prompt": args.deployed_cache_prompt == "true"}
        mismatch = card.deployment_mismatch(deployed)
        if mismatch:
            for k, (measured, shipped) in mismatch.items():
                print(
                    f"[eval] DEPLOYMENT MISMATCH {k}: measured={measured!r} "
                    f"shipped={shipped!r} — these numbers do not describe the deployed "
                    f"configuration",
                    file=sys.stderr,
                )
            card.add(
                Check(
                    "deployment_match",
                    False,
                    "; ".join(
                        f"{k}: measured {a!r} != shipped {b!r}" for k, (a, b) in mismatch.items()
                    ),
                )
            )
        else:
            card.add(Check("deployment_match", True, "measured config matches deployment"))

    card.write(args.out)
    print(f"\n[eval] {args.label or prov.model_path}", file=sys.stderr)
    print(
        f"[eval]   clean {bs.clean_meetings}/{bs.n_meetings} | churn {bs.churn_rate:.1%} "
        f"| median {bs.median_chars_per_point:.0f}ch/pt "
        f"| ungrounded {gs.ungrounded_rate:.1%} of {gs.total_checked}",
        file=sys.stderr,
    )
    for c in card.failed:
        print(f"[eval]   FAIL {c.name}: {c.reason}", file=sys.stderr)
    print(f"[eval] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
