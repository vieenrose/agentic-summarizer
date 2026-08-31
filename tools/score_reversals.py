"""Run and score the INDEPENDENT reversal probe set (`data/reversals_probe`).

    python tools/score_reversals.py --url http://127.0.0.1:8081 \\
        --corpus data/reversals_probe --out runs/sft-dropv6/revprobe

Scores exactly what `arcsum.probe.score_probe` scores, against the planted plan rather
than a hand-written expectation:

* `states_later`      — the subject (or its key term) AND the LATE outcome word appear.
* `states_earlier_as_current` — the EARLY outcome word appears while the late one does
  not, i.e. the summary reports the superseded decision as if it still stood. This is the
  failure G1 exists to catch and it alone fails a case.

Why a separate scorer at all: G1 is two hand-built cases, and `tools/gen_reversals.py`
trains this capability, so a G1 pass would no longer be independent evidence. This set is
generated from scenarios that share no subject, key term, domain or outcome vocabulary
with the training scenarios, so it is the check that a passing G1 is not pattern match.

**Baseline first.** Always record the score BEFORE any reversal training, or there is no
way to tell an improvement from a set that was easy to begin with.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from gen_reversals import CONTROL_SCENARIOS, PROBE_SCENARIOS  # noqa: E402
from gen_reversals import outcome_variants as _outcome_variants  # noqa: E402

from arcsum.agent import run_agent  # noqa: E402
from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.ops import render_op  # noqa: E402
from arcsum.transcript import parse_transcript  # noqa: E402


def _subject_variants(sc) -> tuple[str, ...]:
    """The key term, the full subject, and the subject with its 案-suffix trimmed —
    a summary saying 「農產直銷市集」 has identified the subject just as surely as one
    saying 「農產直銷市集設置案」."""
    trimmed = sc.subject
    for suf in ("設置案", "汰換案", "設定案", "開放案", "採購案", "案"):
        if trimmed.endswith(suf):
            trimmed = trimmed[: -len(suf)]
            break
    return tuple({sc.key_term, sc.subject, trimmed})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--corpus", type=Path, default=REPO / "data/reversals_probe")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default="")
    p.add_argument("--protocol", choices=("edit", "tool"), default="edit",
                   help="SPEC §4.1 step grammar for the agent arm")
    p.add_argument("--control", action="store_true",
                   help="score the NO-REVERSAL control set: the decision stands, and "
                        "reporting a withdrawal that never happened is the failure")
    args = p.parse_args(argv)

    common = {"base_url": args.url, "seed": 0, "raw_completion": True,
              "extra": {"cache_prompt": False}}
    step = LlamaServer(max_tokens=256, **common)
    prose = LlamaServer(max_tokens=1200, repeat_penalty=1.1, **common)

    scenarios = CONTROL_SCENARIOS if args.control else PROBE_SCENARIOS
    #: Words that assert a reversal. On the control set NONE of these may appear about the
    #: subject: a model trained to handle withdrawals can start seeing them everywhere,
    #: and a fabricated withdrawal is a worse deployment defect than a missed one.
    false_reversal = ("撤回", "撤銷", "否決", "駁回", "退回", "終止", "中止", "廢止",
                      "取消", "暫緩", "緩議", "保留", "不予", "收回", "停止")

    results = []
    errors: list[dict] = []
    for sc in scenarios:
        f = next(iter(sorted(args.corpus.glob(f"{sc.slug}-*.txt"))), None)
        if f is None or not f.exists():
            continue
        # `skip` covers a failed READING step; the SYNTHESIS call can 500 too (trap 3),
        # and that raises out of `run_agent`. Both must be survivable: one bad scenario
        # aborting the run would leave one checkpoint unscored while its comparator
        # scored fine, which is a silently unfair comparison rather than a visible gap.
        # An errored scenario is recorded as UNSCORED, never as a probe FAIL.
        try:
            trace = run_agent(parse_transcript(f.read_text(encoding="utf-8")), step,
                              synth_model=prose, on_step_error="skip",
                              protocol=args.protocol)
        except RuntimeError as exc:
            errors.append({"slug": sc.slug, "error": repr(exc)[:200]})
            print(f"[rev-probe] {sc.slug}: UNSCORED ({exc})", file=sys.stderr)
            continue
        text = trace.synthesis.prose.text
        subject_ok = any(v in text for v in _subject_variants(sc))
        early_ok = any(v in text for v in _outcome_variants(sc.early))
        if args.control:
            invented = [w for w in false_reversal if w in text]
            states_later = subject_ok and early_ok  # the decision AS IT STANDS
            states_earlier = False
            passed = subject_ok and not invented
        else:
            late_ok = any(v in text for v in _outcome_variants(sc.late))
            invented = []
            states_later = subject_ok and late_ok
            states_earlier = early_ok and not late_ok
            passed = states_later and not states_earlier
        results.append({
            "slug": sc.slug, "subject_present": subject_ok,
            "states_later": states_later, "states_earlier_as_current": states_earlier,
            "passed": passed, "invented_reversal": invented,
            "prose": text,
            "memory_points": [pt.text for pt in trace.memory.points],
            "failed_steps": trace.failed_steps,
            "steps": [s.raw for s in trace.steps],
            # Per-op verdicts, so a `key_term` that was emitted but is absent from final
            # memory can be attributed. Without this the two mechanisms are
            # indistinguishable in the artifact: (a) the harness REFUSED the op
            # (`point too long`, `duplicate point`, contradiction guard) -- a supervision
            # or cap problem we can fix; (b) a later step DROPped it and the replacement
            # did not carry the term -- `qwen-tools-v6`'s lossy revision, which is
            # revision-specific and points at the corpus instead. `tools/loss_map.py`
            # measured emitted 8/11 vs memory 2/11 on the v5 set, so this gap is where
            # G1 is actually being lost and guessing at its cause is not good enough.
            "op_verdicts": [
                {
                    "step": st.index,
                    "op": render_op(a.op),
                    "applied": a.applied,
                    "reason": a.reason,
                    "note": a.note,
                }
                for st in trace.steps
                for a in st.outcome.results
            ],
        })
        print(f"[rev-probe] {sc.slug}: {'PASS' if results[-1]['passed'] else 'FAIL'} "
              f"(subject={subject_ok} later={states_later} stale={states_earlier})",
              file=sys.stderr)

    passed = sum(1 for r in results if r["passed"])
    report = {
        "label": args.label,
        "protocol": args.protocol,
        "url": args.url,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n": len(results),
        "passed": passed,
        "unscored": errors,
        "results": results,
    }
    args.out.with_name(args.out.name + "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[rev-probe] {passed}/{len(results)} passed"
          f"{f', {len(errors)} UNSCORED' if errors else ''} -> {args.out}_report.json",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
