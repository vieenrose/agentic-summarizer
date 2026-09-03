"""Rewrite `SYNTHESIZE` targets so each is a FUNCTION OF ITS INPUT.

    python tools/regen_synth.py --pool data/staging/sft_pool_mixed.jsonl \\
        --url http://127.0.0.1:8082 --out data/staging/synth_regen.jsonl

**The defect.** Measured on `sft_pool_mixed.jsonl`: of 3,376 specific claims in the 450
`SYNTHESIZE` targets, **1,347 (39.9%) do not appear in the memory the target was written
from**, spread over 45% of rows. That is not a labelling slip — it is how the corpus was
built. SPEC §2.2 stage 3 composes a whole-meeting summary from the segment minutes, and the
row pairs `(final memory) -> (that gold summary)`. The gold legitimately contains detail the
memory never held, so the model is shown, 450 times, that the correct response to a memory
is a summary containing things absent from it. **The student reproduces the rate**: 44%
ungrounded on real ASR.

**Why regeneration rather than filtering.** Filtering was tried first and is refuted
(`runs/clean-e3`): dropping offending rows removes 37% of synthesis supervision,
concentrated at high occupancy — the regime the synthesis cliff already lives in — and the
resulting checkpoint stopped asserting specifics at all (5 claims over 20 meetings against
`v5`'s 27), with starved meetings rising 6 -> 11. **Deleting the fabrication by deleting the
content is not a fix.** Regeneration repairs the row: the teacher writes a summary from the
memory ALONE, so the target is derivable from the input by construction.

**The teacher is asked to do exactly the student's task**, through
`prompts.synth_system_prompt()` and the row's own stored prompt — not a bespoke
instruction. Any drift between the two prompts would put the target off-distribution from
the input the student actually sees at inference, which is the same class of defect being
repaired.

**Every rewrite is verified before it is accepted.** A regenerated target that still
asserts ungrounded specifics is a target that failed the job, and is reported rather than
silently kept; `--max-ungrounded` sets the tolerance. Rows whose rewrite fails verification
keep their ORIGINAL target and are counted, so the pool never silently shrinks — the
failure mode this tool exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.evalkit import grounding  # noqa: E402
from arcsum.prompts import synth_system_prompt  # noqa: E402
from arcsum.prose import finalize  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="the regenerated SYNTHESIZE rows only; merge with clean_pool.py")
    p.add_argument("--url", default="http://127.0.0.1:8082", help="the TEACHER server")
    p.add_argument("--max-ungrounded", type=int, default=0,
                   help="ungrounded specifics tolerated in a REGENERATED target before it "
                        "is rejected and the original kept")
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0, help="0 = all; for a smoke run")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    raw = args.pool.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(ln) for ln in raw if ln.strip()]
    # `build_synth_prompt` is `build_memory_view`, so a SYNTHESIZE row is exactly the one
    # whose prompt is a rendered memory. Reading rows carry a CHUNK, the baseline's reduce
    # rows start with SUMMARIES. Getting this wrong silently regenerates the wrong
    # population — an earlier analysis in this project classified by "no CHUNK" and swept
    # up the baseline's reduce rows too.
    syn = [r for r in rows if r["prompt"].startswith("MEMORY:")]
    if args.limit:
        syn = syn[: args.limit]
    if not syn:
        print("[regen] REFUSED: no SYNTHESIZE rows found", file=sys.stderr)
        return 1

    # `enable_thinking: False` is MANDATORY for this teacher and is honored ONLY on the
    # jinja path (CLAUDE.md trap 10). Serve it with `--jinja`. Without this the model
    # emits its reasoning instead of the summary — measured here: the first attempt
    # returned `我們需要回答使用者：…` ("we need to answer the user: …") as the target,
    # 1,483 characters of deliberation, and every row was correctly rejected by the
    # grounding check for asserting figures the memory never held.
    teacher = LlamaServer(base_url=args.url, max_tokens=args.max_tokens,
                          repeat_penalty=1.1, seed=0, raw_completion=True,
                          extra={"cache_prompt": False,
                                 "chat_template_kwargs": {"enable_thinking": False}})
    system = synth_system_prompt()

    kept = repaired = failed = errored = 0
    before_ung = after_ung = 0
    out_rows = []
    for i, r in enumerate(syn, 1):
        base = grounding.check("", r["completion"], r["prompt"])
        before_ung += base.n_ungrounded
        try:
            new = finalize(teacher(system, r["prompt"]), token_len=heuristic_token_len)
        except Exception as exc:  # one bad row must not lose the whole pass
            errored += 1
            out_rows.append(r)
            after_ung += base.n_ungrounded
            print(f"[regen] {i}/{len(syn)} {r['meeting']}: ERROR {exc}", file=sys.stderr)
            continue

        check = grounding.check("", new.text, r["prompt"])
        # A rewrite that is empty, off-language, or still ungrounded is not an improvement.
        ok = (new.text and not new.lang_flags
              and check.n_ungrounded <= args.max_ungrounded
              and check.n_ungrounded < base.n_ungrounded + 1)
        if ok:
            repaired += 1
            after_ung += check.n_ungrounded
            out_rows.append({**r, "completion": new.text, "regen": True})
        else:
            failed += 1
            after_ung += base.n_ungrounded
            out_rows.append(r)
        if i % 25 == 0:
            print(f"[regen] {i}/{len(syn)} repaired={repaired} kept-original={failed}",
                  file=sys.stderr)
    kept = failed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "pool": str(args.pool), "out": str(args.out),
        "synthesize_rows": len(syn), "repaired": repaired,
        "kept_original": kept, "errors": errored,
        "ungrounded_before": before_ung, "ungrounded_after": after_ung,
        "max_ungrounded": args.max_ungrounded,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
