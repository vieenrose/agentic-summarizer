"""Regenerate `SYNTHESIZE` supervision at the JOURNAL's shape and size (SPEC §4.1 v1.1).

    python tools/gen_journal_synth.py --pool data/staging/sft_pool_v11.jsonl \\
        --url http://127.0.0.1:8087 --out data/staging/synth_journal.jsonl \\
        --report runs/journal-synth-report.json

**The hole this fills.** v1.1 rebuilt the reading step around the journal and left synthesis
reading a v1.0 world. Measured on `sft_pool_v11.jsonl`, all 450 `SYNTHESIZE` rows:

* prompts hold a **median of 13 points and never more than 16** — exactly `POINTS_CAP`,
  because they were written against the v1.0 survivor set;
* **0 rows contain `後改為`**, the journal's supersession rendering, so the single capability
  `revise` exists to produce has no synthesis supervision anywhere;
* targets sit at 38.8 chars per point against a near-constant ~470-character output.

Meanwhile, replaying the pool's own gold ops through the real harness (88.3% applied) puts
**51% of meetings above 16 entries, with a median of 17 and a maximum of 57**. So more than
half the distribution the model meets at serving time was never in its training data, and
output length was never conditioned on input size. That is the mechanism behind the measured
under-rendering: ~40 journal entries in, 346 characters out.

**Why the input is rebuilt by replay rather than edited.** The stored prompt is a v1.0
rendering; there is no edit that turns 16 survivors back into the 40 entries that were
actually recorded, because the evicted ones are simply not in the row. Replaying the gold ops
through `apply_ops` and rendering with `build_synth_prompt` reconstructs the state the
harness would really present — including the journal — so the input is correct by
construction instead of by patch. Ops that do not apply are skipped exactly as they are at
serving time; this is deliberately the live path, not an idealised one.

**Why a coverage gate, and not just the grounding gate.** `regen_synth.py` verifies that a
target asserts nothing absent from its memory, which is necessary and not sufficient: a short,
scrupulously faithful summary passes it while teaching the model to ignore most of its input.
Coverage is the property actually in deficit, so it is measured per row and enforced. A target
must both stay inside the memory and reach most of it.

**The teacher is steered; the stored row is not.** The teacher receives an extra coverage
instruction, but every row is written with the unmodified `synth_system_prompt()` the student
sees at inference. Choosing what behaviour to demonstrate is what supervision is for; changing
the student's prompt is a separate experiment, and CLAUDE.md trap 7 records that a coverage
instruction added prompt-side alone made synthesis worse.

**Teacher provenance changed and this is recorded, not silent.** The pool's original synthesis
targets were authored by Qwen3.8-27B, whose local blobs no longer exist (the Qwen3.6-27B and
35B-A3B GGUFs that remain fail to load against llama.cpp `15586e2d7` — MTP tensor-count
mismatch). This pass uses gemma-3-27b-it. Mixing two teachers inside one supervision slice
would be an uncontrolled variable, so **every** synthesis row is regenerated here rather than
only the oversized ones. SPEC's Gemma exclusion is scoped to the JUDGE, where it exists to
keep evaluation independent of the translator; a Gemma-authored target scored against
Qwen-authored G3 references is more independent than the Qwen-on-Qwen pairing it replaces.

**The supersession markup must not reach the summary.** `build_synth_prompt` renders a
revised point as `原文（後改為：新文）`. That notation is an INTERNAL rendering, and the teacher
copies it verbatim into its prose roughly one time in three unless told not to. Those rows are
exactly the ones carrying G1's revision capability, so unfiltered they would have taught the
student to print harness markup at the user — on the single behaviour v1.1 added `revise` to
support. Rejected on the literal `（後改為：`, never on the phrase `後改為`, which is ordinary
Chinese and is the correct thing for a summary to say.

**Rejections are reported by size bucket, one counter per cause.** Dropping a failed row
silently would shrink supervision exactly at the large-journal end — the regime this tool
exists to serve, and the way `runs/clean-e3` destroyed the model's willingness to assert
anything at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcsum.backends.llama_server import LlamaServer  # noqa: E402
from arcsum.chunker import Chunk  # noqa: E402
from arcsum.evalkit import grounding  # noqa: E402
from arcsum.guards import apply_ops  # noqa: E402
from arcsum.memory import Memory  # noqa: E402
from arcsum.prompts import (  # noqa: E402
    TOOLCALL_PROMPT_VERSION,
    build_synth_prompt,
    synth_system_prompt,
)
from arcsum.prose import finalize  # noqa: E402
from arcsum.tokens import heuristic_token_len  # noqa: E402
from arcsum.toolcalls import parse_tool_calls  # noqa: E402
from arcsum.transcript import Utterance  # noqa: E402

# Steering given to the TEACHER only. The stored row keeps `synth_system_prompt()`.
#
# **The arithmetic clause is not boilerplate — it repairs a measured failure.** Asking for
# full coverage makes the teacher fabricate under pressure, which is CLAUDE.md trap 7's
# "fuller memory is not free" showing up at the data level rather than at inference. On the
# 29+ bucket it produced `九十萬` from a memory holding three separate `30萬` items — it SUMMED
# them — and invented `二十五萬六千` outright. Both were caught only because the grounding gate
# runs alongside the coverage gate; a coverage gate alone would have trained the student on
# invented figures, which is the exact defect this whole pass exists to remove.
COVERAGE_ADDENDUM = (
    "\n\n額外要求（僅適用於本次撰寫）：\n"
    "- 記憶中的每一項都必須寫進摘要，不可遺漏任何一項。\n"
    "- 只能使用記憶中已有的內容，包括其中的數字、金額與日期；"
    "絕對不可加入記憶沒有提到的資訊。\n"
    "- 不可加總、計算、換算或推估任何數字。數字一律照記憶原文引用；"
    "若記憶中有多筆金額，必須分別寫出，不可合併成一個總數。\n"
    "- 寧可省略也不可杜撰：若某項內容不確定，就照記憶原文簡述，不要補充細節。\n"
    "- 記憶項目若標示「（後改為：…）」，表示該事項後來被修正，"
    "摘要需說明原本的決定以及後來改成什麼；"
    "但必須用自然語句敘述（例如「原本…，後來改為…」），"
    "不可將「（後改為：…）」這個括號標記照抄進摘要。\n"
    "- 記憶項目較多時，摘要就應該相應加長，以容納所有項目。\n"
    "- 每一項記憶都要用一個完整的子句交代清楚：說明是誰、做了什麼決定或提出什麼要求，"
    "以及其中的關鍵細節（金額、日期、條號、單位）。不可只用四五個字帶過。\n"
    "- 摘要總長度應與項目數量相稱：大約每項 35 到 45 個字。"
    "例如 20 項約 700 到 900 字。上限為 1000 字，不要超過。"
)

#: The harness's supersession NOTATION, which must never appear in a summary. The teacher
#: copies it verbatim ~1 in 3 times if not told otherwise, and those are precisely the rows
#: carrying G1's revision capability — so a third of the new revision supervision would have
#: taught the student to print internal markup at the user. Checked as a literal rather than
#: on the phrase `後改為`, which is ordinary Chinese and perfectly good prose to emit.
MARKUP_LEAK = "（後改為："


def trigrams(text: str) -> set[str]:
    s = "".join(text.split())
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else ({s} if s else set())


def containment(needle: str, haystack: str) -> float:
    """Fraction of `needle`'s character trigrams present in `haystack`.

    Character trigrams rather than token overlap because a summary legitimately rewrites a
    memory point into running prose — it will not repeat it verbatim, so an exact-match test
    would score a perfectly good rendering as a miss.
    """
    a = trigrams(needle)
    if not a:
        return 1.0
    return len(a & trigrams(haystack)) / len(a)


def coverage(entries: list[str], summary: str, floor: float) -> tuple[int, int]:
    hit = sum(1 for e in entries if containment(e, summary) >= floor)
    return hit, len(entries)


def memory_entries(prompt: str) -> list[str]:
    return [ln[2:].strip() for ln in prompt.splitlines() if ln.startswith("- ")]


def arc_line(prompt: str) -> str:
    for ln in prompt.splitlines():
        if ln.startswith("ARC:"):
            return ln
    return "ARC: -"


def group_prompts(prompt: str, entries: list[str], group: int) -> list[str]:
    """Split a large journal into per-group synthesis prompts, each carrying the same ARC.

    **The teacher has a hard length prior and instructions do not move it.** Measured on the
    supervision teacher: a 28-entry journal produced 664 characters and a 34-entry journal 680,
    i.e. 23.7 and 20.0 characters per entry — barely a mention each — and raising `max_tokens`
    from 1400 to 3000 returned **byte-identical output**. It was not truncated; it stops. The
    explicit "35-45 characters per item, ~700-900 characters for 20 items" clause in
    `COVERAGE_ADDENDUM` did not change it either, which is the same lesson as CLAUDE.md's
    v4->v5 case: a model's entrenched behaviour is changed by what it is shown, not by asking.

    Splitting the journal gives each group the teacher's full ~660-character budget, so density
    scales with the number of groups instead of being divided among all entries. Measured on
    the same two meetings: **23.7 -> 38.1** and **20.0 -> 29.1** characters per entry, with
    coverage at the STRICT 0.45 containment threshold reaching 27/28 and 34/34, and grounding
    still 0 ungrounded.

    Each group repeats the ARC because a group summarised without the meeting's through-line
    reads as a disconnected list; the ARC is what lets the parts concatenate into prose.
    """
    return [
        "MEMORY:\n"
        + arc_line(prompt)
        + "\nPOINTS:\n"
        + "\n".join("- " + e for e in entries[i : i + group])
        + "\n\n"
        for i in range(0, len(entries), group)
    ]


def replay_meeting(steps: list[dict]) -> Memory:
    """Rebuild the end-of-meeting memory the harness would really hold."""
    mem = Memory(token_len=heuristic_token_len)
    for r in sorted(steps, key=lambda x: int(x["step"])):
        ops = parse_tool_calls(r["completion"])
        if not ops:
            continue
        chunk = Chunk(index=int(r["step"]), utterances=[Utterance("S1", "x")], tokens=10)
        apply_ops(mem, ops, chunk, lang_check=False)
    return mem


def bucket(n: int) -> str:
    if n <= 8:
        return "00-08"
    if n <= 16:
        return "09-16"
    if n <= 28:
        return "17-28"
    return "29+"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8087", help="the TEACHER server")
    p.add_argument("--max-ungrounded", type=int, default=0)
    p.add_argument(
        "--min-coverage",
        type=float,
        default=0.70,
        help="fraction of memory entries the target must reach",
    )
    p.add_argument(
        "--containment-floor",
        type=float,
        default=0.30,
        help="trigram containment at which an entry counts as covered. **0.30 accepts a bare "
             "MENTION** — enough to satisfy coverage while saying almost nothing about the "
             "entry, which is how six consecutive pools produced targets averaging ~26 "
             "characters per entry and students that rendered only 80-94%% of what they "
             "recorded (SPEC 5.2.5). Use a higher floor to gate DENSITY.",
    )
    p.add_argument(
        "--min-chars-per-entry",
        type=float,
        default=0.0,
        help="reject a target whose length per memory entry falls below this. The direct "
             "density gate: containment can be satisfied by echoing an entry's distinctive "
             "characters, length cannot.",
    )
    p.add_argument("--max-tokens", type=int, default=1400)
    p.add_argument("--group-size", type=int, default=0,
                   help="synthesise journals larger than this in groups of this size, "
                        "concatenating the parts. 0 disables. See `group_prompts`: the "
                        "teacher's length prior is fixed per call, so density scales with "
                        "the number of calls, not with asking.")
    p.add_argument("--max-summary-tokens", type=int, default=1000,
                   help="SPEC §3's output budget; a group-wise target is trimmed to fit it")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", default="",
                   help="`I/N` — process only meetings whose index mod N == I. Lets several "
                        "instances split the corpus across GPUs; the teacher is a single-stream "
                        "server, so one process leaves every other GPU idle. Sharding is by "
                        "index over the SORTED meeting list, so shards are disjoint, stable "
                        "across runs, and their union is exactly the unsharded set.")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    rows = [
        json.loads(ln) for ln in args.pool.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    # A reading step is a CHUNK prompt whose completion is a tool call; the baseline's MAP
    # rows also carry CHUNK but answer in prose, and replaying them would corrupt the state.
    reading = [r for r in rows if "CHUNK:" in r["prompt"] and "tool_call" in r["completion"]]
    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for r in reading:
        by_meeting[r["meeting"]].append(r)

    meetings = sorted(by_meeting)
    if args.limit:
        meetings = meetings[: args.limit]
    if args.shard:
        i, _, n = args.shard.partition("/")
        idx, total = int(i), int(n)
        if not 0 <= idx < total:
            print(f"[journal-synth] REFUSED: bad shard {args.shard!r}", file=sys.stderr)
            return 1
        meetings = [m for k, m in enumerate(meetings) if k % total == idx]
        print(f"[journal-synth] shard {idx}/{total}: {len(meetings)} meetings",
              file=sys.stderr)
    if not meetings:
        print("[journal-synth] REFUSED: no reading steps found", file=sys.stderr)
        return 1

    system = synth_system_prompt()
    steered = system + COVERAGE_ADDENDUM

    out_rows: list[dict] = []
    # One counter per REJECTION CAUSE. They were lumped in the first version of this tool and
    # that immediately cost a diagnosis: a bucket reading "2 ungrounded" was really covering
    # empty output, off-language output and fabrication at once, which are three different
    # problems with three different fixes.
    per_bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tried": 0, "kept": 0, "empty": 0, "off_language": 0,
                 "markup_leak": 0, "fabricated": 0, "thin": 0,
                 "too_sparse": 0, "error": 0, "retried": 0}
    )
    cov_rates: list[float] = []
    chars: list[int] = []
    entry_counts: list[int] = []
    ungrounded_tot = specifics_tot = 0

    def attempt(prompt: str, seed: int, entries: list[str]):
        client = LlamaServer(
            base_url=args.url,
            max_tokens=args.max_tokens,
            repeat_penalty=1.1,
            seed=seed,
            raw_completion=True,
            extra={"cache_prompt": False,
                   "chat_template_kwargs": {"enable_thinking": False}},
        )
        if args.group_size and len(entries) > args.group_size:
            parts = [
                finalize(client(steered, sub), token_len=heuristic_token_len).text
                for sub in group_prompts(prompt, entries, args.group_size)
            ]
            joined = "".join(parts)
            # §3 caps the product summary; a group-wise target must respect the same budget or
            # it teaches a length the student is not allowed to produce.
            while heuristic_token_len(joined) > args.max_summary_tokens and len(parts) > 1:
                parts.pop()
                joined = "".join(parts)
            return finalize(joined, token_len=heuristic_token_len)
        return finalize(client(steered, prompt), token_len=heuristic_token_len)

    for i, mid in enumerate(meetings, 1):
        mem = replay_meeting(by_meeting[mid])
        prompt = build_synth_prompt(mem)
        entries = memory_entries(prompt)
        b = bucket(len(entries))
        per_bucket[b]["tried"] += 1
        if not entries:
            continue

        # A retry is worth having only because it is RESEEDED: at temperature 0 a second
        # identical request reproduces the first answer by construction, which is the same
        # mistake the G2 judge retry once made.
        new = g = None
        rate = 0.0
        for attempt_no, seed in enumerate((0, 1)):
            if attempt_no:
                per_bucket[b]["retried"] += 1
            try:
                new = attempt(prompt, seed, entries)
            except Exception as exc:  # one bad meeting must not lose the pass
                per_bucket[b]["error"] += 1
                print(f"[journal-synth] {i}/{len(meetings)} {mid}: ERROR {exc}",
                      file=sys.stderr)
                new = None
                continue
            g = grounding.check("", new.text, prompt)
            hit, tot = coverage(entries, new.text, args.containment_floor)
            rate = hit / tot if tot else 0.0
            # **The density floor must respect §3's output budget.** Demanding 28 chars per
            # entry AND capping the target at `--max-summary-tokens` is unsatisfiable above
            # ~35 entries, and the first dense run duly rejected 18 of the 29+ bucket and
            # dropped `max_entries` from 49 to 37 — thinning supervision exactly where the
            # journal matters, which is `runs/clean-e3`'s failure. Above that size the
            # achievable density IS the budget divided by the entry count, so require that
            # instead of an impossible constant.
            floor = min(args.min_chars_per_entry,
                        0.95 * args.max_summary_tokens / max(len(entries), 1))
            dense = not args.min_chars_per_entry or (
                len(new.text) / max(len(entries), 1) >= floor)
            if (new.text and not new.lang_flags
                    and MARKUP_LEAK not in new.text
                    and g.n_ungrounded <= args.max_ungrounded
                    and rate >= args.min_coverage
                    and dense):
                break
        else:
            if new is None:
                continue
            if not new.text:
                per_bucket[b]["empty"] += 1
            elif new.lang_flags:
                per_bucket[b]["off_language"] += 1
            elif MARKUP_LEAK in new.text:
                per_bucket[b]["markup_leak"] += 1
            elif g.n_ungrounded > args.max_ungrounded:
                per_bucket[b]["fabricated"] += 1
            elif args.min_chars_per_entry and len(new.text) / max(len(entries), 1) < min(
                args.min_chars_per_entry,
                0.95 * args.max_summary_tokens / max(len(entries), 1),
            ):
                per_bucket[b]["too_sparse"] += 1
            else:
                per_bucket[b]["thin"] += 1
            continue

        per_bucket[b]["kept"] += 1
        cov_rates.append(rate)
        chars.append(new.chars)
        entry_counts.append(len(entries))
        ungrounded_tot += g.n_ungrounded
        specifics_tot += g.n_checked
        out_rows.append(
            {
                "meeting": mid,
                "step": -1,
                "system": system,
                "prompt": prompt,
                "completion": new.text,
                "is_nop": False,
                "prompt_version": TOOLCALL_PROMPT_VERSION,
                "journal_entries": len(entries),
                "coverage": round(rate, 3),
            }
        )

        if i % 20 == 0:
            print(
                f"[journal-synth] {i}/{len(meetings)} kept={len(out_rows)} "
                f"mean-cov={statistics.mean(cov_rates):.2f}",
                file=sys.stderr,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "pool": str(args.pool),
        "out": str(args.out),
        "teacher_url": args.url,
        "meetings": len(meetings),
        "kept": len(out_rows),
        "min_coverage": args.min_coverage,
        "containment_floor": args.containment_floor,
        "mean_coverage": round(statistics.mean(cov_rates), 4) if cov_rates else None,
        "mean_chars": round(statistics.mean(chars), 1) if chars else None,
        "median_entries": statistics.median(entry_counts) if entry_counts else None,
        "max_entries": max(entry_counts) if entry_counts else None,
        "chars_per_entry": (
            round(statistics.mean([c / n for c, n in zip(chars, entry_counts, strict=True)]), 2)
            if chars
            else None
        ),
        "ungrounded": ungrounded_tot,
        "specifics": specifics_tot,
        "ungrounded_rate": (round(ungrounded_tot / specifics_tot, 4) if specifics_tot else None),
        "by_bucket": {k: dict(v) for k, v in sorted(per_bucket.items())},
        "prompt_version": TOOLCALL_PROMPT_VERSION,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
