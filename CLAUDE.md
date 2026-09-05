# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is the **`next` branch**: a from-scratch redesign, started from an empty tree
(`86e6c67`). Build-out is in progress against `SPEC.md`, now **v1.0** — student
Qwen3.5-0.8B, single-turn tool-call protocol (§4.1). v0.9 (MiniCPM5-1B, edit lines) is
superseded; see "SPEC v1.0" below before touching the agent protocol or the student
model. `src/arcsum/` still supports BOTH protocols side by side (`run_agent(...,
protocol="edit"|"tool")`) so they stay comparable on one harness.

**Commands** (the `.venv` is Python 3.12, `uv`-managed; there is no `uv sync` lockfile
yet, so call the venv's binaries directly):

```bash
.venv/bin/python -m pytest -q                 # whole suite
.venv/bin/python -m pytest tests/test_ops.py -q          # one file
.venv/bin/python -m pytest -k spread -q                  # one test by name
.venv/bin/ruff check .                        # lint
.venv/bin/ruff format src tests               # format
```

The suite must run with **no GPU, no weights, no network, and no optional extra
installed** — that property is load-bearing, not incidental, and is what makes the
harness iterable. Keep it: the model is always a plain `(system, user) -> str` callable,
`token_len` is always injected, and network calls are stubbed at `urllib.request.urlopen`.

**Landed**: the whole harness core (`tokens.py`, `transcript.py`, `chunker.py`,
`memory.py`, `render.py`, `ops.py`, `lang.py`, `guards.py`, `prompts.py`, `prose.py`,
`agent.py` including `SYNTHESIZE`, `backends/llama_server.py`, `baseline.py`,
`probe.py`), the corpus/eval layer (`corpus/`, `metrics/`, `judge/`, `supervision/`),
and 9 of 10 `cli/` entry points: `score`, `report`, `import_corpus`, `probe`, `judge`,
`gen_traces`, `build_sft`, `run_arms`, `device_bench` — all wired to their
`[project.scripts]` names and covered by tests that stub `urllib.request.urlopen`
(or, for `device_bench`, parse the real artifacts committed under `runs/`) rather than
needing a live model or device.

**Not yet written**: `cli/trace_report.py` (deliberately deferred:
`supervision/report.py`'s own docstring insists its rates stay welded to live
`Trace`/`Step`/`Outcome` objects rather than a parallel on-disk schema, since a
numerator/denominator split across a serialization boundary was a real, twice-repeated
bug in the prior project — `cli/gen_traces.py` computes and writes the supervision
report from the live traces it just generated instead, and there is currently no
companion tool that reconstructs a `Trace` from disk later). Installing the package
will give 9 working commands; `arcsum-trace-report` will `ImportError` until built.

**`SPEC.md` is the normative contract.** Where any code disagrees with it, the spec
wins. Read it in full before implementing anything — this file only orients you; it
does not restate the spec's normative detail (formats, caps, gate criteria).

### Five measured traps, all found on 2026-08-27 — do not re-derive

(Four more, measured 2026-08-28, follow this list, then trap 10 on serving flags.)

Each cost real debugging time and is now pinned by a test. The docstrings carry the
numbers; this is the index.

1. **The SFT pool's two balance knobs compound against NOP** (`supervision/sft.py`).
   `downsample_nop` solves its cap against the pool as it stands, then every row
   `oversample_drop` appends dilutes NOP further, so the final share lands *below*
   `max_nop_frac`. The `sft-dropv1` build fell to 25.7% NOP against a teacher rate of
   38.2%; the resulting checkpoint stopped emitting `NOP` and churned instead —
   `DROP` plus a near-identical re-`ADD` — burning up to 45 of a 53-step meeting's
   steps on one topic. **Check the resulting share; never assume it lands at the cap.**
2. **Greedy decoding degenerates into repetition on prose** (`backends/llama_server.py`
   `repeat_penalty`). One synthesis emitted the same sentence eight times, giving that
   eval its worst result. `repeat_penalty=1.1` cut it from 2,053 to 432 characters.
   **Prose calls only** — reading steps emit a fixed op vocabulary, so a repetition
   penalty there would punish the literal `ADD`/`DROP`/`ARC` tokens the format needs.
   The baseline's *map* call is prose too, and gets its own client for exactly this
   reason: penalising only the agent's prose would be the unfair baseline §5.2 forbids.
3. **llama.cpp 500s on its own model's output, and no serving flag is known to fix it.**
   MiniCPM5 occasionally emits an invalid UTF-8 byte (`民�們`); llama.cpp's chat parser
   answers by rejecting the ENTIRE response with a 500 rather than returning one bad
   character in ~300. Fail-fast lost 2 of 20 meetings, and since §5.2's comparison is
   paired that cost *both* arms a meeting and **withheld every G3 gate** for
   `n < min_n`. Neither `--jinja` nor `--skip-chat-parsing` avoids it — both were tried,
   and a server started with `--skip-chat-parsing` still logged four
   `unparsed peg-native output` warnings and still 500'd. `server_error_retries` in
   `backends/llama_server.py` is the only mitigation that currently works, and only
   because of trap 4 below; it cannot rescue a `cache_prompt: false` run.

4. **llama.cpp's prompt cache changes generation — pin `cache_prompt: false` for any
   number you intend to report.** Same model, same seed, same prompt, `temperature=0`:
   cache-on returned 700 characters and cache-off 167, and each setting was internally
   deterministic across three repeats (3/3 byte-identical). So results are reproducible
   *given a cache state* but not across runs, because the cache depends on which
   meetings ran before. This is the mechanism behind trap 3's "same meeting fails in one
   run and succeeds alone", and behind degenerate reading steps that appear and vanish
   between runs. **Weigh this against trap 3 deliberately**: `cache_prompt: false` buys
   reproducibility but disables the retry's only escape route.
5. **G1's probe matched subject terms literally** (`probe.py`). Real output stated both
   reversals perfectly but wrote "B 樓" for "B 棟" and "預算案" for "行銷預算", scoring a
   false FAIL. `subject_terms` is now one tuple of acceptable surface forms per concept.
   This cannot weaken the gate — a stale summary still fails on
   `states_earlier_as_current`, pinned by a negative-control test.

### Four more, measured 2026-08-28 — all are REFUTED hypotheses, do not retry them

The value here is negative: each of these looks obviously right and is wrong on
measurement. Each cost a full experiment.

6. **Within-chunk recency bias is real, but shrinking the chunk does NOT fix G1.** At
   2,500 tokens the reading step records a chunk's tail and drops its head (probe: 「B 棟」
   appears 15× in chunk 0's head and never in the output; real meetings: mean trigram
   containment head 0.221 vs tail 0.322). It is NOT inherited from the teacher — gold is
   head-favoured, 335 vs 265. But end-to-end, G1 still FAILS at budget 1200 AND 800, and
   800 re-triggers the original fixation bug (three identical ARCs refused, one probe
   meeting ending with **0 points**). Chunk size buys ROUGE, not G1 (`runs/chunk1500/`).
7. **Telling the model to "cover the whole chunk" fixes reading and breaks synthesis.**
   One extra SYS line does make the step capture the head — and synthesis then degenerates
   into repeated sentences (1,358 chars, trap 2 again despite `repeat_penalty=1.1`) and
   drops the subject entirely. Probe went from 1 PASS to 0. **Fuller memory is not free.**
8. **Reordering to CHUNK-then-MEMORY is much worse.** Tested on the reasonable theory that
   duplicate ADDs happen because memory sits at the under-attended head of the prompt.
   Measured over 4 meetings: applied-op rate 79.0% → 44.2%, duplicate points 12.2% →
   29.5%, emitted ops 271 → 509. `MEMORY` before `CHUNK` is now measured, not argued.
9. **The memory caps are NOT where the output budget goes.** Of 384 attempted ops at the
   production budget, 24.5% are refused — but `point too long` + `arc too long` together
   are **0.6%**. The waste is repetition: `duplicate point` 14.8% + `arc unchanged` 7.0%,
   i.e. ~22% of every step's output re-emits already-recorded content, which on-device is
   latency spent against G4. Do not raise `POINT_TOKENS` hoping to recover content.

**The long-meeting weakness is FIXED as of `sft-dropv6` — this entry is kept because the
route to the fix is the lesson.** The model used to fixate on the longest meetings,
re-emitting a byte-identical `ARC` while the transcript moved on. Long meetings went
4/9 → 9/9 ROUGE-1 wins and the 53-chunk fixation meeting swung -0.164 → +0.304. Two
things were BOTH required, and neither sufficed alone:

1. **Genuinely new long-meeting supervision** (Phase 4, `data/p4_zh`: 50 meetings, median
   55 chunks vs the pilot's 16). Reweighting the existing 200 could not do it — see the
   dropv3 paragraph below.
2. **A position signal in the prompt** (`PROMPT_VERSION` `sys-v2`). Adding the data alone
   (`sft-dropv4`, `sft-dropv5`) fixed long meetings and BROKE short ones — 10/11 → 6/11 →
   5/11, `corr(meeting length, change) = +0.671` — regardless of NOP mix. `sys-v1`'s step
   prompt carried no step index and no chunk count, so late-step behaviour could not be
   *conditioned*, only absorbed into the global policy. Holding the NOP share exactly
   constant did not separate them (`runs/sft-dropv5/RESULT.md`), which is what ruled out
   the whole family of mixing-ratio fixes.

**Do not retry reweighting for this — it was tried and it regressed.** `sft-dropv3`
(`runs/sft-dropv3/RESULT.md`) raised the late-step share 14.2% → 22.0% via
`oversample_late_steps` and moved ROUGE-1 *away* from its gate: 14/20 (p=0.115) → 12/20
(p=0.503), with rouge2 19→17 and rougeL 19→18. The mechanism was not inert — the
53-chunk fixation meeting improved -0.164 → -0.073 and two meetings flipped to wins —
but it broke four meetings `dropv2` won, because duplicating late steps duplicates
NOP-heavy examples and trades content capture everywhere else for restraint at the end.
**The transferable lesson: stable SHARES did not imply stable BEHAVIOUR.** That build
held NOP (0.353 → 0.359) and DROP (0.320 → 0.325) nearly constant and was still a
regression, because what changed was *which* samples carried those labels. Reading the
shares `build_sft` reports is necessary, not sufficient. Closing this gap needs real
Phase-4 long-meeting supervision, not a different mix of the same 200 meetings — and a
hyperparameter search over `--target-late-frac` against the 20-meeting eval set would be
fitting the only held-out data there is.

### Trap 10, measured 2026-08-31 — the jinja flag is per-ROLE, not per-model-family

**`--jinja` for the teacher, `--no-jinja` for the student. Both are Qwen; the flag is not
about the model, it is about which prompt string the server builds.** Getting this backwards
is silent: it produces plausible-looking output and a plausible-looking failure rate.

Cost: a supervision regeneration launched with `--no-jinja` on the teacher came back
**`dirty=13/13` on every meeting**. It was only catchable because the recorded clean-replay
rate is 51% (`tools/mix_phase4.py`) and 100% is impossible-looking. **Had it returned ~70% I
would have read it as "large chunks are harder for the teacher"** — a plausible, entirely
wrong conclusion, baked into a retrained pool. Keep the 51% number prominent for this reason.

Measured on `/apply-template`, same messages, tail of the rendered prompt:

| server | rendered tail |
|---|---|
| student Qwen3.5-0.8B, `--no-jinja` | `…<\|im_start\|>assistant\n` |
| student Qwen3.5-0.8B, `--jinja` | `…assistant\n<think>\n` |
| teacher Qwen3.8-27B, `--jinja` + `enable_thinking:false` | `…assistant\n<think>\n\n</think>\n\n` |
| teacher Qwen3.8-27B, `--jinja` default | reasoning preamble injected into SYS, then `<think>\n` |
| teacher Qwen3.8-27B, `--no-jinja` | plain ChatML — **no `<think>` at all** |

- **Student needs `--no-jinja`**: its GGUF template unconditionally opens `<think>\n`, and
  `tools/train_toolcalls.py` trained on plain ChatML. Serving under jinja prefixes every
  prompt with tokens the fine-tune never saw.
- **Teacher needs `--jinja`**: `chat_template_kwargs: {enable_thinking: false}` is honored
  ONLY on the jinja path; the legacy builder ignores it silently.

**The counterintuitive part, and the actual mechanism:** `--no-jinja` gives the teacher no
`<think>` marker at all — and that is exactly the failure. Suppression works by PRE-FILLING
an already-closed `<think>\n\n</think>` so the model resumes past it. **Absence of the marker
is not absence of thinking; it is absence of the thing that stops it.** A reasoning model with
no marker opens one itself.

So the real dependency is on `chat_template_kwargs`, not on jinja per se — a client that
prefilled the closed think-block itself could run the teacher `--no-jinja`.

**This reaches further than the teacher.** `cli/run_arms.py` defaults `raw_completion=True`,
so BOTH eval arms render through `/apply-template` too. Serving the student under `--jinja`
would put a stray `<think>` in every prompt of every reported measurement — an eval-wide
confound attributable to a serving flag, not to whatever was being tested. Check the flag
before trusting any number. Note trap 3 tried `--jinja` for a DIFFERENT reason (MiniCPM5's
UTF-8 500s) and found it did not help; that is not a licence to leave it on.

### Trap 11, 2026-09-02 — measuring the WRONG THING is this project's most common failure

Not a new mechanism; a class. Four instances in one session, three of which produced a
plausible number that would have been recorded as a finding. **Every one was caught by an
implausible value, never by process** — which means the process does not catch them.

1. **A server that failed to bind, answering as the previous model.** Killing the old
   `llama-server` and starting the new one in one command: the kill had not completed, the
   new server exited on a port conflict, and the old one served the "new" checkpoint's
   curve. Caught only because two checkpoints returned BYTE-IDENTICAL numbers.
   **Fix, now used everywhere: verify `/props`'s `model_path` before every measurement.**

       curl -s http://127.0.0.1:8081/props | python -c 'import sys,json;print(json.load(sys.stdin)["model_path"])'

2. **`tools/score_reversals.py` defaulted to `--protocol edit`** and scored a v1.0
   tool-call checkpoint under the edit grammar: a clean-looking **0/27**, which reads
   exactly like "this model cannot revise". Real answer under `--protocol tool`: **8/27**.
   The tell was `subject_present=False` on all 27 — a model writing real prose about the
   right meeting cannot miss the subject every time. `--protocol` is now REQUIRED.
3. **A headline number with no artifact behind it.** The "memory details 10.0 -> 2.8"
   collapse that motivated an entire retrain came from an inline script, and **does not
   reproduce** — re-measured, the regression is 10.4 -> 7.8 points, real but a third the
   size. Likewise `v5`'s `3/27` and `selfdistil`'s `12/27` had no artifacts; on re-run they
   are 3/27 and **11/27**. This is the same gap `run_probe.py`'s docstring was written to
   close, reopened by writing a new measurement inline instead of committing it.
4. **`pkill -f <pattern>` matches the agent's OWN shell** and kills the session (exit 144),
   three times. `pgrep -f "cli.run_arms"` in an `until` loop likewise never terminates,
   because the loop's own command line contains the pattern. **Use `pgrep -x llama-server`
   and filter on `/proc/<pid>/cmdline`** — an exact process-name match cannot match bash.

**The generalisable rule: a measurement that cannot be re-run from a committed script and a
recorded configuration is not evidence, it is a memory.** When a number decides a retrain,
commit the tool first. Three of this session's four cost an experiment each.

### Trap 14, 2026-09-05 — a reward pays for what it CREDITS, and the cheapest credit becomes the policy

RAFT on the reading step. Three ops were exploited in turn, each because it was the cheapest
way to collect full credit, and each exploit was invisible in the metric it damaged until a
different column was read. Full record: `runs/raft-reading-outcome.md`.

| op | why it paid | what it produced |
|---|---|---|
| near-duplicate `ADD` | clears the exact-match refusal, so it APPLIES | one step scored 6.75 emitting the same point 3x |
| bare `DROP` | every applied op earned +1, so discarding counted as work | **churn 2.9% → 44.7%**, 4x over G7 |
| `ARC` | replaces one slot but was credited like an accumulating `ADD` | ARC emitted nearly every step, ~48% refused `arc unchanged` |

**The generalisable rule: price an op by what it CONTRIBUTES, not by whether the harness
accepted it.** The harness accepts anything well-formed; acceptance is not achievement.

**The diagnosis was NOT in the churn counter — it was in the memory shape.** Recorded points
rose 366 → 604 while SURVIVING points stayed flat at ~345, so retirements went **18 → 259**.
The model recorded more and threw nearly all of it away. When a behavioural metric moves,
check the conservation quantity beside it.

Two second-order lessons, both of which cost something:

- **Fixing `DROP` credit was not enough, because `idle` keyed off the OP KIND.** With `DROP`
  merely uncredited it was still strictly better than `NOP` (−0.06 vs −0.27), since `NOP` was
  penalised and dropping was free. Caught only because the new test failed. `idle` now means
  **recorded nothing**.
- **The defect was in the POOL before it was in the checkpoint, in a column nobody read.**
  Round 1's pool was audited for grounding (6.4% vs gold's 45.1%) and abstention (NOP 22.4%
  vs 48.2%) — both excellent — while carrying 0.65 drops per add against gold's 0.37.
  `tools/audit_candidates.py` now reports how SELECTION moves each statistic relative to the
  candidate pool; the direction of the move is the reward's signature. Run it before training.
- `raft_reading.py --save-candidates` persists every scored candidate. Sampling is the
  expensive half (~3.5 h) and the reward is the cheap half; all three fixes above cost a full
  re-sample because the losers had been discarded.

**The genuine finding underneath: starvation IS fixable, and it replicates.** Starved
meetings 17/40 → 5/40, NOP 46.2% → 7.9%, specifics +21-44%, on both seeds. Three previous
checkpoints never moved this. It is worth a second round.

### Trap 15, 2026-09-05 — G4 gave THREE different verdicts from one benchmark, all from remembered inputs

`runs/g4-device-measured.md`. In one day: **19.0 min PASS → 20.4 min FAIL → 16.24 min PASS**,
same device, same benchmark. Nothing was wrong except which numbers were remembered.

1. **Decode was benchmarked at DEPTH 0.** A reading step decodes after its own ~3,400-token
   prompt, where the device gives 9.87 t/s against 12.57 at depth 0 — **26% slower where the
   system runs**. Prefill at depth 0 was always right, because SPEC §4.1 lets no history cross
   steps. That single term flipped PASS to FAIL.
2. **Decode LENGTH was inherited from `qwen-tools-v5` (~190 tok/step) and reused for every
   later checkpoint.** Measured on `rl-v3`: **80.3**. Less than half. That flipped it back.

**Decode length is a property of the CHECKPOINT, not the device**, and `rl-v3` is cheap
*because it starves* — 80 tokens/step is what NOPing 46.2% of chunks costs. Measured both ways:

| build | NOP | decode/step | meeting | G4 |
|---|---|---|---|---|
| `rl-v3` | 46.2% | 80.3 tok | **16.24 min** | PASS +18.8% |
| `raft-s0-e1` (unstarved) | 7.9% | 154.3 tok | **18.51 min** | PASS +7.4% |

So recording properly nearly doubles decode, costs 2.3 min, and **still fits. G4 is not the
blocker; churn is.** This also corrects an overstatement of mine from the same day — that
starvation and G4 were in direct tension — which was scaled off the wrong 190-token base and
was roughly 3x too large.

**`arcsum.evalkit.latency` is now normative**: it holds the device constants WITH the depth
each was measured at, and projects from a run's own profile. A G4 claim not produced by that
path is not evidence. Also fixed: `arcsum-eval` was serializing a hand-picked SUBSET of
`BehaviourReport`, so `decode_tokens` and `hedge_points` (the polarity-inversion guard that
must be checked before shipping) never reached disk. Coverage is now asserted against the
dataclass.

**SPEC §7's own latency model was wrong and contradicted §4.1**: it averaged over a KV depth
"ramping ~linearly from 0 toward ~4k" and integrated trapezoidally. That is a conversational
agent's cost shape. Here every step rebuilds the same prompt, so cost is CONSTANT, not an
integral. Corrected in SPEC v1.2, along with the measured profile — 3,018-token prompt and
16.50 steps/meeting, against the 3,400/15.2 the gate had been computed from.

### G2 PASSES 4/4 judges — and the agent is LESS faithful PER CLAIM — 2026-09-05

`runs/g2-panel-instrument.md`. Paired over meetings both arms have:

| judge | absolute (the gate) | per-claim | agent fewer/more |
|---|---|---|---|
| `deepseek-v4-flash` | 50 vs 119 PASS | 16.8% vs 16.4% | 28 / 5 |
| `hy3` | 68 vs 125 PASS | 21.7% vs 16.8% | 27 / 6 |
| `longcat-2.0` | 57 vs 81 PASS | 23.0% vs 13.9% | 15 / 8 |
| `muse-spark-1.3` | 62 vs 79 PASS | 19.7% vs 10.6% | — |

**The gate passes; the rate does not flatter.** The agent wins the absolute count because it
asserts ~2.4x fewer claims (298-314 vs 724-745) in summaries a third as long. The obvious
confound is RULED OUT, not waved at: measured with the judge's own `split_claims`, the
agent's claims are 43.0 median characters against 46.0 — marginally SHORTER, not denser.

**It tracks starvation on every judge** — starved meetings invert more per claim (18.1 vs
16.2, 25.9 vs 19.2, 31.2 vs 22.7). Small n, so directional rather than proof, but it is the
mechanism §4.1 v1.1 already names: an impoverished input is the pressure that produces
invention. **So starvation is plausibly a FAITHFULNESS defect, not only a coverage one.**

**The absolute RATE is still not trustworthy** — the panel ran `--votes 1` against a design
that requires a majority (a judge was measured flipping on identical input), inter-judge
per-meeting agreement is 26-44%, and the judge sees only `top_k=6` retrieved utterances, so a
retrieval miss is indistinguishable from an absent fact. `cli/judge.py` now warns on
`--votes 1`. The ORDERING is solid; the rate is not evidence of a rate.

### Trap 12, 2026-09-03 — the grounding instrument penalised FLUENT zh, and the bias was not common-mode

`evalkit/grounding.py` compared numerals literally, so a claim written `六十` against a source
written `60` counted as FABRICATED. The module's own docstring called this an accepted false
positive, which made it sound like noise. It was systematic: **the corpus writes figures in
Arabic and fluent zh-TW output writes them in CJK**, so the check fired hardest on the
best-written summaries. Found only because it rejected 3 of the first 6 teacher outputs while
building journal synthesis supervision — every one of them faithful.

Re-scored deterministically from the stored per-meeting flagged tokens (`runs/grounding-refold.md`):

| checkpoint | reported | actual |
|---|---|---|
| `qwen-tools-v5` | 33.3% | 27.3% |
| `spec-e3` ep2 | 15.6% | **3.1%** |
| `v11-e3` (SPEC v1.1) | 21.2% | **6.1%** |
| `s234-e3` | 28.6% | **0.0%** |

**The ordering never changed, so no comparison-based conclusion was wrong.** What was wrong is
every ABSOLUTE rate, and unevenly: the correction is ~6 points for `v5` and ~15 for `v11-e3`,
because a model emitting more natural zh trips the bug more often. **A bias that scales with
output quality cannot be subtracted out** — `PROJECT-REVIEW.md`'s agent-vs-baseline-vs-teacher
comparison must be re-measured, not adjusted.

A second defect rode along: `兩` and `〇` were missing from `CJK_NUMBER`, so `兩百萬` matched as
`百萬` and was valued at **1,000,000** — half its real value. That silently compares a correct
figure against the wrong number rather than failing to detect it. A test now asserts every
character `cjk_to_int` parses is one `CJK_NUMBER` detects.

**The transferable part: when an instrument documents a known false positive, check whether it
correlates with the thing being measured.** "Accepted limitation" was doing the work of
"measured and bounded", and it never had been.

### Trap 13, 2026-09-03 — v1.1 changed what "recorded" MEANS, and two instruments were not told

The journal split `Memory.points` (the ≤16 working set the model sees) from
`synthesis_view()` (everything ever recorded). `evalkit/behaviour.py` kept reading `points`,
so three metrics silently changed meaning the day v1.1 landed — **all three in the direction
of flattering the model**:

| metric | what broke |
|---|---|
| `starved` | a meeting that recorded 40 points and retired 24 reads as 16/40 chunks = 0.4/chunk, under the 0.5 floor — the best-accumulating meetings get flagged |
| `chars_per_point` | denominator too small, so **under-rendering could not fire**: 346 chars over 40 recorded is 8.7 ch/pt (fails), but over 16 survivors is 21.6 (passes) |
| G5 retention | had no numerator at all; there was no way to ask whether synthesis USED what reached it |

Fixed with `recorded_points` / `rendered_points` / `retention`, wired into `arcsum-eval`.
Pre-v1.1 reports deserialize unchanged because `recorded_units` falls back to the working set,
which is the correct reading for a trace where the two were the same thing.

**The pattern is now twice in one day** (trap 12 was the other): a protocol change lands, the
measurement code still compiles and still produces plausible numbers, and nothing fails. Both
were found by chasing a number that looked *too good* — 52.6 median chars/point for a model
known to emit 346 characters from a 40-point memory. **When a protocol changes what a noun
means, grep for every consumer of that noun.**

**Third instance, same day: the VALIDATION set was still `tools-v1`.** `train_toolcalls.py`
refuses mixed prompt versions in the TRAIN set and never checked the valid set, so every SPEC
v1.1 build computed `eval_loss` — the signal every "best epoch" claim rests on — against the
superseded protocol. A v1.1 model addresses points by id and can emit `revise`; a v1.0
validation row rewards the text-prefix form it was trained away from. The 20 synthesis rows
are worse still: capped at 16 entries with no `後改為`, they actively penalise journal-shaped
behaviour. **`runs/v12-e3`'s epoch losses were scored this way and its best-epoch pick is not
trustworthy — measure both epochs behaviourally.** Now refused loudly; migrate with
`tools/migrate_pool_v11.py` (`data/staging/valid_tools_v2.jsonl`, 411 rows). Note the migrated
valid set's SYNTHESIS rows are still v1.0-SHAPED — migration rewrites reading rows only — so
`eval_loss` still under-rewards journal synthesis and remains a weak checkpoint selector.

### The synthesis supervision hole that v1.1 left open — fixed 2026-09-03

v1.1 rebuilt the reading step around the journal and left `SYNTHESIZE` reading a v1.0 world.
Measured on `sft_pool_v11.jsonl`, all 450 synthesis rows: prompts hold a **median of 13 points
and never more than 16** (exactly `POINTS_CAP`), **0 rows contain `後改為`** (the journal's
supersession rendering, i.e. the one behaviour `revise` exists to produce had no synthesis
supervision anywhere), and targets sit at a near-constant ~470 characters. Replaying the
pool's own gold ops through the real harness (88.3% applied) puts **51% of meetings above 16
entries, up to 57** — so more than half the serving distribution was absent from training and
output length was never conditioned on input size.

`tools/gen_journal_synth.py` rebuilds the slice by replay; `tools/swap_synth_slice.py` merges
it, replacing a synthesis row **only** when a journal-shaped row exists for that meeting —
149 of the 324 synthesis meetings are synthetic capability rows (`hedge-*`, reversals,
deliberation) with no transcript to replay, and the 12 hedge rows among them are what fixed
`v4`'s polarity inversion. Result: 174 rows, **0 ungrounded across 2,015 specifics** (the old
pool was 39.9% ungrounded), 113 rows above 16 entries, 69 carrying supersession markup.

**Two things the generation caught that would otherwise have shipped, and both argue for
gates in PAIRS:**

1. **Asking for coverage makes the teacher fabricate.** It summed three separate `30萬` memory
   items into `九十萬`. A coverage gate alone would have trained that in; the grounding gate
   caught it. An explicit "do not sum, compute or estimate" clause fixed the 29+ bucket from
   1/3 kept to 3/3.
2. **The teacher copies the harness's `（後改為：…）` markup verbatim into prose ~1 time in 3**,
   on exactly the rows carrying G1's revision capability. Rejected on the literal markup, never
   on the phrase `後改為`, which is ordinary Chinese and correct for a summary to say.

### A SINGLE-SEED A/B IS NOT EVIDENCE HERE — measured 2026-09-03

Three pools x two seeds, all six evaluated on `data/heldout_zh`. Full record in
`runs/journal-synthesis-outcome.md`.

| pool | churn s0 | churn s1 | **spread** | retention s0/s1 |
|---|---|---|---|---|
| `v11` pre-journal | 3.5% (13/40 clean) | 13.3% (5/40) | 9.8 pp | 0.837 / 0.836 |
| `v12` journal | 29.8% (4/40) | 17.0% (6/40) | 12.9 pp | 0.921 / 0.937 |
| `v13` journal+dedup | 36.7% (1/40) | 9.2% (6/40) | **27.4 pp** | 0.936 / 0.936 |

**Seed alone moves churn by up to 27 percentage points and the clean count by 12 of 40** —
larger than the effect any of this project's A/Bs has ever attributed to data. `v11-e3`'s
much-quoted 3.5% is the lucky seed; its own replicate is 13.3%.

**The contrast inside this table is the lesson.** `retention` moves +0.10 between pools with a
within-pool spread of 0.000-0.016 and reproduces at both seeds — that is a real effect. `churn`
differences of the same nominal size are pure noise at n=2. **Which metric a claim rests on
decides whether n=2 is enough**, and the only way to know is to run the replicate.

Two claims from that session are RETRACTED: that `v12` was a decisive churn regression
(p = 2.2e-07 came from comparing ONE run of each), and that near-duplicate redundancy caused
the churn (the mechanism is real in the DATA — 5.6% -> 11.2% — but removing it, 7.2% -> 2.1%,
did not reduce churn: `v13` 23.0% vs `v12` 23.4%). **A measured mechanism in the data is not a
demonstrated cause in the model.**

**A paired sign test over meetings does NOT protect against this.** It measures whether a
difference is consistent across meetings for ONE pair of checkpoints; it is silent on whether
a retrained pair reproduces it. `runs/v12-e3` produced p = 2.2e-07 for a comparison whose
run-to-run term is larger than the effect. Treating one training run as the population is the
error, and it is invisible in the statistics.

**Training costs ~35 minutes** on this setup (921 steps, batch 1 x accum 16, 0.8B full
fine-tune) — not the ~4 h implied elsewhere in this file. Replicates are affordable. Run two
seeds per arm before believing any behavioural delta.

### zh-TW costs real tokens — measured 2026-09-03, `src/arcsum/simplified.py`

Over 16 meetings / 354,995 characters, the same content in Simplified tokenises cheaper:

| tokenizer | zh-TW | zh-CN | saving |
|---|---|---|---|
| Qwen3.5-0.8B (student, 248k vocab) | 1.577 ch/tok | 1.761 ch/tok | **10.5%** |
| Granite (100k vocab) | 0.727 ch/tok | 0.909 ch/tok | **20.0%** |

Chunking is token-based, so this is ~10.5% fewer reading steps: **19.0 -> ~17 min against
G4's 20.00 ceiling**, whose measured margin is 3%. Round-tripping TW->CN->TW alters **0.288%**
of characters (aligned with `difflib`; a positional diff wrongly reports 23.5% because one
inserted character shifts everything after it), almost all benign variants — `畫/劃`, `裡/里`,
`台/臺`, `週/周`.

**Use `tw2sp` going in and `s2tw` — NOT `s2twp` — coming back.** The `p` phrase table performs
vocabulary localisation, not script conversion: it rewrote `發布` to `釋出` and `藉` to `借`.
That is fine for making text tokenise well on the way in and unacceptable in a summary that is
supposed to report what was said.

Not yet trained: this is a train/serve change (the pool is zh-TW throughout, and the stored
`system` field must be converted with it or the fine-tune sees an unseen prompt — trap 10's
failure mode). `tools/convert_pool_zhcn.py` refuses to run without opencc rather than writing
an unconverted pool under a converted name, and verifies numerals survive.

### One-pass long-doc summarisation is CLOSED for the Granite hybrid family — 2026-09-03

`runs/onepass-htiny.md`. `granite-4.0-h-tiny` bf16 (7B total / 1B active), swept 5k->80k
heuristic tokens on a real 91.8k-token meeting: **specifics wander between 1 and 8 with no
trend across a 16x input range**, output pinned at 550-590 chars, and it returns **2 specifics
from an 80k-token meeting** — against 8.5 for the 0.8B map-reduce baseline and 20.1 for the
27B teacher. It also fabricates at mid lengths, unlike `3b-a800m`. At 13.9 GB bf16 it is 5.6x
over SPEC §6's 2.5 GB ceiling; Q4 is still 1.8x over.

Same "faithful but thin" shape now measured at 350m, 1b, 3b-a800m and 7b-a1b, so it is not a
capacity problem that scale fixes within this family. **`PROJECT-REVIEW.md` §4's open question
is answered: there is no viable single-pass alternative, so the agent's real control arm
remains map-reduce.**

Also: **quote Granite context budgets in GRANITE tokens.** 80,000 heuristic tokens rendered as
125,054 granite tokens (1.56x), so a "128k context" Granite holds only ~82k heuristic tokens of
zh-TW and cannot read this corpus's longest meetings in one pass at all.

### SUPERSEDED 2026-09-02: `mixed-e3` BEST-EPOCH is the current best, not `qwen-tools-v5`

Full record in `runs/mixed-e3/RESULT.md`. **Serve `runs/mixed-e3/gguf_best/`** (checkpoint
626, epoch 2) — NOT `gguf/`, which is the last epoch and measurably worse.

| | `v5` (last-epoch, its best config) | **`mixed-e3` best-epoch** |
|---|---|---|
| G1 revision probe (27) | 3/27 | **8/27** |
| G2 faithfulness | PASS 16 vs 58 | **PASS 24 vs 53**, 40/40, 0 judge failures |
| G3 rouge1/2/L | PASS, +0.069/+0.041/+0.057 | **PASS**, +0.053/+0.031/+0.039 |
| real-ASR curated | 17/20, NOP 28% | **19/20, NOP 15%** |
| G4 | 19.0 min measured (v5 only) | **not measured — do this before shipping** |

Still FAILS G1, so §5.2's all-or-nothing decision is unchanged at "ship the baseline".
`mixed-e3` wins on the two deployment-facing axes and pays ~30% of the G3 margin and some
G2 margin (24 vs 16 inversions — it writes longer summaries, so the judge sees more claims).

The pool is `data/staging/sft_pool_mixed.jsonl`: `v5`'s reading rows + 187 TEACHER-memory
and 263 STUDENT-memory synthesis rows, the student side filtered at `MIN_DET=2` (53 dropped)
and >=13-point rows 4x oversampled. **The `MIN_DET` filter is load-bearing and was nearly
dismissed as housekeeping** — see the ablation in `runs/mixed-e3/RESULT.md`: teacher rows
ALONE recover only 1 of 3 G3 gates and cost revision (11/27 -> 4/27) and ASR doing it.

### History: `qwen-tools-v5`, the previous recommendation

`runs/qwen-v2-heldout/RESULT.md` holds the FULL history in one file: `v2` -> `v3` -> `v4`
-> `v5`, in build order, each with what it changed and what it cost. Read top to bottom.
**This supersedes the v0.9 gate table below** (edit lines + MiniCPM5-1B), kept for history.
`SPEC.md` is now v1.0; read its changelog before touching §4/§4.1.

`sft-dropv7` (v0.9, edit-line protocol) is NOT the current recommendation. It was 7/7 on
§5.2 but G1 was trained for a 2-case probe with no independent confirmation, and the
real-ASR check below was not run against it until after `qwen-tools-v3` existed.

**Use `qwen-tools-v5`.** It is strictly better than every earlier v1.0 checkpoint on
everything measured — the earlier `v3`-vs-`v4` trade-off ("passes G3 but unusable on real
ASR" vs "works on ASR but loses G3") turned out to be AVOIDABLE, not fundamental.

| gate | `v3` | `v4` | **`v5`** |
|---|---|---|---|
| G1 revision | FAIL — 0/2; independent probe 2/11 | FAIL — 0/2 | FAIL — corpus limit, see below |
| G2 faithfulness | PASS — 18 vs 68, 39/40 | PASS — 18 vs 63, 39/40 | **PASS — 16 vs 58, 40/40 paired** |
| G3 rouge1 | PASS — 27/13, p=0.038 | FAIL — 25/15, p=0.154 | **PASS — 28/12, +0.069, p=0.017** |
| G3 rouge2 | PASS — 33/7, p=0.000 | FAIL — 26/14, p=0.081 | **PASS — 29/11, +0.041, p=0.006** |
| G3 rougeL | PASS — 34/6, p=0.000 | PASS — 28/12, p=0.017 | **PASS — 35/5, +0.057, p=0.000** |
| real-ASR curated (20) | 9/20 | 16/20 | **17/20** |
| synthesis negation bug | n/a | present, 3/3 seeds | **fixed, 3/3 seeds** |
| G4 budget | *(all superseded)* | *(projections, wrong by 17%)* | **MEASURED on the Reno 7: 19.0 min nominal, 19.4 steady-state, 21.6 worst-case, vs a 20.00 ceiling — `runs/g4-device-measured.md`** |

No checkpoint clears all seven gates — G1 fails on all of them — so §5.2's decision remains
"ship the baseline". Below, unlabelled findings apply to the shared architecture; versions
are named explicitly wherever a finding is checkpoint-specific.

**The `v4` -> `v5` fix is the most transferable lesson here: check what the pool actually
teaches before concluding a model "can't" do something.** `v4` deterministically inverted a
faithful question-form point (`委員質疑...是否應加重刑責` -> `認為...不應加重刑責`, 3/3
seeds). Two prompt-side fixes were tried — one on synthesis, one on the reading step — and
BOTH produced byte-identical output, which looked like "a fine-tuned model at temperature=0
ignores novel system-prompt text". True, but not the cause. Inspecting the pool: of 175
synthesis rows, 5 carried `是否` in memory and only 2 preserved it — **the other 3 dropped
the point entirely**. The majority signal was "question-form points are not worth
carrying", so the model improvised, and inverting was the improvisation. **12 synthesis
rows** (`tools/gen_hedge_synth.py`) flipping that signal to 82% preserving fixed the
inversion, recovered BOTH failing G3 gates, and improved ASR curation 16/20 -> 17/20 at the
same time. The reading step improved too, despite only synthesis rows being added.

**G1: five refuted fix attempts, and a measured loss map — read `runs/g1-study.md` before
trying a sixth.** The most recent (`qwen-tools-v6`, 2026-08-31) corrected a REAL defect —
`gen_reversals.py`'s `late_point` discarded `key_term`, teaching lossy revision across all
68 reversal samples — and the correction verifiably landed at the memory level
(`…改為撤回` -> `…改為撤回，B 棟為最佳方案`). It still did not move the gate: independent
probe 0/11 -> 1/11 (noise) while real-ASR curation FELL 17/20 -> 15/20. **`v5` remains the
recommendation; do not adopt `v6`.**

The loss map is the useful artifact. Tracing `key_term` through all 11 probe scenarios:
reaches MEMORY **5/11**, survives to PROSE **2/11**. The revision template was the THIRD
loss point. The reading step fails to capture the term at all in 6/11 (not a token-cap
issue — every point fits `POINT_TOKENS=25`, checked), and synthesis drops it for 3 of the 5
that do arrive. **No single-point fix can carry this gate**, which is why the partial fix
cost more than it bought. Also refuted: prompt-side instructions at synthesis AND at the
reading step (both byte-identical output — a fine-tuned model at `temperature=0` does not
respond to novel system-prompt text), and `dropv7`'s scale-up of synthetic reversals
(pattern-matched the 2 gate cases while the independent probe fell).

**G1 is a corpus problem, not a model or protocol problem.** Measured
across TWO model families (MiniCPM5-1B, Qwen3.5-0.8B), TWO protocols (edit lines,
tool-call) and FOUR checkpoints: `sft-dropv6` 3/10, `sft-dropv7` 2/10 (trained for it),
`qwen-tools-v2`/`v3` 2/11 each (also trained for it, on a DIFFERENT model, no better).
MeetingBank has essentially no within-meeting reversals to learn from (3.4% of gold items
match reversal language, all legislative boilerplate about repealing external ordinances,
never a decision reversed in the same meeting). Stop trying to fix this by retraining on
more synthetic reversals from the same recipe — the ceiling is the corpus.

**PARTLY SUPERSEDED, 2026-09-02 — a large part of G1's failure was STALE DATA, not the
corpus.** Read `runs/revfix-e3/RESULT.md`. `tools/loss_map.py` run WITH its control arm
(decisions taken and never reversed) shows the control arm loses NOTHING between emission
and memory (73.3% -> 73.3%) while the reversal arm collapses (59.3% -> 14.8%), with zero
harness refusals. So the reading step can retain an identifying detail perfectly well; it
is revision that drops it. Cause: the pool's 68 reversal rows carried tool-call targets
where the replacement `add` had NO key term — **0 of 34 preserving** — while the on-disk
edit-line gold has been correct since 2026-08-31 and `to_toolcalls.py` converts it
correctly. The rows were never regenerated after that fix and were carried from `v5`'s pool
through `selfdistil-e3` into `mixed-e3`. Every v1.0 checkpoint trained on a demonstration
of lossy revision, on the exact capability G1 measures.

Regenerating them (`data/staging/sft_pool_revfix.jsonl`, 26/26 preserving) moves key_term
retention **14.8% -> 48.1% at memory and 3.7% -> 33.3% at prose**, closing the
revision-specific gap from +58.5 to +11.9 points. **It does NOT move the gate** — 8/27 ->
6/27 — because the bottleneck relocated: subject/key_term present rose 13 -> 20 while
"states the late outcome" fell 8 -> 6. The replacement point has a fixed budget and now
spends more of it on the detail. Also refuted there: raising `POINT_TOKENS` 25 -> 32 to
relieve that budget gives 3/27, worse.

**So the corpus claim above is still right about the CEILING and was wrong about the
BINDING CONSTRAINT.** Carry the revfix rows into any future pool, and aim the next G1
attempt at the OUTCOME WORD, checking both columns — they trade against each other.

**§4.1 v1.0's real innovation is single-turn tool calls, not tool calls per se.** A
conventional observe-the-result agent loop was MEASURED and REJECTED: 2 model invocations
per chunk, 1.89x prefill (the second turn re-sends the whole chunk plus the first turn's
output) -> 32-51 min projected, a property of the control flow no fine-tuning fixes. One
batched `update_memory` call with JSON arguments costs only 1.25x the edit-line format's
decode tokens (98 for one-op-per-call, 45 for one batched JSON call, 36 for edit lines).
Also measured and rejected: the chat template's own `tools=` preamble (313-434 tokens/step
vs 187 for a hand-written schema prompt).

**Two integration blockers if you touch Qwen3.5 again:**
- It is `Qwen3_5ForConditionalGeneration` (vision-language), so `AutoModelForCausalLM`
  loads only the text tower.

  **"unsloth cannot train it" — RETRACTED 2026-09-01, this was probably a wrong-loader
  diagnosis recorded as a model limitation.** unsloth's own Qwen3.5 fine-tuning docs
  (`https://unsloth.ai/docs/models/qwen3.5/fine-tune`) list 0.8B / 2B / 4B / 9B / 27B /
  35B-A3B / 122B-A10B as supported, load DENSE models with
  `FastLanguageModel.from_pretrained(..., load_in_16bit=True)`, support **full**
  fine-tuning (not just LoRA), and require `transformers v5` — which this venv has (5.5.0).
  The recorded failure (surfaces as `Qwen3VLProcessor`, TRL reads `eos_token` as a literal
  `'<EOS_TOKEN>'` and aborts) is what the VISION loader path produces; `FastModel` /
  `FastVisionModel` are documented for MoE and vision tuning respectively, NOT for dense
  text fine-tuning.

  This matters for VRAM, which has been the binding constraint: unsloth quotes 2B at 5 GB
  for bf16 LoRA and ~4x that for FFT (~20 GB), against the ~24.7 GB per rank that
  `train_toolcalls.py` + FSDP uses. **Re-test `FastLanguageModel` before assuming
  `tools/train_toolcalls.py` is the only option** — it was written as a workaround for a
  blocker that may not exist.

  `tools/train_toolcalls.py` (plain `transformers.Trainer`, explicit completion-only
  masking) still works and is what every v1.0 checkpoint was built with.
- It carries an MTP head (`mtp_num_hidden_layers: 1`, 15 `mtp.*` tensors) that the
  fine-tune's save path **may or may not** keep, and llama.cpp's GGUF converter asserts
  they exist. Measured 2026-09-01: `runs/qwen-tools-v6/final` has 335 tensors WITH the 15;
  every `runs/qwen-tools-v7/` checkpoint AND its `final` have 320 WITHOUT them — same
  script, same transformers 5.5.0. Do not assume either way. Restoring them from base is
  exact: v6's trained MTP tensors are 15/15 bit-identical to base's, so training never
  touches them. **Use `tools/export_gguf.sh`**, which detects and restores them. (The 153
  `model.visual.*` vision-tower tensors are also dropped, and should stay dropped.)
  Also: 248k vocab OOMs at batch 4 on a 32GB card; use batch 1 with grad accumulation
  (909 steps ≈ 3h50m for a 4,837-row pool at `--batch-size 1 --grad-accum 16`).
- **`load_best_model_at_end` SILENTLY DOES NOT WORK here, and every v1.0 checkpoint is
  therefore the LAST epoch, not the best.** It was configured after `qwen-tools-v2` and
  never verified; this file previously claimed it was fixed. Measured 2026-09-01 by
  comparing a tensor that actually moves during training (an mlp weight —
  `model.norm.weight` is a useless discriminator, it barely changes):

  | | vs best ckpt | vs last ckpt |
  |---|---|---|
  | `v5/final` | diff 3.36e-04 | **IDENTICAL** |
  | `v6/final` | diff 3.51e-04 | **IDENTICAL** |
  | `v7/final` | diff 4.50e-04 | **IDENTICAL** |

  Mechanism: transformers 5.5.0 saves `Qwen3_5ForConditionalGeneration` weights under
  `model.language_model.*` while the best-model reload looks for `model.*`. Nothing
  matches, it logs `There were missing keys in the checkpoint model loaded` listing EVERY
  weight, and leaves the last-epoch weights in memory. `trainer_state.json` still names
  the best checkpoint correctly, so the state file agrees with the intent and disagrees
  with the artifact.

  Eval loss rises at epoch 3 on every run (`v7`: 0.7790 → 0.7712 → **0.8590**), so every
  shipped v1.0 checkpoint is past its minimum. **This also voids the recorded finding that
  overfitting was "ruled out" as the cause of the real-ASR regression** — that retrain was
  described as using best-epoch selection and did not.

  `train_toolcalls.py` now copies the best checkpoint's files directly instead of trusting
  the reload. Verify a new build with the mlp-weight comparison; do not trust the log.

  **Follow-up, 2026-09-02 — do NOT conclude "so always use the best epoch". EVAL LOSS DOES
  NOT ORDER THESE CHECKPOINTS ON ANYTHING §5.2 GATES.** Measured on three builds; all three
  respond differently, and the best-by-loss checkpoint is sometimes the WORST artifact:

  | build | best-by-loss | what actually wins |
  |---|---|---|
  | `qwen-tools-v5` | ckpt-296 (e2) | **e3 (last)** — e2 is worse: ASR 16/20 vs 17/20, probe 1/27 vs 3/27 |
  | `mixed-e3` | ckpt-626 (e2) | **e2 (best)** — ASR 19/20 vs 17/20 AND G3 effects all larger |
  | `selfdistil-e3` | ckpt-306 (e1) | **e2** — e1 has the run's LOWEST loss (0.8318) and NOPs 95% of real meetings, 2/20 curated |

  The `selfdistil-e3` row is the one to remember: its lowest-loss checkpoint is a model that
  abstains on almost everything. **Export both candidate epochs and MEASURE. Never infer the
  artifact from the loss curve** — same lesson as `sft-dropv3`'s "stable SHARES did not imply
  stable BEHAVIOUR", now restated for a stable LOSS.

  Also: `runs/selfdistil-e3/checkpoint-918` is CORRUPTED (truncated safetensors, from a
  disk-full event mid-save), so that run's `final/` is checkpoint-612 — its recorded numbers
  are an epoch-2 measurement, not the last-epoch measurement every other build reports.

### `arcsum-eval` SUPERSEDES `asr_gate.py` — 2026-09-03, after a shipped-and-rolled-back build

`src/arcsum/evalkit/` + `arcsum-eval`. Read `runs/next-iteration-plan.md` for the full
measurement record. **`asr_gate.py`'s `curated` count is now known to reward the failure it
was meant to catch**: it scores a meeting curated when the summary clears a LENGTH floor, and
a 553-character confabulation built from ONE churned memory point clears it easily. The
number that justified shipping `mixed-e3` (19/20) was counting exactly that.

**What happened, because the shape recurs.** `mixed-e3` beat `v5` on every offline gate
(probe 3/27 -> 8/27, real-ASR "curated" 17/20 -> 19/20, all three G3 gates passing), was
published to the demo, and churned on the first real meeting a user ran: 6 chunks -> 1
surviving point, 4 `restates dropped` events, ARC frozen from step 0, then 553 characters of
confident strategy prose. **Every signal was already being recorded and none was
aggregated** — `Outcome.churn_points` fired correctly 4 times and no gate ever reduced it to
a number.

The instruments now exist (`evalkit.behaviour`: churn, starvation, confabulation,
under-rendering, abstention; `evalkit.grounding`: deterministic reference-free fabrication;
`evalkit.provenance`: server identity + corpus content hashing + a comparison that REFUSES).
Measured on 20 real ASR meetings, cache-on:

| build | clean | churn | starved | specifics | ungrounded |
|---|---|---|---|---|---|
| `v5` (shipped) | 8/20 | 30.8% | 7 | 23 | 43.5% |
| `mixed-e3` (rolled back) | **14/20** | 15.4% | **2** | 11 | 18.2% |
| `s234-e3` ep2 | 12/20 | **0.0%** | 5 | 7 | 28.6% |

**Read `specifics` beside `ungrounded` always.** A build asserting 5 specific claims across
20 meetings scores a perfect fabrication rate and is not thereby faithful — it has stopped
saying anything. That is how the S1 grounding FILTER passed while starving the model.

### Three pool defects found by these instruments, with what fixing each did

1. **Churn is LEARNED and removable.** 106 of 4,540 reading rows (2.3%) have an applied
   `ADD` that merely restates a point `DROP`ped in the same step (verified with the
   harness's own `guards.restates_dropped`, NOT a proxy — a crude prefix test reports 43%
   by counting legitimate revision). Deleting them plus 323 no-op ARC ops took churn
   **28.2% -> 0.0%** (`runs/s234-e3`).
2. **323 ARC ops the harness refuses 100% of the time.** 16.9% of consecutive ARC emissions
   re-send the previous step's ARC verbatim; `apply_ops` rejects every one as
   `arc unchanged`. The pool was spending output budget teaching a rejected op.
3. **The `SYNTHESIZE` supervision is structurally unfaithful — the big one.** Of 3,376
   specific claims in the 450 `SYNTHESIZE` targets, **1,347 (39.9%) are absent from the
   memory the target was written from**, across 45% of rows. §2.2 stage 3 composes the
   target from the whole-meeting gold summary, not from the memory, so **the target is not a
   function of its input** — the model is shown 450 times that the right answer to a memory
   contains things absent from it. `tools/regen_synth.py` repairs this by re-asking the
   teacher for a summary of the memory ALONE. **Filtering instead of repairing is REFUTED**
   (`runs/clean-e3`): it removes 37% of synthesis rows, concentrated at high occupancy, and
   the model stops asserting specifics at all.

**When classifying pool rows, classify precisely.** A SYNTHESIZE row is one whose prompt
starts `MEMORY:` (`build_synth_prompt` is `build_memory_view`); reading rows carry a CHUNK;
the baseline's reduce rows start `SUMMARIES:`. Classifying by "no CHUNK" sweeps up the
baseline's reduce rows — done here, and only harmless because there are 4 of them.

### Local evaluation CANNOT reproduce deployed decoding — measured, 2026-09-03

`arcsum-eval --backend llama-cpp` exists to run the DEPLOYED stack, and it still does not
reproduce the demo's failure. Same GGUF, same transcript:

    llama-server, GPU, cache on                 -> 4 points, 0 churn
    llama-cpp-python, CPU, 8 threads, cache on  -> 4 points, 0 churn
    llama-cpp-python, CPU, 2 threads, cache on  -> 4 points, 0 churn   (the demo's setting)
    the HF Space demo itself                    -> 1 point,  4 churn   <- NOT REPRODUCED

Greedy decoding at `temperature=0` is deterministic only for a FIXED floating-point
reduction order, and that order depends on the host CPU and the kernels llama.cpp selects.
**A local scorecard is necessary and not sufficient.** Production logs are the instrument
that actually caught this — which is what the demo's debug-export button is for.

`data/asr_eval_v1/` = the 20 ASR meetings PLUS `dram-supply`, the meeting that failed and
was in NO evaluation corpus. Use it, and re-baseline rather than working around the
comparability refusal.

### Real-ASR gate history — `tools/asr_gate.py` — kept for the regression it caught

**No §5.2 gate protects against this.** G1-G4 are all measured on MeetingBank-derived
text: translated, clean, in-distribution for the training corpus. Real ASR is neither.
Measured today on the SAME 20 real zh-TW meetings (`data/ly_phase3_v2`, Phase 3's corpus):

| checkpoint | curated | NOP rate | mean chars |
|---|---|---|---|
| dropv2-era (2026-08-28) | **17/20** | 41% | 230 |
| `sft-dropv6` (current v0.9, today) | **7/20** | 69% | 92 |
| `qwen-tools-v3` (v1.0, today) | **9/20** | 59% | 122 |

(Corrected once already: "curated" must mean non-trivial synthesis output, not
`points > 0` — `ivod-17704` sets a real ARC and zero POINTS and was miscounted as empty
under the first metric. `tools/asr_gate.py` reports both `curated` (points-based, kept for
continuity) and `mean_summary_chars`; prefer chars-based emptiness in any future read.)

**v0 regressed from 17/20 to ~7/20 across three checkpoints and nothing caught it**, because
every gate since Phase 3 runs on clean text. v1.0 is somewhat better than current v0 (9 vs
7), but BOTH are badly degraded from the dropv2-era number, and that gap — not which
checkpoint is bigger — is the finding to chase. Overfitting was tested as the cause
(retrained `qwen-tools-v3` with best-epoch selection, identical data/hyperparameters) and
ruled out: 9/20, not materially different from the overfit build's 11/20.

**Root-caused, 2026-08-30 — this is real content the model wrongly abstains on, not
corpus garbage, and the mechanism is now identified.** Read all 13-14 NOP'd/empty
transcripts directly. Only ONE (`ivod-17673`, stutter-repeated "他這個 他這個...") is
genuine ASR noise where NOP is correct.

**The model requires the chunk to contain an explicit STATED OUTCOME, and treats
open-ended debate, personal critique, or in-progress Q&A as NOP-worthy even when the
substance is real.** Ruled out first: line length (`17669`/`17681` have similarly long
unbroken lines and DO curate) and speaker count (most NOP'd transcripts have 3-5
speakers, not one). What actually separates them, read directly:

- `ivod-17678` (CURATES): committee dialogue that lands on settled positions — "法案簡化
  刑事訴訟程序...要求...在逐條審查中提升被害人權益保障". Phrased as agreed asks.
- `ivod-17680` (NOP): a single unbroken floor speech advocating for an indigenous-language
  education law — no vote, no other speaker, substantive but never crystallizes into a
  stated position within the chunk.
- `ivod-17699` (NOP): interpellation — personal flood-disaster narrative and political
  criticism of the administration's response. Real content, no proposal.
- `ivod-17666` (NOP): a legislator's clause-by-clause critique of a specific bill article
  (Article 24) — substantively equivalent in KIND to `17678`'s content, but framed as
  ongoing critique/opinion rather than a landed position, and NOP'd anyway.
- `ivod-17701` (NOP): live Q&A between a legislator and an official on evidence-destruction
  safeguards — mid-exchange, no concluded answer within the chunk.

**This tracks directly to corpus provenance.** MeetingBank's gold items are always
RESOLVED agenda-item outcomes ("City Council approved X") — never mid-debate commentary,
critique, or Q&A-in-progress. The model generalized "record resolutions", not "record
substance", which is correct for MeetingBank's council-agenda format and wrong for
legislative committee proceedings, where much of the value IS the deliberation, not just
its eventual vote. **This is a genuine domain-shift gap, not a bug in this codebase** —
closing it needs supervision that includes ongoing deliberation as record-worthy, which
means new teacher instructions/training data, not a prompt tweak.

**Fixed and measured, 2026-08-30 — `qwen-tools-v4`.** 48 synthetic DELIBERATION examples
(`tools/gen_deliberation.py`, same design as the reversal fix: planted gold, independent
probe sharing no subject with training, replay-clean filtering) teaching `ADD -
<role><stance-verb><position>` — attribution, never a resolution — added to `v3`'s pool.
Real ASR: **9/20 -> 16/20 curated, NOP 59% -> 10%**. The negative control held
(`ivod-17673`, genuine noise, still correctly abstains). **Cost: MeetingBank G3 rouge1
flips PASS -> FAIL** (27/13 p=0.038 -> 25/15 p=0.154) — summaries grew 26% longer as the
model now also narrates surrounding deliberation on chunks that DO land on a resolution.
Same shape as dropv4/dropv5's late-step trade: a real capability gain in one regime, paid
for with precision in another. `v4` is the better PRODUCT checkpoint (curates 4/5 of real
meetings vs a checkpoint that abstains on more than half); `v3` is the better GATE-PASSING
one. This is a values call, not something the numbers resolve automatically.

**A new, more serious failure mode surfaced while validating the fix — a synthesis-stage
negation bug, root-caused, not yet fixed.** On the independent deliberation probe (4
scenarios, no subject overlap with training), `forestprotect` produced an INVERSION: the
reading step correctly recorded `委員質疑...是否應加重刑責` ("questions WHETHER it should be
strengthened" — faithful), and `synthesize_memory` deterministically (3/3 seeds) rewrote it
as `認為...不應加重刑責` ("believes it should NOT be strengthened") — asserting the OPPOSITE
polarity as fact. Root cause: no `ADD` target in the training data ever used `是否`
(whether-or-not) phrasing, so the reading step's own paraphrase choice — made live, off any
training example — put synthesis off-distribution, and it guessed a polarity. Wrong.
**Detection is now implemented** (`guards.hedge_marker_in` / `Outcome.hedge_points`,
following the existing "detect and record, never repair in-loop" rule) and **measured
rare**: 1 of 77 ADD points across all 24 ASR+probe meetings (1.3%) — the same case, no
others found. Not fixed: whether to ban the phrasing in training data or instruct synthesis
to preserve question form is unvalidated. **Do not ship a deliberation-trained checkpoint
without this guard active and its `hedge_points` count checked.**

**Run `tools/asr_gate.py` against any future checkpoint before shipping it.** It is not a
phase-gated one-off; it is meant to be run every time, which is the discipline that was
missing when the dropv2->dropv6 regression happened.

### Historical: gate status as of 2026-08-29 — `sft-dropv7` (v0.9, superseded)

### Gate status as of 2026-08-29 — `sft-dropv7` passes ALL SEVEN gates

`runs/dropv7-heldout/RESULT.md`. SPEC §5.2's decision is **"ship the agent"**. Read that
file's caveats before acting on it: G1 was *trained for*, and G4 has never been measured
on the device.

| gate | `sft-dropv7` (40 held-out meetings) |
|---|---|
| G1 revision | PASS — capability deliberately trained; see caveat |
| G2 faithfulness | PASS — **18 vs 109** inversions, 40/40 paired, 0 judge failures |
| G3 rouge1 | PASS — 31/9, +0.095, p=0.001 |
| G3 rouge2 | PASS — 32/8, +0.049, p=0.000 |
| G3 rougeL | PASS — 37/3, +0.067, p=0.000 |
| G4 budget | PASS — 19.58 min **projected**, authorized, never measured on device |

Measured on `data/heldout_zh` — 40 meetings built through the full §2.2 pipeline from
MeetingBank meetings in **no training pool and no prior measurement** (seed 20260828),
because `data/eval20_zh` had been read six times and `sys-v2` was chosen partly on its
evidence. Long-meeting slice on the held-out set: **9/1 wins, +0.202**.

**Three things that must travel with those numbers.**

1. **G1 was trained for, not generalized to.** MeetingBank contains no within-meeting
   reversals — 3.4% of gold items match reversal language and those are legislative
   boilerplate ("repealing Section 5.53.090") reversing EXTERNAL ordinances. The
   capability was taught with synthetic data (`tools/gen_reversals.py`), so G1's two cases
   are no longer independent. The independent 4-case probe went **2/4 → 2/4, no detectable
   gain**; at n=4 it cannot separate capability from pattern match. **Enlarging that probe
   set is the highest-value outstanding measurement.**
2. **G2's per-claim rate favours the BASELINE** (7.3% vs 4.9%). The agent wins on absolute
   inversions partly because it asserts 246 claims against 2,236 — it says less. The gate
   is the absolute count; report both.
3. **G4 has no measurement behind it** — 2.1% margin, throttling unmodelled, acting in the
   failing direction.

**Do not restore any earlier "G1 PASS for dropv2 / 6 of 7" line.** It was never
reproducible: no probe artifact existed and `g1_passed: true` came from the `--g1-passed`
flag. Under a recorded configuration dropv2 fails BOTH probe cases, so **dropv2 is 5 of
7**. Always use `tools/run_probe.py`, which records prose, per-step raw ops and every
generation knob.

**Two measurement bugs found on 2026-08-29 that had hidden most of G2's effect** — both
fixed, both pinned by regression tests:

- `count_inversions(paired_with=...)` required the other arm to have *a record*, not the
  inversions FIELD. ROUGE and judge records share one index and the ROUGE pass covers
  every meeting, so pairing silently did nothing: 35 agent meetings were summed against 19
  baseline meetings ("14 vs 11, FAIL").
- The judge (gpt-oss) can spend its whole budget in the `reasoning_content` channel and
  return empty `content`. It cost 21 of 40 baseline meetings, systematically the LONGEST
  summaries (median 5,087 chars vs 562), so G2 saw only the control arm's shortest
  outputs. The retry now caps `reasoning_effort: low`; re-sending an identical request to
  a `temperature=0` model reproduces the failure by construction. After the fix both arms
  scored 40/40 and the gap moved from 8 vs 11 to **18 vs 109**.

### Two version constants govern what must not drift

| constant | home | changing it means |
|---|---|---|
| `TOKENIZE_VERSION` | `arcsum/tokens.py` | every previously reported metric is incomparable; the golden fixtures assert equality with it, so goldens must be regenerated and re-reviewed |
| `PROMPT_VERSION` | `arcsum/prompts.py` | every prior trace and eval number is incomparable — bump it rather than silently editing a prompt |

`tokens.py` is the single source of truth for "is this character CJK". The prior project
had **three** drifted answers to that question; do not add a fourth. It also keeps three
roles deliberately separate — `char_tokens` (normative, metrics only),
`heuristic_token_len` (non-normative budget estimate, must never produce a reported
number), and `hf_token_len` (the real instrument).

`PROMPT_VERSION` is now `sys-v2`. `prompts.position_line()` is likewise the single
definition of the `POSITION:` prefix — offline tools that re-render a stored pool import
it rather than reproducing the format.

### Phase-4 tooling (`tools/`), and what each exists to prevent

| tool | why it exists |
|---|---|
| `gen_supervision.py` | gold per-step supervision for a translated corpus; reports `dirty_replays` |
| `mix_phase4.py` | merges new supervision into a pool. **51% of teacher gold steps do not replay cleanly** — it replays each one and keeps only ops that applied (a step left empty is DROPPED, never rewritten to NOP). Caps admission by NOP arithmetic; `--hold-nop` keeps the share exactly unchanged |
| `repro_pool_v2.py` | re-renders a stored pool's prompts after a `PROMPT_VERSION` bump. **The pool holds FOUR prompt shapes** (reading step / baseline map / baseline reduce / synthesize) sharing step indices; only reading steps take `POSITION`. Chunk counts are recomputed from the transcript, never inferred from the max step index |
| `run_probe.py` | runs G1 **and records the artifact** with every generation knob. Use this for any G1 claim |
| `measure_grounding.py` | character-trigram containment of each emitted point in its own chunk, per domain. Distinguishes "invented" from "grounded but badly selected" |
| `clean_pool.py` | removes the three measured pool defects (churn rows, no-op ARC ops, ungrounded synthesis targets) and swaps in corrected reversal rows. Reports a before/after count per surgery, and retention BY OCCUPANCY — the grounding filter cuts hardest at high occupancy, which is the regime that already fails |
| `regen_synth.py` | repairs `SYNTHESIZE` targets by re-asking the teacher for a summary of the memory ALONE. **Needs `--jinja` AND `chat_template_kwargs: {enable_thinking: false}`** — without the second the teacher returns its reasoning as the target (measured: 0/20 repaired, 1,483 chars of `我們需要回答使用者：…`) |
| `enrich_points.py` | puts FIGURES back into gold `ADD` targets. 99% of chunks contain a specific but only 42% of gold ADDs carried one, and the student tracks its supervision (33%) — so vagueness is a DATA ceiling, not a capacity limit. **The cap is not the cause**: ADDs with and without a specific have the same length (17.6 vs 17.7 tokens against a cap of 25). Every added figure must appear in ITS OWN CHUNK — the guard rejected 33 teacher fabrications (`850`, `CSULB`, `300/696`) that would otherwise have taught the model to attach plausible numbers, which is strictly worse than the vagueness it replaces |
| `measure_memory.py` | what the READING step captured, read off the memory directly. **No §5.2 gate looks at the reading step in isolation** — G2/G3/ASR all see it only through synthesis, which can mask a thinner memory by writing about it more fluently. `numerals` is the column padding cannot inflate |
| `cliff_curve.py` | synthesis output length vs memory occupancy, against a FIXED point pool. **Always pass `--pool-file`**: on a hand-written clean pool `v5` shows NO cliff, because a clean pool is teacher-shaped and therefore in-distribution. The cliff needs a REAL student-authored memory to appear |

### The prior project is a different, superseded system — and it lives on `pi-agent`

**The mature prior implementation is on `pi-agent`** (tip `3086075`), not `master`.
`master` (`012bd9a`) is an earlier snapshot of the same lineage — it is the merge-base,
with the harness complete but the student never fine-tuned. Anything involving the
fine-tune, GGUF export, the zh campaign, the verifier, on-device integration notes, or
`tools/measure_faithfulness.py` exists **only** on `pi-agent`.

Both are read-only reference: `git show pi-agent:<path>`, `git log pi-agent`. **Do not
copy their code or spec assumptions without checking against `SPEC.md` first.** Key
things that changed between the two designs:

| | `pi-agent`/`master` (prior, superseded) | `next` (this branch, current) |
|---|---|---|
| architecture | 3-model pipeline (student + verifier + judge panel) | single on-device model (MiniCPM5-1B) does both memory curation and synthesis |
| student | `google/functiongemma-270m-it` | MiniCPM5-1B, Q8, 4k context |
| languages | zh-TW **and** en | zh-TW only; en is source material, never a product language |
| transcript format | v1, timestamped (`[m:ss] speaker: text`) | v2, **timestamp-free** (`speaker: text`) |
| output | fixed multi-section notes (TITLE/SUMMARY/DECISIONS/ACTIONS/OPEN/TOPICS), every bullet anchored to a timestamp | single flowing zh-TW prose summary, <1,000 tokens, no anchors |
| memory ops | ADD / UPD / DEL / CMP / NOP | ADD / DROP / ARC / NOP (no rewrite op — measured too heavy at ≤1B in the prior project) |
| corpus | QMSum + VCSum + real recordings | MeetingBank machine-translated to zh-TW (§2.2), plus a small held-out zh-TW audio slice for eval only |
| hardware | 2× RTX 5090 (training); RPi4-class (deploy target) | training hardware TBD; **Oppo Reno 7 5G (CPH2371), CPU-only** is the deploy target (§6) |

The prior project's *measured negative results* still carry weight even though its
design was superseded — e.g. multi-point rewrite ops proved too heavy to learn at ≤1B,
free ReAct-style tool loops were unlearnable at this scale, and a verifier trained on
clean zh collapsed on real noisy zh (0/11 parseable verdicts). `SPEC.md` §8 encodes the
risks carried forward from that experience; treat that lineage as informative, not
binding.

## Architecture this branch is building toward (see `SPEC.md` for full detail)

- **Agent protocol (§4.1):** the transcript is read in ~2,500-token chunks. The harness
  owns a two-slot external memory (`ARC` + `POINTS`, ≤~600 tokens total) and renders it
  fresh into every step's prompt; **no conversation history crosses steps**. The model
  emits only edit lines (`ADD`/`DROP`/`ARC`/`NOP`); the harness applies them
  deterministically, including cap-overflow handling (spread evenly, never
  head-truncate). A final `SYNTHESIZE` call turns the memory into the prose output.
- **Corpus construction (§2.2, §4.2):** MeetingBank → import to format v2 → translate
  en→zh-TW with TranslateGemma-27B → compose a whole-meeting summary with a Qwen teacher
  from the translated full minutes + segment minutes → human validation. Per-step
  training targets are derived by walking chunks and having the teacher convert aligned
  segment minutes (and, for uncovered spans, the full minutes document) into edit lines;
  every gold edit sequence must replay cleanly through the real harness before use.
- **Evaluation (§5):** character-level ROUGE, SacreBLEU (`tokenize=zh`), BERTScore,
  MoverScore, coverage/density, plus a third-family LLM faithfulness judge (never
  Qwen-family — it authored references — or Gemma-family — it did the translation) and
  human review. Aggregate scores are gated against a fair map-reduce baseline (same
  model, same chunk size); ship only if gates G1–G4 (§5.2) clear, otherwise ship the
  baseline and record the negative result.
- **Execution plan (§9)** is phased and gated cheapest-first: Phase 0a (configuration
  check — verifier necessity + en→zh token ratio; no corpus, no device, no training) →
  Phase 0b (on-device latency/RSS/context measurement) → Phase 1 (200-meeting pilot
  corpus) → Phase 2 (pilot fine-tune + baseline + revision probe) → Phase 3 (in-domain
  zh-TW eval slice) → Phase 4 (full 1,250-meeting corpus). Do not skip ahead to a later
  phase's spend before its predecessor's gate has passed.

## Corpus access (audited, SPEC §2.2)

Use the **Zenodo** release, record `7989108` (`MeetingBank.zip`, 606 MB). It has the
word-level diarized transcripts (1,366 files, `segments[].nbest[0].words[]` with
`text`/`offset`/`duration`/`confidence`, plus `segments[].speaker`) and
`Metadata/MeetingBank.json` (1,250 annotated meetings, `itemInfo` with
`startTime`/`endTime`/`Summary`/`type`).

**Do not build the corpus from `huuuyeah/meetingbank` on Hugging Face** — it is a
stripped derivative with no speakers and no timing, and satisfies none of §2.2 stage 1,
even though it is what is currently in the local HF cache.

Two audited facts worth not re-deriving: item spans cover **56.8%** of meeting duration
on average, and **no minutes PDF is distributed** despite the paper describing one —
which is why §4.2 step 3's `ARC` supervision is degraded and the slot is now a Phase-2
ablation (§8 risk 8).

## Device access (SPEC §6, §9 Phase 0b)

The reference device (Oppo Reno 7 5G, CPH2371) is not attached to this workstation's
adb directly, and **`ssh training-machine` from here TIMES OUT** — port 22 is unreachable
despite the host showing online in Tailscale. That blocked every G4 measurement for weeks.

**The working path, confirmed 2026-08-31 — use this:**

```bash
ssh -o ProxyJump=raspberrypi user@100.122.78.108      # then adb as normal
```

Two things make it work, both non-obvious: **the Raspberry Pi CAN reach training-machine on
port 22** even though this workstation cannot, so it serves as a jump host; and the SSH user
is **`user`**, not `luigi` — Tailscale's ACL rejects `luigi` with "tailnet policy does not
permit you to SSH as user". From there `adb shell`/`adb push` work normally, the phone is
attached (`AYBY6HQCMBF6B6KZ`), and `/data/local/tmp/bench/` still holds the Phase-0b
binaries and GGUFs. The on-device llama.cpp (`15586e2d7`) already supports `qwen35`, so no
cross-compile is needed for the v1.0 student — verify with `strings <binary> | grep qwen35`
rather than assuming.

**Checking CPU features: grep `asimddp`, NOT `dotprod`.** That is ARM's name for the flag in
`/proc/cpuinfo`. A check in this session grepped the literal `dotprod`, found nothing on
three ARM hosts, and wrongly concluded none had it. The Reno 7 has `asimddp` (no `i8mm`, no
`sve`); the Raspberry Pi genuinely does not.

**G4 IS NOW MEASURED — see `runs/g4-device-measured.md`.** Headlines: `qwen-tools-v5` Q8_0
at `-C 0xFF` gives pp3400 59.23 t/s / tg190 12.40 t/s -> 72.7 s per reading step ->
**19.0 min per meeting against the 20.00 ceiling**. Over 29.5 minutes of continuous load
**thermal throttling is NOT observable** (prefill -0.2%, decode -2.4%) — the passively-cooled
risk flagged since Phase 0b does not materialise at 0.8B/Q8. The real exposure is
**transient decode stalls from process contention**: 2 of 14 rounds dropped decode 13-37%
with prefill unaffected, and the worst such round puts a meeting at **21.6 min, over
budget**. Steady state is 19.4 min at a 3% margin.

**Do NOT extrapolate cross-model latency ratios across ISA feature boundaries.** A ratio
derived by benching two models on the Raspberry Pi (ARMv8.0, no dotprod) and applied to the
Reno 7 (dotprod) projected 16.2-16.4 min against a measured 19.0 — wrong by 17%, in the
optimistic direction.
Cross-compiling for it: NDK r27c at `~/android-ndk/android-ndk-r27c` (the zip's symlinks
extract as literal text files if unzipped with Python's `zipfile` — rebuild them from
`ZipInfo.external_attr` before relying on the toolchain). Target flags:
`-DANDROID_ABI=arm64-v8a -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod` (the
Reno 7's Cortex-A78 has `dotprod` but no `i8mm`/SVE — do not build for a newer arch
target, it will `SIGILL`).

**Measured headline result, worth not re-deriving**: use `-C 0xFF` (all 8 cores) when
serving on this device, never the prior project's `-C 0xC0` (big-cores-only) —
big-cores-only measured **over** the ~20-minute wall-clock gate (~22.5 min for the
11-step reading phase) while all-cores measured comfortably under it (~12.9 min). Full
numbers in `SPEC.md` §7 and §9 Phase 0b.

A separate ARM proxy host (`nano`, alias in `~/.ssh/config`) was tried first when the
real device wasn't yet known to be reachable; it lacks `dotprod`/`i8mm` (a different,
weaker ARMv8.0 core) and its GCC 8 toolchain has a NEON codegen bug requiring a source
patch to `ggml-cpu-impl.h` to build at all. It is not a reliable Reno-7 stand-in —
prefer the real device via `training-machine` for any future measurement.
