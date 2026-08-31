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

### SPEC v1.0: tool-call protocol + Qwen3.5-0.8B — `qwen-tools-v5` is the current best

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

**§4.1 v1.0's real innovation is single-turn tool calls, not tool calls per se.** A
conventional observe-the-result agent loop was MEASURED and REJECTED: 2 model invocations
per chunk, 1.89x prefill (the second turn re-sends the whole chunk plus the first turn's
output) -> 32-51 min projected, a property of the control flow no fine-tuning fixes. One
batched `update_memory` call with JSON arguments costs only 1.25x the edit-line format's
decode tokens (98 for one-op-per-call, 45 for one batched JSON call, 36 for edit lines).
Also measured and rejected: the chat template's own `tools=` preamble (313-434 tokens/step
vs 187 for a hand-written schema prompt).

**Two integration blockers if you touch Qwen3.5 again:**
- It is `Qwen3_5ForConditionalGeneration` (vision-language). `AutoModelForCausalLM` loads
  only the text tower, which is why **unsloth cannot train it** — it surfaces as a
  `Qwen3VLProcessor` and TRL then reads `eos_token` as a literal `'<EOS_TOKEN>'`
  placeholder and aborts. Use `tools/train_toolcalls.py` (plain `transformers.Trainer`,
  explicit completion-only masking) instead.
- It carries an MTP head (`mtp_num_hidden_layers: 1`, 15 `mtp.*` tensors). **The
  "copy them back from base before converting" instruction that used to sit here is WRONG
  for the current training path and was removed on 2026-08-31.** Verified against
  `runs/qwen-tools-v6/final`: base has 488 tensors / 15 `mtp.*`, the fine-tune has 335 /
  **15 `mtp.*`** — the head is preserved. The 153-tensor difference is entirely
  `model.visual.*`, the vision tower, which a text-only GGUF does not want. Converting
  `final` directly succeeds and reproduces the shipped v6 GGUF's exact size. The old claim
  probably held for the `AutoModelForCausalLM` path unsloth forced; it did not survive the
  move to `tools/train_toolcalls.py`. Use `tools/export_gguf.sh`, which asserts the 15
  tensors are present and stops if the training path changes. Also: 248k vocab OOMs at
  batch 4 on a 32GB card; use batch 1 with grad accumulation (909 steps ≈ 3h45m for a
  4,837-row pool at `--batch-size 1 --grad-accum 16`).
- `save_strategy="epoch"` + `load_best_model_at_end` on eval loss is now the default in
  `train_toolcalls.py`. Both Qwen runs bottomed at epoch 2 and rose at epoch 3; the first
  build (`qwen-tools-v2`) shipped the overfit epoch-3 model before this was fixed.

### Real-ASR is now a standing gate — `tools/asr_gate.py` — because it silently regressed

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
