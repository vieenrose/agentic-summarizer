# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is the **`next` branch**: a from-scratch redesign, started from an empty tree
(`86e6c67`). Build-out is in progress against `SPEC.md` v0.9.

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

**Known open weakness**: on the longest meetings the model still fixates, re-emitting a
byte-identical `ARC` while the transcript moves on. `set_arc` now refuses an unchanged
arc so the instrumentation stops concealing it, but the model behaviour is unfixed. Its
likely cause is coverage: only 1.6% of gold steps sit at index 40+, and the correct NOP
rate *rises* with step index (32% → 51%). More long-meeting supervision is Phase 4 work.
Note it is **not** simply "long meetings are bad" — a 40-chunk meeting is among the
agent's biggest wins, while its worst loss was a 13-chunk meeting (that one was the
repetition bug above).

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
adb directly. It has been reached via `ssh <user>@training-machine` (a tailnet host
with the phone already connected over USB adb) — confirmed working 2026-08-21. From
there, `adb shell`/`adb push`/`adb forward` all work normally against the real device.
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
