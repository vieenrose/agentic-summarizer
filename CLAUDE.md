# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is the **`next` branch**: a from-scratch redesign, started from an empty tree
(`86e6c67`). It currently contains **only `SPEC.md`** — no source, tests, or build
config exist yet. There are therefore no build/lint/test commands to run. Add them to
this section as soon as the first tooling lands (`pyproject.toml`, the harness package,
how to run it on one transcript, how to run a single eval meeting, how to run the G1
revision probe).

A `.venv` (Python 3.12, `uv`-managed) is already provisioned with the training/serving
stack: `torch` 2.11 (cu128), `transformers` 5.5, `trl`, `unsloth`, `peft`, `accelerate`,
`datasets`, `llama_cpp_python`, `gradio`, plus `pytest` and `ruff` for whenever tests and
lint config land. There is no `pyproject.toml` yet, so `uv run` has nothing to key off —
create one before relying on it.

**`SPEC.md` is the normative contract.** Where any code disagrees with it, the spec
wins. Read it in full before implementing anything — this file only orients you; it
does not restate the spec's normative detail (formats, caps, gate criteria).

### The prior project (`master` branch) is a different, superseded system

`master` holds a completed prior implementation of a similarly-named but materially
different design (`src/`, `train/`, `eval/`, `tools/`, its own `CLAUDE.md`/`PLAN.md`/
`RESULTS.md`/`README.md`). It is retained for reference — inspect it with
`git show master:<path>` or `git log master` — but **do not copy its code or its spec
assumptions without checking against `SPEC.md` first.** Key things that changed between
the two:

| | `master` (prior, superseded) | `next` (this branch, current) |
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
- **Execution plan (§9)** is phased and gated cheapest-first: Phase 0 (on-device
  latency/RSS/context measurement, no corpus needed) → Phase 1 (200-meeting pilot
  corpus) → Phase 2 (pilot fine-tune + baseline + revision probe) → Phase 3 (in-domain
  zh-TW eval slice) → Phase 4 (full 1,366-meeting corpus). Do not skip ahead to a later
  phase's spend before its predecessor's gate has passed.
