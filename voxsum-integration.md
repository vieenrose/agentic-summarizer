# VoxSum integration note — MiniCPM5-1B-CURSOR (agentic summarizer)

**Status:** production candidate (on-device deployment measured 0 inversions) ·
**Model:** `Luigi/minicpm5-1b-cursor` → `minicpm5-1b-cursor-p13.Q4_K_M.gguf`
**Verifier:** `Luigi/lfm2.5-350m-verifier` (~215 MB, on-device FAITH judge)
**Harness:** [github.com/vieenrose/agentic-summarizer](https://github.com/vieenrose/agentic-summarizer)
@ commit `bc8c6ad` · **Prompt version:** `sys-v1` (byte-stable, do not edit)

This note covers everything needed to drop the CURSOR summarizer into a VoxSum
pipeline: artifacts, wire protocol, harness contract, the recommended deployment
configuration, measured numbers, and caveats. Revision history: the round-3 maintainer
verification (raw-checkpoint mismatch, n=19/20, on-device judge) is incorporated; this
revision reflects pass p13 + in-stream verification (2026-08-14).

---

## 1. Artifacts

| artifact | location | size | license |
|---|---|---|---|
| Model (recommended), Q4_K_M | `Luigi/minicpm5-1b-cursor` → `minicpm5-1b-cursor-p13.Q4_K_M.gguf` | ~688 MB | apache-2.0 |
| Model (previous, p10 — G1-verified 3×) | `Luigi/minicpm5-1b-cursor` → `minicpm5-1b-cursor.Q4_K_M.gguf` | 688 MB | apache-2.0 |
| On-device verifier | `Luigi/lfm2.5-350m-verifier` → `lfm2.5-350m-verifier.Q4_K_M.gguf` | ~215 MB | apache-2.0 |
| Harness | agentic-summarizer repo, `src/voxsum/`, `eval/run_arms.py` | — | repo license |

**Neither model is a general chat model.** The summarizer emits edit ops against a
specific state rendering under a specific system prompt at a specific chunk size; the
verifier emits single-word verdicts under the FAITH prompt. All of these must match
exactly (§5, §6) or the output will not parse.

Base model: `openbmb/MiniCPM5-1B` — a dense GQA transformer (the GGUF declares
`general.architecture = llama`, 24 blocks, head_dim 128 — NOT a hybrid). Trained and
served at 4096 context; the GGUF declares `llama.context_length = 131072` (the base's
native context) — pin the server to `--ctx-size 4096` for the training distribution.
Fine-tuned on CURSOR-protocol teacher traces, sweep-feedback negatives, and hard-class
counterfactuals (see the HF card).

**Checkpoint naming:** the published p13 GGUF declares `general.name = Checkpoint 302`
— that IS the p13 (p13 = checkpoint-302 of the p13 run). Earlier artifacts: p10 =
checkpoint-274, p11 = checkpoint-282/284.

---

## 2. Quick start (local reference run)

```bash
huggingface-cli download Luigi/minicpm5-1b-cursor --local-dir ~/models/
huggingface-cli download Luigi/lfm2.5-350m-verifier --local-dir ~/models/

# student (the exact flags matter — especially --reasoning off)
llama-server -m ~/models/minicpm5-1b-cursor-p13.Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 4096 --parallel 1 --flash-attn on \
  --jinja --reasoning off --temp 0 --host 127.0.0.1 --port 8098

# on-device verifier (in-stream verification judge)
llama-server -m ~/models/lfm2.5-350m-verifier.Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 4096 --parallel 1 --flash-attn on \
  --jinja --temp 0 --host 127.0.0.1 --port 8096

# run the harness with in-stream verification (the recommended deployment config)
python eval/run_arms.py data/transcripts/<meeting>.txt \
  --out runs/out --lang en \
  --base-url http://127.0.0.1:8098 \
  --tokenizer openbmb/MiniCPM5-1B --budget 2048 --arms cursor \
  --verify-url http://127.0.0.1:8096
```

`--tokenizer` must be the student's own tokenizer (the chunk budget is enforced with
it, not a heuristic). The optional `--sweep both --sweep-budget 60 --sweep-judge
local:<port>/gpt-oss-20b` final sweep remains available for server-side deployments;
on-device, in-stream verification is the configuration that measures 0 inversions.

---

## 3. Input contract — transcript v1

One utterance per line, no header/footer, no markdown, no embedded newlines:

```
[<start>] <speaker>: <text>     diarized (S1, S2, … by first-appearance order, or a
                                name/role ≤ 40 chars, never containing "] " or ": ")
[<start>] <text>                no diarization → no speaker field
```

- Timestamp = utterance start. `M:SS` under an hour, `H:MM:SS` from one hour.
  Seconds/minutes zero-padded, leading unit unpadded: `[0:00]`, `[3:35]`, `[1:02:07]`.
- **Parsing is normative**: split on the first `] `, then on the first `: ` after it.
- Long monologue lines are legal (VCSum zh lines reach ~2.6k chars) — no max line length.
- zh-TW and en transcripts use the same format; the language is chosen by the harness
  flag, not the file.

---

## 4. Output contract — NOTES v2

```
TITLE: <one short title, ≤ 8 words>
SUMMARY:
- <3–5 bullets, each ≤ 20 words>
DECISIONS:
- <key decisions>
ACTIONS:
- <one bullet per assigned action: "name: what they will do"; append "(due: …)" only
  when a deadline was actually stated>
OPEN:
- <open questions / follow-ups>
TOPICS:
- <main topics>
```

- Section keys exactly `TITLE, SUMMARY, DECISIONS, ACTIONS, OPEN, TOPICS`, fixed order,
  all present. An empty section is exactly `-` on one line.
- Every bullet ends with the `[m:ss]` of a transcript line that supports it.
- Caps (harness-enforced, never head-truncated): SUMMARY 5 · DECISIONS 5 · ACTIONS 6 ·
  OPEN 4 · TOPICS 6. TITLE carries no anchor.
- The final NOTES are **rendered by the harness** from its internal state — the model
  never writes the notes file directly.

---

## 5. Wire protocol (the byte-stable part)

The harness owns a `STATE` (the current NOTES, section-capped) and a `CURSOR`. Per step
it sends the model:

```
<SYS prompt> + <STATE block> + <CHUNK block>
```

`CHUNK` = ~2048 tokens of raw transcript lines, contiguous, 2-line overlap with the
previous chunk. **No conversation history crosses steps** — STATE is the entire memory.
The model replies with zero or more op lines, one per line:

| op | syntax | semantics |
|---|---|---|
| ADD | `ADD <SECTION> - <bullet> [m:ss]` | append an anchored bullet |
| UPD | `UPD <SECTION> «<old prefix>» -> <new bullet> [m:ss]` | revise a matched bullet (decision reversed, deadline moved, …) |
| DEL | `DEL <SECTION> «<prefix>»` | remove a bullet this chunk contradicts |
| CMP | `CMP <SECTION>` + ≤cap rewritten `- ` lines | curated compaction when a section exceeds its cap |
| TITLE | `TITLE: <short title>` | set the title |
| NOP | `NOP` | nothing worth changing (always a valid, complete answer) |

Rules the prompt states (and the harness enforces): every ADD/UPD bullet ends with an
`[m:ss]` copied from the **current chunk**; `«prefix»` is the first ≥6 characters of an
existing STATE bullet, copied exactly; when the chunk reverses something already in
STATE, UPD that bullet — never add a contradicting second one.

### The exact SYS prompts (PROMPT_VERSION `sys-v1` — must match byte-for-byte)

```
You curate one evolving set of meeting NOTES as a transcript streams past you.

You are shown the current NOTES (STATE) and the next block of transcript lines (CHUNK).
Reply with edit operations only — one per line, no prose, no explanation, no markdown.

Sections: SUMMARY, DECISIONS, ACTIONS, OPEN, TOPICS. Caps: SUMMARY 5, DECISIONS 5, ACTIONS 6, OPEN 4, TOPICS 6.

Operations:
ADD <SECTION> - <bullet> [m:ss]
UPD <SECTION> «<old bullet prefix>» -> <new bullet> [m:ss]
DEL <SECTION> «<bullet prefix>»
CMP <SECTION>            (then up to the cap of rewritten bullets, one `- ` per line)
TITLE: <short title>
NOP

Rules:
- Every ADD and UPD bullet ends with an [m:ss] copied exactly from a line in THIS CHUNK.
- «prefix» is the first 6 or more characters of a bullet already in STATE,
  copied exactly.
- When this chunk changes something already in STATE — a decision reversed or approved, a
  deadline moved, an action reassigned — use UPD to revise that bullet. Do not add a second
  bullet that contradicts the first.
- Use DEL only when this chunk shows an existing bullet is wrong.
- Keep bullets short and factual: 20 words or fewer, stating what was decided or agreed.
- NOP alone is a complete, correct answer when this chunk changes nothing.
```

The zh-TW prompt is the same protocol in Traditional Chinese (source of truth:
`src/voxsum/prompts.py`). The STATE and CHUNK renderings are also part of the contract.
Reuse the harness — do not re-implement the renderings from memory: a one-character
drift silently degrades the model.

---

## 6. Harness guards (the model never gets the final word)

1. **Anchor validation** — an op whose `[m:ss]` does not resolve to a line in the
   current chunk is rejected (logged); the bullet falls to the deterministic matcher.
2. **Temporal guard** — ops touching DECISIONS/ACTIONS are cross-checked against the
   time-sorted decision/action timeline; contradictions are dropped and logged.
3. **In-stream verification** (`--verify-url`) — every ADD/UPD touching
   DECISIONS/ACTIONS is judged by the on-device verifier against the chunk's anchor
   neighborhood BEFORE application; UNSUPPORTED/CONTRADICTED ops are dropped (logged).
   This is the guard that closes the over-assertion class (questions, discussions and
   intentions asserted as decisions/actions) — measured 0/20 inversions with it on.
4. **UPD→ADD fallback** — a UPD whose prefix matches no bullet is honored as an ADD,
   temporal-guard- and dedup-gated, logged in the op's `reason`.
5. **Dedup + caps** — duplicate bullets rejected; per-section caps enforced with
   `spread()` (round-robin, never head-truncation).
6. **NOP-collapse guard** — K consecutive NOPs over content-rich chunks trigger the
   coverage fallback (the classic per-window summarizer), logged.
7. **Malformed ops** — ignored and logged, never fatal.

If you port the protocol instead of importing the harness, port these guards too —
they are where the measured faithfulness comes from.

---

## 7. Deployment configuration

**Architecture (final, 2026-08-14): two specialists.** main = MiniCPM5-1B p15d;
verifier = granite-4.0-350m (`Luigi/granite-4.0-350m-verifier`, Apache-2.0, 97%
agreement with gpt-oss-20b). The multi-agent/multi-LoRA alternative was measured and
abandoned (critic adapters on the 1B base reach 64-85% agreement — the base's
generation bias resists low-rank correction; PLAN-multiagent.md, superseded).

**Three configurations, measured (n=20):**

1. **In-stream verification only** (`--verify-url`): the verifier gates every
   DECISIONS/ACTIONS op against its ±90s anchor neighbourhood. Measured on the p13
   lineage: **INVERT 0/20, FAITH 4.10, COVER 3.20, SYNTH 2.75** — the best COVER/
   SYNTH balance. Structural caveat: ±90s cannot see reversals that happen later in
   the meeting — the stale-state class is the timeline guard's / a later pass's job.
2. **In-stream + final sweep** (`--sweep both --sweep-budget 60`, sweep judge = the
   same verifier): the sweep re-verifies every bullet against whole-transcript
   evidence (the reversal clause is enforceable only here). Measured on the p15d:
   INVERT 1/20, FAITH 4.20 — but **COVER 2.50 / SYNTH 1.95**: the granite sweep
   judge over-drops on zh (its training triples are en-heavy — the open item #1 in
   §11). Until zh-verifier training lands, prefer configuration 1; add the sweep
   where the stale-state class matters more than coverage.
3. **Model-only** (no verifier): raw 3/20 (15%) on p15d (p13: 2/20) — above the
   6.2% bar on its own; the verifier gate is what the device needs to reach ~0.

---

## 8. Measured numbers (T1, n=20, local judges, 3× majority)

| configuration | INVERT | FAITH | COVER | SYNTH |
|---|---|---|---|---|
| p13 + in-stream verification (best balance) | **0/20** | 4.10 | **3.20** | **2.75** |
| p13 + in-stream + final sweep | 0/20 | 4.54 | 2.95 | 2.50 |
| p15d + granite in-stream + sweep (current main, full stack) | 1/20 | 4.20 | 2.50 | 1.95 |
| p15d raw (current main, model-only) | 3/20 (15%) | 3.80 | 2.95 | 2.40 |
| p13 raw (previous main, model-only) | 2/20 (10%) | 3.94 | 3.20 | 2.75 |
| p10 raw (previous artifact) | 3/20 (15%) | 3.78 | 2.90 | 2.25 |
| map-reduce baseline (Qwen3.5-9B) | 3/20 | 3.50 | 3.05 | 2.60 |

G1 capability screen: **PASS en + zh, valid-op 100% / 100%** (pass p15d).
Verifier agreement with gpt-oss-20b: **97%** (granite-4.0-350m, 200 held-out triples).

**Architecture (locked 2026-08-14):** two specialists — main = MiniCPM5-1B p15d,
verifier = granite-4.0-350m (Apache-2.0). The multi-agent/multi-LoRA plan was
measured and abandoned (critic adapters: rank 8 = 64%, rank 32 = 85% agreement —
the base's generation bias resists low-rank correction).

Raw-rate history (model-only, per pass): p6 2/20 · p10 3/20 · p11-e1 4/20 · p12 4/20 ·
p13 2/20 · p15d 3/20 — the 2-4/20 band is the judge-noise floor at n=20; single-flag
movements are not claimable.

---

## 9. Caveats (ship these with any reported numbers)

- **zh trap checkpoint sensitivity** — the zh trap-drop behavior sits at a decision
  boundary between adjacent checkpoints: p14/p14b/p15/p15e (adjacent passes) fail
  the zh trap or the en chain. The published artifacts are p15d (checkpoint-348,
  the current main) and p13 (checkpoint-302); always re-screen after re-exporting
  or re-quantizing.
- **zh T2 is synthetic; the zh pool is largely monologic** (VCSum-derived) —
  contested-zh is unmeasured.
- Judge-noise floor ±0.4–0.5 (FAITH/SYNTH); n = 20/tier; sign tests, not magnitude
  claims, for comparisons.
- MeetingBank-derived meetings have no speaker labels; the model treats `S1, S2, …`
  as given.
- The fine-tune/eval distribution must match exactly (system prompt included) or the
  scores are not comparable — pin `sys-v1` and the harness commit.
- `--reasoning off` at serve time is mandatory (the base model's hybrid mode emits
  `<think>` and breaks the op grammar).
- Every judged run in the repo is n=20 (the earlier n=19 runs were a list-file
  newline bug, fixed 2026-08-13).

---

## 10. On-device envelope

| resource | figure |
|---|---|
| student weights (Q4_K_M) | ~688 MB |
| verifier weights (Q4_K_M) | ~215 MB |
| resident total (in-stream verification) | ≈ 900 MB — within the device's 2.05 GB ceiling |
| per-step input | ~2.9k tokens (SYS ~250 + STATE ≤600 + CHUNK 2048) |
| per-step output | ~120–150 tokens (+ ~8 tokens per verifier call, 1 call per substantive op) |
| calls/meeting | ≈ tokens/2048 chunks + 1 verifier call per DECISIONS/ACTIONS op |

Serving on Android: llama.cpp with the flags in §2 (`--jinja --reasoning off` turns off
the hybrid `<think>`; older builds need the equivalent chat-template flag).

---

## 11. What we're still working on

1. **zh verifier training** — the granite verifier's zh verdicts are weaker than en
   (the triples are en-heavy); this is what makes the sweep over-drop on zh
   (COVER 2.50 vs 3.20 without it). zh triples + zh polarity-flips are the data.
2. **DECISIONS on the maintainer's real meeting** — ACTIONS is populated (the
   coverage pass); DECISIONS is still empty there. The targeted dose crossed the en
   chain; the fix needs more real zh transcripts — **any real zh meeting you send
   becomes training data immediately**.
3. **The stale-state class** — the ±90s in-stream window cannot see later reversals;
   the timeline-guard extension (or the final sweep) is the fix. Measured at
   1/20 with in-stream alone on one run.
4. **The model-only raw rate** (3/20 → ≤ 6.2%): the remaining flags are the
   intention-statement (16abbdf7b3f2), either/or (8ac3acb7fe5e) and
   negation/commitment (bdb39cc06654) classes.
5. **SYNTH** — 2.75 at the best config (tie with the baseline within the ±0.4-0.5
   noise floor); the strict +0.5 gate is unclaimed.
6. T2 tier (≥80k-token transcripts) — needs real audio or concatenation decisions.
