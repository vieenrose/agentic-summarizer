# VoxSum integration note — MiniCPM5-1B-CURSOR (agentic summarizer)

**Status:** production candidate · **Model:** `Luigi/minicpm5-1b-cursor`
**Harness:** [github.com/vieenrose/agentic-summarizer](https://github.com/vieenrose/agentic-summarizer)
@ commit `730ca7e` · **Prompt version:** `sys-v1` (see §5 — byte-stable, do not edit)

This note covers everything needed to drop the CURSOR summarizer into a VoxSum
pipeline: the artifacts, the wire protocol, the harness contract, the deployment
configuration, and the measured numbers with their caveats. It supersedes the earlier
`feedback-reply-to-voxsumdroid.md` thread (the 350M answers there are historical; this
model is the successor).

---

## 1. Artifacts

| artifact | location | size | license |
|---|---|---|---|
| Model (GGUF, Q4_K_M) | `Luigi/minicpm5-1b-cursor` → `minicpm5-1b-cursor.Q4_K_M.gguf` | ~650 MB | apache-2.0 |
| Model card (full measured record) | same repo, `README.md` | — | — |
| Harness (the CURSOR protocol implementation) | the agentic-summarizer repo, `src/voxsum/`, `eval/run_arms.py` | — | repo license |

**This is not a general chat model.** It emits edit ops against a specific state
rendering, under a specific system prompt, at a specific chunk size. All of those must
match exactly (§5, §6) or the model's output will not parse.

Base model: `openbmb/MiniCPM5-1B` (4k context, linear+full attention hybrid).
Fine-tuned on CURSOR-protocol teacher traces + sweep-feedback negatives (see the card).

---

## 2. Quick start (local reference run)

```bash
# one-time: download
huggingface-cli download Luigi/minicpm5-1b-cursor --local-dir ~/models/

# serve (llama.cpp, the exact flags matter — especially --reasoning off)
llama-server -m ~/models/minicpm5-1b-cursor.Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 4096 --parallel 1 --flash-attn on \
  --jinja --reasoning off --temp 0 --host 127.0.0.1 --port 8098

# run the harness over transcripts
python eval/run_arms.py data/transcripts/<meeting>.txt \
  --out runs/out --lang en \
  --base-url http://127.0.0.1:8098 \
  --tokenizer openbmb/MiniCPM5-1B --budget 2048 --arms cursor \
  --sweep both --sweep-budget 60 --sweep-judge local:8090/gpt-oss-20b
```

`--sweep both` runs the VERIFY/ANCHOR final sweep (§7) — that is the deployment
configuration. `--tokenizer` must be the model's own tokenizer (the chunk budget is
enforced with it, not with a heuristic).

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
`src/voxsum/prompts.py` in the harness repo). The STATE block and CHUNK block renderings
are also part of the contract (STATE per section, bullets with their `[m:ss]`; CHUNK as
raw transcript lines). Reuse the harness — do not re-implement the renderings from
memory: a one-character drift silently degrades the model.

---

## 6. Harness guards (the model never gets the final word)

The harness is deterministic and owns the final word:

1. **Anchor validation** — an op whose `[m:ss]` does not resolve to a line in the
   current chunk is rejected (logged); the bullet falls to the deterministic matcher.
2. **Temporal guard** — ops touching DECISIONS/ACTIONS are cross-checked against the
   time-sorted decision/action timeline; contradictions are dropped and logged
   (the 0%-inversions backstop).
3. **UPD→ADD fallback** — a UPD whose prefix matches no bullet is honored as an ADD
   (the model's intent: this bullet belongs in STATE). Guarded by the temporal guard
   and the dedup check; logged in the op's `reason`. This converts the model's rare
   "UPD against empty state" into a correct ADD instead of a lost bullet.
4. **Dedup + caps** — duplicate bullets are rejected; per-section caps are enforced
   with `spread()` (round-robin over sections, never head-truncation).
5. **NOP-collapse guard** — K consecutive NOPs over content-rich chunks trigger the
   coverage fallback (the classic per-window summarizer), logged.
6. **Malformed ops** — ignored and logged, never fatal.

If you port the protocol instead of importing the harness, port these guards too —
they are where the measured faithfulness comes from.

---

## 7. Deployment configuration: model + sweep

The VERIFY/ANCHOR sweep runs once after the stream ends, per bullet, in loop-free
single calls:

- **VERIFY** — per bullet: ≤6 evidence snippets (anchor neighborhood ∪ lexical top-k
  over the whole transcript, ≤120 chars each) → `KEEP` / `DROP` / `FIX: <rewrite>`.
  FIX is applied as DROP (the model's rewrites were measured unsafe); the bullet then
  falls to the deterministic matcher.
- **ANCHOR** — per bullet: ≤8 lexical top-k candidate lines → the model picks the
  `[m:ss]` that states the claim, or `NONE` (→ deterministic matcher).

**Sweep judge = gpt-oss-20b** (local llama.cpp, the FAITH protocol, 3× majority) — the
sweep is the reason the deployed pipeline measures **0 inversions**. The sweep judge is
a 20B model: on a phone-class device, plan for either (a) running the sweep on a
companion/edge host, or (b) accepting the model-only raw rate below (10%, above the
6.2% on-device bar) until the raw rate is trained down further. The model-side raw rate
is the number we are actively improving; the sweep is the current deployment backstop.

**Sweep budget:** 60 calls/meeting at eval (default `--sweep-budget 60`); each call is
~900 tokens. The sweep is deterministic in structure (budget-gated, loop-free).

---

## 8. Measured numbers (T1, n=20, local judges, 3× majority)

| metric | MiniCPM5-1B-CURSOR (checkpoint-274) | map-reduce baseline (Qwen3.5-9B) |
|---|---|---|
| G1 capability screen (en / zh) | **PASS / PASS** (chain, deadlines, anchored, trap) | — |
| valid-op rate (screen) | en 100% / zh 88%* | — |
| raw INVERT (model only, no sweep, n=20) | **3-4/20 (15-20%)** across passes p10/p11-e1/p12 (the p10 ship-table quote of 2/20 came from p6 — corrected after maintainer verification 2026-08-13) | 3/20 |
| swept INVERT (model + VERIFY/ANCHOR sweep, n=20) | **0/20 (0%)** | — |
| FAITH-claim (1–5) | **4.81–4.84** | 3.50 |
| COVER (1–5) | 2.84–2.89 | 3.05 |
| SYNTH (1–5) | 2.32 | 2.60 (tie within judge noise ±0.4–0.5) |
| prefill | ~2.9k tokens/step, ~1.2× the map-reduce baseline | 1.0× |

\* zh 88% = one redundant duplicate-ADD per screen run, correctly rejected by the
dedup guard; the notes are unaffected.

**On-device status — RESOLVED 2026-08-13:** the objection "a phone has no 20B judge"
is now answered. The sweep judge has been replaced by an on-device verifier:
**`Luigi/lfm2.5-350m-verifier`** (~215 MB Q4_K_M), fine-tuned on the pipeline's own
judged (bullet, evidence, verdict) triples, class-balanced; measured **96% agreement
with the 20B judge** on 200 held-out triples. The full deployment — MiniCPM student
(688 MB) + verifier (215 MB) — runs the sweep on-device and measures, with the
on-device verifier as the sweep judge:

| metric | on-device deployment (student + 350M verifier sweep) |
|---|---|
| G1 screen (en / zh) | PASS / PASS, valid-op **100% / 100%** (pass p12) |
| swept INVERT (n=20) | **0/20** |
| FAITH-claim | **4.54** (baseline 3.50, +1.04) |
| COVER | 2.95 (baseline 3.05, tie) |
| SYNTH | 2.50 (baseline 2.60, tie) |

Memory: 688 + 215 MB ≈ 900 MB if both models resident — within the device's 2.05 GB
ceiling.

**Updated 2026-08-14 (p13 + in-stream verification).** The recommended artifact is now
**`minicpm5-1b-cursor-p13.Q4_K_M.gguf`** (same HF repo), and the harness gained an
**in-stream verification mode** (`--verify-url <verifier endpoint>`): every ADD/UPD
touching DECISIONS/ACTIONS is judged by the on-device verifier against the chunk's
anchor neighborhood BEFORE application; UNSUPPORTED/CONTRADICTED ops are dropped. This
closes the over-assertion class at application time instead of in a final sweep.

| configuration | INVERT (n=20) | FAITH | COVER | SYNTH |
|---|---|---|---|---|
| p13 model-only (no verifier) | 2/20 (10%) | 3.94 | **3.20** | **2.75** |
| p13 + in-stream verification (all on-device) | **0/20** | 4.10 | **3.20** | **2.75** |
| map-reduce baseline | 3/20 | 3.50 | 3.05 | 2.60 |

Model-side progress that produced p13: the over-assertion counterfactual beats
(soft-action, either/or, informal-negation, intention-statement) injected into REAL
transcripts (14 augmented meetings), teacher-traced with beat-assertion records
filtered out. p13 is G1 PASS both languages with 100% valid-op on both.

The model-only raw rate without any verifier is 2/20 (10%) — still above the 6.2% bar
on its own. With the in-stream verifier (part of the harness the device runs), the
device measures 0 inversions at better-than-baseline COVER and SYNTH.

Ship-rule reading (spec §7.7): GT2 (faith) clears decisively — FAITH +1.3 at fewer
inversions than the baseline (0 vs 3, swept). GT3 (SYNTH) is a statistical tie, not a
win. The owner bar (raw < 6.2%) is met by the **deployed** pipeline (model + sweep);
the model-only raw rate is 10% and is the current training target.

---

## 9. Caveats (ship these with any reported numbers)

- **zh trap checkpoint sensitivity** — the zh trap-drop behavior sits at the decision
  boundary between adjacent checkpoints: the published artifact is **checkpoint-274**
  (G1-verified three times); the training final (284) fails the zh trap. Always
  re-screen after re-exporting or re-quantizing.
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

---

## 10. On-device envelope

| resource | figure |
|---|---|
| weights (Q4_K_M) | ~650 MB |
| runtime memory | within the ~785 MB envelope (Q4_K_M, 4k KV) |
| per-step input | ~2.9k tokens (SYS ~250 + STATE ≤600 + CHUNK 2048) |
| per-step output | ~120–150 tokens |
| calls/meeting | ≈ tokens/2048 chunks + ≤60 sweep calls |

Serving on Android: llama.cpp with the flags in §2 (the `--jinja --reasoning off`
combination is what turns off the hybrid `<think>`; older builds need the equivalent
chat-template flag). The 20B sweep judge is not phone-resident — see §7.

---

## 11. What we're still working on

1. The model-only raw rate (10% → ≤ 6.2%): hard-class counterfactual training is the
   active lever (proposed-never-decided, informal-negation, negative-preference,
   reject-action meetings); each pass moves ~1 flag per tier.
2. SYNTH tie → win (more real-meeting arc data; the T2 ≥80k-token tier when real
   transcripts exist).
3. zh duplicate-ADD polish (the 88% valid-op note).
