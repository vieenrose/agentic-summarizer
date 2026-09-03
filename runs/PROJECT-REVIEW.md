# Project review — `next` branch, 2026-09-03

Written after a long measurement session. The short version: **the harness and the
evaluation infrastructure are in good shape; the model is not, and the reason is now
known.** The product target has not been met and there is a real question about whether
this architecture is the right way to reach it.

---

## 1. Where the project stands against its own goal

SPEC §5.2 defines seven gates and an all-or-nothing rule. **No checkpoint has ever cleared
them.** Current state of the two best candidates, all on the same corpus in the deployed
cache-on configuration:

| | `qwen-tools-v5` (shipped) | `spec-e3` ep2 (newest) |
|---|---|---|
| G1 revision | FAIL 3/27 | FAIL |
| G2 faithfulness | PASS 16 vs 58 | **PASS 21 vs 41** |
| G3 rouge1 | PASS +0.069 | **FAIL −0.013** |
| G3 rouge2 | PASS +0.041 | **FAIL +0.007** |
| G3 rougeL | PASS +0.057 | PASS +0.026 |
| G4 budget | 19.0 min measured (3% margin) | not re-measured |
| real-ASR clean | 8/21 | 11/21 |
| real-ASR churn | 26.7% | **0.0%** |
| fabrication rate | 33.3% of 33 specifics | **15.6% of 32** |

`v5` is what ships. **`v5` is not a good product**: it churns on a quarter of its steps and
fabricates a third of the specifics it asserts. It passes the gates anyway, which is the
single most important fact in this review.

---

## 2. The central finding: the gates certified a bad model

### G3's references are ~half unreachable

Measured on the 40 held-out meetings: of 454 specific claims in
`data/heldout_references.json`, **211 (46.5%) do not appear in the transcript being
summarised**. Conservatively — Arabic numbers and Latin identifiers only, where numeral
reformatting cannot be the excuse — still **160 (35%)**. Values like `94009`, `166513`,
`2200000`. They come from MeetingBank's minutes documents, which §2.2 stage 3 composed the
references from. The agent never sees them.

Two consequences:

1. G3 has a ceiling no faithful agent can reach.
2. **The correlation runs backwards.** `v5` fabricates 33.3% and passes 3/3. `spec-e3`
   fabricates 15.6% and passes 1/3. Overlap with a reference full of
   invented-from-elsewhere detail is best achieved by a model that invents in the same
   style. **A gate that rewards fabrication is worse than no gate.**

Newly composed references (teacher reads the whole transcript, `tools/build_reachable_refs.py`)
score **2.6% ungrounded across 503 specifics** — so this is fixable, and 25 of 40 meetings
now have reachable references. The remaining 15 exceed the teacher's context.

### The real-ASR gate rewarded the failure it existed to catch

`asr_gate.py` counts a meeting "curated" when the summary clears a length floor. A
553-character confabulation built from ONE churned memory point clears it. The 19/20 that
justified shipping `mixed-e3` was counting exactly that. `mixed-e3` was published to the
demo and rolled back the same day after a user log showed it churning.

---

## 3. What the supervision was actually teaching

Three defects, all found by pointing the new instruments at the pool rather than the model:

| defect | measurement | fixed? |
|---|---|---|
| churn (`DROP` + near-identical re-`ADD`) | 106 / 4,540 reading rows | **yes** — removing them took churn 28.2% → 0.0% |
| ARC ops the harness ALWAYS refuses | 339 / 2,006 consecutive ARC emissions | **yes** — stripped |
| `SYNTHESIZE` targets asserting things absent from their own memory | **1,347 / 3,376 = 39.9%**, 45% of rows | partly — regeneration → 24.9% |
| gold `ADD` targets carrying no figure though the chunk has one | 58% of targets, while 99% of chunks contain a specific | **yes** — enrichment → 42% → 60% |

The last is the cleanest result of the session. The student **tracks its supervision**:
point-level specificity went 33% → **46%** when the ceiling went 42% → 60%. That confirms
the deficit is a data ceiling, not model capacity, and it predicted the outcome in advance.

**The `SYNTHESIZE` defect is structural, not a labelling slip.** §2.2 pairs
`(memory) → (whole-meeting gold summary)`, so the target is not a function of the input.
The model was shown 450 times that the right answer to a memory contains things absent from
it. It learned exactly that: 44% ungrounded on real ASR.

---

## 4. Is the architecture earning its keep?

**Probably not, on current evidence.** On 25 meetings with reachable references:

| system | chars | specifics/meeting | ungrounded |
|---|---|---|---|
| 27B teacher, ONE PASS | 873 | **20.1** | 3% |
| 0.8B map-reduce baseline | 741 | **8.5** | 5% |
| 0.8B agent (`spec-e3` ep2) | 289 | **3.6** | 7% |

The agent **loses to its own baseline** — 3.6 vs 8.5 grounded specifics, at a slightly
worse fabrication rate. SPEC §5.2's standing "ship the baseline" decision has been correct
all along, and this is the mechanism.

### Can a single long-context pass replace it?

Measured on the actual Reno 7 (`-C 0xFF`, Q4_K_M):

| model | pp2048 | pp8192 | pp32768 |
|---|---|---|---|
| LFM2.5-1.2B | 56.4 | 46.0 | **25.8** |
| granite-4.0-h-1b | 38.3 | 34.6 | (running) |

**80k in 20 min needs ≥67 t/s sustained. Nothing measured is close.** Even 32k at LFM2.5's
25.8 t/s is 21 minutes — over budget. The 1.2B hybrids are out.

Memory was never the constraint: KV at 80k is 0.92 GB (LFM2.5-1.2B), 0.61 GB
(granite-4.0-h-1b), **0.31 GB** (granite-4.0-h-350m) against a 2.5 GB ceiling. And note
**Qwen3-0.6B is disqualified outright** at 114,688 B/token → 8.54 GB, despite being
*smaller* than the current model — smaller parameters do not mean smaller footprint.

The 350M class is still being benchmarked. If it fits on time, the quality question becomes
acute; if it does not, the chunked architecture is justified on latency grounds and the
argument for it becomes much stronger than it looks today.

---

## 5. What is genuinely in good shape

- **The harness.** Zero core dependencies; the whole suite runs with no GPU, weights,
  network or optional extra. 39 test files, all green, lint clean.
- **The instruments built this session** (`src/arcsum/evalkit/`): provenance with a
  comparison that REFUSES, behaviour (churn/starvation/confabulation/under-rendering/
  abstention), deterministic reference-free grounding, and a scorecard that persists
  per-meeting rows. Plus `arcsum-eval` as a single entry point.
- **The discipline of recording refuted hypotheses.** `CLAUDE.md` now carries 11 traps and
  a long list of measured negatives. That is why this session did not re-derive them.

## 6. What is not

- **Disk.** Was at 100% (5.2 GB free) — the exact precondition that corrupted
  `selfdistil-e3/checkpoint-918`. Reclaimed to 57 GB by deleting optimizer states from
  7 exported runs. `runs/` is 229 GB and needs a retention policy.
- **52 uncommitted files**, including the entire `evalkit` package and two new CLI
  entry points. Nothing is committed from this session.
- **G1 has never worked** and five fix attempts are refuted. The loss is now located
  (revision-specific, `key_term` dropped during the replacement) but not fixed.
- **Local evaluation cannot reproduce deployed behaviour.** Same GGUF, same transcript:
  llama-server GPU → 4 points 0 churn; llama-cpp-python CPU → 4 points 0 churn; the actual
  Space → 1 point 4 churn. Greedy decoding is deterministic only for a fixed
  floating-point reduction order, which depends on the host.

---

## 7. Recommendations, in order

1. **Rebuild the G3 reference set** and re-run every gate against it. Until then, no G3
   number means what it appears to mean — including the ones that certified `v5`. 25 of 40
   are done; the remaining 15 need either a longer-context teacher pass or an explicit
   decision to score on the 25.
2. **Commit this session's work.** The evalkit is the most valuable artifact produced and
   it is currently untracked.
3. **Decide the architecture question on the 350M benchmark.** If a single pass fits the
   budget, the agentic memory is unnecessary complexity that is currently *losing* to
   map-reduce. If it does not, that is the strongest evidence yet that the design is right.
4. **Carry the pool fixes forward regardless** — churn removal, ARC stripping, point
   enrichment and synthesis regeneration are all verified data repairs that survive any
   architecture decision.
5. **Do not ship `spec-e3`.** It is better on every deployment-facing axis and worse on the
   gates. Given finding §2, that pattern is now ambiguous rather than damning — but the
   honest move is to fix the gate first, then re-judge.
