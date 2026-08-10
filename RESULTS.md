# RESULTS — teacher screen, gemma-4-31B-it

**Date:** 2026-08-10 · Harness `sys-v1` · 2× RTX 5090 · llama.cpp `llama-server`
Screen: `eval/screen.py` on the planted-fact set (`voxsum.screenset`), no GBNF grammar.

Every number here is from the **teacher screen** (PLAN.md §2b) — a go/no-go on trace
generation. It is **not** a student result and not a ship gate. n is tiny; read it as a
disqualifier, not a ranking.

---

## Headline

**Both quants pass with thinking on. Neither is reliable with thinking off — and the
failure is zh-TW-specific.**

| config | en | zh-TW | wall / meeting | valid-op | anchor (raw) |
|---|---|---|---|---|---|
| Q8_0, thinking **on** | 3/3 PASS | 3/3 PASS | ~55 s | 100% | 100% |
| UD-Q4_K_XL, thinking **on** | 1/1 PASS | 1/1 PASS | ~48 s | 100% | 100% |
| Q8_0, thinking **off** (ctx 16k) | 5/5 PASS | 5/5 PASS | ~3 s | 100% | 100% |
| Q8_0, thinking **off** (ctx 4k) | 1/1 PASS | **0/1 PASS** | ~3 s | 80% | 100% |
| UD-Q4_K_XL, thinking **off** | 5/5 PASS | **1/5 PASS** | ~3 s | 90–100% | 100% |

The Q8-thinking runs span both ctx settings (one at 4096, two at 16384) and passed at
each, with `revised_at_contradiction` true on every language-run — so thinking-on is the
only config that has not produced a single zh-TW inversion.

G1 criteria: decision chain rejected→approved, both deadlines, 100% anchored, no trap.

---

## What the failures actually look like

zh-TW, Q4, thinking off — the notes state the *opposite* of the meeting's outcome:

```
DECISIONS:
- 否決目前的倉庫整併方案 [2:00]        <- "rejects the current plan"
```

The transcript's later line 「倉庫整併方案通過」 ("the plan is approved") is never applied.
The model emitted the early rejection and then left it standing. English on the same
config revised correctly every time.

**This is the exact failure the screen exists to detect, and it is invisible to a keyword
check** — the word "approved" being absent is the only thing wrong, and a summary
containing a real, verbatim-supported decision looks fine to any faithfulness metric that
does not compare polarity across time.

## Conclusions

1. **Teacher agency is language-asymmetric.** The revise-don't-append behaviour that GT3
   depends on is markedly weaker in zh-TW than en at equal capability. Since contested
   zh-TW is already the project's biggest unmeasured caveat (CLAUDE.md §7.8), this is the
   single most important finding here — and it argues the zh trace set needs *more*
   oversampling of revision points than en, not the same.
2. **Thinking is what buys the agency, not the quant.** Q4-with-thinking beats
   Q8-without on the one metric that matters. Trace generation should run **thinking on**;
   per PLAN.md §2c that is legitimate (extra compute on the same input), and only the op
   lines are ever kept as a target.
3. **Q8 vs Q4 is not the deciding axis.** With thinking on, both reach 100% on every
   measure (Q8 n=3, Q4 n=1). If a GPU is needed elsewhere, Q4 on one card is a defensible
   choice; Q8's thinner-evidence margin is not grounds to insist on it.
4. **Anchor copying is a non-issue for the teacher.** 100% raw anchor rate in every
   config, both languages. Whether the *student* can copy digits post-quantisation is a
   separate question and still open.

## Cost of the recommended config

Thinking on, ~15–20 s per step. For ~1200 steps that is **≈5–7 h unattended** on one
server, against ~1 h thinking-off. Given (2), the extra hours buy the behaviour the whole
GT3 bet rests on, so they are worth paying once.

---

## Caveats — read these before quoting any number above

* **n=3 (Q8) / n=1 (Q4) per thinking cell, n=5 per thinking-off cell.** No confidence
  intervals. A single planted meeting per language, synthetic, written by the same author
  as the harness. Q4-with-thinking is the thinnest cell in the table and the one most
  likely to move.
* **Runs are not reproducible despite `seed=0`.** Q4 thinking-off produced zh PASS on the
  first run and FAIL on the next four. Prompt-cache state and slot reuse across `--parallel`
  appear to matter; treat any single screen run as noise and require n≥3.
* **`CTX` changes the result, even with thinking off.** The same Q8 config failed zh at
  `--ctx-size 4096` and passed 5/5 at 16384. `--ctx-size` is divided across `--parallel`
  slots and bounds prompt+output together, so a nominally adequate window can still starve
  a step. Every number above records its ctx for this reason.
* **Two metric bugs were found and fixed while producing this table** — a `valid-op rate`
  of 110% (NOP counted in the numerator but not the denominator) and `anchor_rate_raw`
  scoring NOP/TITLE as natively anchored. Both inflated. Numbers from before those fixes
  are not comparable; these were produced after.
* **The screen is not G1 for the student.** It reuses G1's criteria to disqualify a
  *teacher*. Passing here says nothing about whether FunctionGemma-270M can learn the task.

## Reproducing

```sh
CTX=16384 tools/serve_teacher.sh                    # Q8 across both GPUs
QUANT=UD-Q4_K_XL CTX=16384 tools/serve_teacher.sh   # Q4 on one GPU

python eval/screen.py --thinking --max-tokens 6144 --notes-out /tmp/screen
python eval/screen.py                               # thinking off
```

Exit code 0 = G1 passed on every language screened.
