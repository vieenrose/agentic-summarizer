# Phase 0a item 2 — en→zh-TW token ratio — 2026-08-21

Closes the last open Phase 0a question (SPEC.md §9 Phase 0a item 2): "any available
model, must not wait for TranslateGemma" — a units measurement, not a quality one.

- `measure_ratio.py` — the script run to produce this measurement.
- `ratio_measurement.json` — its raw output (per-segment and aggregate token counts).

**Method.** 5 real English segments from the cached `huuuyeah/meetingbank` HF mirror
(fine for THIS units-only measurement even though SPEC §2.2 requires the authoritative
Zenodo release for the actual training corpus — see CLAUDE.md "Corpus access"),
1,500–6,000 chars each. Translated with the already-running local Qwen3.8-27B instance
(`http://127.0.0.1:8082`, Q8_0, `--reasoning on --reasoning-budget 4096`) via a system
prompt requiring a complete, non-summarized translation, `temperature=0`,
`max_tokens=6000`. Both English and zh-TW text were measured under
`openbmb/MiniCPM5-1B`'s real tokenizer (`arcsum.tokens.hf_token_len`).

**Result: 1.2154613466334165 (4,010 en tokens → 4,874 zh-TW tokens, token-weighted
across all 5 segments).** All 5 calls returned `finish_reason=stop` (no truncation);
per-segment ratios ranged 1.117–1.302, a tight band with no outliers.

**A first attempt (same session, not saved) stalled on empty content.** Root cause:
`max_tokens` was left near the model's `--reasoning-budget 4096`, so the reasoning
trace alone exhausted the budget before any answer content was emitted. Fixed by
raising `max_tokens` to 6000 — well clear of the reasoning budget — which is the
config `measure_ratio.py` uses.

**Not yet done** (recorded, not skipped — see SPEC.md §9 Phase 0a): re-measure on a
larger sample once the Phase 1 pilot corpus itself is available, and re-measure with
TranslateGemma-27B specifically once it is cached, since this result used Qwen (the
eventual composition teacher) as a stand-in translator.

See `SPEC.md` §4.1, §7, §8 risk 6, and §9 Phase 0a for how this number revises the
step-count arithmetic (~11 → ~14 reading steps) and what it leaves open (Phase 0b's
reading-phase wall-clock total needs re-projection at the new step count).
