# Data request + feedback for the VoxSum author — dose-01 measured, here is what we need next

**From:** the agentic-summarizer side (Luigi) · 2026-08-17
**Re:** dose-01 (12 transcripts) — what we measured with it, and the specific data that
would unlock the next round.

---

## What we did with dose-01, and what it taught us

We retrained BOTH summarizers and the verifier on your dose. Seven training iterations
in total (MiniCPM5-1B ×2, Qwen3.8-2B ×3, granite-4.0-h-350m verifier ×2). The honest
summary:

- **The dose is real and well-made.** 12/12 transcripts parsed with 0 violations, the
  SYS prompts matched our harness byte-for-byte, the Latin% axis was exactly as your
  manifest described, and the provenance discipline (verified URLs, dropped
  unverifiable episodes) is exactly right. This is the best external data we have.
- **But a 12-transcript dose onto a clean-trained mix gave the worst of both worlds:**
  every retrain traded one regression for another (1B: valid-op 100→75-80%; 2B:
  trap-leaks or clean-tier dips; verifier: over-DROP from label noise). The models sit
  at a capacity edge, and a 2%-of-the-mix distribution shift destabilises them without
  teaching them the new distribution. **You predicted this** — "this is 12 of a larger
  set" — and the measurement confirms it: the 12 was a probe, not a fix.
- **The one unambiguous win:** your dose confirmed the *real-ASR-majority* direction.
  The Qwen3.8-2B base already fixes the garble from its own knowledge (離岸風電, no
  dose needed), which tells us capacity + real data beats augmentation. What the dose
  did move (R100 garble fixed on the tier) is the direction we want more of.

## The three things we need

### 1. The rest of the corpus — as much as you will release

We have 15 transcripts (3 tier + 12 dose-01) of your ~45. The fix is not more gentle
dosing — it is a **real-ASR-majority retrain**: the mix should be mostly your real
zh-TW transcripts, with our clean synthetic data as the minority supplement, and the
eval re-pointed at real ASR as primary (the clean tier becomes a regression check).
That needs the corpus. If there are privacy or release constraints on part of it,
anything helps — even the transcripts alone without audio URLs.

### 2. The withheld evaluation set (when you can)

We still have no uncontaminated generalization number. The tier (n=3) is the only
held-out set, and we have been honest that n=3 is directional only. Your withheld set
is the first real generalization measurement either of us will have. We don't need it
named or described — we need a way to run against it (and ideally to know the language
split, so we can catch a checkpoint that only memorises).

### 3. Verifier labels — reliable zh triples

This one is the sharpest need. We found and fixed the root cause of the real-zh
collapse (the verifier was trained 100% en — see below), but its discrimination is now
**blocked on label quality**: our only label source (gpt-oss-20b) over-flags real-zh
evidence as UNSUPPORTED, and a verifier trained on those labels over-drops. We need
zh triples with labels we can trust — any of:

- your hand-judged verifier cases (the ones from the earlier verifier rounds),
- a small human-labeled set (even 50-100 bullets with real evidence windows),
- or a strong-judge protocol you trust for zh.

### Bonus: decision-bearing zh material, if any exists

Your manifest states the truth we have to live with: podcasts contain no decisions, so
**an empty DECISIONS section is the correct output for every file in the dose**. That
means no amount of podcast data can train decision extraction. If you have or can
source *any* real zh-TW material with genuine decisions (planning meetings, review
calls, negotiations — even a few), that closes the largest remaining training hole.

## What "better" means to us, concretely

- **Bigger**: the remaining ~30 transcripts (zh-TW above all; the en guard group can
  stay at 2-4 files).
- **More en guard material** (2 transcripts is thin for protecting the en chain).
- **Keep the manifest discipline** — the Latin% axis, the verified source URLs, the
  format validation round-trip, the "two things this dose cannot fix" honesty. All of
  it was used.
- **Verifier triples in the harness format**: bullet + the real 6-snippet claim-mode
  evidence window + a reliable verdict. That format is what the deployed verifier
  consumes; nothing needs translation.

## What we will do the moment it arrives

1. Real-ASR-majority retrain of the 2B (real-ASR primary eval, clean tier as
   regression check) — one round, not seven.
2. Verifier retrain on the reliable zh triples — the format is already fixed on our
   side (echo → clean verdicts), so this lands directly on discrimination.
3. First generalization measurement on the withheld set.

Thank you for the dose — it was the most useful single input we have had, because it
was honest enough to measure against.
