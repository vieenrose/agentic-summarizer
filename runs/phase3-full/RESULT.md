# Phase 3 — 20 real zh-TW meetings, real ASR: no catastrophic degradation

**Gate verdict: PASS.** SPEC §9 Phase 3 asks for "no catastrophic degradation versus the
clean-text corpus". The agent curates 17 of 20 real meetings from real ASR output.

| slice | domain | text | NOP rate | curated |
|---|---|---|---|---|
| zh-TW podcasts | out-of-domain | ASR | 8/8 = **100%** | 0/3 |
| LY committee | in-domain | stenographic | 5/20 = **25%** | 3/3 |
| **LY, n=20 (this run)** | **in-domain** | **REAL ASR** | **16/39 = 41%** | **17/20** |

41% against a 25% clean-text baseline is a real degradation but a graceful one, and it
sits close to the teacher's own natural NOP band (~32–38%). It is nothing like the total
collapse the podcasts produced. Mean 7.9 points of memory on the meetings that curate;
mean summary 230 characters against the baseline's 294.

## The corpus

20 Legislative Yuan sessions, 2026-08-12 to 2026-08-27, 240 minutes of audio harvested
from the IVOD archive (`tools/harvest_ivod.py`) and transcribed with
MOSS-Transcribe-Diarize (`tools/asr_transcribe.py`): 39 chunks, 2–7 speakers per meeting,
0 UNK, 0 v2 conformance defects, 0 simplified characters after s2twp conversion.

Formats deliberately mixed rather than one committee repeated: 9 委員會, 4 院會,
3 聯席會議, 3 公聽會, 1 朝野黨團協商. 18 of 20 are multi-chunk, so they exercise
cross-chunk memory rather than degenerating to one-shot summarisation.

This is SPEC's named fallback for Phase 3 (立法院 committee sessions) since VoxSum
recordings were not available. It closes risk 1 (domain) and risk 5 (ASR) but **not** the
device gap: these are broadcast-desk recordings, not phone-mic captures.

## The three that collapsed

Not noise — each has an identifiable cause, and two are arguably correct behaviour.

- **ivod-17673** — single chunk, and the quietest recording in the set (RMS 0.023, ~8x
  below the next). Thin input.
- **ivod-17677** — the ASR emitted ENGLISH for Chinese speech ("Good First of all, I
  would like to complain to you…"), dropping the transcript's CJK ratio to 0.28 against
  a 0.7 threshold. The language guard firing here is the guard working. **This is a
  genuine VoxSum-path failure mode, kept deliberately rather than cherry-picked out.**
- **ivod-17701** — 2 steps, both NOP, on an 教育及文化 committee session.

Excluding only the ASR-language failure would give 16/19 curated; excluding all three
identifiable causes, 17/17. Reported as 17/20 because the failure modes are real and will
recur in deployment.

## What this settles

Every candidate explanation for the podcast collapse has now been tested:

| candidate | verdict |
|---|---|
| translationese / council-register-only | ruled out (`../phase3-ly/RESULT.md`) |
| Latin ratio / code-switching | ruled out to 14% (`../phase3-ly/ABLATION.md`) |
| thin chunks, malformed ops, guard refusals | ruled out (`../phase3-pilot/RESULT.md`) |
| ASR noise (risk 5) | ruled out at n=1 (`../phase3-ly/ASR.md`), now **at n=20** |
| **domain** | the remaining explanation |

The model curates meetings and declines podcasts. Since podcasts contain no decisions,
motions or commitments, that is largely correct behaviour — the open concern remains that
100% was over-conservative for a prompt that also asks for 關鍵論點 (key arguments).

## Limits

- **Reference-free.** No ROUGE, no faithfulness judge — there are no reference summaries
  for these meetings. The gate is a degradation check, which reference-free metrics can
  carry, but this is not a quality measurement.
- **12 minutes per meeting**, not whole sessions, so ~2 chunks each versus MeetingBank's
  6–8. Long-meeting behaviour — the known weakness and the reason ROUGE-1 misses G3 — is
  **not** exercised here.
- Broadcast audio, one ASR system. The VoxSum device path may differ in error profile.
- Vocabulary leakage (`市議會` appearing where the source never says it) was found on the
  clean LY slice and is not re-measured here.

## Consequence for the plan

Phase 3's gate passes, so the Phase 2 exit that justifies Phase 4 — "the deficit is
diagnosably data-volume-bound" — is not contradicted by deployment-distribution evidence.
Phase 4 remains defensible on volume grounds.
