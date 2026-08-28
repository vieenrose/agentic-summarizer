# Real-ASR probe — risk 5 does not reproduce the collapse

**The agent curates a real zh-TW ASR transcript of a real meeting. The train/deploy ASR
gap is not what broke it on podcasts.**

| slice | domain | text | NOP | memory | summary |
|---|---|---|---|---|---|
| zh-TW podcasts | out-of-domain | ASR | **8/8 = 100%** | empty | fallback |
| LY committee | in-domain | stenographic | 5/20 = 25% | 9/4/3 pts | 258/244/276 ch |
| **LY negotiation** | **in-domain** | **REAL ASR** | **0/2 = 0%** | **2 pts** | **188 ch** |

This was the last confounded variable. Domain, language and script are now all held
fixed against the working case; only transcript quality varies, and curation survives.

## What was run

立法院朝野黨團協商, 2026-08-20 — a cross-party negotiation on the 國防自主無人載具採購
特別條例草案 (defence autonomous-vehicle procurement bill). 12 minutes of real audio
pulled from the LY IVOD stream via the g0v mirror (`v2.ly.govapi.tw/ivods`), transcribed
with **MOSS-Transcribe-Diarize 0.9B** (ASR + diarization, `tools/asr_transcribe.py`),
converted to format v2: 36 utterances, **7 speakers**, 0 UNK, 0 conformance defects.

The output is on-topic and specific — it names the three-committee joint session and the
Executive Yuan's version of the bill, both correct for this meeting:

> 本次會議首先處理一項關於無人機採購協商程式的提案… 該案經經濟、外交及國防、財政三委員會
> 第二次聯席會議審查後提交。

## Two deploy-path findings

**1. The ASR emits SIMPLIFIED Chinese.** MOSS transcribed this zh-TW audio as 会同意 /
我们 / 韩俊. This project is zh-TW only (SPEC §2), every training target is Traditional,
and `lang.simplified_hits` treats simplified characters as a language-guard failure — so
raw ASR output would trip the guard on script alone, confounding a script mismatch with
an ASR-noise effect. `tools/asr_transcribe.py` now converts s2twp by default; after
conversion, `simplified_hits = 0` and `cjk_ratio = 0.94`.

This is a real requirement for the VoxSum integration, not a detail of this experiment:
**the deploy pipeline needs Simplified→Traditional conversion between ASR and the
harness**, or the language guard will fire on every meeting.

**2. Half the IVOD recordings are silent.** The first session sampled
(經費稽核委員會, 2443s) has peak amplitude 0.008 — about −42 dBFS, no speech. The ASR
correctly emitted timestamp/speaker scaffold with no text, and I nearly debugged a model
that was working fine. Check RMS before spending ASR time: of 3 sessions sampled, 2 were
effectively silent and 1 had normal levels (RMS 0.40, peak 1.0).

## Limits

- **n=1 meeting, 12 minutes, 2 chunks.** A strong signal, not a settled result. The
  LY clean-text slice was 20 steps; this is 2.
- ASR utterances here are short and fragmentary, so this meeting produced fewer chunks
  than its wall-clock length suggests.
- No reference summary, so reference-free only: no ROUGE, no judge pass.
- One ASR system (MOSS). VoxSum's on-device stack may have a different error profile.

## Where this leaves the Phase 3 picture

Every candidate explanation for the podcast collapse has now been tested and eliminated
except domain:

| candidate | verdict |
|---|---|
| translationese / council-register-only | ruled out — `RESULT.md` |
| Latin ratio / code-switching | ruled out — `ABLATION.md` (no collapse to 14%) |
| thin chunks, malformed ops, guard refusals | ruled out — `../phase3-pilot/RESULT.md` |
| **ASR noise (risk 5)** | **ruled out — this file** |
| **domain (no decisions to record)** | **the remaining explanation** |

Podcasts contain no decisions, motions or commitments, so some NOP there is correct.
The open concern stays what it was: 100% is over-conservative for a format whose prompt
also asks for 關鍵論點 (key arguments), and the model has narrowed "worth recording"
toward formal proceedings.
