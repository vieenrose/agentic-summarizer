# Phase 3 pilot — the agent collapses to total NOP on out-of-domain zh-TW

**Headline: 8 of 8 reading steps returned bare `NOP`. All three summaries are the
empty-memory fallback constant. The agent built no memory at all.**

This is a *pilot*, not Phase 3 — see "What this is not" below. But the signal is stark
enough to matter before Phase 4 is funded.

## What was run

The 3 real zh-TW ASR transcripts from the prior project
(`pi-agent:asr-transcripts-2026-08-16`), converted v1 -> v2 by `tools/v1_to_v2.py`
(strict: refuses any unrecognised line rather than emitting a `UNK:` utterance).
Verified through the real parser: 0 UNK, 0 v2 conformance defects across 260 lines.

Both arms, `sft-dropv2`, identical pinned config to the n=20 gate run
(`cache_prompt: false`, raw completion route, `repeat_penalty 1.1` on prose only).

## Result

| meeting | chunks | agent | baseline |
|---|---|---|---|
| 01-cerebras-16m | 3 | **fallback constant** (20 chars) | 290 chars |
| 02-materials-17m | 3 | **fallback constant** (20 chars) | 492 chars |
| 03-tsmc-wind-10m | 2 | **fallback constant** (20 chars) | 271 chars |

Every agent summary is byte-identical to `agent.EMPTY_MEMORY_PROSE`
(`本次會議沒有記錄到具體的決議或討論重點。`). Per-step traces:

```
01-cerebras-16m : NOP, NOP, NOP
02-materials-17m: NOP, NOP, NOP
03-tsmc-wind-10m: NOP, NOP
```

## Three explanations ruled out

1. **Not thin chunks.** The chunks are full and content-rich by the harness's own
   `is_content_rich` heuristic: 2481, 2465, 2443, 2485, 1988, 2498 tokens. Only the two
   short tail chunks are not rich, and they are not what drove this.
2. **Not malformed output or guard refusals.** The raw output is the literal string
   `NOP` — parsed cleanly, applied successfully. Nothing was rejected by a cap, a
   language check or the contradiction guard. The model *chose* to record nothing.
3. **Not an inability to read the text.** The SAME model on the SAME chunks produced
   fluent, on-topic zh-TW summaries through the baseline's map path. Comprehension is
   intact; what fails is specifically the curation behaviour.

That third point is the important one. This is not "the model is confused by ASR noise";
it is "the model does not recognise this as something to curate."

## Why this bears on Phase 4

SPEC §9 Phase 3's gate: *"no catastrophic degradation versus the clean-text corpus. A
large drop here means the model learned council-procedure text, not meeting
summarization, and no amount of additional MeetingBank data fixes that."*

Total NOP collapse is the maximal degradation the metric can express. If it reproduces
on real meetings, Phase 4 — translating ~1,050 further American council transcripts —
would be scaling exactly the distribution that produced the failure.

It also matches the prior project's precedent recorded in SPEC §8 risk 5: a verifier
that scored 8/9 on clean zh and **0/11 on real noisy zh**.

## What this is NOT

Stated plainly, because the result is easy to over-read:

- **n=3, and they are PODCASTS, not meetings.** Tech commentary with one or two
  speakers (S1; S1+S2; S1). SPEC Phase 3 requires ~20 genuinely multi-speaker
  recordings. These have no decisions, motions or votes — the very things `ADD`/`ARC`
  are trained to capture — so some NOP is *correct* here.
- **Two variables are confounded.** Domain (podcast vs meeting, risk 1) and ASR noise
  (risk 5) move together in this slice; it cannot say which drives the collapse.
- **It does not by itself fail Phase 3.** It is a cheap early read using assets that
  already existed.

The reason it is still worth acting on: 8/8 with *zero* ops is not a graceful
degradation. A model that merely found this domain harder would emit fewer or worse
ops, not none. And a 17-minute technical discussion with two speakers is not devoid of
recordable content — the baseline extracted 492 characters of it from the same input.

## Suggested next step

Disambiguate domain from ASR noise before spending on Phase 4, using material that is
in-domain (a real multi-speaker meeting with decisions) but ASR-transcribed. 立法院
committee sessions are SPEC's named fallback and are publicly available. If the agent
curates those normally, this pilot is a domain artifact of podcasts and Phase 3 proper
can proceed. If it NOPs there too, the Phase 2 gate's "diagnosably data-volume-bound"
exit does not apply, and Phase 4 should not be funded on volume grounds.
