# Latin-ratio ablation — hypothesis REFUTED, cause narrowed to domain

**Code-switching is not what breaks curation. The podcast collapse is a domain effect.**

## The hypothesis

The three Phase 3 slices differ from the training distribution on three axes, and only
one moved far:

| slice | punct/100 | filler/100 | **latin/100** | result |
|---|---|---|---|---|
| MeetingBank zh (trained on) | 9.27 | 0.59 | **1.73** | — |
| LY committee | 7.42 | 1.59 | **4.82** | 25% NOP, curates fine |
| zh-TW tech podcasts | 5.67 | 2.57 | **10.30** | **100% NOP, total collapse** |

Latin at 6x the training rate, plus the prior project's own note on one of these files
("THE FLIP CASE — 22.3% Latin, both checkpoints answer in English"), made
code-switching the obvious suspect. If true it would reframe the collapse as a
language-mixing problem, not the ASR gap (risk 5).

## The test

Hold domain, speakers and language fixed; vary only the Latin ratio. Starting from LY
transcripts the agent *does* curate, substitute common Chinese terms with the English a
bilingual Taiwanese speaker would actually code-switch to (`委員會`→`committee`,
`資料`→`data`, …), meaning preserved, so any drop cannot be blamed on the text becoming
less informative. `tools/latin_ablation.py`.

## The result

| variant | latin/100 | steps | NOP | NOP% | points | prose |
|---|---|---|---|---|---|---|
| op-msf-19-L00 | 4.38 | 8 | 1 | 12% | 9 | 258 |
| op-msf-19-L04 | 7.19 | 8 | 0 | 0% | 13 | 394 |
| op-msf-19-L09 | 8.25 | 8 | 2 | 25% | 9 | 368 |
| op-msf-19-L18 | **11.23** | 8 | 1 | 12% | 3 | 201 |
| op-msf-20-L00 | 4.51 | 6 | 0 | 0% | 4 | 244 |
| op-msf-20-L04 | 9.24 | 6 | 1 | 17% | 6 | 161 |
| op-msf-20-L09 | 10.75 | 6 | 1 | 17% | 7 | 252 |
| op-msf-20-L18 | **14.01** | 6 | 1 | 17% | 11 | 189 |

**No collapse at any level.** At 14.01% Latin — well *above* the podcasts' 10.30% — the
agent still produced 11 points and a 189-character summary, with a 17% NOP rate inside
the normal band. Not one variant hit the empty-memory fallback.

The hypothesis is refuted. Latin ratio is not the mechanism.

## What that leaves

By elimination, the podcast collapse is a **domain** effect:

- not translationese — the agent curates native zh-TW LY meetings (`RESULT.md`)
- not Latin/code-switching — this ablation, up to 14%
- not thin chunks, malformed ops or guard refusals — ruled out in `../phase3-pilot/RESULT.md`

What is left is that a tech podcast contains no **decisions, motions or commitments** —
the things `ADD`/`ARC` are trained to record. Some NOP there is genuinely *correct*.

**But 100% is still over-conservative, and worth keeping on the risk list.** The step
prompt asks for `關鍵論點、決議或承諾` — key *arguments*, decisions, or commitments — and a
17-minute technical discussion is full of key arguments. The baseline extracted 492
characters of them from the same input. So the model has narrowed "worth recording" to
roughly "formal proceedings", which is narrower than the prompt it was trained against.

## Caveat on the ablation itself

Substitution raises the Latin ratio but produces *lexical* code-switching in otherwise
well-formed sentences. Real ASR code-switching also brings mis-segmentation and
mistranscribed English, which this does not reproduce. So this refutes "high Latin ratio
alone breaks curation"; it does not fully clear ASR-induced language mixing.

**Risk 5 (the train/deploy ASR gap) therefore remains open and untested** — it needs
in-domain zh-TW *audio*, which is not obtainable here (IVOD is unreachable from this
host, and the only zh-TW ASR material available is the out-of-domain podcasts).
