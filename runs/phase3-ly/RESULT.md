# LY committee probe — the agent does NOT collapse on native zh-TW meetings

**The podcast collapse was domain/ASR, not translationese. Phase 4's premise survives.**

| slice | NOP rate | memory | summary |
|---|---|---|---|
| zh-TW podcasts (`runs/phase3-pilot`) | **8/8 = 100%** | empty | fallback constant |
| **LY committee meetings (here)** | **5/20 = 25%** | 9 / 4 / 3 points | 258 / 244 / 276 chars |
| (teacher's natural rate, for scale) | ~32–38% | — | — |

25% sits inside the normal band. The agent curates real Legislative Yuan committee
meetings essentially as it curates MeetingBank.

## Why this was the right control

The pilot confounded three variables. This slice removes two:

| | domain | text | language |
|---|---|---|---|
| podcasts | out-of-domain | ASR noise | native zh |
| **LY committee** | **in-domain meeting** | **clean** | **native zh** |
| MeetingBank (trained on) | in-domain | clean | **translated** |

Since the agent works here, the failure is **not** "the model only understands
translated council register." That was the reading that would have undermined Phase 4,
and it is now ruled out. What remains as the cause of the podcast collapse is domain
(podcasts have no decisions, motions or votes to record) and/or ASR noise — and for the
podcasts, some NOP is genuinely correct.

## Material

3 Open Parliament Committee (OP-MSF) meetings from the Legislative Yuan, 2023
(`https://www.ly.gov.tw/Pages/List.aspx?nodeid=43875`), converted by
`tools/ly_odt_to_v2.py`. Real, hard material: 30–136 utterances, **10–19 speakers**,
6–8 full chunks each. Verified through the real parser: 0 UNK, 0 v2 defects.

## The finding that is NOT good news: training-domain vocabulary leaks in

Terms that appear in the summary but **zero times in the source transcript**:

| meeting | hallucinated term | in source |
|---|---|---|
| op-msf-20 | **市議會** ("city council") | 0 |
| op-msf-21 | 決議案 ("resolution") | 0 |
| op-msf-19 | 公聽會 ("public hearing") | 0 |

`市議會` is the clearest: the Legislative Yuan is a national parliament and the word
never occurs in the transcript, but it appears in the summary — imported from
MeetingBank's American city-council register. This is SPEC §8 risk 1 (the corpus
teaches a different task) and risk 4 (translationese) showing up as measurable
confabulation, not as an abstract concern.

It is a faithfulness defect that the ROUGE gates cannot see, because it only shows up
against out-of-domain input, and every G3 number was measured in-domain.

## What this does and does not settle

**Settles:** the agent is not confined to translated council text. Phase 3 proper is
worth running, and Phase 4's "diagnosably data-volume-bound" justification is not
refuted by the pilot.

**Does not settle:**
- n=3, all from the same committee (OP-MSF) — one venue, one meeting culture.
- These are **stenographic transcripts, not ASR**. The train/deploy ASR gap (risk 5)
  remains entirely untested by this slice — it was the podcasts' other confound and is
  still open.
- No references, so this is reference-free only: no ROUGE, and the faithfulness judge
  was not run. The vocabulary leakage above was found by direct source-vs-summary term
  checks, which is weaker than a judge pass.

## Suggested next step

The remaining open variable is ASR noise on in-domain material. The cheapest decisive
test is the same LY meetings passed through ASR (or their original audio, if
obtainable) — holding domain and language fixed while varying only transcript quality.
That isolates risk 5, which is the one gap SPEC says nothing in the corpus can measure.
