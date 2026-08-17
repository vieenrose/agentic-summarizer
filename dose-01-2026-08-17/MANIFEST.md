# CURSOR training dose 01 — 2026-08-17

12 real-ASR transcripts, 202 min, ~40 CURSOR steps at CHUNK_TOKENS=2048.

Produced by the shipped VoxSum pipeline, so these are the exact shape the summarizer
sees in production.

## Please dose gradually

This model has moved in both directions on small doses: six samples cleared 3 of 4 flags
in one pass, the same six caused a regression at 350M, and p16's DECISIONS dose broke the
English chain. That is why this is 12 of a larger set rather than everything at once —
there is a second batch to correct with if this one regresses.

## Some transcripts are withheld, deliberately

A portion of this corpus is held back as an uncontaminated evaluation set and is not in
this archive. No judgement implied about your work — it is that every real transcript we
previously owned was either sent to you as training data or was already in the p15 set
(`meeting_zh_long.txt` is byte-identical to your repo-root copy, sha256 `7532e02b...8151`).
With nothing held back we cannot distinguish a checkpoint that improved from one that
memorised. We are not describing which files or what properties they have, because that
would let training drift toward them and destroy the thing that makes them useful.

## Contents

### Flip axis — mid/high Latin zh-TW

These carry the highest code-switch density in the corpus. The zh->English output flip
correlates with Latin share, so this is the group aimed at that failure.

| file | dur | utts | spk | Latin % | steps | source |
|---|---|---|---|---|---|---|
| q10.txt | 9m38s | 101 | 1 | 19.4% | 2 | [不住矽谷的台灣二寶爸工程師加拿大溫哥華育兒日記 aka 瘋人院院長](https://anchor.fm/s/23d304dc/podcast/play/86579311/https%3A%2F%2Fd3ctxlq1ktw2nl.cloudfront.net%2Fstaging%2F2024-4-10%2F377209433-44100-2-e9a08ac260ed4.m4a) |
| ho3.txt | 13m25s | 69 | — | 18.8% | 3 | [矽谷輕鬆談 Just Kidding Tech](https://anchor.fm/s/120c33e0/podcast/play/123919776/https%3A%2F%2Fd3ctxlq1ktw2nl.cloudfront.net%2Fstaging%2F2026-7-7%2F429426828-44100-2-454532ce2af49.mp3) |
| q04.txt | 17m12s | 359 | 2 | 17.3% | 4 | [DevMurmur | 讓我們聊聊軟體開發現場的那些事](https://anchor.fm/s/d2c5274/podcast/play/4664127/https%3A%2F%2Fd3ctxlq1ktw2nl.cloudfront.net%2Fstaging%2F2020-02-19%2Fa643a0ac6eadf960f75369b251949334.m4a) |
| ho18.txt | 7m28s | 126 | — | 12.1% | 2 | [NPDP產品經理國際認證問題集](https://rss.soundon.fm/rssf/86d857d1-56b7-4d42-821f-9406561bc2f9/feedurl/2bb3aee2-767f-481f-9b3b-2745069fd349/rssFileVip.mp3?timestamp=1655652987590) |
| q01.txt | 20m31s | 188 | 4 | 11.8% | 4 | [菜鳥的職涯筆記](https://rss.soundon.fm/rssf/36d92788-1be3-4b0c-b37e-2c8706d60fa8/feedurl/8978ebe7-fa35-4e8c-a793-a99198fe9f88/rssFileVip.mp3?timestamp=1776934820626) |
| q06.txt | 16m15s | 317 | 2 | 9.6% | 3 | [冒牌者症候群の下班時間](https://anchor.fm/s/15077e9c/podcast/play/15754360/https%3A%2F%2Fd3ctxlq1ktw2nl.cloudfront.net%2Fstaging%2F2020-06-26%2F97d71545e79eb9f7ab6f76881a3359d6.m4a) |

### ASR-noise robustness — dense low-Latin zh-TW

406-538 utterances each. Real X-ASR output fragments far more than the synthetic
transcripts the checkpoint was trained on; this is the fragmentation itself.

| file | dur | utts | spk | Latin % | steps | source |
|---|---|---|---|---|---|---|
| q08.txt | 22m54s | 516 | 2 | 2.2% | 4 | [職感生活](https://m.cdn.firstory.me/track/cksa76me8qqr70854mjgd8zpc/cl48hp4l7015q01zp2lkh5byr/https%3A%2F%2Fd3mww1g1pfq2pt.cloudfront.net%2FRecord%2Fcksa76me8qqr70854mjgd8zpc%2Fcl48hp4l7015r01zpaoa9gawf.mp3?v=1654868052645) |
| p09.txt | 15m47s | 426 | 2 | 1.6% | 3 | [MIT 作者對談](https://track.fstry.me/p/s4ajd3ty/rss.soundon.fm/rssf/1f3b5d9c-b2bd-48ca-9967-31eded24f1a9/feedurl/843c1f4d-8a68-49e7-b1ab-c64e039a4c8b/rssFileVip.mp3?timestamp=1745438176346) |
| p14.txt | 16m21s | 406 | 2 | 0.8% | 4 | [聽你在NNRR](https://m.cdn.firstory.me/track/ckc8z5u0j6ift0918kaxqntoq/cm3r3znai0fnb01uz1jrhhjas/https%3A%2F%2Fd3mww1g1pfq2pt.cloudfront.net%2FRecord%2Fckc8z5u0j6ift0918kaxqntoq%2Fcm3r3znai0fnc01uz0r641q5d.mp3?v=1732181260166) |
| p12.txt | 21m47s | 538 | 1 | 0.5% | 4 | [玫薇絲的管理碎碎唸](https://m.cdn.firstory.me/track/clsu37tno01dt01z93p0pad7c/cmso3gamw2lh101yq8km38zhk/https%3A%2F%2Ffile.cdn.firstory.me%2FRecord%2Fclsu37tno01dt01z93p0pad7c%2Fcmso3gamw2lh201yq07lj5q3q.mp3?v=1786418525737) |

### English regression guard

NOT here to improve English. A zh-only dose is the shape that broke the en chain on p16,
so English stays represented. Insurance, not a target.

| file | dur | utts | spk | Latin % | steps | source |
|---|---|---|---|---|---|---|
| p03.txt | 16m58s | 337 | 2 | 100.0% `[en]` | 3 | [DevOps Chat](https://feeds.soundcloud.com/stream/2038547401-devopschat-devops-thrives-and-software-supply-chain-is-the-sbom.mp3) |
| p02.txt | 22m10s | 404 | 1 | 100.0% `[en]` | 4 | [Adventures in DevOps](https://dts.podtrac.com/redirect.mp3/api.spreaker.com/download/episode/58841876/stream.mp3) |

Full per-episode titles and URLs in `sources.json`.

## Provenance

- **ASR**: X-ASR zipformer transducer zh-en on LiteRT (`xasr_q8_octav.tflite`) — the
  shipped default (`asrBackend = "x-asr"`). Not a cloud ASR, not a larger offline model:
  the noise here is the noise users actually get.
- **Script**: OpenCC to Traditional, applied at ASR-emit time as the app does. Every ASR
  model in the stack emits Simplified; zh-TW is the target.
- **Diarization**: pyannote segmentation -> CAM++ embeddings -> auto-k spectral clustering,
  where a speaker count is shown. Files with `—` were not diarized; transcript format v1
  permits a line with no speaker field, and that variant is also in production.
- **Format**: transcript format v1, `[m:ss] Sx: text`.
- Every mapping above was verified by matching measured audio duration to the source
  entry. One English episode was dropped from this dose because its source URL could not
  be verified, and a guessed URL is worse than one fewer file.

## Prompts (`prompts/`)

The bytes the harness actually sends, extracted from the deployed Kotlin:

- `sys-v1.en.txt` — 1254 chars, sha256 `0384503278de...`
- `sys-v1.zh-TW.txt` — 636 chars, sha256 `6da8383efa32...`
- `faith-judge.sys.txt` + `faith-judge.user-layout.txt` — the 350M verifier's contract

Per-step user content is `STATE:\n<rendered notes>\n\nCHUNK:\n<lines>`, in that fixed
order. STATE is rendered WITHOUT the promotion/chain guards — those run on the final
product render only, since the model was fine-tuned against un-promoted notes.

## Two things this dose cannot fix

Stated so it is not mistaken for a general quality batch:

1. **No decisions anywhere.** These are podcasts. Across all 45 transcripts we hold, the
   commitment lexicon returns only conversational hits — agreeing with an opinion,
   someone described as refusing a promotion, a historical cancellation recounted. Not one
   is a commitment that could be stated backwards. **An empty DECISIONS section is the
   correct output for every file here**, and training toward populated DECISIONS on this
   material would be training toward fabrication.

2. **The verifier's real-zh collapse is untouched.** granite-350m-verifier-zh answers
   SUPPORTED to fabricated bullets when given the harness's real 6-line noisy zh evidence
   window, while discriminating correctly on short clean lines. That is an evidence-window
   distribution problem, not something more student training addresses.

