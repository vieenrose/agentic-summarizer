# Real zh-TW ASR transcripts for the CURSOR eval tier — 2026-08-16

Three public podcast episodes, transcribed by the shipped VoxSum Android pipeline. **Public
sources, no privacy constraint — usable as training data and as an eval tier.**

Each directory contains:

| file | what it is |
|---|---|
| `source.txt` | series, episode, duration, **audio URL** |
| `transcript.txt` | ASR output in **transcript format v1** (see below) |
| `notes-p13-android.txt` | notes produced on-device (p13 + granite-zh, OPPO CPH2371) |
| `notes-p15d-desktop.txt` | notes from the SAME transcript on x86 (p15d + granite-zh) |
| `stats-p15d.txt` | ops emitted/applied/vetoed/malformed, anchors, inversion audit |

Both checkpoints ran the identical Kotlin port of the harness at `sys-v1`, chunk 2048, ctx 4096,
greedy. ASR is `sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03`.

## Format

Transcript format v1, validated against the normative rules before packaging — **0 violations**
across all three: split on the first `] ` then the first `: ` within 40 chars; `M:SS` under an
hour and `H:MM:SS` from one hour, seconds zero-padded and the leading unit unpadded; one utterance
per line; no header, markdown, CRLF or embedded newlines; timestamps monotonic. They are the
deployed parser's own round-trip, so they are byte-exactly what the summarizer consumes.

Speakers are `S1`/`S2` by first appearance. Two episodes are solo shows (S1 only); that is real,
not a diarization failure.

## What these show

| episode | Latin % | output language | SUM | DEC | ACT | OPEN | TOP | anchors |
|---|---|---|---|---|---|---|---|---|
| 01 cerebras 16m | **22.3%** | **English** X | 1 | 0 | 0 | 1 | 1 | 3/3 |
| 02 materials 17m | 4.9% | Chinese OK | **0** | 0 | 2 | 1 | 3 | 6/6 |
| 03 tsmc-wind 10m | 3.5% | Chinese OK | 2 | 0 | 1 | 0 | 1 | 4/4 |

**Start with 01.** The transcript is 78% Han — unambiguously Chinese — and the harness selects the
zh protocol prompt correctly. Both p13 and p15d answer in **English**, on both ARM and x86. Its
bullets are also ungrounded (<=1 shared token with any line within +/-90s of their own anchor):
"sweepers" appears nowhere in the episode, and "Cerebrus" misspells Cerebras throughout including
the title. p15d additionally emitted **3 of 10 malformed ops** here — every clean-tier run of ours
was 0.

The other two stay in Chinese and are fully grounded, which is what makes 01 informative: the
variable that tracks the flip is the share of English technical vocabulary in the transcript.

Episode 03 shows a second effect: the ASR produced both 離岸風電 and the garbled 離岸封建
("offshore feudalism"); the model copied the garbled form into the summary, on both checkpoints.

DECISIONS is empty everywhere and that is **correct** — podcasts contain no decisions. These
episodes say nothing about decision extraction.

## Two more available

Two further episodes (23m and 24m) were recorded but exceeded our 30-minute processing cap, so
they have audio but no transcript yet. Say the word and we will finish them.
