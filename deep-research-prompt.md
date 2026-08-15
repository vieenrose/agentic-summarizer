# Deep-research prompt — thinking-enabled agentic summarization at sub-1B

(Copy into Google Deep Research. Replace bracketed items as needed.)

## Goal being researched

We are building a production agentic meeting-transcript summarizer that runs
**on-device with a single sub-1B model**: MiniCPM5-1B (a hybrid-reasoning model —
200B tokens of deep-thinking SFT + 200B of hybrid-thinking SFT in its base),
**thinking mode ENABLED**, context ≤ 8192 tokens, streaming edit operations over one
evolving notes state (the CURSOR protocol — no map-reduce, no free ReAct loops).
Inputs: ASR-noisy meeting transcripts in zh-TW and English; outputs: structured,
fully anchored meeting notes. The fine-tune teacher is Qwen3.8-27B (thinking
enabled) generating per-chunk reasoning-plus-operations traces at the same context
size.

## The five research questions (prioritized)

### 1. Teaching bounded reasoning to small models (highest priority)
We need the 1B student to reason *briefly* (a few hundred tokens per step) so the
reasoning + the ~2.9k-token prompt + the operations fit 8k context. The teacher's
reasoning is unbounded (8k–19k chars per step at temp 0.5–1.0).
- What methods control reasoning length in fine-tuned LLMs (budgeted CoT,
  think-token budgets, reasoning-length regularization, "reason briefly" prompt
  engineering, sampling constraints)?
- How does reasoning length trade against quality at ≤1B scale? Is there a measured
  elbow where shorter reasoning stops hurting?
- Any work on distilling long teacher reasoning into short student reasoning
  (reasoning summarization/compression before SFT)?

### 2. Reasoning distillation from 27B → 1B
We SFT the student on (teacher reasoning + ops) targets. Measured context: our
earlier op-only SFT (no reasoning targets) left the student's think-mode untrained
in-role; the base's own reasoning is strong but role-naive.
- What does the literature say about transferring reasoning ability at 20-30× size
  gaps? Which target formats work (full reasoning, truncated, rationale-annotated)?
- Is reasoning in the fine-tune targets needed at all for small models to *use*
  thinking at inference, or does the base's hybrid-thinking prior suffice with
  role-specific SFT?
- Quiet-STAR / self-distillation / SCOTT-style findings relevant here?

### 3. Hybrid-thinking models: fine-tuning the think mode
MiniCPM5-class models serve both think and no-think modes from one checkpoint via an
`enable_thinking` template flag.
- How does fine-tuning interact with the two modes — does SFT on think-mode targets
  degrade the no-think mode (and vice versa)? Is mode-specific SFT necessary?
- Recommended inference sampling per mode (we measured temp 0.5/top-k 40/top-p 0.9
  tames the teacher's reasoning length; the student serves greedy temp 0).

### 4. Streaming agentic loops with reasoning at small scale
Our loop: ~40 chunks per 80k-token meeting; per-step prompt ~2.9k tokens; the
reasoning adds decode tokens (the expensive currency on-device: ~20-25 tok/s).
- Any measurements of reasoning overhead in long-horizon agentic/streaming tasks
  at sub-2B? Does thinking help faithfulness/coverage in long-document agents, or
  is it mostly wasted on short per-chunk decisions?
- Context-length behavior of sub-2B models with reasoning at 8k (RULER/SlimLM-type
  degradation curves).

### 5. Noisy-ASR robustness and agentic summarization quality
The coverage blocker we measured: the student's op emission collapses on real
ASR-noisy zh-TW transcripts (echo-loops, disfluency, homophone errors) even for
meetings in its training set. Zero decision-section ops on a decision-dense hour.
- Dialogue-to-description normalization (NexusSum-style) and other transcript
  preprocessing that small models learn well.
- Does reasoning help small models read through ASR noise, or is noise-robustness a
  data problem only?
- Any agentic summarization papers measuring decision/action extraction fidelity
  with thinking enabled vs disabled.

## Constraints to respect in recommendations

- On-device: single model ~700MB Q4, ≤8k ctx, decode ~20-25 tok/s, thermals matter
  (sustained load regime).
- Byte-stable protocol prompts (the system prompt and state rendering must not
  change — they define the fine-tune/eval distribution).
- Every claim in the notes must be anchored to a transcript line; 0% inversions
  (notes never state the opposite of the transcript).
- Languages: zh-TW (primary) and English.

## Deliverable format

For each question: (a) the named methods/papers with one-line claims, (b) which are
practical at ≤1B on-device, (c) concrete experiment recipes we can run (data
formats, target constructions, sampling, loss/dose choices), (d) what to measure.
Finish with a ranked list of the 5 highest-value experiments for our exact setting.
