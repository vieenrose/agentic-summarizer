# NEXT STEPS — resume point (2026-08-12, after the pi-agent restart)

Everything needed to resume is in this repo; `RESULTS.md` has the full measured record.

## Running right now (nohup'd, survive the restart)

- GPU 1: gpt-oss-20b FAITH judge (8090), en student phase-3 candidate (8093),
  zh student v2 (8094)
- GPU 0: qwen3.6-35B COVER judge (8091) — **currently OFF** (killed for training)
- Restore everything with: `tools/serve_stack.sh`

## Where we are

- **Primary (LFM2.5-350M):** phase-3 (sweep-feedback negatives) raw T1 INVERT
  **4/20 → 3/20** (recorded in RESULTS.md). **G1 regression to fix**: deadlines anchored
  one line off ([5:00] vs 6:00) — the negatives perturbed anchor placement. Candidate
  fix: retrain phase-3 with negatives at ×1.5 instead of ×3, or a targeted
  deadline-anchor data add. The p3 GGUF is served on 8093.
- **Secondaries (all three trained, 6 epochs, checkpoints on disk — finals/GGUFs NOT yet
  saved):**
  - `runs/sft-qwen06` (agentlans/Qwen3-0.6B-notetaker base, checkpoint-240)
  - `runs/sft-lfm-12b` (LFM2.5-1.2B-Thinking, checkpoint-480)
  - `runs/sft-minicpm` (openbmb/MiniCPM5-1B, checkpoint-480)
  - Next: save finals + export Q4_K_M GGUFs → serve → G1 screen (en + zh) → raw T1
    INVERT for the best one → compare with the primary's 3/20 and the owner's 6.2% bar.
- **Owner (VoxSumDroid) thread:** `agentic-summarizer-feedback.md` — verdict open,
  narrowing; re-eval bar = raw INVERT < 6.2%; target metric = raw INVERT.

## Pipeline notes (so a fresh agent doesn't re-learn them)

- Trace data: `data/traces_v2/` (26 waves); SFT builders in `tools/` (split-filtered to
  train; eval tiers held out).
- Training: `train/sft_unsloth.py --resume <ckpt>` for phase-2/3 continuation; pre-tokenized
  dataset path (trl map pickling workaround); `UNSLOTH_COMPILE_DISABLE` NOT usable at 4096
  (OOM), torch 2.10 pinned (2.11 breaks unsloth's fused CE).
- The judge is stochastic on borderline inputs → always 3× majority (eval/judge.py and
  src/voxsum/sweep.py); warm the judge after any restart before runs; never mix instruments'
  JSONs in one report.
- Sweep (src/voxsum/sweep.py) is part of the pipeline: anchor-first, verify-last,
  FAITH-protocol parsing, FIX-as-DROP, full-text delete prefixes.
- HF: weights live on HF only (`Luigi/lfm2.5-350m-cursor-en/-zh`); the GitHub pre-push
  hook refuses weights; `.gitignore` blocks *.gguf/*.safetensors/checkpoints.
