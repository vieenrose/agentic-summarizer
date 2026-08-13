# Next steps (2026-08-13) — MiniCPM5-1B-CURSOR shipped

## Published: Luigi/minicpm5-1b-cursor (checkpoint-274, G1-verified)

## Remaining work
1. **SYNTH gate (tie → win)**: more real-meeting SUMMARY-arc data (the T2 tier when it
   exists) or a higher arc dose — diminishing returns observed (p10 +0.27, p11 +0.00).
2. **zh valid-op 100%**: stop the one redundant duplicate-ADD (a small "no duplicate
   re-ADD" negative set, or accept as documented).
3. **zh trap robustness**: the trap sits at the decision boundary between adjacent
   checkpoints — pick checkpoints by G1 screen, always.
4. **T2 tier (≥80k tokens)**: still needs real audio or concatenation decisions.
5. **Safetensors publish**: the training final (284) fails the zh trap — publish
   safetensors only after re-verifying a final, or retrain with save at the verified
   step.

## Hygiene
- After every GGUF export: verify the file exists AND the server PID changed.
- rm runs/*/checkpoint-* after each pass (disk fills).
- Judge ports: 8090 gpt-oss FAITH / 8091 qwen COVER (both must be up before any judge run).
