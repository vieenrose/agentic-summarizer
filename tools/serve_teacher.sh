#!/usr/bin/env bash
# Serve the teacher across both RTX 5090s (PLAN.md §2b).
#
#   tools/serve_teacher.sh            # Q8_0, layer-split over both GPUs
#   QUANT=UD-Q4_K_XL tools/serve_teacher.sh   # single-card Q4, frees GPU 1
#
# --ctx-size is deliberately the *student's* per-step budget, not the teacher's capacity:
# per PLAN.md §2c the teacher must not see more than the student will, and capping the
# served window makes an over-budget prompt fail loudly instead of quietly succeeding.
set -euo pipefail

REPO="${REPO:-$HOME/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF}"
QUANT="${QUANT:-Q8_0}"
# --ctx-size bounds prompt + output *together*, and --parallel N divides it per slot. It is
# therefore not the enforcement of the §2c prompt budget — agent.py's client-side assertion
# is, and it stays in force regardless of what is set here. Gemma 4 is a thinking model:
# reasoning alone can run past 1.5k tokens, so a 4096 ctx over 2 slots starves it and the
# step returns reasoning with no ops. Use CTX=16384 for thinking runs.
CTX="${CTX:-4096}"
PORT="${PORT:-8080}"
LLAMA="${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}"

model=$(find "$REPO" -name "gemma-4-31B-it-${QUANT}.gguf" | head -1)
mtp=$(find "$REPO" -path '*/MTP/*' -name "mtp-gemma-4-31B-it-*.gguf" | head -1)
[ -n "$model" ] || { echo "no ${QUANT} gguf under $REPO" >&2; exit 1; }

args=(
  -m "$model"
  --n-gpu-layers 999
  --ctx-size "$CTX"
  --parallel 2
  --flash-attn on
  --jinja                 # correct chat template; without it Gemma templating is mishandled
  --temp 1.0 --top-k 64 --top-p 0.95   # Gemma 3/4 recommended sampling
  --host 127.0.0.1 --port "$PORT"
)

# Q8_0 is 33 GB and does not fit one 32 GB card: split layers across both. Layer split
# beats row split here — there is no NVLink on 5090s, so row split would push activations
# over PCIe every layer.
if [ "$QUANT" = "Q8_0" ]; then
  args+=(--split-mode layer --tensor-split 1,1)
else
  args+=(--split-mode none --main-gpu 0)
fi

# Speculative decode via the repo's multi-token-prediction head. Opt-in (MTP=1): the head
# carries architecture `gemma4-assistant`, which this llama.cpp build rejects as a draft
# model — enable only once your build supports it, or the server refuses to start.
if [ "${MTP:-0}" = "1" ] && [ -n "$mtp" ]; then
  args+=(-md "$mtp")
fi

echo "[serve] ${QUANT} $(basename "$model") ctx=${CTX} port=${PORT}"
exec "$LLAMA" "${args[@]}"
