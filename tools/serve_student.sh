#!/usr/bin/env bash
# Serve the fine-tuned student (GGUF) for the G1 screen and eval arms.
#
#   tools/serve_student.sh runs/sft-v1/gguf/model-Q4_K_M.gguf [port]
set -euo pipefail
GGUF="${1:?usage: serve_student.sh <model.gguf> [port]}"
PORT="${2:-8092}"
LLAMA="$HOME/llama.cpp/build-aug/bin/llama-server"
[ -f "$GGUF" ] || { echo "model not found: $GGUF" >&2; exit 1; }
# The student serves on the GPU the teachers no longer need; 4k ctx matches its budget
# (spec §8); greedy for eval reproducibility; the screen must run WITHOUT a grammar and
# the arms run with the same surface, so no grammar flag here.
exec env CUDA_VISIBLE_DEVICES=0 "$LLAMA" -m "$GGUF" \
  --n-gpu-layers 999 --ctx-size 4096 --parallel 1 --flash-attn on --jinja \
  --temp 0 --host 127.0.0.1 --port "$PORT"
