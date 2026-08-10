#!/usr/bin/env bash
# Serve TWO independent teachers, one per GPU (PLAN.md §2b).
#
#   tools/serve_teacher_dual.sh          # QAT q4_0, ports 8080 and 8081
#
# Why two servers rather than one split across both cards:
#
#   * The Q8_0 teacher is 33 GB and does not fit one 32 GB card, so it is layer-split with
#     activations crossing PCIe every layer — there is no NVLink on 5090s. The QAT q4_0 build
#     is 17.7 GB and fits one card whole, so that traffic disappears entirely.
#   * Two whole-model servers give real parallelism across meetings, which a single split
#     server cannot: trace generation is embarrassingly parallel per meeting.
#   * Q4 halves weight-memory bandwidth per token against Q8.
#
# QAT (quantization-aware training) is the part that makes 4-bit acceptable here: the model is
# trained to be quantized rather than rounded afterwards, so quality tracks the unquantized
# model far more closely than a post-training Q4 would. It must still pass the G1 screen on
# BOTH languages before being trusted as a teacher — see RESULTS.md.
set -euo pipefail

MODEL="${MODEL:-$(find "$HOME/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-q4_0-gguf" -name '*q4_0-it.gguf' | head -1)}"
CTX="${CTX:-16384}"
LLAMA="${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}"
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "model not found; set MODEL=" >&2; exit 1; }

for gpu in 0 1; do
  port=$((8080 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$LLAMA" \
    -m "$MODEL" \
    --n-gpu-layers 999 \
    --ctx-size "$CTX" \
    --parallel 2 \
    --flash-attn on \
    --jinja \
    --temp 1.0 --top-k 64 --top-p 0.95 \
    --host 127.0.0.1 --port "$port" \
    > "/tmp/teacher_gpu${gpu}.log" 2>&1 &
  echo "[serve] GPU $gpu -> port $port (pid $!)"
done

echo "[serve] waiting for both to come up..."
for port in 8080 8081; do
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "[serve] port $port ready"; break; }
    sleep 5
  done
done
wait
