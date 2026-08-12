#!/usr/bin/env bash
# Bring up the full serving stack after a restart (run from the repo root).
# Usage: tools/serve_stack.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA="$HOME/llama.cpp/build-aug/bin/llama-server"

echo "[stack] GPU 0: qwen3.6-35B COVER/SYNTH judge (8091)"
QWEN=~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
nohup env CUDA_VISIBLE_DEVICES=0 "$LLAMA" -m "$QWEN" --n-gpu-layers 999 --ctx-size 32768 --parallel 1 \
  --flash-attn on --jinja --reasoning off --temp 0 --host 127.0.0.1 --port 8091 > /tmp/judge_qwen.log 2>&1 &

echo "[stack] GPU 1: gpt-oss-20b FAITH judge (8090)"
GGUF=~/.cache/huggingface/hub/models--FreedomAISVR--gpt-oss-20B-NVFP4-GGUF/snapshots/6bdcb474af98b5e35260a92d8dc193c7f75ac18b/gpt-oss-20b-nvfp4.gguf
nohup env CUDA_VISIBLE_DEVICES=1 "$LLAMA" -m "$GGUF" --n-gpu-layers 999 --ctx-size 32768 --parallel 1 \
  --flash-attn on --jinja --reasoning off --temp 0 --host 127.0.0.1 --port 8090 > /tmp/judge_gptoss.log 2>&1 &

echo "[stack] GPU 1: en student (phase-3 candidate, 8093)"
nohup env CUDA_VISIBLE_DEVICES=1 "$LLAMA" -m runs/sft-lfm-en-p3/gguf_gguf/lfm2.5-350m-en-p3.Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 4096 --parallel 1 --flash-attn on --jinja --temp 0 \
  --host 127.0.0.1 --port 8093 > /tmp/student_en.log 2>&1 &

echo "[stack] GPU 1: zh student (v2, 8094)"
nohup env CUDA_VISIBLE_DEVICES=1 "$LLAMA" -m runs/sft-lfm-zh/gguf_gguf/lfm2.5-350m-zh.Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 4096 --parallel 1 --flash-attn on --jinja --temp 0 \
  --host 127.0.0.1 --port 8094 > /tmp/student_zh.log 2>&1 &

sleep 40
for p in 8090 8091 8093 8094; do
  echo -n "port $p: "; curl -s "http://127.0.0.1:$p/health" && echo
done
echo "[stack] next steps: see NEXT_STEPS.md"
