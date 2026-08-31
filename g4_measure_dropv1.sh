#!/usr/bin/env bash
# G4 measurement for sft-dropv1 on the reference device (Oppo Reno 7, CPH2371).
# Run this yourself via the `!` prefix (needs your terminal's tailnet access, not
# Claude's — the Bash tool's execution context cannot reach training-machine, see
# the network diagnosis earlier in this session).
#
# Reuses the EXACT same cross-compiled binaries and build flags as Phase 0b
# (2026-08-21): arm64-v8a, armv8.2-a+dotprod, no i8mm/SVE — confirmed via `file` +
# CMakeCache.txt inspection, not rebuilt. Adds ONE thing Phase 0b didn't have: a real
# all-cores depth sweep on the actual DEPLOY quant (Q8_0) and THIS checkpoint, rather
# than the Q4_0-derived extrapolation SPEC.md currently carries.
#
# Usage: ! bash g4_measure_dropv1.sh

set -euo pipefail

# `training-machine` doesn't resolve via OS DNS here (MagicDNS not configured in this
# resolver) -- resolve it through Tailscale's own CLI instead, which doesn't depend on
# OS DNS at all. Falls back to the IP confirmed earlier this session if `tailscale`
# itself is unavailable for some reason.
TS_HOST="$(tailscale ip -4 training-machine 2>/dev/null || echo 100.122.78.108)"
echo "resolved training-machine -> $TS_HOST"
# No explicit user: falls back to your local username (matches the original bare
# `training-machine` target) or whatever your own ~/.ssh/config already maps for this
# host/IP -- not guessing a remote username that might be wrong.
SSH_TARGET="$TS_HOST"

LOCAL_GGUF="runs/sft-dropv1/gguf_gguf/final.Q8_0.gguf"
LOCAL_BENCH="$HOME/llama.cpp/build-android/bin/llama-bench"
LOCAL_SERVER="$HOME/llama.cpp/build-android/bin/llama-server"
LOCAL_PAYLOAD="/tmp/g4_rss_payload.json"   # real ~2500-tok step prompt, pre-generated
REMOTE_STAGE="~/bench-dropv1"              # on training-machine
DEVICE_DIR="/data/local/tmp/bench-dropv1"
OUT_DIR="runs/phase0b-dropv1-$(date +%Y-%m-%d)"

for f in "$LOCAL_GGUF" "$LOCAL_BENCH" "$LOCAL_SERVER" "$LOCAL_PAYLOAD"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done

echo "=== 1/5: stage files on training-machine ==="
ssh $SSH_TARGET "mkdir -p $REMOTE_STAGE"
scp "$LOCAL_GGUF" "$LOCAL_BENCH" "$LOCAL_SERVER" "$LOCAL_PAYLOAD" $SSH_TARGET:$REMOTE_STAGE/

echo "=== 2/5: push to the phone over adb ==="
ssh $SSH_TARGET "
  set -e
  adb shell mkdir -p $DEVICE_DIR
  adb push $REMOTE_STAGE/final.Q8_0.gguf $DEVICE_DIR/minicpm5-1b-dropv1-q8_0.gguf
  adb push $REMOTE_STAGE/llama-bench $DEVICE_DIR/llama-bench
  adb push $REMOTE_STAGE/llama-server $DEVICE_DIR/llama-server
  adb push $REMOTE_STAGE/g4_rss_payload.json $DEVICE_DIR/rss_payload.json
  adb shell chmod +x $DEVICE_DIR/llama-bench $DEVICE_DIR/llama-server
  adb shell 'grep Features /proc/cpuinfo | head -1'
"
echo "  ^ confirm 'asimddp' present, 'i8mm' ABSENT before proceeding (Cortex-A78, no i8mm)."

echo "=== 3/5: all-cores depth sweep, Q8_0, THIS checkpoint (0xFF, depth 0 and ~4k) ==="
# -pg 2500,150 matches SPEC's own reading-step shape (2500 prefill, 150 decode).
# -d 0,1446 covers the reading-phase range Phase 0b's own trapezoidal estimate used;
# add ",5542" if you also want the ~8k point refreshed for this checkpoint.
ssh $SSH_TARGET "
  adb shell 'cd $DEVICE_DIR && ./llama-bench -m minicpm5-1b-dropv1-q8_0.gguf \
    -p 2500 -n 150 -pg 2500,150 -d 0,1446 -ub 128 -fa 1 -r 2 \
    -C 0xFF -t 8 -o jsonl > throughput_dropv1_q8_allcores.jsonl 2>&1'
  adb pull $DEVICE_DIR/throughput_dropv1_q8_allcores.jsonl $REMOTE_STAGE/
"

echo "=== 4/5: peak RSS, one REAL 2500-token completion (an actual training-pool step"
echo "    prompt, not synthetic filler), --no-mmap (honest private-storage number) ==="
# The prompt is a FILE on-device (rss_payload.json), never inlined through shell quoting
# -- multi-KB zh text surviving ssh -> adb shell -> bash -c quoting intact is not
# something to rely on, and a mis-quoted prompt would silently measure something other
# than a real step.
ssh $SSH_TARGET "
  adb shell '
    cd $DEVICE_DIR
    ./llama-server -m minicpm5-1b-dropv1-q8_0.gguf -c 4096 --no-mmap -C 0xFF -t 8 --port 8099 &
    SRV_PID=\$!
    sleep 8
    curl -s http://127.0.0.1:8099/completion -H \"Content-Type: application/json\" \
      -d @rss_payload.json > completion_out.json
    grep VmHWM /proc/\$SRV_PID/status
    awk \"/^Pss:/{print; exit}\" /proc/\$SRV_PID/smaps_rollup
    kill \$SRV_PID
  ' | tee $REMOTE_STAGE/rss_dropv1_q8.txt
"
echo "  Sanity-check completion_out.json on-device has real content before trusting the"
echo "  RSS number: ssh $SSH_TARGET \"adb shell 'cat $DEVICE_DIR/completion_out.json'\""

echo "=== 5/5: pull results back to this workstation ==="
mkdir -p "$OUT_DIR"
scp "$SSH_TARGET:$REMOTE_STAGE/throughput_dropv1_q8_allcores.jsonl" "$OUT_DIR/"
scp "$SSH_TARGET:$REMOTE_STAGE/rss_dropv1_q8.txt" "$OUT_DIR/"

echo
echo "Done. Results in $OUT_DIR/"
echo "Next: hand this back and I'll fold it into the real (not extrapolated) G4 number"
echo "via runs/phase0b-2026-08-21's same trapezoidal method, and arcsum-bench report."
