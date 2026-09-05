#!/usr/bin/env bash
# Serve one GGUF and score it on the held-out 40, with the server identity VERIFIED.
#
#   tools/eval_checkpoint.sh runs/raft-s0/gguf_best/model.Q8_0.gguf raft-s0-e2 8091
#
# Exists because every step below has silently produced a wrong number in this project:
#
# * **The server can answer as the PREVIOUS model.** Killing the old `llama-server` and
#   starting the new one in one command raced: the kill had not completed, the new server
#   exited on a port conflict, and the old one served the "new" checkpoint's curve. Caught
#   only because two checkpoints returned BYTE-IDENTICAL numbers. So `/props` `model_path` is
#   checked against the requested file before any measurement runs, and a mismatch aborts.
# * **`pkill -f <pattern>` matches this script's own command line** and has killed the session
#   three times (exit 144). `pgrep -x llama-server` plus a `/proc/<pid>/cmdline` filter is an
#   exact process-NAME match, which cannot match a shell.
# * **`--no-jinja` is required for the student** (trap 10). Its GGUF template unconditionally
#   opens `<think>\n` while `train_toolcalls.py` trained on plain ChatML, so serving under
#   jinja prefixes every prompt with tokens the fine-tune never saw. `cli.run_arms` renders
#   through `/apply-template`, so this would confound every reported number, not just one.
# * **`-np N` divides `-c` among slots.** A run with `-np 4 -c 8192` gives each slot 2,048
#   tokens, which silently failed 5,400 of 5,772 samples. Context is set explicitly here.
set -euo pipefail

GGUF="${1:?usage: eval_checkpoint.sh <gguf> <label> [port]}"
LABEL="${2:?}"
PORT="${3:-8091}"
CORPUS="${CORPUS:-data/heldout_zh}"
OUT="${OUT:-runs/${LABEL}/scorecard_heldout.json}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

[ -f "$GGUF" ] || { echo "no such gguf: $GGUF" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# Reap only OUR server on OUR port. Exact name match; never `pkill -f`.
for pid in $(pgrep -x llama-server 2>/dev/null || true); do
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--port $PORT"; then
    echo "[eval] stopping stale server on :$PORT (pid $pid)"
    kill "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do sleep 1; done
  fi
done

echo "[eval] serving $GGUF on :$PORT"
~/llama.cpp/build/bin/llama-server -m "$GGUF" --port "$PORT" -c 8192 -ngl 99 \
  --no-jinja > "runs/${LABEL}/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

until curl -sf "http://127.0.0.1:$PORT/props" >/dev/null 2>&1; do
  kill -0 $SERVER 2>/dev/null || { echo "[eval] server died; see runs/${LABEL}/server.log" >&2; exit 1; }
  sleep 2
done

SERVED=$(curl -s "http://127.0.0.1:$PORT/props" \
  | .venv/bin/python -c 'import sys,json;print(json.load(sys.stdin)["model_path"])')
case "$SERVED" in
  *"$(basename "$(dirname "$GGUF")")/$(basename "$GGUF")"*|*"$GGUF"*) ;;
  *) echo "[eval] REFUSING: server is serving '$SERVED', not '$GGUF'" >&2; exit 1 ;;
esac
echo "[eval] verified served model: $SERVED"

PYTHONPATH=src .venv/bin/python -m arcsum.cli.eval_all \
  --url "http://127.0.0.1:$PORT" --protocol tool \
  --corpus "$CORPUS" --label "$LABEL" \
  --cache-prompt true --deployed-cache-prompt true \
  --out "$OUT"
echo "[eval] wrote $OUT"
