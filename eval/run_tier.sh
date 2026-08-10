#!/usr/bin/env bash
# Run one eval tier end-to-end against a served student: arms -> judge -> report.
#
#   STUDENT_URL=http://127.0.0.1:8092 FAITH=local:8090/gpt-oss-20b \
#   COVER=local:8091/qwen3.6-27b ./eval/run_tier.sh micro
#
# Environment:
#   STUDENT_URL  llama-server endpoint running the (fine-tuned) student
#   FAITH        FAITH/INVERT judge (local:PORT/NAME or cloud model id)
#   COVER        COVER/SYNTH judge
#   SECOND       optional second-opinion judge (unset => none)
#   BUDGET       chunk token budget (default 2048; micro uses 2048 for GT4 comparability)
set -euo pipefail
cd "$(dirname "$0")/.."
TIER="${1:?usage: run_tier.sh micro|t1}"
STUDENT_URL="${STUDENT_URL:-http://127.0.0.1:8092}"
FAITH="${FAITH:-local:8090/gpt-oss-20b}"
COVER="${COVER:-}"
BUDGET="${BUDGET:-2048}"

[ -n "$COVER" ] || { echo "COVER judge required (local:PORT/NAME)"; exit 2; }

VENV=.venv/bin/python
OUT=runs/"$TIER"
mkdir -p "$OUT/arms" "$OUT/judged"

# 1. Meeting list for this tier, from the manifest.
mapfile -t MEETINGS < <("$VENV" - <<'PY'
import json, sys
rows = json.load(open("data/transcripts/manifest.json"))
for r in rows:
    if r["split"] == sys.argv[1]:
        print(r["meeting_id"], r["lang"])
PY
"$TIER")

echo "[tier] $TIER: ${#MEETINGS[@]} meetings"

# 2. Arms: cursor (student) + baseline (same student model), paired per meeting.
EN=(); ZH=()
for pair in "${MEETINGS[@]}"; do
  mid=${pair%% *}; lang=${pair##* }
  if [ "$lang" = en ]; then EN+=("data/transcripts/$mid.txt"); else ZH+=("data/transcripts/$mid.txt"); fi
done
[ ${#EN[@]} -gt 0 ] && "$VENV" eval/run_arms.py "${EN[@]}" --out "$OUT/arms" --lang en \
  --base-url "$STUDENT_URL" --declarations --budget "$BUDGET"
[ ${#ZH[@]} -gt 0 ] && "$VENV" eval/run_arms.py "${ZH[@]}" --out "$OUT/arms" --lang zh-TW \
  --base-url "$STUDENT_URL" --declarations --budget "$BUDGET"

# 3. Judge every (meeting, arm) notes file.
ARGS=(--faith-model "$FAITH" --cover-model "$COVER")
[ -n "${SECOND:-}" ] && ARGS+=(--second-model "$SECOND") || ARGS+=(--no-second-opinion)
for pair in "${MEETINGS[@]}"; do
  mid=${pair%% *}; lang=${pair##* }
  for arm in cursor baseline; do
    notes="$OUT/arms/$mid.$arm.notes.txt"
    [ -f "$notes" ] || { echo "[tier] missing $notes"; continue; }
    "$VENV" eval/judge.py --notes "$notes" --transcript "data/transcripts/$mid.txt" \
      --system "$arm" --meeting-id "$mid" --out "$OUT/judged/$mid.$arm.json" \
      --budget-usd 5 "${ARGS[@]}" > /dev/null
    echo "[tier] judged $mid/$arm"
  done
done

# 4. Paired report with GT4 from the same run.
"$VENV" eval/report.py "$OUT"/judged/*.json --usage "$OUT/arms/usage.json" \
  --min-n "$( [ "$TIER" = micro ] && echo 6 || echo 20 )" --out "$OUT/report.json" \
  | tee "$OUT/report.txt"
echo "[tier] done: $OUT/report.txt"
