#!/usr/bin/env bash
# Export a fine-tuned Qwen3.5-0.8B checkpoint to Q8_0 GGUF for llama.cpp.
#
#   tools/export_gguf.sh runs/qwen-tools-v7/final runs/qwen-tools-v7/gguf
#
# **No tensor surgery is required, and CLAUDE.md said otherwise until 2026-08-31.**
# That note read: "it carries an MTP head at block 24 (mtp_num_hidden_layers: 1, 15
# tensors) that a text-tower fine-tune drops ... Copy them back from the base checkpoint
# before converting." Verified against `runs/qwen-tools-v6/final`:
#
#   base checkpoint   488 tensors, 15 named mtp.*
#   fine-tuned final  335 tensors, 15 named mtp.*   <- MTP IS PRESERVED
#   difference        153 tensors, ALL model.visual.*  (the vision tower)
#
# `tools/train_toolcalls.py` saves the MTP head; what it drops is the vision tower, which
# the text-only GGUF does not want. Converting `final` directly succeeds and reproduces
# the shipped v6 GGUF byte-for-byte in size (833,591,584 B). Copying MTP tensors "back"
# would have been a no-op at best.
#
# The claim was probably true for the `AutoModelForCausalLM` path that unsloth forced
# (see CLAUDE.md's Qwen3.5 integration notes) and did not survive the move to
# `train_toolcalls.py`. Left here rather than deleted because a future session hitting a
# converter assert should know which of the two paths it is on.
set -euo pipefail

SRC=${1:?usage: export_gguf.sh <checkpoint-dir> <out-dir>}
OUT=${2:?usage: export_gguf.sh <checkpoint-dir> <out-dir>}
LLAMA=${LLAMA_CPP:-$HOME/llama.cpp}
PY=${PYTHON:-.venv/bin/python}

mkdir -p "$OUT"

# Fail loudly if the MTP head is absent: llama.cpp's converter asserts on it, and a
# missing head means the training path changed and this script's assumption no longer
# holds. Better to stop here than to produce a GGUF that will not load.
$PY - "$SRC" <<'PYEOF'
import glob, os, sys
from safetensors import safe_open
src = sys.argv[1]
names = set()
for f in glob.glob(os.path.join(src, "*.safetensors")):
    with safe_open(f, framework="pt") as s:
        names |= set(s.keys())
mtp = [n for n in names if n.startswith("mtp.")]
print(f"[export] {len(names)} tensors, {len(mtp)} MTP")
if len(mtp) != 15:
    raise SystemExit(
        f"[export] ABORT: expected 15 mtp.* tensors, found {len(mtp)}. The training path "
        "changed; see this script's header before working around it."
    )
PYEOF

$PY "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outfile "$OUT/f16.gguf" --outtype f16
"$LLAMA/build/bin/llama-quantize" "$OUT/f16.gguf" "$OUT/Qwen3.5-0.8B.Q8_0.gguf" Q8_0
rm -f "$OUT/f16.gguf"   # ~1.5 GB; the Q8_0 is what is served (SPEC §4)
ls -la "$OUT/Qwen3.5-0.8B.Q8_0.gguf"
