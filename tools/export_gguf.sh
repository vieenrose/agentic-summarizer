#!/usr/bin/env bash
# Export a fine-tuned Qwen3.5-0.8B checkpoint to Q8_0 GGUF for llama.cpp.
#
#   tools/export_gguf.sh runs/qwen-tools-v7/final runs/qwen-tools-v7/gguf
#
# **The MTP head IS dropped from intermediate checkpoints — CLAUDE.md was right, and my
# first correction to it on 2026-08-31 was wrong.** Measured properly on 2026-09-01:
#
#   base checkpoint          488 tensors, 15 mtp.*
#   runs/qwen-tools-v6/final 335 tensors, 15 mtp.*   <- save_model kept them
#   runs/qwen-tools-v7/*     320 tensors,  0 mtp.*   <- checkpoints AND final: dropped
#
# So whether the head survives depends on the save path, and it is not reliable. The first
# correction generalised from a single `v6/final` that happened to retain it.
#
# Restoring the head from base is CORRECT, not a fudge: v6's trained MTP tensors are
# bit-identical to base's, all 15/15, so training never touches them.
#
# Note the vision tower (153 `model.visual.*` tensors) IS dropped and should stay dropped —
# a text-only GGUF does not want it.

set -euo pipefail

SRC=${1:?usage: export_gguf.sh <checkpoint-dir> <out-dir>}
OUT=${2:?usage: export_gguf.sh <checkpoint-dir> <out-dir>}
LLAMA=${LLAMA_CPP:-$HOME/llama.cpp}
PY=${PYTHON:-.venv/bin/python}

mkdir -p "$OUT"

# Restore the MTP head if this save path dropped it. llama.cpp's converter requires the
# 15 `mtp.*` tensors; training never modifies them (verified 15/15 bit-identical to base),
# so taking them from the base checkpoint is exact, not an approximation.
BASE_SNAP=${BASE_SNAP:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/ | head -1)}
$PY - "$SRC" "$BASE_SNAP" <<'PYEOF'
import glob, os, sys
from safetensors import safe_open
from safetensors.torch import save_file

src, base = sys.argv[1], sys.argv[2]

def read(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                out[k] = s.get_tensor(k)
    return out

w = read(src)
have = [k for k in w if k.startswith("mtp.")]
print(f"[export] {len(w)} tensors, {len(have)} MTP")
if len(have) == 15:
    raise SystemExit(0)
mtp = {k: v for k, v in read(base).items() if k.startswith("mtp.")}
if len(mtp) != 15:
    raise SystemExit(f"[export] ABORT: base has {len(mtp)} mtp.* tensors, expected 15")
w.update(mtp)
for f in glob.glob(os.path.join(src, "*.safetensors")):
    os.remove(f)
idx = os.path.join(src, "model.safetensors.index.json")
if os.path.exists(idx):
    os.remove(idx)
save_file(w, os.path.join(src, "model.safetensors"), metadata={"format": "pt"})
print(f"[export] restored 15 MTP tensors from base -> {len(w)} total")
PYEOF

$PY "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outfile "$OUT/f16.gguf" --outtype f16
"$LLAMA/build/bin/llama-quantize" "$OUT/f16.gguf" "$OUT/Qwen3.5-0.8B.Q8_0.gguf" Q8_0
rm -f "$OUT/f16.gguf"   # ~1.5 GB; the Q8_0 is what is served (SPEC §4)
ls -la "$OUT/Qwen3.5-0.8B.Q8_0.gguf"
