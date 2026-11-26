#!/bin/bash
# Logs which MoE experts Inkling-2-Mini selects for each text, per layer.
# Builds llama-imatrix into ./llama-build (Unsloth's install is not touched).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${LLAMA_CPP_DIR:-$HOME/.unsloth/llama.cpp}"   # llama.cpp source with inkling2_loop support
BUILD="$HERE/llama-build"
MODEL="${INKLING_GGUF:-$(ls "$HOME"/.cache/huggingface/hub/models--*Inkling-2-Mini*/snapshots/*/*.gguf 2>/dev/null | grep -v mmproj | head -1)}"
[ -f "$MODEL" ] || { echo "Set INKLING_GGUF=/path/to/Inkling-2-Mini-*.gguf"; exit 1; }

command -v cmake >/dev/null || { echo "cmake not found. Install it with:  brew install cmake"; exit 1; }
if [ ! -x "$BUILD/bin/llama-imatrix" ]; then
  echo "== building llama-imatrix (a few minutes, once)"
  cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$HERE/build.log" 2>&1
  cmake --build "$BUILD" --target llama-imatrix -j >> "$HERE/build.log" 2>&1 \
    || { echo "build failed, see build.log"; tail -20 "$HERE/build.log"; exit 1; }
fi

mkdir -p "$HERE/imatrix"
for f in "$HERE"/corpus/*.txt; do
  n=$(basename "$f" .txt)
  echo "== $n"
  "$BUILD/bin/llama-imatrix" -m "$MODEL" -f "$f" -o "$HERE/imatrix/$n.gguf" \
      -c 128 -ngl 99 --no-ppl --output-format gguf > "$HERE/imatrix/$n.log" 2>&1 \
    || { echo "failed on $n, see imatrix/$n.log"; tail -15 "$HERE/imatrix/$n.log"; exit 1; }
done
echo "Done. Tell Claude to analyse."
