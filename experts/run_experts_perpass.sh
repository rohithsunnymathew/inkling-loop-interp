#!/bin/bash
# Same as run_experts.sh, but records each pass through the loop core separately.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/llama-build"
MODEL="${INKLING_GGUF:-$(ls "$HOME"/.cache/huggingface/hub/models--*Inkling-2-Mini*/snapshots/*/*.gguf 2>/dev/null | grep -v mmproj | head -1)}"
[ -f "$MODEL" ] || { echo "Set INKLING_GGUF=/path/to/Inkling-2-Mini-*.gguf"; exit 1; }
echo "== rebuilding llama-imatrix with the per-pass patch"
cmake --build "$BUILD" --target llama-imatrix -j > "$HERE/build_perpass.log" 2>&1 \
  || { echo "build failed, see build_perpass.log"; tail -20 "$HERE/build_perpass.log"; exit 1; }
mkdir -p "$HERE/imatrix_perpass"
for f in "$HERE"/corpus/*.txt; do
  n=$(basename "$f" .txt); echo "== $n"
  IMATRIX_PER_SLOT=1 "$BUILD/bin/llama-imatrix" -m "$MODEL" -f "$f" -o "$HERE/imatrix_perpass/$n.gguf" \
      -c 128 -ngl 99 --no-ppl --output-format gguf > "$HERE/imatrix_perpass/$n.log" 2>&1 \
    || { echo "failed on $n, see imatrix_perpass/$n.log"; tail -15 "$HERE/imatrix_perpass/$n.log"; exit 1; }
done
echo "Done. Tell Claude to analyse the per-pass run."
