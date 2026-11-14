#!/bin/bash
# Sets xblick-rtdetrv4 up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv), with the extra export (PyTorch for the CPU and what RT-DETRv4's
#   code imports, 1.5 GB), which makes the ONNX file
# * the model in ~/.cache/xblick/rtdetrv4 (the node's cache directory), through xblick-rtdetrv4 --download-only
#   (weights.py, export.py; only what is missing): RT-DETRv4-S by default, its checkpoint (169 MB) from the Google
#   Drive link of RT-DETRv4's README and RT-DETRv4's source (5 MB, a fixed commit) from GitHub, then the ONNX file
#   (42 MB) made from them on this PC. The node does the same at its start.
#
#     tools/setup.sh [--model s|m|l|x]
#
# Needs uv (UV names it, default uv on PATH). Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }

"$UV" sync --frozen --extra export
.venv/bin/xblick-rtdetrv4 --download-only "$@"
echo "xblick-rtdetrv4: set up"
