#!/bin/bash
# Sets xblick-rtdetrv4 up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv), with the extra export (PyTorch for the CPU and what RT-DETRv4's
#   code imports, 1.5 GB), which makes the ONNX file
# * the model in ~/.cache/xblick/rtdetrv4 (the node's cache directory), through xblick-rtdetrv4 --download-only
#   (weights.py, export.py; only what is missing): RT-DETRv4-S by default, its checkpoint (169 MB) from the Google
#   Drive link of RT-DETRv4's README and RT-DETRv4's source (5 MB, a fixed commit) from GitHub, then the ONNX file
#   (42 MB) made from them on this PC. The node does the same at its start.
# * the GPU runtime for --device cuda (runtime.py), shared by the ONNX Runtime nodes, in
#   ~/.cache/xblick/onnxruntime/gpu-<version>-cp<Python>: onnxruntime-gpu with NVIDIA's CUDA 13 and cuDNN 9 libraries
#   from PyPI (2.4 GB), outside the venv, so uv run and uv sync leave it as it is. Without an option: installed when
#   an NVIDIA GPU is present (nvidia-smi lists one, its driver is 580 or newer), else one line says so. --gpu: always
#   (fails if it cannot be installed). --no-gpu: never. It is installed once; a second call, or another ONNX Runtime
#   node that installed it already, reuses it.
#
#     tools/setup.sh [--gpu | --no-gpu] [--model s|m|l|x]
#
# Needs uv (UV names it, default uv on PATH). Run again, it does only what changed. It never asks.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
mode=auto
args=()
for arg in "$@"; do
    case $arg in
        --gpu) [[ $mode != off ]] || { echo "--gpu and --no-gpu exclude each other" >&2; exit 2; }; mode=on ;;
        --no-gpu) [[ $mode != on ]] || { echo "--gpu and --no-gpu exclude each other" >&2; exit 2; }; mode=off ;;
        *) args+=("$arg") ;;
    esac
done

"$UV" sync --frozen --extra export
.venv/bin/xblick-rtdetrv4 --download-only "${args[@]}"
.venv/bin/python -m xblick_rtdetrv4.runtime setup --uv "$(command -v "$UV")" --prog xblick-rtdetrv4 --mode "$mode"
echo "xblick-rtdetrv4: set up"
