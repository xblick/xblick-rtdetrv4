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
# Needs uv (UV names it, default uv on PATH) and git: without them it stops with the command that installs
# them. Run again, it does only what changed. It never asks and never uses sudo.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
PROG=xblick-rtdetrv4

usage() {
    cat <<'EOF'
Usage: tools/setup.sh [--gpu | --no-gpu] [--model NAME]

Sets xblick-rtdetrv4 up in this clone, or updates it. It creates .venv exactly as uv.lock fixes it (ONNX Runtime for
the CPU, the extra export, the core xblick from PyPI), prepares a model in the node's cache directory
(~/.cache/xblick/rtdetrv4) and, on a PC with an NVIDIA GPU, installs the GPU runtime. A second call does only what
changed. It never asks and never uses sudo. Needs git and uv (UV=/path/to/uv names another uv). An NVIDIA GPU is
optional.

  (no option)   the GPU runtime when nvidia-smi lists a GPU with driver 580 or newer
  --gpu         the GPU runtime whatever the check says (an image built on a PC without a GPU)
  --no-gpu      no GPU runtime
  --model NAME  the model to prepare: s (default), m, l, x
  -h, --help    this text
EOF
}

die() {
    echo "setup.sh: $*" >&2
    exit 1
}

for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; esac
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] ||
    die "the node needs Linux on x86-64 (the core's wheels), this is $(uname -s) $(uname -m)"
command -v git > /dev/null || die "no git: sudo apt install git"
command -v "$UV" > /dev/null ||
    die "no uv ($UV): install it with  wget -qO- https://astral.sh/uv/install.sh | sh  and open a new terminal"
mode=auto
args=()
for arg in "$@"; do
    case $arg in
        --gpu) [[ $mode != off ]] || { echo "--gpu and --no-gpu exclude each other" >&2; exit 2; }; mode=on ;;
        --no-gpu) [[ $mode != on ]] || { echo "--gpu and --no-gpu exclude each other" >&2; exit 2; }; mode=off ;;
        *) args+=("$arg") ;;
    esac
done

cat <<EOF
$PROG: setting up $PWD. It downloads what is missing of:
  * .venv as uv.lock fixes it: Python, the core xblick, ONNX Runtime for the CPU and the extra export (PyTorch for
    the CPU and what RT-DETRv4's code imports); about 1.3 GB
  * the model into the node's cache directory, after its terms (default RT-DETRv4-S: its checkpoint, 169 MB, from
    RT-DETRv4's Google Drive link, and RT-DETRv4's code, 5 MB), then the ONNX file made from them on this PC
  * on a PC with an NVIDIA GPU (driver 580 or newer): the GPU runtime, onnxruntime-gpu with NVIDIA's CUDA 13 and
    cuDNN 9 libraries (2.4 GB, NVIDIA's licence), shared by the ONNX Runtime nodes
EOF
"$UV" sync --frozen --extra export
.venv/bin/xblick-rtdetrv4 --download-only "${args[@]}"
.venv/bin/python -m xblick_rtdetrv4.runtime setup --uv "$(command -v "$UV")" --prog xblick-rtdetrv4 --mode "$mode"
echo "$PROG: set up. The commands are in .venv/bin/; the first start: README.md, Install."
