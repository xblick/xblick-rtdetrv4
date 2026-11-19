# Changelog

## 0.1.3 (2026-11-19)

An installation that a beginner can follow from the README alone.

### Added

- `xblick-rtdetrv4-gpu-runtime install`: the GPU runtime for `--device cuda` after `uv tool install`, which has no
  `tools/setup.sh`. `xblick-rtdetrv4-gpu-runtime where` prints its directory.

### Changed

- The README's installation is a numbered path: the prerequisites with their install commands, the node with
  `uv tool install`, the core's command with one line for every node
  (`uv tool install "xblick[rerun,video,cam]>=0.5,<0.6"`), and a first start. The clone with `tools/setup.sh` stays
  the second way.
- `tools/setup.sh --help` describes the script and its options and ends. Before, it ran the setup.
- `tools/setup.sh` checks its prerequisites first. Without git or uv it stops and prints the command that installs
  them. On a PC other than Linux on x86-64, for which the core has no wheels, it stops at once.
- Before its downloads, `tools/setup.sh` lists what it downloads, with the sizes and the licences.

### Fixed

- Without the GPU runtime, `--device cuda` named only `tools/setup.sh --gpu`, which an install with uv does not have.
  It now names `xblick-rtdetrv4-gpu-runtime install`, and `tools/setup.sh --gpu` for a clone.
## 0.1.2 (2026-11-15)

The MPS check of the other nodes, and how to cite RT-DETRv4.

### Changed

- README: how to cite RT-DETRv4.

### Fixed

- With `--device cuda` and `CUDA_MPS_PIPE_DIRECTORY` set, the node stops at the start when no MPS daemon runs.

Compatible with earlier clients: nothing on the wire changed.

## 0.1.1 (2026-11-15)

- The GPU runtime for `--device cuda` is shared by the ONNX Runtime nodes: `onnxruntime-gpu` 1.32.0 with NVIDIA's
  CUDA 13 and cuDNN 9 libraries in `~/.cache/xblick/onnxruntime/gpu-1.32.0-cp<Python>/` (`xblick.paths.cache_dir`,
  so `XBLICK_HOME` and `XDG_CACHE_HOME` move it). Nodes that pin the same version use one copy.
- `tools/setup.sh` installs it when the PC has an NVIDIA GPU (`nvidia-smi` lists one, driver 580 or newer), and says
  before the 2.4 GB download what it found and installs where. `--gpu` installs it always, `--no-gpu` never. A second
  call reuses it. The install takes a lock and goes to `<dir>.part`, so an interrupted one never looks complete.
- `--device cuda` prints "preparing the GPU" at once, then loads CUDA and computes once in a thread of its own. Ctrl-C
  and SIGTERM end the node during that time.
- On the GPU every part of the model runs there (ONNX Runtime's CPU fallback is off). Missing runtime, a GPU that
  cannot compute or a Python without a wheel end the start with one line; it names the step.
- The package swap in `.venv` that 0.1.0 documented is gone: `uv run` brought the CPU package back over it.
- The start line says where the model computes: "model on the CPU with 4 threads" or "model on cuda:0; scaling,
  filtering and drawing on the CPU".

## 0.1.0 (2026-11-14)

The first version: RT-DETRv4 (S, M, L, X) in ONNX Runtime on the CPU or, with onnxruntime-gpu, on a GPU. Detections
as `vision_msgs/Detection2DArray` on `<camera>/rtdetrv4/detections_2d`, the optional overlay
`<camera>/rtdetrv4/overlay` (`--overlay`), diagnostics, options as parameters (`score` and `classes` live). At the
first start the node downloads the checkpoint and RT-DETRv4's source at a fixed commit and makes the ONNX file on
the user's PC (`--download-only`, `--no-download`, `--weights-dir`).
