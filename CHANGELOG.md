# Changelog

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
