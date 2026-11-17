# AGENTS.md

xblick-rtdetrv4: RT-DETRv4 (real-time DETR, COCO, 80 classes; Liao et al. 2025) as a Durchblick node. ONNX Runtime
runs an ONNX file that the node makes on the user's PC from RT-DETRv4's checkpoint (`export.py`, child process, the
extra `export` with PyTorch for the CPU); pre- and post-processing in numpy (`model.py`). CPU by default, GPU with
onnxruntime-gpu. Optional overlay image stream (`--overlay`). Version 0.1.0, tested on a test camera and a published
image, not with a real webcam. Human documentation: `README.md`, `docs/design.md`, `docs/performance.md`.
Transport-wide traps are in the core's `AGENTS.md` (github.com/xblick/xblick). Style of all documentation: the core's
`docs/project/writing-docs.md`.

## Commands

```bash
tools/setup.sh                                          # uv sync --frozen --extra export, then --download-only (s)
.venv/bin/python -m pytest -q                           # ~12 s; real-model tests skip without the ONNX files
.venv/bin/xblick-rtdetrv4 --download-only --model x     # one model: download, export (~1 min), check
.venv/bin/ruff check src tests tools                    # lint; 120 columns
taskset -c 0-3 .venv/bin/python tools/bench.py          # ms per frame of each model in the cache, ~3 min
XBLICK_RTDETRV4_TEST_GPU=cuda:0 .venv/bin/python -m pytest -q tests/test_model.py   # with onnxruntime-gpu
.venv/bin/xblick router & .venv/bin/xblick test-camera &   # router and camera, as the README's Run
.venv/bin/xblick-rtdetrv4 --camera test/camera --overlay
```

## Rules

- `runtime.py` and `tests/test_runtime.py` are the same in every ONNX Runtime node (xblick-yolox, xblick-rtdetrv4):
  change them in all. The GPU runtime lives outside the venv in `~/.cache/xblick/onnxruntime/gpu-<version>-cp<XY>`;
  never swap onnxruntime for onnxruntime-gpu in `.venv` (`uv run` restores the CPU package over it).
- Files: everything in the node's cache directory (`xblick.paths.cache_dir("rtdetrv4")`, `~/.cache/xblick/rtdetrv4`),
  never in the checkout: the checkpoint, the source archive and its unpacked directory, the ONNX file and its record.
  `store = weights.store_for(parser, args)` (`--weights-dir`); `export.ensure_onnx(store, model, ...)` is the one
  entry (start and `--download-only`); model code takes the path it returns.
- Never copy RT-DETRv4's code into this repository. `export.py` imports it from the unpacked archive of
  `weights.COMMIT`. A new commit: new `CODE` size and SHA-256 (measured), the export tested for every model.
- The export runs in a child process (`child_command`), so that PyTorch never loads into the node. It checks ONNX
  Runtime against PyTorch (`compare`) before the file is renamed into place. Keep both.
- Fixed defaults only: `--model s`, `--score 0.4`, `--threads 4`, `--device cpu`. Never derive a default from the
  machine, the installed packages or a measurement.
- `--device cuda[:N]` without a usable CUDA provider is an error. Never fall back to the CPU.
- Overlay: the core's `xblick.overlay` (`add_argument`, `Overlay`, `draw_boxes`). No `Overlay` without `--overlay`.
  Publish the detections before `overlay.put`.
- `model.py` follows the export's format (inputs `images`, `orig_target_sizes`; outputs `labels`, `boxes`, `scores`);
  keep `tests/test_model.py` green after any change of resize or selection.
- Durchblick pin: `xblick>=0.5.5,<0.6`. Dependencies from PyPI, PyTorch from its CPU index (`tool.uv.sources`).
  Branch `ros2`: the core from its git branch `ros2` (`--ros`), until a release has it.
- 120 columns. Docstrings say what, comments say why. Update `README.md` and `docs/` with every change of behaviour.

## Traps

| Trap | What happens | Fix |
| --- | --- | --- |
| Upstream `export_onnx.py` | It traces a batch of 32 frames: 32 GB of memory, and it needs onnxsim. | `export.py` exports one frame (0.8 to 2.5 GB). |
| torch 2.9 | `torch.onnx.export` uses the dynamo exporter by default. | `dynamo=False`; the extra pins `torch>=2.8,<2.9`. |
| `onnx` package | The TorchScript exporter needs it; without it the child fails with "Module onnx is not installed". | In the extra and in `EXPORT_MODULES`. |
| Engine imports | RT-DETRv4's `engine/__init__` imports its training stack: tensorboard, scipy, faster-coco-eval, calflops, transformers. | All in the extra. |
| `HGNetv2.pretrained` | True in the configs: the backbone would download ImageNet weights. | The export sets it False; the checkpoint has every weight. |
| Resize | RT-DETRv4 evaluates on the frame stretched to 640 x 640; its `onnx_inf.py` letterboxes instead. | `preprocess` stretches, as the evaluation and `torch_inf.py`. |
| Google Drive | `drive.usercontent.google.com/download?...&confirm=t` gives the file without the warning page. A Drive quota can refuse downloads for a while. | `weights.DRIVE`. |
| ORT warning on a GPU | onnxruntime-gpu 1.32 prints "No registered plugin EP device found for 'CUDAExecutionProvider'" at load. The session still runs on CUDA (the node checks the providers; 4 ms per frame for S). | None needed. |

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_rtdetrv4/runtime.py` | ONNX Runtime sessions (CPU, GPU), the shared GPU runtime (`setup`, `install`, `detect`, `onnxruntime_for`, `prepare`); the same file in every ONNX Runtime node |
| `src/xblick_rtdetrv4/model.py` | `preprocess`, `postprocess`, `Detector`, `COCO_CLASSES`, `INPUT_SIZE`, `TOP` |
| `src/xblick_rtdetrv4/export.py` | `ensure_onnx`, `download_only`, `check_made`, `unpack`, `child_command`, the child (`export`, `compare`) |
| `src/xblick_rtdetrv4/node.py` | options, the loop, publishing, `draw` (the overlay), diagnostics |
| `src/xblick_rtdetrv4/weights.py` | `MODELS`, `CODE`, `COMMIT`, `TERMS`, `needed`, `path_of`, `STORE` |
| `~/.cache/xblick/rtdetrv4/` | `RTv4-<S/M/L/X>-hgnet.pth`, `RT-DETRv4-<commit>.tar.gz` and its directory, `rtdetrv4_<m>.onnx` and `.json`, locks |
| `tests/` | `test_model.py`, `test_export.py`, `test_weights.py`, `test_node.py`, `test_overlay.py`, `standin.py`, `data/office.png` |
| `tools/` | `setup.sh`, `bench.py`, `check_references.sh` |

## Defaults

| What | Value |
| --- | --- |
| Input | `<camera>/color` (`--color`): `rgb8`, `rgba8`, `bgr8`, `bgra8`, `mono8` |
| Detections | `<camera>/rtdetrv4/detections_2d`, `vision_msgs/Detection2DArray`, class name as `class_id` |
| Overlay | `<camera>/rtdetrv4/overlay`, `rgb8`, only with `--overlay` |
| Diagnostics | `<camera>/diagnostics`, status `rtdetrv4`, once per second |
| `--max-hz` | none (the newest frame, as fast as it goes) |
| `--local` | `auto` (shared memory) |
