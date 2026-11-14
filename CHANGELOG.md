# Changelog

## 0.1.0 (2026-11-14)

The first version: RT-DETRv4 (S, M, L, X) in ONNX Runtime on the CPU or, with onnxruntime-gpu, on a GPU. Detections
as `vision_msgs/Detection2DArray` on `<camera>/rtdetrv4/detections_2d`, the optional overlay
`<camera>/rtdetrv4/overlay` (`--overlay`), diagnostics, options as parameters (`score` and `classes` live). At the
first start the node downloads the checkpoint and RT-DETRv4's source at a fixed commit and makes the ONNX file on
the user's PC (`--download-only`, `--no-download`, `--weights-dir`).
