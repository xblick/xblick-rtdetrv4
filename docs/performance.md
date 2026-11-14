# Speed

This page lists the measured times of the four models on the CPU and on a GPU, the time and memory of the ONNX
file's export, and the node's rate end to end.

## Setup

The PC has two AMD EPYC 9335 with 32 cores each and NVIDIA RTX PRO 6000 Blackwell GPUs (driver 580, CUDA 13.0).
The CPU measurements limited the process to 4 cores (`taskset -c 0-3`) and ONNX Runtime to 4 threads
(`--threads 4`), to come close to a notebook. Other users' jobs ran on the PC (load average 7 to 15) and on other GPUs.
The input was `tests/data/office.png`, 640 x 480. The software was ONNX Runtime 1.32.0 (CPU), onnxruntime-gpu
1.32.0 with NVIDIA's CUDA 13 and cuDNN 9 wheels, and ONNX files made with PyTorch 2.8.0. Date: 2026-11-14.

## Per frame

`tools/bench.py` times the frames after a warm-up: 100 after 10 on the CPU, 200 after 20 on the GPU. "Detect" is the
whole detection: resizing on the CPU, the model, and the selection.

| Model | Detect, median, CPU | Detect, 90th percentile, CPU | Model alone, CPU | Detect, median, GPU | Model alone, GPU |
| --- | --- | --- | --- | --- | --- |
| s | 52.0 ms | 52.4 ms | 49.3 ms | 5.9 ms | 3.9 ms |
| m | 98.6 ms | 99.1 ms | 95.7 ms | 7.7 ms | 5.5 ms |
| l | 132.8 ms | 133.3 ms | 129.9 ms | 9.0 ms | 6.9 ms |
| x | 254.6 ms | 255.9 ms | 251.6 ms | 10.9 ms | 8.7 ms |

```bash
taskset -c 0-3 .venv/bin/python tools/bench.py
.venv/bin/python tools/bench.py --device cuda:0 --frames 200 --warm-up 20   # with onnxruntime-gpu
```

On the GPU, about 2 ms of each frame are the resizing to 640 x 640 and the conversion to float on the CPU. The GPU
measurement did not limit the CPU cores.

By these numbers, S is the model for a PC without a GPU: 19 frames per second at most on 4 threads. M reaches 10
frames per second. L and X reach 7.5 and 4, which suits only scenes that change slowly.

## The export

`xblick-rtdetrv4 --download-only` made each ONNX file from its checkpoint on the CPU, without a core limit. The
times include unpacking the source for the first model and the check against PyTorch.

| Model | Time | Peak memory |
| --- | --- | --- |
| s | 23 s | 0.84 GB |
| m | 28 s | 1.13 GB |
| l | 37 s | 1.53 GB |
| x | 46 s | 2.62 GB |

RT-DETRv4's own `tools/deployment/export_onnx.py` needed 32.1 GB of memory for the S model, because it traces a batch
of 32 frames. The node's export traces one frame. [How the node works](design.md#the-onnx-file) lists the other
differences. Two exports of S with PyTorch 2.8.0 gave the same file, byte for byte.

## Settings

ONNX Runtime's threads sleep between frames (`session.intra_op.allow_spinning` is `0`), so the node uses no CPU
while it waits for a frame. We have not measured the effect of this setting.
