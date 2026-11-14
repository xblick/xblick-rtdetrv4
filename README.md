# xblick-rtdetrv4

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)
[![tests](https://github.com/xblick/xblick-rtdetrv4/actions/workflows/tests.yml/badge.svg)](https://github.com/xblick/xblick-rtdetrv4/actions/workflows/tests.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![device](https://img.shields.io/badge/runs%20on-CPU%20or%20NVIDIA%20GPU-brightgreen)](README.md)

This [Durchblick](https://github.com/xblick/xblick) node finds the objects of COCO's 80 classes in a camera's colour
stream. It runs on a CPU without a GPU: RT-DETRv4-S needs 52 ms per 640 x 480 frame on 4 CPU threads of our test
PC. On a GPU each of the four models needs 6 to 11 ms.

The node uses [RT-DETRv4](https://github.com/AIMIA-PKU/RT-DETRv4) by Liao et al. (2025), a real-time detection
transformer that learned from the vision foundation model DINOv3 during its training. ONNX Runtime runs the model,
and numpy does the pre- and post-processing. The node makes the ONNX file on your PC from RT-DETRv4's checkpoint.

## What it publishes

The node publishes for the newest colour frame, as often as the device finishes one. Each message has the stamp and
the camera frame of its colour frame.

| Topic | Message type | Content |
| --- | --- | --- |
| `<camera>/rtdetrv4/detections_2d` | `vision_msgs/Detection2DArray` | one detection per object, the best first. `class_id` is the class name, such as `person`, and `score` is the model's score from 0 to 1. The box is in pixels of the colour image. |
| `<camera>/rtdetrv4/overlay` | image stream, encoding `rgb8` | only with `--overlay`: the colour image with the boxes, class names and scores drawn on it |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | status `rtdetrv4` once per second: frames per second, ms per frame, objects per frame, ms from the frame to the results |

A frame without objects gives an empty array. `--node` changes the `rtdetrv4` part of the topics. The node reads
`<camera>/color` in the encoding `rgb8`, `rgba8`, `bgr8`, `bgra8` or `mono8`.

The overlay is off by default, so that the topics hold only results. With `--overlay` the node draws in a thread of
its own after it has published the detections, so the drawing never delays them. It draws only while a subscriber
exists. Without `--overlay` the stream does not exist at all.
[How the node works](docs/design.md#the-overlay) describes the rules.

## Requirements

| What | Requirement |
| --- | --- |
| CPU | an x86-64 PC with Linux. We measured on 4 cores of an AMD EPYC 9335. |
| GPU | none. An NVIDIA GPU is optional, see "On a GPU". |
| Memory | 0.9 GB (S) to 2.6 GB (X) while the node makes the ONNX file, once per model |
| Durchblick | 0.5.5 or newer, below 0.6 (Python package `xblick`) |
| Python | 3.10 to 3.13, which `tools/setup.sh` puts into `.venv` |
| Tools for the setup | uv |
| Disk | 1.3 GB for `.venv` with the export's packages. Per model, the checkpoint and the ONNX file: 0.2 GB (S), 0.4 GB (M), 0.6 GB (L) or 1.3 GB (X). |

Model licences:

| Part | Licence |
| --- | --- |
| RT-DETRv4's code | Apache 2.0, as the repository's LICENSE states. It builds on RT-DETR, D-FINE and DEIM, whose files keep their authors' copyright lines. |
| RT-DETRv4's checkpoints (S, M, L, X) | RT-DETRv4's README links them on Google Drive. It states no terms for them. |
| DINOv3, the teacher in RT-DETRv4's training | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) by Meta. The node neither downloads nor runs DINOv3. Neither RT-DETRv4 nor Meta states whether this licence extends to detectors trained with DINOv3 as the teacher. |

All four checkpoints come from the same source with the same statements, so the node offers all four. The node
prints these terms before it downloads a model. This repository contains no copy of RT-DETRv4's code or
checkpoints.

## Install

```bash
tools/setup.sh
```

The script creates `.venv` exactly as `uv.lock` states. It installs ONNX Runtime for the CPU, Durchblick from PyPI
and the extra `export`: PyTorch for the CPU and the packages that RT-DETRv4's code imports. Then it runs
`xblick-rtdetrv4 --download-only`, which prepares RT-DETRv4-S in `~/.cache/xblick/rtdetrv4/`.

1. It downloads the checkpoint (169 MB) from RT-DETRv4's Google Drive link and RT-DETRv4's source (5 MB) at a fixed
   commit from GitHub. It checks the size and the SHA-256 of both.
2. It makes the ONNX file (42 MB) from them on the CPU, in about 25 s. It checks the file against PyTorch on a test
   frame.

`tools/setup.sh --model x` prepares another model. The script does only what is missing.

The node also prepares a missing model at its start. A machine that must never start a download on its own starts
the node with `--no-download`. The node then ends at once when a download is needed, and it names the command. Once
the ONNX file exists, the node reads only that file and needs none of the export's packages.

The node is not on PyPI yet. We plan to publish it there, so that `pip install "xblick-rtdetrv4[export]"` installs
it.

### Where the weights are

The node keeps its files in its cache directory, `~/.cache/xblick/rtdetrv4/`. `XBLICK_HOME=/dir` moves it to
`/dir/cache/rtdetrv4/`, and `XDG_CACHE_HOME` moves it as well. `--weights-dir DIR` names another directory.
[Where files are](https://github.com/xblick/xblick/blob/main/docs/reference/files.md) lists the places, and it shows
how to copy the cache to a PC without internet. After the export you may delete the checkpoint `RTv4-*-hgnet.pth`.
The node needs it again only to make the ONNX file anew.

## Run

Start a router, a camera, the node and a viewer, each in its own terminal. The camera name is an example.

```bash
.venv/bin/xblick router                                  # the Zenoh router of this PC
.venv/bin/xblick test-camera                             # a test pattern on test/camera/color
.venv/bin/xblick-rtdetrv4 --camera test/camera --overlay
# on a PC with an NVIDIA GPU: add --device cuda (see "On a GPU")
.venv/bin/xblick view test/camera/color test/camera/rtdetrv4/overlay
```

`xblick view` opens VLC, or offers the browser view at `http://127.0.0.1:8090/`. The test pattern has no objects
in it. A webcam has: `xblick cam /dev/video0` publishes `cam/video0/color`, and `--camera cam/video0` takes it.
`xblick cam` needs the core's `cam` extra, which `.venv` does not have and `tools/setup.sh` does not install.
Install it separately for `xblick cam`: `pip install "xblick[cam]"`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera CAMERA` | required | the base name of the camera, such as `cam/video0` |
| `--model NAME` | `s` | `s`, `m`, `l` or `x`, RT-DETRv4-S to RT-DETRv4-X. All take 640 x 640. |
| `--score S` | 0.4 | the least score of an object, the threshold of RT-DETRv4's inference scripts. Live. |
| `--classes NAME ...` | all 80 | only these COCO classes, also comma-separated: `--classes person,cup`. Live. |
| `--overlay` | off | also publish `<camera>/rtdetrv4/overlay` |
| `--threads N` | 4 | the CPU threads of ONNX Runtime for one frame |
| `--device DEVICE` | `cpu` | `cpu`, `cuda` (GPU 0) or `cuda:N`. A GPU that ONNX Runtime cannot use ends the start with one line. |
| `--max-hz HZ` | every frame | the most frames per second. The node always takes the newest frame. |
| `--local` | `auto` | how frames of a camera on this host reach the node: `auto` or `shm` |
| `--color TOPIC` | `<camera>/color` | the colour stream |
| `--node NAME` | `rtdetrv4` | the node's part of the topics |
| `--no-download` | off | end at once when a download is needed |
| `--download-only` | off | download the missing files, make the ONNX file, check everything and end. `--camera` may be left out. |
| `--weights-dir DIR` | the cache directory | where the files are and go, in place of `~/.cache/xblick/rtdetrv4/` |

Every default is a fixed value. The node never chooses a model, a device or a thread count from the machine.
RT-DETRv4 needs no non-maximum suppression, so the node has no option for it.

Every option is also a parameter of the node `<camera>/rtdetrv4`, which Durchblick's terminal interface (`xblick`,
tab Parameters) and `xblick param` show. `score` and `classes` change while the node runs. The other options apply at
the next start.

## On a GPU

`uv.lock` installs ONNX Runtime for the CPU. For a GPU, replace it in `.venv` by `onnxruntime-gpu` with NVIDIA's CUDA
13 and cuDNN 9 libraries from PyPI (2.5 GB):

```bash
uv pip uninstall --python .venv/bin/python onnxruntime
uv pip install --python .venv/bin/python "onnxruntime-gpu[cuda,cudnn]==1.32.0"
.venv/bin/xblick-rtdetrv4 --camera test/camera --overlay --device cuda
```

`tools/setup.sh` installs the CPU package again, because it restores `.venv` as `uv.lock` states. A GPU that ONNX
Runtime cannot use ends the start with one line. The node never falls back to the CPU. The export of the ONNX file
always runs on the CPU.

## Which model

The table gives the accuracy that RT-DETRv4's README states for COCO val2017
([its table](https://github.com/AIMIA-PKU/RT-DETRv4#-performance)), and our milliseconds per 640 x 480 frame for the
whole detection: resizing, model and selection. We measured on 4 threads of an AMD EPYC 9335 with the process
limited to 4 cores, and on one RTX PRO 6000, as medians (2026-11-14).

| Model | AP (upstream) | ONNX file | Checkpoint | ms per frame, 4 CPU threads | ms per frame, GPU |
| --- | --- | --- | --- | --- | --- |
| s | 49.8 | 42 MB | 169 MB | 52 | 5.9 |
| m | 53.7 | 78 MB | 318 MB | 99 | 7.7 |
| l | 55.4 | 124 MB | 505 MB | 133 | 9.0 |
| x | 57.0 | 248 MB | 1010 MB | 255 | 10.9 |

Without a GPU, take S. Its model alone allows 19 frames per second on 4 threads, and the node reached 10 to 15 on
our shared PC. M reaches about 9. L and X need 133 and 255 ms per frame on the CPU, which suits only scenes that
change slowly. On a GPU each of the four models kept up with a test camera at 30 frames per second.
[Speed](docs/performance.md) has the setup and the node's rate end to end.

## Limits

- We tested the node on a test camera and on a replayed image, as separate processes. We have not tested it with a
  real webcam. The test PC has no display, so a script checked the overlay through `xblick view`'s web page.
- The export needs PyTorch 2.8. We have not tested the exporter of PyTorch 2.9 and newer.
- Google Drive can refuse downloads of a file for a while when many people download it. The download then fails
  with an HTTP error. Start again later, or copy the cache from a PC that has the model.
- RT-DETRv4 knows only COCO's 80 classes. It does not track, so a detection has no id.
- The node runs on Linux on x86-64, because the core's wheels exist only there.

## Documentation

- [How the node works](docs/design.md): the ONNX file's export, the model's input and output, the selection, the
  overlay, the tests
- [Speed](docs/performance.md): the measurements and their setup
- Durchblick: [streams and topics](https://github.com/xblick/xblick/blob/main/docs/concepts/streams-and-topics.md),
  [sessions and routers](https://github.com/xblick/xblick/blob/main/docs/concepts/sessions.md)

## Licence and credits

RT-DETRv4 is by Zijun Liao, Yian Zhao, Xin Shan, Yu Yan, Chang Liu, Lei Lu, Xiangyang Ji and Jie Chen
([AIMIA-PKU/RT-DETRv4](https://github.com/AIMIA-PKU/RT-DETRv4), [arXiv:2510.25257](https://arxiv.org/abs/2510.25257)),
under the Apache License 2.0. "Requirements" lists what is known about its checkpoints.

The code of this repository is under the Apache License 2.0, see [LICENSE](LICENSE). [NOTICE](NOTICE) names the
copyright holder and the parts of others, among them the test image of the TUM RGB-D dataset (CC BY 4.0).
