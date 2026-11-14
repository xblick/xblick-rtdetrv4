# How the node works

This page describes where the ONNX file comes from, the model's input and output, the selection of detections, the
overlay and the tests.

## The ONNX file

RT-DETRv4 publishes PyTorch checkpoints on Google Drive and no ONNX files. The node therefore makes the ONNX file on
your PC, once per model, in its cache directory. `export.ensure_onnx` does it at the node's start and for
`--download-only`.

1. `xblick.weights` downloads the checkpoint (`RTv4-S-hgnet.pth`) and GitHub's archive of RT-DETRv4's source at a
   fixed commit. It checks the size and the SHA-256 of both.
2. The node unpacks the source next to the archive, once.
3. A child process builds the model from the source's config with RT-DETRv4's own code. It loads the checkpoint's EMA
   weights, switches model and post-processor to their deploy form and exports both with `torch.onnx.export`.
4. The child runs the ONNX file in ONNX Runtime and the model in PyTorch on a fixed test frame. Each of the 20 best
   detections of PyTorch must be in ONNX Runtime's output with its class, a score within 0.002 and corners within
   1 pixel. Only then does the child rename the file into place.
5. A record next to the file (`rtdetrv4_s.onnx.json`) states the checkpoint's SHA-256, the commit, the PyTorch
   version, the file's size and SHA-256 and the check's result. The node refuses an ONNX file whose record names
   another checkpoint or another size.

The export follows RT-DETRv4's `tools/deployment/export_onnx.py`, with three differences.

| Upstream script | This node | Reason |
| --- | --- | --- |
| traces a batch of 32 frames | traces one frame | The upstream script needed 32 GB of memory for the S model. One frame needs 0.8 GB (S) to 2.5 GB (X). |
| simplifies the file with onnxsim | no simplification | onnxsim is one more native package. ONNX Runtime optimises the graph when it loads the file. |
| checks the file with `onnx.checker` | compares ONNX Runtime with PyTorch | The comparison checks the numbers, which the checker does not. |

The export runs in a child process, so that PyTorch never loads into the node. It needs the extra `export`: PyTorch
for the CPU and the packages that RT-DETRv4's `engine` package imports at its import, among them its training stack
(tensorboard, faster-coco-eval, calflops, transformers). `tools/setup.sh` installs the extra. Once the ONNX file
exists, the node needs none of these packages, and a start reads only the ONNX file and its record.

The export does not need DINOv3. The teacher model appears only in the training configs, and the checkpoints hold the
detector's weights only. The node never downloads DINOv3 code or weights.

## The model

The ONNX file has two inputs. `images` is a (1, 3, 640, 640) float32 array: the frame resized to 640 x 640 with
Pillow's bilinear resampling, in RGB order, with values from 0 to 1. The resize does not keep the aspect ratio,
because RT-DETRv4 trained and evaluated on frames stretched this way (`Resize` in `configs/base/dataloader.yml`).
`orig_target_sizes` is a (1, 2) int64 array with the frame's width and height. The model scales its boxes to this
size, so the boxes come out in pixels of the frame.

The model returns its 300 best (query, class) pairs: `labels` (1, 300) int64 COCO class indices, `boxes` (1, 300, 4)
float32 corners x1, y1, x2, y2, and `scores` (1, 300) float32 sigmoid scores, best first. The class index counts the
80 COCO classes in their usual order, which `model.COCO_CLASSES` lists.

## The selection

1. The node keeps the pairs with a score of at least `--score`, and only the classes of `--classes`.
2. It sorts them by score and clips the boxes to the frame.

RT-DETRv4 needs no non-maximum suppression, because each of its 300 queries predicts one object and its training
matches each object to one query. Two pairs can still share a query with two classes. The node keeps both, as
RT-DETRv4's own scripts do.

`--score` defaults to 0.4, the threshold that RT-DETRv4's inference scripts (`tools/inference/*.py`) use for drawing.
`model.py` holds the pre- and post-processing in numpy. It follows the format above and copies no code of RT-DETRv4.

## The overlay

| Rule | This node |
| --- | --- |
| Option | `--overlay`, off by default, read-only (it applies at the start) |
| Key | `<camera>/<node>/overlay`, an ordinary image stream |
| Encoding | `rgb8`, the colour frame's size, its stamp, its camera frame and its camera_info. Remote readers get JPEG. |
| Without the option | no publisher, so no topic and no info record |
| Order | the node publishes the detections first, then hands the frame to the overlay thread |
| Load | the thread draws only while a subscriber exists. It keeps one frame, and a newer frame replaces a waiting one. |
| Code | the core's `xblick.overlay`: `Overlay` publishes and draws in its thread, `draw_boxes` draws. The node's `draw` gives the texts and colours. |

The drawing shows each box in a fixed colour per class, with the class name and the score above the box.

## Tests

| File | What |
| --- | --- |
| `tests/test_model.py` | resize and selection on synthetic data, the real models on `office.png` on the CPU and, with `XBLICK_RTDETRV4_TEST_GPU=cuda:N`, on a GPU (skip without the ONNX files in the cache directory) |
| `tests/test_export.py` | the record and its checks, unpacking the source, the comparison of ONNX Runtime with PyTorch, the extra's packages |
| `tests/test_weights.py` | the file list, `--download-only`, `--no-download`, `--weights-dir` and a start without the model, against a local HTTP server, with a stand-in for the export's child |
| `tests/test_node.py` | conversions and options, the node as its own process with a stand-in model and with the real S model |
| `tests/test_overlay.py` | the drawing: boxes and labels where the detections are |

The stand-in model (`tests/standin.py`) replaces only ONNX Runtime. It returns 300 pairs in which a person and a cup
score high, so the node's pre- and post-processing run as they do with the real model. The tests need neither
PyTorch nor the network, so CI runs them without the extra.
