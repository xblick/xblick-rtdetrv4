"""RT-DETRv4 on the CPU or a GPU: ONNX Runtime runs the model, numpy does the rest.

    detector = Detector(weights.path_of("s"), threads=4)
    found = detector.detect(rgb, score=0.5)                  # rgb: (H, W, 3) uint8
    found.boxes, found.scores, found.labels                  # x1 y1 x2 y2 in pixels of rgb; COCO class indices

The ONNX file is made on the user's PC from the upstream checkpoint (``export.py``). Its format is that of
RT-DETRv4's ``tools/deployment/export_onnx.py``, which this node's export follows. The pre- and post-processing below
are written for this node in numpy from that format; no code of RT-DETRv4 is copied:

* input ``images``, (1, 3, 640, 640) float32: the frame resized to 640 x 640 without keeping its aspect ratio
  (bilinear), RGB, values 0 to 1, no normalisation. RT-DETRv4's evaluation (``configs/base/dataloader.yml``,
  ``Resize`` then ``ConvertPILImage``) and its ``tools/inference/torch_inf.py`` prepare a frame the same way.
* input ``orig_target_sizes``, (1, 2) int64: the frame's width and height. The model scales its boxes to them.
* outputs ``labels`` (1, 300) int64, ``boxes`` (1, 300, 4) float32 x1, y1, x2, y2 in frame pixels, ``scores``
  (1, 300) float32: the 300 best (query, class) pairs by score, best first. A score is a sigmoid. The model has no
  non-maximum suppression and needs none: each query predicts one object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_SIZE = 640  # the side of the square input of every RT-DETRv4 model (eval_spatial_size in its configs)
TOP = 300  # the detections the model returns per frame (PostProcessor.num_top_queries)

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass
class Detections:
    """Objects in a frame, best first: ``boxes`` (n, 4) float32 x1, y1, x2, y2 in pixels of the frame, ``scores``
    (n,) float32, ``labels`` (n,) int64 indices into COCO_CLASSES."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> Detections:
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


def preprocess(rgb: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """The model's inputs for an (H, W, 3) uint8 RGB frame: ``images`` (1, 3, size, size) float32, the frame resized
    to size x size (bilinear, Pillow), values 0 to 1; and ``orig_target_sizes`` (1, 2) int64, the frame's width and
    height."""
    h, w = rgb.shape[:2]
    scaled = rgb if (w, h) == (size, size) else np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
    images = (scaled.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).astype(np.float32)
    return np.ascontiguousarray(images), np.array([[w, h]], np.int64)


def postprocess(labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, frame_size: tuple[int, int],
                score: float, classes: set[int] | None = None) -> Detections:
    """Detections of one frame from the model's outputs (first dimension 1 removed): the pairs that score at least
    ``score`` (and whose class is one of ``classes``; None: all), best first, boxes clipped to the frame of
    ``frame_size`` (w, h). No suppression: RT-DETRv4 predicts each object once."""
    chosen = scores >= score
    if classes is not None:
        chosen &= np.isin(labels, list(classes))
    if not chosen.any():
        return Detections.empty()
    boxes, scores, labels = boxes[chosen].astype(np.float32), scores[chosen], labels[chosen]
    order = np.argsort(-scores, kind="stable")
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    w, h = frame_size
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    return Detections(boxes, scores.astype(np.float32), labels.astype(np.int64))


class Detector:
    """One RT-DETRv4 model in ONNX Runtime, ``threads`` CPU threads for one frame at a time; ``device`` ``cpu``, or
    ``cuda:N`` (ONNX Runtime's CUDA provider on GPU N, which the onnxruntime-gpu package brings: without it an
    error, never the CPU instead)."""

    def __init__(self, path: Path, threads: int, device: str = "cpu"):
        self.size = INPUT_SIZE
        self.session = self._load(path, threads, device)

    def _load(self, path: Path, threads: int, device: str):
        import onnxruntime as ort  # noqa: PLC0415 - loaded with the model

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # the threads for one model call
        options.inter_op_num_threads = 1  # one call at a time
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # the threads sleep between frames (a notebook's battery) instead of spinning while they wait for work
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        if device != "cpu":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(f"--device {device}: this onnxruntime has no CUDA provider (it has "
                                   f"{', '.join(ort.get_available_providers())}); install onnxruntime-gpu in its "
                                   "place, or use --device cpu")
            # CUDA and cuDNN of the nvidia-* wheels that onnxruntime-gpu[cuda,cudnn] installs (else the system's)
            ort.preload_dlls()
            providers = [("CUDAExecutionProvider", {"device_id": int(device.split(":")[1])}), "CPUExecutionProvider"]
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if device != "cpu" and session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"--device {device}: ONNX Runtime could not use the GPU ({session.get_providers()})")
        inputs = {i.name: list(i.shape) for i in session.get_inputs()}
        outputs = [o.name for o in session.get_outputs()]
        expected = {"images": [1, 3, self.size, self.size], "orig_target_sizes": [1, 2]}
        if inputs != expected or outputs != ["labels", "boxes", "scores"]:
            raise ValueError(f"{path}: inputs {inputs} and outputs {outputs}, expected {expected} and labels, boxes, "
                             "scores (an RT-DETRv4 ONNX file of this node's export)")
        if device != "cpu":  # a GPU that loads but cannot compute (no cuDNN) fails here, at the start
            try:
                self._run(session, *preprocess(np.zeros((self.size, self.size, 3), np.uint8), self.size))
            except Exception as error:  # noqa: BLE001 - ONNX Runtime's errors: one line for the user
                raise RuntimeError(f"--device {device}: ONNX Runtime failed on the GPU: {str(error)[:300]}") from None
        return session

    @staticmethod
    def _run(session, images: np.ndarray, sizes: np.ndarray):
        labels, boxes, scores = session.run(None, {"images": images, "orig_target_sizes": sizes})
        return labels[0], boxes[0], scores[0]

    def infer(self, images: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The model's labels (300,), boxes (300, 4) and scores (300,) for one input."""
        return self._run(self.session, images, sizes)

    def detect(self, rgb: np.ndarray, score: float, classes: set[int] | None = None) -> Detections:
        """The objects in an (H, W, 3) uint8 RGB frame (see ``postprocess``)."""
        labels, boxes, scores = self.infer(*preprocess(rgb, self.size))
        return postprocess(labels, boxes, scores, (rgb.shape[1], rgb.shape[0]), score, classes)
