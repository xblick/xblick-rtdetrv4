"""Synthetic RT-DETRv4 outputs, for the tests: ``outputs_for`` builds the model's 300 (label, box, score) entries in
which the given objects are found, and ``StandIn`` is a Detector that returns them without a model (no ONNX Runtime,
no weights)."""

import numpy as np

from xblick_rtdetrv4 import model

# boxes as fractions of the frame (x1, y1, x2, y2), class, score: a person and a cup
OBJECTS = [((0.1, 0.125, 0.5, 0.8), 0, 0.9), ((0.65, 0.4, 0.75, 0.5), 41, 0.6)]


def outputs_for(objects, width: int, height: int):
    """labels (300,), boxes (300, 4) in pixels of a width x height frame and scores (300,), best first: the objects,
    then filler entries of score 0.01 (as the model returns its 300 best pairs, most of them unlikely)."""
    labels = np.full(model.TOP, 56, np.int64)  # chair
    boxes = np.tile(np.array([[0, 0, 4, 4]], np.float32), (model.TOP, 1))
    scores = np.full(model.TOP, 0.01, np.float32)
    for i, ((x1, y1, x2, y2), label, score) in enumerate(sorted(objects, key=lambda o: -o[2])):
        labels[i], scores[i] = label, score
        boxes[i] = (x1 * width, y1 * height, x2 * width, y2 * height)
    return labels, boxes, scores


class StandIn(model.Detector):
    def _load(self, path, threads, device):
        return None

    def infer(self, images, sizes):
        assert images.shape == (1, 3, self.size, self.size) and images.dtype == np.float32
        assert sizes.shape == (1, 2) and sizes.dtype == np.int64
        return outputs_for(OBJECTS, int(sizes[0, 0]), int(sizes[0, 1]))
