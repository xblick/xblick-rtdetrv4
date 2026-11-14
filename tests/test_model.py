"""Pre- and post-processing on synthetic data (no model), and the real models on a frame of an office (they skip
without the ONNX files in the node's cache directory)."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from standin import OBJECTS, StandIn, outputs_for

from xblick_rtdetrv4 import model, weights


def test_preprocess_resizes_to_640_without_keeping_the_aspect_ratio_in_rgb_from_0_to_1():
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[:, :160] = (255, 0, 0)  # the left half red, the right half blue
    rgb[:, 160:] = (0, 0, 255)
    images, sizes = model.preprocess(rgb)
    assert images.shape == (1, 3, 640, 640) and images.dtype == np.float32 and images.flags.c_contiguous
    assert sizes.tolist() == [[320, 240]] and sizes.dtype == np.int64  # width, height
    assert tuple(images[0, :, 0, 0]) == (1.0, 0.0, 0.0) and tuple(images[0, :, 639, 0]) == (1.0, 0.0, 0.0)
    assert tuple(images[0, :, 639, 639]) == (0.0, 0.0, 1.0)  # stretched: the bottom row is the frame's, no padding
    assert tuple(images[0, :, 320, 300]) == (1.0, 0.0, 0.0) and tuple(images[0, :, 320, 340]) == (0.0, 0.0, 1.0)


def test_preprocess_takes_a_640_square_as_it_is():
    rgb = np.random.default_rng(0).integers(0, 256, (640, 640, 3), dtype=np.uint8)
    images, sizes = model.preprocess(rgb)
    np.testing.assert_array_equal(images[0], rgb.transpose(2, 0, 1) / np.float32(255))
    assert sizes.tolist() == [[640, 640]]


def test_postprocess_keeps_the_objects_above_the_score_best_first():
    labels, boxes, scores = outputs_for(OBJECTS, 640, 480)
    found = model.postprocess(labels, boxes, scores, (640, 480), score=0.4)
    assert found.labels.tolist() == [0, 41]
    np.testing.assert_allclose(found.scores, [0.9, 0.6], rtol=1e-6)
    np.testing.assert_allclose(found.boxes, [[64, 60, 320, 384], [416, 192, 480, 240]], atol=1e-3)
    assert len(model.postprocess(labels, boxes, scores, (640, 480), 0.7)) == 1
    assert model.postprocess(labels, boxes, scores, (640, 480), 0.3, {41}).labels.tolist() == [41]
    assert len(model.postprocess(labels, boxes, scores, (640, 480), 0.95)) == 0
    assert len(model.postprocess(labels, boxes, scores, (640, 480), 0.005)) == model.TOP  # no suppression


def test_postprocess_sorts_and_clips_to_the_frame():
    labels = np.array([2, 0], np.int64)
    boxes = np.array([[-20, 10, 100, 450], [5, 5, 50, 50]], np.float32)
    scores = np.array([0.5, 0.8], np.float32)
    found = model.postprocess(labels, boxes, scores, (416, 300), 0.3)
    assert found.labels.tolist() == [0, 2]
    np.testing.assert_allclose(found.boxes, [[5, 5, 50, 50], [0, 10, 100, 300]], atol=1e-3)


def test_the_stand_in_detector_goes_through_the_whole_pipeline():
    detector = StandIn(Path("none"), 4)
    found = detector.detect(np.zeros((480, 640, 3), np.uint8), 0.4)
    assert found.labels.tolist() == [0, 41]
    np.testing.assert_allclose(found.boxes[0], [64, 60, 320, 384], atol=1e-3)


OFFICE = Path(__file__).parent / "data" / "office.png"
PLAUSIBLE = {"chair", "bottle", "cup", "dining table", "book", "tv", "laptop", "keyboard", "scissors", "vase",
             "potted plant", "remote", "mouse", "cell phone", "bowl"}


def office_objects(name: str, device: str):
    path = weights.path_of(name)
    if not path.is_file():
        pytest.skip(f"no {path.name} (xblick-rtdetrv4 --download-only --model {name})")
    rgb = np.asarray(Image.open(OFFICE).convert("RGB"))
    found = model.Detector(path, 4, device).detect(rgb, 0.4)
    return [model.COCO_CLASSES[i] for i in found.labels], found


def check_office(names, found):
    # the frame shows two chairs and bottles on a desk: robust to the variant, no exact boxes
    assert names.count("chair") >= 2 and "bottle" in names, names
    assert found.scores[names.index("chair")] > 0.8
    confident = [n for n, s in zip(names, found.scores, strict=True) if s >= 0.6]  # below: a handbag, a ball (x, l)
    assert set(confident) <= PLAUSIBLE, names
    chairs = found.boxes[[n == "chair" for n in names]]
    assert (chairs[:, 3] > 400).all()  # both chairs stand in the lower part of the 480 rows


@pytest.mark.parametrize("name", list(weights.MODELS))
def test_the_real_model_finds_the_office_objects(name):
    check_office(*office_objects(name, "cpu"))


@pytest.mark.skipif(not os.environ.get("XBLICK_RTDETRV4_TEST_GPU"),
                    reason="XBLICK_RTDETRV4_TEST_GPU=cuda:N names a GPU (and needs onnxruntime-gpu)")
@pytest.mark.parametrize("name", list(weights.MODELS))
def test_the_real_model_finds_the_office_objects_on_a_gpu(name):
    check_office(*office_objects(name, os.environ["XBLICK_RTDETRV4_TEST_GPU"]))
