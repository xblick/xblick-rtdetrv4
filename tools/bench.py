"""Milliseconds a frame of each RT-DETRv4 model on this PC: the whole detect (resize, model, selection) on
tests/data/office.png (640 x 480), after a warm-up; the median and the 90th percentile, and the model alone.

    taskset -c 0-3 .venv/bin/python tools/bench.py [--threads 4] [--frames 100] [--device cpu|cuda:N] [--models s m]

It measures the models whose ONNX file is in the node's cache directory (XBLICK_HOME moves it).
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from xblick_rtdetrv4 import model, weights

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--frames", type=int, default=100)
parser.add_argument("--warm-up", type=int, default=10)
parser.add_argument("--device", default="cpu")
parser.add_argument("--models", nargs="*", default=list(weights.MODELS))
args = parser.parse_args()
rgb = np.asarray(Image.open(Path(__file__).parents[1] / "tests" / "data" / "office.png").convert("RGB"))
print(f"{rgb.shape[1]} x {rgb.shape[0]}, {args.device}, {args.threads} threads, {args.frames} frames after "
      f"{args.warm_up}")
for name in args.models:
    path = weights.path_of(name)
    if not path.is_file():
        print(f"{name}: no {path} (xblick-rtdetrv4 --download-only --model {name})")
        continue
    detector = model.Detector(path, args.threads, args.device)
    for _ in range(args.warm_up):
        detector.detect(rgb, 0.4)
    total, inner = [], []
    for _ in range(args.frames):
        t0 = time.perf_counter()
        inputs = model.preprocess(rgb)
        t1 = time.perf_counter()
        detector.infer(*inputs)
        t2 = time.perf_counter()
        detector.detect(rgb, 0.4)
        t3 = time.perf_counter()
        total.append((t3 - t2) * 1e3)
        inner.append((t2 - t1) * 1e3)
    q = lambda xs, p: float(np.percentile(xs, p))  # noqa: E731
    print(f"{name}: detect {statistics.median(total):7.1f} ms (p90 {q(total, 90):.1f}), "
          f"model alone {statistics.median(inner):7.1f} ms", flush=True)
