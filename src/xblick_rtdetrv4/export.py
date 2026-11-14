"""The ONNX file of a model, made on this PC from RT-DETRv4's checkpoint with RT-DETRv4's own model code.

RT-DETRv4 publishes PyTorch checkpoints, no ONNX files. ``ensure_onnx`` (the node's start, ``--download-only``)
makes the missing one in the node's cache directory:

1. ``xblick.weights`` downloads the checkpoint and RT-DETRv4's source at a fixed commit (``weights.needed``).
2. The source archive is unpacked next to it, once.
3. A child process (``python -m xblick_rtdetrv4.export``, this module's ``main``) builds the model from the
   source's config, loads the checkpoint's EMA weights, switches model and post-processor to their deploy form and
   exports both with ``torch.onnx.export``, as RT-DETRv4's ``tools/deployment/export_onnx.py`` does. This code is
   written for this node. It differs from that script in three points: a batch of one frame (the script traces 32
   frames and needs 32 GB of memory), no onnxsim, and a check of the ONNX file against PyTorch on a test input.
4. ``<onnx>.json`` records what the file was made from. The node refuses an ONNX file whose record names another
   checkpoint.

The child needs PyTorch and the packages that RT-DETRv4's code imports (the extra ``export``, which
``tools/setup.sh`` installs). The node itself needs none of them once the ONNX file exists.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import warnings
from pathlib import Path

from xblick import weights as store_lib

from . import weights

LOCK = ".export.lock"  # in the root: one export at a time, also across processes
# the modules RT-DETRv4's ``engine`` package imports (its requirements.txt), and the PyPI names that bring them
EXPORT_MODULES = {"torch": "torch", "torchvision": "torchvision", "yaml": "pyyaml", "scipy": "scipy",
                  "tensorboard": "tensorboard", "faster_coco_eval": "faster-coco-eval", "calflops": "calflops",
                  "transformers": "transformers", "onnx": "onnx", "onnxruntime": "onnxruntime"}
CHECK_TOP = 20  # the best detections of the test input that the ONNX file must reproduce
SCORE_TOLERANCE = 2e-3  # the largest difference of a score between ONNX Runtime and PyTorch
BOX_TOLERANCE = 1.0  # the same for a box corner, in pixels of the 640 x 480 test frame


def record_of(onnx: Path) -> Path:
    """The record next to an ONNX file: what it was made from."""
    return onnx.with_name(onnx.name + ".json")


def check_made(onnx: Path, model: str, verify: bool = False):
    """WeightsError unless ``onnx`` was made by this node's export from ``model``'s pinned checkpoint and has the size
    its record states (``verify``: also its SHA-256)."""
    record = record_of(onnx)
    if not record.is_file():
        raise store_lib.WeightsError(f"{onnx} has no record {record.name} (not made by this node's export): remove it, "
                                     "then start again")
    made = json.loads(record.read_text())
    expected = weights.MODELS[model][2]
    if made.get("checkpoint_sha256") != expected:
        raise store_lib.WeightsError(f"{onnx} was made from another checkpoint ({made.get('checkpoint_sha256')}, "
                                     f"expected {expected}): remove it and {record.name}, then start again")
    if onnx.stat().st_size != made.get("size"):
        raise store_lib.WeightsError(f"{onnx}: {onnx.stat().st_size} bytes, its record states {made.get('size')}: "
                                     f"remove it and {record.name}, then start again")
    if verify and store_lib.sha256_of(onnx) != made.get("sha256"):
        raise store_lib.WeightsError(f"{onnx}: wrong SHA-256, its record states {made.get('sha256')}: remove it and "
                                     f"{record.name}, then start again")


def missing_modules() -> list[str]:
    """The PyPI names of the export's packages that are not installed."""
    return [name for module, name in EXPORT_MODULES.items() if importlib.util.find_spec(module) is None]


def ensure_onnx(store: store_lib.Store, model: str, *, download: bool = True, verify: bool = False, options=(),
                out=None) -> Path:
    """The ONNX file of ``model`` in ``store``: there and made from the pinned checkpoint, or made now (downloads
    first, see the module). WeightsError with one line for the user when it cannot be made."""
    out = out or sys.stdout
    onnx = store.root / weights.onnx_name(model)
    if onnx.is_file():
        check_made(onnx, model, verify)
        return onnx
    store.ensure(weights.needed(model), download=download, verify=verify, out=out, options=options)
    missing = missing_modules()
    if missing:
        raise store_lib.WeightsError(f"{onnx.name} is made from the checkpoint on this PC, which needs "
                                     f"{', '.join(missing)}: run tools/setup.sh (it installs the extra export), then "
                                     f"{store.command(options)}")
    with _locked(store.root / LOCK, out):
        if onnx.is_file():  # another process made it while this one waited
            check_made(onnx, model, verify)
            return onnx
        code = unpack(store.root / weights.code_name(), store.root)
        checkpoint = store.root / weights.checkpoint_name(model)
        print(f"{weights.PROG}: making {onnx.name} from {checkpoint.name} with RT-DETRv4's code "
              f"(commit {weights.COMMIT[:12]}) on the CPU, about a minute", file=out, flush=True)
        started = time.monotonic()
        child = subprocess.Popen(child_command(code, model, checkpoint, onnx), stderr=subprocess.STDOUT)
        main = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGTERM, _terminate) if main else None
        try:
            status = child.wait()
        except (KeyboardInterrupt, _Terminated) as error:  # Ctrl-C or SIGTERM: the child goes too, its file with it
            child.terminate()
            child.wait()
            onnx.with_name(onnx.name + ".part").unlink(missing_ok=True)  # SIGTERM skips the child's own cleanup
            how, exit_code = ("Ctrl-C", 130) if isinstance(error, KeyboardInterrupt) else ("SIGTERM", 143)
            raise store_lib.WeightsError(f"stopped ({how}) while making {onnx.name}; no partial file kept",
                                         exit_code) from None
        finally:
            if main:
                signal.signal(signal.SIGTERM, previous)
        if status != 0 or not onnx.is_file():
            raise store_lib.WeightsError(f"the export of {onnx.name} failed (exit code {status}, its output above)")
        check_made(onnx, model, verify=True)
        print(f"{weights.PROG}: {onnx.name} made and checked in {time.monotonic() - started:.0f} s", file=out,
              flush=True)
    return onnx


class _Terminated(BaseException):
    """SIGTERM while the child exports (a BaseException: no ``except Exception`` on the way swallows it)."""


def _terminate(signum, frame):
    raise _Terminated


def child_command(code: Path, model: str, checkpoint: Path, onnx: Path) -> list[str]:
    """The export's child process (``main``); tests replace it."""
    return [sys.executable, "-m", "xblick_rtdetrv4.export", "--code", str(code), "--model", model, "--checkpoint",
            str(checkpoint), "--out", str(onnx)]


def download_only(store: store_lib.Store, model: str, options=(), out=None) -> int:
    """``--download-only``: the ONNX file of ``model`` there (downloaded and made where missing, every file checked
    against its SHA-256), one line at the end; the exit code."""
    out = out or sys.stdout
    try:
        onnx = ensure_onnx(store, model, verify=True, options=options, out=out)
    except store_lib.WeightsError as error:
        return store.fail(error)
    print(f"{weights.PROG}: {onnx.name} ready in {onnx.parent}", file=out, flush=True)
    return 0


def unpack(archive: Path, root: Path) -> Path:
    """RT-DETRv4's source from ``archive`` (GitHub's archive of COMMIT), unpacked into ``root`` once: its top
    directory. Unpacked into a temporary directory first, renamed when complete."""
    final = root / f"RT-DETRv4-{weights.COMMIT}"
    if final.is_dir():
        return final
    if not hasattr(tarfile, "data_filter"):
        raise store_lib.WeightsError(f"unpacking {archive.name} needs tarfile's data filter (Python 3.10.12, 3.11.4 "
                                     "or newer)")
    tmp = root / f"{final.name}.part"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")
    inner = tmp / final.name
    if not (inner / "engine").is_dir():
        raise store_lib.WeightsError(f"{archive.name} holds no {final.name}/engine: remove it, then start again")
    inner.rename(final)
    tmp.rmdir()
    return final


@contextlib.contextmanager
def _locked(path: Path, out):
    """The export's lock (fcntl): another process exporting, wait for it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{weights.PROG}: another process makes the ONNX file; waiting for it", file=out, flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def test_frame():
    """The check's input: a fixed 640 x 480 frame of smooth colour gradients and blocks (no file, no download)."""
    import numpy as np  # noqa: PLC0415

    y, x = np.mgrid[0:480, 0:640].astype(np.float32)
    rgb = np.stack((x / 640 * 255, y / 480 * 255, (x + y) % 256), -1)
    for i in range(6):  # a few flat blocks, so that the model sees edges
        x0, y0 = 40 + 95 * i, 60 + 50 * (i % 3)
        rgb[y0:y0 + 120, x0:x0 + 70] = (30 * i, 255 - 40 * i, 120)
    return rgb.astype(np.uint8)


def export(code: Path, model: str, checkpoint: Path, out: Path):
    """The child's work (see the module): ``out`` and its record, or an exception."""
    import onnxruntime as ort  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from .model import preprocess  # noqa: PLC0415

    # the tracer warns at every Python branch on a shape, torch 2.8 of its own exporter's future, the constant
    # folding of slices it leaves: the check against PyTorch below is what counts
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", message="You are using the legacy TorchScript-based ONNX export")
    warnings.filterwarnings("ignore", message="Constant folding - Only steps=1")
    sys.path.insert(0, str(code))
    from engine.core import YAMLConfig  # noqa: PLC0415 - RT-DETRv4's code, pinned by weights.CODE

    cfg = YAMLConfig(str(code / weights.config_of(model)))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # the checkpoint has every weight: no backbone download
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])

    class Deployed(torch.nn.Module):
        """Model and post-processor in their deploy form, as RT-DETRv4's export script combines them."""

        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    deployed = Deployed().eval()
    images, sizes = (torch.from_numpy(a) for a in preprocess(test_frame()))
    with torch.no_grad():
        reference = [t.numpy()[0] for t in deployed(images, sizes)]
    part = out.with_name(out.name + ".part")
    try:
        with torch.no_grad():
            torch.onnx.export(deployed, (images, sizes), str(part), input_names=["images", "orig_target_sizes"],
                              output_names=["labels", "boxes", "scores"], opset_version=16, do_constant_folding=True,
                              dynamo=False)
        session = ort.InferenceSession(str(part), providers=["CPUExecutionProvider"])
        got = [a[0] for a in session.run(None, {"images": images.numpy(), "orig_target_sizes": sizes.numpy()})]
        differences = compare(reference, got)
        record = {"model": model, "checkpoint": checkpoint.name, "checkpoint_sha256": weights.MODELS[model][2],
                  "code": f"{weights.UPSTREAM} at {weights.COMMIT}", "torch": torch.__version__,
                  "opset": 16, "size": part.stat().st_size, "sha256": store_lib.sha256_of(part),
                  "check": differences, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        record_of(out).write_text(json.dumps(record, indent=2) + "\n")
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)
    print(f"{weights.PROG}: ONNX Runtime against PyTorch on the test frame: {differences}", flush=True)


def compare(reference, got) -> dict:
    """ValueError unless each of the CHECK_TOP best detections of PyTorch (labels, boxes, scores) has one in ONNX
    Runtime's output with its class, a score within SCORE_TOLERANCE and corners within BOX_TOLERANCE. The order may
    differ where two scores are nearly equal. Returns the largest differences."""
    import numpy as np  # noqa: PLC0415

    (labels, boxes, scores), (labels2, boxes2, scores2) = reference, got
    if labels2.shape != labels.shape or boxes2.shape != boxes.shape:
        raise ValueError(f"ONNX Runtime's outputs {labels2.shape}, {boxes2.shape}; PyTorch's {labels.shape}, "
                         f"{boxes.shape}")
    worst_score = worst_box = 0.0
    for i in range(min(CHECK_TOP, len(scores))):
        same = np.flatnonzero((labels2 == labels[i]) & (np.abs(scores2 - scores[i]) <= SCORE_TOLERANCE))
        if not len(same):
            raise ValueError(f"PyTorch's detection {i} (class {labels[i]}, score {scores[i]:.4f}) is not in ONNX "
                             "Runtime's output")
        corners = np.abs(boxes2[same] - boxes[i]).max(1)
        j = same[corners.argmin()]
        if corners.min() > BOX_TOLERANCE:
            raise ValueError(f"PyTorch's detection {i}: its box differs by {corners.min():.2f} px in ONNX Runtime")
        worst_score = max(worst_score, float(abs(scores2[j] - scores[i])))
        worst_box = max(worst_box, float(corners.min()))
    return {"detections": min(CHECK_TOP, len(scores)), "best score": round(float(scores[0]), 4),
            "largest score difference": worst_score, "largest box difference px": round(worst_box, 4)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m xblick_rtdetrv4.export",
                                     description="The export's child process (ensure_onnx starts it).")
    parser.add_argument("--code", type=Path, required=True, help="RT-DETRv4's unpacked source")
    parser.add_argument("--model", choices=tuple(weights.MODELS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the ONNX file; its record goes next to it")
    args = parser.parse_args(argv)
    export(args.code, args.model, args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
