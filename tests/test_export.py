"""The ONNX file's export around the child process: the record and its checks, unpacking the source archive, the
comparison of ONNX Runtime with PyTorch, and the packages it needs. No PyTorch, no network. The export itself is
tested with the real checkpoints (README, docs/performance.md)."""

import io
import json
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from standin import OBJECTS, outputs_for
from xblick import weights as store

from xblick_rtdetrv4 import export, weights

TOP = f"RT-DETRv4-{weights.COMMIT}"


def made(onnx: Path, model: str = "s", data: bytes = b"onnx", **record):
    """An ONNX file and its record as the export writes them."""
    onnx.write_bytes(data)
    entry = {"checkpoint_sha256": weights.MODELS[model][2], "size": len(data),
             "sha256": store.sha256_of(onnx)} | record
    export.record_of(onnx).write_text(json.dumps(entry))


def test_the_record_must_name_the_pinned_checkpoint_and_the_size(tmp_path):
    onnx = tmp_path / "rtdetrv4_s.onnx"
    onnx.write_bytes(b"onnx")
    with pytest.raises(store.WeightsError, match="has no record rtdetrv4_s.onnx.json"):
        export.check_made(onnx, "s")
    made(onnx)
    export.check_made(onnx, "s", verify=True)
    with pytest.raises(store.WeightsError, match="made from another checkpoint"):
        export.check_made(onnx, "m")
    made(onnx, size=3)
    with pytest.raises(store.WeightsError, match="4 bytes, its record states 3"):
        export.check_made(onnx, "s")
    made(onnx, sha256="0" * 64)
    export.check_made(onnx, "s")  # the SHA-256 only with verify (--download-only): a start reads no 40 MB
    with pytest.raises(store.WeightsError, match="wrong SHA-256"):
        export.check_made(onnx, "s", verify=True)


def archive(path: Path, members: dict[str, bytes]):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_the_source_is_unpacked_once_into_its_commit_directory(tmp_path):
    source = tmp_path / "code.tar.gz"
    archive(source, {f"{TOP}/engine/__init__.py": b"", f"{TOP}/configs/a.yml": b"a: 1\n"})
    code = export.unpack(source, tmp_path)
    assert code == tmp_path / TOP and (code / "configs" / "a.yml").read_text() == "a: 1\n"
    assert not (tmp_path / f"{TOP}.part").exists()
    source.unlink()
    assert export.unpack(source, tmp_path) == code  # there: the archive is not read again


def test_an_archive_without_the_code_or_with_paths_outside_is_refused(tmp_path):
    source = tmp_path / "code.tar.gz"
    archive(source, {f"{TOP}/README.md": b""})
    with pytest.raises(store.WeightsError, match="holds no"):
        export.unpack(source, tmp_path)
    archive(source, {f"{TOP}/engine/__init__.py": b"", "../outside.py": b""})
    with pytest.raises(tarfile.TarError):
        export.unpack(source, tmp_path / "root")
    assert not (tmp_path / "outside.py").exists()


def test_onnx_runtime_must_reproduce_pytorchs_best_detections():
    reference = outputs_for([*OBJECTS, ((0.2, 0.2, 0.3, 0.3), 39, 0.3), ((0.6, 0.6, 0.7, 0.7), 41, 0.3)], 640, 480)
    assert export.compare(reference, reference)["largest score difference"] == 0.0
    labels, boxes, scores = (a.copy() for a in reference)
    labels[[2, 3]], boxes[[2, 3]] = labels[[3, 2]], boxes[[3, 2]]  # two equal scores swapped: the order may differ
    scores[0] += 1e-4
    boxes[1] += 0.5
    differences = export.compare(reference, (labels, boxes, scores))
    assert differences["largest score difference"] == pytest.approx(1e-4, rel=0.01)
    assert differences["largest box difference px"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="is not in ONNX"):
        export.compare(reference, (np.where(labels == 0, 1, labels), boxes, scores))
    with pytest.raises(ValueError, match="box differs by 5.00 px"):
        export.compare(reference, (labels, reference[1] + np.array([5, 0, 0, 0], np.float32), scores))


def test_the_test_frame_is_fixed():
    frame = export.test_frame()
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
    assert (frame == export.test_frame()).all() and len(np.unique(frame.reshape(-1, 3), axis=0)) > 1000


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib: Python 3.11")
def test_the_export_extra_brings_every_package_the_child_imports():
    import tomllib

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    extra = {r.split(">")[0].split("<")[0].split("=")[0].strip() for r in pyproject["project"]["optional-dependencies"]
             ["export"]}
    assert set(export.EXPORT_MODULES.values()) - {"onnxruntime"} == extra
    assert "onnxruntime" in pyproject["project"]["dependencies"][1]  # a dependency of the node itself


# a process that makes rtdetrv4_s.onnx with a stand-in child, which writes its partial file and then waits
INTERRUPTED = """
import hashlib, sys
from pathlib import Path
from xblick import weights as store
from xblick_rtdetrv4 import export, weights
root = Path(sys.argv[1])
data = b"checkpoint"
(root / weights.checkpoint_name("s")).write_bytes(data)
(root / f"RT-DETRv4-{weights.COMMIT}" / "engine").mkdir(parents=True)  # unpacked already
weights.MODELS["s"] = ("id", len(data), hashlib.sha256(data).hexdigest())
code = (root / weights.code_name())
code.write_bytes(b"code")
weights.CODE = ("url", 4, hashlib.sha256(b"code").hexdigest())
export.missing_modules = lambda: []
CHILD = ("import sys, time; open(sys.argv[1] + '.part', 'w').write('x'); "
         "print('child: started', flush=True); time.sleep(60)")
export.child_command = lambda code, model, checkpoint, onnx: [sys.executable, "-c", CHILD, str(onnx)]
sys.exit(export.download_only(store.Store(weights.PROG, root, weights.TERMS), "s"))
"""


def test_sigterm_while_the_child_exports_ends_both_and_leaves_no_partial_file(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", INTERRUPTED, str(tmp_path)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    lines = []
    for line in process.stdout:
        lines.append(line)
        if "child: started" in line:
            break
    assert (tmp_path / "rtdetrv4_s.onnx.part").exists(), lines
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=20) == 143
    assert "stopped (SIGTERM) while making rtdetrv4_s.onnx; no partial file kept" in process.stderr.read()
    assert not list(tmp_path.glob("rtdetrv4_s.onnx*"))
