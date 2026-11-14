"""The node's weights: its list of files, and --download-only, --no-download and a start with the model missing,
against a local HTTP server (no internet) and with a stand-in for the export's child process (no PyTorch). The
download itself (locks, checks, interrupts) is xblick.weights' and tested in the core."""

import contextlib
import hashlib
import http.server
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest
from xblick import paths
from xblick import weights as store
from xblick.session import open_session

from xblick_rtdetrv4 import export, node, weights

DATA = os.urandom(3 * 1024 * 1024 + 17)  # a stand-in checkpoint
DATA_SHA = hashlib.sha256(DATA).hexdigest()


def code_archive() -> bytes:
    """A stand-in of RT-DETRv4's source archive: its top directory with an engine package."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(f"RT-DETRv4-{weights.COMMIT}/engine/__init__.py")
        tar.addfile(info, io.BytesIO(b""))
    return buffer.getvalue()


CODE = code_archive()


@pytest.fixture
def server():
    """A local HTTP server: the source archive at /code.tar.gz, DATA at every other path; records each GET."""
    gets = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            gets.append(self.path)
            body = CODE if self.path == "/code.tar.gz" else DATA
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", gets
    httpd.shutdown()
    httpd.server_close()


# the export's child replaced: it writes an ONNX file and its record as the real one does, from the checkpoint's bytes
CHILD = """
import hashlib, json, sys
from pathlib import Path
onnx, checkpoint, sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = b"onnx of " + checkpoint.read_bytes()[:16]
Path(str(onnx) + ".json").write_text(json.dumps({"checkpoint_sha256": sha, "size": len(data),
                                                 "sha256": hashlib.sha256(data).hexdigest()}))
onnx.write_bytes(data)
print("child: made", onnx.name)
"""


@pytest.fixture
def home(tmp_path, monkeypatch, server):
    """The node's STORE in tmp_path, XBLICK_HOME too (nothing reaches this PC's cache); every checkpoint a stand-in on
    the local server, the source archive too; the export's child a stand-in. Returns the store's root and the
    child's calls."""
    root = tmp_path / "weights"
    monkeypatch.setenv("XBLICK_HOME", str(tmp_path / "xblick-home"))
    monkeypatch.setattr(weights, "STORE", store.Store(weights.PROG, root, weights.TERMS, (), repo=tmp_path))
    url, _ = server
    monkeypatch.setattr(weights, "DRIVE", f"{url}/drive/{{}}")
    for name in weights.MODELS:
        monkeypatch.setitem(weights.MODELS, name, (f"id-{name}", len(DATA), DATA_SHA))
    monkeypatch.setattr(weights, "CODE", (f"{url}/code.tar.gz", len(CODE), hashlib.sha256(CODE).hexdigest()))
    monkeypatch.setattr(export, "missing_modules", lambda: [])
    calls = []

    def child(code, model, checkpoint, onnx):
        calls.append((code, model))
        return [sys.executable, "-c", CHILD, str(onnx), str(checkpoint), weights.MODELS[model][2]]

    monkeypatch.setattr(export, "child_command", child)
    return root, calls


def test_the_files_are_the_checkpoints_rtdetrv4s_readme_links_and_its_source_at_a_fixed_commit():
    drive = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
    assert {name: [(w.path.name, w.source.url, w.size, w.sha256) for w in weights.needed(name)][0]
            for name in weights.MODELS} == {  # measured from the downloaded files, 2026-11-14
        "s": ("RTv4-S-hgnet.pth", drive.format("1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z"), 169_278_400,
              "238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee"),
        "m": ("RTv4-M-hgnet.pth", drive.format("1O-YpP4X-quuOXbi96y2TKkztbjroP5mX"), 317_911_052,
              "8f1bd54437e9a227dc41ff3f65986f15097a4aab27e6a4e944c845e441f27501"),
        "l": ("RTv4-L-hgnet.pth", drive.format("1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8"), 505_400_981,
              "935c691dbd4f282a6a4ba59a57d77f13ad6963411dbb1b29615a72af2452f6e1"),
        "x": ("RTv4-X-hgnet.pth", drive.format("19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3"), 1_010_474_984,
              "2925ec2e53d48e6141db8601c8656566219e696cd63bf8d6955190cae6bfd3f9")}
    code = weights.needed("s")[1]
    assert (code.path, code.source.url, code.size, code.sha256) == (
        Path("RT-DETRv4-55fefaaed7ef.tar.gz"),
        "https://github.com/AIMIA-PKU/RT-DETRv4/archive/55fefaaed7efe2a5f72d0a18fd4e05965e35c292.tar.gz", 5_047_744,
        "a12773e0038da92539ba87bcb3ecd3891ee7912b4cbd015db85cc9e90ad3ffd6")
    assert weights.needed("x")[1] == code  # one source for every model
    assert weights.config_of("m") == "configs/rtv4/rtv4_hgnetv2_m_coco.yml"


def test_the_store_is_the_cache_directory(tmp_path):
    assert weights.STORE.root == paths.cache_dir("rtdetrv4") and weights.STORE.node == weights.NODE == "rtdetrv4"
    assert weights.STORE.former == ()  # a new node: no former place
    assert weights.path_of("s") == paths.cache_dir("rtdetrv4") / "rtdetrv4_s.onnx"
    env = {**os.environ, "XBLICK_HOME": str(tmp_path)}  # moves it, and asking creates nothing
    root = subprocess.run([sys.executable, "-c", "from xblick_rtdetrv4.weights import STORE; print(STORE.root)"],
                          env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert root == str(tmp_path / "cache" / "rtdetrv4") and not list(tmp_path.iterdir())


def test_weights_dir_is_a_read_only_parameter_and_download_only_none():
    from xblick.parameters import Parameters

    parser = node.make_parser()
    args = store.parse_args(parser, ["--camera", "test/camera", "--weights-dir", "/w"])
    params = Parameters(None, "rtdetrv4", "test/camera")
    params.configure(args)
    params.declare_options(parser, args, live=node.LIVE, leave_out=("camera", "node", "download_only"))
    assert params.descriptor("weights_dir").read_only and params["weights_dir"] == "/w"
    assert "download_only" not in params.names()


def test_download_only_downloads_and_makes_the_model_option_and_then_has_nothing_to_do(home, server, capsys):
    root, calls = home
    _, gets = server
    assert node.main(["--download-only", "--model", "m"]) == 0
    out = capsys.readouterr().out
    assert (root / "RTv4-M-hgnet.pth").read_bytes() == DATA and not (root / "RTv4-S-hgnet.pth").exists()
    assert (root / "rtdetrv4_m.onnx").read_bytes() == b"onnx of " + DATA[:16]
    assert (root / f"RT-DETRv4-{weights.COMMIT}" / "engine").is_dir()
    assert calls == [(root / f"RT-DETRv4-{weights.COMMIT}", "m")]
    assert "RT-DETRv4 is the work of its authors" in out and "DINOv3 License" in out
    assert "making rtdetrv4_m.onnx" in out and "rtdetrv4_m.onnx ready in" in out
    assert sorted(gets) == ["/code.tar.gz", "/drive/id-m"]
    assert node.main(["--download-only", "--model", "m"]) == 0 and len(gets) == 2 and len(calls) == 1
    assert "rtdetrv4_m.onnx ready" in capsys.readouterr().out


def test_a_made_onnx_file_needs_neither_the_checkpoint_nor_a_download(home, server, capsys):
    root, calls = home
    _, gets = server
    assert node.main(["--download-only"]) == 0
    (root / "RTv4-S-hgnet.pth").unlink()
    assert export.ensure_onnx(weights.STORE, "s", download=False) == root / "rtdetrv4_s.onnx"
    assert len(gets) == 2 and len(calls) == 1


def test_weights_dir_is_where_the_files_go(home, tmp_path, capsys):
    root, _ = home
    there = tmp_path / "elsewhere"
    assert node.main(["--download-only", "--weights-dir", str(there)]) == 0
    assert (there / "rtdetrv4_s.onnx").is_file() and (there / "RTv4-S-hgnet.pth").is_file() and not root.exists()
    with pytest.raises(SystemExit) as exit_:
        node.main(["--download-only", "--weights-dir", "${XBLICK_NO_SUCH_VARIABLE}/w"])
    assert exit_.value.code == 2 and "--weights-dir" in capsys.readouterr().err


def test_download_only_refuses_a_changed_checkpoint(home, monkeypatch, capsys):
    root, calls = home
    monkeypatch.setitem(weights.MODELS, "s", ("id-s", len(DATA), "0" * 64))
    assert node.main(["--download-only"]) == 1
    assert "wrong SHA-256" in capsys.readouterr().err and not (root / "RTv4-S-hgnet.pth").exists() and calls == []


def test_without_the_export_packages_the_line_names_them_and_the_setup(home, monkeypatch, capsys):
    root, calls = home
    monkeypatch.setattr(export, "missing_modules", lambda: ["torch", "onnx"])
    assert node.main(["--download-only"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "needs torch, onnx: run tools/setup.sh" in err[0], err
    assert "xblick-rtdetrv4 --download-only --model s" in err[0] and calls == []
    assert (root / "RTv4-S-hgnet.pth").is_file()  # downloaded: the export can start without a download later


def test_a_failed_export_leaves_no_onnx_file(home, monkeypatch, capsys):
    root, _ = home
    monkeypatch.setattr(export, "child_command", lambda *args: [sys.executable, "-c", "raise SystemExit(3)"])
    assert node.main(["--download-only"]) == 1
    assert "the export of rtdetrv4_s.onnx failed (exit code 3" in capsys.readouterr().err
    assert not (root / "rtdetrv4_s.onnx").exists()


def test_an_onnx_file_of_another_checkpoint_is_refused(home, capsys):
    root, _ = home
    assert node.main(["--download-only"]) == 0
    record = export.record_of(root / "rtdetrv4_s.onnx")
    record.write_text(json.dumps(json.loads(record.read_text()) | {"checkpoint_sha256": "1" * 64}))
    assert node.main(["--download-only"]) == 1
    assert "made from another checkpoint" in capsys.readouterr().err


class Started(Exception):
    """The node got past its weights."""


@pytest.fixture
def start(monkeypatch):
    """A start's options with a Zenoh session to connect to; the model replaced by raising Started."""
    def stop(*args):
        raise Started(args)

    monkeypatch.setattr(node, "load_detector", stop)
    kept = []  # the node's parameters and session: a start that raised Started left them open

    class Kept(node.node.Parameters):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            kept.append(self)

    monkeypatch.setattr(node.node, "Parameters", Kept)
    opened = node.open_from_args
    monkeypatch.setattr(node, "open_from_args", lambda args: kept.append(opened(args)) or kept[-1])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    sigterm = signal.getsignal(signal.SIGTERM)  # node.start sets its own
    yield ["--camera", "test/camera", "--connect", f"127.0.0.1:{port}"]
    for closable in kept:
        with contextlib.suppress(Exception):  # closed already by a node that ended
            closable.close()
    signal.signal(signal.SIGTERM, sigterm)
    session.close()


def test_no_download_with_the_model_missing_ends_with_one_line(home, server, start, capsys):
    root, calls = home
    _, gets = server
    assert node.main([*start, "--no-download"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "RTv4-S-hgnet.pth" in err[0], err
    assert "xblick-rtdetrv4 --download-only --model s" in err[0], err
    assert gets == [] and calls == [] and not (root / "rtdetrv4_s.onnx").exists()


def test_a_start_with_the_model_missing_downloads_and_makes_it_first(home, server, start, capsys):
    root, _ = home
    with pytest.raises(Started) as started:
        node.main([*start, "--model", "l", "--device", "cuda:1", "--threads", "2"])
    assert started.value.args[0] == (root / "rtdetrv4_l.onnx", 2, "cuda:1")
    assert "RT-DETRv4 is the work of its authors" in capsys.readouterr().out
