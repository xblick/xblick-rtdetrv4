"""The node's weights: which files it reads, where they come from and what it makes of them.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-rtdetrv4
--download-only``) both use it. ``xblick.weights`` downloads and checks the files, ``export.py`` makes the ONNX file.

* ``MODELS``: name -> (Google Drive file id, size in bytes, SHA-256) of the checkpoint RT-DETRv4's README links for
  that model, in ``RTv4-<NAME>-hgnet.pth`` of the node's cache directory (``~/.cache/xblick/rtdetrv4``,
  ``--weights-dir``). Size and SHA-256 were measured from the files downloaded on 2026-11-14. The README states no
  checksums. The file names are those Google Drive gives.
* ``CODE``: RT-DETRv4's source at the fixed commit ``COMMIT``, as GitHub's archive of it (size and SHA-256 measured
  on 2026-11-14). The export imports its model code. The node never runs it while it detects.
* ``needed(model)``: the files the export of ``--model`` reads, their paths relative to the store's root;
  ``onnx_name(model)``: the ONNX file the node reads, made from them on this PC
* ``store_for(parser, args)``: ``STORE`` at ``--weights-dir``; ``options(args)``: the options that select the weights,
  for the command the lines name
* ``STORE``: the download (``xblick.weights.Store``), its root the node's cache directory
"""

from __future__ import annotations

from pathlib import Path

from xblick import weights

PROG = "xblick-rtdetrv4"
NODE = "rtdetrv4"  # the short name: its cache directory is ~/.cache/xblick/rtdetrv4
FORMER_GPU_RUNTIME = []  # earlier places of a GPU runtime of this node: none (it never had its own)
REPO = Path(__file__).resolve().parents[2]
UPSTREAM = "https://github.com/AIMIA-PKU/RT-DETRv4"
COMMIT = "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"  # main on 2026-11-14
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"  # no page, the file itself
MODELS = {  # name -> (Google Drive id, size, SHA-256): into <root>/RTv4-<NAME>-hgnet.pth
    "s": ("1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z", 169_278_400,
          "238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee"),
    "m": ("1O-YpP4X-quuOXbi96y2TKkztbjroP5mX", 317_911_052,
          "8f1bd54437e9a227dc41ff3f65986f15097a4aab27e6a4e944c845e441f27501"),
    "l": ("1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8", 505_400_981,
          "935c691dbd4f282a6a4ba59a57d77f13ad6963411dbb1b29615a72af2452f6e1"),
    "x": ("19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3", 1_010_474_984,
          "2925ec2e53d48e6141db8601c8656566219e696cd63bf8d6955190cae6bfd3f9"),
}
CODE = (f"{UPSTREAM}/archive/{COMMIT}.tar.gz", 5_047_744,
        "a12773e0038da92539ba87bcb3ecd3891ee7912b4cbd015db85cc9e90ad3ffd6")
TERMS = """\
RT-DETRv4 is the work of its authors (Liao et al., 2025), under their terms, and you are the one who uses it:
  * RT-DETRv4's repository (github.com/AIMIA-PKU/RT-DETRv4) is under the Apache License 2.0, as its LICENSE states.
    Its code builds on RT-DETR, D-FINE and DEIM, whose files keep their authors' copyright lines.
  * The checkpoints, which its README links on Google Drive, come with no terms of their own.
  * The authors trained them on COCO, with Meta's DINOv3 (ViT-B/16) as the teacher whose features the detector
    learned to match. DINOv3 is under the DINOv3 License (github.com/facebookresearch/dinov3). The node neither
    downloads nor runs DINOv3. Whether its licence extends to the trained detectors is not stated by either party.
  * The node makes the ONNX file from the checkpoint on this PC with RT-DETRv4's own model code (downloaded as
    GitHub's archive of a fixed commit)."""

# the download (xblick.weights): the root is the node's cache directory, its lock there
STORE = weights.Store(PROG, node=NODE, terms=TERMS, repo=REPO)


def checkpoint_name(model: str) -> Path:
    """The checkpoint of ``--model``, relative to a store's root (the name Google Drive gives it)."""
    return Path(f"RTv4-{model.upper()}-hgnet.pth")


def code_name() -> Path:
    """RT-DETRv4's source archive, relative to a store's root."""
    return Path(f"RT-DETRv4-{COMMIT[:12]}.tar.gz")


def onnx_name(model: str) -> Path:
    """The ONNX file of ``--model`` that this PC makes, relative to a store's root."""
    return Path(f"rtdetrv4_{model}.onnx")


def config_of(model: str) -> str:
    """The model's config in RT-DETRv4's source, relative to its top directory."""
    return f"configs/rtv4/rtv4_hgnetv2_{model}_coco.yml"


def needed(model: str) -> list[weights.Weight]:
    """The downloads that the ONNX file of ``--model`` (a name of MODELS) is made from: the checkpoint and the code."""
    drive_id, size, sha256 = MODELS[model]
    url, code_size, code_sha256 = CODE
    return [weights.Weight(checkpoint_name(model), weights.Url(DRIVE.format(drive_id)), size, sha256),
            weights.Weight(code_name(), weights.Url(url), code_size, code_sha256)]


def path_of(model: str, store: weights.Store | None = None) -> Path:
    """Where the ONNX file of ``--model`` is in ``store`` (default STORE: the node's cache directory)."""
    return (store or STORE).root / onnx_name(model)


def options(args) -> list[str]:
    """The options that select the weights (and --weights-dir), for the download command the lines name."""
    return ["--model", args.model] + (["--weights-dir", args.weights_dir] if args.weights_dir else [])


def store_for(parser, args) -> weights.Store:
    """STORE at --weights-dir (not given: the node's cache directory); a wrong --weights-dir is an argument error."""
    try:
        return STORE.at(args.weights_dir)
    except ValueError as error:
        parser.error(f"--weights-dir: {error}")
