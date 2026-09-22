#!/usr/bin/env python3
"""Write reports/env/<date>-environment.md — the state of the real machine.

Generated, never hand-written: every field comes from the live environment
(CONTRACTS-P01 §4). The two items the design could not settle without an
installation are answered here in plain words:

    R17 — which cache received the backbone (``.hf``, ``.torch`` or both),
          the SHA-256 of every file received, and the source identifier.
    R18 — which class accepts ``pre_processor`` / ``post_processor``.

    .venv/bin/python tools/env_report.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import cv2
import torch
from anomalib.models.image.patchcore.lightning_model import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / ".hf"
TORCH_DIR = ROOT / ".torch"
REPORT = ROOT / "reports" / "env" / f"{date.today().isoformat()}-environment.md"


def pip_check() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip() or "no output"


def version_of(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def opencv_distributions() -> list[str]:
    names: list[str] = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"] or ""
        if name.lower().startswith("opencv"):
            names.append(name)
    return sorted(names)


def gui_backend_line(build_info: str) -> str | None:
    for line in build_info.splitlines():
        if line.strip().startswith("GUI:"):
            return line.strip()
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_inventory(cache_dir: Path) -> list[tuple[str, str, int]]:
    inventory: list[tuple[str, str, int]] = []
    if not cache_dir.is_dir():
        return inventory
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            inventory.append(
                (str(path.relative_to(ROOT)), sha256_of(path), path.stat().st_size)
            )
    return inventory


def hub_repo_ids(cache_dir: Path) -> list[str]:
    repos: list[str] = []
    if (cache_dir / "hub").is_dir():
        for blob_dir in sorted((cache_dir / "hub").glob("models--*")):
            repos.append(blob_dir.name.removeprefix("models--").replace("--", "/"))
    return repos


def r17_answer(hf: list[tuple[str, str, int]], torchc: list[tuple[str, str, int]]) -> str:
    which = (
        "`.hf`" if hf and not torchc
        else "`.torch`" if torchc and not hf
        else "both `.hf` and `.torch`" if hf and torchc
        else "neither — the caches are empty"
    )
    return which


def main() -> int:
    torch_sig = inspect.signature(PatchcoreModel.__init__)
    lightning_sig = inspect.signature(Patchcore.__init__)

    torch_params = list(torch_sig.parameters)
    lightning_params = list(lightning_sig.parameters)
    processors_owner = (
        "the Lightning wrapper `Patchcore`"
        if "pre_processor" in lightning_params and "post_processor" in lightning_params
        and "pre_processor" not in torch_params and "post_processor" not in torch_params
        else None
    )

    build_info = cv2.getBuildInformation()
    gui = gui_backend_line(build_info)

    cuda = torch.version.cuda or "none"
    available = torch.cuda.is_available()
    if available:
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_line = (
            f"`is_available() = True` — {device_name},"
            f" {total_memory / (1 << 30):.1f} GiB total"
        )
    else:
        device_name = "unavailable"
        total_memory = 0
        gpu_line = "`is_available() = False` — no usable CUDA device"

    hf_inventory = cache_inventory(HF_DIR)
    torch_inventory = cache_inventory(TORCH_DIR)
    received = r17_answer(hf_inventory, torch_inventory)
    sources = hub_repo_ids(HF_DIR)
    source_line = ", ".join(sources) if sources else "not exposed by the library"

    v4l2 = shutil.which("v4l2-ctl")
    tkinter_version = None
    import tkinter

    tkinter_version = tkinter.TkVersion

    distributions = opencv_distributions()
    one_headless = len(distributions) == 1 and distributions[0] == "opencv-python-headless"
    if one_headless:
        opencv_note = "exactly one, headless, as required"
    else:
        opencv_note = "NOT the single headless build required"
    joined = ", ".join(distributions) if distributions else "none"

    lines = [
        f"# Environment report — {date.today().isoformat()}",
        "",
        "Generated by `tools/env_report.py`; every field is read from the live environment.",
        "",
        "## Python and pip",
        "",
        f"- Python: {sys.version.split()[0]} (`{sys.executable}`)",
        f"- `pip check`: {pip_check()}",
        "",
        "## Torch and the GPU",
        "",
        f"- torch {version_of('torch')}, built with CUDA {cuda}",
        f"- {gpu_line}",
        "",
        "## Library versions",
        "",
        f"- torchvision {version_of('torchvision')}",
        f"- anomalib {version_of('anomalib')}",
        f"- lightning {version_of('lightning')}",
        f"- timm {version_of('timm')}",
        f"- numpy {version_of('numpy')}",
        f"- pillow {version_of('pillow')}",
        "",
        "## OpenCV",
        "",
        f"- version: {cv2.__version__}",
        f"- GUI backend: {gui if gui else 'none — headless build'}",
        f"- installed distributions: {joined} — {opencv_note}",
        "",
        "## Tk",
        "",
        f"- tkinter Tk version: {tkinter_version}",
        "",
        "## R17 — where the backbone lives",
        "",
        f"The backbone was received by {received}.",
        f"Source identifier: {source_line}.",
        "",
        "| cache | file | SHA-256 | bytes |",
        "|---|---|---|---|",
    ]
    for cache_name, inventory in ((".hf", hf_inventory), (".torch", torch_inventory)):
        for rel_path, digest, size in inventory:
            lines.append(f"| {cache_name} | `{rel_path}` | `{digest}` | {size} |")
    if not hf_inventory and not torch_inventory:
        lines.append("| — | the caches are empty — run `tools/prefetch_weights.py` | | |")

    lines += [
        "",
        "## R18 — who owns the processors",
        "",
        f"- `PatchcoreModel.__init__{torch_sig}`",
        f"- Lightning `Patchcore.__init__{lightning_sig}`",
    ]
    if processors_owner:
        lines += [
            "",
            f"Conclusion: `pre_processor` / `post_processor` are accepted by {processors_owner},"
            " not by the torch `PatchcoreModel` — using the torch model directly avoids them,"
            " as the design assumed.",
        ]
    else:
        lines += [
            "",
            "Conclusion: **contradicts the design** — the processors are NOT where"
            " `docs/15-model.md` §5 says they are. The planner corrects the design.",
        ]

    lines += [
        "",
        "## v4l2-ctl",
        "",
        f"- {v4l2 if v4l2 else 'not on the PATH — Ali installs v4l-utils (manual step)'}",
        "",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT.relative_to(ROOT)}")
    print(f"R17: received by {received}; source {source_line}")
    print(f"R18: processors accepted by {processors_owner or 'unresolved'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
