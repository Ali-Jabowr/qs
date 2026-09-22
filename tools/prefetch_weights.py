#!/usr/bin/env python3
"""Fetch the pretrained PatchCore backbone once, into the repo's own caches.

Sets ``HF_HOME=.hf`` and ``TORCH_HOME=.torch`` before anything imports torch,
timm or anomalib, then builds Anomalib's torch model exactly as the station
will (``wide_resnet50_2``, layers ``layer2`` and ``layer3``, pretrained) —
which downloads the backbone — and finally records which of the two caches
received files and the SHA-256 of every file received (review item R17).

The record goes to ``reports/env/weights-manifest.json`` and to stdout;
``tools/env_report.py`` reads the caches again when it writes the
environment report.

    .venv/bin/python tools/prefetch_weights.py

This is a setup-time tool: it is the one step allowed to open a network
connection (ADR-0016). Everything after it runs offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / ".hf"
TORCH_DIR = ROOT / ".torch"
MANIFEST = ROOT / "reports" / "env" / "weights-manifest.json"

os.environ["HF_HOME"] = str(HF_DIR)
os.environ["TORCH_HOME"] = str(TORCH_DIR)

import torch  # noqa: E402
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_cache(cache_dir: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(cache_dir.rglob("*")) if cache_dir.is_dir() else []:
        if path.is_file() and not path.is_symlink():
            files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_of(path),
                    "bytes": path.stat().st_size,
                }
            )
    return files


def hub_repo_ids(cache_dir: Path) -> list[str]:
    """HF cache layout encodes the source: models--<org>--<name>."""
    repos: list[str] = []
    if cache_dir.is_dir():
        for blob_dir in sorted((cache_dir / "hub").glob("models--*")):
            repos.append(blob_dir.name.removeprefix("models--").replace("--", "/"))
    return repos


def main() -> int:
    print(f"building PatchcoreModel backbone={BACKBONE} layers={LAYERS} pretrained ...")
    model = PatchcoreModel(layers=LAYERS, backbone=BACKBONE, pre_trained=True)

    probe = torch.zeros(1, 3, 256, 256)
    features = model.feature_extractor(probe)
    received = sorted(features)
    print(f"feature layers hooked: {received}")
    if sorted(LAYERS) != received:
        print(f"ERROR: expected layers {LAYERS}, got {received}", file=sys.stderr)
        return 1
    for name, tensor in features.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    manifest = {
        "date": date.today().isoformat(),
        "model": "PatchcoreModel",
        "backbone": BACKBONE,
        "layers": LAYERS,
        "caches": {
            ".hf": {
                "hub_repo_ids": hub_repo_ids(HF_DIR),
                "files": walk_cache(HF_DIR),
            },
            ".torch": {
                "hub_repo_ids": [],
                "files": walk_cache(TORCH_DIR),
            },
        },
    }

    hf_count = len(manifest["caches"][".hf"]["files"])
    torch_count = len(manifest["caches"][".torch"]["files"])
    if hf_count and not torch_count:
        received_by = ".hf"
    elif torch_count and not hf_count:
        received_by = ".torch"
    elif hf_count and torch_count:
        received_by = "both"
    else:
        received_by = "neither"

    print()
    print(
        f"R17 (first look): files received by .hf: {hf_count},"
        f" by .torch: {torch_count} -> {received_by}"
    )
    for cache in (".hf", ".torch"):
        for entry in manifest["caches"][cache]["files"]:
            print(f"  {cache}  {entry['path']}  sha256:{entry['sha256']}")
    if manifest["caches"][".hf"]["hub_repo_ids"]:
        print(f"  source: {', '.join(manifest['caches']['.hf']['hub_repo_ids'])}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest written to {MANIFEST.relative_to(ROOT)}")

    if hf_count == 0 and torch_count == 0:
        print(
            "ERROR: the caches received no files — the backbone was already cached elsewhere",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
