"""The real-environment tests — GPU, caches, offline load. Run by `make env-check`.

Every test here is marked ``env``: it needs the venv, the RTX 3050 and the
prefetched weights cache, so it is excluded from `make check`. The imports of
torch and anomalib live inside the tests for the same reason.

Test 4 proves the station's real start-up path: the backbone loads from the
repo caches with the network off (ADR-0016). Test 5 proves test 4 used the
caches — pointed at an empty directory, the same load must fail.
"""

from __future__ import annotations

import importlib.metadata
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.env

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / ".hf"
TORCH_DIR = ROOT / ".torch"


def build_pretrained_patchcore() -> list[str]:
    """Build the torch PatchCore model exactly as the station will, pretrained."""
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    model = PatchcoreModel(
        layers=["layer2", "layer3"],
        backbone="wide_resnet50_2",
        pre_trained=True,
    )
    features = model.feature_extractor(torch.zeros(1, 3, 256, 256))
    return sorted(features)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Any socket connection attempt raises — the station never opens one."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise OSError("network disabled by test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield


def point_caches(monkeypatch: pytest.MonkeyPatch, hf_home: Path, torch_home: Path) -> None:
    """Point HF_HOME / TORCH_HOME at the given dirs, for real.

    huggingface_hub and torch.hub bind their cache locations at import time,
    so the module globals are patched too, not only the environment variables.
    """
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    import huggingface_hub.constants as hub_constants
    import huggingface_hub.file_download as hub_file_download
    import torch.hub

    hub_cache = str(hf_home / "hub")
    monkeypatch.setattr(hub_constants, "HF_HOME", str(hf_home), raising=False)
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", hub_cache, raising=False)
    monkeypatch.setattr(hub_constants, "HUGGINGFACE_HUB_CACHE", hub_cache, raising=False)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.setattr(hub_file_download, "HF_HUB_CACHE", hub_cache, raising=False)
    monkeypatch.setattr(hub_file_download, "HUGGINGFACE_HUB_CACHE", hub_cache, raising=False)
    monkeypatch.setattr(torch.hub, "_hub_dir", str(torch_home / "hub"), raising=False)


def test_gpu_is_available_and_is_the_3050() -> None:
    import torch

    assert torch.cuda.is_available()
    assert "3050" in torch.cuda.get_device_name(0)


def test_exactly_one_opencv_distribution_and_it_is_headless() -> None:
    names = sorted(
        (dist.metadata["Name"] or "").lower()
        for dist in importlib.metadata.distributions()
        if (dist.metadata["Name"] or "").lower().startswith("opencv")
    )
    assert names == ["opencv-python-headless"]


def test_tkinter_works_and_tk_is_86_or_newer() -> None:
    import tkinter

    assert tkinter.TkVersion >= 8.6


def test_offline_backbone_load_from_the_repo_caches(no_network, monkeypatch) -> None:
    """With sockets blocked and the repo caches in place, the build succeeds."""
    point_caches(monkeypatch, HF_DIR, TORCH_DIR)
    assert build_pretrained_patchcore() == ["layer2", "layer3"]


def test_the_same_load_fails_with_empty_caches(no_network, monkeypatch, tmp_path) -> None:
    """Empty caches and the network off — the build must fail, proving test 4
    really loaded from the cache."""
    point_caches(monkeypatch, tmp_path / "hf", tmp_path / "torch")
    with pytest.raises((OSError, ValueError, RuntimeError)):
        build_pretrained_patchcore()
