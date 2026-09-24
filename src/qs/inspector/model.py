"""The PatchCore seam: what the station asks of a model, and the pins it is held to.

Everything here is torch-free, so the commands and their tests can depend on it without
loading a model. `qs.inspector.patchcore` is the real implementation — the one module that
imports Anomalib (ADR-0015); tests pass a fake numpy one.

- **Pins** (docs/15 §5, ADR-0020): the backbone by its full timm tag, the SHA-256 of its
  cached `model.safetensors`, the layers and the neighbours.
- **The weights check** (ADR-0016, ADR-0020): `load_backbone` resolves the cached weights
  file with no network call, compares its hash with the pin and only then builds the model.
- **The device**: `device_for` maps `inspector.backend` to a torch device and never falls
  back from CUDA to the CPU (docs/16 §3).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from qs.errors import ConfigError

BACKBONE = "wide_resnet50_2.racm_in1k"
BACKBONE_SHA256 = "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000"
HUB_REPO = "models--timm--wide_resnet50_2.racm_in1k"  # timm/wide_resnet50_2.racm_in1k
WEIGHTS_FILE = "model.safetensors"
LAYERS = ("layer2", "layer3")
NUM_NEIGHBORS = 9
NORMALISATION_MEAN = (0.485, 0.456, 0.406)  # ImageNet, as `window.to_input` applies it
NORMALISATION_STD = (0.229, 0.224, 0.225)

_BACKENDS = {"torch-cuda": "cuda", "torch-cpu": "cpu"}


@dataclass(frozen=True, slots=True)
class FitSettings:
    """How one memory bank is fitted: the coreset share, the seed and the torch device."""

    coreset_ratio: float
    seed: int
    device: str


@dataclass(frozen=True, slots=True)
class Fitted:
    """A fitted memory bank (rows × features, float32) and the peak GPU memory, if on CUDA."""

    bank: np.ndarray
    peak_gpu_bytes: int | None


@dataclass(frozen=True, slots=True)
class Scored:
    """One window scored: the anomaly map at `input_px`, PatchCore's own image score, and
    the backbone and search timings in milliseconds."""

    anomaly_map: np.ndarray
    pred_score: float
    backbone_ms: float
    search_ms: float


class Scorer(Protocol):
    """Scores one `window.to_input` array (CHW float32) against a fixed memory bank."""

    def score(self, window_input: np.ndarray) -> Scored: ...


class PatchCore(Protocol):
    """The model seam: fit windows into a bank, and build a scorer from a bank."""

    def fit(self, windows: Sequence[np.ndarray], settings: FitSettings) -> Fitted: ...

    def scorer(self, bank: np.ndarray) -> Scorer: ...


def device_for(backend: str, *, cuda_available: bool) -> str:
    """The torch device for `inspector.backend`; `torch-cuda` without CUDA is refused."""
    if backend not in _BACKENDS:
        raise ConfigError(f"inspector.backend: {backend!r} is not one of {sorted(_BACKENDS)}")
    if backend == "torch-cuda" and not cuda_available:
        raise ConfigError(
            "inspector.backend: torch-cuda, but CUDA is not available — no CPU fallback "
            "(docs/16 §3)"
        )
    return _BACKENDS[backend]


def cached_weights(hub_dir: Path) -> Path | None:
    """The backbone's cached `model.safetensors` under `hub_dir`, by `refs/main`; no network."""
    repo = hub_dir / HUB_REPO
    try:
        revision = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    path = repo / "snapshots" / revision / WEIGHTS_FILE
    return path if revision and path.is_file() else None


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_backbone[T](resolve: Callable[[], Path | None], build: Callable[[], T]) -> T:
    """Check the cached backbone weights against the pin, then — and only then — build.

    A missing file or a different hash is a `ConfigError`: the station refuses to start
    rather than run on weights nobody pinned (ADR-0020).
    """
    path = resolve()
    if path is None or not path.is_file():
        raise ConfigError(
            f"weights: {BACKBONE} {WEIGHTS_FILE} is not in the local cache "
            f"({path or 'unresolved'}) — run tools/prefetch_weights.py once, online (ADR-0016)"
        )
    digest = sha256_of(path)
    if digest != BACKBONE_SHA256:
        raise ConfigError(
            f"weights: {path} has SHA-256 {digest[:12]}…, not the pinned "
            f"{BACKBONE_SHA256[:12]}… of {BACKBONE} (ADR-0020)"
        )
    return build()
