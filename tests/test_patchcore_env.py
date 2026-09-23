"""The real PatchCore module on the laptop: offline load, determinism, memory, one score.

Every test is marked ``env`` — it needs the RTX 3050 and the `.hf` weights cache, so
`make check` skips it and `make env-check` runs it. Sockets are blocked throughout: the
backbone must come from the repo cache (ADR-0016).
"""

from __future__ import annotations

import math
import socket
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.errors import ConfigError
from qs.inspector.model import BACKBONE, FitSettings
from qs.inspector.window import cut_window, to_input

pytestmark = pytest.mark.env

WINDOW_PX = 400
INPUT_PX = 256
GIB = 1 << 30


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Any socket connection attempt raises — nothing is downloaded at runtime."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise OSError("network disabled by test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield


def synthetic_windows(count: int, seed: int = 7) -> list[np.ndarray]:
    """`count` model inputs: pale noisy discs of varying size on a dark floor, windowed."""
    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(count):
        frame = np.full((1080, 1920, 3), 20, dtype=np.uint8)
        centre = (960 + int(rng.integers(-40, 41)), 540 + int(rng.integers(-40, 41)))
        cv2.circle(frame, centre, int(rng.integers(170, 200)), (200, 200, 200), -1)
        noise = rng.normal(0.0, 6.0, frame.shape)
        frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        window, _ = cut_window(frame, (float(centre[0]), float(centre[1])), WINDOW_PX)
        windows.append(to_input(window, INPUT_PX))
    return windows


@pytest.fixture(scope="module")
def windows() -> list[np.ndarray]:
    return synthetic_windows(60)


def test_the_backbone_loads_by_full_tag_with_the_hash_checked(no_network: None) -> None:
    from qs.inspector.patchcore import AnomalibPatchCore

    model = AnomalibPatchCore("cuda")
    assert model.backbone_tag == BACKBONE
    assert model.pretrained_tag == "racm_in1k"


def test_a_wrong_hash_is_refused_before_the_model_is_built(
    no_network: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qs.inspector import patchcore

    def never_built(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model was built before the weights were checked")

    monkeypatch.setattr(patchcore, "PatchcoreModel", never_built)
    other = tmp_path / "model.safetensors"
    other.write_bytes(b"other weights")
    with pytest.raises(ConfigError, match="^weights: .*not the pinned"):
        patchcore.AnomalibPatchCore("cuda", resolve=lambda: other)
    with pytest.raises(ConfigError, match="^weights: .*not in the local cache"):
        patchcore.AnomalibPatchCore("cuda", resolve=lambda: None)


def test_fitting_is_deterministic_by_seed_and_fits_the_card(
    no_network: None, windows: list[np.ndarray]
) -> None:
    import torch

    from qs.inspector.patchcore import AnomalibPatchCore

    model = AnomalibPatchCore("cuda")
    first = model.fit(windows, FitSettings(coreset_ratio=0.1, seed=42, device="cuda"))
    second = model.fit(windows, FitSettings(coreset_ratio=0.1, seed=42, device="cuda"))
    other = model.fit(windows, FitSettings(coreset_ratio=0.1, seed=43, device="cuda"))

    assert first.bank.shape == (int(60 * 1024 * 0.1), 1536) == (6144, 1536)
    assert first.bank.dtype == np.float32
    assert torch.equal(torch.from_numpy(first.bank), torch.from_numpy(second.bank))
    assert not torch.equal(torch.from_numpy(first.bank), torch.from_numpy(other.bank))
    peaks = [f.peak_gpu_bytes for f in (first, second, other)]
    assert all(p is not None and 0 < p < 3.5 * GIB for p in peaks)
    print(f"peak GPU memory, 60 windows: {max(p or 0 for p in peaks) / 2**20:.0f} MiB")


def test_scoring_one_window(no_network: None, windows: list[np.ndarray]) -> None:
    from qs.inspector.patchcore import AnomalibPatchCore

    model = AnomalibPatchCore("cuda")
    fitted = model.fit(windows[:8], FitSettings(coreset_ratio=0.1, seed=42, device="cuda"))
    scored = model.scorer(fitted.bank).score(synthetic_windows(1, seed=99)[0])

    assert scored.anomaly_map.shape == (INPUT_PX, INPUT_PX)
    assert scored.anomaly_map.dtype == np.float32
    assert np.isfinite(scored.anomaly_map).all()
    assert math.isfinite(scored.pred_score)
    assert scored.backbone_ms > 0
    assert scored.search_ms > 0
    print(f"backbone {scored.backbone_ms:.1f} ms · search {scored.search_ms:.1f} ms")


def test_the_same_window_scores_the_same_twice(
    no_network: None, windows: list[np.ndarray]
) -> None:
    from qs.inspector.patchcore import AnomalibPatchCore

    model = AnomalibPatchCore("cuda")
    fitted = model.fit(windows[:8], FitSettings(coreset_ratio=0.1, seed=42, device="cuda"))
    scorer = model.scorer(fitted.bank)
    probe = synthetic_windows(1, seed=5)[0]
    first, second = scorer.score(probe), scorer.score(probe)
    assert np.array_equal(first.anomaly_map, second.anomaly_map)
    assert first.pred_score == second.pred_score
