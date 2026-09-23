"""`AppearanceInspector` on the real backbone: NFR-03, a blot, and the CUDA timing.

Every test is marked ``env`` — it needs the RTX 3050 and the `.hf` weights cache, so
`make check` skips it and `make env-check` runs it. Sockets are blocked throughout: the
backbone must come from the repo cache (ADR-0016). A real bundle is trained once per module
through `qs train`, on 30 synthetic good disc frames (10 units × 3 rotations), into
`tmp_path`; the example config is rewritten to name it. The example's threshold means
nothing for these scores, so only the scores are compared, never the labels.
"""

from __future__ import annotations

import socket
import statistics
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from appearance_fixtures import config_naming, frame_of
from conftest import EXAMPLE
from qs.__main__ import build_parser
from qs.cmd import train
from qs.contracts import CHECK_APPEARANCE, Verdict
from qs.inspector.appearance import AppearanceInspector

pytestmark = pytest.mark.env

WINDOW_PX = 400
UNITS = 10
ROTATIONS = 3
FRAMES = 20


@pytest.fixture(scope="module")
def no_network() -> Iterator[None]:
    """Any socket connection attempt raises — nothing is downloaded at runtime."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise OSError("network disabled by test")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(socket.socket, "connect", blocked)
        patched.setattr(socket, "create_connection", blocked)
        yield


BLOT_OFFSET = (60, -40)


def disc_frame(seed: int, *, blot: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    """A pale noisy disc of radius 180–199 on the dark floor, near the frame's centre, and
    its centre; with `blot`, a dark round stain of radius 30 on the unit at `BLOT_OFFSET`."""
    rng = np.random.default_rng(seed)
    frame = np.full((1080, 1920, 3), 20, dtype=np.uint8)
    centre = (960 + int(rng.integers(-40, 41)), 540 + int(rng.integers(-40, 41)))
    cv2.circle(frame, centre, int(rng.integers(180, 200)), (200, 200, 200), -1)
    if blot:
        blot_at = (centre[0] + BLOT_OFFSET[0], centre[1] + BLOT_OFFSET[1])
        cv2.circle(frame, blot_at, 30, (60, 60, 60), -1)
    noise = rng.normal(0.0, 6.0, frame.shape)
    return np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8), centre


def good_disc(seed: int) -> np.ndarray:
    """A good disc frame."""
    return disc_frame(seed)[0]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("appearance")


@pytest.fixture(scope="module")
def model_id(workdir: Path, no_network: None) -> str:
    """A real bundle, trained through `qs train` on 30 good frames."""
    good = workdir / "dataset" / "train" / "good"
    good.mkdir(parents=True)
    for unit in range(1, UNITS + 1):
        for rotation in range(ROTATIONS):
            pixels = good_disc(1000 + 10 * unit + rotation)
            cv2.imwrite(str(good / f"u{unit:03d}_good_G2_r{rotation}.png"), pixels)
    args = build_parser().parse_args([
        "train", "--config", str(EXAMPLE), "--dataset", str(workdir / "dataset"),
        "--window-px", str(WINDOW_PX), "--product", "mochi",
        "--models", str(workdir / "models"),
    ])
    assert train.main(args) == 0
    [directory] = (workdir / "models").iterdir()
    return directory.name


@pytest.fixture(scope="module")
def inspector(workdir: Path, model_id: str, no_network: None) -> AppearanceInspector:
    loaded = config_naming(workdir, model_id)
    assert loaded.config.inspector.backend == "torch-cuda"
    inspector = AppearanceInspector(loaded.config_hash, workdir / "models")
    inspector.load(loaded.config)
    assert inspector.model_id == model_id
    return inspector


def _score(verdict: Verdict) -> float:
    [appearance] = [c for c in verdict.checks if c.name == CHECK_APPEARANCE]
    assert appearance.value is not None
    return appearance.value


def test_the_same_frames_score_the_same_twice_nfr_03(inspector: AppearanceInspector) -> None:
    frames = [frame_of(good_disc(seed), seq=seed) for seed in range(FRAMES)]
    first = [inspector.inspect(frame) for frame in frames]
    second = [inspector.inspect(frame) for frame in frames]

    for a, b in zip(first, second, strict=True):
        assert abs(_score(a) - _score(b)) <= 1e-6
        assert a.checks[0] == b.checks[0]
        assert a.checks[0].passed is True  # every good disc is inside the example tolerances
    median_ms = statistics.median(v.inspect_ms for v in second)
    scores = [_score(v) for v in first]
    print(
        f"\nCUDA inspect_ms over {FRAMES} frames: median {median_ms:.1f} ms "
        f"(min {min(v.inspect_ms for v in second):.1f}, "
        f"max {max(v.inspect_ms for v in second):.1f}) · good scores "
        f"{min(scores):.4f} … {max(scores):.4f}"
    )


def test_a_dark_blot_scores_above_every_good_disc(inspector: AppearanceInspector) -> None:
    good = [_score(inspector.inspect(frame_of(good_disc(seed)))) for seed in range(FRAMES)]
    pixels, centre = disc_frame(500, blot=True)
    blotted = inspector.inspect(frame_of(pixels))
    print(f"\nblot {_score(blotted):.4f} against the highest good disc {max(good):.4f}")
    assert _score(blotted) > max(good)
    assert blotted.heatmap is not None
    assert blotted.window_box is not None
    # the hottest pixel of the heatmap is on the blot, in window coordinates
    y, x = np.unravel_index(int(np.argmax(blotted.heatmap)), blotted.heatmap.shape)
    x0, y0 = blotted.window_box[:2]
    blot_x, blot_y = centre[0] + BLOT_OFFSET[0] - x0, centre[1] + BLOT_OFFSET[1] - y0
    assert (x - blot_x) ** 2 + (y - blot_y) ** 2 <= 60**2
