"""Shared fixtures for the Inspector's synthetic frames (docs/15 §2).

Pale discs, ellipses and blobs — and bright starch specks — drawn in memory on a black
1920×1080 BGR floor, against the example config's `inspector.localize`. Nothing here needs
a camera, a display or a GPU, and nothing is written anywhere.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from qs.config import GeometryConfig, LocalizeConfig, load_config
from qs.contracts import CameraReadback, TriggerEvent
from qs.inspector.model import FitSettings, Fitted, Scored
from qs.screen import KeyPress
from qs.session import (
    FrameLog,
    Product,
    SessionRecord,
    save_frame,
    write_plan,
    write_session,
)
from qs.splitplan import Plan, Shot, make_plan, shot_list, substitute_spares

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"

FRAME_SHAPE = (1080, 1920)
FLOOR_GREY = 20  # the black floor of the acceptance frames
UNIT_GREY = 200  # a pale unit on it
DARK_GREY = 30  # the dark filling of a tear, seen from above
SPECK_GREY = 255


def _bgr(grey: int) -> tuple[int, int, int]:
    return (grey, grey, grey)


def _clear_of(
    box: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    avoid: tuple[tuple[int, int, int], ...],
) -> bool:
    """Whether a speck's box clears the placed specks by 8 px and every avoid circle by 12 px."""
    x0, y0, x1, y1 = box
    for ox0, oy0, ox1, oy1 in placed:
        if x0 - 8 <= ox1 and ox0 <= x1 + 8 and y0 - 8 <= oy1 and oy0 <= y1 + 8:
            return False
    for cx, cy, radius in avoid:
        near_x = min(max(x0, cx), x1)
        near_y = min(max(y0, cy), y1)
        if (cx - near_x) ** 2 + (cy - near_y) ** 2 <= (radius + 12) ** 2:
            return False
    return True


@dataclass(frozen=True, slots=True)
class Paint:
    """Drawing primitives for synthetic frames: pale shapes on a black floor, in place."""

    def floor(self, grey: int = FLOOR_GREY) -> np.ndarray:
        """A black 1920×1080 BGR floor at `grey`."""
        return np.full(FRAME_SHAPE + (3,), grey, dtype=np.uint8)

    def disc(
        self,
        pixels: np.ndarray,
        centre: tuple[int, int],
        radius: int,
        grey: int = UNIT_GREY,
    ) -> None:
        """A filled disc."""
        cv2.circle(pixels, centre, radius, _bgr(grey), -1)

    def ellipse(
        self,
        pixels: np.ndarray,
        centre: tuple[int, int],
        axes: tuple[int, int],
        angle: int,
        grey: int = UNIT_GREY,
    ) -> None:
        """A filled ellipse: `axes` the half-sizes, `angle` the rotation in degrees."""
        cv2.ellipse(pixels, centre, axes, angle, 0, 360, _bgr(grey), -1)

    def wedge_disc(
        self,
        pixels: np.ndarray,
        centre: tuple[int, int],
        radius: int,
        degrees: int = 30,
        grey: int = UNIT_GREY,
    ) -> None:
        """A disc with a wedge of `degrees` cut out to its centre — the notched unit."""
        half = degrees / 2
        cv2.ellipse(pixels, centre, (radius, radius), 0, half, 360 - half, _bgr(grey), -1)

    def blob(
        self, pixels: np.ndarray, box: tuple[int, int, int, int], grey: int = UNIT_GREY
    ) -> None:
        """A filled rectangle, `box` = (x, y, w, h) exactly `w` × `h` pixels."""
        x, y, w, h = box
        cv2.rectangle(pixels, (x, y), (x + w - 1, y + h - 1), _bgr(grey), -1)

    def specks(
        self,
        pixels: np.ndarray,
        count: int,
        *,
        grey: int = SPECK_GREY,
        seed: int = 0,
        avoid: tuple[tuple[int, int, int], ...] = (),
    ) -> None:
        """`count` bright 1–2 px specks scattered on the floor; the same seed, the same layout.

        Specks stay at least 8 px from each other and 12 px outside every `avoid`
        (centre, radius) circle, so no speck survives the blur, the threshold and the
        open as part of a unit, and none bridges to a unit in the close.
        """
        rng = np.random.default_rng(seed)
        placed: list[tuple[int, int, int, int]] = []
        attempts = 0
        while len(placed) < count:
            attempts += 1
            if attempts > 10_000 + 100 * count:
                raise RuntimeError("could not scatter the specks")
            x = int(rng.integers(8, FRAME_SHAPE[1] - 10))
            y = int(rng.integers(8, FRAME_SHAPE[0] - 10))
            side = int(rng.integers(1, 3))
            box = (x, y, x + side - 1, y + side - 1)
            if not _clear_of(box, placed, avoid):
                continue
            placed.append(box)
            pixels[y : y + side, x : x + side] = grey


@pytest.fixture(scope="session")
def localize_config() -> LocalizeConfig:
    """The example config's `inspector.localize`: roi [360, 0, 1200, 1080], threshold 70,
    min_area_px 20000, border_margin_px 10."""
    return load_config(EXAMPLE).config.inspector.localize


@pytest.fixture(scope="session")
def geometry_config() -> GeometryConfig:
    """The example config's `threshold.geometry`: area_px [98000, 131000], axis_ratio_min 0.90,
    solidity_min 0.96."""
    return load_config(EXAMPLE).config.threshold.geometry


@pytest.fixture
def paint() -> Paint:
    """The synthetic-frame drawing primitives."""
    return Paint()


@pytest.fixture(scope="session")
def mask_dilate_px() -> int:
    """The example config's `inspector.mask_dilate_px`: 12."""
    return load_config(EXAMPLE).config.inspector.mask_dilate_px


class FakeScorer:
    """A numpy scorer: each input pixel's distance to the nearest bank row, as the map."""

    def __init__(self, bank: np.ndarray) -> None:
        self.bank = bank

    def score(self, window_input: np.ndarray) -> Scored:
        pixels = window_input.reshape(3, -1).T  # (H·W, 3)
        distances = np.sqrt(((pixels[:, None, :] - self.bank[None, :, :]) ** 2).sum(axis=2))
        anomaly_map = distances.min(axis=1).reshape(window_input.shape[1:]).astype(np.float32)
        return Scored(anomaly_map, float(anomaly_map.max()), backbone_ms=1.0, search_ms=2.0)


class FakePatchCore:
    """The model seam in numpy (`qs.inspector.model.PatchCore`): the bank is each window's
    mean colour, so a window unlike the training windows scores high. It records every
    fit's windows and settings for the tests to inspect."""

    def __init__(self) -> None:
        self.fits: list[tuple[list[np.ndarray], FitSettings]] = []

    def fit(self, windows: Sequence[np.ndarray], settings: FitSettings) -> Fitted:
        self.fits.append((list(windows), settings))
        bank = np.stack([w.reshape(3, -1).mean(axis=1) for w in windows]).astype(np.float32)
        return Fitted(bank=bank, peak_gpu_bytes=None)

    def scorer(self, bank: np.ndarray) -> FakeScorer:
        return FakeScorer(bank)


@pytest.fixture
def fake_patchcore() -> FakePatchCore:
    """A fresh numpy stand-in for the real PatchCore module."""
    return FakePatchCore()


class BlotScorer:
    """A numpy scorer whose map is each pixel's distance to the nearest bank row."""

    def score(self, window_input: np.ndarray) -> Scored:
        pixels = window_input.reshape(3, -1).T  # (H·W, 3)
        distances = np.sqrt(((pixels[:, None, :] - self.bank[None, :, :]) ** 2).sum(axis=2))
        anomaly_map = distances.min(axis=1).reshape(window_input.shape[1:]).astype(np.float32)
        return Scored(anomaly_map, float(anomaly_map.max()), backbone_ms=1.0, search_ms=2.0)

    def __init__(self, bank: np.ndarray) -> None:
        self.bank = bank


class BlotPatchCore:
    """The model seam for the pilot's blot fixtures: the bank is each fit window's brightest
    pixel — the unit's own grey — so a plain unit scores low away from the floor rim inside
    the dilated mask, and a dark blot on the unit scores far higher. It records every fit's
    windows and settings for the tests to inspect."""

    def __init__(self) -> None:
        self.fits: list[tuple[list[np.ndarray], FitSettings]] = []

    def fit(self, windows: Sequence[np.ndarray], settings: FitSettings) -> Fitted:
        self.fits.append((list(windows), settings))
        bank = np.stack([w.max(axis=(1, 2)) for w in windows]).astype(np.float32)
        return Fitted(bank=bank, peak_gpu_bytes=None)

    def scorer(self, bank: np.ndarray) -> BlotScorer:
        return BlotScorer(bank)


# --- the P08 synthetic lean session (tickets #80–#83) ---------------------------------------
# One lean dataset session at a small frame size, written with P05b's own planner and
# writers: `make_plan`, `shot_list`, `save_frame`, `FrameLog`, `write_session`. Discs,
# blotted discs, flattened and oval discs, empty and edge frames; a matching tmp config.

BUILD_SESSION = "2026-09-25-mvp"
SMALL_W, SMALL_H = 320, 240
BENCH_CAMERA = CameraReadback(10000, 0, 4500, 30, True)
PHONE_CAMERA = CameraReadback(20000, 100, None, None, False)
STARTED_AT = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
CARD_BOX = (296, 8, 16, 16)  # the grey card, where the small config's reference_patch roi sits
UNIT_RADIUS = 26
BLOT_RADIUS = 8
BLOT_GREY = 40


@dataclass(frozen=True, slots=True)
class ShotRecord:
    """One recorded frame, enough for `write_session` (`session.Recorded`)."""

    event: TriggerEvent = TriggerEvent(1, 0.0, STARTED_AT, "test")
    reference_patch: float | None = 0.62
    camera: CameraReadback = BENCH_CAMERA


def bench_product() -> Product:
    """The product record of the synthetic session."""
    return Product(
        brand="Mochi-Dom",
        flavour="strawberry",
        shop="Magnit",
        best_before=date(2026, 9, 28),
        purchased_at=datetime(2026, 9, 25, 9, 10, tzinfo=UTC),
    )


def draw_lean_frame(unit: str, state: str, rotation: int, damage: str | None) -> np.ndarray:
    """One small synthetic frame: a pale disc on a black floor; a dent or tear as a dark blot
    on the disc; deformed-flat flattened, deformed-oval squeezed; `empty` a bare floor with
    the grey card; `edge` a disc over the inspection area's edge."""
    paint = Paint()
    pixels = np.full((SMALL_H, SMALL_W, 3), FLOOR_GREY, dtype=np.uint8)
    paint.blob(pixels, CARD_BOX, 158)  # the grey card reads 0.62
    if state == "empty":
        return pixels
    if state == "edge":
        paint.disc(pixels, (272, 120), UNIT_RADIUS)
        return pixels
    centre = (150 + 6 * rotation, 116 - 4 * rotation)
    if state == "dmg" and damage == "deformed-flat":
        paint.ellipse(pixels, centre, (30, 15), 0)
        return pixels
    if state == "dmg" and damage == "deformed-oval":
        paint.ellipse(pixels, centre, (22, 26), 0)
        return pixels
    paint.disc(pixels, centre, UNIT_RADIUS)
    if state == "dmg":
        paint.disc(pixels, (centre[0] + 7, centre[1] - 5), BLOT_RADIUS, BLOT_GREY)
    return pixels


def write_lean_session(
    root: Path,
    *,
    device: str = "/dev/video0",
    camera: CameraReadback = BENCH_CAMERA,
    broken: Sequence[tuple[str, int]] = (),
) -> tuple[Plan, Path]:
    """The synthetic lean dataset session under `root/BUILD_SESSION`, exactly as P05b writes
    one: `plan.yml`, every rule-named PNG, `frames.csv` and `session.yml` last. `broken`
    breaks units as `qs import-phone --broken` does: the frames the unit got are saved and
    logged `discarded`, the next spare takes its split and damage. The plan and the folder."""
    folder = root / BUILD_SESSION
    plan = make_plan("lean", 20260925)
    write_plan(folder, plan, "sheet\n")
    shot = substitute_spares(plan, "G2", list(broken)) if broken else None
    steps = shot.steps if shot is not None else shot_list(plan, "G2")
    damage = shot.damage if shot is not None else plan.damage
    discarded = {file for files in (shot.discarded if shot else {}).values() for file in files}
    record = ShotRecord(camera=camera)
    with FrameLog(folder) as log:
        for step in steps:
            if not isinstance(step, Shot):
                continue
            frame = step.frame
            pixels = draw_lean_frame(frame.unit, frame.state, frame.rotation,
                                     damage.get(frame.unit))
            sha = save_frame(folder, frame, pixels)
            log.append_row(step.file, frame, record, None, None, sha,
                           "discarded" if step.file in discarded else "saved")
    written = SessionRecord(
        session_id=BUILD_SESSION,
        geometry="G2",
        product=bench_product(),
        device=device,
        camera_at_open=camera,
        plan=plan,
        frames=(record,),
        substitution=shot if shot is not None and shot.discarded else None,
    )
    write_session(folder, written, by="qs import-phone")
    return plan, folder


@pytest.fixture(scope="session")
def lean_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic lean session — 265 frames — written once for the whole test run."""
    _, folder = write_lean_session(tmp_path_factory.mktemp("lean-session"))
    return folder


@pytest.fixture(scope="session")
def broken_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same lean session with one test unit broken after its good frames: the first
    spare took its place in test, and the unit's 3 good frames are logged `discarded`."""
    plan = make_plan("lean", 20260925)
    _, folder = write_lean_session(tmp_path_factory.mktemp("broken-session"),
                                   broken=[(plan.test[0], 3)])
    return folder


@pytest.fixture(scope="session")
def phone_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same lean session as the phone shot it (ADR-0022): device `phone:…`, unverified."""
    _, folder = write_lean_session(tmp_path_factory.mktemp("phone-session"),
                                   device="phone:Xiaomi Redmi Note 12", camera=PHONE_CAMERA)
    return folder


@pytest.fixture
def session_copy(tmp_path: Path, lean_session: Path) -> Path:
    """The lean session copied into this test's `tmp_path`, for the tests that tamper."""
    target = tmp_path / BUILD_SESSION
    shutil.copytree(lean_session, target)
    return target


@pytest.fixture(scope="session")
def bench_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The station config matching the small frames, cut from the example: the ROIs and
    tolerances in their pixels, so the tickets that replay the session can load it."""
    text = (Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"
            ).read_text(encoding="utf-8")
    for old, new in (
        ("width: 1920", f"width: {SMALL_W}"),
        ("height: 1080", f"height: {SMALL_H}"),
        ("[1780, 40, 100, 100]", f"[{CARD_BOX[0]}, {CARD_BOX[1]}, {CARD_BOX[2]}, "
                                 f"{CARD_BOX[3]}]"),
        ("[360, 0, 1200, 1080]", "[8, 0, 272, 240]"),
        ("min_area_px: 20000", "min_area_px: 500"),
        ("border_margin_px: 10", "border_margin_px: 4"),
        ("mask_dilate_px: 12", "mask_dilate_px: 4"),
        ("area_px:        [98000, 131000]", "area_px:        [1600, 2600]"),
        ("axis_ratio_min: 0.90", "axis_ratio_min: 0.85"),
        ("solidity_min:   0.96", "solidity_min:   0.95"),
    ):
        assert old in text, f"the example no longer contains {old!r}"
        text = text.replace(old, new)
    path = tmp_path_factory.mktemp("bench-config") / "station.yml"
    path.write_text(text, encoding="utf-8")
    return path


class FakeScreen:
    """A `ScreenLike` without a display (#87): keeps a copy of every presented image and
    plays scripted batches of key presses, one batch per `keys()` call (empty when none is
    left). `close()` sets `closed` and counts its calls. `last_pump_end` is set by the test
    (#88); each scripted `KeyPress` carries its own `after_s`.

    For the station's loop (#89) it can also act as the window does:
    - `clock`: every `present()` is a pump that ends at `clock()`, and `keys()` stamps each
      press it returns with the end of the pump before the last one, as `Screen` does;
    - `held`: a key held down — once the script is used up, every `keys()` call delivers one
      press of it (auto-repeat at the loop's pace);
    - `close_when(calls)`: closes itself after the `keys()` call for which it is true;
    - `keep`: only the newest `keep` presented images are kept."""

    def __init__(
        self,
        script: Sequence[Sequence[KeyPress]] = (),
        *,
        clock: Callable[[], float] | None = None,
        held: str | None = None,
        close_when: Callable[[int], bool] | None = None,
        keep: int | None = None,
    ) -> None:
        self.presented: list[Image.Image] = []
        self.script: list[list[KeyPress]] = [list(batch) for batch in script]
        self.closed = False
        self.close_calls = 0
        self.last_pump_end = 0.0 if clock is None else clock()
        self.presents = 0
        self.keys_calls = 0
        self._clock = clock
        self._held = held
        self._close_when = close_when
        self._keep = keep
        self._pump_before = self.last_pump_end

    def present(self, image: Image.Image) -> None:
        self.presents += 1
        self.presented.append(image.copy())
        if self._keep is not None and len(self.presented) > self._keep:
            del self.presented[: len(self.presented) - self._keep]
        if self._clock is not None:
            self._pump_before = self.last_pump_end
            self.last_pump_end = self._clock()

    def keys(self) -> list[KeyPress]:
        self.keys_calls += 1
        if self.script:
            batch = self.script.pop(0)
        elif self._held is not None:
            batch = [KeyPress(self._held, 1000 + self.keys_calls)]
        else:
            batch = []
        if self._close_when is not None and self._close_when(self.keys_calls):
            self.closed = True
        if self._clock is not None:
            batch = [KeyPress(p.keysym, p.tk_ms, self._pump_before) for p in batch]
        return batch

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
