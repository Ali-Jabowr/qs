"""Burst of 1 on the bench implementations (#130): file replay, webcam and the appearance
inspector give what they gave before bursts; a burst of 2 is refused. Synthetic frames and the
existing fakes only: no camera, no GPU."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from appearance_fixtures import config_naming, frame_of
from qs.contracts import CHECK_APPEARANCE, CHECK_GEOMETRY, Label
from qs.inspector.appearance import AppearanceInspector
from qs.sources.replay import ReplaySource
from qs.sources.webcam import WebcamSource
from station_fixtures import FakeClock, event, loaded_config, textured, write_frames
from test_appearance import ConstantScorer, Factory, _bundle, _disc
from test_v4l2 import FakeControls
from test_webcam import FakeCapture, image, small_config
from test_webcam import event as webcam_event


def _replay(tmp_path: Path) -> ReplaySource:
    files = write_frames(tmp_path / "shoot", 1)
    source = ReplaySource(loaded_config(tmp_path).config, files, clock=FakeClock(step=0.001))
    source.open()
    return source


def test_replay_burst_of_one_is_the_file_of_its_seq(tmp_path: Path) -> None:
    (frame,) = _replay(tmp_path).capture_burst(event(1), 1)
    assert np.array_equal(frame.pixels, textured(1))
    assert frame.event == event(1)


def test_replay_burst_of_two_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="burst of 2"):
        _replay(tmp_path).capture_burst(event(1), 2)


@pytest.fixture
def webcam() -> Iterator[tuple[WebcamSource, float]]:
    base = time.perf_counter()
    script = [
        (base - 0.05, image(10)),
        (base + 0.010, image(30)),  # exactly t0 + exposure: stale
        (base + 0.020, image(40)),
        (base + 0.030, image(50)),
    ]
    cap = FakeCapture(script)
    source = WebcamSource(
        small_config(), open_capture=lambda device: cap, open_controls=lambda d: FakeControls()
    )
    source.open()
    yield source, base
    cap.finish()
    source.close()


def test_webcam_burst_of_one_is_the_earliest_frame_after_trigger_and_exposure(
    webcam: tuple[WebcamSource, float],
) -> None:
    source, base = webcam
    (frame,) = source.capture_burst(webcam_event(base), 1)
    assert frame.grabbed_at == pytest.approx(base + 0.020, abs=1e-9)
    assert int(frame.pixels[0, 0, 0]) == 40


def test_webcam_burst_of_two_is_refused(webcam: tuple[WebcamSource, float]) -> None:
    source, base = webcam
    with pytest.raises(ValueError, match="burst of 2"):
        source.capture_burst(webcam_event(base), 2)


def test_appearance_burst_of_one_is_the_single_frame_verdict(tmp_path: Path) -> None:
    models = tmp_path / "models"
    loaded = config_naming(tmp_path, _bundle(models))
    inspector = AppearanceInspector(
        loaded.config_hash, models, validation=False,
        make_scorer=Factory(ConstantScorer(1.0, pred_score=0.75)), cuda_available=lambda: True,
    )
    inspector.load(loaded.config)
    verdict = inspector.inspect_burst((frame_of(_disc(190)),))
    assert verdict.label is Label.PASS
    assert tuple(check.name for check in verdict.checks) == (CHECK_GEOMETRY, CHECK_APPEARANCE)
    assert verdict.checks[1].value == 1.0
    assert verdict.unit_box == (770, 350, 381, 381)


def test_appearance_burst_of_two_is_refused(tmp_path: Path) -> None:
    inspector = AppearanceInspector("hash", tmp_path / "models")
    frame = frame_of(_disc(190))
    with pytest.raises(ValueError, match="burst of 2"):
        inspector.inspect_burst((frame, frame))
