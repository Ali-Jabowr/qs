"""`ReplayTrigger` and `ReplaySource` (#69, docs/11 §6): order, metadata, the hash refusal,
frame validation — each directly on the source."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from qs.errors import CaptureError
from qs.sources.replay import ReplayError, ReplaySource, replay_files, replay_pair
from station_fixtures import (
    HEIGHT,
    WIDTH,
    FakeClock,
    loaded_config,
    sha256_file,
    textured,
    write_frames,
    write_frames_csv,
    write_png,
)


def test_files_are_the_pngs_directly_in_the_directory_in_name_order(tmp_path: Path) -> None:
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    for name in ("b.png", "a.png", "c.png"):
        write_png(shoot / name, textured(1))
    write_png(shoot / "z.jpg", textured(1))
    (shoot / "sub").mkdir()
    write_png(shoot / "sub" / "0.png", textured(1))
    assert [p.name for p in replay_files(shoot)] == ["a.png", "b.png", "c.png"]


def test_a_missing_or_empty_directory_cannot_be_replayed(tmp_path: Path) -> None:
    with pytest.raises(ReplayError):
        replay_files(tmp_path / "nowhere")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ReplayError):
        replay_files(tmp_path / "empty")


def test_the_trigger_gives_one_event_per_file_then_is_exhausted(tmp_path: Path) -> None:
    config = loaded_config(tmp_path).config
    files = write_frames(tmp_path / "shoot", 3)
    clock = FakeClock(step=0.5)
    trigger, _ = replay_pair(config, files, "shoot", clock=clock)
    events = []
    for _ in range(3):
        assert trigger.exhausted is False
        event = trigger.poll()
        assert event is not None
        events.append(event)
    assert trigger.exhausted is True
    assert trigger.poll() is None  # never blocks, never repeats
    assert [e.seq for e in events] == [1, 2, 3]
    assert {e.source_id for e in events} == {"replay:shoot"}
    assert events[0].t0 < events[1].t0 < events[2].t0
    for e in events:
        assert e.wall.utcoffset() == timedelta(0)


def _source(tmp_path: Path, files: list[Path]) -> ReplaySource:
    source = ReplaySource(loaded_config(tmp_path).config, files, clock=FakeClock(step=0.001))
    source.open()
    return source


def _event(tmp_path: Path, files: list[Path], seq: int):  # type: ignore[no-untyped-def]
    trigger, _ = replay_pair(loaded_config(tmp_path).config, files, "x", clock=FakeClock())
    for _ in range(seq):
        event = trigger.poll()
    return event


def test_a_good_frame_carries_pixels_readback_and_measured_patch(tmp_path: Path) -> None:
    files = write_frames(tmp_path / "shoot", 1)
    source = _source(tmp_path, files)
    frame = source.capture(_event(tmp_path, files, 1))
    assert frame.pixels.shape == (HEIGHT, WIDTH, 3) and frame.pixels.dtype == np.uint8
    assert np.array_equal(frame.pixels, textured(1))
    assert frame.camera.verified is False and frame.camera.exposure_us is None
    assert frame.reference_patch == pytest.approx(158 / 255)
    assert frame.capture_ms > 0


@pytest.mark.parametrize("fault", ["hash", "size", "flat", "not_png", "grey", "duplicate"])
def test_capture_refuses_a_frame_it_cannot_trust(tmp_path: Path, fault: str) -> None:
    shoot = tmp_path / "shoot"
    files = write_frames(shoot, 1)
    path = files[0]
    if fault == "hash":
        write_frames_csv(shoot, [{"file": path.name, "verified": "true", "sha256": "0" * 64}])
    elif fault == "size":
        write_png(path, np.zeros((HEIGHT, WIDTH + 1, 3), np.uint8) + textured(1)[:, :1])
    elif fault == "flat":
        write_png(path, np.full((HEIGHT, WIDTH, 3), 90, np.uint8))
    elif fault == "not_png":
        path.write_text("not an image")
    elif fault == "grey":
        write_png(path, textured(1)[:, :, 0])
    else:
        row = {"file": path.name, "verified": "true", "sha256": sha256_file(path)}
        write_frames_csv(shoot, [row, row])
    source = _source(tmp_path, files)
    with pytest.raises(CaptureError):
        source.capture(_event(tmp_path, files, 1))


def test_a_matching_row_is_used_and_needs_every_value_to_be_verified(tmp_path: Path) -> None:
    shoot = tmp_path / "shoot"
    files = write_frames(shoot, 2)
    write_frames_csv(shoot, [
        {"file": files[0].name, "exposure_us": 10000, "gain": 0, "wb_kelvin": 4500,
         "focus": 30, "verified": "true", "sha256": sha256_file(files[0])},
        {"file": files[1].name, "exposure_us": 10000, "gain": 0, "wb_kelvin": None,
         "focus": 30, "verified": "true", "sha256": sha256_file(files[1])},
    ])
    source = _source(tmp_path, files)
    first = source.capture(_event(tmp_path, files, 1))
    second = source.capture(_event(tmp_path, files, 2))
    assert (first.camera.exposure_us, first.camera.verified) == (10000, True)
    assert second.camera.wb_kelvin is None and second.camera.verified is False


def test_replay_has_no_preview(tmp_path: Path) -> None:
    files = write_frames(tmp_path / "shoot", 1)
    assert _source(tmp_path, files).preview() is None
