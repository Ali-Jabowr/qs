"""`qs train` on a line config: good cups from a labels file and its video to a bundle (#108).

The drawn video, at 10 fps, 13 frames. Frames 0–9 show a full (pale) cup, its centre alternating
between x 120 and x 200; frame 10 is labelled hand, 11 nothing, 12 empty cup. The frames labelled
unit are 0–9, so with a 0.2 s gap (2 frames) the time split trains on 0–6 and drops 7–9
(`tests/test_line_split.py` works the rule by hand).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from conftest import FakePatchCore
from qs.__main__ import build_parser
from qs.cmd import train
from station_fixtures import config_mapping
from test_cup import drawn_frame

FPS = 10.0
VIDEO = "line.avi"
UNIT_FRAMES = range(10)
TRAINED = ["line.avi#0", "line.avi#1", "line.avi#2", "line.avi#3", "line.avi#4", "line.avi#5",
           "line.avi#6"]


def _frames(no_cup_at: int | None = None) -> list[np.ndarray]:
    cups = [drawn_frame("none" if i == no_cup_at else "pale", centre=(120 + 80 * (i % 2), 120))
            for i in UNIT_FRAMES]
    return cups + [drawn_frame("none", seed=2), drawn_frame("none", seed=3), drawn_frame("empty")]


def _footage(root: Path, frames: Sequence[np.ndarray]) -> Path:
    footage = root / "footage"
    footage.mkdir()
    writer = cv2.VideoWriter(str(footage / VIDEO), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (320, 240))
    assert writer.isOpened()
    for pixels in frames:
        writer.write(pixels)
    writer.release()
    return footage


def _labels(root: Path) -> Path:
    frames = {f"{VIDEO}#{i}": {"label": "unit", "flavour": "white"} for i in UNIT_FRAMES}
    frames |= {
        f"{VIDEO}#10": {"label": "hand", "flavour": None},
        f"{VIDEO}#11": {"label": "nothing", "flavour": None},
        f"{VIDEO}#12": {"label": "empty_cup", "flavour": None},
    }
    path = root / "top.json"
    path.write_text(json.dumps({"version": 1, "source": "line", "frames": frames, "cups": []}),
                    encoding="utf-8")
    return path


def _config(root: Path) -> Path:
    data = config_mapping(root, inspector_type="line")
    data["trigger"]["type"] = "cup"
    data["camera"]["type"] = "replay"
    data["inspector"]["checks"] = ["empty_cup"]
    data["inspector"]["line"] = {
        "zone": [20, 30, 240, 180],
        "rim_size_px": [105, 105],
        "direction": "right",
        "fps": FPS,
        "warmth_min": 5.0,
        "burst": 3,
        "warmth_spread_max": 40.0,
        "max_speed_px_s": 320.0,
    }
    path = root / "line.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _run(tmp_path: Path, fake: FakePatchCore, *flags: str, no_cup_at: int | None = None) -> int:
    args = build_parser().parse_args([
        "train", "--config", str(_config(tmp_path)), "--labels", str(_labels(tmp_path)),
        "--footage", str(_footage(tmp_path, _frames(no_cup_at))), "--window-px", "40",
        "--input-px", "32", "--product", "mochi-top", "--models", str(tmp_path / "models"),
        "--gap-s", "0.2", *flags,
    ])
    return train.main(args, make_patchcore=lambda device: fake, cuda_available=lambda: True)


def _bundle(tmp_path: Path) -> dict[str, object]:
    (bundle,) = (tmp_path / "models").iterdir()
    loaded: dict[str, object] = yaml.safe_load((bundle / "bundle.yml").read_text("utf-8"))
    return loaded


def test_the_bundle_holds_the_training_block_of_the_unit_frames_only(tmp_path: Path) -> None:
    assert _run(tmp_path, FakePatchCore()) == 0
    bundle = _bundle(tmp_path)
    assert [image["file"] for image in bundle["images"]] == TRAINED  # type: ignore[attr-defined]
    assert bundle["units"] == ["line.avi"]


def test_the_window_follows_the_cup_across_the_frame(tmp_path: Path) -> None:
    fake = FakePatchCore()
    assert _run(tmp_path, fake) == 0
    (windows, _), = fake.fits
    # Frame 0's cup is at x 120 and frame 1's at x 200: both windows lie inside the моти.
    assert np.abs(windows[0] - windows[1]).max() < 0.5


def test_a_training_frame_with_no_cup_is_refused_and_no_bundle_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, FakePatchCore(), no_cup_at=2) == 1
    assert capsys.readouterr().out.strip() == "refused: line.avi#2: no cup wholly in the zone"
    assert not (tmp_path / "models").exists()


def test_a_line_config_needs_the_footage_with_its_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = build_parser().parse_args([
        "train", "--config", str(_config(tmp_path)), "--labels", str(_labels(tmp_path)),
        "--window-px", "40", "--product", "mochi-top", "--models", str(tmp_path / "models"),
    ])
    assert train.main(args, make_patchcore=lambda device: FakePatchCore(),
                      cuda_available=lambda: True) == 2
    assert capsys.readouterr().out.strip() == "a line config trains with --labels and --footage"
