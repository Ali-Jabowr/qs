"""A line station with `warmth_min` judges each cup by the median warmth of a burst (#132).

The video is drawn lossless at 8 fps; a cup crosses at 40 px a frame, one crossing being the nine
steps of `PATH_X` and four frames of steel. A crossing that starts at frame s is seen wholly in the
zone from x 90, the third step, so the trigger fires at s + 2 and the burst is frames s + 3, s + 4
and s + 5: the cup at x 130, 170 and 210. Warmth is the mean of red minus blue over the inner half
of the rim. Empty steel is 0, a pink моти 50 and a pale one 10 (their colours in `test_cup`);
`_warmed` adds red over a disc of radius 35 inside the rim. The drawn crossings:

    0       empty, no tint                                  БРАК
    1       pink                                            ГОДЕН
    2       pale                                            ГОДЕН
    3       empty, one burst frame warmed by 20             БРАК (median 0; the maximum is 20)
    4       empty, one burst frame warmed by 60             НЕТ РЕШЕНИЯ (spread 60 over 40)
    5       empty, the cup missing from one burst frame     НЕТ РЕШЕНИЯ (2 of 3 frames)
    6       empty, warmed by 0, 12 and 30                   ГОДЕН (median 12); evidence: middle
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.cmd.score_line import play_video
from qs.config import LoadedConfig, load_config
from qs.contracts import InspectionRecord
from test_cup import drawn_frame
from warmth_fixtures import FPS, warmth_config

PATH_X = (10, 50, 90, 130, 170, 210, 250, 290, 330)
BURST_X = (130, 170, 210)
VIDEO = "line.avi"


def _warmed(pixels: np.ndarray, centre: tuple[int, int], red: int) -> np.ndarray:
    tint = np.zeros(pixels.shape[:2], dtype=np.uint8)
    cv2.circle(tint, centre, 35, 255, -1)
    warmed = pixels.copy()
    warmed[tint > 0, 2] = np.clip(warmed[tint > 0, 2].astype(np.int16) + red, 0, 255)
    return warmed


def _crossing(contents: str, tints: dict[int, int] | None = None,
              missing: int | None = None) -> list[np.ndarray]:
    """One crossing: `tints` maps the x of a burst frame to the red added over its cup, and
    `missing` is the x of a frame drawn with no cup."""
    frames = []
    for x in PATH_X:
        shown = "none" if x == missing else contents
        pixels = drawn_frame(shown, centre=(x, 120), seed=x)
        if tints and x in tints:
            pixels = _warmed(pixels, (x, 120), tints[x])
        frames.append(pixels)
    return frames + [drawn_frame("none", seed=100 + i) for i in range(4)]


def _video(root: Path) -> Path:
    steel = [drawn_frame("none", seed=100 + i) for i in range(3)]
    crossings = (
        _crossing("empty")
        + _crossing("pink")
        + _crossing("pale")
        + _crossing("empty", {170: 20})
        + _crossing("empty", {170: 60})
        + _crossing("empty", missing=170)
        + _crossing("empty", {130: 0, 170: 12, 210: 30})
    )
    path = root / VIDEO
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (320, 240))
    assert writer.isOpened()
    for pixels in steel + crossings + steel:
        writer.write(pixels)
    writer.release()
    return path


def _loaded(root: Path) -> LoadedConfig:
    return load_config(warmth_config(root))


@pytest.fixture(scope="module")
def records(tmp_path_factory: pytest.TempPathFactory) -> list[InspectionRecord]:
    root = tmp_path_factory.mktemp("warmth")
    found, _ = play_video(_loaded(root), _video(root))
    return found


def test_each_cup_gets_the_verdict_of_its_median_warmth(records: list[InspectionRecord]) -> None:
    assert [(r.verdict.label.value, r.verdict.reason) for r in records] == [
        ("defect", None),
        ("pass", None),
        ("pass", None),
        ("defect", None),
        ("no_verdict", "placement"),
        ("no_verdict", "placement"),
        ("pass", None),
    ]


@pytest.mark.parametrize(
    ("index", "warmth", "passed", "failed_on"),
    [(0, 0.0, False, "warmth<min"), (1, 50.0, True, None), (2, 10.0, True, None),
     (3, 0.0, False, "warmth<min")],
)
def test_the_record_holds_the_median_the_threshold_the_frame_count_and_the_spread(
    records: list[InspectionRecord], index: int, warmth: float, passed: bool,
    failed_on: str | None,
) -> None:
    (check,) = records[index].verdict.checks

    assert check.name == "empty_cup"
    assert check.value == pytest.approx(warmth, abs=2.0)
    assert check.limit == 5.0
    assert check.passed is passed
    assert check.failed_on == failed_on
    assert dict(check.measures)["frames"] == 3.0


def test_one_warm_frame_shows_in_the_spread_but_not_in_the_median(
    records: list[InspectionRecord],
) -> None:
    (check,) = records[3].verdict.checks

    assert dict(check.measures)["spread"] == pytest.approx(20.0, abs=2.0)
    assert check.value == pytest.approx(0.0, abs=2.0)


def test_the_evidence_frame_is_the_burst_frame_closest_to_the_median(
    records: list[InspectionRecord],
) -> None:
    record = records[6]
    x, _, w, _ = record.verdict.unit_box or (0, 0, 0, 0)
    assert x + w / 2 == pytest.approx(170.0, abs=3.0)
    decoded = cv2.imdecode(np.frombuffer(record.image_jpeg or b"", np.uint8), cv2.IMREAD_COLOR)
    patch = decoded[115:125, 165:175].astype(np.int16)
    assert (patch[..., 2] - patch[..., 0]).mean() == pytest.approx(12.0, abs=4.0)
