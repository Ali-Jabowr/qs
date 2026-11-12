"""The lens-hole circle `lens_mask_px` is left out of every warmth measurement (#133, ADR-0029
decision 2).

A pink моти has warmth 50 and `warmth_min` is 45, so it is ГОДЕН. A black disc of radius 14 is drawn
over every frame at (150, 120), as the lens hole's dark reflection would be: it stays where it is
while the cup crosses. The inner half of the rim is a disc of radius 26, so the spot lies in it at
the first two burst frames (cup at x 130 and 170) and pulls the median under 45. Its diameter, 28,
is under the smallest rim (50), so it is not itself a cup.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.cmd.score_line import play_video
from qs.config import load_config
from qs.inspector.line import warmth
from strict_fixtures import crossing, steel, write_video
from warmth_fixtures import warmth_config

SPOT = (150, 120)


def _with_spot(frames: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for pixels in frames:
        drawn = pixels.copy()
        cv2.circle(drawn, SPOT, 14, (0, 0, 0), -1)
        out.append(drawn)
    return out


@pytest.mark.parametrize(
    ("lens_mask_px", "label", "value"),
    [
        ([10, 10, 3], "defect", 40.0),  # masked elsewhere, the spot drags a pink моти (50) under 45
        ([150, 120, 18], "pass", 50.0),  # masked, the spot is left out
    ],
)
def test_a_dark_spot_at_the_masked_circle_does_not_change_a_full_cups_verdict(
    tmp_path: Path, lens_mask_px: list[int], label: str, value: float
) -> None:
    video = write_video(
        tmp_path / "line.avi", _with_spot(steel(3) + crossing("pink") + steel(3))
    )
    loaded = load_config(warmth_config(tmp_path, warmth_min=45.0, lens_mask_px=lens_mask_px))

    records, _ = play_video(loaded, video)

    assert [r.verdict.label.value for r in records] == [label]
    (check,) = records[0].verdict.checks
    assert check.value == pytest.approx(value, abs=1.5)


def test_a_mask_covering_the_whole_inner_half_measures_nothing() -> None:
    pixels = np.full((240, 320, 3), 128, dtype=np.uint8)
    rim = ((160.0, 120.0), (100.0, 100.0), 0.0)

    assert warmth(pixels, rim, (160, 120, 60)) is None


@pytest.mark.parametrize(
    ("burst_min_frames", "labels"),
    [
        (3, ["no_verdict"]),  # the cup at x 170 is wholly masked: 2 frames of 3 are fewer than 3
        (2, ["pass"]),        # the two frames that can be measured are enough
    ],
)
def test_a_burst_frame_the_mask_covers_is_skipped_and_burst_min_frames_decides(
    tmp_path: Path, burst_min_frames: int, labels: list[str]
) -> None:
    video = write_video(tmp_path / "line.avi", steel(3) + crossing("pink") + steel(3))
    loaded = load_config(warmth_config(
        tmp_path, warmth_min=45.0, lens_mask_px=[170, 120, 28], burst_min_frames=burst_min_frames
    ))

    records, _ = play_video(loaded, video)

    assert [r.verdict.label.value for r in records] == labels
