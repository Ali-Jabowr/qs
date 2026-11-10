"""A ring that stays still while the chain moves is a reflection, not a cup (#133, ADR-0029
decision 3), seen through `play_video`, the engine of `qs score-line`.

A thin ring of radius 35 is drawn at (60, 190) over every frame, low in the zone so a cup crossing
at y 120 does not cover its centre. With `still_ring_px` set:

- the chain moving past it: the ring gives no record once it has been seen at the same place on 3
  moving frames, and the cups give one each. The ring is in frame 0, before that can be known, and
  a rim is followed by default, so it makes one `no_verdict` record at the start of the video and
  none after it (a station already running has confirmed its reflections);
- a cup standing for six frames in a stop of the chain, with the ring still in the picture: that
  cup gets exactly one record, and the ring none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.cmd.score_line import play_video
from qs.config import load_config
from strict_fixtures import PATH_X, ROW, crossing, standing, steel, with_ring, write_video
from test_cup import drawn_frame
from warmth_fixtures import warmth_config

RING = ((60, 190), 35)


def _stopping_crossing() -> list:
    """A cup that runs to x 130, stands there six frames, then runs on and out of view."""
    head = [drawn_frame("pale", centre=(x, ROW), seed=x) for x in PATH_X[:4]]
    rest = [drawn_frame("pale", centre=(x, ROW), seed=200 + x) for x in PATH_X[4:]]
    return head + standing(6, contents="pale", seed=300) + rest + steel(4, seed=400)


@pytest.mark.parametrize(
    ("frames", "labels"),
    [
        # three cups pass the ring: the startup record of the ring, then one for each cup
        (crossing("pink") + crossing("pale", seed=500) + crossing("pink", seed=900),
         ["no_verdict", "pass", "pass", "pass"]),
        # a cup stands in a stop beside the ring: exactly one record for it
        (crossing("pink") + _stopping_crossing(), ["no_verdict", "pass", "pass"]),
    ],
)
def test_a_still_ring_on_a_moving_chain_gives_no_record_and_a_standing_cup_gives_one(
    tmp_path: Path, frames: list, labels: list[str]
) -> None:
    video = write_video(tmp_path / "line.avi", with_ring(steel(1) + frames + steel(3), *RING))
    loaded = load_config(warmth_config(tmp_path, still_ring_px=3))

    records, _ = play_video(loaded, video)

    assert [r.verdict.label.value for r in records] == labels


def test_a_cup_standing_from_the_first_frame_of_a_stopped_chain_gets_one_record(
    tmp_path: Path,
) -> None:
    video = write_video(tmp_path / "line.avi", standing(10, contents="pink") + steel(3))
    loaded = load_config(warmth_config(tmp_path, still_ring_px=3))

    records, _ = play_video(loaded, video)

    assert [r.verdict.label.value for r in records] == ["pass"]


@pytest.mark.parametrize("still_ring_px", [None, 3])
def test_a_moving_cup_found_on_every_other_frame_still_gets_its_record(
    tmp_path: Path, still_ring_px: int | None
) -> None:
    frames = crossing("pink")
    gappy = [f if i % 2 == 0 else steel(1, seed=900 + i)[0] for i, f in enumerate(frames)]
    video = write_video(tmp_path / "line.avi", steel(3) + gappy + steel(3))
    loaded = load_config(warmth_config(tmp_path, still_ring_px=still_ring_px))

    records, _ = play_video(loaded, video)

    assert [r.verdict.label.value for r in records] == ["no_verdict"]
