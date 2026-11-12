"""A line station with `rim_size_px` judges only a rim within 5% of it (#133, ADR-0029 decision 3).

The drawn cup has a rim of about 105 px across (half-axes 50). It crosses the zone once and is a
pink моти, so a cup that is judged is ГОДЕН.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.cmd.score_line import play_video
from qs.config import load_config
from strict_fixtures import crossing, steel, write_video
from warmth_fixtures import warmth_config


@pytest.mark.parametrize(
    ("rim_size_px", "labels"),
    [
        ([108, 108], ["pass"]),  # the rim (105) is 3% under the set size: judged
        ([102, 102], ["pass"]),  # 3% over
        ([117, 117], []),        # 10% under: no cup, no record
        ([90, 90], []),          # 15% over the rim, 10% over its inner edge (99)
    ],
)
def test_a_rim_off_the_set_size_by_more_than_5_percent_gives_no_record(
    tmp_path: Path, rim_size_px: list[int], labels: list[str]
) -> None:
    video = write_video(tmp_path / "line.avi", steel(3) + crossing("pink") + steel(3))
    loaded = load_config(warmth_config(tmp_path, rim_size_px=rim_size_px))

    records, _ = play_video(loaded, video)

    assert [r.verdict.label.value for r in records] == labels
