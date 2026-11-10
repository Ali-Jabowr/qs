"""A line station with `rim_size_px` judges only a rim within 5% of it (#133, ADR-0029 decision 3).

The drawn cup has a rim of about 100 px across (half-axes 50). It crosses the zone once and is a
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
        (103, ["pass"]),   # the rim is 3% under the set size: judged
        (97, ["pass"]),    # 3% over
        (112, []),         # 10% under: no cup, no record
        (90, []),          # 10% over
        (None, ["pass"]),  # no key: the old axis range alone, as before
    ],
)
def test_a_rim_off_the_set_size_by_more_than_5_percent_gives_no_record(
    tmp_path: Path, rim_size_px: float | None, labels: list[str]
) -> None:
    video = write_video(tmp_path / "line.avi", steel(3) + crossing("pink") + steel(3))
    loaded = load_config(warmth_config(tmp_path, rim_size_px=rim_size_px))

    records, _ = play_video(loaded, video)

    assert [r.verdict.label.value for r in records] == labels
