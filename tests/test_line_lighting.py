"""A line station may run with no reference patch: the visit's footage has no grey card in view,
so the lighting check would end every cup in NO_VERDICT(`lighting`). `lighting.reference_patch:
null` turns the check off, and only a station with `inspector.line` may do so; the bench keeps it.
Frames are drawn; the card's corner is painted over with glare, as on the line's steel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from evidence_records import fast_sqlite
from qs.cmd.station import main
from qs.config import load_config
from qs.errors import ConfigError
from station_fixtures import PATCH_ROI, FakeClock, rows, station_args, write_config
from test_line_station import BLANK_AFTER, _config_file, _crossing, _steel, _write_shoot


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def _no_card(frames: list[np.ndarray]) -> list[np.ndarray]:
    x, y, w, h = PATCH_ROI
    for pixels in frames:
        pixels[y : y + h, x : x + w] = 255
    return frames


def _without_patch(path: Path) -> Path:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["lighting"]["reference_patch"] = None
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_a_line_station_may_leave_the_reference_patch_out(tmp_path: Path) -> None:
    loaded = load_config(_without_patch(_config_file(tmp_path)))
    assert loaded.config.lighting.reference_patch is None


def test_a_bench_station_may_not_leave_the_reference_patch_out(tmp_path: Path) -> None:
    path = _without_patch(write_config(tmp_path))
    with pytest.raises(ConfigError, match="lighting.reference_patch: only a line station"):
        load_config(path)


def test_a_line_station_without_the_patch_judges_frames_with_no_card(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shoot = _write_shoot(
        tmp_path, _no_card(_steel(3) + _crossing("empty") + _crossing("pale") + _steel(BLANK_AFTER))
    )
    config = _without_patch(_config_file(tmp_path))

    code = main(station_args(config, shoot), clock=FakeClock(step=1 / 256))

    assert code == 0
    assert "events 2 · pass 1 · defect 1 · no_verdict 0" in capsys.readouterr().out
    records = rows(tmp_path / "evidence" / "qs.db", "SELECT * FROM inspection ORDER BY seq")
    assert [r["label"] for r in records] == ["defect", "pass"]
    assert [r["reference_patch"] for r in records] == [None, None]
