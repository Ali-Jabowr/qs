"""A line station that leaves out its wording gets the line's, never the bench's (#104)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from qs.config import load_config
from station_fixtures import config_mapping


def _written(root: Path, *, line: bool, display: dict[str, Any]) -> Path:
    data = config_mapping(root, inspector_type="line" if line else "stub")
    if line:
        data["inspector"]["checks"] = ["empty_cup"]
        data["inspector"]["line"] = {
            "zone": [20, 0, 240, 240], "rim_axis_px": [50, 140], "blob_share_min": 0.3,
        }
    data["display"].update(display)
    path = root / "station.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("line", "display", "ready", "placement"),
    [
        (True, {}, "ГОТОВ", "стаканчик не проверен"),
        (True, {"ready_line": "ЖДУ", "placement_line": "мимо зоны"}, "ЖДУ", "мимо зоны"),
        (False, {}, None, None),
    ],
)
def test_a_line_station_without_its_wording_gets_the_lines(
    tmp_path: Path, line: bool, display: dict[str, Any], ready: str | None,
    placement: str | None,
) -> None:
    loaded = load_config(_written(tmp_path, line=line, display=display)).config.display
    assert (loaded.ready_line, loaded.placement_line) == (ready, placement)
