"""The defects file: which labelled full passes of the test clips are off-colour (#175)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qs.errors import ConfigError
from qs.labels import Labels, load_labels
from qs.lane.defects import Defect, load_defects
from qs.lane.split import Split

SPLIT = Split({"T-1.mkv": ("train", "a"), "E-1.mkv": ("test", "b")})


def _labels(root: Path) -> Labels:
    cups = [{"video": "E-1.mkv", "order": 1, "state": "full_pink", "frame": 50},
            {"video": "E-1.mkv", "order": 2, "state": "empty", "frame": 82},
            {"video": "T-1.mkv", "order": 1, "state": "full", "frame": 50}]
    path = root / "L.json"
    path.write_text(json.dumps({"version": 1, "source": "L", "frames": {}, "cups": cups}))
    return load_labels(path, "L")


def _defects(root: Path, entries: str) -> Path:
    path = root / "defects.yml"
    path.write_text(f"version: 1\ndefects:\n{entries}")
    return path


def test_a_defect_on_a_labelled_full_pass_of_a_test_clip_is_loaded(tmp_path: Path) -> None:
    path = _defects(tmp_path, "  - {video: E-1.mkv, frame: 50, kind: patch, unit: D01}\n")
    assert load_defects(path, _labels(tmp_path), SPLIT) == (Defect("E-1.mkv", 50, "patch", "D01"),)


@pytest.mark.parametrize("kind", ["patch", "colour", "shape"])
def test_the_three_kinds_are_accepted(tmp_path: Path, kind: str) -> None:
    path = _defects(tmp_path, f"  - {{video: E-1.mkv, frame: 50, kind: {kind}, unit: D01}}\n")
    assert load_defects(path, _labels(tmp_path), SPLIT)[0].kind == kind


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("{video: E-1.mkv, frame: 51, kind: patch, unit: D01}", "E-1.mkv frame 51"),
        ("{video: E-1.mkv, frame: 82, kind: patch, unit: D01}", "E-1.mkv frame 82"),
        ("{video: E-1.mkv, frame: 50, kind: dent, unit: D01}", "kind 'dent'"),
        ("{video: T-1.mkv, frame: 50, kind: patch, unit: D01}", "T-1.mkv is a train clip"),
        ("{video: Z-1.mkv, frame: 50, kind: patch, unit: D01}", "Z-1.mkv is not in the split"),
    ],
)
def test_a_defect_that_breaks_the_contract_is_refused_in_one_line(
    tmp_path: Path, entry: str, message: str
) -> None:
    path = _defects(tmp_path, f"  - {entry}\n")
    with pytest.raises(ConfigError, match=message) as refused:
        load_defects(path, _labels(tmp_path), SPLIT)
    assert "\n" not in str(refused.value)
