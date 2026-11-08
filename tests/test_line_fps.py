"""A line bundle records the fps it was split at, and `qs line-alarms` checks it (#112).

The time split's gap in frames is `round(gap_s * fps)`: trained at 10 fps with a 0.2 s gap, frames
0–27 train and calibration starts at 30; read at 5 fps the gap is one frame and calibration starts
at 29, two frames from trained frame 27, with the training block unchanged. Footage, labels and
config are those of `tests/test_line_provenance.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qs.errors import ConfigError
from qs.inspector.bundle import load_bundle
from test_bundle import _rehashed
from test_line_provenance import _alarms, _train


def _set_fps(root: Path, fps: float) -> None:
    """Rewrite the line config `_train` used with another `inspector.line.fps`."""
    path = root / "line.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["inspector"]["line"]["fps"] = fps
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_a_line_bundle_records_the_fps_it_was_split_at(tmp_path: Path) -> None:
    bundle = _train(tmp_path)

    data = yaml.safe_load((bundle / "bundle.yml").read_text(encoding="utf-8"))
    assert data.get("line_fps") == 10.0
    assert load_bundle(bundle.parent, bundle.name).line_fps == 10.0


def test_a_config_fps_other_than_the_bundles_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _train(tmp_path)
    _set_fps(tmp_path, 5.0)
    capsys.readouterr()

    assert _alarms(tmp_path, bundle.name) == 1
    assert capsys.readouterr().out.strip() == (
        "refused: inspector.line.fps 5 is not the 10 fps the model was split at, so the gap in "
        "frames would differ from training; use the fps it was trained with"
    )


def test_a_bundle_with_no_recorded_fps_warns_and_still_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _train(tmp_path)
    older = _rehashed(bundle.parent, bundle.name, drop="line_fps")
    _set_fps(tmp_path, 5.0)
    capsys.readouterr()

    assert _alarms(tmp_path, older) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "WARNING: the bundle records no line_fps (made before #112), so inspector.line.fps is "
        "not checked against training"
    )


@pytest.mark.parametrize(
    ("line_fps", "match"),
    [
        (0, "line_fps: 0 is not a finite number above 0"),
        (True, "line_fps: True is not a finite number above 0"),
        ("ten", "line_fps: 'ten' is not a finite number above 0"),
    ],
)
def test_a_malformed_fps_record_is_refused(tmp_path: Path, line_fps: object, match: str) -> None:
    bundle = _train(tmp_path)
    edited = _rehashed(bundle.parent, bundle.name, {"line_fps": line_fps})

    with pytest.raises(ConfigError, match=match):
        load_bundle(bundle.parent, edited)
