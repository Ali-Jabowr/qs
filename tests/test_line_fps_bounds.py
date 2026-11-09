"""A bundle's recorded `line_fps` must be a finite number above 0 (#120).

`tests/test_line_fps.py` pins 0, a bool and a string; these pin the rest of the guard in
`qs.inspector.bundle._line_fps`: infinity and NaN (the `isfinite` part) and a negative number
(the `<= 0` part). A YAML hand edit can write `.inf` or `.nan`, and either would make the gap in
frames meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.errors import ConfigError
from qs.inspector.bundle import load_bundle
from test_bundle import _rehashed
from test_line_provenance import _train


@pytest.mark.parametrize(
    ("line_fps", "match"),
    [
        (float("inf"), "line_fps: inf is not a finite number above 0"),
        (float("nan"), "line_fps: nan is not a finite number above 0"),
        (0, "line_fps: 0 is not a finite number above 0"),
        (-10.0, "line_fps: -10.0 is not a finite number above 0"),
    ],
)
def test_a_line_fps_that_is_not_a_finite_number_above_0_is_refused(
    tmp_path: Path, line_fps: float, match: str
) -> None:
    bundle = _train(tmp_path)
    edited = _rehashed(bundle.parent, bundle.name, {"line_fps": line_fps})

    with pytest.raises(ConfigError, match=match):
        load_bundle(bundle.parent, edited)
