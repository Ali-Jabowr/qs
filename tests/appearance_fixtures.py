"""Helpers for the `AppearanceInspector` tests: a config naming a bundle, and a `Frame`.

The config is the example's, rewritten in `tmp_path` so that `inspector.model` and
`threshold.set_on.model` both name the bundle under test, as `qs calibrate` (P08) will
write them. Nothing here needs a camera, a display or a GPU.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from conftest import EXAMPLE
from qs.config import LoadedConfig, load_config
from qs.contracts import CameraReadback, Frame, TriggerEvent

EXAMPLE_MODEL = "patchcore-mochi-v1-3f9a1c2b7d4e"


def config_naming(
    tmp_path: Path, model_id: str, *, replace: tuple[tuple[str, str], ...] = ()
) -> LoadedConfig:
    """The example config with its model replaced by `model_id`, and any `(old, new)` text
    replacements applied, written to `tmp_path` and loaded."""
    text = EXAMPLE.read_text(encoding="utf-8")
    assert text.count(EXAMPLE_MODEL) == 2, "the example no longer names its model twice"
    text = text.replace(EXAMPLE_MODEL, model_id)
    for old, new in replace:
        assert old in text, f"the example no longer contains {old!r}"
        text = text.replace(old, new)
    path = tmp_path / "station.yml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def frame_of(pixels: np.ndarray, seq: int = 1) -> Frame:
    """A captured frame around `pixels`, as the Pipeline hands it to the Inspector."""
    event = TriggerEvent(seq=seq, t0=100.0, wall=datetime(2026, 9, 26, tzinfo=UTC),
                         source_id="replay:test")
    camera = CameraReadback(exposure_us=10000, gain=0, wb_kelvin=4500, focus=30, verified=True)
    return Frame(event, pixels, 100.03, camera, 0.62, 30.0)
