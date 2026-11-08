"""The line bundle records how it was split, and `qs line-alarms` checks it (#112).

Footage, labels and config are those of `tests/test_line_alarms.py`: one 10 fps video of 40 frames,
all labelled unit; with a 0.2 s gap the split trains on frames 0–27.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import FakePatchCore
from qs.__main__ import build_parser
from qs.cmd import train
from qs.errors import ConfigError
from qs.inspector.bundle import LineRecord, load_bundle
from test_bundle import _rehashed
from test_line_alarms import _config, _footage, _labels


def _train(root: Path) -> Path:
    """Train on all 40 frames with a 0.2 s gap; the bundle's directory."""
    models = root / "models"
    args = build_parser().parse_args([
        "train", "--config", str(_config(root)), "--labels", str(_labels(root)),
        "--footage", str(_footage(root)), "--models", str(models), "--gap-s", "0.2",
        "--window-px", "40", "--input-px", "32", "--product", "mochi-top",
    ])
    assert train.main(args, make_patchcore=lambda device: FakePatchCore(),
                      cuda_available=lambda: True) == 0
    (bundle,) = models.iterdir()
    return bundle


def test_a_line_bundle_records_the_gap_and_the_labels_file_it_was_split_with(
    tmp_path: Path,
) -> None:
    bundle = _train(tmp_path)

    data = yaml.safe_load((bundle / "bundle.yml").read_text(encoding="utf-8"))
    assert data.get("line") == {"gap_s": 0.2, "labels": "top.json"}


def test_the_loaded_bundle_carries_its_split(tmp_path: Path) -> None:
    bundle = _train(tmp_path)

    assert load_bundle(bundle.parent, bundle.name).line == LineRecord(0.2, "top.json")


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ({"gap_s": -1, "labels": "top.json"}, "line.gap_s: -1 is not a finite number at or"),
        ({"gap_s": True, "labels": "top.json"}, "line.gap_s: True is not a finite number"),
        ({"gap_s": 0.2, "labels": ""}, "line.labels: '' is not a file name"),
        ({"gap_s": 0.2}, "line: missing key 'labels'"),
    ],
)
def test_a_malformed_split_record_is_refused(
    tmp_path: Path, line: dict[str, object], match: str
) -> None:
    bundle = _train(tmp_path)
    edited = _rehashed(bundle.parent, bundle.name, {"line": line})

    with pytest.raises(ConfigError, match=match):
        load_bundle(bundle.parent, edited)
