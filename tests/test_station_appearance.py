"""`inspector.type: appearance` through `qs station --replay`: P07's `AppearanceInspector`
behind the same Pipeline as the stub (docs/11 §4).

The frames are 1920×1080 synthetic discs on P06's black floor, with the grey card painted
into the reference patch. The example config is rewritten to name a bundle written with a
fake bank into `tmp_path/models`, and the test runs from `tmp_path`, so the station's
`models/` and `evidence/` are both there. The scorer is a numpy fake whose anomaly map is one
constant: `LOW`, or `HIGH` when the window's centre carries the bright mark painted on the
one unit scripted to fail. Both values are exact in float32 and far from the threshold of
1.83, so no verdict here sits on the float32 rounding boundary. No GPU, no weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from appearance_fixtures import config_naming
from evidence_records import fast_sqlite
from qs.cmd.station import main
from qs.config import load_config
from qs.errors import ConfigError
from qs.inspector.model import Scored
from station_fixtures import disc_frame, rows, station_args, write_fake_bundle, write_png

LOW = 1.0
HIGH = 4.0
# Channel 0 of `window.to_input` is (grey / 255 − 0.485) / 0.229: 1.31 on the unit's grey
# 200, 2.16 on the mark's 250.
MARK_LEVEL = 1.8


class MarkScorer:
    """A scorer whose map is `HIGH` everywhere when the input's centre pixel is the mark, and
    `LOW` otherwise. Stateless: a frame's score depends on that frame alone."""

    def score(self, window_input: np.ndarray) -> Scored:
        _, height, width = window_input.shape
        value = HIGH if window_input[0, height // 2, width // 2] > MARK_LEVEL else LOW
        anomaly_map = np.full((height, width), value, dtype=np.float32)
        return Scored(anomaly_map, value, backbone_ms=1.0, search_ms=2.0)


class Factory:
    """`make_scorer`: records the device and bank it was asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray]] = []

    def __call__(self, device: str, bank: np.ndarray) -> MarkScorer:
        self.calls.append((device, bank))
        return MarkScorer()


FRAMES = (  # (file, pixels, expected label, expected reason)
    ("f01.png", disc_frame(190), "pass", None),
    ("f02.png", disc_frame(185, (930, 520)), "pass", None),
    ("f03.png", disc_frame(190, mark=True), "defect", None),
    ("f04.png", disc_frame(195, (990, 560)), "pass", None),
    ("f05.png", disc_frame(None), "no_verdict", "placement"),
    ("f06.png", disc_frame(188), "pass", None),
)


@pytest.fixture
def station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, Path]:
    """From `tmp_path`: the config naming a fresh fake bundle, the bundle's `model_id`, and the
    six frames of `FRAMES` in `shoot/`."""
    fast_sqlite(monkeypatch)
    monkeypatch.chdir(tmp_path)
    model_id = write_fake_bundle(tmp_path / "models")
    config_naming(tmp_path, model_id)
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    for name, pixels, _, _ in FRAMES:
        write_png(shoot / name, pixels)
    return tmp_path / "station.yml", model_id, shoot


def test_appearance_replay_gives_decided_verdicts_with_the_bundles_model_id(
    tmp_path: Path, station: tuple[Path, str, Path]
) -> None:
    config, model_id, shoot = station
    factory = Factory()
    code = main(station_args(config, shoot), make_scorer=factory, cuda_available=lambda: True)
    assert code == 0

    [(device, bank)] = factory.calls  # one scorer, on the backend's device, the bundle's bank
    assert device == "cuda"
    assert bank.shape == (20, 8)
    database = tmp_path / "evidence" / "qs.db"
    records = rows(database, "SELECT * FROM inspection ORDER BY seq")
    assert [(r["label"], r["reason"]) for r in records] == [
        (label, reason) for _, _, label, reason in FRAMES
    ]
    threshold_id = load_config(config).config.threshold.id
    assert all(r["model_id"] == model_id for r in records)
    assert all(r["threshold_id"] == threshold_id for r in records)

    defect = records[2]
    assert defect["heatmap_path"] is not None
    heatmap = tmp_path / "evidence" / defect["heatmap_path"]
    assert heatmap.read_bytes().startswith(b"\x89PNG")
    checks = {
        c["name"]: c
        for c in rows(database, "SELECT * FROM inspection_check WHERE inspection_id = ?",
                      (defect["id"],))
    }
    assert checks["geometry"]["passed"] == 1
    appearance = checks["appearance"]
    assert appearance["passed"] == 0
    assert appearance["failed_on"] == "score>threshold"
    assert (appearance["value"], appearance["limit_value"]) == (HIGH, 1.83)

    for record in (records[0], records[1], records[3], records[5]):
        [passed] = rows(
            database,
            "SELECT * FROM inspection_check WHERE inspection_id = ? AND name = 'appearance'",
            (record["id"],),
        )
        assert (passed["passed"], passed["value"]) == (1, LOW)
    placement = records[4]
    assert placement["image_status"] == "stored"  # the empty floor keeps its image
    assert rows(database, "SELECT * FROM inspection_check WHERE inspection_id = ?",
                (placement["id"],)) == []


def test_a_scorer_factory_that_refuses_stops_the_station_with_no_evidence(
    tmp_path: Path, station: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, shoot = station

    def refuses(device: str, bank: np.ndarray) -> MarkScorer:
        raise ConfigError(
            "inspector.backend: torch-cuda, but CUDA is not available — no CPU fallback "
            "(docs/16 §3)"
        )

    capsys.readouterr()
    assert main(station_args(config, shoot), make_scorer=refuses,
                cuda_available=lambda: True) == 2
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "no CPU fallback" in out, out
    assert "Traceback" not in out
    assert not (tmp_path / "evidence").exists()


def test_torch_cuda_without_cuda_is_refused_and_no_scorer_is_built_on_any_device(
    tmp_path: Path, station: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, shoot = station
    factory = Factory()
    capsys.readouterr()
    assert main(station_args(config, shoot), make_scorer=factory,
                cuda_available=lambda: False) == 2
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "no CPU fallback" in out, out
    assert factory.calls == []
    assert not (tmp_path / "evidence").exists()
