"""The line inspector's appearance check, with a fake scorer, on drawn cups (#109).

A line config listing `[empty_cup, appearance]` names a small random bundle written into
`tmp_path`. The scorer is the numpy fake of `test_appearance`: its anomaly map is one constant
value, so the masked maximum, the score, is exactly that value. The drawn cup's rim is centred
on (130, 120) with half-axes of 50 px; the window is 128 px, so it starts at (66, 56), and its
corners are about 90 px from the centre: outside the cup's inside dilated by 12 px, so masked
to 0. The threshold is the example's, 1.83. No GPU, no weights, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from qs.config import LoadedConfig, load_config
from qs.contracts import CHECK_APPEARANCE, CHECK_EMPTY_CUP, Frame, Label
from qs.inspector.bundle import BundleInfo, ImageRecord, write_bundle
from qs.inspector.line import LineInspector
from station_fixtures import PATCH_GREY, VERIFIED, config_mapping, event
from test_appearance import ConstantScorer, Factory, _bank
from test_cup import drawn_frame

HASH = "c" * 64
WINDOW_PX = 128
INPUT_PX = 64


@pytest.fixture
def models(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def loaded(tmp_path: Path, models: Path) -> LoadedConfig:
    images = tuple(
        ImageRecord(f"line.avi#{i}", f"cup{i:03d}", f"{i:02d}".ljust(64, "b")) for i in (1, 2, 3)
    )
    info = BundleInfo(input_px=INPUT_PX, window_px=WINDOW_PX, coreset_ratio=0.1, seed=42,
                      device="cuda", data_source="own", images=images)
    model_id = write_bundle(models, "mochi", _bank(), info)
    data = config_mapping(tmp_path, inspector_type="line", model=model_id)
    data["inspector"]["checks"] = ["empty_cup", "appearance"]
    data["inspector"]["line"] = {
        "zone": [20, 0, 240, 240],
        "rim_axis_px": [50, 140],
        "blob_share_min": 0.3,
    }
    path = tmp_path / "line.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(path)


def _inspector(loaded: LoadedConfig, models: Path, scorer: ConstantScorer) -> LineInspector:
    inspector = LineInspector(
        HASH, models, make_scorer=Factory(scorer), cuda_available=lambda: True
    )
    inspector.load(loaded.config)
    return inspector


def _frame(pixels: np.ndarray) -> Frame:
    return Frame(event(1), pixels, 0.0, VERIFIED, PATCH_GREY / 255, 5.0)


def test_a_full_cup_with_a_low_score_passes_with_both_checks(
    loaded: LoadedConfig, models: Path
) -> None:
    inspector = _inspector(loaded, models, ConstantScorer(1.0, pred_score=0.75))

    verdict = inspector.inspect(_frame(drawn_frame("pale")))

    assert verdict.label is Label.PASS
    empty_cup, appearance = verdict.checks
    assert empty_cup.name == CHECK_EMPTY_CUP and empty_cup.passed is True
    assert appearance.name == CHECK_APPEARANCE and appearance.passed is True
    assert appearance.value == 1.0
    assert appearance.limit == 1.83
    assert appearance.measures == (("pred_score", 0.75),)
    assert verdict.window_box == (66, 56, WINDOW_PX, WINDOW_PX)
    assert verdict.heatmap is not None
    assert verdict.heatmap.shape == (WINDOW_PX, WINDOW_PX)
    assert verdict.heatmap[WINDOW_PX // 2, WINDOW_PX // 2] == 1.0  # inside the cup
    assert verdict.heatmap[0, 0] == 0.0  # steel beyond the rim, masked off
    assert verdict.model_id == inspector.model_id == loaded.config.inspector.model


def test_a_full_cup_with_a_high_score_is_a_defect_on_appearance(
    loaded: LoadedConfig, models: Path
) -> None:
    inspector = _inspector(loaded, models, ConstantScorer(5.0))

    verdict = inspector.inspect(_frame(drawn_frame("pale")))

    assert verdict.label is Label.DEFECT
    empty_cup, appearance = verdict.checks
    assert empty_cup.passed is True
    assert appearance.passed is False
    assert appearance.value == 5.0
    assert appearance.failed_on == "score>threshold"


def test_an_empty_cup_still_runs_the_appearance_check(
    loaded: LoadedConfig, models: Path
) -> None:
    scorer = ConstantScorer(1.0)
    inspector = _inspector(loaded, models, scorer)

    verdict = inspector.inspect(_frame(drawn_frame("empty")))

    assert verdict.label is Label.DEFECT
    assert [check.name for check in verdict.checks] == [CHECK_EMPTY_CUP, CHECK_APPEARANCE]
    assert verdict.checks[0].passed is False
    assert len(scorer.inputs) == 2  # the warm-up at load, then this cup
