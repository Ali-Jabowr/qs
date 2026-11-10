"""The appearance check of a warmth station runs on the burst's evidence frame (#132).

Three frames show the same empty cup at x 130, 170 and 210, warmed by 0, 12 and 30 over red, so
the median is 12 and the middle frame is the evidence. The appearance window is 128 px centred on
the rim, so on the middle frame it starts at x 170 - 64 = 106. The scorer is `test_appearance`'s
constant fake: no GPU, no weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.config import load_config
from qs.contracts import CHECK_APPEARANCE, CHECK_EMPTY_CUP, Frame, Label
from qs.inspector.bundle import BundleInfo, ImageRecord, write_bundle
from qs.inspector.line import LineInspector
from station_fixtures import PATCH_GREY, VERIFIED, event
from test_appearance import ConstantScorer, Factory, _bank
from test_cup import drawn_frame
from test_line_warmth_score import _warmed
from warmth_fixtures import warmth_config

HASH = "c" * 64


def test_the_appearance_check_runs_on_the_frame_closest_to_the_median(tmp_path: Path) -> None:
    models = tmp_path / "models"
    images = tuple(
        ImageRecord(f"line.avi#{i}", f"cup{i:03d}", f"{i:02d}".ljust(64, "b")) for i in (1, 2, 3)
    )
    info = BundleInfo(input_px=64, window_px=128, coreset_ratio=0.1, seed=42,
                      device="cuda", data_source="own", images=images)
    model_id = write_bundle(models, "mochi", _bank(), info)
    path = warmth_config(tmp_path, model=model_id, checks=("empty_cup", "appearance"))
    config = load_config(path).config
    inspector = LineInspector(
        HASH, models, make_scorer=Factory(ConstantScorer(1.0, pred_score=0.75)),
        cuda_available=lambda: True,
    )
    inspector.load(config)
    frames = tuple(
        Frame(event(1), _warmed(drawn_frame("empty", centre=(x, 120), seed=x), (x, 120), red),
              0.0, VERIFIED, PATCH_GREY / 255, 5.0)
        for x, red in ((130, 0), (170, 12), (210, 30))
    )

    verdict = inspector.inspect_burst(frames)

    assert [check.name for check in verdict.checks] == [CHECK_EMPTY_CUP, CHECK_APPEARANCE]
    assert verdict.label is Label.PASS
    assert verdict.evidence_index == 1
    assert verdict.window_box is not None
    assert verdict.window_box[0] == pytest.approx(106, abs=3)
