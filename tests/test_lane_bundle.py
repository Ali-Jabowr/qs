"""The lane model, its prediction and its hashed bundle (#156)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qs.errors import ConfigError
from qs.lane.model import build, load_lane_bundle, predict, write_lane_bundle

RECORD = {"classes": ["full", "empty", "none"], "input_px": 64, "seed": 42}


def _inputs(n: int) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n, 3, 64, 64)).astype(np.float32)


def test_predict_gives_one_row_of_three_probabilities_per_input() -> None:
    torch.manual_seed(0)
    probabilities = predict(build(None), _inputs(5), "cpu")
    assert probabilities.shape == (5, 3)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(5), abs=1e-5)


def test_the_loaded_bundle_predicts_what_the_written_model_did(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build(None)
    model_id = write_lane_bundle(tmp_path, "mochi", model, RECORD)
    loaded = load_lane_bundle(tmp_path, model_id)
    assert model_id == "lane-mochi-v1"
    assert loaded.record["input_px"] == 64
    assert loaded.record["classes"] == ["full", "empty", "none"]
    assert predict(loaded.model, _inputs(5), "cpu") == pytest.approx(
        predict(model, _inputs(5), "cpu"), abs=1e-6
    )


def test_the_next_bundle_takes_the_next_free_number(tmp_path: Path) -> None:
    write_lane_bundle(tmp_path, "mochi", build(None), RECORD)
    assert write_lane_bundle(tmp_path, "mochi", build(None), RECORD) == "lane-mochi-v2"


def test_a_bundle_whose_model_changed_by_one_byte_is_refused(tmp_path: Path) -> None:
    model_id = write_lane_bundle(tmp_path, "mochi", build(None), RECORD)
    path = tmp_path / model_id / "model.pt"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ConfigError, match="bundle lane-mochi-v1: model.pt"):
        load_lane_bundle(tmp_path, model_id)
