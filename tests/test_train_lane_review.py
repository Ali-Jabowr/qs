"""`qs train-lane` after review: seeding, refusals as one line, the between rule, bundle checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from qs.__main__ import run
from qs.cmd import train_lane
from qs.errors import ConfigError
from qs.inspector.window import to_input
from qs.labels import load_labels
from qs.lane.lanefile import Lane, crop
from qs.lane.model import build, load_lane_bundle, write_lane_bundle
from test_train_lane import WINDOW, E, F, G, _states, _world


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_seed_decides_the_model(tmp_path: Path) -> None:
    world = _world(tmp_path, _states(10))
    assert world.run(seed="42") == 0
    assert world.run(seed="42") == 0
    assert world.run(seed="7") == 0
    first, again, other = (world.bundle(n)["model_sha256"] for n in (1, 2, 3))
    assert first == again
    assert first != other


def test_a_second_key_on_a_labelled_frame_is_refused_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, _states(10))
    load_labels(world.labels, "T-1").answer_cup("T-1.mkv", E, 328)
    assert world.run() == 1
    assert capsys.readouterr().out.strip().splitlines() == [
        "train-lane: refused: T-1.mkv: pass 10 (full) and pass 11 (empty) "
        "both claim frame 326 with different classes"
    ]


def test_a_training_failure_is_one_line_and_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def out_of_memory(*_: Any) -> None:
        raise RuntimeError("CUDA out of memory\nTried to allocate 2 GiB")

    monkeypatch.setattr(train_lane, "_train_model", out_of_memory)
    world = _world(tmp_path, _states(10))
    assert world.run() == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        "train-lane: training failed: CUDA out of memory"
    ]
    assert not world.models.exists()


def test_without_any_none_sample_it_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, [F, E] * 5)
    assert world.run("--between", "40", "60") == 1
    assert capsys.readouterr().out.strip().splitlines() == [
        "train-lane: refused: no none sample to train on: label a gap, or widen --between"
    ]


def test_the_between_rule_can_be_set_and_is_recorded(tmp_path: Path) -> None:
    world = _world(tmp_path, [F, E] * 5)
    assert world.run() == 0
    assert world.bundle(1)["between"] == [16, 48]
    assert world.bundle(1)["samples_per_class"]["none"] == 21


def test_without_a_full_pass_it_refuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    world = _world(tmp_path, [E, G] * 5)
    assert world.run() == 1
    assert capsys.readouterr().out.strip().splitlines() == [
        "train-lane: refused: no full pass to train on"
    ]


def test_a_validation_without_an_empty_pass_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, [F, E, F, E, F, E, F, E, F, F])
    assert world.run() == 0
    out = capsys.readouterr().out
    assert "validation samples per class: full 10, empty 0, none 6" in out
    assert "WARN: no empty sample in validation" in out
    assert world.bundle(1)["validation_per_class"] == {"full": 10, "empty": 0, "none": 6}


def test_the_training_constants_are_in_the_bundle(tmp_path: Path) -> None:
    world = _world(tmp_path, _states(10))
    assert world.run() == 0
    record = world.bundle(1)
    assert record["batch_size"] == 32
    assert record["learning_rate"] == 1e-4
    assert record["shift_px"] == 8
    assert record["brightness"] == [0.9, 1.1]


def test_the_inputs_are_hashed_when_they_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, _states(10))
    before = (_sha(world.lane), _sha(world.labels), _sha(world.split))
    real = train_lane._train_model

    def meanwhile(*args: Any) -> None:
        world.lane.write_text(world.lane.read_text() + "\n# edited\n")
        world.labels.write_text(world.labels.read_text() + "\n")
        world.split.write_text(world.split.read_text() + "\n# edited\n")
        real(*args)

    monkeypatch.setattr(train_lane, "_train_model", meanwhile)
    assert world.run() == 0
    record = world.bundle(1)
    assert (record["lane_sha256"], record["labels_sha256"], record["split_sha256"]) == before


@pytest.mark.parametrize("window", [WINDOW, [0, 0, 64, 64], [64, 32, 64, 64]])
def test_an_unshifted_input_is_what_inference_computes(window: list[int]) -> None:
    frame = np.random.default_rng(1).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    lane = Lane((window[0], window[1], window[2], window[3]), 30.0)
    padded = train_lane._padded_crop(frame, lane)
    assert np.array_equal(
        train_lane._model_input(padded, lane, 64), to_input(crop(frame, lane), 64)
    )


def test_cuda_asked_for_and_missing_is_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    world = _world(tmp_path, _states(10))
    assert world.run("--device", "cuda") == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        "train-lane: device: --device cuda, but torch sees no CUDA device"
    ]


def test_the_command_is_dispatched_by_the_qs_entry_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, _states(10), {"C-1.mkv": "calibrate", "E-1.mkv": "test"})
    code = run([
        "train-lane", "--lane", str(world.lane), "--labels", str(world.labels),
        "--footage", str(world.footage), "--split", str(world.split), "--product", "mochi",
        "--models", str(world.models),
    ])
    assert code == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        "train-lane: the split file has no train clip"
    ]


@pytest.mark.parametrize(
    "record",
    [
        {"classes": ["empty", "full", "none"], "input_px": 64},
        {"classes": ["full", "empty", "none"], "input_px": 0},
        {"classes": ["full", "empty", "none"], "input_px": "64"},
        {"classes": ["full", "empty", "none"], "input_px": True},
        {"classes": ["full", "empty", "none"]},
    ],
)
def test_a_bundle_with_the_wrong_classes_or_input_size_is_refused(
    tmp_path: Path, record: dict[str, Any]
) -> None:
    model_id = write_lane_bundle(tmp_path, "mochi", build(None), record)
    with pytest.raises(ConfigError, match="bundle lane-mochi-v1: "):
        load_lane_bundle(tmp_path, model_id)
