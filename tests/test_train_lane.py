"""`qs train-lane`: the lane classifier trained on drawn clips, on the CPU (#156)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from qs.__main__ import build_parser
from qs.cmd import train_lane
from qs.labels import CupState, load_labels
from qs.lane.split import Split, write_split

ROOT = Path(__file__).resolve().parent.parent
FRAME_W, FRAME_H = 128, 96
WINDOW = [32, 16, 64, 64]
FIRST, STEP = 40, 32

F, E, G = CupState.FULL, CupState.EMPTY, CupState.GAP
CYCLE = (F, F, E, G)


def _states(n: int) -> list[CupState]:
    return [CYCLE[i % 4] for i in range(n)]


def _frame(state: CupState | None) -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W, 3), 130, np.uint8)
    centre = (WINDOW[0] + WINDOW[2] // 2, WINDOW[1] + WINDOW[3] // 2)
    if state == F:
        cv2.circle(frame, centre, 22, (200, 120, 240), -1)
    if state == E:
        cv2.circle(frame, centre, 22, (30, 30, 30), 5)
    return frame


def _draw_clip(path: Path, states: list[CupState]) -> list[int]:
    """A video whose pass `i` is at frame `FIRST + STEP * i`; the cup shows 3 frames either side."""
    frames = FIRST + STEP * len(states) + 20
    at = {FIRST + STEP * i: s for i, s in enumerate(states)}
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"FFV1"), 30.0, (FRAME_W, FRAME_H))
    for n in range(frames):
        near = [s for f, s in at.items() if abs(f - n) <= 3]
        writer.write(_frame(near[0] if near else None))
    writer.release()
    return sorted(at)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class World:
    root: Path
    footage: Path
    lane: Path
    labels: Path
    split: Path
    models: Path

    def run(self, *extra: str, seed: str = "42", epochs: str = "6") -> int:
        argv = [
            "train-lane", "--lane", str(self.lane), "--labels", str(self.labels),
            "--footage", str(self.footage), "--split", str(self.split), "--product", "mochi",
            "--seed", seed, "--epochs", epochs, "--input-px", "64", "--models", str(self.models),
            "--device", "cpu", *extra,
        ]
        return train_lane.main(build_parser().parse_args(argv), weights=lambda: None)

    def bundle(self, number: int) -> dict[str, Any]:
        return yaml.safe_load((self.models / f"lane-mochi-v{number}" / "bundle.yml").read_text())


def _world(root: Path, states: list[CupState], split_clips: dict[str, str] | None = None) -> World:
    footage = root / "footage"
    footage.mkdir()
    frames = _draw_clip(footage / "T-1.mkv", states)
    (footage / "C-1.mkv").write_bytes(b"calibrate clip, never decoded")
    (footage / "E-1.mkv").write_bytes(b"test clip, never decoded")
    labels = root / "labels.json"
    book = load_labels(labels, "T-1")
    for state, frame in zip(states, frames, strict=True):
        book.answer_cup("T-1.mkv", state, frame)
    book.answer_cup("C-1.mkv", F, 50)
    book.answer_cup("E-1.mkv", E, 60)
    book.answer_cup("E-1.mkv", F, 90)
    lane = root / "lane.yml"
    lane.write_text(yaml.safe_dump({"version": 1, "window": WINDOW, "fps": 30.0}))
    roles = split_clips or {"T-1.mkv": "train", "C-1.mkv": "calibrate", "E-1.mkv": "test"}
    split = root / "split.yml"
    write_split(Split({v: (r, _sha(footage / v)) for v, r in roles.items()}), split)
    return World(root, footage, lane, labels, split, root / "models")


def test_a_clean_run_writes_a_bundle_that_learned_the_three_classes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, _states(40))
    assert world.run() == 0
    record = world.bundle(1)
    assert record["classes"] == ["full", "empty", "none"]
    assert record["seed"] == 42
    assert record["epochs"] == 6
    assert record["input_px"] == 64
    assert record["lane"] == {"window": WINDOW, "fps": 30.0}
    assert record["model_sha256"] == _sha(world.models / "lane-mochi-v1" / "model.pt")
    assert record["lane_sha256"] == _sha(world.lane)
    assert record["labels_sha256"] == _sha(world.labels)
    assert record["split_sha256"] == _sha(world.split)
    assert record["pretrained_sha256"] is None
    assert record["train_clips"] == {"T-1.mkv": _sha(world.footage / "T-1.mkv")}
    confusion = np.array(record["validation_confusion"])
    assert np.trace(confusion) / confusion.sum() >= 0.9
    out = capsys.readouterr().out
    assert "model_id: lane-mochi-v1" in out
    assert "validation confusion" in out


def test_the_same_command_twice_gives_the_same_validation_predictions(tmp_path: Path) -> None:
    world = _world(tmp_path, _states(40))
    assert world.run() == 0
    assert world.run() == 0
    first, second = world.bundle(1), world.bundle(2)
    assert first["validation_predicted"] == second["validation_predicted"]
    assert first["samples_per_class"] == second["samples_per_class"]


def test_the_last_fifth_of_the_passes_is_validation_and_the_rest_trains(tmp_path: Path) -> None:
    world = _world(tmp_path, _states(10))
    assert world.run() == 0
    record = world.bundle(1)
    assert record["validation_passes"] == [["T-1.mkv", 9], ["T-1.mkv", 10]]
    assert np.sum(record["validation_confusion"]) == 16
    assert record["samples_per_class"] == {"full": 20, "empty": 10, "none": 31}


def test_passes_in_calibrate_and_test_clips_are_never_read(tmp_path: Path) -> None:
    world = _world(tmp_path, _states(10))
    assert world.run() == 0
    assert list(world.bundle(1)["train_clips"]) == ["T-1.mkv"]


def test_a_split_with_no_train_clip_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, _states(10), {"C-1.mkv": "calibrate", "E-1.mkv": "test"})
    assert world.run() == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        "train-lane: the split file has no train clip"
    ]


def test_a_train_clip_changed_since_the_split_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, _states(10))
    with (world.footage / "T-1.mkv").open("ab") as handle:
        handle.write(b"x")
    assert world.run() == 2
    out = capsys.readouterr().out
    assert "T-1.mkv changed since the split" in out
    assert not world.models.exists()


def test_without_an_empty_pass_among_the_train_passes_it_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, [F, G] * 5)
    assert world.run() == 1
    assert capsys.readouterr().out.strip().splitlines() == [
        "train-lane: refused: no empty pass to train on"
    ]
    assert not world.models.exists()


@pytest.mark.env
def test_the_pinned_weights_train_one_epoch_on_the_gpu(tmp_path: Path) -> None:
    from qs.lane.weights import resnet18_weights

    world = _world(tmp_path, _states(10))
    argv = [
        "train-lane", "--lane", str(world.lane), "--labels", str(world.labels),
        "--footage", str(world.footage), "--split", str(world.split), "--product", "mochi",
        "--epochs", "1", "--input-px", "64", "--models", str(world.models), "--device", "cuda",
    ]
    code = train_lane.main(
        build_parser().parse_args(argv), weights=lambda: resnet18_weights(ROOT / ".torch")
    )
    assert code == 0
    assert len(world.bundle(1)["pretrained_sha256"]) == 64
