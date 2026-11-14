"""The threshold rule (#158): hand-worked on 300 scores, then the file and the command.

The drawn clips are plain grey 64x48 frames at 10 fps, 120 frames. The lane is [8, 4, 32, 32].
`C-1.mkv` (calibrate) has full passes at frames 20, 60 and 100; `T-1.mkv` (train) and `X-1.mkv`
(test) are never to be opened.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from qs.__main__ import build_parser
from qs.cmd import threshold_lane
from qs.lane.lanefile import Lane
from qs.lane.model import build, write_lane_bundle
from qs.lane.split import Split, file_sha256, write_split
from qs.lane.threshold import Threshold, lane_threshold, load_threshold, write_threshold

FRAMES = 120
Probs = Callable[..., np.ndarray]


@pytest.mark.parametrize(
    ("scores", "far", "expected"),
    [
        ([0.01] * 298 + [0.6, 0.7], 1, 0.5),
        ([0.01] * 297 + [0.6, 0.7, 0.8], 1, 0.5),
        ([0.01] * 295 + [0.55, 0.6, 0.7, 0.8, 0.9], 1, 0.6),
        ([0.01] * 300, 1, 0.5),
        ([0.01] * 50 + [0.9], 1, 0.9),
    ],
)
def test_the_threshold_is_half_unless_more_than_far_percent_lie_above_it(
    scores: list[float], far: int, expected: float
) -> None:
    assert lane_threshold(scores, far) == expected


def test_no_full_pass_raises_naming_the_reason() -> None:
    with pytest.raises(ValueError, match="no full pass to set the threshold on"):
        lane_threshold([], 1)


def _threshold() -> Threshold:
    return Threshold(
        model_id="lane-mochi-v1", far_pct=1, value=0.6, n_full=300, n_above_half=5, n_unseen=2,
        labels_sha256="a" * 64,
    )


def test_a_written_threshold_loads_equal(tmp_path: Path) -> None:
    write_threshold(_threshold(), tmp_path / "t.yml")
    assert load_threshold(tmp_path / "t.yml") == _threshold()


def _write_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (64, 48))
    assert writer.isOpened()
    for n in range(FRAMES):
        pixels = np.full((48, 64, 3), 128, dtype=np.uint8)
        pixels[10:20, 10:20] = n
        writer.write(pixels)
    writer.release()


def _setup(
    root: Path,
    calibrate: tuple[tuple[str, int], ...] = (("full", 20), ("full", 60), ("full", 100)),
    test: tuple[tuple[str, int], ...] = (("full", 7),),
) -> dict[str, Path]:
    footage = root / "footage"
    footage.mkdir()
    (footage / "T-1.mkv").write_bytes(b"not a video: train")
    roles = {"T-1.mkv": "train"}
    cups = [{"video": "T-1.mkv", "order": 1, "state": "full", "frame": 5}]
    if calibrate:
        _write_clip(footage / "C-1.mkv")
        roles["C-1.mkv"] = "calibrate"
        cups += [{"video": "C-1.mkv", "order": i + 1, "state": s, "frame": f}
                 for i, (s, f) in enumerate(calibrate)]
    if test:
        roles["X-1.mkv"] = "test"  # the test clip is not even on disk
        cups += [{"video": "X-1.mkv", "order": i + 1, "state": s, "frame": f}
                 for i, (s, f) in enumerate(test)]
    labels = root / "L.json"
    labels.write_text(json.dumps({"version": 1, "source": "L", "frames": {}, "cups": cups}))
    lane = root / "lane.yml"
    lane.write_text(yaml.safe_dump({"version": 1, "window": [8, 4, 32, 32], "fps": 10}))
    hashes = {v: "f" * 64 if v == "X-1.mkv" else file_sha256(footage / v) for v in roles}
    split = root / "split.yml"
    write_split(Split({v: (r, hashes[v]) for v, r in roles.items()}), split)
    return {"footage": footage, "labels": labels, "lane": lane, "split": split}


def _bundle(root: Path, empty_bias: float | None = None) -> str:
    torch.manual_seed(0)
    model = build(None)
    if empty_bias is not None:
        with torch.no_grad():
            model.fc.weight.zero_()  # type: ignore[operator]
            model.fc.bias.copy_(torch.tensor([0.0, empty_bias, 0.0]))  # type: ignore[operator]
    record = {"classes": ["full", "empty", "none"], "input_px": 32, "seed": 1}
    return write_lane_bundle(root / "models", "mochi", model, record)


def _probs(runs: dict[int, float]) -> Probs:
    """A stand-in for the model: frames within 2 of a key are cup frames with that P(empty)."""

    def probs(video: Path, lane: Lane, model: object, device: str, *, input_px: int) -> np.ndarray:
        rows = np.tile(np.array([0.0, 0.0, 1.0]), (FRAMES, 1))
        for centre, empty in runs.items():
            rows[max(centre - 2, 0) : centre + 3] = [1.0 - empty, empty, 0.0]
        return rows

    return probs


def _run(root: Path, paths: dict[str, Path], model_id: str, probs: Probs | None = None) -> int:
    args = build_parser().parse_args([
        "threshold-lane", "--bundle", model_id, "--lane", str(paths["lane"]),
        "--labels", str(paths["labels"]), "--footage", str(paths["footage"]),
        "--split", str(paths["split"]), "--out", str(root / "threshold.yml"),
        "--models", str(root / "models"), "--device", "cpu",
    ])
    if probs is None:
        return threshold_lane.main(args)
    return threshold_lane.main(args, probs=probs)


def test_a_pass_without_a_run_is_counted_unseen_and_not_scored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    assert _run(tmp_path, paths, _bundle(tmp_path), _probs({20: 0.1, 100: 0.2})) == 0
    out = capsys.readouterr().out
    assert "full passes used 2" in out
    assert "above 0.5: 0" in out
    assert "n_unseen 1" in out
    assert "threshold 0.5" in out
    saved = load_threshold(tmp_path / "threshold.yml")
    assert (saved.n_full, saved.n_above_half, saved.n_unseen, saved.value) == (2, 0, 1, 0.5)


def test_the_file_carries_the_bundle_id_and_the_labels_hash(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    model_id = _bundle(tmp_path)
    assert _run(tmp_path, paths, model_id, _probs({20: 0.1, 60: 0.1, 100: 0.2})) == 0
    saved = load_threshold(tmp_path / "threshold.yml")
    assert saved.model_id == model_id == "lane-mochi-v1"
    assert saved.labels_sha256 == hashlib.sha256(paths["labels"].read_bytes()).hexdigest()
    assert saved.far_pct == 1


def test_the_real_model_scores_a_run_end_to_end(tmp_path: Path) -> None:
    paths = _setup(tmp_path, calibrate=(("full", 60),))
    assert _run(tmp_path, paths, _bundle(tmp_path, empty_bias=5.0)) == 0
    saved = load_threshold(tmp_path / "threshold.yml")
    assert (saved.n_full, saved.n_above_half, saved.n_unseen) == (1, 1, 0)
    assert saved.value == pytest.approx(0.98670, abs=1e-4)


def test_without_a_calibrate_clip_the_test_clip_is_never_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path, calibrate=())
    assert _run(tmp_path, paths, _bundle(tmp_path), _probs({})) == 2
    assert "no calibrate clip in the split file" in capsys.readouterr().out
    assert not (tmp_path / "threshold.yml").exists()


def test_a_calibrate_clip_with_no_full_pass_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path, calibrate=(("empty", 20), ("gap", 60)))
    assert _run(tmp_path, paths, _bundle(tmp_path), _probs({20: 0.9})) == 1
    assert "refused: no full pass to set the threshold on" in capsys.readouterr().out
    assert not (tmp_path / "threshold.yml").exists()


def test_an_existing_output_file_is_not_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    (tmp_path / "threshold.yml").write_text("keep me")
    assert _run(tmp_path, paths, _bundle(tmp_path), _probs({20: 0.1})) == 2
    assert "already exists" in capsys.readouterr().out
    assert (tmp_path / "threshold.yml").read_text() == "keep me"


def test_a_calibrate_clip_changed_since_the_split_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    with (paths["footage"] / "C-1.mkv").open("ab") as clip:
        clip.write(b"x")
    assert _run(tmp_path, paths, _bundle(tmp_path), _probs({20: 0.1})) == 2
    assert "C-1.mkv changed since the split" in capsys.readouterr().out
