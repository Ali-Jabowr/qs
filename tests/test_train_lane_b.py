"""`qs train-lane-b`: PatchCore fitted on lane crops of the train clips' full passes (#159).

The drawn clips are plain grey 64x48 frames at 10 fps; frame `n` carries a stamp of brightness
`n` in its top-left corner so that every frame has its own pixels. The lane is the window
[8, 4, 32, 32]. `T-1.mkv` has four passes: full@50, empty@82, full_pink@114, gap@146.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from conftest import FakePatchCore
from qs.__main__ import build_parser
from qs.cmd import train_lane_b
from qs.errors import InferenceError
from qs.inspector.bundle import load_bundle
from qs.inspector.model import Scored
from qs.lane.lanefile import Lane
from qs.lane.split import Split, file_sha256, write_split

FRAMES = 150
T1_PASSES = [("full", 50), ("empty", 82), ("full_pink", 114), ("gap", 146)]


def _write_clip(path: Path, frames: int = FRAMES) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (64, 48))
    assert writer.isOpened()
    for n in range(frames):
        pixels = np.full((48, 64, 3), 128, dtype=np.uint8)
        pixels[10:20, 10:20] = n
        writer.write(pixels)
    writer.release()


def _setup(
    root: Path,
    t1: list[tuple[str, int]] = T1_PASSES,
    others: tuple[tuple[str, str, str, int], ...] = (),
) -> dict[str, Path]:
    """Clips T-1 (train) plus `others` as (clip, role, state, frame); the labels, lane, split."""
    footage = root / "footage"
    footage.mkdir()
    _write_clip(footage / "T-1.mkv")
    roles = {"T-1.mkv": "train"}
    cups = [{"video": "T-1.mkv", "order": i + 1, "state": s, "frame": f}
            for i, (s, f) in enumerate(t1)]
    for clip, role, state, frame in others:
        (footage / f"{clip}.mkv").write_bytes(f"not a video: {clip}".encode())
        roles[f"{clip}.mkv"] = role
        cups.append({"video": f"{clip}.mkv", "order": 1, "state": state, "frame": frame})
    labels = root / "L.json"
    labels.write_text(json.dumps({"version": 1, "source": "L", "frames": {}, "cups": cups}))
    lane = root / "lane.yml"
    lane.write_text(yaml.safe_dump({"version": 1, "window": [8, 4, 32, 32], "fps": 10}))
    split = root / "split.yml"
    write_split(Split({v: (r, file_sha256(footage / v)) for v, r in roles.items()}), split)
    return {"footage": footage, "labels": labels, "lane": lane, "split": split}


def _run(root: Path, paths: dict[str, Path], fake: FakePatchCore) -> int:
    args = build_parser().parse_args([
        "train-lane-b", "--lane", str(paths["lane"]), "--labels", str(paths["labels"]),
        "--footage", str(paths["footage"]), "--split", str(paths["split"]),
        "--product", "mochi-lane", "--input-px", "32", "--models", str(root / "models"),
    ])
    return train_lane_b.main(args, make_patchcore=lambda device: fake, cuda_available=lambda: True)


def _bundle(root: Path) -> dict[str, object]:
    (bundle,) = (root / "models").iterdir()
    loaded: dict[str, object] = yaml.safe_load((bundle / "bundle.yml").read_text("utf-8"))
    return loaded


def test_each_full_pass_of_a_train_clip_is_one_fitted_image(tmp_path: Path) -> None:
    fake = FakePatchCore()
    assert _run(tmp_path, _setup(tmp_path), fake) == 0
    (windows, settings), = fake.fits
    assert len(windows) == 2
    assert (settings.coreset_ratio, settings.seed, settings.device) == (0.1, 42, "cuda")
    bundle = _bundle(tmp_path)
    assert [i["file"] for i in bundle["images"]] == ["T-1.mkv#50", "T-1.mkv#114"]  # type: ignore[attr-defined]
    assert bundle["units"] == ["T-1.mkv"]
    assert bundle["window_px"] == 32


def test_a_frame_with_no_rim_is_still_fitted_and_the_rim_finder_is_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("the rim finder was called")

    monkeypatch.setattr("qs.inspector.cup.find_cup", never_called)
    fake = FakePatchCore()
    assert _run(tmp_path, _setup(tmp_path, [("full", 50)]), fake) == 0
    assert len(fake.fits[0][0]) == 1


@pytest.mark.parametrize("role", ["calibrate", "test"])
def test_a_full_pass_in_a_clip_that_is_not_train_is_not_read(tmp_path: Path, role: str) -> None:
    fake = FakePatchCore()
    paths = _setup(tmp_path, others=(("X-1", role, "full", 7),))
    assert _run(tmp_path, paths, fake) == 0
    assert [i["file"] for i in _bundle(tmp_path)["images"]] == [  # type: ignore[attr-defined]
        "T-1.mkv#50", "T-1.mkv#114"]


def test_no_full_pass_in_the_train_clips_is_refused_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path, [("empty", 82), ("gap", 146)],
                   others=(("C-1", "calibrate", "full", 7),))
    assert _run(tmp_path, paths, FakePatchCore()) == 1
    assert capsys.readouterr().out == "refused: no full pass in the train clips\n"
    assert not (tmp_path / "models").exists()


def test_a_train_clip_changed_since_the_split_is_refused_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    with (paths["footage"] / "T-1.mkv").open("ab") as clip:
        clip.write(b"x")
    assert _run(tmp_path, paths, FakePatchCore()) == 2
    assert capsys.readouterr().out.startswith("T-1.mkv changed since the split")
    assert not (tmp_path / "models").exists()


def test_the_written_bundle_loads(tmp_path: Path) -> None:
    assert _run(tmp_path, _setup(tmp_path), FakePatchCore()) == 0
    (bundle_dir,) = (tmp_path / "models").iterdir()
    bundle = load_bundle(tmp_path / "models", bundle_dir.name)
    assert (bundle.input_px, bundle.window_px, bundle.data_source) == (32, 32, "own")
    assert bundle.bank.shape == (2, 3)


class _Scorer:
    def __init__(self, score: float) -> None:
        self.value = score

    def score(self, window_input: np.ndarray) -> Scored:
        return Scored(np.zeros((1, 1), np.float32), self.value, 1.0, 1.0)


LANE = Lane((8, 4, 32, 32), 10.0)


def test_lane_b_score_is_the_scorers_pred_score() -> None:
    pixels = np.full((48, 64, 3), 128, dtype=np.uint8)
    assert train_lane_b.lane_b_score(pixels, LANE, _Scorer(1.5), 32) == 1.5


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_lane_b_score_refuses_a_non_finite_score(value: float) -> None:
    pixels = np.full((48, 64, 3), 128, dtype=np.uint8)
    with pytest.raises(InferenceError, match="pred_score"):
        train_lane_b.lane_b_score(pixels, LANE, _Scorer(value), 32)


@pytest.mark.env
def test_the_real_patchcore_fits_lane_crops_on_the_gpu_with_the_cached_backbone(
    tmp_path: Path,
) -> None:
    paths = _setup(tmp_path)
    args = build_parser().parse_args([
        "train-lane-b", "--lane", str(paths["lane"]), "--labels", str(paths["labels"]),
        "--footage", str(paths["footage"]), "--split", str(paths["split"]),
        "--product", "mochi-lane", "--input-px", "64", "--models", str(tmp_path / "models"),
    ])
    assert train_lane_b.main(args) == 0
    (bundle_dir,) = (tmp_path / "models").iterdir()
    bundle = load_bundle(tmp_path / "models", bundle_dir.name)
    assert bundle.device == "cuda"
    assert bundle.bank.shape[1] == 1536
