"""`qs train` through `main()` with the fake model, on a synthetic dataset in `tmp_path`.

8 units × 3 rotations of pale discs, `uNNN_good_G2_rK.png`, drawn once per module and
linked into each test's own `train/good`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from conftest import EXAMPLE, FakePatchCore, Paint
from qs.__main__ import build_parser
from qs.cmd import train
from qs.config import _MODEL_ID
from qs.inspector.model import PatchCore, load_backbone

UNITS = (12, 3, 25, 7, 18, 1, 30, 9)
PINNED = "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000"
BUNDLE_KEYS = {
    "backbone", "backbone_sha256", "layers", "input_px", "window_px", "normalisation",
    "coreset_ratio", "num_neighbors", "seed", "device", "versions", "data_source",
    "validation_only", "units", "images",
}


def _disc_frame(unit: int, rotation: int) -> np.ndarray:
    paint = Paint()
    frame = paint.floor()
    paint.disc(frame, (960 + 7 * rotation, 540 - 5 * rotation), 180 + unit % 5)
    return frame


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The 24 good frames, written once."""
    root = tmp_path_factory.mktemp("frames")
    for unit in UNITS:
        for rotation in range(3):
            path = root / f"u{unit:03d}_good_G2_r{rotation}.png"
            cv2.imwrite(str(path), _disc_frame(unit, rotation))
    return root


def _dataset(tmp_path: Path, frames: Path, extra: dict[str, np.ndarray] | None = None) -> Path:
    good = tmp_path / "dataset" / "train" / "good"
    good.mkdir(parents=True)
    for path in frames.iterdir():
        (good / path.name).symlink_to(path)
    for name, pixels in (extra or {}).items():
        cv2.imwrite(str(good / name), pixels)
    (tmp_path / "dataset" / "calib" / "good").mkdir(parents=True)  # never read
    cv2.imwrite(str(tmp_path / "dataset" / "calib" / "good" / "u099_good_G2_r0.png"),
                Paint().floor())
    return tmp_path / "dataset"


def _run(
    tmp_path: Path,
    dataset: Path,
    *extra: str,
    model: Callable[[str], PatchCore] | None = None,
    cuda: bool = True,
    config: Path = EXAMPLE,
) -> int:
    args = build_parser().parse_args([
        "train", "--config", str(config), "--dataset", str(dataset), "--window-px", "400",
        "--product", "mochi", "--models", str(tmp_path / "models"), *extra,
    ])
    fake = FakePatchCore()
    return train.main(
        args,
        make_patchcore=model or (lambda device: fake),
        cuda_available=lambda: cuda,
    )


def _one_line(capsys: pytest.CaptureFixture[str]) -> str:
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1, lines
    return lines[0]


def _no_bundle(tmp_path: Path) -> None:
    models = tmp_path / "models"
    assert not models.exists() or list(models.iterdir()) == []


def test_help_lists_train() -> None:
    assert "train" in build_parser().format_help()


def test_train_writes_a_bundle_and_prints_its_model_id(
    tmp_path: Path, frames: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakePatchCore()
    code = _run(tmp_path, _dataset(tmp_path, frames), "--input-px", "224", "--seed", "7",
                model=lambda device: fake)
    out = capsys.readouterr().out
    assert code == 0, out
    [model_id] = re.findall(r"^model_id: (\S+)$", out, re.MULTILINE)
    assert _MODEL_ID.fullmatch(model_id)
    assert f"\ninspector.model: {model_id}\n" in f"\n{out}"
    directory = tmp_path / "models" / model_id
    assert sorted(p.name for p in directory.iterdir()) == ["bundle.yml", "memory_bank.pt"]

    [(windows, settings)] = fake.fits
    assert len(windows) == 24
    assert all(w.shape == (3, 224, 224) and w.dtype == np.float32 for w in windows)
    assert (settings.coreset_ratio, settings.seed, settings.device) == (0.1, 7, "cuda")


def test_the_bundle_records_what_it_was_trained_on(
    tmp_path: Path, frames: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _dataset(tmp_path, frames)
    assert _run(tmp_path, dataset) == 0
    [directory] = (tmp_path / "models").iterdir()
    text = (directory / "bundle.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert set(data) == BUNDLE_KEYS
    assert data["units"] == sorted(f"u{unit:03d}" for unit in UNITS)
    good = dataset / "train" / "good"
    assert len(data["images"]) == 24
    for image in data["images"]:
        assert image["sha256"] == hashlib.sha256((good / image["file"]).read_bytes()).hexdigest()
        assert image["file"].startswith(image["unit"] + "_good_G2_r")
    assert data["data_source"] == "own"
    assert data["validation_only"] is False
    assert (data["window_px"], data["input_px"], data["seed"]) == (400, 256, 42)
    assert (data["backbone"], data["backbone_sha256"]) == ("wide_resnet50_2.racm_in1k", PINNED)
    assert not re.search(r"\d{2}:\d{2}|\d{4}-\d{2}-\d{2}", text)
    joined = (directory / "memory_bank.pt").read_bytes() + (directory / "bundle.yml").read_bytes()
    assert directory.name.endswith("-" + hashlib.sha256(joined).hexdigest()[:12])


def test_a_second_run_is_v2_and_leaves_the_first_untouched(
    tmp_path: Path, frames: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _dataset(tmp_path, frames)
    assert _run(tmp_path, dataset) == 0
    [first] = (tmp_path / "models").iterdir()
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    assert _run(tmp_path, dataset, "--seed", "43") == 0
    names = sorted(p.name for p in (tmp_path / "models").iterdir())
    assert [re.sub(r"-[0-9a-f]{12}$", "", n) for n in names] == [
        "patchcore-mochi-v1", "patchcore-mochi-v2"
    ]
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("p01_good_G2_r0.png", "not a dataset unit"),
        ("c01_good_G2_r0_pass1.png", "not a dataset unit"),
        ("x01_empty_G2_r0.png", "not a dataset unit"),
        ("u040_dmg_G2_r0.png", "state dmg"),
        ("u040_good_G3_r0.png", "geometry G3"),
        ("u040_good_G2.png", "parts"),
        ("frame.png", "parts"),
    ],
)
def test_a_frame_that_does_not_belong_in_train_good_is_refused(
    tmp_path: Path, frames: Path, capsys: pytest.CaptureFixture[str], name: str, reason: str
) -> None:
    dataset = _dataset(tmp_path, frames, {name: Paint().floor()})
    assert _run(tmp_path, dataset) == 1
    line = _one_line(capsys)
    assert line.startswith(f"refused: {name}: ")
    assert reason in line
    _no_bundle(tmp_path)


def test_a_frame_with_no_unit_is_refused(
    tmp_path: Path, frames: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _dataset(tmp_path, frames, {"u050_good_G2_r0.png": Paint().floor()})
    assert _run(tmp_path, dataset) == 1
    assert _one_line(capsys) == "refused: u050_good_G2_r0.png: placement: no unit"
    _no_bundle(tmp_path)


def test_an_empty_train_good_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "dataset" / "train" / "good").mkdir(parents=True)
    assert _run(tmp_path, tmp_path / "dataset") == 1
    assert "no .png frames" in _one_line(capsys)
    _no_bundle(tmp_path)


def test_an_invalid_config_cannot_run(
    tmp_path: Path, frames: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yml"
    bad.write_text(EXAMPLE.read_text(encoding="utf-8").replace("backend:   torch-cuda",
                                                                "backend:   tpu"))
    assert _run(tmp_path, _dataset(tmp_path, frames), config=bad) == 2
    assert _one_line(capsys).startswith("config: ")
    _no_bundle(tmp_path)


def test_torch_cuda_without_cuda_never_falls_back(
    tmp_path: Path, frames: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, _dataset(tmp_path, frames), cuda=False) == 2
    assert "no CPU fallback" in _one_line(capsys)
    _no_bundle(tmp_path)


@pytest.mark.parametrize("content", [None, b"not the pinned weights"])
def test_weights_missing_or_another_hash_cannot_run(
    tmp_path: Path, frames: Path, capsys: pytest.CaptureFixture[str], content: bytes | None
) -> None:
    weights = tmp_path / "model.safetensors"
    if content is not None:
        weights.write_bytes(content)
    built: list[str] = []

    def make(device: str) -> PatchCore:
        return load_backbone(lambda: weights, lambda: built.append(device) or FakePatchCore())

    assert _run(tmp_path, _dataset(tmp_path, frames), model=make) == 2
    line = _one_line(capsys)
    assert line.startswith("weights: ")
    assert built == []
    _no_bundle(tmp_path)


@pytest.mark.parametrize(
    "flags",
    [["--product", "Mochi"], ["--window-px", "0"], ["--coreset", "0"], ["--coreset", "1.5"],
     ["--input-px", "-1"], ["--seed", "-1"]],
)
def test_bad_flags_are_usage_errors(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(
            ["train", "--config", "c", "--dataset", "d", "--window-px", "400",
             "--product", "mochi", *flags]
        )
    assert caught.value.code == 2
