"""`qs mvtec-check` through `main()` with fakes, on a synthetic MVTec-shaped tree.

`bottle/train/good` holds 10 grey 300 × 300 squares, `test/good` 4 of them and `test/broken`
4 with a dark blot. The blot fake scores the blots high, so the good and broken scores are
fully separated; the inverted fake scores by content mean, reversed, so both AUROCs are 0.
Nothing needs a GPU, the weights cache, a camera or a display, and nothing is downloaded.
"""

from __future__ import annotations

import io
import itertools
import os
import re
import shutil
import socket
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from conftest import BlotPatchCore
from qs.__main__ import build_parser
from qs.cmd import mvtec_check
from qs.errors import ConfigError
from qs.inspector.bundle import load_bundle
from qs.inspector.model import FitSettings, Fitted, PatchCore, Scored, Scorer
from qs.inspector.window import to_input

SIZE = 300  # the synthetic images; the whole-image window is therefore 300 px
GOOD_GREY = 200
BLOT_GREY = 30
BLOT_RADIUS = 24
DAY = "2026-09-23"


def _square(blot: bool) -> np.ndarray:
    """One grey square; with a blot, a dark disc at its centre."""
    pixels = np.full((SIZE, SIZE, 3), GOOD_GREY, dtype=np.uint8)
    if blot:
        cv2.circle(pixels, (SIZE // 2, SIZE // 2), BLOT_RADIUS, (BLOT_GREY,) * 3, -1)
    return pixels


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """The synthetic MVTec AD tree: bottle, 10 train squares, 4 + 4 test images."""
    root = tmp_path / "mvtec"
    good = root / "bottle" / "train" / "good"
    good.mkdir(parents=True)
    for index in range(10):
        cv2.imwrite(str(good / f"{index:03d}.png"), _square(False))
    for name, blot in (("good", False), ("broken", True)):
        folder = root / "bottle" / "test" / name
        folder.mkdir(parents=True)
        for index in range(4):
            cv2.imwrite(str(folder / f"{index:03d}.png"), _square(blot))
    return root


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Any socket connection attempt raises — the command opens none."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise OSError("network disabled by test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield


class _RecordingScorer:
    """The blot scorer with every input it received recorded."""

    def __init__(self, inner: Scorer, seen: list[np.ndarray]) -> None:
        self._inner = inner
        self.seen = seen

    def score(self, window_input: np.ndarray) -> Scored:
        self.seen.append(window_input)
        return self._inner.score(window_input)


class RecordingBlot(BlotPatchCore):
    """The blot fake, recording every input the scorer received."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[np.ndarray] = []

    def scorer(self, bank: np.ndarray) -> Scorer:
        return _RecordingScorer(super().scorer(bank), self.seen)


class InvertedPatchCore:
    """A fake that scores by content mean: the blots score lowest.

    The map is one constant per image — the mean — so the station's masked maximum and
    `pred_score` agree, and every good image outranks every broken one: AUROC 0.
    """

    def __init__(self) -> None:
        self.fits: list[tuple[list[np.ndarray], FitSettings]] = []

    def fit(self, windows: list[np.ndarray], settings: FitSettings) -> Fitted:
        self.fits.append((list(windows), settings))
        return Fitted(bank=np.zeros((1, 1), dtype=np.float32), peak_gpu_bytes=100 << 20)

    def scorer(self, bank: np.ndarray) -> Scorer:
        return self

    def score(self, window_input: np.ndarray) -> Scored:
        value = float(window_input.mean())
        anomaly_map = np.full(window_input.shape[1:], value, dtype=np.float32)
        return Scored(anomaly_map, value, backbone_ms=1.0, search_ms=2.0)


def _run(
    tmp_path: Path,
    root: Path,
    *extra: str,
    fake: PatchCore | None = None,
    model: Callable[[str], PatchCore] | None = None,
    cuda: bool = True,
    hide: Callable[[], None] | None = None,
) -> tuple[int, str]:
    """One `mvtec-check` run; code and one-line output."""
    args = build_parser().parse_args([
        "mvtec-check", "--root", str(root), "--category", "bottle",
        "--models", str(tmp_path / "models"), "--report-dir", str(tmp_path / "reports"),
        *extra,
    ])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = mvtec_check.main(
            args,
            make_patchcore=model or (lambda device: fake if fake is not None else RecordingBlot()),
            cuda_available=lambda: cuda,
            hide_cuda=hide if hide is not None else (lambda: None),
        )
    return code, buffer.getvalue()


# --- the pure arithmetic --------------------------------------------------------------------


def test_help_lists_mvtec_check() -> None:
    assert "mvtec-check" in build_parser().format_help()


def test_the_auroc_on_hand_examples() -> None:
    assert mvtec_check.auroc([0.1, 0.2], [0.5, 0.6]) == 1.0
    assert mvtec_check.auroc([0.5, 0.6], [0.1, 0.2]) == 0.0
    assert mvtec_check.auroc([0.3, 0.3], [0.3, 0.3]) == 0.5
    assert mvtec_check.auroc([0.1, 0.2, 0.3], [0.3, 0.8, 0.9]) == 8.5 / 9


def test_the_auroc_needs_both_groups() -> None:
    with pytest.raises(ValueError, match="at least one normal and one anomalous"):
        mvtec_check.auroc([], [0.5])
    with pytest.raises(ValueError, match="at least one normal and one anomalous"):
        mvtec_check.auroc([0.5], [])


# --- the passing run ------------------------------------------------------------------------


def test_the_passing_run(
    tmp_path: Path,
    tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_network: None,
) -> None:
    monkeypatch.setattr(mvtec_check, "_today", lambda: DAY)
    # The clock steps 5 ms per read, so the whole model call row is 5.0 — not the two
    # phases' 1.0 + 2.0 added up.
    ticks = itertools.count(0, 0.005)
    monkeypatch.setattr(mvtec_check, "_clock", lambda: next(ticks))
    fake = RecordingBlot()
    code, out = _run(tmp_path, tree, fake=fake)
    assert code == 0, out

    [(windows, settings)] = fake.fits
    assert len(windows) == 10
    assert all(w.shape == (3, 256, 256) and w.dtype == np.float32 for w in windows)
    assert (settings.coreset_ratio, settings.seed, settings.device) == (0.1, 42, "cuda")
    paths = sorted((tree / "bottle" / "test").glob("*/*.png"))
    assert len(fake.seen) == len(paths) == 8
    for seen, path in zip(fake.seen, paths, strict=True):
        assert seen.shape == (3, 256, 256)
        pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(seen, to_input(pixels, 256))

    [model_id] = re.findall(r"^model_id: (\S+)$", out, re.MULTILINE)
    assert model_id.startswith("patchcore-mvtec-bottle-v")
    models = tmp_path / "models"
    record = yaml.safe_load((models / model_id / "bundle.yml").read_text(encoding="utf-8"))
    assert record["data_source"] == "external:mvtec-ad"
    assert record["validation_only"] is True
    assert record["units"] == ["good"]
    assert len(record["images"]) == 10
    reloaded = load_bundle(models, model_id, validation=True)
    assert reloaded.validation_only and reloaded.data_source == "external:mvtec-ad"
    with pytest.raises(ConfigError, match="ADR-0016"):
        load_bundle(models, model_id)

    assert sorted(p.name for p in (tmp_path / "reports").iterdir()) == [
        f"{DAY}-bottle-cuda.md"
    ]
    report = (tmp_path / "reports" / f"{DAY}-bottle-cuda.md").read_text(encoding="utf-8")
    assert "train/good 10 (fit)" in report
    assert "test 8 in 2 classes — broken 4 (anomalous), good 4 (normal)" in report
    assert "pred_score (Anomalib's image score) AUROC: 1.000" in report
    assert "masked maximum (the station's score) AUROC: 1.000" in report
    assert "published image AUROC: 1.000" in report
    assert "PatchCore README shipped in the pinned package" in report
    assert "pass line (both ≥ 0.98): PASS" in report
    assert "| backbone | 1.0 | 1.0 | 1.0 |" in report
    assert "| search | 2.0 | 2.0 | 2.0 |" in report
    assert "| whole model call | 5.0 | 5.0 | 5.0 |" in report
    assert "| whole model call | 3.0" not in report
    assert "peak GPU memory" not in report
    assert f"- model_id: {model_id}" in report
    assert "validation-only: refused by the station outside validation mode" in report
    assert "ADR-0016" in report
    assert out.rstrip().endswith(f"report: {tmp_path / 'reports' / f'{DAY}-bottle-cuda.md'}")


# --- below the pass line --------------------------------------------------------------------


def test_an_inverted_scorer_is_below_the_line(
    tmp_path: Path, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mvtec_check, "_today", lambda: DAY)
    code, out = _run(tmp_path, tree, fake=InvertedPatchCore())
    assert code == 1, out
    report = (tmp_path / "reports" / f"{DAY}-bottle-cuda.md").read_text(encoding="utf-8")
    assert "pred_score (Anomalib's image score) AUROC: 0.000" in report
    assert "masked maximum (the station's score) AUROC: 0.000" in report
    assert (
        "pass line (both ≥ 0.98): FAIL — pred_score 0.000 and masked maximum 0.000 "
        "below the line"
    ) in report
    assert "peak GPU memory (fit): 100 MiB" in report
    assert "below the line" in out
    assert out.rstrip().endswith(f"report: {tmp_path / 'reports' / f'{DAY}-bottle-cuda.md'}")


# --- refusals -------------------------------------------------------------------------------


def test_a_non_square_image_is_refused(tmp_path: Path, tree: Path) -> None:
    cv2.imwrite(
        str(tree / "bottle" / "test" / "good" / "000.png"),
        np.full((SIZE, 200, 3), GOOD_GREY, dtype=np.uint8),
    )
    code, out = _run(tmp_path, tree)
    lines = out.strip().splitlines()
    assert code == 1, out
    assert len(lines) == 1
    assert lines[0] == "refused: bottle/test/good/000.png: 200 × 300 is not square"
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "reports").exists()


def test_a_missing_category_cannot_run(tmp_path: Path, no_network: None) -> None:
    root = tmp_path / "mvtec"
    root.mkdir()
    code, out = _run(tmp_path, root)
    lines = out.strip().splitlines()
    assert code == 2
    assert len(lines) == 1
    assert lines[0] == (
        f"mvtec: {root / 'bottle'} does not exist — download the bottle category "
        "there first (S1.1)"
    )


def test_a_category_without_train_images_cannot_run(tmp_path: Path, tree: Path) -> None:
    for path in (tree / "bottle" / "train" / "good").glob("*.png"):
        path.unlink()
    code, out = _run(tmp_path, tree)
    assert code == 2, out
    assert "no .png training images" in out


def test_a_category_without_an_anomalous_class_cannot_run(
    tmp_path: Path, tree: Path
) -> None:
    shutil.rmtree(tree / "bottle" / "test" / "broken")
    code, out = _run(tmp_path, tree)
    assert code == 2, out
    assert "no anomalous class" in out


def test_an_unreadable_png_cannot_run(tmp_path: Path, tree: Path) -> None:
    (tree / "bottle" / "test" / "good" / "000.png").write_bytes(b"not a png")
    code, out = _run(tmp_path, tree)
    assert code == 2, out
    assert "not a readable PNG" in out


def test_a_category_without_a_published_number_cannot_run(
    tmp_path: Path, tree: Path
) -> None:
    (tree / "screw" / "train" / "good").mkdir(parents=True)
    code, out = _run(tmp_path, tree, "--category", "screw")
    assert code == 2, out
    assert "no published AUROC" in out and "bottle" in out


def test_torch_cuda_without_cuda_cannot_run(tmp_path: Path, tree: Path) -> None:
    code, out = _run(tmp_path, tree, cuda=False)
    lines = out.strip().splitlines()
    assert code == 2
    assert len(lines) == 1 and "no CPU fallback" in lines[0]


def test_a_bad_device_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args([
            "mvtec-check", "--root", "r", "--category", "bottle", "--device", "tpu",
        ])
    assert caught.value.code == 2


# --- the cpu path ---------------------------------------------------------------------------


class _StubTensor:
    """Enough of a torch tensor for `load_bundle` to hand back a bank."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def numpy(self) -> np.ndarray:
        return self._array


def test_cpu_without_bundle_cannot_run(tmp_path: Path, tree: Path) -> None:
    hidden: list[bool] = []
    code, out = _run(tmp_path, tree, "--device", "cpu", hide=lambda: hidden.append(True))
    lines = out.strip().splitlines()
    assert code == 2, out
    assert len(lines) == 1
    assert lines[0] == "cpu: --device cpu scores an existing bundle only — give --bundle MODEL_ID"
    assert hidden == []
    assert not (tmp_path / "models").exists()


def test_cpu_hides_cuda_before_the_model_module_and_pins_four_threads(
    tmp_path: Path, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mvtec_check, "_today", lambda: DAY)
    assert _run(tmp_path, tree)[0] == 0
    [bundle_dir] = sorted((tmp_path / "models").iterdir())
    model_id = bundle_dir.name

    # The bank the bundle file holds: the 200-grey triple, as the blot fake's fit stores it.
    window = to_input(_square(False), 256)
    bank = np.stack([window.max(axis=(1, 2))] * 2).astype(np.float32)

    threads: list[int] = []
    order: list[str] = []
    stub = types.ModuleType("torch")
    stub.Tensor = _StubTensor  # type: ignore[attr-defined]
    stub.set_num_threads = lambda count: threads.append(count)  # type: ignore[attr-defined]
    stub.load = lambda *args, **kwargs: _StubTensor(bank)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", stub)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    def hide() -> None:
        order.append("hide")
        mvtec_check._hide_cuda()

    def make(device: str) -> PatchCore:
        order.append("model")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        return RecordingBlot()

    code, out = _run(
        tmp_path, tree, "--device", "cpu", "--bundle", model_id, model=make, hide=hide
    )
    assert code == 0, out
    assert threads == [4]
    assert order == ["hide", "model"]
    report = (tmp_path / "reports" / f"{DAY}-bottle-cpu.md").read_text(encoding="utf-8")
    assert "input_px 256" in report
    assert "peak GPU memory" not in report
    assert f"- model_id: {model_id}" in report
    assert "validation-only: refused by the station outside validation mode" in report

    code, out = _run(tmp_path, tree, "--device", "cuda", "--bundle", model_id)
    assert code == 0, out
    cuda_report = (tmp_path / "reports" / f"{DAY}-bottle-cuda.md").read_text(
        encoding="utf-8"
    )
    assert "- peak GPU memory: not measured — reused bundle, no fit" in cuda_report
