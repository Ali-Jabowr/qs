"""`qs latency` through `main()` with a fake scorer and synthetic frames in `tmp_path`.

Seven PNGs are written: five discs (which localize and score) and two empty floors (which
raise `PlacementFault`, so they are excluded from the sample and counted, on every pass).
The scripted scorer hands back fixed `backbone_ms`/`search_ms` values per call, so the
report's percentiles can be checked against hand-computed `numpy.percentile` values, and
the whole `inspect()` is timed around it. CUDA is faked available; `--cpu` is observed
through the command's `hide_cuda` seam. No GPU, no weights, no network.
"""

from __future__ import annotations

import io
import os
import re
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import pytest

from appearance_fixtures import config_naming
from conftest import Paint
from qs.__main__ import build_parser
from qs.cmd import latency
from qs.config import LoadedConfig
from qs.inspector.model import Scored

DAY = "2026-09-23"


class ScriptedScorer:
    """The scorer whose timings are the call-th of the scripted lists, in call order.

    Call 0 is `load`'s warm-up, call 1 the replay warm-up verdict; the timed verdicts are
    calls 1 … count.
    """

    def __init__(self, backbone: list[float], search: list[float]) -> None:
        self.backbone, self.search = backbone, search
        self.calls = 0

    def score(self, window_input: np.ndarray) -> Scored:
        index = self.calls
        self.calls += 1
        anomaly_map = np.zeros(window_input.shape[1:], dtype=np.float32)
        return Scored(anomaly_map, 0.5, backbone_ms=self.backbone[index],
                      search_ms=self.search[index])


@pytest.fixture
def frames(tmp_path: Path) -> Path:
    """Seven frames by name: discs at 001, 003, 005, 007, 009 and empty floors at 002, 008."""
    directory = tmp_path / "frames"
    directory.mkdir()
    paint = Paint()
    for index in range(1, 8):
        pixels = paint.floor()
        if index not in (2, 6):
            paint.disc(pixels, (960, 540), 190)
        cv2.imwrite(str(directory / f"{index:03d}_replay.png"), pixels)
    return directory


@pytest.fixture
def loaded(tmp_path: Path) -> LoadedConfig:
    """The example config, rewritten to name a real (small) bundle under `tmp_path`."""
    from qs.inspector.bundle import BundleInfo, ImageRecord, write_bundle

    bank = np.random.default_rng(3).random((20, 8), dtype=np.float32)
    images = tuple(
        ImageRecord(f"u{u:03d}_good_G2_r0.png", f"u{u:03d}", f"{u:02d}".ljust(64, "b"))
        for u in (1, 2, 3)
    )
    info = BundleInfo(input_px=64, window_px=400, coreset_ratio=0.1, seed=42,
                      device="cuda", data_source="own", images=images)
    model_id = write_bundle(tmp_path / "models", "mochi", bank, info)
    return config_naming(tmp_path, model_id)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loaded: LoadedConfig,
    frames: Path,
    *extra: str,
    scorer: ScriptedScorer | None = None,
    cuda: bool = True,
    hide: Callable[[], None] | None = None,
    order: list[str] | None = None,
) -> tuple[int, str, Path]:
    """One `latency` run; its exit code, its one-line output, and the report's path."""
    scorer = scorer or ScriptedScorer([1.0] * 400, [2.0] * 400)
    report_dir = tmp_path / "reports"
    device_label = "cpu" if "--cpu" in extra else "cuda"
    monkeypatch.setattr(latency, "MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(latency, "_today", lambda: DAY)
    args = build_parser().parse_args([
        "latency", "--config", str(loaded.path), "--frames", str(frames),
        "--report-dir", str(report_dir), *extra,
    ])

    def make(device: str, bank: np.ndarray) -> ScriptedScorer:
        if order is not None:
            order.append("scorer")
        return scorer

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = latency.main(args, make_scorer=make, cuda_available=lambda: cuda,
                            hide_cuda=hide if hide is not None else lambda: None)
    path = report_dir / f"{DAY}-{loaded.config.inspector.model}-{device_label}.md"
    return code, buffer.getvalue(), path


def _sample(scorer: ScriptedScorer, count: int) -> tuple[list[float], list[float]]:
    """The timings of the timed verdicts: scorer calls 1 … count (call 0 is the warm-up)."""
    return scorer.backbone[1:count + 1], scorer.search[1:count + 1]


def test_twenty_inspections_loop_and_exclude_the_empty_frames(
    tmp_path: Path, loaded: LoadedConfig, frames: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = ScriptedScorer([10.0 + 0.5 * i for i in range(30)],
                            [5.0 + 0.1 * i for i in range(30)])
    code, out, path = _run(monkeypatch, tmp_path, loaded, frames, "--count", "20",
                           scorer=scorer, hide=lambda: None)

    assert code == 0, out
    assert scorer.calls == 22  # load's warm-up + the replay warm-up + 20 timed verdicts
    report = path.read_text(encoding="utf-8")
    assert "inspections: 20" in report
    assert "placement-fault frames excluded: 8" in report  # 2 per pass, 4 passes


def test_the_percentiles_are_numpy_linear(
    tmp_path: Path, loaded: LoadedConfig, frames: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = ScriptedScorer([10.0 + 0.5 * i for i in range(30)],
                            [5.0 + 0.1 * i for i in range(30)])
    code, out, path = _run(monkeypatch, tmp_path, loaded, frames, "--count", "20",
                           scorer=scorer, hide=lambda: None)

    assert code == 0, out
    backbone, search = _sample(scorer, 20)
    report = path.read_text(encoding="utf-8")
    for name, samples in (("backbone", backbone), ("search", search)):
        row = re.search(rf"\| {name} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", report)
        assert row is not None, report
        for got, q in zip(row.groups(), (50.0, 95.0, 99.0), strict=True):
            assert got == f"{float(np.percentile(samples, q)):.1f}", (name, q)


def test_the_gate_b_line_passes_at_p95_130(tmp_path: Path, loaded: LoadedConfig,
                                           frames: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    # a scripted p95 of 130 ms: the timed backbone samples are constant 130
    scorer = ScriptedScorer([130.0] * 25, [1.0] * 25)
    code, out, path = _run(monkeypatch, tmp_path, loaded, frames, "--count", "20", "--cpu",
                           scorer=scorer, hide=lambda: None)

    assert code == 0, out
    report = path.read_text(encoding="utf-8")
    assert "Gate B proxy: p95 × 3 = 390.0 ms ≤ 400 ms → PASS (ADR-0017)" in report
    assert "N100 estimate: p95 × 2 … p95 × 3 = 260.0 … 390.0 ms" in report
    assert "not a Gate B result" in report  # 20 inspections, below the 200
    # no score, threshold or label anywhere in a CPU report
    assert "defect" not in report.lower()
    assert "threshold_id" not in report


def test_the_gate_b_line_fails_at_p95_140_and_lists_the_fallbacks(
    tmp_path: Path, loaded: LoadedConfig, frames: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = ScriptedScorer([140.0] * 25, [1.0] * 25)
    code, out, path = _run(monkeypatch, tmp_path, loaded, frames, "--count", "20", "--cpu",
                           scorer=scorer, hide=lambda: None)

    assert code == 0, out
    report = path.read_text(encoding="utf-8")
    assert "Gate B proxy: p95 × 3 = 420.0 ms ≤ 400 ms → FAIL (ADR-0017)" in report
    assert "N100 estimate: p95 × 2 … p95 × 3 = 280.0 … 420.0 ms" in report
    for step in ("coreset 1%", "input 224", "ResNet-18", "PaDiM"):
        assert step in report


def test_the_gpu_run_carries_the_bench_line_and_no_gate_b_verdict(
    tmp_path: Path, loaded: LoadedConfig, frames: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = ScriptedScorer([80.0] * 25, [1.0] * 25)
    code, out, path = _run(monkeypatch, tmp_path, loaded, frames, "--count", "20",
                           scorer=scorer, hide=lambda: None)

    assert code == 0, out
    report = path.read_text(encoding="utf-8")
    assert "PatchCore p95 80.0 ms against 100 ms (docs/10 §5) — reported, not gated" in report
    assert "Gate B proxy" not in report
    assert "-cuda.md" not in path.name or path.name.endswith("-cuda.md")


def test_the_cpu_override_is_observed_through_the_seam(
    tmp_path: Path, loaded: LoadedConfig, frames: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    order: list[str] = []
    config_path = loaded.path
    before = config_path.read_bytes()
    scorer = ScriptedScorer([1.0] * 25, [1.0] * 25)

    def hide() -> None:
        order.append("hide")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def make(device: str, bank: np.ndarray) -> ScriptedScorer:
        order.append("scorer")
        seen["device"] = device
        seen["env"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        return scorer

    args = build_parser().parse_args([
        "latency", "--config", str(config_path), "--frames", str(frames),
        "--count", "5", "--cpu", "--report-dir", str(tmp_path / "reports"),
    ])
    monkeypatch.setattr(latency, "MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(latency, "_today", lambda: DAY)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = latency.main(args, make_scorer=make, cuda_available=lambda: True, hide_cuda=hide)
    out = buffer.getvalue()
    path = tmp_path / "reports" / f"{DAY}-{loaded.config.inspector.model}-cpu.md"
    assert code == 0, out
    assert seen["device"] == "cpu"
    assert seen["env"] == ""  # CUDA_VISIBLE_DEVICES="" before the scorer was built
    assert order == ["hide", "scorer"]  # hide_cuda ran before the scorer was built
    assert config_path.read_bytes() == before  # the config file's bytes are unchanged
    report = path.read_text(encoding="utf-8")
    assert "device: torch-cpu" in report


def test_below_200_is_not_a_gate_b_result(tmp_path: Path, loaded: LoadedConfig,
                                          frames: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    code, out, path = _run(monkeypatch, tmp_path, loaded, frames, "--count", "50",
                           hide=lambda: None)
    assert code == 0, out
    assert "below the 200 ADR-0017 requires: not a Gate B result" in (
        path.read_text(encoding="utf-8"))


def test_only_empty_frames_exit_1(tmp_path: Path, loaded: LoadedConfig,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    paint = Paint()
    cv2.imwrite(str(directory / "001_replay.png"), paint.floor())
    code, out, path = _run(monkeypatch, tmp_path, loaded, directory, "--count", "5",
                           hide=lambda: None)
    assert code == 1
    assert "no frame" in out and "placement-fault" in out
    assert not path.exists()


def test_a_missing_directory_exits_2(tmp_path: Path, loaded: LoadedConfig,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    code, out, path = _run(monkeypatch, tmp_path, loaded, tmp_path / "nope",
                           hide=lambda: None)
    assert code == 2
    assert "is not a directory" in out
    assert "Traceback" not in out


def test_a_bad_config_exits_2(tmp_path: Path, loaded: LoadedConfig, frames: Path,
                              monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.yml"
    bad.write_text("version: 99\n", encoding="utf-8")
    args = build_parser().parse_args([
        "latency", "--config", str(bad), "--frames", str(frames),
        "--report-dir", str(tmp_path / "reports"),
    ])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = latency.main(args, make_scorer=lambda d, b: ScriptedScorer([1.0], [1.0]),
                            cuda_available=lambda: True, hide_cuda=lambda: None)
    assert code == 2
    assert "Traceback" not in buffer.getvalue()
    assert buffer.getvalue().strip().count("\n") == 0  # one line
