"""`qs latency`: the Gate B proxy — PatchCore's latency on this machine (ADR-0017, docs/15 §10).

    bin/qs latency --config PATH --frames DIR [--cpu] [--count 200]
                   [--report-dir reports/latency]

The directory's PNGs are replayed in name order, looping, through `AppearanceInspector` —
with the camera excluded — until `--count` inspections have produced a verdict. The first
verdict is a warm-up and is excluded from the sample. Frames that end in `PlacementFault`
never reach the model, and timing them would flatter p95, so they are left out of the
sample and counted. The backbone and the search are timed by the scorer (its `Scored`
carries both, in milliseconds); the whole `inspect()` is timed around it.

`--cpu` sets `CUDA_VISIBLE_DEVICES=""` before torch is imported, pins torch to 4 threads
and overrides the backend to `torch-cpu` in memory — `dataclasses.replace`; the config
file is never written. Scores and labels are never reported on the CPU, because the
threshold was calibrated on CUDA (docs/12 §4).

The report, `reports/latency/<date>-<model_id>-<cpu|cuda>.md`, holds the count, the frames
directory, the excluded placement frames and the device; p50, p95 and p99 (numpy's default
linear percentile) of the backbone, the search and the whole inspection; on the CPU the
Gate B line — p95 × 3 against 400 ms, PASS or FAIL — and the N100 estimate as the range
p95 × 2 … p95 × 3, with the docs/15 §10 fallback order on FAIL; on the GPU the bench
budget line (PatchCore ≤ 100 ms, docs/10 §5), reported and not gated. A run under 200
inspections is marked not a Gate B result. Nothing is changed automatically.

Exit codes: 0 report written · 1 no frame produced a verdict · 2 could not run (config,
bundle, CUDA for a GPU run). Every refusal is one line, never a traceback.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import cv2
import numpy as np

from qs.config import LoadedConfig, load_config
from qs.contracts import CameraReadback, Frame, TriggerEvent
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.inspector.appearance import AppearanceInspector
from qs.inspector.model import Scored, Scorer
from qs.sources.replay import replay_files

MODELS_ROOT = "models"  # the bundles, relative to the working directory, as `qs train` writes
THREADS = 4  # the CPU run pins torch to 4 threads (docs/15 §10)
GATE_B_MS = 400  # the N100 budget the ×3 proxy is judged against (ADR-0017)
BENCH_MS = 100  # the bench budget for PatchCore's p95 (docs/10 §5)
MIN_INSPECTIONS = 200  # the fewest inspections a Gate B result may rest on (ADR-0017)
FALLBACKS = (
    "coreset 1% — qs train --coreset 0.01",
    "input 224 — qs train --input-px 224",
    "ResNet-18",
    "PaDiM",
)  # the docs/15 §10 fallback order, tried from the top; nothing is changed automatically


def count(text: str) -> int:
    """argparse type: an integer above zero — the verdicts to time."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not above zero")
    return value


def _real_scorer(device: str, bank: np.ndarray) -> Scorer:
    from qs.inspector.appearance import _real_scorer

    return _real_scorer(device, bank)


def _cuda_available() -> bool:
    from qs.inspector.patchcore import cuda_available

    return cuda_available()


def _hide_cuda() -> None:
    """`CUDA_VISIBLE_DEVICES=""` before any torch import, then torch on 4 threads.

    The model module imports torch when it is first imported; set first, the environment
    keeps that torch off the GPU. The tests observe this through the command's factory seam.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(THREADS)


def _today() -> str:
    """The report's date, YYYY-MM-DD; the tests pin it."""
    return date.today().isoformat()


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


class _TimedScorer:
    """A scorer that copies each call's backbone and search milliseconds into the samples."""

    def __init__(self, inner: Scorer, backbone: list[float], search: list[float]) -> None:
        self._inner = inner
        self._backbone, self._search = backbone, search

    def score(self, window_input: np.ndarray) -> Scored:
        scored = self._inner.score(window_input)
        self._backbone.append(scored.backbone_ms)
        self._search.append(scored.search_ms)
        return scored


def _percentile(samples: list[float], q: float) -> float:
    """One percentile of the samples, numpy's default (linear) method."""
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def _row(name: str, samples: list[float]) -> str:
    """One latency table row: the measure and its p50, p95 and p99."""
    return (f"| {name} | {_percentile(samples, 50):.1f} | {_percentile(samples, 95):.1f} | "
            f"{_percentile(samples, 99):.1f} |")


def _frame(path: Path, pixels: np.ndarray, seq: int) -> Frame:
    """One replayed frame, as the Pipeline would hand it to the Inspector."""
    event = TriggerEvent(seq=seq, t0=float(seq), wall=datetime.now(UTC),
                         source_id=f"replay:{path.name}")
    camera = CameraReadback(exposure_us=None, gain=None, wb_kelvin=None, focus=None,
                            verified=False)
    return Frame(event, pixels, float(seq), camera, None, 0.0)


def _write_report(report_dir: Path, result: dict[str, object]) -> Path:
    """The report: numbers only, no images — `reports/` is in git."""
    report_dir.mkdir(parents=True, exist_ok=True)
    model_id: str = result["model_id"]  # type: ignore[assignment]
    device_label: str = result["device_label"]  # type: ignore[assignment]
    count: int = result["count"]  # type: ignore[assignment]
    backbone: list[float] = result["backbone_ms"]  # type: ignore[assignment]
    lines = [
        f"# Latency — {model_id} ({device_label})",
        "",
        f"- frames: {result['frames_dir']} · inspections: {count} "
        f"(the first verdict is a warm-up, excluded from the sample) · "
        f"placement-fault frames excluded: {result['excluded']}",
        f"- device: {result['backend']}"
        + (" — torch pinned to 4 threads" if device_label == "cpu" else ""),
        "",
        "## Latency",
        "",
        "| measure | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|",
        _row("backbone", backbone),
        _row("search", result["search_ms"]),  # type: ignore[arg-type]
        _row("inspection", result["inspect_ms"]),  # type: ignore[arg-type]
        "",
    ]
    p95 = _percentile(backbone, 95)
    if device_label == "cpu":
        triple = 3.0 * p95
        passed = triple <= GATE_B_MS
        lines += [
            "## Gate B proxy",
            "",
            f"- Gate B proxy: p95 × 3 = {triple:.1f} ms ≤ {GATE_B_MS} ms → "
            f"{'PASS' if passed else 'FAIL'} (ADR-0017)",
            f"- N100 estimate: p95 × 2 … p95 × 3 = {2.0 * p95:.1f} … {triple:.1f} ms",
        ]
        if not passed:
            lines += [
                "- the docs/15 §10 fallback order, tried from the top — nothing is changed "
                "automatically:",
                *(f"  {index + 1}. {step}" for index, step in enumerate(FALLBACKS)),
            ]
        lines += [""]
    else:
        lines += [
            "## Bench budget",
            "",
            f"- PatchCore p95 {p95:.1f} ms against {BENCH_MS} ms (docs/10 §5) — "
            "reported, not gated",
            "",
        ]
    if count < MIN_INSPECTIONS:
        lines += [
            f"- {count} inspections is below the {MIN_INSPECTIONS} ADR-0017 requires: "
            "not a Gate B result",
            "",
        ]
    path = report_dir / f"{_today()}-{model_id}-{device_label}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _config(args: argparse.Namespace, hide_cuda: Callable[[], None]) -> LoadedConfig:
    """The loaded config, with the backend overridden in memory on `--cpu`."""
    loaded = load_config(Path(args.config))
    if args.cpu:
        hide_cuda()
        loaded = dataclasses.replace(
            loaded,
            config=dataclasses.replace(
                loaded.config,
                inspector=dataclasses.replace(loaded.config.inspector, backend="torch-cpu"),
            ),
        )
    return loaded


def main(
    args: argparse.Namespace,
    *,
    make_scorer: Callable[[str, np.ndarray], Scorer] = _real_scorer,
    cuda_available: Callable[[], bool] = _cuda_available,
    hide_cuda: Callable[[], None] = _hide_cuda,
) -> int:
    """Replay the frames and time them; 0 report written, 1 no verdict, 2 cannot run."""
    try:
        return _run(args, make_scorer, cuda_available, hide_cuda)
    except KeyboardInterrupt:
        print("stopped by the operator — no report written")
        return 2


def _run(
    args: argparse.Namespace,
    make_scorer: Callable[[str, np.ndarray], Scorer],
    cuda_available: Callable[[], bool],
    hide_cuda: Callable[[], None],
) -> int:
    try:
        loaded = _config(args, hide_cuda)
    except ConfigError as exc:
        print(f"config: {_first_line(exc)}")
        return 2

    frames_dir = Path(args.frames)
    try:
        paths = replay_files(frames_dir)
    except Exception as exc:  # noqa: BLE001 — ReplayError and OSError alike, one line
        print(f"frames: {_first_line(exc)}")
        return 2

    pixels: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"frames: {path.name}: not a readable PNG")
            return 2
        pixels.append(image)

    backbone: list[float] = []
    search: list[float] = []

    def timed_scorer(bank_device: str, bank: np.ndarray) -> Scorer:
        return _TimedScorer(make_scorer(bank_device, bank), backbone, search)

    inspector = AppearanceInspector(
        loaded.config_hash, MODELS_ROOT,
        make_scorer=timed_scorer, cuda_available=cuda_available,
    )
    try:
        inspector.load(loaded.config)
    except ConfigError as exc:
        print(f"bundle: {_first_line(exc)}")
        return 2

    def stream() -> Iterator[Frame]:
        seq = 0
        while True:  # looping: the directory is replayed from the top, in name order
            for path, image in zip(paths, pixels, strict=True):
                seq += 1
                yield _frame(path, image, seq)

    excluded = 0
    inspect_ms: list[float] = []
    timed = 0  # verdicts kept in the sample so far
    warm_up = False
    pass_verdicts = 0  # verdicts in the pass being replayed, the warm-up included
    try:
        for index, frame in enumerate(stream()):
            if index > 0 and index % len(paths) == 0:
                if pass_verdicts == 0:
                    break  # a full pass with no verdict at all: none is coming
                pass_verdicts = 0
            try:
                verdict = inspector.inspect(frame)
            except PlacementFault:
                excluded += 1
                continue
            pass_verdicts += 1
            if not warm_up:
                warm_up = True  # the first verdict warms the path up; it is not timed
                continue
            inspect_ms.append(verdict.inspect_ms)
            timed += 1
            if timed >= args.count:
                break
    except InferenceError as exc:
        print(f"inference: {_first_line(exc)}")
        return 2
    if timed == 0:
        print(f"frames: no frame in {frames_dir} produced a verdict — "
              f"{excluded} placement-fault frames excluded")
        return 1

    # scorer call 0 is `load`'s warm-up; verdict i of the run scored with call i
    result: dict[str, object] = {
        "model_id": inspector.model_id,
        "device_label": "cpu" if args.cpu else "cuda",
        "backend": loaded.config.inspector.backend,
        "frames_dir": str(frames_dir),
        "count": timed,
        "excluded": excluded,
        "backbone_ms": backbone[1:timed + 1],
        "search_ms": search[1:timed + 1],
        "inspect_ms": inspect_ms,
    }
    try:
        path = _write_report(Path(args.report_dir), result)
    except OSError as exc:
        print(f"report: {_first_line(exc)}")
        return 2
    print(f"latency: {timed} inspections on {result['device_label']} — report: {path}")
    return 0
