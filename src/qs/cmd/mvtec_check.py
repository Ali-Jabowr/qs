"""`qs mvtec-check`: the MVTec AD check — our code path on one category (docs/15 §5).

    bin/qs mvtec-check --root DIR --category bottle [--device cuda|cpu]
                       [--bundle MODEL_ID] [--models models] [--report-dir reports/mvtec]

It reads `<root>/<category>/train/good/*.png` and `<root>/<category>/test/<class>/*.png` and
never downloads anything: class `good` is normal, every other class is anomalous. The whole
image is the window, and it must be square. From `window.to_input`'s single `INTER_AREA`
resize to `input_px` onward this is the station's path, with a mask of all ones through
`window.score_map`. On `--device cuda` it fits the model on `train/good` and writes a bundle
with `data_source: external:mvtec-ad` and product `mvtec-<category>` — validation-only — or
reuses `--bundle`, loaded with `validation=True`. On `--device cpu` it sets
`CUDA_VISIBLE_DEVICES=""` before torch is imported, pins torch to 4 threads and scores an
existing bundle only.

Every test image is scored. The report, `reports/mvtec/<date>-<category>-<device>.md`, holds
the image and class counts; the image AUROC of `pred_score` (Anomalib's image score) and of
the masked maximum (the station's score) — rank-based, ties counted as half, our own few
lines; the published WRN-50 image AUROC for the category, with its source (the PatchCore
README shipped in the pinned package); the pass line, both AUROCs ≥ 0.98; p50/p95/p99 of the
backbone, the search and the whole model call per test image on the device; the peak GPU
memory on CUDA; and the bundle's `model_id` with the confirmation that `load_bundle` refuses
it outside validation mode.

Exit codes: 0 both AUROCs on the pass line · 1 below the pass line, or a non-square image ·
2 could not run (missing data — the category is downloaded once, at setup, S1.1; CUDA,
weights, bundle). Every refusal is one line, never a traceback.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from qs.errors import ConfigError, InferenceError
from qs.inspector.bundle import (
    PRODUCT,
    Bundle,
    BundleInfo,
    ImageRecord,
    load_bundle,
    write_bundle,
)
from qs.inspector.model import FitSettings, PatchCore, device_for
from qs.inspector.window import score_map, to_input

INPUT_PX = 256  # docs/15 §5 — the input the published numbers were produced at
CORESET_RATIO = 0.1
SEED = 42
PASS_LINE = 0.98  # docs/15 §5 — "near" the published number means both AUROCs at least this
DATA_SOURCE = "external:mvtec-ad"
DATA_UNIT = "good"  # the unit ID recorded for every train/good image in the bundle
THREADS = 4  # docs/15 §10 — the CPU run pins torch to 4 threads

# Anomalib 2.6.2's PatchCore README, shipped in the pinned package: the Image-Level AUC
# table, row "Wide ResNet-50". Only the categories pinned here can be checked.
PUBLISHED_AUROC: dict[str, float] = {"bottle": 1.000}
PUBLISHED_SOURCE = (
    "Anomalib 2.6.2's PatchCore README shipped in the pinned package "
    "(Wide ResNet-50, Image-Level AUC)"
)


class _Refused(Exception):
    """The run is refused: exit 1, one line."""


class _CannotRun(Exception):
    """The run cannot start: exit 2, one line."""


_clock = time.perf_counter  # the whole model call's clock, read around each scorer.score()


def category(text: str) -> str:
    """argparse type: an MVTec category name, e.g. `bottle`."""
    if PRODUCT.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(f"{text!r}: lower-case words joined by '-', e.g. bottle")
    return text


def _real_patchcore(device: str) -> PatchCore:
    from qs.inspector.patchcore import AnomalibPatchCore

    return AnomalibPatchCore(device)


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


def auroc(normal: Sequence[float], anomalous: Sequence[float]) -> float:
    """The image AUROC of two score groups: rank-based, ties counted as half.

    The share of (normal, anomalous) pairs the anomalous score wins, a tie half — the
    Mann-Whitney U, divided by the pair count. An empty group has no AUROC.
    """
    good = np.asarray(normal, dtype=np.float64)
    bad = np.asarray(anomalous, dtype=np.float64)
    if good.size == 0 or bad.size == 0:
        raise ValueError("an AUROC needs at least one normal and one anomalous score")
    wins = int(np.count_nonzero(bad[:, None] > good)) + int(
        np.count_nonzero(bad[:, None] == good)
    ) / 2
    return wins / (good.size * bad.size)


@dataclass(frozen=True, slots=True)
class _Pixels:
    """One MVTec image: its path under the root, the SHA-256 of its bytes, its pixels."""

    rel: str
    sha256: str
    pixels: np.ndarray


@dataclass(frozen=True, slots=True)
class _Row:
    """One test image's scores and model timings, in milliseconds.

    `whole_ms` is the whole `scorer.score()` call, measured on the device around it —
    not the two phases added up.
    """

    anomalous: bool
    pred: float
    station: float
    backbone_ms: float
    search_ms: float
    whole_ms: float


@dataclass(frozen=True, slots=True)
class _Result:
    """Everything the report states, gathered by the run."""

    category: str
    device: str
    root: str
    fitted: bool
    train_count: int
    classes: tuple[tuple[str, int], ...]
    test_count: int
    input_px: int
    auroc_pred: float
    auroc_station: float
    published: float
    backbone_ms: tuple[float, ...]
    search_ms: tuple[float, ...]
    whole_ms: tuple[float, ...]
    peak_gpu_bytes: int | None
    model_id: str
    refusal: str


def _rel(path: Path, root: Path) -> str:
    """The path under the root, as the refusals name it."""
    return path.relative_to(root).as_posix()


def _read(path: Path, root: Path) -> _Pixels:
    """One image's pixels and digest; `_Refused` when not square, `_CannotRun` unreadable."""
    where = _rel(path, root)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise _CannotRun(f"{where}: {_first_line(exc)}") from None
    pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise _CannotRun(f"{where}: not a readable PNG")
    height, width = pixels.shape[:2]
    if height != width:
        raise _Refused(f"{where}: {width} × {height} is not square")
    return _Pixels(where, hashlib.sha256(payload).hexdigest(), pixels)


def _train_images(root: Path, data: Path) -> list[_Pixels]:
    """`train/good`'s images, decoded; the fit needs at least one."""
    directory = data / "train" / "good"
    paths = sorted(directory.glob("*.png")) if directory.is_dir() else []
    if not paths:
        raise _CannotRun(f"{directory}: no .png training images")
    return [_read(path, root) for path in paths]


def _test_images(
    root: Path, data: Path
) -> tuple[list[tuple[_Pixels, bool]], tuple[tuple[str, int], ...]]:
    """Every test class's images as `(image, anomalous)`, classes sorted, and their counts."""
    test_dir = data / "test"
    if not test_dir.is_dir():
        raise _CannotRun(f"{test_dir} is not a directory")
    class_dirs = sorted(path for path in test_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise _CannotRun(f"{test_dir}: no class directories")
    images: list[tuple[_Pixels, bool]] = []
    counts: dict[str, int] = {}
    for class_dir in class_dirs:
        paths = sorted(class_dir.glob("*.png"))
        if not paths:
            raise _CannotRun(f"{class_dir}: no .png test images")
        anomalous = class_dir.name != "good"
        images.extend((_read(path, root), anomalous) for path in paths)
        counts[class_dir.name] = len(paths)
    if not any(not anomalous for _, anomalous in images):
        raise _CannotRun(f"{test_dir}: no good class with images — the AUROC needs one")
    if not any(anomalous for _, anomalous in images):
        raise _CannotRun(f"{test_dir}: no anomalous class — the AUROC needs one")
    return images, tuple(sorted(counts.items()))


def _pass_line(pred: float, station: float) -> str:
    """`PASS`, or `FAIL` naming every AUROC below the line."""
    if pred >= PASS_LINE and station >= PASS_LINE:
        return "PASS"
    below = " and ".join(
        f"{name} {value:.3f}"
        for name, value in (("pred_score", pred), ("masked maximum", station))
        if value < PASS_LINE
    )
    return f"FAIL — {below} below the line"


def _percentile(samples: Sequence[float], q: float) -> float:
    """One percentile of the samples."""
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def _latency_row(name: str, samples: Sequence[float]) -> str:
    """One latency table row: the measure and its p50, p95 and p99."""
    return (f"| {name} | {_percentile(samples, 50):.1f} | {_percentile(samples, 95):.1f} | "
            f"{_percentile(samples, 99):.1f} |")


def _write_report(report_dir: Path, result: _Result) -> Path:
    """The report: numbers only, no images — `reports/` is in git."""
    report_dir.mkdir(parents=True, exist_ok=True)
    trained = (
        f"train/good {result.train_count} (fit)" if result.fitted
        else "fit: none — reused bundle"
    )
    classes = ", ".join(
        f"{name} {count} ({'normal' if name == 'good' else 'anomalous'})"
        for name, count in result.classes
    )
    lines = [
        f"# MVTec check — {result.category} ({result.device})",
        "",
        f"- root {result.root} · window: the whole image, mask all ones · "
        f"input_px {result.input_px}",
        f"- images: {trained} · test {result.test_count} in {len(result.classes)} "
        f"classes — {classes}",
        "",
        "## Scores",
        "",
        f"- pred_score (Anomalib's image score) AUROC: {result.auroc_pred:.3f}",
        f"- masked maximum (the station's score) AUROC: {result.auroc_station:.3f}",
        f"- published image AUROC: {result.published:.3f} — {PUBLISHED_SOURCE}",
        f"- pass line (both ≥ {PASS_LINE:g}): {_pass_line(result.auroc_pred,
                                                            result.auroc_station)}",
        "",
        f"## Latency — per test image on {result.device}",
        "",
        "| measure | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|",
        _latency_row("backbone", result.backbone_ms),
        _latency_row("search", result.search_ms),
        _latency_row("whole model call", result.whole_ms),
        "",
    ]
    if result.peak_gpu_bytes is not None:
        lines += [f"- peak GPU memory (fit): {result.peak_gpu_bytes / 2**20:.0f} MiB", ""]
    elif not result.fitted and result.device == "cuda":
        lines += ["- peak GPU memory: not measured — reused bundle, no fit", ""]
    lines += [
        "## Bundle",
        "",
        f"- model_id: {result.model_id}",
        "- validation-only: refused by the station outside validation mode",
        f"  - load_bundle without validation: {result.refusal}",
        "",
    ]
    path = report_dir / f"{_today()}-{result.category}-{result.device}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(
    args: argparse.Namespace,
    *,
    make_patchcore: Callable[[str], PatchCore] = _real_patchcore,
    cuda_available: Callable[[], bool] = _cuda_available,
    hide_cuda: Callable[[], None] = _hide_cuda,
) -> int:
    """Check one MVTec category; 0 on the pass line, 1 below it or non-square, 2 cannot run."""
    try:
        return _check(args, make_patchcore, cuda_available, hide_cuda)
    except KeyboardInterrupt:
        print("stopped by the operator — no report written")
        return 2


def _check(
    args: argparse.Namespace,
    make_patchcore: Callable[[str], PatchCore],
    cuda_available: Callable[[], bool],
    hide_cuda: Callable[[], None],
) -> int:
    if args.device == "cpu" and args.bundle is None:
        print("cpu: --device cpu scores an existing bundle only — give --bundle MODEL_ID")
        return 2
    if args.device == "cuda":
        try:
            device_for("torch-cuda", cuda_available=cuda_available())
        except ConfigError as exc:
            print(exc)
            return 2
    else:
        hide_cuda()

    root = Path(args.root)
    data = root / args.category
    if not data.is_dir():
        print(f"mvtec: {data} does not exist — download the {args.category} category "
              "there first (S1.1)")
        return 2
    published = PUBLISHED_AUROC.get(args.category)
    if published is None:
        pinned = ", ".join(sorted(PUBLISHED_AUROC))
        print(f"category: {args.category!r} has no published AUROC — pinned: {pinned}")
        return 2

    models_root = Path(args.models)
    bundle: Bundle | None = None
    if args.bundle is not None:
        try:
            bundle = load_bundle(models_root, args.bundle, validation=True)
        except ConfigError as exc:
            print(f"bundle: {exc}")
            return 2
    input_px = INPUT_PX if bundle is None else bundle.input_px

    try:
        train_pixels = [] if bundle is not None else _train_images(root, data)
        test_images, classes = _test_images(root, data)
    except _Refused as exc:
        print(f"refused: {exc}")
        return 1
    except _CannotRun as exc:
        print(f"mvtec: {exc}")
        return 2

    device = args.device
    try:
        model = make_patchcore(device)
    except ConfigError as exc:
        print(exc)
        return 2
    peak_gpu_bytes: int | None = None
    model_id: str
    bank: np.ndarray
    if bundle is None:
        try:
            fitted = model.fit(
                [to_input(image.pixels, input_px) for image in train_pixels],
                FitSettings(CORESET_RATIO, SEED, device),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"fit: {_first_line(exc)}")
            return 2
        info = BundleInfo(
            input_px=input_px,
            window_px=int(train_pixels[0].pixels.shape[0]),
            coreset_ratio=CORESET_RATIO,
            seed=SEED,
            device=device,
            data_source=DATA_SOURCE,
            images=tuple(
                ImageRecord(image.rel, DATA_UNIT, image.sha256) for image in train_pixels
            ),
        )
        try:
            model_id = write_bundle(models_root, f"mvtec-{args.category}", fitted.bank, info)
        except (ConfigError, OSError) as exc:
            print(f"bundle: {_first_line(exc)}")
            return 2
        bank = fitted.bank
        peak_gpu_bytes = fitted.peak_gpu_bytes
    else:
        bank, model_id = bundle.bank, bundle.model_id
    try:
        scorer = model.scorer(bank)
    except (RuntimeError, ValueError) as exc:
        print(f"scorer: {_first_line(exc)}")
        return 2

    rows: list[_Row] = []
    try:
        for image, anomalous in test_images:
            window_input = to_input(image.pixels, input_px)
            started = _clock()
            scored = scorer.score(window_input)
            whole_ms = (_clock() - started) * 1000.0
            side = int(image.pixels.shape[0])
            mask = np.ones((side, side), dtype=np.uint8)
            _, station = score_map(
                scored.anomaly_map, mask, (0, 0, side, side), side, 0
            )
            rows.append(_Row(anomalous, scored.pred_score, station, scored.backbone_ms,
                             scored.search_ms, whole_ms))
    except InferenceError as exc:
        print(f"inference: {exc}")
        return 2
    except (RuntimeError, ValueError) as exc:
        print(f"score: {_first_line(exc)}")
        return 2

    normal = [row for row in rows if not row.anomalous]
    damaged = [row for row in rows if row.anomalous]
    auroc_pred = auroc([row.pred for row in normal], [row.pred for row in damaged])
    auroc_station = auroc([row.station for row in normal], [row.station for row in damaged])
    passed = auroc_pred >= PASS_LINE and auroc_station >= PASS_LINE

    try:
        load_bundle(models_root, model_id)
    except ConfigError as exc:
        refusal = _first_line(exc)
    else:
        print(f"bundle {model_id}: it loaded without validation — a validation-only bundle "
              "must be refused (ADR-0016)")
        return 2

    result = _Result(
        category=args.category,
        device=device,
        root=str(root),
        fitted=bundle is None,
        train_count=len(train_pixels),
        classes=classes,
        test_count=len(test_images),
        input_px=input_px,
        auroc_pred=auroc_pred,
        auroc_station=auroc_station,
        published=published,
        backbone_ms=tuple(row.backbone_ms for row in rows),
        search_ms=tuple(row.search_ms for row in rows),
        whole_ms=tuple(row.whole_ms for row in rows),
        peak_gpu_bytes=peak_gpu_bytes,
        model_id=model_id,
        refusal=refusal,
    )
    try:
        path = _write_report(Path(args.report_dir), result)
    except OSError as exc:
        print(f"report: {_first_line(exc)}")
        return 2
    print(f"{args.category} on {device}: pred_score AUROC {auroc_pred:.3f}, masked maximum "
          f"{auroc_station:.3f} — {_pass_line(auroc_pred, auroc_station)}")
    print(f"model_id: {model_id}")
    print(f"report: {path}")
    return 0 if passed else 1
