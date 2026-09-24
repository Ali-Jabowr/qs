"""`qs train`: a dataset's `train/good` in, a verified model bundle out (FR-53, docs/15 §5, §8).

    bin/qs train --config PATH --dataset DIR --window-px N --product P
                 [--seed 42] [--coreset 0.1] [--input-px 256] [--models models]

It reads `DIR/train/good/*.png` and nothing else. Every name must parse under the frame-name
rule as a dataset unit (`uNNN`) in state `good`, all under one geometry; every frame must
localize to exactly one unit with the config's `inspector.localize`. Each unit's window of
`--window-px`, resized once to `--input-px`, goes to PatchCore, fitted on the device
`inspector.backend` names — `torch-cuda` never falls back to the CPU. The bundle records
`data_source: own`, the unit IDs and every image's SHA-256. It prints the `model_id` and the
line for `inspector.model`; it edits no config (`qs calibrate`, P08, writes it with the
threshold).

Exit codes: 0 written · 1 refused (a name, a geometry, a placement) · 2 could not run
(config, CUDA, weights). Every refusal is one line, never a traceback.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from qs.config import StationConfig, load_config
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.inspector.bundle import PRODUCT, BundleInfo, ImageRecord, write_bundle
from qs.inspector.localize import localize
from qs.inspector.model import FitSettings, PatchCore, device_for
from qs.inspector.window import cut_window, to_input
from qs.names import FrameName, FrameNameError, parse_name


def product(text: str) -> str:
    """argparse type: a product name for the model_id, e.g. `mochi`."""
    if PRODUCT.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(f"{text!r}: lower-case words joined by '-', e.g. mochi")
    return text


def positive_int(text: str) -> int:
    """argparse type: an integer above zero."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not above zero")
    return value


def coreset(text: str) -> float:
    """argparse type: the coreset share, in (0, 1]."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"{value:g} is not in (0, 1]")
    return value


def seed(text: str) -> int:
    """argparse type: a seed NumPy accepts, 0 to 2**32 − 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError(f"{value} is not in 0 … 2**32 − 1")
    return value


class _Refused(Exception):
    """The run is refused: exit 1, one line."""


def _training_frames(good: Path) -> tuple[list[tuple[Path, FrameName]], str]:
    """Every `*.png` in `train/good` by name, checked before any is read; and the geometry."""
    paths = sorted(good.glob("*.png"))
    if not paths:
        raise _Refused(f"{good}: no .png frames to train on")
    frames: list[tuple[Path, FrameName]] = []
    for path in paths:
        try:
            name = parse_name(path.name)
        except FrameNameError as exc:
            raise _Refused(f"{path.name}: {exc}") from None
        if not name.unit.startswith("u"):
            raise _Refused(
                f"{path.name}: {name.unit} is not a dataset unit (uNNN) — train/good holds "
                "dataset units only"
            )
        if name.state != "good":
            raise _Refused(f"{path.name}: state {name.state} — train/good holds good frames only")
        if frames and name.geometry != frames[0][1].geometry:
            raise _Refused(
                f"{path.name}: geometry {name.geometry}, but {frames[0][0].name} is "
                f"{frames[0][1].geometry} — one model per geometry"
            )
        frames.append((path, name))
    return frames, frames[0][1].geometry


def _window_input(
    path: Path, config: StationConfig, window_px: int, input_px: int
) -> tuple[np.ndarray, str]:
    """One frame's model input and the SHA-256 of the bytes it was decoded from."""
    data = path.read_bytes()
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise _Refused(f"{path.name}: not a readable PNG")
    try:
        silhouette = localize(pixels, config.inspector.localize)
    except PlacementFault as exc:
        raise _Refused(f"{path.name}: placement: {exc.detail}") from None
    except InferenceError as exc:
        raise _Refused(f"{path.name}: inference: {exc}") from None
    window, _ = cut_window(pixels, silhouette.centroid, window_px)
    return to_input(window, input_px), hashlib.sha256(data).hexdigest()


def _real_patchcore(device: str) -> PatchCore:
    from qs.inspector.patchcore import AnomalibPatchCore

    return AnomalibPatchCore(device)


def _cuda_available() -> bool:
    from qs.inspector.patchcore import cuda_available

    return cuda_available()


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def main(
    args: argparse.Namespace,
    *,
    make_patchcore: Callable[[str], PatchCore] = _real_patchcore,
    cuda_available: Callable[[], bool] = _cuda_available,
) -> int:
    """Train one bundle; 0 written, 1 refused, 2 could not run."""
    try:
        return _train(args, make_patchcore, cuda_available)
    except KeyboardInterrupt:
        print("stopped by the operator — no bundle written")
        return 2


def _train(
    args: argparse.Namespace,
    make_patchcore: Callable[[str], PatchCore],
    cuda_available: Callable[[], bool],
) -> int:
    try:
        config = load_config(Path(args.config)).config
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    try:
        device = device_for(config.inspector.backend, cuda_available=cuda_available())
    except ConfigError as exc:
        print(exc)
        return 2
    good = Path(args.dataset) / "train" / "good"
    if not good.is_dir():
        print(f"dataset: {good} is not a directory")
        return 2

    inputs: list[np.ndarray] = []
    images: list[ImageRecord] = []
    try:
        frames, geometry = _training_frames(good)
        for path, name in frames:
            window_input, digest = _window_input(path, config, args.window_px, args.input_px)
            inputs.append(window_input)
            images.append(ImageRecord(path.name, name.unit, digest))
    except _Refused as exc:
        print(f"refused: {exc}")
        return 1
    except OSError as exc:
        print(f"dataset: {_first_line(exc)}")
        return 2

    try:
        model = make_patchcore(device)
    except ConfigError as exc:
        print(exc)
        return 2
    try:
        fitted = model.fit(inputs, FitSettings(args.coreset, args.seed, device))
    except (RuntimeError, ValueError) as exc:
        print(f"fit: {_first_line(exc)}")
        return 2
    info = BundleInfo(
        input_px=args.input_px,
        window_px=args.window_px,
        coreset_ratio=args.coreset,
        seed=args.seed,
        device=device,
        data_source="own",
        images=tuple(images),
    )
    models = Path(args.models)
    try:
        model_id = write_bundle(models, args.product, fitted.bank, info)
    except (ConfigError, OSError) as exc:
        print(f"bundle: {_first_line(exc)}")
        return 2

    units = sorted({image.unit for image in images})
    rows, features = fitted.bank.shape
    print(
        f"trained on {len(images)} frames of {len(units)} units, geometry {geometry}: "
        f"window {args.window_px} px → input {args.input_px} px, device {device}"
    )
    print(f"memory bank: {rows} rows × {features} · coreset {args.coreset:g} · seed {args.seed}")
    if fitted.peak_gpu_bytes is not None:
        print(f"peak GPU memory: {fitted.peak_gpu_bytes / 2**20:.0f} MiB")
    print(f"bundle: {models / model_id}")
    print(f"model_id: {model_id}")
    print(f"inspector.model: {model_id}")
    return 0
