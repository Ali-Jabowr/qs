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

On a line config (`inspector.type: line`, #108) it reads a labels file and its videos instead:

    bin/qs train --config PATH --labels PATH --footage DIR --window-px N --product P
                 [--gap-s 3] [--seed 42] [--coreset 0.1] [--input-px 256] [--models models]

The labels file covers one camera position, so a run makes that position's model. The frames
labelled `unit` are the good ones; no other label trains. They are split by time
(`qs.eval.line_split`, `inspector.line.fps` and `--gap-s`) and only the training block is read.
Each of its frames must show a cup wholly in the zone (`find_cup`); the window is centred on
the rim. An image's unit is its video, and its SHA-256 is that of the decoded frame's pixels.

Exit codes: 0 written · 1 refused (a name, a geometry, a placement) · 2 could not run
(config, CUDA, weights). Every refusal is one line, never a traceback.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from qs.config import LineConfig, StationConfig, load_config
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.eval.line_split import LineSplitError, split_units, video_frame
from qs.inspector.appearance import real_cuda_available
from qs.inspector.bundle import PRODUCT, BundleInfo, ImageRecord, write_bundle
from qs.inspector.cup import find_cup, no_cup_reason
from qs.inspector.localize import localize
from qs.inspector.model import FitSettings, PatchCore, device_for
from qs.inspector.window import cut_window, to_input
from qs.labels import FrameReadError, LabelsError, VideoFrames, load_labels
from qs.names import FrameName, FrameNameError, parse_name

# Seconds dropped before each held-out block of a line split when `--gap-s` is not given; the
# flag has no argparse default so that a bench config can refuse it.
DEFAULT_GAP_S = 3.0


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


def non_negative_float(text: str) -> float:
    """argparse type: a finite number at or above zero."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{value:g} is not a finite number at or above 0")
    return value


class _Refused(Exception):
    """The run is refused: exit 1, one line."""


class _CannotRun(Exception):
    """The run could not start: exit 2, one line."""


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


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def main(
    args: argparse.Namespace,
    *,
    make_patchcore: Callable[[str], PatchCore] = _real_patchcore,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> int:
    """Train one bundle; 0 written, 1 refused, 2 could not run."""
    try:
        return _train(args, make_patchcore, cuda_available)
    except KeyboardInterrupt:
        print("stopped by the operator — no bundle written")
        return 2


def _dataset_inputs(
    args: argparse.Namespace, config: StationConfig
) -> tuple[list[np.ndarray], list[ImageRecord], str]:
    """A dataset's `train/good`: the model inputs, their records, and what they were."""
    good = Path(args.dataset) / "train" / "good"
    if not good.is_dir():
        raise _CannotRun(f"dataset: {good} is not a directory")
    inputs: list[np.ndarray] = []
    images: list[ImageRecord] = []
    try:
        frames, geometry = _training_frames(good)
        for path, name in frames:
            window_input, digest = _window_input(path, config, args.window_px, args.input_px)
            inputs.append(window_input)
            images.append(ImageRecord(path.name, name.unit, digest))
    except OSError as exc:
        raise _CannotRun(f"dataset: {_first_line(exc)}") from None
    units = {image.unit for image in images}
    trained_on = f"trained on {len(images)} frames of {len(units)} units, geometry {geometry}"
    return inputs, images, trained_on


def _line_inputs(
    args: argparse.Namespace, line: LineConfig
) -> tuple[list[np.ndarray], list[ImageRecord], str]:
    """The training block of one position's `unit` frames: inputs, records, what they were."""
    if args.labels is None or args.footage is None:
        raise _CannotRun("a line config trains with --labels and --footage")
    if line.fps is None:
        raise _CannotRun("inspector.line.fps: line training needs this key to split by time")
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise _CannotRun(f"labels file {labels_path} not found")
    try:
        labels = load_labels(labels_path, labels_path.stem)
    except LabelsError as exc:
        raise _CannotRun(f"labels: {exc}") from None
    try:
        gap_s = DEFAULT_GAP_S if args.gap_s is None else args.gap_s
        split = split_units(labels, fps=line.fps, gap_s=gap_s)
    except LineSplitError as exc:
        raise _Refused(str(exc)) from None
    if not split.train:
        raise _Refused(f"{labels_path.name}: no frame labelled unit to train on")
    inputs: list[np.ndarray] = []
    images: list[ImageRecord] = []
    videos: dict[str, VideoFrames] = {}
    try:
        for frame_id in split.train:
            video, index = video_frame(frame_id)
            if video not in videos:
                videos[video] = VideoFrames(Path(args.footage) / video)
            pixels = videos[video].read(index)
            cup = find_cup(pixels, line)
            if cup is None:
                raise _Refused(f"{frame_id}: {no_cup_reason(pixels, line)}")
            window, _ = cut_window(pixels, cup.rim[0], args.window_px)
            inputs.append(to_input(window, args.input_px))
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            images.append(ImageRecord(frame_id, video, digest))
    except FrameReadError as exc:
        raise _CannotRun(f"footage: {exc}") from None
    finally:
        for frames in videos.values():
            frames.close()
    trained_on = (
        f"trained on {len(images)} frames of {len(videos)} videos, "
        f"held out {len(split.calibrate)} to calibrate and {len(split.test)} to test, "
        f"{len(split.dropped)} dropped in the gaps"
    )
    return inputs, images, trained_on


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
    line = config.inspector.line if config.inspector.type == "line" else None
    try:
        if line is None:
            if args.labels is not None:
                raise _CannotRun("--labels trains a line config only (inspector.type: line)")
            if args.footage is not None or args.gap_s is not None:
                raise _CannotRun(
                    "--footage and --gap-s are for a line config only (inspector.type: line)"
                )
            inputs, images, trained_on = _dataset_inputs(args, config)
        else:
            inputs, images, trained_on = _line_inputs(args, line)
    except _Refused as exc:
        print(f"refused: {exc}")
        return 1
    except _CannotRun as exc:
        print(exc)
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

    rows, features = fitted.bank.shape
    print(
        f"{trained_on}: window {args.window_px} px → input {args.input_px} px, device {device}"
    )
    print(f"memory bank: {rows} rows × {features} · coreset {args.coreset:g} · seed {args.seed}")
    if fitted.peak_gpu_bytes is not None:
        print(f"peak GPU memory: {fitted.peak_gpu_bytes / 2**20:.0f} MiB")
    print(f"bundle: {models / model_id}")
    print(f"model_id: {model_id}")
    print(f"inspector.model: {model_id}")
    return 0
