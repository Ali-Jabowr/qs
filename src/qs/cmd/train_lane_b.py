"""`qs train-lane-b`: PatchCore fitted on lane crops of the train clips' full passes (#159).

    qs train-lane-b --lane lane.yml --labels L.json --footage DIR --split split.yml --product P
                    [--seed 42] [--coreset 0.1] [--input-px 256] [--models models]
                    [--device cuda|cpu]

Method B needs no empty cup. Each full pass (`full`, `full_white`, `full_pink`) of a clip the
split marks `train` gives one image: the lane crop at the labelled frame. The rim finder is
never called. Only train clips are hashed and read.

Exit codes: 0 written · 1 refused (no full pass) · 2 could not run.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from qs.cmd.train import _first_line, _real_patchcore, coreset, positive_int, product, seed
from qs.errors import ConfigError, InferenceError
from qs.inspector.appearance import real_cuda_available
from qs.inspector.bundle import BundleInfo, ImageRecord, pixels_sha256, write_bundle
from qs.inspector.model import FitSettings, PatchCore, Scorer, device_for
from qs.inspector.window import to_input
from qs.labels import CupState, FrameReadError, LabelsError, VideoFrames, load_labels
from qs.lane.lanefile import Lane, crop, load_lane
from qs.lane.split import Split, file_sha256, load_split, verify

_FULL = (CupState.FULL, CupState.FULL_WHITE, CupState.FULL_PINK)


class _Refused(Exception):
    """The run is refused: exit 1, one line."""


class _CannotRun(Exception):
    """The run could not start: exit 2, one line."""


def add_parser(commands: Any) -> None:
    """Register `train-lane-b` on the `qs` subcommands."""
    parser = commands.add_parser(
        "train-lane-b", help="PatchCore on the lane crop of every full pass of the train clips"
    )
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--labels", required=True, metavar="PATH",
                        help="the labels file (qs label)")
    parser.add_argument("--footage", required=True, metavar="DIR", help="the folder of clips")
    parser.add_argument("--split", required=True, metavar="PATH", help="the split file")
    parser.add_argument("--product", required=True, type=product, metavar="P",
                        help="for the model_id, e.g. mochi-lane")
    parser.add_argument("--seed", type=seed, default=42, metavar="N", help="coreset seed (42)")
    parser.add_argument("--coreset", type=coreset, default=0.1, metavar="R",
                        help="coreset share (0.1)")
    parser.add_argument("--input-px", type=positive_int, default=256, metavar="N",
                        help="the model input (256)")
    parser.add_argument("--max-passes", type=positive_int, default=150, metavar="N",
                        help="fit at most N full passes, evenly spaced in time (150, about "
                        "0.9 GiB of embeddings at 256 px)")
    parser.add_argument("--states", default="stub")
    parser.add_argument("--models", default="models", metavar="DIR",
                        help="where bundles are written (models)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="the torch device (cuda; never a silent fallback to cpu)")


def lane_b_score(pixels: np.ndarray, lane: Lane, scorer: Scorer, input_px: int) -> float:
    """Method B's score of one frame: PatchCore's `pred_score` of the lane crop."""
    score = float(scorer.score(to_input(crop(pixels, lane), input_px)).pred_score)
    if not math.isfinite(score):
        raise InferenceError(f"pred_score is not finite: {score}")
    return score


def _train_videos(split: Split, footage: Path) -> tuple[str, ...]:
    """The train clips, after checking only their bytes against the split file."""
    train = split.videos("train")
    verify(Split({video: split.clips[video] for video in train}), footage, file_sha256)
    return train


def _evenly(items: list[tuple[str, int]], count: int) -> list[tuple[str, int]]:
    """`count` of `items`, the first and last included and the rest evenly spaced."""
    if len(items) <= count:
        return items
    if count == 1:
        return items[:1]
    return [items[i * (len(items) - 1) // (count - 1)] for i in range(count)]


def _inputs(
    args: argparse.Namespace, lane: Lane
) -> tuple[list[np.ndarray], list[ImageRecord], dict[str, int], int]:
    """The model inputs and records, each train clip's full-pass count, and the passes found."""
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise _CannotRun(f"labels file {labels_path} not found")
    try:
        labels = load_labels(labels_path, labels_path.stem)
        split = load_split(Path(args.split))
        footage = Path(args.footage)
        train = _train_videos(split, footage)
    except (LabelsError, ConfigError) as exc:
        raise _CannotRun(str(exc)) from None
    passes = {video: [cup.frame for cup in labels.cups_of(video) if cup.state in _FULL]
              for video in train}
    found = [(video, frame) for video, frames in passes.items() for frame in frames]
    if not found:
        raise _Refused("no full pass in the train clips")
    inputs: list[np.ndarray] = []
    images: list[ImageRecord] = []
    chosen = _evenly(found, args.max_passes)
    for video in dict.fromkeys(v for v, _ in chosen):
        try:
            reader = VideoFrames(footage / video)
        except FrameReadError as exc:
            raise _CannotRun(f"footage: {exc}") from None
        try:
            for _, frame in (c for c in chosen if c[0] == video):
                pixels = reader.read(frame)
                inputs.append(to_input(crop(pixels, lane), args.input_px))
                images.append(ImageRecord(f"{video}#{frame}", video, pixels_sha256(pixels)))
        except FrameReadError as exc:
            raise _CannotRun(f"footage: {exc}") from None
        except ConfigError as exc:
            raise _CannotRun(str(exc)) from None
        finally:
            reader.close()
    return inputs, images, {v: len(f) for v, f in passes.items()}, len(found)


def _train(
    args: argparse.Namespace,
    make_patchcore: Callable[[str], PatchCore],
    cuda_available: Callable[[], bool],
) -> int:
    try:
        device = device_for(
            "torch-cuda" if args.device == "cuda" else "torch-cpu",
            cuda_available=cuda_available(),
        )
        lane = load_lane(Path(args.lane))
    except ConfigError as exc:
        print(exc)
        return 2
    try:
        inputs, images, per_clip, found = _inputs(args, lane)
    except _Refused as exc:
        print(f"refused: {exc}")
        return 1
    except _CannotRun as exc:
        print(exc)
        return 2
    for video, count in per_clip.items():
        print(f"{video}: {count} full passes")
    if len(images) < found:
        print(f"used {len(images)} of {found} full passes")
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
        window_px=lane.window[2],
        coreset_ratio=args.coreset,
        seed=args.seed,
        device=device,
        data_source="own",
        images=tuple(images),
        line=None,
        line_fps=None,
    )
    models = Path(args.models)
    try:
        model_id = write_bundle(models, args.product, fitted.bank, info)
    except (ConfigError, OSError) as exc:
        print(f"bundle: {_first_line(exc)}")
        return 2
    rows, features = fitted.bank.shape
    print(f"trained on {len(images)} full passes of {len({i.unit for i in images})} clips: "
          f"lane crop {lane.window[2]}x{lane.window[3]} → input {args.input_px} px, "
          f"device {device}")
    print(f"memory bank: {rows} rows × {features} · coreset {args.coreset:g} · seed {args.seed}")
    if fitted.peak_gpu_bytes is not None:
        print(f"peak GPU memory: {fitted.peak_gpu_bytes / 2**20:.0f} MiB")
    print(f"bundle: {models / model_id}")
    print(f"model_id: {model_id}")
    return 0


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
