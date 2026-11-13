"""`qs train-lane`: the three-class lane classifier, trained and bundled (#156).

    qs train-lane --lane lane.yml --labels L.json --footage DIR --split split.yml --product P
                  [--seed 42] [--epochs 8] [--input-px 224] [--models models] [--device cuda|cpu]

It trains on the labelled passes of the split file's `train` clips and on nothing else: a pass in
any other clip is not read, and a train clip whose bytes changed since the split is refused. The
last 20% of the passes (by video, then frame; rounded up) are validation and never trained on.
Each sample's input is `to_input(crop(frame, lane), input_px)`; training samples are first
shifted by up to 8 px in x and y and scaled in brightness by 0.9..1.1. ResNet18 on the pinned
ImageNet weights, Adam at 1e-4, batches of 32, cross-entropy weighted by inverse class frequency.
The same seed gives the same model on the same device. The bundle goes to
`models/lane-<product>-v<N>/` (`qs.lane.model`); the command prints the validation confusion
counts, peak GPU memory, the minutes taken and the `model_id`.

Exit codes: 0 written · 1 refused (nothing to learn from) · 2 could not run (a file, a hash, CUDA).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from qs.cmd.train import positive_int, product, seed
from qs.errors import ConfigError
from qs.inspector.window import to_input
from qs.labels import Cup, FrameReadError, LabelsError, VideoFrames, load_labels
from qs.lane.lanefile import Lane, crop, load_lane
from qs.lane.model import build, predict, write_lane_bundle
from qs.lane.samples import CLASSES, Sample, samples
from qs.lane.split import Split, file_sha256, load_split, verify
from qs.lane.weights import resnet18_weights

ROOT = Path(__file__).resolve().parents[3]
SHIFT_PX = 8
BRIGHTNESS = (0.9, 1.1)
BATCH = 32
BETWEEN = [16, 48]
LEARNING_RATE = 1e-4


class _Refused(Exception):
    """Nothing to learn from: exit 1, one line."""


def add_parser(commands: Any) -> None:
    """Register `train-lane` on the `qs` subcommands."""
    parser = commands.add_parser("train-lane", help="train the lane classifier")
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--labels", required=True, metavar="PATH", help="the labels file")
    parser.add_argument("--footage", required=True, metavar="DIR", help="the clips' folder")
    parser.add_argument("--split", required=True, metavar="PATH", help="the split file")
    parser.add_argument("--product", required=True, type=product, help="e.g. mochi")
    parser.add_argument("--seed", type=seed, default=42)
    parser.add_argument("--epochs", type=positive_int, default=8)
    parser.add_argument("--input-px", type=positive_int, default=224)
    parser.add_argument("--between", type=positive_int, nargs=2, default=BETWEEN,
                        metavar=("MIN", "MAX"),
                        help="frames between two passes that count as the chain moving on, "
                        "whose space gives `none` samples (default 16 48)")
    parser.add_argument("--models", default="models", metavar="DIR")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="cuda never falls back to the CPU")


def _pinned_weights() -> Path:
    return resnet18_weights(ROOT / ".torch")


def _seed_everything(value: int) -> None:
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _padded_crop(frame: np.ndarray, lane: Lane) -> np.ndarray:
    """The lane window plus `SHIFT_PX` on every side, edge pixels repeated past the frame, so a
    shifted window can be cut from it. The window itself is checked by `crop`."""
    crop(frame, lane)
    x, y, w, h = lane.window
    height, width = frame.shape[:2]
    top, left = y - SHIFT_PX, x - SHIFT_PX
    bottom, right = y + h + SHIFT_PX, x + w + SHIFT_PX
    region = frame[max(top, 0) : min(bottom, height), max(left, 0) : min(right, width)]
    return cv2.copyMakeBorder(
        region, max(-top, 0), max(bottom - height, 0), max(-left, 0), max(right - width, 0),
        cv2.BORDER_REPLICATE,
    )


def _model_input(
    padded: np.ndarray, lane: Lane, input_px: int, dx: int = 0, dy: int = 0, factor: float = 1.0
) -> np.ndarray:
    """The input for the window shifted by `(dx, dy)`; with no shift and factor 1 it is
    `to_input(crop(frame, lane), input_px)`."""
    _, _, w, h = lane.window
    window = padded[SHIFT_PX + dy : SHIFT_PX + dy + h, SHIFT_PX + dx : SHIFT_PX + dx + w]
    if factor != 1.0:
        window = cv2.convertScaleAbs(window, alpha=factor)
    return to_input(window, input_px)


def _validation_keys(cups: list[Cup]) -> list[tuple[str, int]]:
    """The last 20% of the passes (rounded up), by video then frame."""
    ordered = sorted(cups, key=lambda c: (c.video, c.frame, c.order))
    count = -(-len(ordered) * 2 // 10)
    return [(c.video, c.order) for c in ordered[len(ordered) - count :]]


def _read_crops(
    footage: Path, wanted: list[Sample], lane: Lane
) -> dict[tuple[str, int], np.ndarray]:
    """The padded crop of every wanted frame, each video read once, in frame order."""
    crops: dict[tuple[str, int], np.ndarray] = {}
    for video in sorted({s.video for s in wanted}):
        frames = VideoFrames(footage / video)
        try:
            for frame in sorted({s.frame for s in wanted if s.video == video}):
                crops[(video, frame)] = _padded_crop(frames.read(frame), lane)
        finally:
            frames.close()
    return crops


def _frame_counts(footage: Path, videos: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for video in videos:
        frames = VideoFrames(footage / video)
        counts[video] = len(frames.ids)
        frames.close()
    return counts


def _class_weights(counts: Counter[str], device: str) -> torch.Tensor:
    total = sum(counts.values())
    weights = [total / (len(CLASSES) * counts[c]) if counts[c] else 0.0 for c in CLASSES]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_model(
    model: torch.nn.Module,
    images: list[np.ndarray],
    targets: list[int],
    lane: Lane,
    args: argparse.Namespace,
) -> None:
    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    counts = Counter(CLASSES[t] for t in targets)
    loss_fn = torch.nn.CrossEntropyLoss(weight=_class_weights(counts, args.device))
    model.to(args.device).train()
    for _ in range(args.epochs):
        order = rng.permutation(len(images))
        for start in range(0, len(order), BATCH):
            chosen = order[start : start + BATCH]
            shifts = rng.integers(-SHIFT_PX, SHIFT_PX + 1, size=(len(chosen), 2))
            factors = rng.uniform(*BRIGHTNESS, size=len(chosen))
            batch = np.stack([
                _model_input(images[i], lane, args.input_px, int(dx), int(dy), float(f))
                for i, (dx, dy), f in zip(chosen, shifts, factors, strict=True)
            ])
            labels = torch.tensor([targets[i] for i in chosen], device=args.device)
            optimizer.zero_grad()
            loss = loss_fn(model(torch.from_numpy(batch).to(args.device)), labels)
            loss.backward()
            optimizer.step()


def _validate(
    model: torch.nn.Module, images: list[np.ndarray], targets: list[int], lane: Lane,
    args: argparse.Namespace,
) -> tuple[list[int], list[list[int]]]:
    """Each validation sample's predicted class, and the confusion counts (rows true)."""
    predicted: list[int] = []
    for start in range(0, len(images), 64):
        batch = np.stack([_model_input(i, lane, args.input_px) for i in images[start : start + 64]])
        predicted.extend(int(i) for i in predict(model, batch, args.device).argmax(axis=1))
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    for true, guess in zip(targets, predicted, strict=True):
        confusion[true][guess] += 1
    return predicted, confusion


def _run(args: argparse.Namespace, weights: Callable[[], Path | None]) -> int:
    footage = Path(args.footage)
    lane_path, labels_path, split_path = Path(args.lane), Path(args.labels), Path(args.split)
    lane_sha = file_sha256(lane_path)  # hashed when read, not after minutes of training
    split_sha = file_sha256(split_path)
    lane = load_lane(lane_path)
    split = load_split(split_path)
    train_videos = list(split.videos("train"))
    if not train_videos:
        raise ConfigError("the split file has no train clip")
    train_clips = {v: split.clips[v] for v in train_videos}
    verify(Split(train_clips), footage, file_sha256)
    if not labels_path.is_file():
        raise ConfigError(f"labels file {labels_path} not found")
    labels_sha = file_sha256(labels_path)
    try:
        labelled = load_labels(labels_path, labels_path.stem).cups
    except LabelsError as exc:
        raise ConfigError(f"labels: {exc}") from exc
    cups = [c for c in labelled if c.video in train_clips]
    if not cups:
        raise _Refused("no labelled pass in a train clip")
    validation = set(_validation_keys(cups))
    if len(validation) == len(cups):
        raise _Refused("at least two labelled passes are needed, one to validate on")
    counts = _frame_counts(footage, train_videos)
    if args.between[0] > args.between[1]:
        raise ConfigError(f"--between {args.between[0]} {args.between[1]}: MIN is above MAX")
    try:
        every = samples(cups, counts, between=(args.between[0], args.between[1]))
    except ValueError as exc:
        raise _Refused(str(exc)) from exc
    trained = [s for s in every if (s.video, s.order) not in validation]
    held = [s for s in every if (s.video, s.order) in validation]
    per_class = Counter(s.cls for s in trained)
    for cls in ("empty", "full"):
        if not per_class[cls]:
            raise _Refused(f"no {cls} pass to train on")
    if not per_class["none"]:
        raise _Refused("no none sample to train on: label a gap, or widen --between")
    held_per_class = Counter(s.cls for s in held)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ConfigError("device: --device cuda, but torch sees no CUDA device")

    started = time.monotonic()
    pretrained = weights()
    crops = _read_crops(footage, every, lane)
    _seed_everything(args.seed)
    model = build(pretrained)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        _train_model(
            model, [crops[(s.video, s.frame)] for s in trained],
            [CLASSES.index(s.cls) for s in trained], lane, args,
        )
    except RuntimeError as exc:
        first = str(exc).strip().splitlines()
        raise ConfigError(f"training failed: {first[0] if first else type(exc).__name__}") from exc
    predicted, confusion = _validate(
        model, [crops[(s.video, s.frame)] for s in held],
        [CLASSES.index(s.cls) for s in held], lane, args,
    )
    peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    minutes = (time.monotonic() - started) / 60

    record = {
        "product": args.product,
        "classes": list(CLASSES),
        "input_px": args.input_px,
        "lane": {"window": list(lane.window), "fps": lane.fps},
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "pretrained_sha256": file_sha256(pretrained) if pretrained is not None else None,
        "lane_sha256": lane_sha,
        "labels_sha256": labels_sha,
        "split_sha256": split_sha,
        "batch_size": BATCH,
        "learning_rate": LEARNING_RATE,
        "shift_px": SHIFT_PX,
        "brightness": list(BRIGHTNESS),
        "between": list(args.between),
        "train_clips": {video: sha for video, (_, sha) in train_clips.items()},
        "samples_per_class": {c: per_class[c] for c in CLASSES},
        "validation_per_class": {c: held_per_class[c] for c in CLASSES},
        "validation_passes": [list(key) for key in sorted(validation)],
        "validation_predicted": predicted,
        "validation_confusion": confusion,
    }
    models = Path(args.models)
    model_id = write_lane_bundle(models, args.product, model, record)

    print(f"{len(cups)} passes: {len(cups) - len(validation)} train, {len(validation)} validation; "
          f"{len(labelled) - len(cups)} passes outside the train clips not read")
    print("samples per class: " + ", ".join(f"{c} {per_class[c]}" for c in CLASSES))
    print("validation samples per class: " + ", ".join(f"{c} {held_per_class[c]}" for c in CLASSES))
    for cls in CLASSES:
        if not held_per_class[cls]:
            print(f"WARN: no {cls} sample in validation; its row of the table is blind")
    print(f"validation confusion (rows true, columns predicted; {' '.join(CLASSES)}): "
          f"{yaml.safe_dump(confusion, default_flow_style=True).strip()}")
    if peak is not None:
        print(f"peak GPU memory: {peak / 2**20:.0f} MiB")
    print(f"minutes: {minutes:.1f}")
    print(f"bundle: {models / model_id}")
    print(f"model_id: {model_id}")
    return 0


def main(
    args: argparse.Namespace, *, weights: Callable[[], Path | None] = _pinned_weights
) -> int:
    """Train one bundle; 0 written, 1 refused, 2 could not run. `weights` gives the pretrained
    file, or None for random weights (tests)."""
    try:
        return _run(args, weights)
    except _Refused as exc:
        print(f"train-lane: refused: {exc}")
        return 1
    except (ConfigError, FrameReadError, OSError) as exc:
        print(f"train-lane: {exc}")
        return 2
    except KeyboardInterrupt:
        print("train-lane: stopped by the operator — no bundle written")
        return 2
