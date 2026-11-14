"""A small drawn rig for `qs freeze` and `qs compare` tests (#160).

Three clips (`T-1` train, `C-1` calibrate, `E-1` test) are plain byte files: the tests inject the
frame probabilities, so no video is ever decoded. The lane bundle is a fake `model.pt` with a
`bundle.yml`; nothing loads it as a network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from qs.__main__ import build_parser
from qs.cmd import freeze
from qs.lane.split import Split, file_sha256, write_split

MODEL_ID = "lane-mochi-v1"
HEAD = "a" * 40
E1_CUPS = [("full", 100), ("empty", 200), ("empty", 300), ("full", 400), ("full", 500),
           ("gap", 600), ("empty", 700), ("gap", 800)]


@dataclass(frozen=True)
class Rig:
    root: Path
    footage: Path
    models: Path
    labels: Path
    lane: Path
    split: Path
    threshold: Path
    model_file: Path
    freeze: Path


def make_rig(root: Path, threshold: float = 0.5) -> Rig:
    footage = root / "footage"
    footage.mkdir()
    roles = {"T-1.mkv": "train", "C-1.mkv": "calibrate", "E-1.mkv": "test"}
    for name in roles:
        (footage / name).write_bytes(f"bytes of {name}".encode())
    cups = [{"video": "T-1.mkv", "order": 1, "state": "full", "frame": 50},
            {"video": "C-1.mkv", "order": 1, "state": "empty", "frame": 60}]
    cups += [{"video": "E-1.mkv", "order": i + 1, "state": s, "frame": f}
             for i, (s, f) in enumerate(E1_CUPS)]
    labels = root / "L.json"
    labels.write_text(json.dumps({"version": 1, "source": "L", "frames": {}, "cups": cups}))
    lane = root / "lane.yml"
    lane.write_text(yaml.safe_dump({"version": 1, "window": [8, 4, 32, 32], "fps": 10}))
    split = root / "split.yml"
    write_split(Split({v: (r, file_sha256(footage / v)) for v, r in roles.items()}), split)
    bundle = root / "models" / MODEL_ID
    bundle.mkdir(parents=True)
    model_file = bundle / "model.pt"
    model_file.write_bytes(b"model weights")
    (bundle / "bundle.yml").write_text(yaml.safe_dump({
        "model_id": MODEL_ID, "model_sha256": hashlib.sha256(b"model weights").hexdigest()}))
    thr = root / "threshold.yml"
    thr.write_text(yaml.safe_dump({
        "version": 1, "model_id": MODEL_ID, "far_pct": 1, "value": threshold, "n_full": 300,
        "n_above_half": 2, "n_unseen": 0, "labels_sha256": "0" * 64}))
    return Rig(root, footage, root / "models", labels, lane, split, thr, model_file,
               root / "freeze.yml")


def fake_git(head: str = HEAD, dirty: str = "") -> Callable[[Sequence[str]], str]:
    """A git runner: `status` answers `dirty`, `rev-parse` answers `head`."""

    def run(args: Sequence[str]) -> str:
        if args[0] == "status":
            return dirty
        if args[0] == "rev-parse":
            return head + "\n"
        raise AssertionError(f"unexpected git call {list(args)}")

    return run


def clock() -> datetime:
    return datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def freeze_args(rig: Rig, *extra: str) -> list[str]:
    return ["freeze", "--lane-bundle", MODEL_ID, "--threshold", str(rig.threshold),
            "--lane", str(rig.lane), "--split", str(rig.split), "--out", str(rig.freeze),
            "--models", str(rig.models), *extra]


def run_freeze(rig: Rig, *extra: str, git: Callable[[Sequence[str]], str] | None = None) -> int:
    args = build_parser().parse_args(freeze_args(rig, *extra))
    return freeze.main(args, git=git or fake_git(), clock=clock)
