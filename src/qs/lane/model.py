"""The lane classifier: a ResNet18 with three outputs, and its hashed bundle (#156).

A bundle is `models/lane-<product>-v<N>/` holding `model.pt` (the state dict) and `bundle.yml`
(everything that made the model, with the SHA-256 of `model.pt`). Loading re-checks that hash.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from qs.errors import ConfigError
from qs.inspector.bundle import PRODUCT, _versions
from qs.lane.samples import CLASSES

MODEL_FILE = "model.pt"
BUNDLE_FILE = "bundle.yml"
_BATCH = 64


def build(weights: Path | None) -> torch.nn.Module:
    """A ResNet18 with a three-class head. With `weights` (the pinned ImageNet file) the body
    starts from them; with None it has random weights, which is what tests use."""
    # torchvision ships no type information; the module is loaded by name and typed here.
    model: Any = importlib.import_module("torchvision.models").resnet18(weights=None)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    return cast(torch.nn.Module, model)


def predict(model: torch.nn.Module, inputs: np.ndarray, device: str) -> np.ndarray:
    """Class probabilities, shape `(n, 3)`, for `inputs` of shape `(n, 3, H, W)`; the model is
    left in eval mode on `device`."""
    model.to(device).eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), _BATCH):
            batch = torch.from_numpy(np.ascontiguousarray(inputs[start : start + _BATCH]))
            logits = model(batch.to(device))
            rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, len(CLASSES)), dtype=np.float32)


def _next_number(models_root: Path, product: str) -> int:
    pattern = re.compile(rf"lane-{re.escape(product)}-v(\d+)")
    numbers = [
        int(match.group(1))
        for entry in (models_root.iterdir() if models_root.is_dir() else ())
        if entry.is_dir() and (match := pattern.fullmatch(entry.name)) is not None
    ]
    return max(numbers, default=0) + 1


def write_lane_bundle(
    models_root: Path, product: str, model: torch.nn.Module, record: dict[str, Any]
) -> str:
    """Write the next free `lane-<product>-v<N>` and return its model_id. `record` is what the
    caller knows; `model_id`, `model_sha256` and `versions` are added. A bundle is never
    overwritten."""
    if PRODUCT.fullmatch(product) is None:
        raise ConfigError(f"product: {product!r} must match [a-z0-9]+(-[a-z0-9]+)*, e.g. mochi")
    buffer = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buffer)
    model_bytes = buffer.getvalue()
    models_root.mkdir(parents=True, exist_ok=True)
    model_id = f"lane-{product}-v{_next_number(models_root, product)}"
    final = models_root / model_id
    if final.exists():
        raise ConfigError(f"{final}: already exists — a bundle is never overwritten")
    full = {
        **record,
        "model_id": model_id,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "versions": _versions(),
    }
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=models_root))
    try:
        (staging / MODEL_FILE).write_bytes(model_bytes)
        (staging / BUNDLE_FILE).write_text(yaml.safe_dump(full, sort_keys=False), encoding="utf-8")
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return model_id


@dataclass(frozen=True, slots=True)
class LaneBundle:
    """A verified bundle: the model on the CPU, and every field of its `bundle.yml`."""

    model_id: str
    path: Path
    model: torch.nn.Module
    record: dict[str, Any]


def load_lane_bundle(models_root: Path, model_id: str) -> LaneBundle:
    """The bundle `model_id`, after `model.pt` matches the hash its `bundle.yml` records."""
    path = models_root / model_id
    try:
        record = yaml.safe_load((path / BUNDLE_FILE).read_text(encoding="utf-8"))
        data = (path / MODEL_FILE).read_bytes()
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"bundle {model_id}: cannot be read: {exc}") from exc
    if not isinstance(record, dict) or not isinstance(record.get("model_sha256"), str):
        raise ConfigError(f"bundle {model_id}: {BUNDLE_FILE} has no model_sha256")
    actual = hashlib.sha256(data).hexdigest()
    if actual != record["model_sha256"]:
        raise ConfigError(
            f"bundle {model_id}: {MODEL_FILE} has SHA-256 {actual[:8]}…, "
            f"not the recorded {record['model_sha256'][:8]}…"
        )
    model = build(None)
    try:
        model.load_state_dict(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True))
    except (RuntimeError, ValueError) as exc:
        raise ConfigError(f"bundle {model_id}: {MODEL_FILE} does not fit the model: {exc}") from exc
    return LaneBundle(model_id, path, model, record)
