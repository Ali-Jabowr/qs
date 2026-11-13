"""The lane classifier: a ResNet18 with three outputs, and its hashed bundle (#156)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def build(weights: Path | None) -> torch.nn.Module:
    raise NotImplementedError


def predict(model: torch.nn.Module, inputs: np.ndarray, device: str) -> np.ndarray:
    raise NotImplementedError


def write_lane_bundle(
    models_root: Path, product: str, model: torch.nn.Module, record: dict[str, Any]
) -> str:
    raise NotImplementedError


def load_lane_bundle(models_root: Path, model_id: str) -> Any:
    raise NotImplementedError
