"""The pre-registered threshold rule and the threshold file (#158)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from qs.errors import ConfigError
from qs.eval.threshold import UnitScore, choose_threshold

HALF = 0.5


@dataclass(frozen=True, slots=True)
class Threshold:
    """The chosen value and what it came from."""

    model_id: str
    far_pct: int
    value: float
    n_full: int
    n_above_half: int
    n_unseen: int
    labels_sha256: str


def lane_threshold(scores: Sequence[float], far_pct: int) -> float:
    """0.5 when at most `far_pct` percent of the full-pass `scores` lie above it, else the value
    `choose_threshold` gives at `far_pct` (always above 0.5 then)."""
    if not scores:
        raise ValueError("no full pass to set the threshold on")
    units = [UnitScore(str(i), float(s), False) for i, s in enumerate(scores)]
    above = sum(1 for s in scores if s > HALF)
    if above <= far_pct * len(scores) // 100:
        return HALF
    value = choose_threshold(units, far_pct).value
    if value is None or not math.isfinite(value):
        raise ValueError(f"no finite threshold at {far_pct}%: {value}")
    return max(HALF, value)


def write_threshold(threshold: Threshold, path: Path) -> None:
    body = {"version": 1, **asdict(threshold)}
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def load_threshold(path: Path) -> Threshold:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data["version"] != 1:
            raise ValueError(f"version {data['version']}")
        return Threshold(
            model_id=str(data["model_id"]), far_pct=int(data["far_pct"]),
            value=float(data["value"]), n_full=int(data["n_full"]),
            n_above_half=int(data["n_above_half"]), n_unseen=int(data["n_unseen"]),
            labels_sha256=str(data["labels_sha256"]),
        )
    except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"{path} is not a threshold file: {exc!r}") from exc
