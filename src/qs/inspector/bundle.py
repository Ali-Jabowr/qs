"""The model bundle: a memory bank and the record of exactly how it was made (docs/15 §8).

    models/<model_id>/
        memory_bank.pt   the coreset (torch.save of the memory_bank tensor)
        bundle.yml       backbone, backbone_sha256, layers, input_px, window_px,
                         normalisation {mean, std}, coreset_ratio, num_neighbors, seed, device,
                         versions {python, torch, torchvision, anomalib, timm, numpy, opencv},
                         data_source (own | external:<name>), validation_only,
                         units (sorted unit IDs), images [{file, unit, sha256}]

`model_id = patchcore-<product>-v<n>-<first 12 hex of sha256(memory_bank.pt ‖ bundle.yml)>`,
where `n` is one more than the highest version of that product already under the models
root. `bundle.yml` holds no wall-clock time, the bundle is written into a temporary
directory and renamed into place, and an existing directory is never overwritten.

`load_bundle` refuses — each time with a one-line `ConfigError`, because the station then
refuses to start (docs/16 §2) — a missing directory or file, files that no longer hash to
the `model_id`, a missing or unknown key, a backbone other than the pinned one, a
`validation_only` flag inconsistent with `data_source`, and a validation-only bundle
outside validation mode (ADR-0016). Validation mode is an explicit argument, never a
config key and never a default. torch is imported only to save and load the bank.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qs.config import _MODEL_ID
from qs.errors import ConfigError
from qs.inspector.model import (
    BACKBONE,
    BACKBONE_SHA256,
    LAYERS,
    NORMALISATION_MEAN,
    NORMALISATION_STD,
    NUM_NEIGHBORS,
)

BANK_FILE = "memory_bank.pt"
BUNDLE_FILE = "bundle.yml"
PRODUCT = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_EXTERNAL = re.compile(r"external:[a-z0-9]+(-[a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEYS = (
    "backbone", "backbone_sha256", "layers", "input_px", "window_px", "normalisation",
    "coreset_ratio", "num_neighbors", "seed", "device", "versions", "data_source",
    "validation_only", "units", "images",
)
_VERSIONS = {
    "python": None,
    "torch": "torch",
    "torchvision": "torchvision",
    "anomalib": "anomalib",
    "timm": "timm",
    "numpy": "numpy",
    "opencv": "opencv-python-headless",
}
_DEVICES = ("cuda", "cpu")


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One training image: its file name, its unit ID and the SHA-256 of its bytes."""

    file: str
    unit: str
    sha256: str


@dataclass(frozen=True, slots=True)
class BundleInfo:
    """What a caller knows about a fit; the pins and library versions are added on write."""

    input_px: int
    window_px: int
    coreset_ratio: float
    seed: int
    device: str
    data_source: str
    images: tuple[ImageRecord, ...]


@dataclass(frozen=True, slots=True)
class Bundle:
    """A verified bundle. `bank` is rows × features, float32 and read-only."""

    model_id: str
    path: Path
    bank: np.ndarray
    backbone: str
    backbone_sha256: str
    input_px: int
    window_px: int
    coreset_ratio: float
    seed: int
    device: str
    versions: Mapping[str, str]
    data_source: str
    validation_only: bool
    units: tuple[str, ...]
    images: tuple[ImageRecord, ...]


def _versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for key, dist in _VERSIONS.items():
        if dist is None:
            found[key] = platform.python_version()
            continue
        try:
            found[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            found[key] = "not installed"
    return found


def _bank_bytes(bank: np.ndarray) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(torch.from_numpy(np.array(bank, dtype=np.float32, copy=True)), buffer)
    return buffer.getvalue()


def _check_bank(bank: np.ndarray) -> None:
    if bank.dtype != np.float32 or bank.ndim != 2 or bank.shape[0] == 0:
        raise ValueError(f"a memory bank is non-empty rows × features float32, not "
                         f"{bank.dtype} {bank.shape}")


def _next_version(models_root: Path, product: str) -> int:
    pattern = re.compile(rf"patchcore-{re.escape(product)}-v(\d+)-[0-9a-f]{{12}}")
    versions = [
        int(match.group(1))
        for entry in (models_root.iterdir() if models_root.is_dir() else ())
        if entry.is_dir() and (match := pattern.fullmatch(entry.name)) is not None
    ]
    return max(versions, default=0) + 1


def write_bundle(models_root: Path, product: str, bank: np.ndarray, info: BundleInfo) -> str:
    """Write `models_root/<model_id>/` and return the new `model_id` (docs/15 §8)."""
    if PRODUCT.fullmatch(product) is None:
        raise ConfigError(f"product: {product!r} must match [a-z0-9]+(-[a-z0-9]+)*, e.g. mochi")
    if info.data_source != "own" and _EXTERNAL.fullmatch(info.data_source) is None:
        raise ConfigError(f"data_source: {info.data_source!r} is not own or external:<name>")
    if not info.images:
        raise ConfigError("images: a bundle records at least one training image")
    _check_bank(bank)
    record: dict[str, Any] = {
        "backbone": BACKBONE,
        "backbone_sha256": BACKBONE_SHA256,
        "layers": list(LAYERS),
        "input_px": info.input_px,
        "window_px": info.window_px,
        "normalisation": {"mean": list(NORMALISATION_MEAN), "std": list(NORMALISATION_STD)},
        "coreset_ratio": info.coreset_ratio,
        "num_neighbors": NUM_NEIGHBORS,
        "seed": info.seed,
        "device": info.device,
        "versions": _versions(),
        "data_source": info.data_source,
        "validation_only": info.data_source != "own",
        "units": sorted({image.unit for image in info.images}),
        "images": [
            {"file": image.file, "unit": image.unit, "sha256": image.sha256}
            for image in info.images
        ],
    }
    bank_bytes = _bank_bytes(bank)
    yml_bytes = yaml.safe_dump(record, sort_keys=False, allow_unicode=True).encode("utf-8")
    suffix = hashlib.sha256(bank_bytes + yml_bytes).hexdigest()[:12]
    models_root.mkdir(parents=True, exist_ok=True)
    model_id = f"patchcore-{product}-v{_next_version(models_root, product)}-{suffix}"
    if _MODEL_ID.fullmatch(model_id) is None:
        raise ConfigError(f"{model_id!r} is not a full model_id")
    final = models_root / model_id
    if final.exists():
        raise ConfigError(f"{final}: already exists — a bundle is never overwritten")
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=models_root))
    try:
        (staging / BANK_FILE).write_bytes(bank_bytes)
        (staging / BUNDLE_FILE).write_bytes(yml_bytes)
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return model_id


class _Refusal:
    """Builds the one-line `ConfigError`s for one bundle."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def __call__(self, detail: str) -> ConfigError:
        return ConfigError(f"bundle {self.model_id}: {detail}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(data: Mapping[str, Any], key: str, refuse: _Refusal) -> int:
    value = data[key]
    if not _is_int(value) or value <= 0:
        raise refuse(f"{key}: {value!r} is not a positive integer")
    return int(value)


def _exact_keys(data: object, keys: tuple[str, ...], where: str, refuse: _Refusal) -> None:
    if not isinstance(data, dict):
        raise refuse(f"{where}: not a mapping")
    missing = [key for key in keys if key not in data]
    if missing:
        raise refuse(f"{where}: missing key {missing[0]!r}")
    unknown = sorted(str(key) for key in data if key not in keys)
    if unknown:
        raise refuse(f"{where}: unknown key {unknown[0]!r}")


def _images(value: object, units: tuple[str, ...], refuse: _Refusal) -> tuple[ImageRecord, ...]:
    if not isinstance(value, list) or not value:
        raise refuse("images: not a non-empty list")
    records = []
    for index, entry in enumerate(value):
        where = f"images[{index}]"
        _exact_keys(entry, ("file", "unit", "sha256"), where, refuse)
        file, unit, digest = entry["file"], entry["unit"], entry["sha256"]
        if not isinstance(file, str) or not file:
            raise refuse(f"{where}.file: {file!r} is not a file name")
        if unit not in units:
            raise refuse(f"{where}.unit: {unit!r} is not in units")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise refuse(f"{where}.sha256: {digest!r} is not a SHA-256")
        records.append(ImageRecord(file, unit, digest))
    if {record.unit for record in records} != set(units):
        raise refuse("units: not exactly the units of the images")
    return tuple(records)


def _parse(data: object, refuse: _Refusal) -> dict[str, Any]:
    """Every rule on `bundle.yml`, in key order; the checked values."""
    _exact_keys(data, _KEYS, BUNDLE_FILE, refuse)
    assert isinstance(data, dict)
    if data["backbone"] != BACKBONE:
        raise refuse(f"backbone: {data['backbone']!r} is not the pinned {BACKBONE!r} (ADR-0020)")
    if data["backbone_sha256"] != BACKBONE_SHA256:
        raise refuse(
            f"backbone_sha256: {str(data['backbone_sha256'])[:12]!r}… is not the pinned "
            f"{BACKBONE_SHA256[:12]}… (ADR-0020)"
        )
    if data["layers"] != list(LAYERS):
        raise refuse(f"layers: {data['layers']!r} is not {list(LAYERS)!r} (docs/15 §5)")
    input_px = _positive_int(data, "input_px", refuse)
    window_px = _positive_int(data, "window_px", refuse)
    expected = {"mean": list(NORMALISATION_MEAN), "std": list(NORMALISATION_STD)}
    if data["normalisation"] != expected:
        raise refuse(f"normalisation: {data['normalisation']!r} is not ImageNet's {expected!r}")
    ratio = data["coreset_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, int | float) or not 0 < ratio <= 1:
        raise refuse(f"coreset_ratio: {ratio!r} is not in (0, 1]")
    if data["num_neighbors"] != NUM_NEIGHBORS or not _is_int(data["num_neighbors"]):
        raise refuse(f"num_neighbors: {data['num_neighbors']!r} is not {NUM_NEIGHBORS}")
    seed = data["seed"]
    if not _is_int(seed) or seed < 0:
        raise refuse(f"seed: {seed!r} is not a non-negative integer")
    if data["device"] not in _DEVICES:
        raise refuse(f"device: {data['device']!r} is not one of {', '.join(_DEVICES)}")
    versions = data["versions"]
    _exact_keys(versions, tuple(_VERSIONS), "versions", refuse)
    if not all(isinstance(value, str) for value in versions.values()):
        raise refuse("versions: every version is a string")
    source = data["data_source"]
    if not isinstance(source, str) or (source != "own" and _EXTERNAL.fullmatch(source) is None):
        raise refuse(f"data_source: {source!r} is not own or external:<name>")
    flag = data["validation_only"]
    if not isinstance(flag, bool):
        raise refuse(f"validation_only: {flag!r} is not true or false")
    if flag != (source != "own"):
        raise refuse(
            f"validation_only: {str(flag).lower()} does not match data_source {source!r} "
            "(true exactly when the data is external, ADR-0016)"
        )
    units = data["units"]
    if (
        not isinstance(units, list)
        or not all(isinstance(unit, str) for unit in units)
        or units != sorted(set(units))
    ):
        raise refuse("units: not a sorted list of distinct unit IDs")
    return {
        "input_px": input_px, "window_px": window_px, "coreset_ratio": float(ratio),
        "seed": int(seed), "device": data["device"], "versions": dict(versions),
        "data_source": source, "validation_only": flag, "units": tuple(units),
        "images": _images(data["images"], tuple(units), refuse),
    }


def _load_bank(bank_bytes: bytes, refuse: _Refusal) -> np.ndarray:
    import torch

    try:
        tensor = torch.load(io.BytesIO(bank_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:  # a corrupt or foreign file: one line, never a traceback
        raise refuse(f"{BANK_FILE}: cannot be read ({type(exc).__name__})") from exc
    if not isinstance(tensor, torch.Tensor):
        raise refuse(f"{BANK_FILE}: holds a {type(tensor).__name__}, not a tensor")
    bank = tensor.numpy().copy()
    try:
        _check_bank(bank)
    except ValueError as exc:
        raise refuse(f"{BANK_FILE}: {exc}") from exc
    bank.setflags(write=False)
    return bank


def load_bundle(models_root: Path, model_id: str, *, validation: bool = False) -> Bundle:
    """Load and verify `models_root/<model_id>/`, or refuse with a one-line `ConfigError`.

    A validation-only bundle loads only when `validation` is exactly `True` (ADR-0016).
    """
    refuse = _Refusal(model_id)
    if _MODEL_ID.fullmatch(model_id) is None:
        raise refuse("not a full model_id (patchcore-<product>-v<n>-<12 hex>)")
    directory = models_root / model_id
    if not directory.is_dir():
        raise refuse(f"no such bundle directory under {models_root}")
    try:
        bank_bytes = (directory / BANK_FILE).read_bytes()
    except OSError as exc:
        raise refuse(f"{BANK_FILE} is missing or unreadable") from exc
    try:
        yml_bytes = (directory / BUNDLE_FILE).read_bytes()
    except OSError as exc:
        raise refuse(f"{BUNDLE_FILE} is missing or unreadable") from exc
    suffix = hashlib.sha256(bank_bytes + yml_bytes).hexdigest()[:12]
    if suffix != model_id[-12:]:
        raise refuse(
            f"the files hash to {suffix}, which does not match its model_id — the bundle "
            "was changed after it was written (docs/16 §2)"
        )
    try:
        data = yaml.safe_load(yml_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise refuse(f"{BUNDLE_FILE}: not valid YAML") from exc
    fields = _parse(data, refuse)
    if fields["validation_only"] and validation is not True:
        raise refuse(
            f"validation-only (data_source {fields['data_source']}) — refused outside "
            "validation mode; a model trained on external data never runs the station "
            "(ADR-0016)"
        )
    return Bundle(
        model_id=model_id,
        path=directory,
        bank=_load_bank(bank_bytes, refuse),
        backbone=BACKBONE,
        backbone_sha256=BACKBONE_SHA256,
        **fields,
    )
