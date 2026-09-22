import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from qs.errors import ConfigError

_TRIGGER_TYPES = ("keyboard", "replay")
_CAMERA_TYPES = ("webcam", "phone", "replay")
_CAMERA_FORMATS = ("YUYV", "MJPG")
_LIGHTING_GEOMETRIES = ("G1", "G2", "G3")
_INSPECTOR_TYPES = ("appearance", "stub")
_INSPECTOR_BACKENDS = ("torch-cuda", "torch-cpu")
_MODEL_ID = re.compile(r"^patchcore-[a-z0-9]+(-[a-z0-9]+)*-v\d+-[0-9a-f]{12}$")


@dataclass(frozen=True, slots=True)
class TriggerConfig:
    type: str
    key: str


@dataclass(frozen=True, slots=True)
class CameraConfig:
    type: str
    device: str
    width: int
    height: int
    format: str
    power_line_hz: int
    exposure_us: int
    gain: int
    wb_kelvin: int
    focus: int
    capture_timeout_ms: int


@dataclass(frozen=True, slots=True)
class ReferencePatchConfig:
    roi: tuple[int, int, int, int]
    expect: float
    band: float


@dataclass(frozen=True, slots=True)
class LightingConfig:
    geometry: str
    reference_patch: ReferencePatchConfig


@dataclass(frozen=True, slots=True)
class LocalizeConfig:
    roi: tuple[int, int, int, int]
    threshold: int
    min_area_px: int
    border_margin_px: int


@dataclass(frozen=True, slots=True)
class InspectorConfig:
    type: str
    localize: LocalizeConfig
    mask_dilate_px: int
    model: str
    backend: str


@dataclass(frozen=True, slots=True)
class SetOnConfig:
    model: str
    backend: str
    calib_units: int
    localize_threshold: int
    mask_dilate_px: int


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    area_px: tuple[int, int]
    axis_ratio_min: float
    solidity_min: float


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    id: str
    value: float
    policy: str
    set_on: SetOnConfig
    geometry: GeometryConfig


@dataclass(frozen=True, slots=True)
class LabelsConfig:
    pass_: str
    defect: str
    no_verdict: str


@dataclass(frozen=True, slots=True)
class DisplayConfig:
    labels: LabelsConfig
    font: str


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    root: str
    retention_days: int


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    budget_ms: int
    breach_action: str


@dataclass(frozen=True, slots=True)
class StationConfig:
    version: int
    station_id: str
    product: str
    trigger: TriggerConfig
    camera: CameraConfig
    lighting: LightingConfig
    inspector: InspectorConfig
    threshold: ThresholdConfig
    display: DisplayConfig
    evidence: EvidenceConfig
    latency: LatencyConfig


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    config: StationConfig
    config_hash: str  # 64 hex characters
    path: Path


def _kind(value: object) -> str:
    return "null" if value is None else type(value).__name__


def _key(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {_kind(value)}")
    return value


def _check_keys(mapping: Mapping[str, Any], expected: tuple[str, ...], path: str) -> None:
    unknown = sorted(set(mapping) - set(expected), key=str)
    if unknown:
        raise ConfigError("unknown key(s): " + ", ".join(_key(path, str(k)) for k in unknown))
    missing = sorted(set(expected) - set(mapping))
    if missing:
        raise ConfigError("missing key(s): " + ", ".join(_key(path, k) for k in missing))


def _str(mapping: Mapping[str, Any], key: str, parent: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ConfigError(f"{_key(parent, key)}: expected a string, got {_kind(value)}")
    return value


def _int(mapping: Mapping[str, Any], key: str, parent: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{_key(parent, key)}: expected an integer, got {_kind(value)}")
    return value


def _float(mapping: Mapping[str, Any], key: str, parent: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{_key(parent, key)}: expected a number, got {_kind(value)}")
    return float(value)


def _enum(
    mapping: Mapping[str, Any], key: str, parent: str, values: tuple[str, ...]
) -> str:
    value = _str(mapping, key, parent)
    if value not in values:
        raise ConfigError(f"{_key(parent, key)}: {value!r} is not one of: {', '.join(values)}")
    return value


def _roi(mapping: Mapping[str, Any], key: str, parent: str) -> tuple[int, int, int, int]:
    value = mapping[key]
    path = _key(parent, key)
    if not isinstance(value, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) for x in value
    ):
        raise ConfigError(f"{path}: expected four integers [x, y, w, h], got {value!r}")
    if len(value) != 4:
        raise ConfigError(f"{path}: expected four integers [x, y, w, h], got {value!r}")
    return (value[0], value[1], value[2], value[3])


def _pair(mapping: Mapping[str, Any], key: str, parent: str) -> tuple[int, int]:
    value = mapping[key]
    path = _key(parent, key)
    if not isinstance(value, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) for x in value
    ):
        raise ConfigError(f"{path}: expected two integers [min, max], got {value!r}")
    if len(value) != 2:
        raise ConfigError(f"{path}: expected two integers [min, max], got {value!r}")
    return (value[0], value[1])


def _parse_trigger(value: object, path: str) -> TriggerConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("type", "key"), path)
    return TriggerConfig(
        type=_enum(mapping, "type", path, _TRIGGER_TYPES),
        key=_str(mapping, "key", path),
    )


def _parse_camera(value: object, path: str) -> CameraConfig:
    mapping = _mapping(value, path)
    _check_keys(
        mapping,
        (
            "type",
            "device",
            "width",
            "height",
            "format",
            "power_line_hz",
            "exposure_us",
            "gain",
            "wb_kelvin",
            "focus",
            "capture_timeout_ms",
        ),
        path,
    )
    return CameraConfig(
        type=_enum(mapping, "type", path, _CAMERA_TYPES),
        device=_str(mapping, "device", path),
        width=_int(mapping, "width", path),
        height=_int(mapping, "height", path),
        format=_enum(mapping, "format", path, _CAMERA_FORMATS),
        power_line_hz=_int(mapping, "power_line_hz", path),
        exposure_us=_int(mapping, "exposure_us", path),
        gain=_int(mapping, "gain", path),
        wb_kelvin=_int(mapping, "wb_kelvin", path),
        focus=_int(mapping, "focus", path),
        capture_timeout_ms=_int(mapping, "capture_timeout_ms", path),
    )


def _parse_reference_patch(value: object, path: str) -> ReferencePatchConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("roi", "expect", "band"), path)
    return ReferencePatchConfig(
        roi=_roi(mapping, "roi", path),
        expect=_float(mapping, "expect", path),
        band=_float(mapping, "band", path),
    )


def _parse_lighting(value: object, path: str) -> LightingConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("geometry", "reference_patch"), path)
    return LightingConfig(
        geometry=_enum(mapping, "geometry", path, _LIGHTING_GEOMETRIES),
        reference_patch=_parse_reference_patch(
            mapping["reference_patch"], _key(path, "reference_patch")
        ),
    )


def _parse_localize(value: object, path: str) -> LocalizeConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("roi", "threshold", "min_area_px", "border_margin_px"), path)
    return LocalizeConfig(
        roi=_roi(mapping, "roi", path),
        threshold=_int(mapping, "threshold", path),
        min_area_px=_int(mapping, "min_area_px", path),
        border_margin_px=_int(mapping, "border_margin_px", path),
    )


def _parse_inspector(value: object, path: str) -> InspectorConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("type", "localize", "mask_dilate_px", "model", "backend"), path)
    return InspectorConfig(
        type=_enum(mapping, "type", path, _INSPECTOR_TYPES),
        localize=_parse_localize(mapping["localize"], _key(path, "localize")),
        mask_dilate_px=_int(mapping, "mask_dilate_px", path),
        model=_str(mapping, "model", path),
        backend=_enum(mapping, "backend", path, _INSPECTOR_BACKENDS),
    )


def _parse_set_on(value: object, path: str) -> SetOnConfig:
    mapping = _mapping(value, path)
    _check_keys(
        mapping,
        ("model", "backend", "calib_units", "localize_threshold", "mask_dilate_px"),
        path,
    )
    return SetOnConfig(
        model=_str(mapping, "model", path),
        backend=_str(mapping, "backend", path),
        calib_units=_int(mapping, "calib_units", path),
        localize_threshold=_int(mapping, "localize_threshold", path),
        mask_dilate_px=_int(mapping, "mask_dilate_px", path),
    )


def _parse_geometry(value: object, path: str) -> GeometryConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("area_px", "axis_ratio_min", "solidity_min"), path)
    return GeometryConfig(
        area_px=_pair(mapping, "area_px", path),
        axis_ratio_min=_float(mapping, "axis_ratio_min", path),
        solidity_min=_float(mapping, "solidity_min", path),
    )


def _parse_threshold(value: object, path: str) -> ThresholdConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("id", "value", "policy", "set_on", "geometry"), path)
    return ThresholdConfig(
        id=_str(mapping, "id", path),
        value=_float(mapping, "value", path),
        policy=_str(mapping, "policy", path),
        set_on=_parse_set_on(mapping["set_on"], _key(path, "set_on")),
        geometry=_parse_geometry(mapping["geometry"], _key(path, "geometry")),
    )


def _parse_labels(value: object, path: str) -> LabelsConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("pass", "defect", "no_verdict"), path)
    return LabelsConfig(
        pass_=_str(mapping, "pass", path),
        defect=_str(mapping, "defect", path),
        no_verdict=_str(mapping, "no_verdict", path),
    )


def _parse_display(value: object, path: str) -> DisplayConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("labels", "font"), path)
    return DisplayConfig(
        labels=_parse_labels(mapping["labels"], _key(path, "labels")),
        font=_str(mapping, "font", path),
    )


def _parse_evidence(value: object, path: str) -> EvidenceConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("root", "retention_days"), path)
    return EvidenceConfig(
        root=_str(mapping, "root", path),
        retention_days=_int(mapping, "retention_days", path),
    )


def _parse_latency(value: object, path: str) -> LatencyConfig:
    mapping = _mapping(value, path)
    _check_keys(mapping, ("budget_ms", "breach_action"), path)
    return LatencyConfig(
        budget_ms=_int(mapping, "budget_ms", path),
        breach_action=_str(mapping, "breach_action", path),
    )


def _parse_station(mapping: Mapping[str, Any]) -> StationConfig:
    _check_keys(
        mapping,
        (
            "version",
            "station_id",
            "product",
            "trigger",
            "camera",
            "lighting",
            "inspector",
            "threshold",
            "display",
            "evidence",
            "latency",
        ),
        "",
    )
    return StationConfig(
        version=_int(mapping, "version", ""),
        station_id=_str(mapping, "station_id", ""),
        product=_str(mapping, "product", ""),
        trigger=_parse_trigger(mapping["trigger"], "trigger"),
        camera=_parse_camera(mapping["camera"], "camera"),
        lighting=_parse_lighting(mapping["lighting"], "lighting"),
        inspector=_parse_inspector(mapping["inspector"], "inspector"),
        threshold=_parse_threshold(mapping["threshold"], "threshold"),
        display=_parse_display(mapping["display"], "display"),
        evidence=_parse_evidence(mapping["evidence"], "evidence"),
        latency=_parse_latency(mapping["latency"], "latency"),
    )


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _validate(config: StationConfig) -> None:
    if config.camera.power_line_hz == 50 and config.camera.exposure_us % 10000 != 0:
        raise ConfigError(
            f"camera.exposure_us: {config.camera.exposure_us} is not a multiple of 10000 "
            "(10 ms), required with 50 Hz mains to avoid flicker banding"
        )
    patch = config.lighting.reference_patch.roi
    localize = config.inspector.localize.roi
    if _overlaps(patch, localize):
        raise ConfigError(
            "lighting.reference_patch.roi: overlaps inspector.localize.roi — the localiser "
            "would find the grey card as a unit"
        )
    width = config.camera.width
    height = config.camera.height
    for path, (x, y, w, h) in (
        ("lighting.reference_patch.roi", patch),
        ("inspector.localize.roi", localize),
    ):
        if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
            raise ConfigError(
                f"{path}: [{x}, {y}, {w}, {h}] is outside the {width}x{height} frame"
            )
    if _MODEL_ID.fullmatch(config.inspector.model) is None:
        raise ConfigError(
            f"inspector.model: {config.inspector.model!r} is not a full model_id — expected "
            "patchcore-<product>-v<n>-<12 hex> (15-model.md §8)"
        )
    if config.threshold.set_on.model != config.inspector.model:
        raise ConfigError(
            f"threshold.set_on.model: {config.threshold.set_on.model!r} does not match "
            f"inspector.model {config.inspector.model!r}"
        )
    if config.threshold.set_on.backend != config.inspector.backend:
        raise ConfigError(
            f"threshold.set_on.backend: {config.threshold.set_on.backend!r} does not match "
            f"inspector.backend {config.inspector.backend!r}"
        )
    area = config.threshold.geometry.area_px
    if area[0] >= area[1]:
        raise ConfigError(
            f"threshold.geometry.area_px: min {area[0]} is not below max {area[1]}"
        )
    for path, value in (
        ("threshold.geometry.axis_ratio_min", config.threshold.geometry.axis_ratio_min),
        ("threshold.geometry.solidity_min", config.threshold.geometry.solidity_min),
    ):
        if not 0 < value <= 1:
            raise ConfigError(f"{path}: {value} is outside (0, 1]")
    reference_patch = config.lighting.reference_patch
    for path, value in (
        ("lighting.reference_patch.band", reference_patch.band),
        ("lighting.reference_patch.expect", reference_patch.expect),
    ):
        if not 0 < value < 1:
            raise ConfigError(f"{path}: {value} is outside (0, 1)")
    labels = config.display.labels
    for path, text in (
        ("display.labels.pass", labels.pass_),
        ("display.labels.defect", labels.defect),
        ("display.labels.no_verdict", labels.no_verdict),
    ):
        if not text:
            raise ConfigError(f"{path}: must not be empty")


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def canonical_hash(mapping: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical(mapping), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path) -> LoadedConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the config file: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: expected a mapping at the top level, got {_kind(raw)}")
    config = _parse_station(raw)
    _validate(config)
    return LoadedConfig(config=config, config_hash=canonical_hash(raw), path=path)
