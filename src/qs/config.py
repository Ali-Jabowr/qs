"""The station config: a strict loader over `yaml.safe_load`, and its canonical hash.

Nothing here has a default — a missing key is a `ConfigError`, never a silent value (C6),
except `inspector.checks`, which may be left out, `inspector.line`, which only a line inspector
has, its `direction` and `fps`, which only a cup trigger needs, and `display.pass_expiry_s`,
`display.ready_line` and `display.placement_line`, which only a line screen sets (#104; left
out on a line station, the line's own wording fills them), and `lighting.own_dc_light`, which
defaults to false (true only on a line station, ADR-0030) — and every rule of CONTRACTS §3.3
refuses the config naming the offending key, so a bad config fails at start-up instead of
quietly ruining a shoot.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from qs.contracts import CHECK_APPEARANCE, CHECK_EMPTY_CUP, CHECK_GEOMETRY
from qs.errors import ConfigError

_TRIGGER_TYPES = ("keyboard", "replay", "cup")
TRAVEL_DIRECTIONS = ("right", "left", "down", "up")  # the way the chain moves in the picture
_CAMERA_TYPES = ("webcam", "phone", "replay")
_CAMERA_FORMATS = ("YUYV", "MJPG")
_LIGHTING_GEOMETRIES = ("G1", "G2", "G3")
_INSPECTOR_TYPES = ("appearance", "line", "stub")
_INSPECTOR_BACKENDS = ("torch-cuda", "torch-cpu")
KNOWN_CHECKS = (CHECK_GEOMETRY, CHECK_APPEARANCE, CHECK_EMPTY_CUP)
# What a config without `inspector.checks` runs. Fixed on purpose: a check added to
# KNOWN_CHECKS later must not start running on stations that never asked for it.
_BENCH_CHECKS = frozenset({CHECK_GEOMETRY, CHECK_APPEARANCE})
# The check lists a line inspector runs. Any other listed check would never run, and the
# Pipeline's guard would refuse every verdict. `empty_cup` is in every one: an empty cup is
# never judged only by a model trained on full cups.
_LINE_CHECK_SETS = (
    frozenset({CHECK_EMPTY_CUP}),
    frozenset({CHECK_EMPTY_CUP, CHECK_APPEARANCE}),
)
_LINE_READY = "ГОТОВ"  # a line station's READY line when display.ready_line is left out
_LINE_PLACEMENT = "стаканчик не проверен"  # and its placement line
_RIM_AXIS_MIN_PX = 16  # below this, the inside of a found rim has too few pixels to measure
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
    reference_patch: ReferencePatchConfig | None  # None: no lighting check; a line station only
    own_dc_light: bool = False  # own PWM-free DC light, line only (ADR-0030)


@dataclass(frozen=True, slots=True)
class LocalizeConfig:
    roi: tuple[int, int, int, int]
    threshold: int
    min_area_px: int
    border_margin_px: int


@dataclass(frozen=True, slots=True)
class LineConfig:
    zone: tuple[int, int, int, int]
    rim_axis_px: tuple[int, int]
    blob_share_min: float
    direction: str | None = None
    fps: float | None = None
    # The median-warmth verdict over a burst (ADR-0029 §4); a station without `warmth_min` keeps
    # `blob_share`. `max_speed_px_s` is the fastest the chain moves, for the zone-length rule.
    warmth_min: float | None = None
    burst: int | None = None
    burst_min_frames: int = 3
    warmth_spread_max: float | None = None
    max_speed_px_s: float | None = None
    # The strict finder (ADR-0029 §2-3). `rim_size_px`: the rim's (minor, major) axes measured on
    # the mount, each accepted within ±5% of its own value. `still_ring_px`: a ring moving less
    # than this per frame while the chain moves is a reflection. `lens_mask_px`: (x, y, radius)
    # left out of every warmth measure.
    rim_size_px: tuple[int, int] | None = None
    still_ring_px: float | None = None
    lens_mask_px: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class InspectorConfig:
    type: str
    localize: LocalizeConfig
    mask_dilate_px: int
    model: str
    backend: str
    checks: frozenset[str] = _BENCH_CHECKS
    line: LineConfig | None = None


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
    pass_expiry_s: float | None = None  # None: ГОДЕН stays until the next verdict (the bench)
    ready_line: str | None = None  # None: the bench's "place the item and press SPACE"
    placement_line: str | None = None  # None: the bench's "place the item in the frame"


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


def _check_keys(
    mapping: Mapping[str, Any],
    expected: tuple[str, ...],
    path: str,
    optional: tuple[str, ...] = (),
) -> None:
    unknown = sorted(set(mapping) - set(expected) - set(optional), key=str)
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


def _bool(mapping: Mapping[str, Any], key: str, parent: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{_key(parent, key)}: expected a boolean, got {_kind(value)}")
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


def _circle(mapping: Mapping[str, Any], key: str, parent: str) -> tuple[int, int, int]:
    value = mapping[key]
    path = _key(parent, key)
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    ):
        raise ConfigError(f"{path}: expected three integers [x, y, radius], got {value!r}")
    return (value[0], value[1], value[2])


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
    _check_keys(mapping, ("geometry", "reference_patch"), path, optional=("own_dc_light",))
    return LightingConfig(
        geometry=_enum(mapping, "geometry", path, _LIGHTING_GEOMETRIES),
        reference_patch=(
            None
            if mapping["reference_patch"] is None
            else _parse_reference_patch(mapping["reference_patch"], _key(path, "reference_patch"))
        ),
        own_dc_light=_bool(mapping, "own_dc_light", path) if "own_dc_light" in mapping else False,
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


def _refuse_malformed_checks(mapping: Mapping[str, Any], parent: str) -> None:
    if "checks" not in mapping:
        return
    names = mapping["checks"]
    path = _key(parent, "checks")
    if not isinstance(names, list):
        raise ConfigError(f"{path}: expected a list, got {_kind(names)}")
    if not names:
        raise ConfigError(f"{path}: must list at least one check")
    for name in names:
        if not isinstance(name, str):
            raise ConfigError(f"{path}: expected a string, got {_kind(name)}")
        if name not in KNOWN_CHECKS:
            raise ConfigError(f"{path}: {name!r} is not one of: {', '.join(KNOWN_CHECKS)}")
        if names.count(name) > 1:
            raise ConfigError(f"{path}: {name!r} is repeated")


_STRICT_KEYS = ("rim_size_px", "still_ring_px", "lens_mask_px")
_WARMTH_KEYS = ("warmth_min", "burst", "burst_min_frames", "warmth_spread_max", "max_speed_px_s")


def _parse_warmth(mapping: Mapping[str, Any], path: str) -> dict[str, Any]:
    """The warmth keys present, parsed: all or none of them, with `burst_min_frames` optional."""
    if "warmth_min" not in mapping:
        extra = [key for key in _WARMTH_KEYS if key in mapping]
        if extra:
            raise ConfigError(f"{_key(path, extra[0])}: needs {_key(path, 'warmth_min')} to be set")
        return {}
    for key in ("burst", "warmth_spread_max", "max_speed_px_s", "direction", "fps"):
        if key not in mapping:
            raise ConfigError(f"{_key(path, key)}: a station with warmth_min needs this key")
    parsed: dict[str, Any] = {
        "warmth_min": _float(mapping, "warmth_min", path),
        "burst": _int(mapping, "burst", path),
        "warmth_spread_max": _float(mapping, "warmth_spread_max", path),
        "max_speed_px_s": _float(mapping, "max_speed_px_s", path),
    }
    if "burst_min_frames" in mapping:
        parsed["burst_min_frames"] = _int(mapping, "burst_min_frames", path)
    return parsed


def _parse_line(value: object, path: str) -> LineConfig:
    mapping = _mapping(value, path)
    _check_keys(
        mapping,
        ("zone", "rim_axis_px", "blob_share_min"),
        path,
        optional=("direction", "fps", *_WARMTH_KEYS, *_STRICT_KEYS),
    )
    warmth = _parse_warmth(mapping, path)
    return LineConfig(
        **warmth,
        zone=_roi(mapping, "zone", path),
        rim_axis_px=_pair(mapping, "rim_axis_px", path),
        blob_share_min=_float(mapping, "blob_share_min", path),
        direction=(
            _enum(mapping, "direction", path, TRAVEL_DIRECTIONS) if "direction" in mapping else None
        ),
        fps=_float(mapping, "fps", path) if "fps" in mapping else None,
        rim_size_px=_pair(mapping, "rim_size_px", path) if "rim_size_px" in mapping else None,
        still_ring_px=(
            _float(mapping, "still_ring_px", path) if "still_ring_px" in mapping else None
        ),
        lens_mask_px=_circle(mapping, "lens_mask_px", path) if "lens_mask_px" in mapping else None,
    )


def _parse_inspector(value: object, path: str) -> InspectorConfig:
    mapping = _mapping(value, path)
    _check_keys(
        mapping,
        ("type", "localize", "mask_dilate_px", "model", "backend"),
        path,
        optional=("checks", "line"),
    )
    _refuse_malformed_checks(mapping, path)
    checks = frozenset(mapping["checks"]) if "checks" in mapping else _BENCH_CHECKS
    inspector_type = _enum(mapping, "type", path, _INSPECTOR_TYPES)
    if inspector_type == "appearance" and CHECK_APPEARANCE not in checks:
        # Such a station would load the model and never score with it.
        raise ConfigError(
            f"{_key(path, 'checks')}: an appearance inspector must list {CHECK_APPEARANCE!r}"
        )
    if inspector_type == "line" and checks not in _LINE_CHECK_SETS:
        allowed = " or ".join(str(sorted(listed)) for listed in _LINE_CHECK_SETS)
        raise ConfigError(f"{_key(path, 'checks')}: a line inspector must list {allowed}")
    if inspector_type != "line" and CHECK_EMPTY_CUP in checks:
        raise ConfigError(
            f"{_key(path, 'checks')}: {CHECK_EMPTY_CUP!r} runs only on a line inspector"
        )
    if inspector_type == "line" and "line" not in mapping:
        raise ConfigError(f"{_key(path, 'line')}: a line inspector needs this block")
    if inspector_type != "line" and "line" in mapping:
        raise ConfigError(f"{_key(path, 'line')}: only a line inspector takes this block")
    return InspectorConfig(
        type=inspector_type,
        localize=_parse_localize(mapping["localize"], _key(path, "localize")),
        mask_dilate_px=_int(mapping, "mask_dilate_px", path),
        model=_str(mapping, "model", path),
        backend=_enum(mapping, "backend", path, _INSPECTOR_BACKENDS),
        checks=checks,
        line=_parse_line(mapping["line"], _key(path, "line")) if "line" in mapping else None,
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
    _check_keys(
        mapping, ("labels", "font"), path,
        optional=("pass_expiry_s", "ready_line", "placement_line"),
    )
    return DisplayConfig(
        labels=_parse_labels(mapping["labels"], _key(path, "labels")),
        font=_str(mapping, "font", path),
        pass_expiry_s=(
            _float(mapping, "pass_expiry_s", path) if "pass_expiry_s" in mapping else None
        ),
        ready_line=_str(mapping, "ready_line", path) if "ready_line" in mapping else None,
        placement_line=(
            _str(mapping, "placement_line", path) if "placement_line" in mapping else None
        ),
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
    config = StationConfig(
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
    return dataclasses.replace(config, display=_line_wording(config.display, config.inspector))


def _line_wording(display: DisplayConfig, inspector: InspectorConfig) -> DisplayConfig:
    """A line station's display, with the line's wording wherever the file leaves it out, so a
    line never shows the bench's "place the item and press SPACE"."""
    if inspector.type != "line":
        return display
    return dataclasses.replace(
        display,
        ready_line=display.ready_line if display.ready_line is not None else _LINE_READY,
        placement_line=(
            display.placement_line if display.placement_line is not None else _LINE_PLACEMENT
        ),
    )


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _check_in_frame(path: str, roi: tuple[int, int, int, int], config: StationConfig) -> None:
    x, y, w, h = roi
    width, height = config.camera.width, config.camera.height
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
        raise ConfigError(f"{path}: [{x}, {y}, {w}, {h}] is outside the {width}x{height} frame")


def _validate_line(line: LineConfig, config: StationConfig) -> None:
    _check_in_frame("inspector.line.zone", line.zone, config)
    patch = config.lighting.reference_patch
    if patch is not None and _overlaps(patch.roi, line.zone):
        raise ConfigError(
            "lighting.reference_patch.roi: overlaps inspector.line.zone — the grey card "
            "would be read as part of the view of a cup"
        )
    low, high = line.rim_axis_px
    if low >= high:
        raise ConfigError(f"inspector.line.rim_axis_px: min {low} is not below max {high}")
    if low < _RIM_AXIS_MIN_PX:
        raise ConfigError(f"inspector.line.rim_axis_px: min {low} is below {_RIM_AXIS_MIN_PX}")
    if not 0 < line.blob_share_min < 1:
        raise ConfigError(f"inspector.line.blob_share_min: {line.blob_share_min} is outside (0, 1)")
    if line.fps is not None and (not math.isfinite(line.fps) or line.fps <= 0):
        raise ConfigError(f"inspector.line.fps: {line.fps} is not a finite number above 0")
    _validate_strict(line, config)
    if line.warmth_min is not None:
        if config.camera.type != "replay":
            raise ConfigError(
                "inspector.line.warmth_min: needs camera.type replay: multi-frame capture is "
                "only available from recorded video for now"
            )
        _validate_warmth(line)


def _validate_strict(line: LineConfig, config: StationConfig) -> None:
    """The strict-finder keys: each a finite number that can work with the rest of the block."""
    low, high = line.rim_axis_px
    size = line.rim_size_px
    if size is not None and not low <= size[0] <= size[1] <= high:
        raise ConfigError(
            f"inspector.line.rim_size_px: [{size[0]}, {size[1]}] is not [minor, major] within "
            f"rim_axis_px {low}-{high}"
        )
    still = line.still_ring_px
    if still is not None:
        if not math.isfinite(still) or still <= 0:
            raise ConfigError(f"inspector.line.still_ring_px: {still} is not above 0")
        if config.trigger.type != "cup":
            raise ConfigError("inspector.line.still_ring_px: needs trigger.type cup")
    if line.lens_mask_px is not None:
        x, y, radius = line.lens_mask_px
        if radius < 1:
            raise ConfigError(f"inspector.line.lens_mask_px: radius {radius} is below 1")
        _check_in_frame("inspector.line.lens_mask_px", (x - radius, y - radius, 2 * radius,
                                                         2 * radius), config)
        if line.warmth_min is None:
            raise ConfigError(
                "inspector.line.lens_mask_px: needs warmth_min: only the warmth measure uses it"
            )


def _validate_warmth(line: LineConfig) -> None:
    """A warmth station's numbers, and the zone rule of docs/11 "Bursts": the zone is at least one
    rim plus `burst` frame periods of travel long (one to the first frame, burst - 1 between)."""
    burst, spread, speed, fps = line.burst, line.warmth_spread_max, line.max_speed_px_s, line.fps
    if burst is None or spread is None or speed is None or fps is None or line.direction is None:
        raise ConfigError("inspector.line: the warmth keys are incomplete")  # the parser ensures
    if burst < 1:
        raise ConfigError(f"inspector.line.burst: {burst} is below 1")
    if not 1 <= line.burst_min_frames <= burst:
        raise ConfigError(
            f"inspector.line.burst_min_frames: {line.burst_min_frames} is not within 1 and "
            f"burst ({burst})"
        )
    if not math.isfinite(spread) or spread <= 0:
        raise ConfigError(f"inspector.line.warmth_spread_max: {spread} is not above 0")
    if not math.isfinite(speed) or speed <= 0:
        raise ConfigError(f"inspector.line.max_speed_px_s: {speed} is not above 0")
    needed = line.rim_axis_px[1] + burst * speed / fps
    zone_w, zone_h = line.zone[2], line.zone[3]
    available = zone_w if line.direction in ("left", "right") else zone_h
    if available < needed:
        raise ConfigError(
            f"inspector.line.zone: {available} px along the travel is shorter than a rim "
            f"({line.rim_axis_px[1]} px) plus {burst} frames of travel ({needed:.0f} px in all)"
        )


def _validate_cup_trigger(config: StationConfig) -> None:
    line = config.inspector.line
    if line is None:
        raise ConfigError("trigger.type: cup needs inspector.line, with its inspection zone")
    if line.direction is None:
        raise ConfigError("inspector.line.direction: a cup trigger needs this key")
    if line.fps is None:
        raise ConfigError("inspector.line.fps: a cup trigger needs this key")


def _validate(config: StationConfig) -> None:
    if config.inspector.line is not None:
        _validate_line(config.inspector.line, config)
    if config.trigger.type == "cup":
        _validate_cup_trigger(config)
    if config.camera.exposure_us <= 0:
        raise ConfigError(f"camera.exposure_us: {config.camera.exposure_us} is not above 0")
    if config.lighting.own_dc_light and config.inspector.line is None:
        raise ConfigError(
            "lighting.own_dc_light: only a line station may declare its own DC light — "
            "the bench keeps the flicker rule"
        )
    if (
        config.camera.power_line_hz == 50
        and config.camera.exposure_us % 10000 != 0
        and not config.lighting.own_dc_light
    ):
        raise ConfigError(
            f"camera.exposure_us: {config.camera.exposure_us} is not a multiple of 10000 "
            "(10 ms), required with 50 Hz mains to avoid flicker banding"
        )
    reference_patch = config.lighting.reference_patch
    if reference_patch is None and config.inspector.line is None:
        raise ConfigError(
            "lighting.reference_patch: only a line station may run without it — the bench "
            "judges lighting on every frame (FR-14)"
        )
    localize = config.inspector.localize.roi
    rois = [("inspector.localize.roi", localize)]
    if reference_patch is not None:
        if _overlaps(reference_patch.roi, localize):
            raise ConfigError(
                "lighting.reference_patch.roi: overlaps inspector.localize.roi — the localiser "
                "would find the grey card as a unit"
            )
        rois.insert(0, ("lighting.reference_patch.roi", reference_patch.roi))
    for path, roi in rois:
        _check_in_frame(path, roi, config)
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
    if config.threshold.set_on.localize_threshold != config.inspector.localize.threshold:
        raise ConfigError(
            f"threshold.set_on.localize_threshold: "
            f"{config.threshold.set_on.localize_threshold} does not match "
            f"inspector.localize.threshold {config.inspector.localize.threshold}"
        )
    if config.threshold.set_on.mask_dilate_px != config.inspector.mask_dilate_px:
        raise ConfigError(
            f"threshold.set_on.mask_dilate_px: {config.threshold.set_on.mask_dilate_px} "
            f"does not match inspector.mask_dilate_px {config.inspector.mask_dilate_px}"
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
    if reference_patch is not None:
        for path, value in (
            ("lighting.reference_patch.band", reference_patch.band),
            ("lighting.reference_patch.expect", reference_patch.expect),
        ):
            if not 0 < value < 1:
                raise ConfigError(f"{path}: {value} is outside (0, 1)")
    display = config.display
    labels = display.labels
    for path, text in (
        ("display.labels.pass", labels.pass_),
        ("display.labels.defect", labels.defect),
        ("display.labels.no_verdict", labels.no_verdict),
        ("display.ready_line", display.ready_line),
        ("display.placement_line", display.placement_line),
    ):
        if text is not None and not text.strip():
            raise ConfigError(f"{path}: must not be empty or whitespace-only")
    expiry = display.pass_expiry_s
    if expiry is not None and not (math.isfinite(expiry) and expiry > 0):
        raise ConfigError(f"display.pass_expiry_s: {expiry} is not a finite number above 0")
    threshold = config.threshold.value
    if not math.isfinite(threshold) or threshold <= 0:
        raise ConfigError(
            f"threshold.value: {threshold} is not a finite number above 0 — a NaN or "
            "infinite threshold would pass every appearance check (fails open)"
        )


def _refuse_duplicate_keys(node: yaml.Node, path: str) -> None:
    if not isinstance(node, yaml.MappingNode):
        return
    seen: set[str] = set()
    for key_node, value_node in node.value:
        if not isinstance(key_node, yaml.ScalarNode):
            _refuse_duplicate_keys(value_node, path)
            continue
        key = str(key_node.value)
        if key in seen:
            raise ConfigError(
                f"{_key(path, key)}: this key is repeated within one mapping — "
                "YAML would silently keep only the last value"
            )
        seen.add(key)
        _refuse_duplicate_keys(value_node, _key(path, key))


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
        document = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if document is not None:
        _refuse_duplicate_keys(document, "")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: expected a mapping at the top level, got {_kind(raw)}")
    config = _parse_station(raw)
    _validate(config)
    return LoadedConfig(config=config, config_hash=canonical_hash(raw), path=path)
