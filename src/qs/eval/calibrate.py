"""The calibration record, the `threshold_id`, and the candidate and output configs
(docs/12 §4, docs/15 §7 steps 5–6; FR-54; ticket #81).

The record holds unit IDs and numbers only — no wall-clock time — and the `threshold_id`
is `cal-<local date>-far<N>-<first 6 hex of the SHA-256 of its canonical JSON>` (sorted keys,
no whitespace, UTF-8; `config.canonical_hash`). The output config is the input's parsed
mapping, key order kept, with exactly `inspector.model` and the whole `threshold` block
replaced (#77): a new file, never an overwrite. Pure: everything here is built in memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from qs.config import GeometryConfig, SetOnConfig, StationConfig, ThresholdConfig, canonical_hash
from qs.contracts import CheckResult

CANDIDATE_ID = "cal-candidate"
CANDIDATE_VALUE = 1.0  # a placeholder never written and whose labels are never read


def with_threshold(
    raw: Mapping[str, Any], model: str, threshold: Mapping[str, Any]
) -> dict[str, Any]:
    """The input mapping with `inspector.model` and the whole `threshold` block replaced."""
    out = dict(raw)
    inspector = dict(raw["inspector"])
    inspector["model"] = model
    out["inspector"] = inspector
    out["threshold"] = dict(threshold)
    return out


def threshold_block(
    threshold_id: str,
    value: float,
    *,
    model: str,
    config: StationConfig,
    geometry: GeometryConfig,
    calib_units: int,
    target: int,
) -> dict[str, Any]:
    """The `threshold` block as the config and the loader shape it (docs/12 §4)."""
    return {
        "id": threshold_id,
        "value": value,
        "policy": f"far_{target}pct_on_calib",
        "set_on": {
            "model": model,
            "backend": config.inspector.backend,
            "calib_units": calib_units,
            "localize_threshold": config.inspector.localize.threshold,
            "mask_dilate_px": config.inspector.mask_dilate_px,
        },
        "geometry": {
            "area_px": list(geometry.area_px),
            "axis_ratio_min": geometry.axis_ratio_min,
            "solidity_min": geometry.solidity_min,
        },
    }


def candidate_config(
    config: StationConfig, model: str, geometry: GeometryConfig
) -> StationConfig:
    """The scoring config built in memory: `--model` and the derived tolerances in, a
    placeholder threshold value whose labels are never read."""
    threshold = ThresholdConfig(
        id=CANDIDATE_ID,
        value=CANDIDATE_VALUE,
        policy="far_0pct_on_calib",
        set_on=SetOnConfig(
            model=model,
            backend=config.inspector.backend,
            calib_units=0,
            localize_threshold=config.inspector.localize.threshold,
            mask_dilate_px=config.inspector.mask_dilate_px,
        ),
        geometry=geometry,
    )
    return replace(config, inspector=replace(config.inspector, model=model),
                   threshold=threshold)


def config_text(header: str, mapping: Mapping[str, Any]) -> str:
    """The output config's text: the header comment, then the mapping in key order."""
    return header + yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True)


def threshold_id(record: Mapping[str, Any], target: int) -> str:
    """`cal-<local date>-far<N>-<first 6 hex of the record's canonical JSON>`."""
    suffix = canonical_hash(record)[:6]
    return f"cal-{date.today().isoformat()}-far{target}-{suffix}"


def record_path(report_dir: Path, threshold_id: str) -> Path:
    return report_dir / f"{threshold_id}.yml"


def check_summary(check: CheckResult) -> dict[str, Any]:
    """One geometry `CheckResult` as the record keeps it: passed, `failed_on`, measures."""
    measures = dict(check.measures)
    return {
        "passed": check.passed,
        "failed_on": check.failed_on,
        "area_px": measures["area_px"],
        "axis_ratio": measures["axis_ratio"],
        "solidity": measures["solidity"],
    }


@dataclass(frozen=True, slots=True)
class UnitRecord:
    """One calibration unit as the record keeps it: its score and per-rotation geometry."""

    unit: str
    score: float
    rotations: dict[str, dict[str, Any]]


def build_record(
    *,
    model: str,
    backend: str,
    target: int,
    dataset: Path,
    config_hash: str,
    localize_threshold: int,
    mask_dilate_px: int,
    units: list[UnitRecord],
    n: int,
    a: int,
    g: int,
    flagged: list[str],
    tolerances: Any,
    threshold: float,
) -> dict[str, Any]:
    """The calibration record (docs/15 §7 step 5): unit IDs and numbers only."""
    sigma = tolerances.sigma
    means = tolerances.means
    return {
        "model": model,
        "backend": backend,
        "policy": f"far_{target}pct_on_calib",
        "target": target,
        "dataset": str(dataset),
        "config_hash": config_hash,
        "localize_threshold": localize_threshold,
        "mask_dilate_px": mask_dilate_px,
        "units": {
            entry.unit: {
                "score": entry.score,
                "rotations": dict(sorted(entry.rotations.items())),
            }
            for entry in sorted(units, key=lambda u: u.unit)
        },
        "n": n,
        "a": a,
        "g": g,
        "flagged": sorted(flagged),
        "tolerances": {
            "area_px": list(tolerances.config.area_px),
            "axis_ratio_min": tolerances.config.axis_ratio_min,
            "solidity_min": tolerances.config.solidity_min,
            "derivation": {
                "units": tolerances.units,
                "frames": tolerances.frames,
                "sigma": {name: sigma[name] for name in ("area_px", "axis_ratio",
                                                         "solidity")},
                "mean_range": {
                    name: list(tolerances.mean_range[name])
                    for name in ("area_px", "axis_ratio", "solidity")
                },
                "means": {
                    unit: {name: means[unit][name] for name in ("area_px", "axis_ratio",
                                                                "solidity")}
                    for unit in sorted(means)
                },
            },
        },
        "threshold": threshold,
    }


def margin(check: CheckResult, geometry: GeometryConfig) -> str:
    """How far each failing measure is past its limit, one clause per failing rule (A2)."""
    measures = dict(check.measures)
    clauses: list[str] = []
    for rule in (check.failed_on or "").split(", "):
        limit: float
        if rule == "area_px<min":
            limit = float(geometry.area_px[0])
            clauses.append(f"area_px {measures['area_px']:.0f} is "
                           f"{limit - measures['area_px']:.0f} below the minimum {limit:.0f}")
        elif rule == "area_px>max":
            limit = float(geometry.area_px[1])
            clauses.append(f"area_px {measures['area_px']:.0f} is "
                           f"{measures['area_px'] - limit:.0f} above the maximum {limit:.0f}")
        elif rule == "axis_ratio<min":
            limit = geometry.axis_ratio_min
            clauses.append(f"axis_ratio {measures['axis_ratio']:.3f} is "
                           f"{limit - measures['axis_ratio']:.3f} below the minimum "
                           f"{limit:.3f}")
        elif rule == "solidity<min":
            limit = geometry.solidity_min
            clauses.append(f"solidity {measures['solidity']:.3f} is "
                           f"{limit - measures['solidity']:.3f} below the minimum "
                           f"{limit:.3f}")
    return "; ".join(clauses)
