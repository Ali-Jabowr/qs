"""`qs evaluate`: the INDICATIVE operating table (FR-55, FR-56; docs/14 §4, §7, §9).

    bin/qs evaluate --config PATH --golden DIR [--models models] [--report-dir reports/eval]

The frozen golden set, the calibrated config and the bundle in; the report and its per-unit
CSV out. Every check runs before the first write: the set against its own manifest
(`verify_golden`), against the model (E1 — no golden unit or frame may be in the bundle)
and against the calibration (`reports/calib/<threshold.id>.yml` — beside the report
directory — names this model and exactly the set's calib units). Every golden frame is then
scored through the station's own Pipeline (`score_frames`, the config as loaded, never
validation mode), the calibration threshold is recomputed and compared to 1e-6, units are
aggregated (E2), and the report is written — `INDICATIVE` in the title, every rate per unit
and carrying its Clopper-Pearson bound; the controls and the drift line (E9, A3/#79), the
paired differences (ADR-0014), every unit whose label changed since the newest earlier CSV
in the report directory with the same golden hash, and latency last — the path of the
newest `reports/latency/*-<model_id>-cpu.md` beside the report directory (`qs latency`,
ADR-0017), or the command that makes it.

Exit codes: 0 written · 1 refused (nothing written), or written with a placement FAIL, the
early warning or the drift line · 2 could not run (config, bundle, CUDA, missing golden
set). Every refusal is one line, never a traceback.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from qs.config import LabelsConfig, StationConfig, load_config
from qs.contracts import Label
from qs.errors import ConfigError
from qs.eval import report as rep
from qs.eval.golden import GoldenRefused, verify_golden
from qs.eval.scoring import FrameResult, score_frames
from qs.eval.threshold import UnitScore, choose_threshold
from qs.eval.units import (
    Aggregated,
    TargetResult,
    aggregate,
    control_passes,
    drift,
    early_warning,
    latency,
    pairs,
    target_result,
)
from qs.inspector.appearance import AppearanceInspector
from qs.inspector.bundle import Bundle, load_bundle
from qs.inspector.model import Scorer
from qs.station import real_cuda_available, real_scorer

_POLICY = re.compile(r"far_(\d+)pct_on_calib")
TARGETS = (1, 5, 10)
DENT_RECALL_FLOOR = 0.70


class _Refused(Exception):
    """The evaluation is refused: exit 1, the cause named, nothing written."""


def _session_yml(golden: Path) -> dict[str, Any]:
    """The golden set's copy of `session.yml`; missing or unreadable is exit 2."""
    path = golden / "session.yml"
    try:
        data: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OSError(f"{path}: {exc}") from None
    if not isinstance(data, dict):
        raise OSError(f"{path}: not a session.yml")
    return data


def _check_set(config: StationConfig, session: dict[str, Any], rows: list[dict[str, str]]
               ) -> None:
    """The set's geometry and camera against the config (ADR-0022); a mismatch refuses."""
    geometries = session.get("geometries")
    geometry = geometries[0] if isinstance(geometries, list) and geometries and all(
        isinstance(g, str) for g in geometries) else None
    if not isinstance(geometry, str):
        raise _Refused("session.yml: geometries is a list of G1, G2, G3")
    if config.lighting.geometry != geometry:
        raise _Refused(
            f"lighting.geometry {config.lighting.geometry} but the golden set was shot "
            f"under {geometry} — one model per geometry (docs/15 §7 step 1)"
        )
    camera = session.get("camera")
    device = camera.get("device") if isinstance(camera, dict) else None
    if not isinstance(device, str):
        raise _Refused("session.yml: camera.device is missing")
    phone_shot = device.startswith("phone:")
    phone_config = config.camera.type == "phone"
    if phone_shot != phone_config:
        raise _Refused(
            f"the golden set was shot by {'the phone' if phone_shot else 'the webcam'} "
            f"({device}) but camera.type is {config.camera.type} — what is learned from "
            "phone frames belongs to the phone (ADR-0022)"
        )
    row_geometry = {row["geometry"] for row in rows}
    if row_geometry != {geometry}:
        raise _Refused(
            f"the manifest names geometry {', '.join(sorted(row_geometry))} but "
            f"session.yml says {geometry} — the set disagrees with itself"
        )


def _check_bundle(bundle: Bundle, rows: list[dict[str, str]]) -> None:
    """E1 where the number is made: no golden unit or frame may be in the bundle."""
    units = {row["unit"] for row in rows}
    shared_units = sorted(units & set(bundle.units))
    if shared_units:
        raise _Refused(
            f"the bundle was trained on {', '.join(shared_units)}, a golden unit — E1 is "
            "asserted where the number is made (docs/14 §8)"
        )
    hashes = {row["sha256"] for row in rows}
    shared_images = sorted(image.file for image in bundle.images
                           if image.sha256 in hashes)
    if shared_images:
        raise _Refused(
            f"the bundle was trained on {', '.join(shared_images)}, a golden frame's "
            "bytes — E1 is asserted where the number is made (docs/14 §8)"
        )


def _calibration_record(
    config: StationConfig, rows: list[dict[str, str]], report_dir: Path
) -> dict[str, Any]:
    """The record `qs calibrate` wrote: it must name this model and exactly the set's
    calib units, and `set_on.calib_units` their number (docs/14 §4 step 2)."""
    calib_units = sorted({row["unit"] for row in rows if row["role"] == "calib"})
    path = report_dir.parent / "calib" / f"{config.threshold.id}.yml"
    if not path.is_file():
        raise _Refused(
            f"no calibration record at {path} — a threshold is never used without its "
            "record (docs/12 §4)"
        )
    try:
        record: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise _Refused(f"{path}: {exc}") from None
    if not isinstance(record, dict):
        raise _Refused(f"{path}: not a calibration record")
    if record.get("model") != config.inspector.model:
        raise _Refused(
            f"{path}: the record names model {record.get('model')!r} but the config "
            f"names {config.inspector.model!r} — the threshold was not set on this model"
        )
    units = record.get("units")
    if not isinstance(units, dict) or sorted(units) != calib_units:
        raise _Refused(
            f"{path}: the record's calibration units are not exactly the golden set's "
            f"({', '.join(calib_units)}) — a threshold is never mixed across sets "
            "(docs/14 §8)"
        )
    if config.threshold.set_on.calib_units != len(calib_units):
        raise _Refused(
            f"threshold.set_on.calib_units is {config.threshold.set_on.calib_units} but "
            f"the golden set has {len(calib_units)} calib units — the threshold was not "
            "set on this set"
        )
    return record


def _check_calibration(
    units: tuple[Aggregated, ...], config: StationConfig,
    results: tuple[FrameResult, ...], rows: list[dict[str, str]],
) -> None:
    """Every calib frame has a verdict, and the threshold reproduces to 1e-6."""
    for row, result in zip(rows, results, strict=True):
        if row["role"] == "calib" and result.label is Label.NO_VERDICT:
            raise _Refused(
                f"{row['file']}: НЕТ РЕШЕНИЯ ({result.reason}) — a threshold is never set "
                "on fewer units than it claims (docs/15 §7 step 3)"
            )
    match = _POLICY.fullmatch(config.threshold.policy)
    if match is None:
        raise _Refused(
            f"threshold.policy {config.threshold.policy!r} is not far_<N>pct_on_calib — "
            "the threshold cannot be recomputed on this set"
        )
    calib = [u for u in units if u.role == "calib"]
    choice = choose_threshold(
        [UnitScore(u.unit, u.score or 0.0, bool(u.geometry_failed)) for u in calib],
        int(match.group(1)),
    )
    recomputed = "not reachable" if choice.value is None else f"{choice.value:.6f}"
    if choice.value is None or abs(choice.value - config.threshold.value) > 1e-6:
        raise _Refused(
            f"the threshold recomputed on this set's calib units is {recomputed} but the "
            f"config says {config.threshold.value:.6f} — the threshold was not set on "
            "this set, or the scores changed (docs/14 §4)"
        )


def _placement(
    results: tuple[FrameResult, ...], rows: list[dict[str, str]]
) -> tuple[bool, tuple[tuple[str, str], ...]]:
    """Every placement frame НЕТ РЕШЕНИЯ(`placement`) → PASS, else each frame and what
    it got (docs/14 §9)."""
    bad: list[tuple[str, str]] = []
    for row, result in zip(rows, results, strict=True):
        if row["role"] != "placement":
            continue
        if result.label is Label.NO_VERDICT:
            if result.reason == "placement":
                continue
            got = result.reason or ""
        else:
            got = result.label.value
        bad.append((row["file"], got))
    return (not bad, tuple(bad))


def _report_path(report_dir: Path, stem: str) -> Path:
    """`<stem>.md`, or the next free `-2`, `-3` — a report is never overwritten."""
    path = report_dir / f"{stem}.md"
    n = 1
    while path.exists():
        n += 1
        path = report_dir / f"{stem}-{n}.md"
    return path


_REPORT_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})-(.+?)(?:-(\d+))?\.csv")


def _previous_run(report_dir: Path, golden_hash: str) -> tuple[str, str] | None:
    """The newest earlier per-unit CSV in `report_dir` on this golden set: (name, text).

    Newest by the file name — its date, then its `-2`, `-3` — never by a clock. A CSV that
    is not a report's (another header) or carries another golden hash is ignored."""
    found: list[tuple[tuple[str, int, str], str]] = []
    for path in sorted(report_dir.glob("*.csv")) if report_dir.is_dir() else []:
        match = _REPORT_NAME.fullmatch(path.name)
        if match is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) < 2 or tuple(rows[0]) != rep.CSV_COLUMNS:
            continue
        if {row[0] for row in rows[1:] if row} != {golden_hash}:
            continue
        found.append(((match.group(1), int(match.group(3) or 1), path.name), text))
    if not found:
        return None
    (_, _, name), text = max(found)
    return name, text


def _latency_report(report_dir: Path, model_id: str) -> str | None:
    """The newest `qs latency --cpu` report of this model beside the report directory."""
    directory = report_dir.parent / "latency"
    reports = sorted(directory.glob(f"*-{model_id}-cpu.md")) if directory.is_dir() else []
    return str(reports[-1]) if reports else None


def _invocation(args: argparse.Namespace) -> str:
    return (f"qs evaluate --config {args.config} --golden {args.golden} "
            f"--models {args.models} --report-dir {args.report_dir}")


def _summary(out: Path, csv_path: Path, targets: tuple[TargetResult, ...],
             placement_ok: bool, warning: str | None, drift_line: str | None) -> None:
    at5 = next(t for t in targets if t.target == 5)
    recall = " · ".join(f"{r.cls} {r.caught}/{r.total}" for r in at5.recall)
    far = at5.far_combined
    print(f"report: {out}")
    print(f"csv: {csv_path}")
    print(f"far 5%: threshold {at5.choice.value:.6f} · recall {recall} · "
          f"false alarms {far.flagged}/{far.total} (upper bound {far.upper * 100:.1f}%)")
    print(f"placement: {'PASS' if placement_ok else 'FAIL'}")
    if drift_line is not None:
        print(drift_line)
    if warning is not None:
        print(warning)


def main(
    args: argparse.Namespace,
    *,
    make_scorer: Callable[[str, Any], Scorer] = real_scorer,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> int:
    """Evaluate; 0 written, 1 refused or written with a stop line, 2 could not run."""
    try:
        return _evaluate(args, make_scorer, cuda_available)
    except _Refused as exc:
        print(f"evaluate: {exc}")
        return 1


def _evaluate(
    args: argparse.Namespace,
    make_scorer: Callable[[str, Any], Scorer],
    cuda_available: Callable[[], bool],
) -> int:
    try:
        loaded = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    config = loaded.config
    golden = Path(args.golden)
    if not golden.is_dir():
        print(f"golden: {golden} is not a directory")
        return 2
    try:
        rows = verify_golden(golden)
    except GoldenRefused as exc:
        raise _Refused(
            f"{exc} — a report is never built on an unfrozen set (FR-56, docs/14 §8)"
        ) from None
    try:
        session = _session_yml(golden)
    except OSError as exc:
        print(f"golden: {exc}")
        return 2
    _check_set(config, session, rows)
    try:
        bundle = load_bundle(Path(args.models), config.inspector.model)
    except ConfigError as exc:
        print(f"bundle: {exc}")
        return 2
    _check_bundle(bundle, rows)
    report_dir = Path(args.report_dir)
    _calibration_record(config, rows, report_dir)
    inspector = AppearanceInspector(
        loaded.config_hash, models_root=Path(args.models),
        make_scorer=make_scorer, cuda_available=cuda_available,
    )
    try:
        inspector.load(config)
    except ConfigError as exc:
        print(f"inspector: {exc}")
        return 2
    files = [golden / "frames" / row["file"] for row in rows]
    results = score_frames(config, loaded.config_hash, files, inspector=inspector,
                           name="evaluate")
    units = aggregate(rows, results)
    _check_calibration(units, config, results, rows)
    targets = tuple(target_result(target, units) for target in TARGETS)
    placement_ok, placement_bad = _placement(results, rows)
    warning = early_warning(targets)
    controls = control_passes(rows, results)
    paired = pairs(units)
    drifted = drift(controls, paired)
    labels = LabelsConfig(config.display.labels.pass_, config.display.labels.defect,
                          config.display.labels.no_verdict)
    built = rep.Report(
        product=config.product,
        geometry=config.lighting.geometry,
        golden_name=golden.name,
        golden_hash=(golden / "manifest.sha256").read_text(encoding="utf-8").strip(),
        model_id=config.inspector.model,
        threshold_id=config.threshold.id,
        config_hash=loaded.config_hash,
        invocation=_invocation(args),
        camera_device=str(session.get("camera", {}).get("device", "")),
        camera_is_phone=config.camera.type == "phone",
        counts=rep.counts(rows),
        targets=targets,
        units=units,
        placement_ok=placement_ok,
        placement_bad=placement_bad,
        early_warning=warning,
        labels=labels,
        controls=controls,
        drift=drifted,
        pairs=paired,
        latency=latency(results),
        backend=config.inspector.backend,
        latency_report=_latency_report(report_dir, config.inspector.model),
        latency_command=(f"bin/qs latency --config {args.config} "
                         f"--frames {golden / 'frames'} --cpu"),
    )
    csv_text = rep.csv_text(built)
    previous = _previous_run(report_dir, built.golden_hash)
    built = replace(built, changes=rep.changed_units(
        previous[0] if previous else None, previous[1] if previous else "", csv_text))
    stem = f"{date.today().isoformat()}-{config.product}-{config.lighting.geometry}"
    out = _report_path(report_dir, stem)
    report_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(rep.markdown(built), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    csv_path.write_text(csv_text, encoding="utf-8")
    drift_line = drifted.line()
    _summary(out, csv_path, targets, placement_ok, warning, drift_line)
    return 0 if placement_ok and warning is None and drift_line is None else 1
