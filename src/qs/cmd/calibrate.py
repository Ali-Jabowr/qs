"""`qs calibrate`: derived tolerances and the threshold, under one `threshold_id` (FR-54).

    bin/qs calibrate --config PATH --dataset DIR --model MODEL_ID --out PATH
                     [--far 5] [--models models] [--report-dir reports/calib]

Steps 1–7 of docs/15 §7: the refusals before any scoring (#77, E1, ADR-0022), the tolerances
derived from `train/good` ∪ `calib/good` (docs/15 §3), the calibration scores through the
station's own Pipeline (`qs.eval.scoring.score_frames`), the threshold by `choose_threshold`
(A2, #78), then the record `reports/calib/<threshold_id>.yml` and the new config — written to
a temporary file, checked with `load_config`, renamed to `--out`; the record after it. The
input config is only read; `--out` must not exist.

Exit codes: 0 done · 1 refused (or geometry alone past the target, A2) · 2 could not run
(config, bundle, backend, missing input). Every refusal is one line — the not-reachable
refusal lists each flagged unit, rotation, rule and margin — never a traceback.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import yaml

from qs.config import GeometryConfig, StationConfig, canonical_hash, load_config
from qs.contracts import CheckResult, Label
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.eval import calibrate as cal
from qs.eval.scoring import FrameResult, score_frames
from qs.eval.threshold import ThresholdChoice, UnitScore, choose_threshold
from qs.inspector.appearance import AppearanceInspector
from qs.inspector.bundle import load_bundle
from qs.inspector.geometry import GeometryMeasures, measure
from qs.inspector.localize import localize
from qs.inspector.model import Scorer
from qs.inspector.tolerances import Tolerances, derive_tolerances
from qs.names import FrameNameError, parse_name
from qs.station import real_cuda_available, real_scorer


class _Refused(Exception):
    """The calibration is refused: exit 1, the cause named, nothing written."""


Far = int


def _dataset_units(directory: Path) -> set[str]:
    """The units of one dataset folder, from the frame names; a non-parsing name refuses."""
    units: set[str] = set()
    for path in sorted(directory.glob("*.png")):
        try:
            units.add(parse_name(path.name).unit)
        except FrameNameError as exc:
            raise _Refused(f"{path.name}: {exc}") from None
    if not units:
        raise _Refused(f"{directory}: no .png frames")
    return units


def _session_yml(dataset: Path) -> dict[str, object]:
    """The dataset's `session.yml` — a copy `qs build` wrote; missing or unreadable is exit 2."""
    path = dataset / "session.yml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"{path}: {exc}") from None
    try:
        data: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OSError(f"{path}: {exc}") from None
    if not isinstance(data, dict):
        raise OSError(f"{path}: not a session.yml")
    return data


def _check_session(
    config: StationConfig, dataset: Path, session: dict[str, object]
) -> None:
    """The dataset's geometry and camera against the config; the geometry, when they agree."""
    geometries = session.get("geometries")
    if not isinstance(geometries, list) or not geometries or not all(
        isinstance(g, str) for g in geometries
    ):
        raise _Refused(f"{dataset / 'session.yml'}: geometries is a list of G1, G2, G3")
    geometry = geometries[0]
    if config.lighting.geometry != geometry:
        raise _Refused(
            f"lighting.geometry {config.lighting.geometry} but the dataset was shot under "
            f"{geometry} — one model per geometry (docs/15 §7 step 1)"
        )
    camera = session.get("camera")
    device = camera.get("device") if isinstance(camera, dict) else None
    if not isinstance(device, str):
        raise _Refused(f"{dataset / 'session.yml'}: camera.device is missing")
    phone_shot = device.startswith("phone:")
    phone_config = config.camera.type == "phone"
    if phone_shot != phone_config:
        raise _Refused(
            f"the dataset was shot by {'the phone' if phone_shot else 'the webcam'} "
            f"({device}) but camera.type is {config.camera.type} — what is learned from "
            "phone frames belongs to the phone (ADR-0022)"
        )


def _measures(
    config: StationConfig, dataset: Path
) -> dict[str, list[GeometryMeasures]]:
    """Every `train/good` ∪ `calib/good` frame localized and measured (docs/15 §3, step 2)."""
    per_unit: dict[str, list[GeometryMeasures]] = {}
    for split in ("train", "calib"):
        directory = dataset / split / "good"
        for path in sorted(directory.glob("*.png")):
            try:
                name = parse_name(path.name)
            except FrameNameError as exc:
                raise _Refused(f"{path.name}: {exc}") from None
            data = path.read_bytes()
            pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise _Refused(f"{path.name}: not a readable PNG")
            try:
                silhouette = localize(pixels, config.inspector.localize)
                measured = measure(silhouette)
            except PlacementFault as exc:
                raise _Refused(f"{path.name}: placement: {exc.detail}") from None
            except InferenceError as exc:
                raise _Refused(f"{path.name}: inference: {exc}") from None
            per_unit.setdefault(name.unit, []).append(measured)
    return per_unit


def _tolerances(config: StationConfig, dataset: Path) -> Tolerances:
    """P06's `derive_tolerances` over the good frames of train ∪ calib."""
    try:
        tolerances = derive_tolerances(_measures(config, dataset))
    except ValueError as exc:
        raise _Refused(f"the tolerances cannot be derived: {exc}") from None
    if tolerances.config.area_px[0] >= tolerances.config.area_px[1]:
        # A real shoot has measurement noise, so σₙ > 0 and the range spans several pixels;
        # pixel-identical synthetic frames are the one way to get here. The loader refuses a
        # single-pixel range and nothing widens a derivation by itself (docs/15 §3, A2).
        raise _Refused(
            f"the derived area range is degenerate: area_px "
            f"[{tolerances.config.area_px[0]}, {tolerances.config.area_px[1]}] — the config "
            "loader refuses a single-pixel range and nothing widens a derivation "
            "automatically (docs/15 §3)"
        )
    return tolerances


def _units(
    results: tuple[FrameResult, ...],
) -> tuple[list[cal.UnitRecord], list[UnitScore], dict[str, dict[str, CheckResult]]]:
    """Per unit: the score as the maximum over its rotations, and whether any rotation
    failed geometry (docs/14 §4 step 3). The raw geometry `CheckResult`s come back too, for
    the not-reachable report."""
    scores: dict[str, float] = {}
    checks: dict[str, dict[str, CheckResult]] = {}
    failed: dict[str, bool] = {}
    for result in results:
        unit = result.name.unit
        assert result.score is not None and result.geometry is not None  # scored frames only
        scores[unit] = max(scores.get(unit, float("-inf")), result.score)
        checks.setdefault(unit, {})[f"r{result.name.rotation}"] = result.geometry
        failed[unit] = failed.get(unit, False) or not result.geometry.passed
    records = [
        cal.UnitRecord(
            unit,
            scores[unit],
            {r: cal.check_summary(check) for r, check in sorted(checks[unit].items())},
        )
        for unit in sorted(scores)
    ]
    unit_scores = [UnitScore(u, scores[u], failed[u]) for u in sorted(scores)]
    return records, unit_scores, checks


def _not_reachable(
    choice: ThresholdChoice, checks: dict[str, dict[str, CheckResult]],
    tolerances: Tolerances, target: Far,
) -> _Refused:
    """The A2 refusal: each flagged unit, rotation, rule and margin, next to the tolerances."""
    geometry = tolerances.config
    lines = [
        f"geometry alone flags {choice.g} of {choice.n} units at a far {target}% target "
        f"that allows {choice.a} — the tolerances are the problem, not the threshold; "
        "nothing widened, nothing written (A2, docs/15 §7)"
    ]
    for unit in choice.flagged:
        for rotation, check in sorted(checks[unit].items()):
            if not check.passed:
                lines.append(
                    f"  {unit} {rotation}: {check.failed_on} — {cal.margin(check, geometry)}"
                )
    sigma = tolerances.sigma
    lines.append(
        f"  tolerances: area_px [{geometry.area_px[0]}, {geometry.area_px[1]}] · "
        f"axis_ratio_min {geometry.axis_ratio_min:.3f} · "
        f"solidity_min {geometry.solidity_min:.3f} · σₙ area_px {sigma['area_px']:.3f}, "
        f"axis_ratio {sigma['axis_ratio']:.3f}, solidity {sigma['solidity']:.3f}"
    )
    return _Refused("\n".join(lines))


def _command(args: argparse.Namespace) -> str:
    parts = ["qs calibrate", "--config", str(args.config), "--dataset", str(args.dataset),
             "--model", args.model, "--out", str(args.out), "--far", str(args.far)]
    return " ".join(parts)


def _write_config(out: Path, text: str) -> int:
    """The config to a temporary file, checked with `load_config`, renamed to `--out` (#77)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".yml")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    try:
        load_config(temp)
    except ConfigError as exc:
        temp.unlink(missing_ok=True)
        print(f"calibrate: the new config does not load: {exc}")
        return 2
    os.replace(temp, out)
    return 0


def main(
    args: argparse.Namespace,
    *,
    make_scorer: Callable[[str, np.ndarray], Scorer] = real_scorer,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> int:
    """Calibrate; 0 done, 1 refused, 2 could not run."""
    try:
        return _calibrate(args, make_scorer, cuda_available)
    except _Refused as exc:
        print(f"calibrate: {exc}")
        return 1


def _calibrate(
    args: argparse.Namespace,
    make_scorer: Callable[[str, np.ndarray], Scorer],
    cuda_available: Callable[[], bool],
) -> int:
    # Step 1 — the refusals before any scoring.
    try:
        loaded = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    config = loaded.config
    out = Path(args.out)
    if out.exists():
        raise _Refused(f"--out {out} exists — calibration writes a new file, never an "
                       "overwrite (#77)")
    dataset = Path(args.dataset)
    folders = (dataset / "train" / "good", dataset / "calib" / "good")
    for directory in folders:
        if not directory.is_dir():
            print(f"dataset: {directory} is not a directory")
            return 2
    try:
        session = _session_yml(dataset)
    except OSError as exc:
        print(f"dataset: {exc}")
        return 2
    _check_session(config, dataset, session)
    try:
        bundle = load_bundle(Path(args.models), args.model)
    except ConfigError as exc:
        print(f"bundle: {exc}")
        return 2
    train_units = _dataset_units(folders[0])
    calib_units = _dataset_units(folders[1])
    bundle_units = set(bundle.units)
    shared = sorted(bundle_units & calib_units)
    if shared:
        raise _Refused(
            f"the bundle was trained on {', '.join(shared)}, a calibration unit — a "
            "threshold is never set on units the model has memorised (E1, docs/15 §7 step 2)"
        )
    if bundle_units != train_units:
        missing = sorted(train_units - bundle_units)
        extra = sorted(bundle_units - train_units)
        detail = (f"missing {', '.join(missing)}; " if missing else "") + (
            f"unknown {', '.join(extra)}; " if extra else ""
        )
        raise _Refused(f"the bundle's units are not the dataset's train/good units: "
                       f"{detail}calibrate only the model's own training units (E1)")
    # Step 2 — the tolerances, derived never typed.
    tolerances = _tolerances(config, dataset)
    # Step 3 — the calibration scores, through the station's own path.
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidate = cal.candidate_config(config, args.model, tolerances.config)
    candidate_block = cal.threshold_block(
        cal.CANDIDATE_ID, cal.CANDIDATE_VALUE, model=args.model, config=config,
        geometry=tolerances.config, calib_units=len(calib_units), target=args.far,
    )
    candidate_hash = canonical_hash(cal.with_threshold(raw, args.model, candidate_block))
    inspector = AppearanceInspector(
        candidate_hash, models_root=Path(args.models),
        make_scorer=make_scorer, cuda_available=cuda_available,
    )
    try:
        inspector.load(candidate)
    except ConfigError as exc:
        print(f"inspector: {exc}")
        return 2
    files = sorted(folders[1].glob("*.png"))
    results = score_frames(candidate, candidate_hash, files, inspector=inspector,
                           name="calibrate")
    for result in results:
        if result.label is Label.NO_VERDICT:
            raise _Refused(
                f"{result.file.name}: НЕТ РЕШЕНИЯ ({result.reason}) — a threshold is never "
                "set on fewer units than it claims (docs/15 §7 step 3)"
            )
    # Step 4 — the threshold.
    records, unit_scores, checks = _units(results)
    choice = choose_threshold(unit_scores, args.far)
    if choice.value is None:
        raise _not_reachable(choice, checks, tolerances, args.far)
    assert choice.value is not None
    # Step 5 — the record and its id.
    record = cal.build_record(
        model=args.model, backend=config.inspector.backend, target=args.far,
        dataset=dataset, config_hash=loaded.config_hash,
        localize_threshold=config.inspector.localize.threshold,
        mask_dilate_px=config.inspector.mask_dilate_px,
        units=records, n=choice.n, a=choice.a, g=choice.g,
        flagged=list(choice.flagged), tolerances=tolerances, threshold=choice.value,
    )
    id_ = cal.threshold_id(record, args.far)
    # Step 6 — the config first, then the record (A1).
    block = cal.threshold_block(
        id_, choice.value, model=args.model, config=config, geometry=tolerances.config,
        calib_units=choice.n, target=args.far,
    )
    report = cal.record_path(Path(args.report_dir), id_)
    header = (
        f"# Written by `{_command(args)}`.\n"
        f"# input: {args.config} (config_hash {loaded.config_hash})\n"
        f"# threshold_id: {id_} — the calibration record: {report}\n"
    )
    code = _write_config(out, cal.config_text(header, cal.with_threshold(raw, args.model,
                                                                       block)))
    if code != 0:
        return code
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _report(choice, id_, tolerances.config, out, report, args.far)
    return 0


def _report(
    choice: ThresholdChoice, id_: str, geometry: GeometryConfig, out: Path, report: Path,
    target: Far,
) -> None:
    """Step 7: the id, the threshold, the flags per check and combined, tolerances, paths."""
    print(f"threshold_id: {id_}")
    print(f"threshold: {choice.value} — far {target}% on {choice.n} calib units "
          f"(a = {choice.a}, g = {choice.g})")
    if choice.below_one_over_n:
        print(f"the target is below 1/n: one unit of {choice.n} is "
              f"{100.0 / choice.n:.1f}%")
    print(f"geometry flagged {choice.g} of {choice.n} · appearance flagged "
          f"{choice.k - choice.g} of {choice.n} · combined {choice.k} of {choice.n}")
    if choice.a - choice.g == 0:
        print(f"the highest calibration score: 0 of {choice.n} flagged")
    print(f"tolerances: area_px [{geometry.area_px[0]}, {geometry.area_px[1]}] · "
          f"axis_ratio_min {geometry.axis_ratio_min:.3f} · "
          f"solidity_min {geometry.solidity_min:.3f}")
    print(f"config: {out}")
    print(f"record: {report}")
