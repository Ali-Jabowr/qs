"""`qs pilot-report`: the lighting decision's numbers (docs/13 §5, ADR-0013).

    bin/qs pilot-report --config PATH --session DIR [--report-dir reports/pilot]

It reads the pilot session's `session.yml` (`kind: pilot`, `geometries`, `units.pilot`,
`damage`, and any `discarded` units, which are skipped) and its `pNN_<state>_G<k>_r<n>.png`
frames, and reports per geometry, in ADR-0013's order:

1. **Localization** of every frame at the config's fixed threshold: counts per outcome, and
   every frame that is not exactly one unit listed with its fault detail.
2. **Threshold suggestion**: `suggest_threshold` over that geometry's good frames; the
   256-bin histogram goes to the CSV beside the report.
3. **Silhouette**: `rotation_spread` per good unit. The geometry passes only if every good
   unit localized in all 3 rotations and every spread is under 2%. Otherwise it is out.
4. **Window**: the median `diameter_px` of the good frames and `window_px = ⌈2 × median⌉`.
5. **Separation, leave-one-out**, through the model seam: for each good unit, fit on every
   other good unit's good frames at the suggested window, then score the held-out unit's
   rotations and every damaged unit's `dmg` rotations, masked to the silhouette dilated by
   the config's `mask_dilate_px`. A unit's score is the maximum over its rotations;
   gap = lowest damaged score − highest held-out good score, spread the sample standard
   deviation of the held-out good scores, separation = gap / spread, printed with n.
6. **Ranking**: the geometries that pass the silhouette, by separation. A tie — equal to two
   decimals — goes to G2, or to G3 in a G1 + G3 session. If none passes, the report says so
   and names ADR-0013's two-frame fallback.

The report is `reports/pilot/<session>.md` and `<session>-histogram.csv`: numbers only, no
images, and no model is written — the pilot's units are discarded after the decision
(docs/13 §5). The report recommends; Ali sets `lighting.geometry` and
`inspector.localize.threshold` by hand and writes the sentence. Exit codes: 0 report written
with a recommendation · 1 refused (a broken session, or the session is not a pilot), or the
report is written but no geometry passed · 2 could not run (config, missing `session.yml`,
CUDA, weights). Every refusal is one line, never a traceback.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from qs.config import StationConfig, load_config
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.inspector.geometry import GeometryMeasures, measure
from qs.inspector.localize import Silhouette, localize
from qs.inspector.model import FitSettings, PatchCore, Scorer, device_for
from qs.inspector.pilot import Separation, rank, separation, suggest_window
from qs.inspector.tolerances import ThresholdSuggestion, rotation_spread, suggest_threshold
from qs.inspector.window import cut_window, score_map, to_input
from qs.names import FrameName, FrameNameError, parse_name

INPUT_PX = 256  # docs/15 §5 — the model input the pilot's throwaway models are built at
CORESET_RATIO = 0.1
SEED = 42
SPREAD_LIMIT = 0.02  # ADR-0013: the silhouette criterion, a fraction of the mean area


class _Refused(Exception):
    """The run is refused: exit 1, one line."""


class _CannotRun(Exception):
    """The run cannot start: exit 2, one line."""


def _real_patchcore(device: str) -> PatchCore:
    from qs.inspector.patchcore import AnomalibPatchCore

    return AnomalibPatchCore(device)


def _cuda_available() -> bool:
    from qs.inspector.patchcore import cuda_available

    return cuda_available()


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


@dataclass(frozen=True, slots=True)
class _Session:
    """What the report needs from `session.yml` (docs/13 §8)."""

    session_id: str
    geometries: tuple[str, ...]
    units: tuple[str, ...]
    damage: frozenset[str]
    discarded: frozenset[str]


def _string_list(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _CannotRun(f"{what}: a list of names")
    return tuple(value)


def _string_keys(value: object, what: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _CannotRun(f"{what}: a mapping of names")
    return frozenset(value)


def _pilot_session(session: Path) -> _Session:
    """The pilot's `session.yml`, read and checked; `_Refused` when it is not a pilot."""
    path = session / "session.yml"
    if not path.is_file():
        raise _CannotRun(f"{path} not found")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _CannotRun(f"{path}: {_first_line(exc)}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise _CannotRun(f"{path}: {_first_line(exc)}") from None
    if not isinstance(data, dict):
        raise _CannotRun(f"{path}: not a session record")
    if data.get("kind") != "pilot":
        raise _Refused(f"not a pilot session: kind {data.get('kind')!r}")
    units_record = data.get("units")
    if not isinstance(units_record, dict):
        raise _CannotRun(f"{path}: units: a mapping with a pilot list")
    session_id = data.get("session_id")
    if not isinstance(session_id, str):
        raise _CannotRun(f"{path}: session_id: a string")
    return _Session(
        session_id=session_id,
        geometries=_string_list(data.get("geometries"), f"{path}: geometries"),
        units=_string_list(units_record.get("pilot"), f"{path}: units.pilot"),
        damage=_string_keys(data.get("damage"), f"{path}: damage"),
        discarded=_string_keys(data.get("discarded"), f"{path}: discarded"),
    )


@dataclass(frozen=True, slots=True)
class _Frame:
    """One pilot frame: its name, path, pixels, and what the localiser found in it."""

    name: FrameName
    path: Path
    pixels: np.ndarray
    silhouette: Silhouette | None = None
    measures: GeometryMeasures | None = None
    fault: str | None = None


@dataclass(frozen=True, slots=True)
class _Windowed:
    """One frame's model input, and what `score_map` needs to mask its map."""

    model_input: np.ndarray
    mask: np.ndarray
    box: tuple[int, int, int, int]


@dataclass(slots=True)
class _GeometryReport:
    """One geometry's numbers, built up as the run proceeds."""

    name: str
    good: list[_Frame]
    dmg: list[_Frame]
    counts: list[tuple[str, int]] = field(default_factory=list)
    faults: list[tuple[str, str]] = field(default_factory=list)
    suggestion: ThresholdSuggestion | None = None
    spreads: dict[str, float] = field(default_factory=dict)
    incomplete: list[tuple[str, int]] = field(default_factory=list)
    passes: bool = False
    median: float | None = None
    window_px: int | None = None
    sep: Separation | None = None


class _Model:
    """The model seam, built once when the first separation needs it.

    The weights are checked only when a separation is actually computed: a run that refuses
    every geometry never touches them.
    """

    def __init__(self, make: Callable[[str], PatchCore], device: str) -> None:
        self._make = make
        self._device = device
        self._model: PatchCore | None = None

    def get(self) -> PatchCore:
        if self._model is None:
            self._model = self._make(self._device)
        return self._model


def _load_frames(
    session: Path, pilot: _Session, config: StationConfig
) -> dict[str, dict[str, list[_Frame]]]:
    """Every pilot frame by geometry and state, localized at the config's fixed threshold."""
    frames: dict[str, dict[str, list[_Frame]]] = {
        geometry: {"good": [], "dmg": []} for geometry in pilot.geometries
    }
    for path in sorted(session.glob("*.png")):
        try:
            name = parse_name(path.name)
        except FrameNameError as exc:
            raise _Refused(f"{path.name}: {exc}") from None
        if name.unit in pilot.discarded:
            continue
        if name.unit not in pilot.units:
            raise _Refused(f"{path.name}: unit {name.unit} is not in the session")
        if name.geometry not in frames:
            raise _Refused(f"{path.name}: geometry {name.geometry} is not in the session")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise _CannotRun(f"{path.name}: {_first_line(exc)}") from None
        pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise _Refused(f"{path.name}: not a readable PNG")
        silhouette: Silhouette | None = None
        measures: GeometryMeasures | None = None
        fault: str | None = None
        try:
            silhouette = localize(pixels, config.inspector.localize)
            measures = measure(silhouette)
        except PlacementFault as exc:
            fault = exc.detail
        except InferenceError as exc:
            fault = str(exc)
        frames[name.geometry][name.state].append(
            _Frame(name, path, pixels, silhouette, measures, fault)
        )
    return frames


def _count(report: _GeometryReport) -> None:
    """The localization counts per outcome, and every faulted frame with its detail."""
    counts: dict[str, int] = {}
    for frame in [*report.good, *report.dmg]:
        counts[frame.fault or "ok"] = counts.get(frame.fault or "ok", 0) + 1
        if frame.fault is not None:
            report.faults.append((frame.path.name, frame.fault))
    report.counts = ([("ok", counts.pop("ok"))] if "ok" in counts else []) + list(
        counts.items()
    )


def _silhouette(report: _GeometryReport, good_units: Sequence[str]) -> None:
    """Each good unit's rotation spread; the geometry passes the 2% criterion or is out."""
    complete: dict[str, list[GeometryMeasures]] = {}
    for unit in good_units:
        localized = [
            frame
            for frame in report.good
            if frame.name.unit == unit and frame.measures is not None
        ]
        if len(localized) == 3:
            measures: list[GeometryMeasures] = []
            for frame in sorted(localized, key=lambda f: f.name.rotation):
                if frame.measures is not None:
                    measures.append(frame.measures)
            complete[unit] = measures
        else:
            report.incomplete.append((unit, len(localized)))
    if complete:
        report.spreads = rotation_spread(complete)
    report.passes = bool(good_units) and not report.incomplete and all(
        spread < SPREAD_LIMIT for spread in report.spreads.values()
    )


def _windowed(frames: Sequence[_Frame], window_px: int) -> list[_Windowed]:
    """The localized frames' model inputs, cut at `window_px` and resized once."""
    out: list[_Windowed] = []
    for frame in frames:
        if frame.silhouette is None:
            continue
        window, box = cut_window(frame.pixels, frame.silhouette.centroid, window_px)
        out.append(_Windowed(to_input(window, INPUT_PX), frame.silhouette.mask, box))
    return out


def _unit_score(
    windows: Sequence[_Windowed], scorer: Scorer, window_px: int, dilate_px: int
) -> float:
    """The unit's score: the maximum over its rotations of the masked map's maximum."""
    return max(
        score_map(scorer.score(w.model_input).anomaly_map, w.mask, w.box, window_px, dilate_px)[1]
        for w in windows
    )


def _separate(
    report: _GeometryReport,
    good_units: Sequence[str],
    damaged_units: Sequence[str],
    model: PatchCore,
    device: str,
    window_px: int,
    dilate_px: int,
) -> Separation:
    """The leave-one-out separation: one fit per good unit, never seeing its own frames."""
    good = {unit: _windowed([f for f in report.good if f.name.unit == unit], window_px)
            for unit in good_units}
    damaged = {unit: _windowed([f for f in report.dmg if f.name.unit == unit], window_px)
               for unit in damaged_units}
    good_scores: list[float] = []
    damaged_by_fold: list[list[float]] = []
    for held in good_units:
        fit_windows = [w.model_input for unit in good_units if unit != held for w in good[unit]]
        fitted = model.fit(fit_windows, FitSettings(CORESET_RATIO, SEED, device))
        scorer = model.scorer(fitted.bank)
        good_scores.append(_unit_score(good[held], scorer, window_px, dilate_px))
        damaged_by_fold.append(
            [_unit_score(windows, scorer, window_px, dilate_px)
             for windows in damaged.values() if windows]
        )
    return separation(good_scores, damaged_by_fold)


def _compute(
    session: Path,
    pilot: _Session,
    config: StationConfig,
    models: _Model,
    device: str,
) -> list[_GeometryReport]:
    """Every geometry's numbers, in `session.yml`'s order; silhouette first, then separation."""
    frames = _load_frames(session, pilot, config)
    roi = config.inspector.localize.roi
    dilate_px = config.inspector.mask_dilate_px
    good_units = [unit for unit in pilot.units if unit not in pilot.damage]
    damaged_units = [unit for unit in pilot.units if unit in pilot.damage]
    reports: list[_GeometryReport] = []
    for name in pilot.geometries:
        report = _GeometryReport(name=name, good=frames[name]["good"], dmg=frames[name]["dmg"])
        _count(report)
        suggestion = (
            suggest_threshold([f.pixels for f in report.good], roi) if report.good else None
        )
        report.suggestion = suggestion
        _silhouette(report, good_units)
        if report.passes and suggestion is not None:
            diameters = [f.measures.diameter_px for f in report.good if f.measures is not None]
            median, window_px = suggest_window(diameters)
            report.median, report.window_px = median, window_px
            report.sep = _separate(
                report, good_units, damaged_units, models.get(), device, window_px, dilate_px
            )
        reports.append(report)
    return reports


def _silhouette_text(report: _GeometryReport) -> str:
    """The silhouette line: every good unit's spread, and any unit that failed it."""
    spreads = ", ".join(
        f"{unit} {report.spreads[unit] * 100:.2f}%" for unit in sorted(report.spreads)
    )
    parts = ([spreads] if spreads else []) + [
        f"{unit}: {count} of 3 rotations" for unit, count in report.incomplete
    ]
    verdict = "pass" if report.passes else "out"
    detail = " — " + ", ".join(parts) if parts else ""
    return f"silhouette, spread < {SPREAD_LIMIT * 100:g}%: {verdict}{detail}"


def _geometry_section(report: _GeometryReport, histogram: str) -> list[str]:
    """One geometry's report lines, in ADR-0013's order."""
    total = len(report.good) + len(report.dmg)
    lines = [f"## {report.name}", ""]
    lines.append(f"- frames: {total} — {len(report.good)} good, {len(report.dmg)} dmg")
    lines.append(
        "- localization: "
        + ", ".join(f"{count} {outcome}" for outcome, count in report.counts)
    )
    lines.extend(f"  - {name}: {detail}" for name, detail in report.faults)
    if report.suggestion is None:
        lines.append("- threshold suggestion: none — no good frames")
        lines.append("- silhouette: out — no good frames")
        return lines
    lines.append(f"- threshold suggestion: {report.suggestion.value:.0f} — histogram: {histogram}")
    lines.append(f"- {_silhouette_text(report)}")
    if report.median is not None and report.window_px is not None:
        lines.append(
            f"- window: median diameter {report.median:.2f} px → window_px {report.window_px}"
        )
    if report.sep is not None:
        lines.append(
            f"- separation: {report.sep.separation:.2f} — gap {report.sep.gap:.4f}, "
            f"spread {report.sep.spread:.4f}, n {report.sep.n}"
        )
    return lines


def _write(
    report_dir: Path,
    session_name: str,
    reports: Sequence[_GeometryReport],
    ordered: Sequence[str],
    config_path: str,
    threshold: int,
    started: float,
) -> Path:
    """The report and its histogram CSV; the paths hold only these two files."""
    report_dir.mkdir(parents=True, exist_ok=True)
    histogram_name = f"{session_name}-histogram.csv"
    with (report_dir / histogram_name).open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("geometry", "bin", "count"))
        for report in reports:
            if report.suggestion is None:
                continue
            for bin_, count in enumerate(report.suggestion.histogram):
                writer.writerow((report.name, bin_, int(count)))

    lines = [
        f"# Pilot report — {session_name}",
        "",
        f"config {config_path} · localize threshold {threshold} · "
        f"wall time {time.monotonic() - started:.1f} s",
        "",
    ]
    for report in reports:
        lines.extend(_geometry_section(report, histogram_name))
        lines.append("")
    lines.append("## Ranking")
    lines.append("")
    if ordered:
        for geometry in ordered:
            sep = next(r.sep for r in reports if r.name == geometry)
            assert sep is not None
            lines.append(f"- {geometry}: separation {sep.separation:.2f}")
        for report in reports:
            if not report.passes:
                lines.append(f"- {report.name}: out by silhouette")
        lines.append("")
        lines.append(f"Recommendation: {ordered[0]}.")
        lines.append("")
        lines.append(
            "The report recommends; Ali decides — lighting.geometry and "
            "inspector.localize.threshold are set by hand, then the sentence (docs/13 §5)."
        )
    else:
        lines.append("No geometry passes the silhouette criterion.")
        lines.append("")
        lines.append(
            "Fallback: ADR-0013's two-frame design — a backlit frame for geometry, "
            "a low-angle one for appearance."
        )
    path = report_dir / f"{session_name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(
    args: argparse.Namespace,
    *,
    make_patchcore: Callable[[str], PatchCore] = _real_patchcore,
    cuda_available: Callable[[], bool] = _cuda_available,
) -> int:
    """Report one pilot session; 0 recommended, 1 refused or nothing passed, 2 cannot run."""
    try:
        return _report(args, make_patchcore, cuda_available)
    except KeyboardInterrupt:
        print("stopped by the operator — no report written")
        return 2


def _report(
    args: argparse.Namespace,
    make_patchcore: Callable[[str], PatchCore],
    cuda_available: Callable[[], bool],
) -> int:
    started = time.monotonic()
    try:
        config = load_config(Path(args.config)).config
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    try:
        device = device_for(config.inspector.backend, cuda_available=cuda_available())
    except ConfigError as exc:
        print(exc)
        return 2
    session = Path(args.session)
    try:
        pilot = _pilot_session(session)
    except _CannotRun as exc:
        print(f"session: {exc}")
        return 2
    except _Refused as exc:
        print(f"refused: {exc}")
        return 1
    try:
        reports = _compute(session, pilot, config, _Model(make_patchcore, device), device)
    except _Refused as exc:
        print(f"refused: {exc}")
        return 1
    except ConfigError as exc:
        print(exc)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(f"separation: {_first_line(exc)}")
        return 2
    except OSError as exc:
        print(f"session: {_first_line(exc)}")
        return 2
    passing = {
        report.name: report.sep.separation
        for report in reports
        if report.passes and report.sep is not None
    }
    ordered = rank(passing, pilot.geometries)
    session_name = pilot.session_id
    try:
        path = _write(
            Path(args.report_dir), session_name, reports, ordered, args.config,
            config.inspector.localize.threshold, started,
        )
    except OSError as exc:
        print(f"report: {_first_line(exc)}")
        return 2
    print(f"report: {path}")
    if ordered:
        print(f"recommendation: {ordered[0]}")
        return 0
    print("no geometry passed the silhouette criterion")
    return 1
