"""Limits from block C good units, and the locked table of the test block (#177).

`derive_limits` turns the `qs unit-scores` rows of block C into limits, one set per state
(`full`, `full_white`, `full_pink` apart, because their colours differ). `judge` says which
methods flag one scored pass. `render_report` counts the test block's passes against the limits:
good units flagged and not answered, each with its bound (`qs.eval.stats`), and defective units
caught only when every one of their passes is flagged, since the passes of one unit are not
independent. A pass whose cut-out failed is no answer for every method (fail to НЕТ РЕШЕНИЯ).

The shape columns (`area_px`, `axis_ratio`, `solidity`) are optional: with them absent or empty
no shape limit is written and the table has no shape column.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qs.cmd.train_lane_b import _FULL
from qs.errors import ConfigError
from qs.eval.stats import lower_bound, upper_bound
from qs.lane.defects import Defect
from qs.lane.unit_scores import ScoreRow, colour_shift

MIN_ROWS = 30
_COLOUR_KINDS = ("patch", "colour")
_TITLES = {"spread": "colour spread", "shift": "colour shift", "patch": "PatchCore",
           "colour": "hand-written colour", "shape": "shape"}
_STATE_ORDER = tuple(state.value for state in _FULL)
_REQUIRED = ("video", "frame", "state", "fault", "colour_spread", "median_l", "median_a",
             "median_b", "patch_score")
_FLOATS = ("colour_spread", "median_l", "median_a", "median_b", "patch_score", "area_px",
           "axis_ratio", "solidity")


@dataclass(frozen=True, slots=True)
class Scored:
    """One full pass as the limits and the report read it. Scores are None when the pass had no
    cut-out (`fault` says why), when its state has no PatchCore bank, or, for the shape
    numbers, when `qs unit-scores` did not measure them."""

    video: str
    frame: int
    state: str
    fault: str
    colour_spread: float | None = None
    median_l: float | None = None
    median_a: float | None = None
    median_b: float | None = None
    patch_score: float | None = None
    area_px: float | None = None
    axis_ratio: float | None = None
    solidity: float | None = None


@dataclass(frozen=True, slots=True)
class StateLimits:
    """The limits of one state, and how many block C rows they rest on."""

    n_usable: int
    n_faulted: int
    colour_reference: tuple[float, float, float]
    colour_shift_max: float
    colour_spread_max: float
    patch_score_max: float | None = None
    area_px: tuple[float, float] | None = None
    axis_ratio_min: float | None = None
    solidity_min: float | None = None


@dataclass(frozen=True, slots=True)
class Derived:
    """The limits by state name, and one line for each state that got none."""

    limits: dict[str, StateLimits]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class LimitsFile:
    """The thresholds file: what it was made from (hashes) and the limits.
    `patchcore` maps a state to its bundle's (model_id, model.pt SHA-256)."""

    lane_sha256: str
    unit_sha256: str
    patchcore: dict[str, tuple[str, str]]
    states: dict[str, StateLimits]


# --- reading the scores -----------------------------------------------------------------


def from_score_rows(rows: Sequence[ScoreRow]) -> list[Scored]:
    """The rows of `score_block`. The shape numbers are taken when the row has them (the shape
    columns are a later ticket's), else they stay None."""
    return [
        Scored(r.video, r.frame, r.state, r.fault, r.colour_spread, r.median_l, r.median_a,
               r.median_b, r.patch_score, getattr(r, "area_px", None),
               getattr(r, "axis_ratio", None), getattr(r, "solidity", None))
        for r in rows
    ]


def read_scores(path: Path) -> list[Scored]:
    """The rows of a `qs unit-scores` CSV; an empty cell is None, absent shape columns too."""
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in _REQUIRED if name not in (reader.fieldnames or [])]
            if missing:
                raise ConfigError(f"{path}: not a scores file, no column {', '.join(missing)}")
            return [_scored(path, number, record) for number, record in enumerate(reader, 2)]
    except (OSError, csv.Error) as error:
        raise ConfigError(f"{path}: cannot read the scores: {error}") from error


def _scored(path: Path, line: int, record: Mapping[str, str]) -> Scored:
    try:
        numbers = {name: (float(record[name]) if record.get(name) else None) for name in _FLOATS}
        return Scored(record["video"], int(record["frame"]), record["state"], record["fault"],
                      **numbers)
    except ValueError as error:
        raise ConfigError(f"{path}: line {line}: {error}") from error


# --- the limits -------------------------------------------------------------------------


def _usable(row: Scored) -> bool:
    return not row.fault and None not in (
        row.colour_spread, row.median_l, row.median_a, row.median_b)


def _top_limit(values: Sequence[float]) -> float:
    """The value at index n // 100 of the values, largest first: at most 1% lie above it, and
    under 100 values it is the largest."""
    return sorted(values, reverse=True)[len(values) // 100]


def _colour(row: Scored) -> tuple[float, float, float]:
    assert row.median_l is not None and row.median_a is not None and row.median_b is not None
    return row.median_l, row.median_a, row.median_b


def _limits_of(rows: Sequence[Scored], n_faulted: int) -> StateLimits:
    colours = [_colour(row) for row in rows]
    reference = (statistics.median(c[0] for c in colours),
                 statistics.median(c[1] for c in colours),
                 statistics.median(c[2] for c in colours))
    patches = [row.patch_score for row in rows if row.patch_score is not None]
    areas = [row.area_px for row in rows if row.area_px is not None]
    ratios = [row.axis_ratio for row in rows if row.axis_ratio is not None]
    solids = [row.solidity for row in rows if row.solidity is not None]
    shaped = len(areas) == len(ratios) == len(solids) == len(rows)
    spreads = [row.colour_spread for row in rows if row.colour_spread is not None]
    return StateLimits(
        n_usable=len(rows),
        n_faulted=n_faulted,
        colour_reference=reference,
        colour_shift_max=_top_limit([colour_shift(c, reference) for c in colours]),
        colour_spread_max=_top_limit(spreads),
        patch_score_max=_top_limit(patches) if patches else None,
        area_px=(min(areas), max(areas)) if shaped else None,
        axis_ratio_min=min(ratios) if shaped else None,
        solidity_min=min(solids) if shaped else None,
    )


def derive_limits(rows: Sequence[Scored]) -> Derived:
    """The limits of every state with at least 30 usable rows (a row with a fault, or with a
    colour score missing, is left out and counted); one note for each state with fewer."""
    states = list(dict.fromkeys(row.state for row in rows))
    limits: dict[str, StateLimits] = {}
    notes: list[str] = []
    for state in states:
        of_state = [row for row in rows if row.state == state]
        usable = [row for row in of_state if _usable(row)]
        if len(usable) < MIN_ROWS:
            notes.append(f"{state}: {len(usable)} usable rows, fewer than {MIN_ROWS}; no limits")
        else:
            limits[state] = _limits_of(usable, len(of_state) - len(usable))
    return Derived(limits, notes)


# --- the file ---------------------------------------------------------------------------


def _state_body(limits: StateLimits) -> dict[str, object]:
    body: dict[str, object] = {
        "n_usable": limits.n_usable,
        "n_faulted": limits.n_faulted,
        "colour_reference": list(limits.colour_reference),
        "colour_shift_max": limits.colour_shift_max,
        "colour_spread_max": limits.colour_spread_max,
    }
    if limits.patch_score_max is not None:
        body["patch_score_max"] = limits.patch_score_max
    if limits.area_px is not None:
        body["area_px"] = list(limits.area_px)
        body["axis_ratio_min"] = limits.axis_ratio_min
        body["solidity_min"] = limits.solidity_min
    return body


def write_limits(path: Path, limits: LimitsFile) -> None:
    """Write the thresholds file; an existing file is never overwritten."""
    body = {
        "version": 1,
        "lane_sha256": limits.lane_sha256,
        "unit_sha256": limits.unit_sha256,
        "patchcore": {state: {"model_id": model_id, "model_sha256": sha}
                      for state, (model_id, sha) in limits.patchcore.items()},
        "states": {state: _state_body(found) for state, found in limits.states.items()},
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(yaml.safe_dump(body, sort_keys=False))
    except FileExistsError as error:
        raise ConfigError(f"{path} already exists; the limits are never overwritten") from error
    except OSError as error:
        raise ConfigError(f"{path}: cannot be written: {error}") from error


def _state_limits(body: Mapping[str, Any]) -> StateLimits:
    r, area = body["colour_reference"], body.get("area_px")
    return StateLimits(
        n_usable=int(body["n_usable"]),
        n_faulted=int(body["n_faulted"]),
        colour_reference=(float(r[0]), float(r[1]), float(r[2])),
        colour_shift_max=float(body["colour_shift_max"]),
        colour_spread_max=float(body["colour_spread_max"]),
        patch_score_max=_optional(body, "patch_score_max"),
        area_px=None if area is None else (float(area[0]), float(area[1])),
        axis_ratio_min=_optional(body, "axis_ratio_min"),
        solidity_min=_optional(body, "solidity_min"),
    )


def _optional(body: Mapping[str, Any], key: str) -> float | None:
    value = body.get(key)
    return None if value is None else float(value)


def load_limits(path: Path) -> LimitsFile:
    """The thresholds file at `path`; anything not as `write_limits` writes it is a ConfigError."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data["version"] != 1:
            raise ValueError("not a version 1 mapping")
        return LimitsFile(
            lane_sha256=str(data["lane_sha256"]),
            unit_sha256=str(data["unit_sha256"]),
            patchcore={state: (str(b["model_id"]), str(b["model_sha256"]))
                       for state, b in data["patchcore"].items()},
            states={state: _state_limits(body) for state, body in data["states"].items()},
        )
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: cannot read the thresholds file: {error}") from error
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        raise ConfigError(f"{path}: not a thresholds file: {error!r}") from error


# --- judging a pass ---------------------------------------------------------------------

FLAG, OK, NONE = "flag", "ok", "none"


def _above(value: float | None, limit: float) -> str:
    if value is None:
        return NONE
    return FLAG if value > limit else OK


def _shape(row: Scored, limits: StateLimits) -> str:
    assert limits.area_px is not None
    assert limits.axis_ratio_min is not None and limits.solidity_min is not None
    if row.area_px is None or row.axis_ratio is None or row.solidity is None:
        return NONE
    outside = (row.area_px < limits.area_px[0] or row.area_px > limits.area_px[1]
               or row.axis_ratio < limits.axis_ratio_min or row.solidity < limits.solidity_min)
    return FLAG if outside else OK


def judge(row: Scored, limits: StateLimits) -> dict[str, str]:
    """Each method's verdict on `row`: `flag`, `ok` or `none` (no answer). A value exactly at
    its limit is not flagged. A faulted pass is `none` everywhere. Methods the state has no limit
    for are left out; `colour` is the hand-written row, flagged when spread or shift is."""
    wanted = ["spread", "shift", *(["patch"] if limits.patch_score_max is not None else []),
              "colour", *(["shape"] if limits.area_px is not None else [])]
    if row.fault:
        return {method: NONE for method in wanted}
    verdict = {"spread": _above(row.colour_spread, limits.colour_spread_max)}
    if row.median_l is None or row.median_a is None or row.median_b is None:
        verdict["shift"] = NONE
    else:
        verdict["shift"] = _above(
            colour_shift(_colour(row), limits.colour_reference), limits.colour_shift_max)
    if limits.patch_score_max is not None:
        verdict["patch"] = _above(row.patch_score, limits.patch_score_max)
    colour = (verdict["spread"], verdict["shift"])
    verdict["colour"] = FLAG if FLAG in colour else NONE if NONE in colour else OK
    if limits.area_px is not None:
        verdict["shape"] = _shape(row, limits)
    return verdict


# --- the table --------------------------------------------------------------------------


def _line(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _of(n: int, k: int) -> str:
    return f"{k} of {n}"


def _defective_units(
    rows: Sequence[Scored], defects: Sequence[Defect]
) -> dict[str, tuple[str, list[Scored]]]:
    """Each defective unit's name mapped to its kind and the rows of its passes."""
    by_pass = {(row.video, row.frame): row for row in rows}
    units: dict[str, tuple[str, list[Scored]]] = {}
    for defect in defects:
        row = by_pass.get((defect.video, defect.frame))
        if row is None:
            continue
        kind, passes = units.setdefault(defect.unit, (defect.kind, []))
        if kind != defect.kind:
            raise ConfigError(f"unit {defect.unit} is given as {kind} and as {defect.kind}")
        passes.append(row)
    return units


def _state_table(
    state: str, limits: StateLimits, rows: Sequence[Scored], defects: Sequence[Defect]
) -> list[str]:
    defective = {(d.video, d.frame) for d in defects}
    good = [row for row in rows if (row.video, row.frame) not in defective]
    units = _defective_units(rows, defects)
    methods = list(judge(rows[0], limits))
    verdicts = {method: Counter(judge(row, limits)[method] for row in good) for method in methods}
    out = [f"### {state}", "",
           f"limits from {limits.n_usable} good block C units; "
           f"{limits.n_faulted} with a fault left out", "",
           _line(["", *(_TITLES[m] for m in methods)]), _line(["---"] * (len(methods) + 1))]
    n = len(good)
    if n:
        flagged = [_of(n, verdicts[m][FLAG]) for m in methods]
        bounds = [f"{upper_bound(n, verdicts[m][FLAG]):.4f}" for m in methods]
        none = [_of(n, verdicts[m][NONE]) for m in methods]
    else:
        flagged, bounds, none = (["no good unit"] * len(methods), ["-"] * len(methods),
                                 ["no good unit"] * len(methods))
    out += [_line(["good units flagged", *flagged]), _line(["upper bound", *bounds]),
            _line(["good units, no answer", *none])]
    passes_cells, caught_cells, lower_cells = [], [], []
    for method in methods:
        kinds = ("shape",) if method == "shape" else _COLOUR_KINDS
        mine = [passes for kind, passes in units.values() if kind in kinds]
        if not mine:
            passes_cells.append("no real defective unit")
            caught_cells += ["-"]
            lower_cells += ["-"]
            continue
        results = [[judge(row, limits)[method] for row in passes] for passes in mine]
        n_passes = sum(len(r) for r in results)
        caught = sum(all(v == FLAG for v in r) for r in results)
        passes_cells.append(_of(n_passes, sum(v == FLAG for r in results for v in r)))
        caught_cells.append(_of(len(results), caught))
        lower_cells.append(f"{lower_bound(len(results), caught):.4f}")
    out += [_line(["defective passes flagged", *passes_cells]),
            _line(["defective units caught", *caught_cells]),
            _line(["lower bound", *lower_cells])]
    for unit in sorted(units):
        kind, passes = units[unit]
        relevant = [m for m in methods if (m == "shape") == (kind == "shape")]
        flags = ", ".join(
            f"{_TITLES[m]} {_of(len(passes), sum(judge(r, limits)[m] == FLAG for r in passes))}"
            for m in relevant)
        out.append(f"- {unit} ({kind}), {len(passes)} passes: {flags}")
    return out


def render_report(
    limits: LimitsFile, rows: Sequence[Scored], defects: Sequence[Defect],
    rescored_run: int | None = None,
) -> str:
    """One table per state, from the test block's rows. A state with no limits, or no test
    pass, gets one line. With `rescored_run`, a first line `RESCORED (run N)` stands before it.
    A defect that matches no row raises ConfigError."""
    seen = {(row.video, row.frame) for row in rows}
    for defect in defects:
        if (defect.video, defect.frame) not in seen:
            raise ConfigError(f"defect {defect.unit}: {defect.video} frame {defect.frame} "
                              f"was not scored")
    out = [f"RESCORED (run {rescored_run})", ""] if rescored_run is not None else []
    names = [s for s in _STATE_ORDER if s in {r.state for r in rows} | set(limits.states)]
    for state in names:
        of_state = [row for row in rows if row.state == state]
        if state not in limits.states:
            out += [f"{state}: no limits, no result ({len(of_state)} passes)", ""]
        elif not of_state:
            out += [f"{state}: no test pass", ""]
        else:
            mine = [d for d in defects
                    if any((r.video, r.frame) == (d.video, d.frame) for r in of_state)]
            out += [*_state_table(state, limits.states[state], of_state, mine), ""]
    return "\n".join(out).rstrip("\n")

