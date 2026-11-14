"""`qs unit-thresholds`: the limits of the off-colour methods, from block C (#177).

    qs unit-thresholds --scores calibrate.csv --lane lane.yml --unit unit.yml
                       [--patchcore STATE=MODEL_ID ...] --out unit-thresholds.yml
                       [--models models]

`calibrate.csv` is the output of `qs unit-scores --block calibrate`. For each state with at least
30 usable rows (rows with a fault are left out and counted) the limits are derived by
`qs.lane.unit_limits` and written with the SHA-256 of the lane file, of `unit.yml` and of each
PatchCore bundle's `model.pt`: `qs unit-report` refuses the test clips if any of them changed.
An existing output file is refused. Give `--patchcore` for exactly the states whose scores hold a
`patch_score`.

Exit codes: 0 written · 1 refused (no state has enough rows) · 2 could not run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qs.cmd.train_lane_b import _FULL
from qs.errors import ConfigError
from qs.lane.freeze import model_file_sha256
from qs.lane.lanefile import load_lane
from qs.lane.split import file_sha256
from qs.lane.unit import load_unit
from qs.lane.unit_limits import (
    MIN_ROWS,
    LimitsFile,
    StateLimits,
    derive_limits,
    read_scores,
    write_limits,
)


def add_parser(commands: Any) -> None:
    """Register `unit-thresholds` on the `qs` subcommands."""
    parser = commands.add_parser(
        "unit-thresholds", help="the limits of the off-colour methods, from the calibrate scores"
    )
    parser.add_argument("--scores", required=True, metavar="PATH",
                        help="the CSV of `qs unit-scores --block calibrate`")
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--unit", required=True, metavar="PATH", help="the unit settings file")
    parser.add_argument("--patchcore", action="append", default=[], metavar="STATE=MODEL_ID",
                        help="the lane-b bundle of one state; repeat for each state")
    parser.add_argument("--out", required=True, metavar="PATH",
                        help="the thresholds file to write (never overwritten)")
    parser.add_argument("--models", default="models", metavar="DIR",
                        help="where bundles are read (models)")


def _bundles(entries: list[str], models: Path) -> dict[str, tuple[str, str]]:
    """Each `STATE=MODEL_ID` mapped to (model_id, SHA-256 of the bundle's model.pt)."""
    found: dict[str, tuple[str, str]] = {}
    valid = ", ".join(state.value for state in _FULL)
    for entry in entries:
        state, separator, model_id = entry.partition("=")
        if not separator or state not in {s.value for s in _FULL}:
            raise ConfigError(f"--patchcore {entry!r}: expected STATE=MODEL_ID with STATE one of "
                              f"{valid}")
        if state in found:
            raise ConfigError(f"--patchcore: state {state!r} is given twice")
        try:
            found[state] = (model_id, model_file_sha256(models, model_id))
        except ConfigError as error:
            raise ConfigError(f"--patchcore {state}: {error}") from error
    return found


def _match_banks(
    limits: dict[str, StateLimits], patchcore: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """The banks of the states that got limits; a bank and a patch limit must go together."""
    for state, found in limits.items():
        if found.patch_score_max is not None and state not in patchcore:
            raise ConfigError(f"{state}: the scores hold a patch_score; give --patchcore "
                              f"{state}=MODEL_ID")
        if found.patch_score_max is None and state in patchcore:
            raise ConfigError(f"--patchcore {state}: the scores hold no patch_score for {state}; "
                              f"score them with the same --patchcore")
    return {state: bank for state, bank in patchcore.items() if state in limits}


def _run(args: argparse.Namespace) -> int:
    lane_path, unit_path = Path(args.lane), Path(args.unit)
    load_lane(lane_path)
    load_unit(unit_path)
    patchcore = _bundles(args.patchcore, Path(args.models))
    derived = derive_limits(read_scores(Path(args.scores)))
    for note in derived.notes:
        print(note)
    if not derived.limits:
        print(f"refused: no state has {MIN_ROWS} usable rows")
        return 1
    banks = _match_banks(derived.limits, patchcore)
    try:
        lane_sha, unit_sha = file_sha256(lane_path), file_sha256(unit_path)
    except OSError as error:
        raise ConfigError(f"cannot hash the lane or unit file: {error}") from error
    write_limits(Path(args.out), LimitsFile(lane_sha, unit_sha, banks, derived.limits))
    for state, found in derived.limits.items():
        print(f"{state}: limits from {found.n_usable} usable rows, "
              f"{found.n_faulted} with a fault left out")
    return 0


def main(args: argparse.Namespace) -> int:
    """Write the limits; 0 written, 1 refused, 2 could not run."""
    try:
        return _run(args)
    except ConfigError as error:
        print(f"unit-thresholds: {error}")
        return 2
