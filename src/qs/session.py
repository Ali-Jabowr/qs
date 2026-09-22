"""The session folder `raw/<session>/`: the only code that writes into it.

This ticket holds the split plan: `plan.yml`, the plan as data, and `plan.md`, its printable
tray sheet. A plan is written once — never overwritten, because units already sitting in trays
must not be silently reassigned — and read back through `Plan`, which refuses a unit in two
splits (FR-52).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from qs.splitplan import Plan, PlanError

PLAN_FILE = "plan.yml"
SHEET_FILE = "plan.md"
_SPLITS = ("control", "train", "calib", "test", "spare")
_HEADER = "# The split plan, written by `qs split-plan`. Never edited by hand.\n"


class PlanExists(Exception):
    """The session already has a plan; it is left as it is."""


def _plan_data(plan: Plan) -> dict[str, Any]:
    return {
        "kind": "dataset",
        "size": plan.size,
        "seed": plan.seed,
        "units": {name: list(getattr(plan, name)) for name in _SPLITS},
        "damage": dict(plan.damage),
        "order": list(plan.order),
    }


def write_plan(folder: Path, plan: Plan, sheet: str) -> Path:
    """Write `plan.yml` and `plan.md` into `folder`; refuse if `plan.yml` exists."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / PLAN_FILE
    text = _HEADER + yaml.safe_dump(
        _plan_data(plan), sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    try:
        with path.open("x", encoding="utf-8") as out:  # exclusive: never overwrites
            out.write(text)
    except FileExistsError:
        raise PlanExists(
            f"{path} exists — refusing to overwrite it: its units may already be in trays"
        ) from None
    (folder / SHEET_FILE).write_text(sheet, encoding="utf-8")
    return path


def _strings(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PlanError(f"{where}: a list of unit names")
    return tuple(value)


def read_plan(folder: Path) -> Plan:
    """The plan in `folder`; refuses a file that is not a valid plan."""
    path = folder / PLAN_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PlanError(f"{path}: {exc}") from None
    if not isinstance(data, dict) or sorted(data) != sorted(
        ("kind", "size", "seed", "units", "damage", "order")
    ):
        raise PlanError(f"{path}: keys are kind, size, seed, units, damage, order")
    if data["kind"] != "dataset":
        raise PlanError(f"{path}: kind {data['kind']!r} — this reads dataset plans")
    units = data["units"]
    if not isinstance(units, dict) or sorted(units) != sorted(_SPLITS):
        raise PlanError(f"{path}: units are {', '.join(_SPLITS)}")
    damage = data["damage"]
    if not isinstance(damage, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in damage.items()
    ):
        raise PlanError(f"{path}: damage maps a test unit to its damage")
    if not isinstance(data["size"], str) or type(data["seed"]) is not int:
        raise PlanError(f"{path}: size is a word, seed an integer")
    return Plan(
        size=data["size"],
        seed=data["seed"],
        train=_strings(units["train"], "units.train"),
        calib=_strings(units["calib"], "units.calib"),
        test=_strings(units["test"], "units.test"),
        control=_strings(units["control"], "units.control"),
        spare=_strings(units["spare"], "units.spare"),
        damage=dict(damage),
        order=_strings(data["order"], "order"),
    )
