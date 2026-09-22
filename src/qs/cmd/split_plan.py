"""`qs split-plan`: the seeded dataset plan and its tray sheet (docs/13 §6, ADR-0014).

Run before any frame. Assigns every unit to one split by seed, gives each test unit its damage,
fixes the shooting order, writes `plan.yml` and a printable `plan.md` into `raw/<session>/`, and
prints the tray sheet. Refuses to overwrite a plan (exit 1).
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

from qs.names import GEOMETRIES
from qs.session import PlanExists, write_plan
from qs.splitplan import (
    PLACEMENT_EDGE,
    PLACEMENT_EMPTY,
    Plan,
    PlanError,
    Prompt,
    Shot,
    cell,
    make_plan,
    shot_list,
)

_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DATED = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def session_id(text: str) -> str:
    """argparse type: a plain folder name, so the plan lands inside `--root`."""
    if _SESSION.fullmatch(text) is None or ".." in text:
        raise argparse.ArgumentTypeError(
            f"{text!r}: letters, digits, '.', '_' and '-' only, e.g. 2026-09-25-mvp"
        )
    return text


def default_seed(session: str, today: date | None = None) -> int:
    """The session date as YYYYMMDD (docs/13 §8) — from the session ID when it starts with
    one, so the plan can be regenerated on another day; today's date otherwise."""
    match = _DATED.match(session)
    if match is not None:
        try:
            day = date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            day = today or date.today()
    else:
        day = today or date.today()
    return int(day.strftime("%Y%m%d"))


def _tray(plan: Plan) -> list[str]:
    rows = ["| Cell | Unit | Split | Damage |", "|---:|---|---|---|"]
    for unit in plan.control:
        rows.append(f"| {unit} | {unit} | control | — |")
    for unit in sorted((*plan.train, *plan.calib, *plan.test, *plan.spare)):
        split = plan.split_of(unit)
        note = plan.damage.get(unit, "—")
        if split == "spare":
            note = "in no split — only if a unit breaks"
        rows.append(f"| {cell(unit)} | {unit} | {split} | {note} |")
    return rows


def _order(plan: Plan) -> list[str]:
    """The shooting order, one line per unit — collapsed from the shot list itself."""
    lines: list[str] = []
    last: tuple[str, int | None] | None = None
    n = 0
    # The geometry is in no line of the outline; any valid one builds the list.
    for step in shot_list(plan, GEOMETRIES[0]):
        if isinstance(step, Prompt):
            if step.kind == "brush":
                lines.append("   - brush the floor")
            continue
        assert isinstance(step, Shot)
        unit, pass_ = step.frame.unit, step.frame.pass_
        if unit.startswith("c"):
            key: tuple[str, int | None] = ("control", pass_)
        elif unit.startswith("x"):
            key = ("placement", None)
        else:
            key = (unit, None)
        if key == last:
            continue
        last = key
        n += 1
        if key[0] == "control":
            lines.append(f"{n}. controls {', '.join(plan.control)} — pass {pass_}, 3 rotations")
        elif key[0] == "placement":
            lines.append(
                f"{n}. placement {PLACEMENT_EMPTY[0]}–{PLACEMENT_EMPTY[-1]} empty, "
                f"{PLACEMENT_EDGE[0]}–{PLACEMENT_EDGE[-1]} edge, one frame each"
            )
        elif unit in plan.damage:
            lines.append(
                f"{n}. {unit} (cell {cell(unit)}) test — 3 good, "
                f"damage: {plan.damage[unit]}, 3 dmg"
            )
        else:
            lines.append(f"{n}. {unit} (cell {cell(unit)}) {plan.split_of(unit)} — 3 good")
    return lines


def sheet(session: str, plan: Plan) -> str:
    """The printable tray sheet: cell → unit → split → damage, then the order."""
    text = [
        f"# Split plan — {session}",
        "",
        f"Size {plan.size} · seed {plan.seed} · {len(plan.train)} train · {len(plan.calib)} "
        f"calib · {len(plan.test)} test · {len(plan.control)} control · "
        f"{len(plan.spare)} spare",
        "",
        "## Tray sheet",
        "",
        *_tray(plan),
        "",
        "## Shooting order",
        "",
        *_order(plan),
    ]
    return "\n".join(text) + "\n"


def main(args: argparse.Namespace) -> int:
    """Plan the session; 0 written, 1 refused, 2 could not write."""
    seed = args.seed if args.seed is not None else default_seed(args.session)
    try:
        plan = make_plan(args.size, seed)
    except PlanError as exc:
        print(f"plan: {exc}")
        return 1
    text = sheet(args.session, plan)
    try:
        path = write_plan(Path(args.root) / args.session, plan, text)
    except PlanExists as exc:
        print(f"plan: {exc}")
        return 1
    except OSError as exc:
        print(f"plan: {exc}")
        return 2
    print(text, end="")
    print(f"\nplan: {path} · seed {plan.seed} — print {path.with_name('plan.md')}")
    return 0
