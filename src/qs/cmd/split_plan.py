"""`qs split-plan`: the seeded dataset plan and its tray sheet (docs/13 §6, ADR-0014).

Run before any frame. Assigns every unit to one split by seed, gives each test unit its damage,
fixes the shooting order, writes `plan.yml` and a printable `plan.md` into `raw/<session>/`, and
prints the tray sheet. Refuses to overwrite a plan (exit 1).

`--kind pilot` plans the lighting pilot instead (docs/13 §5, ADR-0013): units p01…, which are
dented and torn, the geometries (`--geometries G1,G2,G3` or `G1,G3`) and the order.

`--shot-list --geometry G` plans nothing: it reads `plan.yml` and writes `shots-<G>.md`, the
numbered photo list for the phone path (ADR-0022); on a pilot plan it writes `shots-pilot.md`,
and the plan's geometries replace `--geometry`. Row k is the k-th `Shot` of `shot_list` (or
`pilot_shot_list`), the functions `qs shoot` and `qs import-phone` use, so paper and import
cannot disagree.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

from qs.names import DEGREES, GEOMETRIES
from qs.session import (
    PLAN_FILE,
    PlanExists,
    ShotListDiffers,
    read_session_plan,
    write_plan,
    write_shot_list,
)
from qs.splitplan import (
    PILOT_GEOMETRIES,
    PLACEMENT_EDGE,
    PLACEMENT_EMPTY,
    PilotPlan,
    Plan,
    PlanError,
    Prompt,
    Shot,
    Step,
    cell,
    make_pilot_plan,
    make_plan,
    pilot_geometries,
    pilot_shot_list,
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


def geometry_list(text: str) -> tuple[str, ...]:
    """argparse type: the pilot's geometries, `G1,G2,G3` or `G1,G3` (docs/13 §5)."""
    try:
        return pilot_geometries(text)
    except PlanError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


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


def pilot_sheet(session: str, plan: PilotPlan) -> str:
    """The printable pilot sheet: cell → unit → damage, then the order (docs/13 §5)."""
    rows = ["| Cell | Unit | Pilot |", "|---:|---|---|"]
    for unit in plan.units:
        rows.append(f"| {cell(unit)} | {unit} | {plan.damage.get(unit, 'kept good')} |")
    geometries = ", ".join(plan.geometries)
    order = ", ".join(plan.order)
    damaged = [u for u in plan.order if u in plan.damage]
    text = [
        f"# Pilot plan — {session}",
        "",
        f"Size {plan.size} · seed {plan.seed} · {len(plan.units)} units · "
        f"{sum(d == 'dent' for d in plan.damage.values())} dent · "
        f"{sum(d == 'tear' for d in plan.damage.values())} tear · "
        f"{len(plan.good)} kept good · geometries {geometries}",
        "",
        "## Tray sheet",
        "",
        *rows,
        "",
        "## Shooting order",
        "",
        f"1. For each of {geometries}: move the lamps, then every unit good, 3 rotations, "
        f"in this order: {order}",
        f"2. Damage {', '.join(f'{u} ({plan.damage[u]})' for u in damaged)}",
        f"3. For each of {geometries}: move the lamps, then {', '.join(damaged)} dmg, "
        "3 rotations",
    ]
    return "\n".join(text) + "\n"


_CHECK = (
    "**CHECK — {n} photos so far. Copy the photos over USB, run `qs import-phone --check`, "
    "then carry on.**"
)


def _cell_text(text: str) -> str:
    return text.replace("|", "\\|")


def _what(shot: Shot) -> str:
    """The unit and its tray cell, as the tray sheet names them."""
    unit = shot.frame.unit
    if unit.startswith("c"):
        return f"{unit} · control"
    if unit.startswith("x"):
        where = "no unit in view" if shot.frame.state == "empty" else "a unit touching the edge"
        return f"{unit} · placement, {where}"
    return f"{unit} · cell {cell(unit)}"


def shots_text(session: str, plan: Plan, geometry: str) -> str:
    """The printable photo list: one numbered row per photo, prompts in bold, two CHECK rows."""
    steps = shot_list(plan, geometry)
    photos = [step for step in steps if isinstance(step, Shot)]
    # The CHECK rows follow the last photo of the controls' pass 1 and of their pass 2.
    checks = {
        max(i for i, s in enumerate(photos) if s.frame.pass_ == pass_) + 1 for pass_ in (1, 2)
    }
    about = f"size {plan.size} · geometry {geometry}"
    return _photo_list(f"{session} · {geometry}", session, plan.seed, about, steps, checks)


def pilot_shots_text(session: str, plan: PilotPlan) -> str:
    """The printable pilot list: lamp and damage steps in bold, one CHECK row after the first
    geometry's last good photo."""
    steps = pilot_shot_list(plan)
    photos = [step for step in steps if isinstance(step, Shot)]
    first = plan.geometries[0]
    checks = {max(i for i, s in enumerate(photos)
                  if s.frame.geometry == first and s.frame.state == "good") + 1}
    about = f"size {plan.size} · pilot · geometries {', '.join(plan.geometries)}"
    return _photo_list(f"{session} · pilot", session, plan.seed, about, steps, checks)


def _photo_list(title: str, session: str, seed: int, about: str,
                steps: tuple[Step, ...], checks: set[int]) -> str:
    """One numbered row per photo, prompts in bold, a CHECK row after each photo in `checks`."""
    photos = [step for step in steps if isinstance(step, Shot)]
    table = ["| # | ✓ | redo | Unit · cell | State | Rotation | File |",
             "|---:|---|---|---|---|---:|---|"]
    n = 0
    for step in steps:
        if isinstance(step, Prompt):
            table.append(f"| | | | **{_cell_text(step.instruction)}** | | | |")
            continue
        n += 1
        frame = step.frame
        table.append(
            f"| {n} | | | {_what(step)} | {frame.state} | {DEGREES[frame.rotation]}° "
            f"| `{step.file}` |"
        )
        if n in checks:
            table.append(f"| | | | {_CHECK.format(n=n)} | | | |")
    text = [
        f"# Shot list — {title}",
        "",
        f"Session {session} · seed {seed} · {about} · **{len(photos)} photos**",
        "",
        "**Never delete a photo on the phone — retake it at once and tick `redo` on its row.**",
        "",
        "Photo k on the phone is row k. Bold rows are steps, not photos.",
        "",
        *table,
    ]
    return "\n".join(text) + "\n"


def _shot_list(args: argparse.Namespace, kind: str) -> int:
    """Write the session's photo list; 0 written or already there, 1 differs, 2 cannot run."""
    if args.size is not None or args.seed is not None:
        print("shots: --shot-list reads the existing plan — drop --size and --seed")
        return 2
    folder = Path(args.root) / args.session
    if not (folder / PLAN_FILE).is_file():
        print(f"shots: {folder / PLAN_FILE} not found — run `qs split-plan` for this "
              "session first")
        return 2
    try:
        plan = read_session_plan(folder)
    except PlanError as exc:
        print(f"shots: {exc}")
        return 2
    if kind == "pilot" and not isinstance(plan, PilotPlan):
        print(f"shots: {folder / PLAN_FILE} is a dataset plan, not a pilot — drop --kind pilot")
        return 2
    if isinstance(plan, PilotPlan):
        if args.geometry is not None:
            print(f"shots: a pilot plan carries its geometries ({', '.join(plan.geometries)}) "
                  "— drop --geometry")
            return 2
        label, text = "pilot", pilot_shots_text(args.session, plan)
        count = sum(isinstance(step, Shot) for step in pilot_shot_list(plan))
    else:
        if args.geometry is None:
            print(f"shots: --shot-list needs --geometry, one of {', '.join(GEOMETRIES)}")
            return 2
        label, text = args.geometry, shots_text(args.session, plan, args.geometry)
        count = sum(isinstance(step, Shot) for step in shot_list(plan, args.geometry))
    try:
        path, fresh = write_shot_list(folder, label, text)
    except ShotListDiffers as exc:
        print(f"shots: {exc}")
        return 1
    except OSError as exc:
        print(f"shots: {exc}")
        return 2
    if fresh:
        print(f"shots: {path} · {count} photos — print it")
    else:
        print(f"shots: {path} is already there and identical — print it")
    return 0


def main(args: argparse.Namespace) -> int:
    """Plan the session, or with `--shot-list` write its photo list.

    0 written, 1 refused, 2 could not run.
    """
    kind = getattr(args, "kind", "dataset")
    geometries = getattr(args, "geometries", None)
    if args.shot_list:
        if geometries is not None:
            print("shots: --shot-list reads the geometries from plan.yml — drop --geometries")
            return 2
        return _shot_list(args, kind)
    if args.geometry is not None:
        print("plan: --geometry goes with --shot-list")
        return 2
    if geometries is not None and kind != "pilot":
        print("plan: --geometries goes with --kind pilot")
        return 2
    if args.size is None:
        print("plan: --size is required to plan a session (standard or lean)")
        return 2
    seed = args.seed if args.seed is not None else default_seed(args.session)
    plan: Plan | PilotPlan
    try:
        if kind == "pilot":
            plan = make_pilot_plan(args.size, seed, geometries or PILOT_GEOMETRIES[0])
        else:
            plan = make_plan(args.size, seed)
    except PlanError as exc:
        print(f"plan: {exc}")
        return 2 if kind == "pilot" else 1
    text = pilot_sheet(args.session, plan) if isinstance(plan, PilotPlan) else sheet(
        args.session, plan)
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
