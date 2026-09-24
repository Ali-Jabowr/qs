"""The split plan: which unit goes to which split, how it is damaged, and the shooting order.

Pure: `make_plan(size, seed) -> Plan` and `shot_list(plan, geometry) -> steps`; for the
lighting pilot, `make_pilot_plan(size, seed, geometries) -> PilotPlan` and
`pilot_shot_list(plan) -> steps` (docs/13 §5). No I/O, no clock. Every unit's split is fixed
by the seed before the first frame, so the shooting order can be random without any split being
tied to time (docs/13 §6, ADR-0014). A `Plan` with a unit in two splits cannot be constructed
(FR-52).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from qs.names import DEGREES, ROTATIONS, FrameName, format_name


class PlanError(ValueError):
    """A plan that breaks a split rule: refused, never written, never shot.

    Not a `QSError`: those are the Pipeline's NO_VERDICT reasons.
    """


@dataclass(frozen=True, slots=True)
class Sizes:
    """One column of docs/13 §1 (and docs/14 §9)."""

    train: int
    calib: int
    dent: int
    tear: int
    deformed: int
    spare: int

    @property
    def test(self) -> int:
        return self.dent + self.tear + self.deformed


SIZES = {
    "standard": Sizes(train=20, calib=15, dent=12, tear=9, deformed=9, spare=5),
    "lean": Sizes(train=18, calib=10, dent=10, tear=7, deformed=7, spare=3),
}
CONTROLS = ("c01", "c02", "c03")
PLACEMENT_EMPTY = tuple(f"x{n:02d}" for n in range(1, 6))
PLACEMENT_EDGE = tuple(f"x{n:02d}" for n in range(6, 11))
DAMAGES = ("dent", "tear", "deformed-flat", "deformed-oval")
BRUSH_EVERY = 10  # units (docs/13 §6)
_SPLITS = ("control", "train", "calib", "test", "spare")

# docs/13 §7, verbatim, so the recipe is repeatable at Gate A.
RECIPES = {
    "dent": "A rounded tip about 10 mm across — the end of a marker cap — pressed straight down "
    "about 3 mm, once, off-centre on the top",
    "tear": "A clean knife slit 5–10 mm through the skin on the top or upper side, opened "
    "slightly so the filling shows",
    "deformed-flat": "Flattened: pressed from above with a flat plate until about a third of the "
    "height is gone",
    "deformed-oval": "Oval: squeezed from two sides",
}


def cell(unit: str) -> int:
    """The tray cell of a dataset unit: its number (docs/13 §6)."""
    return int(unit[1:])


@dataclass(frozen=True, slots=True)
class Plan:
    """A dataset session's split plan. Constructing one that breaks a split rule raises."""

    size: str
    seed: int
    train: tuple[str, ...]
    calib: tuple[str, ...]
    test: tuple[str, ...]
    control: tuple[str, ...]
    spare: tuple[str, ...]
    damage: dict[str, str] = field(hash=False)  # test unit → dent | tear | deformed-flat | -oval
    order: tuple[str, ...]  # the shooting order of the split units; no control, no spare

    def __post_init__(self) -> None:
        _check(self)

    def split_of(self, unit: str) -> str:
        for name in ("train", "calib", "test", "control", "spare"):
            if unit in getattr(self, name):
                return name
        raise PlanError(f"{unit} is in no split")


def _check(plan: Plan) -> None:
    # Two splits first: that is the error that ruins a dataset quietly (E1, FR-52).
    seen: dict[str, str] = {}
    for name in ("train", "calib", "test", "control", "spare"):
        for unit in getattr(plan, name):
            if unit in seen:
                where = "twice in " + name if seen[unit] == name else f"in {seen[unit]} and {name}"
                raise PlanError(f"{unit} is {where}: a unit is in exactly one split (FR-52)")
            seen[unit] = name
    sizes = SIZES.get(plan.size)
    if sizes is None:
        raise PlanError(f"size {plan.size!r}: one of {', '.join(SIZES)}")
    wanted = {"train": sizes.train, "calib": sizes.calib, "test": sizes.test,
              "control": len(CONTROLS), "spare": sizes.spare}
    for name, count in wanted.items():
        if len(getattr(plan, name)) != count:
            raise PlanError(f"{name} has {len(getattr(plan, name))} units; {plan.size} is {count}")
    if plan.control != CONTROLS:
        raise PlanError(f"control is {', '.join(plan.control)}; it is {', '.join(CONTROLS)}")
    total = sizes.train + sizes.calib + sizes.test
    if sorted((*plan.train, *plan.calib, *plan.test, *plan.spare)) != [
        f"u{n:03d}" for n in range(1, total + sizes.spare + 1)
    ] or any(cell(u) <= total for u in plan.spare):
        raise PlanError(f"units are u001…u{total:03d} in the splits, then the spares")
    split_units = Counter((*plan.train, *plan.calib, *plan.test))
    if Counter(plan.order) != split_units:
        raise PlanError("order: every split unit exactly once, and no control or spare")
    if set(plan.damage) != set(plan.test):
        raise PlanError("damage: exactly the test units carry a damage class")
    bad = sorted(u for u, d in plan.damage.items() if d not in DAMAGES)
    if bad:
        raise PlanError(f"damage of {bad[0]}: one of {', '.join(DAMAGES)}")
    got = Counter(d.split("-")[0] for d in plan.damage.values())
    for kind in ("dent", "tear", "deformed"):
        if got[kind] != getattr(sizes, kind):
            raise PlanError(f"damage: {got[kind]} {kind}; {plan.size} is {getattr(sizes, kind)}")


def make_plan(size: str, seed: int) -> Plan:
    """The plan for this size and seed; the same seed gives the same plan."""
    sizes = SIZES.get(size)
    if sizes is None:
        raise PlanError(f"size {size!r}: one of {', '.join(SIZES)}")
    rng = random.Random(seed)
    total = sizes.train + sizes.calib + sizes.test
    units = [f"u{n:03d}" for n in range(1, total + 1)]
    spare = tuple(f"u{n:03d}" for n in range(total + 1, total + sizes.spare + 1))

    shuffled = list(units)
    rng.shuffle(shuffled)
    train = shuffled[: sizes.train]
    calib = shuffled[sizes.train : sizes.train + sizes.calib]
    test = shuffled[sizes.train + sizes.calib :]

    by_damage = sorted(test)
    rng.shuffle(by_damage)
    damage = dict.fromkeys(by_damage[: sizes.dent], "dent")
    damage |= dict.fromkeys(by_damage[sizes.dent : sizes.dent + sizes.tear], "tear")
    deformed = by_damage[sizes.dent + sizes.tear :]
    damage |= {u: ("deformed-flat", "deformed-oval")[i % 2] for i, u in enumerate(deformed)}

    order = list(units)
    rng.shuffle(order)
    return Plan(
        size=size,
        seed=seed,
        train=tuple(sorted(train)),
        calib=tuple(sorted(calib)),
        test=tuple(sorted(test)),
        control=CONTROLS,
        spare=spare,
        damage=dict(sorted(damage.items())),
        order=tuple(order),
    )


# --- the lighting pilot (docs/13 §5, ADR-0013) ----------------------------------------------


@dataclass(frozen=True, slots=True)
class PilotSizes:
    """One pilot size of docs/13 §5: the units, and how many of them are damaged."""

    units: int
    dent: int
    tear: int


PILOT_SIZES = {
    "standard": PilotSizes(units=10, dent=2, tear=2),
    "lean": PilotSizes(units=6, dent=1, tear=1),
}
# G1, G2, G3; or G1 and G3 when the house has one lamp — G2 needs two (docs/13 §3, §5).
PILOT_GEOMETRIES = (("G1", "G2", "G3"), ("G1", "G3"))
PILOT_DAMAGES = ("dent", "tear")


def pilot_geometries(text: str) -> tuple[str, ...]:
    """`G1,G2,G3` or `G1,G3` as a tuple; anything else is refused, the allowed lists named."""
    geometries = tuple(part.strip() for part in text.split(","))
    if geometries not in PILOT_GEOMETRIES:
        allowed = " or ".join(",".join(g) for g in PILOT_GEOMETRIES)
        raise PlanError(f"geometries {text!r}: the pilot compares {allowed} (docs/13 §5)")
    return geometries


@dataclass(frozen=True, slots=True)
class PilotPlan:
    """A lighting-pilot session's plan: units `p01`…, which of them are damaged how, the
    geometries in the order they are shot, and the seeded shooting order of the units.
    Constructing one that breaks a pilot rule raises."""

    size: str
    seed: int
    geometries: tuple[str, ...]
    units: tuple[str, ...]
    damage: dict[str, str] = field(hash=False)  # damaged unit → dent | tear
    order: tuple[str, ...]  # every pilot unit once, in the order it is shot

    def __post_init__(self) -> None:
        sizes = PILOT_SIZES.get(self.size)
        if sizes is None:
            raise PlanError(f"size {self.size!r}: one of {', '.join(PILOT_SIZES)}")
        pilot_geometries(",".join(self.geometries))
        if self.units != tuple(f"p{n:02d}" for n in range(1, sizes.units + 1)):
            raise PlanError(f"units are p01…p{sizes.units:02d} for a {self.size} pilot")
        if sorted(self.order) != list(self.units):
            raise PlanError("order: every pilot unit exactly once")
        if not set(self.damage) <= set(self.units):
            raise PlanError("damage: only pilot units are damaged")
        got = Counter(self.damage.values())
        if set(got) - set(PILOT_DAMAGES):
            raise PlanError(f"damage: one of {', '.join(PILOT_DAMAGES)}")
        for kind in PILOT_DAMAGES:
            if got[kind] != getattr(sizes, kind):
                raise PlanError(
                    f"damage: {got[kind]} {kind}; a {self.size} pilot is {getattr(sizes, kind)}"
                )

    @property
    def good(self) -> tuple[str, ...]:
        """The units kept good all through."""
        return tuple(u for u in self.units if u not in self.damage)


def make_pilot_plan(
    size: str, seed: int, geometries: tuple[str, ...] = PILOT_GEOMETRIES[0]
) -> PilotPlan:
    """The pilot plan for this size, seed and geometries; the same seed gives the same plan."""
    sizes = PILOT_SIZES.get(size)
    if sizes is None:
        raise PlanError(f"size {size!r}: one of {', '.join(PILOT_SIZES)}")
    rng = random.Random(seed)
    units = tuple(f"p{n:02d}" for n in range(1, sizes.units + 1))
    shuffled = list(units)
    rng.shuffle(shuffled)
    damage = dict.fromkeys(shuffled[: sizes.dent], "dent")
    damage |= dict.fromkeys(shuffled[sizes.dent : sizes.dent + sizes.tear], "tear")
    order = list(units)
    rng.shuffle(order)
    return PilotPlan(
        size=size,
        seed=seed,
        geometries=tuple(geometries),
        units=units,
        damage=dict(sorted(damage.items())),
        order=tuple(order),
    )


@dataclass(frozen=True, slots=True)
class Shot:
    """One frame to take: its name and what the screen tells Ali."""

    frame: FrameName
    instruction: str

    @property
    def file(self) -> str:
        return format_name(self.frame)


@dataclass(frozen=True, slots=True)
class Prompt:
    """A step between frames: `damage` a test unit, or `brush` the floor."""

    kind: str
    unit: str | None
    instruction: str


Step = Shot | Prompt


def _rotations(unit: str, state: str, geometry: str, what: str, pass_: int | None) -> list[Step]:
    return [
        Shot(FrameName(unit, state, geometry, r, pass_), f"{what} · {DEGREES[r]}° · {state}")
        for r in ROTATIONS
    ]


def _unit_steps(unit: str, split: str, damage: str | None, geometry: str,
                note: str = "") -> list[Step]:
    """One dataset unit: 3 good rotations; a test unit's damage step and 3 dmg rotations."""
    what = f"{unit} · cell {cell(unit)} · {split}{note}"
    steps = _rotations(unit, "good", geometry, what, None)
    if damage is not None:
        steps.append(Prompt(
            "damage", unit,
            f"Take {unit} out and damage it on the board — {damage}: {RECIPES[damage]}. "
            "Put it back in the rig.",
        ))
        steps.extend(_rotations(unit, "dmg", geometry, f"{what} · {damage}", None))
    return steps


def _session_steps(
    plan: Plan, geometry: str, unit_steps: Callable[[str], list[Step]]
) -> tuple[Step, ...]:
    """The dataset session around the units; `unit_steps` gives each split unit's steps."""
    steps: list[Step] = []

    def controls(pass_: int) -> None:
        for unit in plan.control:
            steps.extend(_rotations(unit, "good", geometry, f"{unit} · control, pass {pass_}",
                                    pass_))

    controls(1)
    half = len(plan.order) // 2
    for done, unit in enumerate(plan.order, start=1):
        steps.extend(unit_steps(unit))
        if done % BRUSH_EVERY == 0:
            steps.append(Prompt("brush", None, "Brush the floor of the rig clear of starch."))
        if done == half:
            controls(2)
    controls(3)
    for unit in PLACEMENT_EMPTY:
        steps.append(Shot(FrameName(unit, "empty", geometry, 0, None),
                          f"{unit} · placement · no unit in view"))
    for unit in PLACEMENT_EDGE:
        steps.append(Shot(FrameName(unit, "edge", geometry, 0, None),
                          f"{unit} · placement · a unit touching the edge of the inspection area"))
    return tuple(steps)


def shot_list(plan: Plan, geometry: str) -> tuple[Step, ...]:
    """The dataset session, step by step (docs/13 §6, ADR-0014).

    Controls pass 1 → every split unit in the plan order, 3 rotations; a test unit's good frames
    are followed at once by its damage step and its dmg frames → controls pass 2 after half the
    units → the rest → controls pass 3 → placement x01–x05 empty, x06–x10 edge, at r0. A floor
    brush every 10 units.
    """
    return _session_steps(
        plan, geometry,
        lambda unit: _unit_steps(unit, plan.split_of(unit), plan.damage.get(unit), geometry),
    )


# docs/13 §5, the "How" column, verbatim: where the lamps go for each geometry.
LAMPS = {
    "G1": "Low-angle, one side: one lamp 20–30° above the floor, from the left",
    "G2": "Low-angle, two sides: two lamps at the same angle, opposite each other",
    "G3": "Diffuse: the lamp aimed at white card or through a tracing-paper tent",
}


def _lamps(geometry: str, round_: str) -> Prompt:
    return Prompt(
        "lamps", None,
        f"Lamps to {geometry} — {LAMPS[geometry]}. Then leave them for every {round_} photo "
        f"under {geometry}.",
    )


def pilot_shot_list(plan: PilotPlan) -> tuple[Step, ...]:
    """The pilot session, step by step (docs/13 §5, ADR-0013).

    For each geometry: a lamp step, then every unit good in 3 rotations, in the plan order.
    Then one damage step for the damaged units. Then for each geometry again: a lamp step and
    the damaged units dmg in 3 rotations. The lamps move per geometry, not per unit; no
    controls, no placement frames.
    """
    return _pilot_steps(plan, {})[0]


def _pilot_steps(
    plan: PilotPlan, after: dict[str, int]
) -> tuple[tuple[Step, ...], dict[str, tuple[str, ...]]]:
    """The pilot steps with each unit in `after` cut after its first n frames; the steps, and
    each cut unit's frame files as shot."""
    shot_of: Counter[str] = Counter()
    kept: dict[str, list[str]] = {unit: [] for unit in after}
    steps: list[Step] = []

    def rotations(unit: str, state: str, geometry: str) -> None:
        what = f"{unit} · cell {cell(unit)} · pilot"
        if unit in plan.damage:
            what += f" · {plan.damage[unit]}"
        for step in _rotations(unit, state, geometry, what, None):
            assert isinstance(step, Shot)
            if unit in after:
                if shot_of[unit] >= after[unit]:
                    continue
                kept[unit].append(step.file)
            shot_of[unit] += 1
            steps.append(step)

    damaged = [unit for unit in plan.order if unit in plan.damage]
    for geometry in plan.geometries:
        steps.append(_lamps(geometry, "good"))
        for unit in plan.order:
            rotations(unit, "good", geometry)
    goods = 3 * len(plan.geometries)
    # A unit that broke before the damage step is not damaged: it is out of the session.
    to_damage = [u for u in damaged if u not in after or after[u] > goods]
    if to_damage:
        recipes = "; ".join(f"{u} — {plan.damage[u]}: {RECIPES[plan.damage[u]]}"
                            for u in to_damage)
        steps.append(Prompt(
            "damage", None,
            f"Take {', '.join(to_damage)} out and damage each on the board — {recipes}. "
            "Put each back in its cell.",
        ))
    for geometry in plan.geometries:
        steps.append(_lamps(geometry, "dmg"))
        for unit in damaged:
            rotations(unit, "dmg", geometry)
    return tuple(steps), {unit: tuple(files) for unit, files in kept.items()}


@dataclass(frozen=True, slots=True)
class Substitution:
    """The session as shot when units broke (assumption A1, #28).

    `steps` is the shot list with each broken unit cut after the frames it was shot in and the
    next unused spare's steps in its place. `units` (control, train, calib, test, spare) and
    `damage` are the splits as shot: the spare stands where the broken unit stood; with no spare
    left the split is one short. `discarded` maps each broken unit to its frame files, in order;
    `replaced` maps it to the spare that took its place, or None.
    """

    steps: tuple[Step, ...]
    units: dict[str, tuple[str, ...]]
    damage: dict[str, str]
    discarded: dict[str, tuple[str, ...]]
    replaced: dict[str, str | None]


def substitute_spares(
    plan: Plan, geometry: str, broken: Sequence[tuple[str, int]]
) -> Substitution:
    """The session with each `(unit, n)` broken after its first n frames; pure.

    The next unused spare takes the unit's split and damage class at the same place in the
    order, from its first rotation, under its own number. The broken unit is discarded whole —
    a test unit broken after its good frames too, so every test unit keeps its pair. A spare
    that breaks in turn is replaced by the next one. Shared by `qs import-phone` (#53) and
    `qs shoot` (#33), so both paths give the same `session.yml`.
    """
    after: dict[str, int] = {}
    for unit, n in broken:
        if unit in after:
            raise PlanError(f"{unit} is marked broken twice")
        after[unit] = n
    spares = list(plan.spare)
    units = {name: list(getattr(plan, name)) for name in _SPLITS}
    damage = dict(plan.damage)
    discarded: dict[str, tuple[str, ...]] = {}
    replaced: dict[str, str | None] = {}

    def unit_steps(unit: str, split: str, dmg: str | None, note: str = "") -> list[Step]:
        block = _unit_steps(unit, split, dmg, geometry, note)
        n = after.get(unit)
        if n is None:
            return block
        frames = sum(isinstance(step, Shot) for step in block)
        if not 0 <= n < frames:
            raise PlanError(
                f"{unit} broke after {n} photos: it has {frames}, so 0 to {frames - 1}"
            )
        taken: list[Step] = []
        for step in block:
            if sum(isinstance(s, Shot) for s in taken) == n:
                break
            taken.append(step)
        discarded[unit] = tuple(step.file for step in taken if isinstance(step, Shot))
        spare = spares.pop(0) if spares else None
        replaced[unit] = spare
        members = units[split]
        at = members.index(unit)
        if dmg is not None:
            del damage[unit]
        if spare is None:
            del members[at]
            return taken
        members[at] = spare
        units["spare"].remove(spare)
        if dmg is not None:
            damage[spare] = dmg
        return taken + unit_steps(spare, split, dmg, f" · spare for {unit}")

    steps = _session_steps(
        plan, geometry,
        lambda unit: unit_steps(unit, plan.split_of(unit), plan.damage.get(unit)),
    )
    never = sorted(set(after) - set(discarded))
    if never:
        raise PlanError(
            f"{never[0]} is not shot in this session — a broken unit is a split unit or a "
            "spare in use"
        )
    return Substitution(
        steps=steps,
        units={name: tuple(members) for name, members in units.items()},
        damage=dict(sorted(damage.items())),
        discarded=discarded,
        replaced=replaced,
    )


def discard_broken(plan: PilotPlan, broken: Sequence[tuple[str, int]]) -> Substitution:
    """The pilot as shot when units broke after their first n frames; pure.

    The pilot has no spares, so a broken unit is discarded whole: its frames so far are kept on
    disk and listed under `discarded`, its later shots leave the list, and it leaves
    `units.pilot` and `damage` (A1, #28). The `Substitution` shape is `substitute_spares`'s,
    so `qs import-phone` treats both alike; `replaced` maps every broken unit to None.
    """
    after: dict[str, int] = {}
    for unit, n in broken:
        if unit in after:
            raise PlanError(f"{unit} is marked broken twice")
        if unit not in plan.units:
            raise PlanError(f"{unit} is not shot in this session — the pilot units are "
                            f"{plan.units[0]}…{plan.units[-1]}")
        frames = 3 * len(plan.geometries) * (2 if unit in plan.damage else 1)
        if not 0 <= n < frames:
            raise PlanError(f"{unit} broke after {n} photos: it has {frames}, so 0 to "
                            f"{frames - 1}")
        after[unit] = n
    steps, discarded = _pilot_steps(plan, after)
    return Substitution(
        steps=steps,
        units={"pilot": tuple(u for u in plan.units if u not in after)},
        damage={u: d for u, d in plan.damage.items() if u not in after},
        discarded=discarded,
        replaced=dict.fromkeys(after),
    )
