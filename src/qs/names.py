"""The frame-name rule (docs/12-data-model.md §2) — the one place it lives.

    <unit>_<state>_<geometry>_r<rotation>[_pass<k>].png

The name is the unit's identity: split by unit and aggregate by unit are enforced from it, so
`qs shoot` writes names with `format_name` and `qs build` (P08) reads them with `parse_name`.
Anything outside the rule is refused, with the offending part named.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUFFIX = ".png"  # dataset frames are PNG, never JPEG (docs/12 §2)
GEOMETRIES = ("G1", "G2", "G3")
ROTATIONS = (0, 1, 2)  # 0°, 120°, 240°
DEGREES = {0: 0, 1: 120, 2: 240}
PASSES = (1, 2, 3)  # controls only: start, middle, end

# Unit prefix → (digits, the states that unit can be shot in). The prefix says what the unit
# is for: a dataset unit is shot good and damaged, a control is never damaged, a placement
# frame shows no unit or a unit at the edge.
_UNITS = {
    "u": (3, ("good", "dmg")),  # dataset
    "c": (2, ("good",)),  # control — every control frame carries its pass
    "p": (2, ("good", "dmg")),  # pilot
    "x": (2, ("empty", "edge")),  # placement
}
_UNIT = re.compile(r"([a-z])(\d+)")


class FrameNameError(ValueError):
    """A frame name outside the rule; `part` names what is wrong.

    Not a `QSError`: those are the Pipeline's NO_VERDICT reasons, and this is a tool's refusal.
    """

    def __init__(self, part: str, detail: str) -> None:
        super().__init__(f"{part}: {detail}")
        self.part = part


@dataclass(frozen=True, slots=True)
class FrameName:
    """The parts of one frame name. `pass_` is set for a control frame and only for one."""

    unit: str
    state: str
    geometry: str
    rotation: int
    pass_: int | None

    @property
    def is_control(self) -> bool:
        return self.unit.startswith("c")


def _check(name: FrameName) -> None:
    match = _UNIT.fullmatch(name.unit)
    if match is None or match.group(1) not in _UNITS:
        raise FrameNameError("unit", f"{name.unit!r} is not uNNN, cNN, pNN or xNN")
    digits, states = _UNITS[match.group(1)]
    if len(match.group(2)) != digits or int(match.group(2)) == 0:
        raise FrameNameError(
            "unit", f"{name.unit!r}: {match.group(1)} takes {digits} digits, from 1"
        )
    if name.state not in states:
        raise FrameNameError(
            "state", f"{name.state!r} for unit {name.unit}: one of {', '.join(states)}"
        )
    if name.geometry not in GEOMETRIES:
        raise FrameNameError("geometry", f"{name.geometry!r}: one of {', '.join(GEOMETRIES)}")
    if name.rotation not in ROTATIONS:
        raise FrameNameError("rotation", f"r{name.rotation}: one of r0, r1, r2")
    if name.is_control:
        if name.pass_ not in PASSES:
            raise FrameNameError(
                "pass", f"control {name.unit} needs pass1, pass2 or pass3, not {name.pass_}"
            )
    elif name.pass_ is not None:
        raise FrameNameError("pass", f"pass{name.pass_} on {name.unit}: controls only")


def format_name(name: FrameName) -> str:
    """The file name for these parts; refuses parts outside the rule."""
    _check(name)
    tail = "" if name.pass_ is None else f"_pass{name.pass_}"
    return f"{name.unit}_{name.state}_{name.geometry}_r{name.rotation}{tail}{SUFFIX}"


def parse_name(file_name: str) -> FrameName:
    """The parts of a file name; refuses a name outside the rule, naming the part."""
    if not file_name.endswith(SUFFIX):
        raise FrameNameError("suffix", f"{file_name!r} is not a {SUFFIX} file")
    parts = file_name[: -len(SUFFIX)].split("_")
    if len(parts) not in (4, 5):
        raise FrameNameError(
            "parts", f"{file_name!r}: <unit>_<state>_<geometry>_r<rotation>[_pass<k>]"
        )
    unit, state, geometry, rotation = parts[:4]
    if re.fullmatch(r"r\d", rotation) is None:
        raise FrameNameError("rotation", f"{rotation!r}: one of r0, r1, r2")
    pass_: int | None = None
    if len(parts) == 5:
        if re.fullmatch(r"pass\d", parts[4]) is None:
            raise FrameNameError("pass", f"{parts[4]!r}: pass1, pass2 or pass3")
        pass_ = int(parts[4][4:])
    name = FrameName(unit, state, geometry, int(rotation[1:]), pass_)
    _check(name)
    return name
