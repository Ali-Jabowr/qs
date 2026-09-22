"""The session folder `raw/<session>/`: the only code that writes into it.

- `plan.yml`, the split plan as data, and `plan.md`, its printable tray sheet. A plan is written
  once — never overwritten, because units already sitting in trays must not be silently
  reassigned — and read back through `Plan`, which refuses a unit in two splits (FR-52).
- The frames, PNG under the name rule (docs/12 §2), each written once and never overwritten.
- `frames.csv`, one row per frame as it is saved (assumption A2, #29): what was recorded at the
  shot cannot be recovered after it (FR-15, docs/11 §6).
- `session.yml` (docs/13 §8), written once at the end of the session.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import cv2
import numpy as np
import yaml

from qs.contracts import CameraReadback, Frame
from qs.names import FrameName, format_name
from qs.splitplan import PLACEMENT_EDGE, PLACEMENT_EMPTY, Plan, PlanError

PLAN_FILE = "plan.yml"
SHEET_FILE = "plan.md"
FRAMES_FILE = "frames.csv"
SESSION_FILE = "session.yml"
FRAMES_COLUMNS = (
    "file", "unit", "state", "geometry", "rotation", "pass", "wall_utc", "grabbed_at",
    "capture_ms", "reference_patch", "exposure_us", "gain", "wb_kelvin", "focus", "verified",
    "sha256", "status",
)
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


def session_started(folder: Path) -> bool:
    """True if the session already has a frame, a frame log or a `session.yml`."""
    return any((folder / name).exists() for name in (FRAMES_FILE, SESSION_FILE)) or any(
        folder.glob("*.png")
    )


def save_frame(folder: Path, name: FrameName, pixels: np.ndarray) -> str:
    """Write the frame as a lossless PNG under its rule name; its SHA-256 as hex.

    Encoded in memory and written with exclusive create — the same OpenCV PNG encoder as
    `cv2.imwrite`, but a file already on disk raises `FileExistsError` instead of being
    replaced: a raw original is never overwritten.
    """
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError(f"{format_name(name)}: OpenCV could not encode the frame as PNG")
    data = encoded.tobytes()
    with (folder / format_name(name)).open("xb") as out:
        out.write(data)
    return hashlib.sha256(data).hexdigest()


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class FrameLog:
    """`frames.csv`: created with its header, one row appended and flushed per frame."""

    def __init__(self, folder: Path) -> None:
        self._file: TextIO = (folder / FRAMES_FILE).open("x", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(FRAMES_COLUMNS)
        self._file.flush()

    def append(self, name: FrameName, frame: Frame, sha256: str, status: str) -> None:
        camera = frame.camera
        self._writer.writerow(_cell(v) for v in (
            format_name(name), name.unit, name.state, name.geometry, name.rotation, name.pass_,
            frame.event.wall.isoformat(), frame.grabbed_at, frame.capture_ms,
            frame.reference_patch, camera.exposure_us, camera.gain, camera.wb_kelvin,
            camera.focus, camera.verified, sha256, status,
        ))
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> FrameLog:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class Product:
    """The моти shot, given before the first frame (assumption A3, #30).

    `brand` is None when the моти is the client's own: `session.yml` then says
    `client_product: true` instead (docs/17 §5).
    """

    brand: str | None
    flavour: str
    shop: str
    best_before: date
    purchased_at: datetime


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """What `session.yml` records; `frames` are the saved frames, first to last."""

    session_id: str
    geometry: str
    product: Product
    device: str
    camera_at_open: CameraReadback | None
    plan: Plan
    frames: tuple[Frame, ...]


def _session_data(record: SessionRecord) -> dict[str, Any]:
    product = record.product
    about: dict[str, Any] = (
        {"brand": product.brand} if product.brand is not None else {"client_product": True}
    )
    about |= {"flavour": product.flavour, "shop": product.shop,
              "best_before": product.best_before}
    at_open = record.camera_at_open
    first, last = record.frames[0], record.frames[-1]
    units: dict[str, Any] = {
        name: list(getattr(record.plan, name))
        for name in ("control", "train", "calib", "test", "spare")
    }
    units["placement"] = [*PLACEMENT_EMPTY, *PLACEMENT_EDGE]
    return {
        "session_id": record.session_id,
        "kind": "dataset",
        "geometries": [record.geometry],
        "product": about,
        "purchased_at": product.purchased_at,
        "started_at": first.event.wall,
        "ended_at": last.event.wall,
        "camera": {
            "device": record.device,
            "exposure_us": None if at_open is None else at_open.exposure_us,
            "gain": None if at_open is None else at_open.gain,
            "wb_kelvin": None if at_open is None else at_open.wb_kelvin,
            "focus": None if at_open is None else at_open.focus,
            # every saved frame's read-back verified; one unreadable setting says false
            "verified": all(frame.camera.verified for frame in record.frames),
        },
        "reference_patch": {"start": first.reference_patch, "end": last.reference_patch},
        "split_seed": record.plan.seed,
        "units": units,
        "damage": dict(record.plan.damage),
        "notes": "",
    }


def write_session(folder: Path, record: SessionRecord) -> Path:
    """Write `session.yml` (docs/13 §8) once; refuses if it exists or no frame was saved."""
    if not record.frames:
        raise ValueError("a session.yml needs at least one saved frame")
    path = folder / SESSION_FILE
    text = "# Written by `qs shoot`. Only `notes` may be filled by hand, before the backup.\n"
    text += yaml.safe_dump(
        _session_data(record), sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    with path.open("x", encoding="utf-8") as out:  # exclusive: never overwrites
        out.write(text)
    return path
