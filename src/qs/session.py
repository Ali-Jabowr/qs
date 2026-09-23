"""The session folder `raw/<session>/`: the only code that writes into it.

- `plan.yml`, the split plan as data, and `plan.md`, its printable tray sheet. A plan is written
  once — never overwritten, because units already sitting in trays must not be silently
  reassigned — and read back through `Plan`, which refuses a unit in two splits (FR-52).
- `shots-<geometry>.md`, the numbered photo list for the phone path (ADR-0022), written once:
  the paper in Ali's hand and `qs import-phone` must read the same order.
- The frames, PNG under the name rule (docs/12 §2), each written once and never overwritten.
- `frames.csv`, one row per frame as it is saved (assumption A2, #29): what was recorded at the
  shot cannot be recovered after it (FR-15, docs/11 §6).
- `session.yml` (docs/13 §8), written once at the end of the session.
- `rejected/`, the spoiled shots of a redo, never deleted and never overwritten (A1, #28).
- From `qs import-phone` only (ADR-0022): `phone/`, the phone's original files byte for byte,
  and `import.csv`, the manifest from each original to its frame.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, TextIO

import cv2
import numpy as np
import yaml

from qs.contracts import CameraReadback, Frame, TriggerEvent
from qs.names import FrameName, format_name
from qs.splitplan import PLACEMENT_EDGE, PLACEMENT_EMPTY, Plan, PlanError, Substitution

PLAN_FILE = "plan.yml"
SHEET_FILE = "plan.md"
FRAMES_FILE = "frames.csv"
SESSION_FILE = "session.yml"
IMPORT_FILE = "import.csv"
REJECTED_DIR = "rejected"
PHONE_DIR = "phone"
FRAMES_COLUMNS = (
    "file", "unit", "state", "geometry", "rotation", "pass", "wall_utc", "grabbed_at",
    "capture_ms", "reference_patch", "exposure_us", "gain", "wb_kelvin", "focus", "verified",
    "sha256", "status",
)
IMPORT_COLUMNS = (
    "original", "original_sha256", "exif_time", "frame_file", "png_sha256", "status",
)
_SPLITS = ("control", "train", "calib", "test", "spare")
_HEADER = "# The split plan, written by `qs split-plan`. Never edited by hand.\n"


class PlanExists(Exception):
    """The session already has a plan; it is left as it is."""


class ShotListDiffers(Exception):
    """The session has a shot list for this geometry that differs; it is left as it is."""


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


def write_shot_list(folder: Path, geometry: str, text: str) -> tuple[Path, bool]:
    """Write `shots-<geometry>.md` into `folder` once; the path, and whether it was written.

    A list already on disk is never replaced: byte-identical → `(path, False)`; different →
    `ShotListDiffers`, because a printed copy may already be at the bench and the import maps
    photo k to row k.
    """
    path = folder / f"shots-{geometry}.md"
    data = text.encode("utf-8")
    try:
        with path.open("xb") as out:  # exclusive: never overwrites
            out.write(data)
    except FileExistsError:
        if path.read_bytes() == data:
            return path, False
        raise ShotListDiffers(
            f"{path} exists and differs from the plan's list — left untouched; a printed copy "
            "may already be in use"
        ) from None
    return path, True


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
    """True if the session already has a frame, a log, a `session.yml`, a rejected shot or
    an imported original."""
    names = (FRAMES_FILE, SESSION_FILE, IMPORT_FILE, REJECTED_DIR, PHONE_DIR)
    return any((folder / name).exists() for name in names) or any(folder.glob("*.png"))


def _write_png(path: Path, pixels: np.ndarray) -> str:
    """Encode in memory and write with exclusive create; the SHA-256 of the bytes written.

    The same OpenCV PNG encoder as `cv2.imwrite`, but a file already on disk raises
    `FileExistsError` instead of being replaced: a raw original is never overwritten.
    """
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError(f"{path.name}: OpenCV could not encode the frame as PNG")
    data = encoded.tobytes()
    with path.open("xb") as out:
        out.write(data)
    return hashlib.sha256(data).hexdigest()


def save_frame(folder: Path, name: FrameName, pixels: np.ndarray) -> str:
    """Write the frame as a lossless PNG under its rule name; its SHA-256 as hex."""
    return _write_png(folder / format_name(name), pixels)


def rejected_file(name: FrameName, nth: int) -> str:
    """The path, relative to the session, of the nth spoiled take of a shot (A1, #28):
    `rejected/<name>.png`, then `_2`, `_3` … for a second or third redo of the same shot."""
    stem = format_name(name)[: -len(".png")]
    return f"{REJECTED_DIR}/{stem}{'' if nth == 1 else f'_{nth}'}.png"


def save_rejected(folder: Path, name: FrameName, nth: int, pixels: np.ndarray) -> tuple[str, str]:
    """Write the nth spoiled take of a shot into `rejected/`; its relative path and SHA-256."""
    file = rejected_file(name, nth)
    (folder / REJECTED_DIR).mkdir(exist_ok=True)
    return file, _write_png(folder / file, pixels)


def copy_original(folder: Path, original: Path) -> str:
    """Copy a phone file byte for byte into `phone/`, never over another; its SHA-256."""
    data = original.read_bytes()
    (folder / PHONE_DIR).mkdir(exist_ok=True)
    with (folder / PHONE_DIR / original.name).open("xb") as out:
        out.write(data)
    return hashlib.sha256(data).hexdigest()


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class Recorded(Protocol):
    """What the frame log and `session.yml` keep of a frame. A `Frame` is one; so is a phone
    photo's record, which has no pixels in memory and no kernel timestamps (ADR-0022)."""

    @property
    def event(self) -> TriggerEvent: ...

    @property
    def reference_patch(self) -> float | None: ...

    @property
    def camera(self) -> CameraReadback: ...


class _CsvLog:
    """A CSV created with its header — never over an existing one — and flushed per row."""

    def __init__(self, path: Path, columns: tuple[str, ...]) -> None:
        self._file: TextIO = path.open("x", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(columns)
        self._file.flush()

    def _row(self, values: tuple[object, ...]) -> None:
        self._writer.writerow(_cell(v) for v in values)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class FrameLog(_CsvLog):
    """`frames.csv`: created with its header, one row appended and flushed per frame."""

    def __init__(self, folder: Path) -> None:
        super().__init__(folder / FRAMES_FILE, FRAMES_COLUMNS)

    def append(self, name: FrameName, frame: Frame, sha256: str, status: str) -> None:
        self.append_row(format_name(name), name, frame, frame.grabbed_at, frame.capture_ms,
                        sha256, status)

    def append_row(
        self,
        file: str,
        name: FrameName,
        frame: Recorded,
        grabbed_at: float | None,
        capture_ms: float | None,
        sha256: str,
        status: str,
    ) -> None:
        """One row; `file` is relative to the session (`rejected/…` for a spoiled take)."""
        camera = frame.camera
        self._row((
            file, name.unit, name.state, name.geometry, name.rotation, name.pass_,
            frame.event.wall.isoformat(), grabbed_at, capture_ms,
            frame.reference_patch, camera.exposure_us, camera.gain, camera.wb_kelvin,
            camera.focus, camera.verified, sha256, status,
        ))

    def __enter__(self) -> FrameLog:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ImportLog(_CsvLog):
    """`import.csv` (ADR-0022): one row per phone photo, in import order."""

    def __init__(self, folder: Path) -> None:
        super().__init__(folder / IMPORT_FILE, IMPORT_COLUMNS)

    def append(self, original: str, original_sha256: str, exif_time: datetime,
               frame_file: str, png_sha256: str, status: str) -> None:
        self._row((original, original_sha256, exif_time.isoformat(), frame_file, png_sha256,
                   status))

    def __enter__(self) -> ImportLog:
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
    """What `session.yml` records; `frames` are the saved frames, first to last.

    `substitution`, when units broke, gives the splits as shot and the `discarded` units
    (A1, #28); without it the splits are the plan's.
    """

    session_id: str
    geometry: str
    product: Product
    device: str
    camera_at_open: CameraReadback | None
    plan: Plan
    frames: tuple[Recorded, ...]
    substitution: Substitution | None = None


def _session_data(record: SessionRecord) -> dict[str, Any]:
    product = record.product
    about: dict[str, Any] = (
        {"brand": product.brand} if product.brand is not None else {"client_product": True}
    )
    about |= {"flavour": product.flavour, "shop": product.shop,
              "best_before": product.best_before}
    at_open = record.camera_at_open
    first, last = record.frames[0], record.frames[-1]
    shot = record.substitution
    units: dict[str, Any] = {
        name: list(getattr(record.plan, name) if shot is None else shot.units[name])
        for name in _SPLITS
    }
    units["placement"] = [*PLACEMENT_EMPTY, *PLACEMENT_EDGE]
    damage = record.plan.damage if shot is None else shot.damage
    discarded = {} if shot is None else {unit: list(files)
                                          for unit, files in shot.discarded.items()}
    data: dict[str, Any] = {
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
        "damage": dict(damage),
    }
    if discarded:  # only when a unit broke (A1, #28)
        data["discarded"] = discarded
    data["notes"] = ""
    return data


def write_session(folder: Path, record: SessionRecord, by: str = "qs shoot") -> Path:
    """Write `session.yml` (docs/13 §8) once; refuses if it exists or no frame was saved."""
    if not record.frames:
        raise ValueError("a session.yml needs at least one saved frame")
    path = folder / SESSION_FILE
    text = f"# Written by `{by}`. Only `notes` may be filled by hand, before the backup.\n"
    text += yaml.safe_dump(
        _session_data(record), sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    with path.open("x", encoding="utf-8") as out:  # exclusive: never overwrites
        out.write(text)
    return path
