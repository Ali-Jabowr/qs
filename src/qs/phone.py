"""Phone photos for the dataset (ADR-0022): EXIF read, lock check, ordering and mapping.

The phone shoots offline and is not a `FrameSource`. Its EXIF is the per-frame read-back that
comes after the shoot: every photo must carry the same exposure time, ISO, white-balance mode
(manual), focal length, make, model, orientation and frame size, with no digital zoom. A missing
tag counts as a difference. Photo k, in capture-time order, is shot k of the printed list.

EXIF is read with Pillow only (already pinned); nothing here touches the network. Reading a
photo is the only I/O; the checks and the mapping are pure.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from qs.config import CameraConfig
from qs.session import rejected_file
from qs.splitplan import Prompt, Shot, Step

JPEG = (".jpg", ".jpeg")
# Formats a phone may write that the import refuses: HEIC and DNG need a new dependency, a
# video is not a photo (ADR-0022, #51 item 2).
REFUSED = {
    ".heic": "HEIC", ".heif": "HEIF", ".dng": "DNG", ".mp4": "video", ".mov": "video",
    ".3gp": "video", ".mkv": "video", ".avi": "video", ".webm": "video",
}
PATCH_LIMIT = 0.01  # the grey card varies by less than 1% (docs/13 §4, camera-test's limit)

# EXIF tags, IFD0 and the Exif sub-IFD.
_EXIF_IFD = 0x8769
_MAKE, _MODEL, _ORIENTATION = 0x010F, 0x0110, 0x0112
_EXPOSURE_TIME, _ISO = 0x829A, 0x8827  # ISO: PhotographicSensitivity, once ISOSpeedRatings
_DATETIME_ORIGINAL, _OFFSET_ORIGINAL, _SUBSEC_ORIGINAL = 0x9003, 0x9011, 0x9291
_FOCAL_LENGTH, _WHITE_BALANCE, _DIGITAL_ZOOM = 0x920A, 0xA403, 0xA404
_OFFSET = re.compile(r"([+-])(\d{2}):?(\d{2})")


class PhoneRefused(Exception):
    """The photos cannot become this session: the import writes nothing (exit 1)."""


class Unsupported(Exception):
    """A file in the phone folder the import cannot read at all (exit 2)."""


@dataclass(frozen=True, slots=True)
class Photo:
    """One phone photo as its EXIF describes it. `None` is a tag that is missing or unreadable;
    `size` is (width, height) with the EXIF orientation applied."""

    path: Path
    taken: datetime | None  # aware
    time_problem: str | None
    exposure_s: Fraction | None
    iso: int | None
    white_balance: int | None  # 0 auto, 1 manual
    focal_mm: Fraction | None
    zoom: Fraction | str | None  # None: no DigitalZoomRatio tag; str: a tag not a ratio
    make: str | None
    model: str | None
    orientation: int | None
    size: tuple[int, int]

    @property
    def name(self) -> str:
        return self.path.name


def list_photos(folder: Path) -> list[Path]:
    """The JPEGs in `folder`, not recursive, by name; `Unsupported` if a HEIC, DNG or video
    is there (ADR-0022). Other files are left alone."""
    photos: list[Path] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in REFUSED:
            raise Unsupported(
                f"{path.name} is {REFUSED[suffix]} — the import reads JPEG only (ADR-0022): "
                "set the phone to JPEG at maximum quality"
            )
        if suffix in JPEG:
            photos.append(path)
    return photos


def utc_offset(text: str) -> timezone:
    """A UTC offset written ±HH:MM."""
    match = _OFFSET.fullmatch(text.strip())
    if match is None or int(match[3]) >= 60 or int(match[2]) > 14:
        raise ValueError(f"{text!r}: a UTC offset as ±HH:MM, e.g. +03:00")
    delta = timedelta(hours=int(match[2]), minutes=int(match[3]))
    return timezone(-delta if match[1] == "-" else delta)


def _fraction(value: object) -> Fraction | None:
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(value, bool) or not isinstance(numerator, int) or not isinstance(
        denominator, int
    ) or denominator == 0:
        return None
    return Fraction(numerator, denominator)


def _integer(value: object) -> int | None:
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip("\x00 \t") or None


def _taken(sub: dict[int, object], fallback: tzinfo | None) -> tuple[datetime | None, str | None]:
    raw = _text(sub.get(_DATETIME_ORIGINAL))
    if raw is None:
        return None, "no capture time (DateTimeOriginal)"
    try:
        local = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None, f"capture time {raw!r} unreadable"
    subsec = _text(sub.get(_SUBSEC_ORIGINAL))
    if subsec is not None and subsec.isdigit():
        local = local.replace(microsecond=int(subsec[:6].ljust(6, "0")))
    offset_text = _text(sub.get(_OFFSET_ORIGINAL))
    zone: tzinfo | None = fallback
    if offset_text is not None:
        try:
            zone = utc_offset(offset_text)
        except ValueError:
            return None, f"capture offset {offset_text!r} unreadable"
    if zone is None:
        return None, "no UTC offset in the EXIF (OffsetTimeOriginal) — give --utc-offset"
    return local.replace(tzinfo=zone), None


def read_photo(path: Path, fallback_offset: tzinfo | None = None) -> Photo:
    """The photo's EXIF and decoded size; `PhoneRefused` if Pillow cannot open it."""
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise PhoneRefused(f"{path.name}: not a JPEG ({image.format})")
            exif = image.getexif()
            sub = dict(exif.get_ifd(_EXIF_IFD))
            width, height = image.size
    except (OSError, SyntaxError, ValueError) as exc:
        raise PhoneRefused(f"{path.name}: unreadable — {exc}") from None
    orientation = _integer(exif.get(_ORIENTATION))
    if orientation is not None and not 1 <= orientation <= 8:
        orientation = None
    if orientation in (5, 6, 7, 8):  # a quarter turn: width and height swap
        width, height = height, width
    zoom_value = sub.get(_DIGITAL_ZOOM)
    zoom: Fraction | str | None = None
    if zoom_value is not None:
        zoom = _fraction(zoom_value)
        if zoom is None and getattr(zoom_value, "numerator", None) == 0:
            zoom = Fraction(0)  # 0/0: EXIF's "digital zoom not used"
        elif zoom is None:
            zoom = repr(zoom_value)
    taken, problem = _taken(sub, fallback_offset)
    return Photo(
        path=path,
        taken=taken,
        time_problem=problem,
        exposure_s=_fraction(sub.get(_EXPOSURE_TIME)),
        iso=_integer(sub.get(_ISO)),
        white_balance=_integer(sub.get(_WHITE_BALANCE)),
        focal_mm=_fraction(sub.get(_FOCAL_LENGTH)),
        zoom=zoom,
        make=_text(exif.get(_MAKE)),
        model=_text(exif.get(_MODEL)),
        orientation=orientation,
        size=(width, height),
    )


def decode(path: Path) -> np.ndarray:
    """The photo as 8-bit, 3-channel BGR — as `qs shoot`'s frames are — with the EXIF
    orientation applied exactly once."""
    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image)
        rgb = upright.convert("RGB")
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[:, :, ::-1])


def in_order(photos: Sequence[Photo]) -> list[Photo]:
    """The photos by capture time, then file name. `PhoneRefused` if a photo has no usable
    capture time, or if time order and file-name order disagree (both files named)."""
    timeless = [p for p in photos if p.taken is None]
    if timeless:
        raise PhoneRefused(
            "capture time: " + "; ".join(f"{p.name}: {p.time_problem}" for p in timeless)
        )
    ordered = sorted(photos, key=lambda p: (p.taken, p.name))
    for before, after in zip(ordered, ordered[1:], strict=False):
        if after.name < before.name:
            raise PhoneRefused(
                f"order: {after.name} was taken after {before.name} but its name sorts before "
                "it — time order and file-name order disagree; the mapping would be a guess"
            )
    return ordered


# --- the lock -----------------------------------------------------------------------------


def _exposure(value: object) -> str:
    if not isinstance(value, Fraction):
        return "missing"
    us = value * 1_000_000
    shown = f"{us.numerator}" if us.denominator == 1 else f"{float(us):.2f}"
    seconds = f"1/{value.denominator}" if value.numerator == 1 else f"{float(value):g}"
    return f"{seconds} s ({shown} µs)"


def _plain(value: object) -> str:
    if value is None:
        return "missing"
    if isinstance(value, Fraction):
        return f"{float(value):g}"
    if isinstance(value, tuple):
        return "x".join(str(v) for v in value)
    return str(value)


def _white_balance(value: object) -> str:
    return {0: "auto (0)", 1: "manual (1)"}.get(value, _plain(value)) if isinstance(
        value, int
    ) else "missing"


_FIELDS: tuple[tuple[str, Callable[[Photo], object], Callable[[object], str]], ...] = (
    ("exposure time", lambda p: p.exposure_s, _exposure),
    ("ISO", lambda p: p.iso, _plain),
    ("white balance", lambda p: p.white_balance, _white_balance),
    ("focal length", lambda p: p.focal_mm,
     lambda v: "missing" if v is None else f"{_plain(v)} mm"),
    ("make", lambda p: p.make, _plain),
    ("model", lambda p: p.model, _plain),
    ("orientation", lambda p: p.orientation, _plain),
    ("frame size", lambda p: p.size, _plain),
)


def _where(names: Sequence[str], total: int) -> str:
    if len(names) == total and total > 1:
        return f"all {total} photos"
    shown = ", ".join(names[:10])
    return shown if len(names) <= 10 else f"{shown} and {len(names) - 10} more"


def _grouped(pairs: Sequence[tuple[Photo, object]], show: Callable[[object], str],
             total: int) -> str:
    by_value: dict[str, list[str]] = {}
    for photo, value in pairs:
        by_value.setdefault(show(value), []).append(photo.name)
    return "; ".join(f"{value} on {_where(names, total)}" for value, names in by_value.items())


@dataclass(frozen=True, slots=True)
class LockResult:
    """The EXIF lock over a set of photos: one PASS line, or a FAIL line per failure."""

    passed: bool
    lines: tuple[str, ...]


def check_lock(photos: Sequence[Photo], camera: CameraConfig) -> LockResult:
    """The EXIF lock (ADR-0022) over every photo, against the config's camera."""
    total = len(photos)
    fails: list[str] = []
    for label, get, show in _FIELDS:
        values = [get(p) for p in photos]
        majority = Counter(values).most_common(1)[0][0]
        odd = [(p, v) for p, v in zip(photos, values, strict=True)
               if v is None or v != majority]
        if odd:
            head = (f"{label}: missing" if majority is None else
                    f"{label}: {show(majority)} on {total - len(odd)} of {total}")
            fails.append(f"FAIL {head} — differs: {_grouped(odd, show, total)}")

    flicker = Fraction(1, 2 * camera.power_line_hz)  # 10 ms at 50 Hz (docs/13 §3)
    wanted = Fraction(camera.exposure_us, 1_000_000)
    bad: list[tuple[Photo, object]] = [
        (p, p.exposure_s) for p in photos
        if p.exposure_s is not None
        and ((p.exposure_s / flicker).denominator != 1 or p.exposure_s != wanted)
    ]
    if bad:
        fails.append(
            f"FAIL exposure time: must be a whole multiple of {float(flicker) * 1000:g} ms "
            f"({camera.power_line_hz} Hz flicker) and equal camera.exposure_us "
            f"{camera.exposure_us} µs — {_grouped(bad, _exposure, total)}"
        )
    bad = [(p, p.white_balance) for p in photos
           if p.white_balance is not None and p.white_balance != 1]
    if bad:
        fails.append(
            "FAIL white balance: must be manual (1), a fixed preset or kelvin value — "
            f"{_grouped(bad, _white_balance, total)}"
        )
    bad = [(p, p.zoom) for p in photos
           if p.zoom is not None and (isinstance(p.zoom, str) or p.zoom not in (0, 1))]
    if bad:
        fails.append(
            "FAIL digital zoom: DigitalZoomRatio must be absent, 0 or 1 (main 1× lens only) — "
            f"{_grouped(bad, _plain, total)}"
        )
    size = (camera.width, camera.height)
    bad = [(p, p.size) for p in photos if p.size != size]
    if bad:
        fails.append(
            f"FAIL frame size: must be camera.width x height {size[0]}x{size[1]}, the pixels "
            f"the ROIs are in — {_grouped(bad, _plain, total)}"
        )
    if fails:
        return LockResult(False, tuple(fails))
    first = photos[0]
    return LockResult(True, (
        f"PASS EXIF lock: {total} photos · {_exposure(first.exposure_s)} · ISO {first.iso} · "
        f"WB manual · {_plain(first.focal_mm)} mm · {first.make} {first.model} · "
        f"orientation {first.orientation} · {_plain(first.size)}",
    ))


def patch_variation(readings: Sequence[float]) -> float:
    """(max − min) / mean of the grey-card readings — `qs camera-test`'s formula."""
    mean = sum(readings) / len(readings)
    return (max(readings) - min(readings)) / mean if mean > 0 else float("inf")


# --- the mapping --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mapped:
    """Photo `index` (import order) is a take of `shot`, written as `file`, relative to the
    session. `status` is saved, rejected (a redo's spoiled take) or discarded (a broken
    unit's frame, A1)."""

    index: int
    shot: Shot
    number: int  # the shot's number in the adjusted list, from 1
    file: str
    status: str
    nth: int = 0  # a rejected take: the first, second … spoiled take of its shot


@dataclass(frozen=True, slots=True)
class Mapping:
    photos: tuple[Mapped, ...]
    shots_taken: int
    shots_total: int
    expected_photos: int


def map_photos(
    count: int, steps: Sequence[Step], redo: Sequence[str], discarded: frozenset[str]
) -> Mapping:
    """Photo k is shot k. `redo` names a shot's frame file once per spoiled take — a file,
    not a number, because the printed rows and the shots as shot differ once a unit breaks:
    the first photo of such a shot goes to `rejected/`, and the next is the shot.
    `PhoneRefused` for a redo of a shot that is not in `steps`. Fewer photos than the list
    map partly; the caller decides."""
    shots = [step for step in steps if isinstance(step, Shot)]
    spoiled = Counter(redo)
    wrong = sorted(set(spoiled) - {shot.file for shot in shots})
    if wrong:
        raise PhoneRefused(f"--redo {wrong[0]}: no such shot in this session")
    mapped: list[Mapped] = []
    taken = 0
    for number, shot in enumerate(shots, start=1):
        for nth in range(1, spoiled[shot.file] + 1):
            if len(mapped) == count:
                break
            mapped.append(Mapped(len(mapped), shot, number, rejected_file(shot.frame, nth),
                                 "rejected", nth))
        if len(mapped) == count:
            break
        status = "discarded" if shot.file in discarded else "saved"
        mapped.append(Mapped(len(mapped), shot, number, shot.file, status))
        taken = number
    return Mapping(tuple(mapped), taken, len(shots), len(shots) + sum(spoiled.values()))


def next_step(steps: Sequence[Step], taken: int) -> tuple[list[Prompt], Shot | None]:
    """The prompts before shot `taken + 1`, and that shot (None when every shot is taken)."""
    prompts: list[Prompt] = []
    seen = 0
    for step in steps:
        if isinstance(step, Prompt):
            if seen == taken:
                prompts.append(step)
            continue
        seen += 1
        if seen == taken + 1:
            return prompts, step
    return prompts, None


def gap_warnings(mapping: Mapping, photos: Sequence[Photo]) -> list[str]:
    """A test unit whose last-good → first-dmg gap is not its largest: where a slip in the
    order usually shows (ADR-0022). Warnings only."""
    by_unit: dict[str, list[tuple[Mapped, datetime]]] = {}
    for entry in mapping.photos:
        taken = photos[entry.index].taken
        if entry.status == "saved" and taken is not None:
            by_unit.setdefault(entry.shot.frame.unit, []).append((entry, taken))
    lines: list[str] = []
    for unit, entries in by_unit.items():
        states = [entry.shot.frame.state for entry, _ in entries]
        if "dmg" not in states or "good" not in states:
            continue
        gaps = [(b[1] - a[1], a[0], b[0]) for a, b in zip(entries, entries[1:], strict=False)]
        at = states.index("dmg") - 1
        damage_gap = gaps[at][0]
        widest = max(gaps, key=lambda gap: gap[0])
        if damage_gap < widest[0]:
            lines.append(
                f"WARNING {unit}: {damage_gap.total_seconds():g} s between its last good and "
                f"first dmg photo, but {widest[0].total_seconds():g} s between "
                f"{photos[widest[1].index].name} and {photos[widest[2].index].name} — check "
                "the order there"
            )
    return lines
