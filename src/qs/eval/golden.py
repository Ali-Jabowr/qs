"""The dataset build and the golden set (docs/12 §2, docs/13 §6; FR-52, FR-56).

`plan_build` is pure: one raw session in, the planned dataset tree and golden set out — or a
`BuildRefused` naming the unit, file or field. Every check runs before the first write, so a
refused build leaves nothing half-built and `DIR` is never touched. `write_build` copies each
tree into a temporary directory beside its target and renames it into place, so an existing
target is never overwritten: identical content → "already built", different → refused.
`verify_golden` checks a golden set on disk — the same check `qs evaluate` (ticket 3) runs
before it scores a frame.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from operator import attrgetter
from pathlib import Path
from typing import Any

import yaml

from qs.names import FrameName, FrameNameError, parse_name
from qs.session import FRAMES_FILE, PLAN_FILE, SESSION_FILE, read_plan
from qs.splitplan import DAMAGES, PLACEMENT_EDGE, PLACEMENT_EMPTY, Plan, PlanError

MANIFEST_COLUMNS = ("file", "unit", "role", "state", "class", "geometry", "rotation",
                    "pass", "sha256")
HASH_FILE = "manifest.sha256"
FRAMES_DIR = "frames"
ROLES = ("calib", "test", "control", "placement")
NAME_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_SPLITS = ("control", "train", "calib", "test", "spare", "placement")
_PREFIX = {"control": "c", "train": "u", "calib": "u", "test": "u", "spare": "u",
           "placement": "x"}
_CLASS_OF = {"dent": "dent", "tear": "tear", "deformed-flat": "deformed",
             "deformed-oval": "deformed"}


class BuildRefused(ValueError):
    """The session cannot be built: refused, naming the unit, file or field. Nothing written."""


class GoldenRefused(ValueError):
    """The golden set does not match its manifest: refused, naming the cause."""


@dataclass(frozen=True, slots=True)
class Planned:
    """One file a build writes: `rel` under its tree's root, its bytes — a byte copy of
    `source` when it is set, the literal `data` otherwise — and its mode (frames 0444)."""

    rel: str
    source: Path | None
    data: bytes | None
    sha256: str
    mode: int

    def bytes(self) -> bytes:
        if self.source is None:
            assert self.data is not None
            return self.data
        return self.source.read_bytes()


@dataclass(frozen=True, slots=True)
class Tree:
    """One target tree: its root under `--data` and every file planned into it."""

    root: Path
    files: tuple[Planned, ...]


@dataclass(frozen=True, slots=True)
class Build:
    """A planned build: both trees, the golden hash, the per-split counts, and whether each
    target is already there with identical content (nothing to write)."""

    session: Path
    geometry: str
    name: str
    dataset: Tree
    golden: Tree
    golden_hash: str
    dataset_done: bool
    golden_done: bool
    counts: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class _Session:
    """The parsed `session.yml`: the fields the build reads, as shot."""

    kind: str
    geometry: str
    started_at: datetime
    split_seed: int
    units: dict[str, tuple[str, ...]]
    damage: dict[str, str]
    discarded: frozenset[str]


def _session_yml(session: Path) -> dict[str, Any]:
    path = session / SESSION_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BuildRefused(f"{path}: {exc}") from None
    if not isinstance(data, dict):
        raise BuildRefused(f"{path}: not a session.yml")
    return data


def _strings(path: Path, data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise BuildRefused(f"{path}: {key} is a list of unit names")
    return tuple(value)


_KIND_WHY = {
    "pilot": "a pilot chose the lighting and was thrown away — it can never help measure it "
             "(docs/13 §5)",
    "gate_a": "Gate A is a separate session, shot and evaluated on its own (docs/14 §6)",
}


def _session(session_dir: Path) -> _Session:
    data = _session_yml(session_dir)
    path = session_dir / SESSION_FILE
    kind = data.get("kind")
    if kind != "dataset":
        if kind in _KIND_WHY:
            raise BuildRefused(f"{path}: kind {kind!r} — {_KIND_WHY[kind]}")
        raise BuildRefused(f"{path}: kind {kind!r} — one of pilot, dataset, gate_a")
    geometries = data.get("geometries")
    if (not isinstance(geometries, list) or not geometries
            or not all(isinstance(g, str) for g in geometries)):
        raise BuildRefused(f"{path}: geometries is a list of G1, G2, G3")
    if len(geometries) > 1:
        raise BuildRefused(
            f"{path}: geometries {', '.join(geometries)} — a dataset session is shot under "
            "one geometry; the pilot compares all three (docs/13 §5)"
        )
    started = data.get("started_at")
    if not isinstance(started, datetime):
        raise BuildRefused(f"{path}: started_at is a time")
    seed = data.get("split_seed")
    if type(seed) is not int:
        raise BuildRefused(f"{path}: split_seed is an integer")
    units_raw = data.get("units")
    if not isinstance(units_raw, dict) or sorted(units_raw) != sorted(_SPLITS):
        raise BuildRefused(f"{path}: units are {', '.join(_SPLITS)}")
    units = {name: _strings(path, units_raw, name) for name in _SPLITS}
    damage = data.get("damage")
    if not isinstance(damage, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in damage.items()
    ):
        raise BuildRefused(f"{path}: damage maps a test unit to its damage class")
    discarded_raw = data.get("discarded", {})
    if not isinstance(discarded_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, list) and all(isinstance(f, str) for f in v)
        for k, v in discarded_raw.items()
    ):
        raise BuildRefused(f"{path}: discarded maps a broken unit to its frame files")
    return _Session(
        kind="dataset",
        geometry=geometries[0],
        started_at=started,
        split_seed=seed,
        units=units,
        damage=dict(damage),
        discarded=frozenset(discarded_raw),
    )


def _check_splits(shot: _Session, plan: Plan) -> None:
    """The splits as shot against the plan: disjoint (E1), prefixes, membership (FR-52)."""
    seen: dict[str, str] = {}
    for name in _SPLITS:
        for unit in shot.units[name]:
            if unit in seen:
                raise BuildRefused(
                    f"{unit} is in {seen[unit]} and {name} — a unit is in exactly one "
                    "split (E1, FR-52)"
                )
            seen[unit] = name
    for name in _SPLITS:
        for unit in shot.units[name]:
            if not unit.startswith(_PREFIX[name]):
                raise BuildRefused(
                    f"{unit} is in {name}: a {name} unit starts with {_PREFIX[name]!r}"
                )
    for name in ("train", "calib", "test"):
        allowed = set(getattr(plan, name)) | set(plan.spare)
        for unit in shot.units[name]:
            if unit not in allowed:
                raise BuildRefused(
                    f"{unit} is in {name} but in neither the plan's {name} nor its spares"
                )
    for name in ("control", "spare"):
        for unit in shot.units[name]:
            if unit not in getattr(plan, name):
                raise BuildRefused(f"{unit} is in {name} but not in the plan's {name}")
    placement = (*PLACEMENT_EMPTY, *PLACEMENT_EDGE)
    for unit in shot.units["placement"]:
        if unit not in placement:
            raise BuildRefused(f"{unit} is a placement frame; they are {placement[0]}…"
                               f"{placement[-1]}")
    for unit in shot.discarded:
        if unit in seen:
            raise BuildRefused(
                f"{unit} is discarded and in {seen[unit]} — a discarded unit is in no split"
            )
    for unit, cls in shot.damage.items():
        if cls not in DAMAGES:
            raise BuildRefused(f"{unit}: damage {cls!r} — one of {', '.join(DAMAGES)}")
        if unit not in shot.units["test"]:
            raise BuildRefused(f"{unit}: damage on a unit that is not in the test split")
    for unit in shot.units["test"]:
        if unit not in shot.damage:
            raise BuildRefused(f"{unit}: a test unit with no damage class")


def _rows(session_dir: Path) -> dict[str, dict[str, str]]:
    """`frames.csv` by file name; a file logged twice is refused."""
    path = session_dir / FRAMES_FILE
    try:
        with path.open(encoding="utf-8", newline="") as log:
            reader = csv.DictReader(log)
            columns = reader.fieldnames or []
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                file = row.get("file")
                if file is None:
                    raise BuildRefused(f"{path}: every row names its file")
                if file in rows:
                    raise BuildRefused(f"{file}: twice in {path}")
                rows[file] = dict(row)
    except OSError as exc:
        raise BuildRefused(f"{path}: {exc}") from None
    if not {"file", "sha256", "status"} <= set(columns):
        raise BuildRefused(f"{path}: columns file, sha256 and status")
    return rows


def _check_frames(
    session_dir: Path, shot: _Session
) -> dict[str, list[tuple[str, FrameName]]]:
    """The frames against `frames.csv` and the shoot protocol; each unit's frames by name."""
    rows = _rows(session_dir)
    logged = {file: row for file, row in rows.items()
              if row["status"] in ("saved", "discarded")}
    pngs = {p.name for p in session_dir.glob("*.png")}
    for file in sorted(set(pngs) - set(logged)):
        raise BuildRefused(f"{file}: a PNG with no frames.csv row as saved or discarded")
    for file in sorted(set(logged) - pngs):
        raise BuildRefused(
            f"{file}: a frames.csv row as {logged[file]['status']} with no PNG on disk"
        )
    names: dict[str, FrameName] = {}
    for file in sorted(logged):
        try:
            name = parse_name(file)
        except FrameNameError as exc:
            raise BuildRefused(f"{file}: {exc}") from None
        if name.geometry != shot.geometry:
            raise BuildRefused(
                f"{file}: geometry {name.geometry} but the session is shot under "
                f"{shot.geometry}"
            )
        names[file] = name
    for file, row in logged.items():
        if row["status"] != "saved":
            continue
        digest = hashlib.sha256((session_dir / file).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise BuildRefused(
                f"{file}: SHA-256 {digest} but the shoot recorded {row['sha256']}"
            )
    frames: dict[str, list[tuple[str, FrameName]]] = {}
    for file in sorted(logged):
        frames.setdefault(names[file].unit, []).append((file, names[file]))
    for split in ("control", "train", "calib", "test", "placement"):
        for unit in shot.units[split]:
            if unit not in frames:
                raise BuildRefused(f"{unit}: no frame of it in this session")
            _check_unit(unit, frames[unit], shot)
    for unit in frames:
        if unit in shot.discarded:
            continue
        listed = any(unit in shot.units[name] for name in ("control", "train", "calib",
                                                           "test", "placement"))
        if not listed:
            if unit in shot.units["spare"]:
                raise BuildRefused(
                    f"{unit}: a spare with frames, in no split — a spare is shot only when "
                    "it takes a broken unit's place"
                )
            raise BuildRefused(f"{unit}: frames but in no list of this session")
    return frames


def _expected_keys(unit: str, shot: _Session) -> tuple[tuple[str, int, int | None], ...]:
    """The (state, rotation, pass) frames the protocol asks of this unit (docs/13 §6)."""
    if unit in shot.units["control"]:
        return tuple(("good", r, p) for p in (1, 2, 3) for r in (0, 1, 2))
    if unit in shot.units["train"] or unit in shot.units["calib"]:
        return tuple(("good", r, None) for r in (0, 1, 2))
    if unit in shot.units["test"]:
        return tuple((state, r, None) for state in ("good", "dmg") for r in (0, 1, 2))
    return (("empty", 0, None),) if int(unit[1:]) <= 5 else (("edge", 0, None),)


def _check_unit(unit: str, group: list[tuple[str, FrameName]], shot: _Session) -> None:
    want = _expected_keys(unit, shot)
    got = {(name.state, name.rotation, name.pass_) for _, name in group}
    for state, rotation, pass_ in want:
        if (state, rotation, pass_) not in got:
            tail = f", pass{pass_}" if pass_ is not None else ""
            raise BuildRefused(f"{unit}: no {state} frame at r{rotation}{tail}")
    for file, name in group:
        if (name.state, name.rotation, name.pass_) in want:
            continue
        tail = f", pass{name.pass_}" if name.pass_ is not None else ""
        raise BuildRefused(
            f"{file}: {unit} is not shot in state {name.state} at r{name.rotation}{tail}"
        )


def _copy(rel: str, source: Path, sha256: str, mode: int = 0o444) -> Planned:
    """A byte copy of `source` at `rel`; frames are read-only (0444)."""
    return Planned(rel, source, None, sha256, mode)


def _literal(rel: str, data: bytes, mode: int = 0o644) -> Planned:
    return Planned(rel, None, data, hashlib.sha256(data).hexdigest(), mode)


def _manifest_bytes(
    frames: list[tuple[str, FrameName, str, str]], sha_of: Callable[[str], str]
) -> bytes:
    """`manifest.csv`: the nine columns, one row per frame, in file-name order."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(MANIFEST_COLUMNS)
    for file, name, role, cls in frames:
        writer.writerow((
            file, name.unit, role, name.state, cls, name.geometry, f"r{name.rotation}",
            "" if name.pass_ is None else f"pass{name.pass_}", sha_of(file),
        ))
    return buffer.getvalue().encode("utf-8")


def plan_build(session_dir: Path, data: Path, product: str, name: str | None) -> Build:
    """The whole build, planned: every check runs before the first file is written."""
    shot = _session(session_dir)
    try:
        plan = read_plan(session_dir)
    except PlanError as exc:
        raise BuildRefused(f"{session_dir / PLAN_FILE}: {exc}") from None
    if shot.split_seed != plan.seed:
        raise BuildRefused(
            f"split_seed {shot.split_seed} but the plan's seed is {plan.seed} — the splits "
            "as shot come from the plan's seed (docs/13 §6)"
        )
    _check_splits(shot, plan)
    frames = _check_frames(session_dir, shot)
    for label, value in (("--product", product), ("--name", name)):
        if value is not None and NAME_RE.fullmatch(value) is None:
            raise BuildRefused(f"{label} {value!r}: lower-case words joined by '-'")
    if name is None:
        name = f"mvp-{shot.started_at.date().isoformat()}"
    rows = _rows(session_dir)
    sha_of = {file: row["sha256"] or "" for file, row in rows.items()}
    session_bytes = (session_dir / SESSION_FILE).read_bytes()

    dataset_files: list[Planned] = []
    golden_frames: list[tuple[str, FrameName, str, str]] = []
    for unit in shot.units["train"]:
        for file, _ in frames[unit]:
            dataset_files.append(_copy(f"train/good/{file}", session_dir / file, sha_of[file]))
    for unit in shot.units["calib"]:
        for file, name_ in frames[unit]:
            dataset_files.append(_copy(f"calib/good/{file}", session_dir / file, sha_of[file]))
            golden_frames.append((file, name_, "calib", ""))
    for unit in shot.units["test"]:
        cls = _CLASS_OF[shot.damage[unit]]
        for file, name_ in frames[unit]:
            folder = f"test/{'good' if name_.state == 'good' else cls}"
            dataset_files.append(_copy(f"{folder}/{file}", session_dir / file, sha_of[file]))
            golden_frames.append((file, name_, "test", cls))
    for unit in shot.units["control"]:
        for file, name_ in frames[unit]:
            golden_frames.append((file, name_, "control", ""))
    for unit in shot.units["placement"]:
        for file, name_ in frames[unit]:
            golden_frames.append((file, name_, "placement", ""))
    golden_frames.sort(key=lambda entry: entry[0])
    manifest = _manifest_bytes(golden_frames, lambda file: sha_of[file])
    golden_hash = hashlib.sha256(manifest).hexdigest()
    dataset_files.append(_literal(SESSION_FILE, session_bytes))
    golden_files = [
        _copy(f"{FRAMES_DIR}/{file}", session_dir / file, sha_of[file])
        for file, *_ in golden_frames
    ]
    golden_files.append(_literal("manifest.csv", manifest))
    golden_files.append(_literal(HASH_FILE, f"{golden_hash}\n".encode()))
    golden_files.append(_literal(SESSION_FILE, session_bytes))
    key = attrgetter("rel")
    dataset = Tree(data / "datasets" / product / shot.geometry,
                   tuple(sorted(dataset_files, key=key)))
    golden = Tree(data / "golden" / name, tuple(sorted(golden_files, key=key)))
    counts = tuple(
        (split, len(shot.units[split]), sum(len(frames[u]) for u in shot.units[split]))
        for split in ("train", "calib", "test", "control", "placement")
    )
    for tree, what in ((dataset, "dataset"), (golden, "golden")):
        differs = _existing_differs(tree)
        if differs is not None:
            raise BuildRefused(f"the {what} {tree.root} exists with different content — "
                               f"{differs}")
    return Build(
        session=session_dir, geometry=shot.geometry, name=name, dataset=dataset,
        golden=golden, golden_hash=golden_hash, dataset_done=dataset.root.is_dir(),
        golden_done=golden.root.is_dir(), counts=counts,
    )


def _existing_differs(tree: Tree) -> str | None:
    """What an existing target disagrees on, or None when it is absent or identical."""
    root = tree.root
    if not root.exists():
        return None
    if not root.is_dir():
        return "it is not a directory"
    existing = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    planned = {f.rel for f in tree.files}
    extra = sorted(existing - planned)
    if extra:
        return f"{extra[0]} is not part of the build"
    missing = sorted(planned - existing)
    if missing:
        return f"{missing[0]} is missing"
    for planned_file in tree.files:
        if planned_file.bytes() != (root / planned_file.rel).read_bytes():
            return f"{planned_file.rel} differs"
    return None


def write_build(build: Build) -> None:
    """Write what is missing. Each tree goes into a temporary directory beside its target
    and is renamed into place, so an existing target is never overwritten; the golden set
    is verified before it lands."""
    if not build.dataset_done:
        _write_tree(build.dataset)
    if not build.golden_done:
        _write_tree(build.golden, verify=True)


def _write_tree(tree: Tree, *, verify: bool = False) -> None:
    tree.root.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=tree.root.parent, prefix=f".{tree.root.name}."))
    try:
        for planned in tree.files:
            path = tmp / planned.rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(planned.bytes())
            os.chmod(path, planned.mode)
        if verify:
            verify_golden(tmp)
        os.chmod(tmp, 0o755)
        tmp.rename(tree.root)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _golden_rows(manifest: bytes, path: Path) -> list[dict[str, str]]:
    """The manifest's rows; the nine columns exactly, one row per frame, no file twice."""
    reader = csv.DictReader(io.StringIO(manifest.decode("utf-8"), newline=""))
    if reader.fieldnames != list(MANIFEST_COLUMNS):
        raise GoldenRefused(f"{path}: the columns are {', '.join(MANIFEST_COLUMNS)}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in reader:
        cells = {k: ("" if v is None else v) for k, v in record.items() if k is not None}
        empty = sorted(k for k, v in cells.items() if v == "" and k not in ("class", "pass"))
        if len(cells) != len(MANIFEST_COLUMNS) or empty:
            raise GoldenRefused(f"{path}: {', '.join(empty) or 'a row'} without the nine "
                                "columns filled")
        file = cells["file"]
        if file in seen:
            raise GoldenRefused(f"{file}: twice in the manifest")
        seen.add(file)
        rows.append(cells)
    return rows


def verify_golden(golden: Path) -> list[dict[str, str]]:
    """A golden set on disk against its own manifest; the rows when it holds.

    The hash file against `manifest.csv`, the frame files against the manifest and their
    hashes, every name against its row, roles against prefixes, one unit per role.
    """
    manifest_path = golden / "manifest.csv"
    hash_path = golden / HASH_FILE
    frames = golden / FRAMES_DIR
    try:
        manifest = manifest_path.read_bytes()
        recorded = hash_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise GoldenRefused(f"{golden}: {exc}") from None
    digest = hashlib.sha256(manifest).hexdigest()
    if recorded != digest:
        raise GoldenRefused(
            f"{HASH_FILE} says {recorded} but manifest.csv hashes to {digest} — the set "
            "was edited"
        )
    rows = _golden_rows(manifest, manifest_path)
    listed = [row["file"] for row in rows]
    try:
        on_disk = sorted(p.name for p in frames.iterdir() if p.is_file())
    except OSError as exc:
        raise GoldenRefused(f"{frames}: {exc}") from None
    for file in sorted(set(on_disk) - set(listed)):
        raise GoldenRefused(f"{file}: in {FRAMES_DIR}/ but not in the manifest")
    for file in sorted(set(listed) - set(on_disk)):
        raise GoldenRefused(f"{file}: in the manifest but not in {FRAMES_DIR}/")
    for row in rows:
        file = row["file"]
        try:
            frame_sha = hashlib.sha256((frames / file).read_bytes()).hexdigest()
        except OSError as exc:
            raise GoldenRefused(f"{file}: {exc}") from None
        if frame_sha != row["sha256"]:
            raise GoldenRefused(
                f"{file}: hashes to {frame_sha} but the manifest records {row['sha256']}"
            )
        try:
            name = parse_name(file)
        except FrameNameError as exc:
            raise GoldenRefused(f"{file}: {exc}") from None
        row_name = (row["unit"], row["state"], row["geometry"], row["rotation"], row["pass"])
        frame_name = (name.unit, name.state, name.geometry, f"r{name.rotation}",
                      "" if name.pass_ is None else f"pass{name.pass_}")
        if row_name != frame_name:
            raise GoldenRefused(
                f"{file}: the manifest's row (unit {row['unit']}, state {row['state']}, "
                f"rotation {row['rotation']}) does not match the name"
            )
    _check_roles(rows)
    return rows


def _check_roles(rows: list[dict[str, str]]) -> None:
    """Roles fit prefixes, one role per unit, and no pilot unit (docs/13 §5)."""
    role_of: dict[str, str] = {}
    for row in rows:
        file, unit, role = row["file"], row["unit"], row["role"]
        prefix = {"calib": "u", "test": "u", "control": "c", "placement": "x"}.get(role)
        if prefix is None:
            raise GoldenRefused(f"{file}: role {role!r} — one of {', '.join(ROLES)}")
        if not unit.startswith(prefix):
            if unit.startswith("p"):
                raise GoldenRefused(
                    f"{file}: a pilot unit is never in a golden set (docs/13 §5)"
                )
            raise GoldenRefused(f"{file}: role {role} takes a {prefix}-unit, not {unit}")
        if unit in role_of and role_of[unit] != role:
            raise GoldenRefused(
                f"{unit}: in roles {role_of[unit]} and {role} — one unit, one role"
            )
        role_of[unit] = role
