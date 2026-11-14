"""The defects file (#175): which labelled full passes of the test clips are off-colour.

    version: 1
    defects:
      - {video: E-1.mkv, frame: 50, kind: patch, unit: D01}

`patch`: part of the unit is off-colour; `colour`: the whole unit is; `shape`: the outline is.
Only test clips may carry a defect: one in a train clip would enter PatchCore's bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from qs.cmd.train_lane_b import _FULL
from qs.errors import ConfigError
from qs.labels import Labels
from qs.lane.split import Split

KINDS = ("patch", "colour", "shape")
_KEYS = {"video", "frame", "kind", "unit"}


@dataclass(frozen=True, slots=True)
class Defect:
    """A defective unit: the full pass at `frame` of `video`, of `kind`, named `unit`."""

    video: str
    frame: int
    kind: str
    unit: str


def _defect(path: Path, entry: object, labels: Labels, split: Split) -> Defect:
    if not isinstance(entry, dict) or entry.keys() != _KEYS:
        raise ConfigError(f"{path}: every defect needs exactly {sorted(_KEYS)}: {entry!r}")
    video, frame, kind, unit = entry["video"], entry["frame"], entry["kind"], entry["unit"]
    if not isinstance(video, str) or not isinstance(unit, str) or not unit:
        raise ConfigError(f"{path}: video and unit must be text: {entry!r}")
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise ConfigError(f"{path}: {video}: frame must be a whole number, not {frame!r}")
    if kind not in KINDS:
        raise ConfigError(f"{path}: {video} frame {frame}: kind {kind!r} is not one of "
                          f"{', '.join(KINDS)}")
    if video not in split.clips:
        raise ConfigError(f"{path}: {video} is not in the split file")
    role = split.role_of(video)
    if role != "test":
        raise ConfigError(f"{path}: {video} is a {role} clip; defects belong to test clips only")
    if not any(cup.frame == frame and cup.state in _FULL for cup in labels.cups_of(video)):
        raise ConfigError(f"{path}: {video} frame {frame} is not a labelled full pass")
    return Defect(video, frame, kind, unit)


def load_defects(path: Path, labels: Labels, split: Split) -> tuple[Defect, ...]:
    """The defects in `path`, or one `ConfigError` line naming the first that breaks the file's
    contract."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        text = str(error).replace("\n", " ")
        raise ConfigError(f"{path}: cannot read the defects file: {text}") from error
    if not isinstance(data, dict) or data.keys() != {"version", "defects"}:
        raise ConfigError(f"{path}: the defects file needs exactly 'version' and 'defects'")
    if data["version"] != 1:
        raise ConfigError(f"{path}: key 'version' must be 1, not {data['version']!r}")
    if not isinstance(data["defects"], list):
        raise ConfigError(f"{path}: key 'defects' must be a list")
    return tuple(_defect(path, entry, labels, split) for entry in data["defects"])
