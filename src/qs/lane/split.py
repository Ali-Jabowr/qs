"""The time split as a file: every recorded clip has one role and a pinned hash (#154).

A block is one clip, so the split is a small file:
`{version: 1, clips: {"T-1.mkv": {role, sha256}}}`.
Training and calibration can then be told, by code, never to open a `test` clip.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from qs.errors import ConfigError
from qs.inspector.model import sha256_of

ROLES = ("train", "calibrate", "test")

Hasher = Callable[[Path], str]

# The one file-hash helper; the split's callers import it from here.
file_sha256: Hasher = sha256_of


@dataclass(frozen=True)
class Split:
    """Each clip's file name mapped to its role and the SHA-256 of its bytes."""

    clips: dict[str, tuple[str, str]]

    def videos(self, role: str) -> tuple[str, ...]:
        return tuple(video for video, (r, _) in self.clips.items() if r == role)

    def role_of(self, video: str) -> str:
        if video not in self.clips:
            raise ConfigError(f"{video} is not in the split file")
        return self.clips[video][0]


def _hash(hasher: Hasher, video: Path) -> str:
    try:
        return hasher(video)
    except OSError as exc:
        raise ConfigError(f"{video.name} cannot be read: {exc}") from exc


def count_frames(video: Path) -> tuple[int, int] | None:
    """(decoded, the container's count) of a video; None if it cannot be opened or reports no
    frames (an unreadable clip is not this check's to name)."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            return None
        container = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if container <= 0:
            return None
        decoded = 0
        while capture.grab():
            decoded += 1
        return decoded, container
    finally:
        capture.release()


def _sidecar(footage: Path, clip: str) -> tuple[str, bool]:
    """The recorded hash and the drift flag of a clip, from `<clip>.json`."""
    path = footage / f"{clip}.json"
    if not path.is_file():
        raise ConfigError(f"{path.name} not found in {footage}")
    try:
        data = json.loads(path.read_text())
        recorded, drifted = data["sha256"], data["drifted"]
    except OSError as exc:
        raise ConfigError(f"{path.name} cannot be read: {exc}") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"{path.name} is not a clip sidecar: {exc!r}") from exc
    if not isinstance(recorded, str) or not isinstance(drifted, bool):
        raise ConfigError(f"{path.name} is not a clip sidecar: sha256 must be text and "
                          f"drifted a boolean")
    return recorded, drifted


def build_split(footage: Path, train: list[str], calibrate: list[str], test: list[str],
                hasher: Hasher, allow_drifted: bool = False) -> Split:
    """Give each named clip its role after checking its sidecar and hash."""
    given = {"train": train, "calibrate": calibrate, "test": test}
    for role in ROLES:
        if not given[role]:
            raise ConfigError(f"every role needs at least one clip: none given for {role}")
    clips: dict[str, tuple[str, str]] = {}
    for role in ROLES:
        for clip in dict.fromkeys(given[role]):
            if given[role].count(clip) > 1:
                raise ConfigError(f"{clip} is listed twice under {role}")
    for clip in dict.fromkeys(c for role in ROLES for c in given[role]):
        roles = [role for role in ROLES if clip in given[role]]
        if len(roles) > 1:
            raise ConfigError(f"{clip} is given two roles: {', '.join(roles)}")
    for role in ROLES:
        for clip in given[role]:
            recorded, drifted = _sidecar(footage, clip)
            if drifted and not allow_drifted:
                raise ConfigError(f"{clip} drifted while recording (--allow-drifted to keep it)")
            video = f"{clip}.mkv"
            actual = _hash(hasher, footage / video)
            if actual != recorded:
                raise ConfigError(f"{video} differs from its sidecar: "
                                  f"{actual[:8]} now, {recorded[:8]} recorded")
            counts = count_frames(footage / video)
            if counts is not None and counts[0] != counts[1]:
                raise ConfigError(f"{video} decodes {counts[0]} frames but its container says "
                                  f"{counts[1]}: frame numbers are not reliable; "
                                  f"run qs clip-fix")
            clips[video] = (role, actual)
    return Split(clips)


def write_split(split: Split, path: Path) -> None:
    body = {"version": 1,
            "clips": {v: {"role": r, "sha256": h} for v, (r, h) in split.clips.items()}}
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def load_split(path: Path) -> Split:
    try:
        data = yaml.safe_load(path.read_text())
        if data["version"] != 1:
            raise ValueError(f"version {data['version']}")
        clips = {str(v): (str(c["role"]), str(c["sha256"])) for v, c in data["clips"].items()}
    except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"{path} is not a split file: {exc!r}") from exc
    for video, (role, _) in clips.items():
        if role not in ROLES:
            raise ConfigError(f"{path}: {video} has unknown role {role!r}")
    return Split(clips)


def verify(split: Split, footage: Path, hasher: Hasher) -> None:
    """Raise naming the first clip whose bytes no longer match the split file."""
    for video, (_, pinned) in split.clips.items():
        actual = _hash(hasher, footage / video)
        if actual != pinned:
            raise ConfigError(f"{video} changed since the split: "
                              f"{actual[:8]} now, {pinned[:8]} pinned")


def require_roles(split: Split, videos: Iterable[str], allowed: set[str]) -> None:
    """Raise naming the first video whose role is not in `allowed`."""
    for video in videos:
        role = split.role_of(video)
        if role not in allowed:
            raise ConfigError(f"{video} has role {role}, not one of: {', '.join(sorted(allowed))}")
