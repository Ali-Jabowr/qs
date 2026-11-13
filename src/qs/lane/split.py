"""Stub."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

Hasher = Callable[[Path], str]


@dataclass(frozen=True)
class Split:
    clips: dict[str, tuple[str, str]]

    def videos(self, role: str) -> tuple[str, ...]:
        raise NotImplementedError

    def role_of(self, video: str) -> str:
        return ""


def file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split(footage: Path, train: list[str], calibrate: list[str], test: list[str],
                hasher: Hasher, allow_drifted: bool = False) -> Split:
    raise NotImplementedError


def write_split(split: Split, path: Path) -> None:
    raise NotImplementedError


def load_split(path: Path) -> Split:
    raise NotImplementedError


def verify(split: Split, footage: Path, hasher: Hasher) -> None:
    return None


def require_roles(split: Split, videos: Iterable[str], allowed: set[str]) -> None:
    return None
