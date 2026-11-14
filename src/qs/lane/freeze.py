"""The freeze file: what a scoring of the test block rests on, hashed (#160).

`freeze.yml` holds the SHA-256 of the lane bundle's `model.pt`, the threshold file, the lane file
and the split file, the commit, and optionally method B's model and threshold. `check_freeze`
raises, naming the file or the commits, when any of them is no longer what was frozen.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from qs.errors import ConfigError
from qs.lane.split import file_sha256

_KEYS = {"version", "git_commit", "lane_bundle", "threshold_sha256", "lane_sha256",
         "split_sha256", "patchcore", "created_utc"}


@dataclass(frozen=True, slots=True)
class Freeze:
    """Everything the freeze file records; `patchcore` is (model_id, threshold) or None."""

    git_commit: str
    model_id: str
    model_sha256: str
    threshold_sha256: str
    lane_sha256: str
    split_sha256: str
    patchcore: tuple[str, float] | None
    created_utc: str


def run_git(args: Sequence[str]) -> str:
    """The standard output of `git <args>` in the current folder; ConfigError if it fails."""
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ConfigError(f"cannot run git: {exc}") from exc
    if done.returncode != 0:
        detail = done.stderr.strip().splitlines()[:1] or [f"exit {done.returncode}"]
        raise ConfigError(f"git {args[0]} failed: {detail[0]}")
    return done.stdout


def require_clean_tree(git: Callable[[Sequence[str]], str]) -> None:
    """ConfigError, in one line, if a tracked file has uncommitted changes."""
    dirty = [line for line in git(["status", "--porcelain", "--untracked-files=no"]).splitlines()
             if line.strip()]
    if dirty:
        raise ConfigError(f"the working tree has uncommitted changes to {len(dirty)} tracked "
                          f"file(s); commit them first")


def threshold_of(path: Path) -> tuple[str, float]:
    """The `model_id` and `value` of a threshold file (the spec's `threshold.yml`)."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_id, value = data["model_id"], data["value"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise ConfigError(f"{path}: not a threshold file: {exc!r}") from exc
    if (
        not isinstance(model_id, str)
        or isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ConfigError(f"{path}: model_id must be text and value a finite number")
    return model_id, float(value)


def model_file_sha256(models: Path, model_id: str) -> str:
    """The SHA-256 of the bundle's `model.pt`, after it matches the hash `bundle.yml` records."""
    bundle = models / model_id
    try:
        recorded = yaml.safe_load((bundle / "bundle.yml").read_text(encoding="utf-8"))
        recorded_sha = recorded["model_sha256"]
        actual = file_sha256(bundle / "model.pt")
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise ConfigError(f"bundle {model_id}: cannot be read from {models}: {exc!r}") from exc
    if actual != recorded_sha:
        raise ConfigError(f"bundle {model_id}: model.pt has SHA-256 {actual[:8]}, "
                          f"not the recorded {str(recorded_sha)[:8]}")
    return actual


def write_freeze(path: Path, freeze: Freeze) -> None:
    """Write the freeze file; an existing file is never overwritten."""
    body = {
        "version": 1,
        "git_commit": freeze.git_commit,
        "lane_bundle": {"model_id": freeze.model_id, "model_sha256": freeze.model_sha256},
        "threshold_sha256": freeze.threshold_sha256,
        "lane_sha256": freeze.lane_sha256,
        "split_sha256": freeze.split_sha256,
        "patchcore": None if freeze.patchcore is None else {
            "model_id": freeze.patchcore[0], "threshold": freeze.patchcore[1]},
        "created_utc": freeze.created_utc,
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(yaml.safe_dump(body, sort_keys=False))
    except FileExistsError as exc:
        raise ConfigError(f"{path} already exists; a freeze is never overwritten") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot be written: {exc}") from exc


def load_freeze(path: Path) -> Freeze:
    """The freeze file at `path`; a missing or extra key or a wrong type raises ConfigError."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: cannot read the freeze file: {exc}") from exc
    if not isinstance(data, dict) or set(data) != _KEYS or data["version"] != 1:
        raise ConfigError(f"{path}: not a freeze file (version 1 with keys {sorted(_KEYS)})")
    try:
        bundle, patchcore = data["lane_bundle"], data["patchcore"]
        frozen = Freeze(
            git_commit=data["git_commit"],
            model_id=bundle["model_id"],
            model_sha256=bundle["model_sha256"],
            threshold_sha256=data["threshold_sha256"],
            lane_sha256=data["lane_sha256"],
            split_sha256=data["split_sha256"],
            patchcore=None if patchcore is None else (patchcore["model_id"],
                                                       float(patchcore["threshold"])),
            created_utc=data["created_utc"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: not a freeze file: {exc!r}") from exc
    texts = [frozen.git_commit, frozen.model_id, frozen.model_sha256, frozen.threshold_sha256,
             frozen.lane_sha256, frozen.split_sha256, frozen.created_utc]
    if not all(isinstance(text, str) for text in texts):
        raise ConfigError(f"{path}: not a freeze file: every hash, id and time must be text")
    return frozen


def check_freeze(
    freeze: Freeze, *, models: Path, threshold: Path, lane: Path, split: Path, head: str
) -> None:
    """Raise, naming the file or both commits, unless every frozen input is still as frozen."""
    pinned = [
        (models / freeze.model_id / "model.pt", freeze.model_sha256),
        (threshold, freeze.threshold_sha256),
        (lane, freeze.lane_sha256),
        (split, freeze.split_sha256),
    ]
    for path, frozen in pinned:
        try:
            actual = file_sha256(path)
        except OSError as exc:
            raise ConfigError(f"{path} cannot be read: {exc}") from exc
        if actual != frozen:
            raise ConfigError(f"{path} changed since the freeze: "
                              f"{actual[:8]} now, {frozen[:8]} frozen")
    if head != freeze.git_commit:
        raise ConfigError(f"HEAD is {head[:8]}, the freeze was made at {freeze.git_commit[:8]}")
