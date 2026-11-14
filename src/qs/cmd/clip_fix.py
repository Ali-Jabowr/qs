"""`qs clip-fix VIDEO` (stub)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any


def add_parser(commands: Any) -> None:
    parser = commands.add_parser("clip-fix", help="renumber a clip's frames without re-encoding")
    parser.add_argument("video", metavar="VIDEO")


def main(
    args: argparse.Namespace,
    *,
    run_ffmpeg: Callable[[list[str]], int] | None = None,
    count_missing: Callable[[Path], int] | None = None,
    file_hash: Callable[[Path], str] | None = None,
) -> int:
    return 3
