"""The `qs` command: `bin/qs <subcommand> …`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """The `qs` parser with every subcommand."""
    parser = argparse.ArgumentParser(prog="qs", description="QS bench station.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    camera = commands.add_parser(
        "camera-test", help="the camera lock test (docs/13-data-collection.md §4)"
    )
    camera.add_argument("--config", required=True, metavar="PATH", help="the station config")
    camera.add_argument("--frames", type=int, default=50, metavar="N", help="frames in the test")
    camera.add_argument(
        "--interval-s", type=float, default=6.0, metavar="S", help="seconds between frames"
    )
    camera.add_argument(
        "--out", metavar="DIR", default=None,
        help="where the frames go (default data/raw/<YYYY-MM-DD>-locktest)",
    )
    camera.add_argument(
        "--report-dir", metavar="DIR", default="reports/camera", help="where the report goes"
    )
    camera.add_argument("--no-screen", action="store_true", help="run without the window")

    from qs.cmd.split_plan import session_id

    plan = commands.add_parser(
        "split-plan", help="the seeded split plan and tray sheet (docs/13-data-collection.md §6)"
    )
    plan.add_argument(
        "--session", required=True, type=session_id, metavar="ID", help="e.g. 2026-09-25-mvp"
    )
    plan.add_argument(
        "--size", required=True, choices=("standard", "lean"), help="docs/13 §1, OQ-8"
    )
    plan.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help="the split seed (default the session date as YYYYMMDD)",
    )
    plan.add_argument(
        "--root", default="data/raw", metavar="DIR", help="where session folders live"
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments and dispatch to the subcommand."""
    args = build_parser().parse_args(argv)
    if args.command == "split-plan":
        from qs.cmd import split_plan

        return split_plan.main(args)
    import os

    import cv2

    # OpenCV's own warnings would print beside our one-line errors; ours say it all.
    # OPENCV_LOG_LEVEL, when set, still wins — for diagnosing a camera.
    if "OPENCV_LOG_LEVEL" not in os.environ:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    if args.command == "camera-test":
        from qs.cmd import camera_test

        return camera_test.main(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(run())
