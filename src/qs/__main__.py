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

    from qs.cmd.split_plan import geometry_list, session_id
    from qs.names import GEOMETRIES

    plan = commands.add_parser(
        "split-plan",
        help="the seeded split plan and tray sheet, dataset or pilot (docs/13 §5, §6)",
    )
    plan.add_argument(
        "--session", required=True, type=session_id, metavar="ID", help="e.g. 2026-09-25-mvp"
    )
    plan.add_argument(
        "--size", choices=("standard", "lean"), help="docs/13 §1, OQ-8 (required to plan)"
    )
    plan.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help="the split seed (default the session date as YYYYMMDD)",
    )
    plan.add_argument(
        "--root", default="data/raw", metavar="DIR", help="where session folders live"
    )
    plan.add_argument(
        "--kind", choices=("dataset", "pilot"), default="dataset",
        help="the MVP dataset (default) or the lighting pilot (docs/13 §5)",
    )
    plan.add_argument(
        "--geometries", type=geometry_list, default=None, metavar="G1,G2,G3",
        help="with --kind pilot: G1,G2,G3 (default) or G1,G3 when the house has one lamp",
    )
    plan.add_argument(
        "--shot-list", action="store_true",
        help="plan nothing: write the numbered photo list shots-<G>.md from plan.yml (ADR-0022)",
    )
    plan.add_argument(
        "--geometry", choices=GEOMETRIES, default=None, help="with --shot-list: the lighting"
    )

    from qs.cmd.shoot import best_before, purchased_at

    shoot = commands.add_parser(
        "shoot", help="the guided dataset session (docs/13-data-collection.md §6, §8)"
    )
    shoot.add_argument("--config", required=True, metavar="PATH", help="the station config")
    shoot.add_argument(
        "--session", required=True, type=session_id, metavar="ID",
        help="the session planned by split-plan",
    )
    shoot.add_argument("--flavour", required=True, metavar="F", help="the моти's flavour")
    shoot.add_argument("--shop", required=True, metavar="S", help="where it was bought")
    shoot.add_argument(
        "--best-before", required=True, type=best_before, metavar="DATE", help="YYYY-MM-DD"
    )
    shoot.add_argument(
        "--purchased-at", required=True, type=purchased_at, metavar="ISO",
        help="e.g. 2026-09-25T09:10+03:00",
    )
    maker = shoot.add_mutually_exclusive_group(required=True)
    maker.add_argument("--brand", metavar="B", help="the brand on the pack")
    maker.add_argument(
        "--client-product", action="store_true",
        help="the client's own моти: recorded as client_product, no brand (docs/17 §5)",
    )
    shoot.add_argument(
        "--root", default="data/raw", metavar="DIR", help="where session folders live"
    )

    from qs.cmd.import_phone import broken_unit, offset, redo_list

    phone = commands.add_parser(
        "import-phone",
        help="phone photos into a dataset or pilot session, or --check the phone's lock "
        "(ADR-0022)",
    )
    phone.add_argument("--config", required=True, metavar="PATH", help="the phone config")
    phone.add_argument(
        "--from", dest="source", required=True, metavar="DIR",
        help="the phone's photos, copied over USB (only read)",
    )
    phone.add_argument(
        "--session", type=session_id, default=None, metavar="ID",
        help="the session planned by split-plan (with --check: the mid-session check)",
    )
    phone.add_argument(
        "--check", action="store_true",
        help="write nothing: the day-zero lock test, or with --session where the shoot is",
    )
    phone.add_argument("--flavour", metavar="F", help="the моти's flavour")
    phone.add_argument("--shop", metavar="S", help="where it was bought")
    phone.add_argument("--best-before", type=best_before, metavar="DATE", help="YYYY-MM-DD")
    phone.add_argument(
        "--purchased-at", type=purchased_at, metavar="ISO", help="e.g. 2026-09-25T09:10+03:00"
    )
    phone_maker = phone.add_mutually_exclusive_group()
    phone_maker.add_argument("--brand", metavar="B", help="the brand on the pack")
    phone_maker.add_argument(
        "--client-product", action="store_true",
        help="the client's own моти: recorded as client_product, no brand (docs/17 §5)",
    )
    phone.add_argument(
        "--redo", type=redo_list, action="append", default=[], metavar="K[,K…]",
        help="row K of the printed list was spoiled and retaken — a spare's photo, which has "
        "no row, by its frame name (u053_good_G2_r1); repeat it for each redo (A1)",
    )
    phone.add_argument(
        "--broken", type=broken_unit, action="append", default=[], metavar="UNIT:N",
        help="the unit broke after N of its photos; the next spare takes its place — in a "
        "pilot, which has no spares, the unit is discarded (A1)",
    )
    phone.add_argument(
        "--utc-offset", type=offset, default=None, metavar="±HH:MM",
        help="the phone clock's offset, when the EXIF has none",
    )
    phone.add_argument(
        "--root", default="data/raw", metavar="DIR", help="where session folders live"
    )

    from qs.cmd.train import coreset, positive_int, product, seed

    train = commands.add_parser(
        "train", help="PatchCore on a dataset's train/good → a model bundle (FR-53, docs/15 §8)"
    )
    train.add_argument("--config", required=True, metavar="PATH", help="the station config")
    train.add_argument(
        "--dataset", required=True, metavar="DIR", help="the dataset; only DIR/train/good is read"
    )
    train.add_argument(
        "--window-px", required=True, type=positive_int, metavar="N",
        help="the window around each unit, from the pilot report",
    )
    train.add_argument(
        "--product", required=True, type=product, metavar="P", help="for the model_id, e.g. mochi"
    )
    train.add_argument("--seed", type=seed, default=42, metavar="N", help="coreset seed (42)")
    train.add_argument(
        "--coreset", type=coreset, default=0.1, metavar="R",
        help="coreset share (0.1; 0.01 is the first latency fallback, docs/15 §10)",
    )
    train.add_argument(
        "--input-px", type=positive_int, default=256, metavar="N",
        help="the model input (256; 224 is a latency fallback, docs/15 §10)",
    )
    train.add_argument(
        "--models", default="models", metavar="DIR", help="where bundles are written (models)"
    )

    from qs.cmd import station as station_cmd  # P04: qs station --replay

    station_cmd.add_parser(commands)
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
    if args.command == "shoot":
        from qs.cmd import shoot

        return shoot.main(args)
    if args.command == "import-phone":
        from qs.cmd import import_phone

        return import_phone.main(args)
    if args.command == "train":
        from qs.cmd import train

        return train.main(args)
    if args.command == "station":  # P04
        from qs.cmd import station

        return station.main(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(run())
