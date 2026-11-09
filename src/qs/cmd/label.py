"""`qs label`: hand labels for the first visit's footage, in a thin window over `qs.labels`.

`--mode frames` labels single frames of a folder of images, or of a video with `--every N`, as
unit, empty cup, nothing or hand. `--mode cups` counts whole cups in order through one video, as
full or empty. Every answer is on disk before the next key is read, so stopping is always safe:
Q or closing the window stops, and the next run resumes. Only counts are printed, and no image is
ever written.
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qs.cmd.train import positive_int
from qs.config import DisplayConfig, load_config
from qs.errors import ConfigError
from qs.labels import (
    CupSession,
    FrameReadError,
    FrameSession,
    ImageFolder,
    Labels,
    LabelsError,
    VideoFrames,
    default_labels_path,
    load_labels,
    source_name,
)
from qs.screen import KeyPress, Screen

_TICK_S = 0.05
_QUIT = ("q", "Q")
_SHIFT = 0x1  # the Shift bit of a Tk key event's `state`
_SHIFTED = ("Left", "Right")


def named_key(keysym: str, state: int) -> str:
    """The key as the sessions name it: a shifted arrow is its own key, `Shift-Right`."""
    if state & _SHIFT and keysym in _SHIFTED:
        return f"Shift-{keysym}"
    return keysym


class _LabelScreen(Screen):
    """The operator window, also telling a shifted arrow from a plain one: `KeyPress` carries
    only a keysym, and Tk gives both the same keysym."""

    def _on_key(self, event: tk.Event[tk.Misc]) -> None:
        name = named_key(str(event.keysym), int(event.state))
        self._keys.append(KeyPress(name, int(event.time), self._last_pump_end))


def add_parser(commands: Any) -> None:
    """Register `label` on the `qs` subcommands."""
    parser = commands.add_parser(
        "label", help="hand labels for visit footage: single frames, or whole cups through a video"
    )
    parser.add_argument("--config", required=True, metavar="PATH", help="the station config")
    parser.add_argument(
        "--mode", required=True, choices=("frames", "cups"),
        help="frames: unit, empty cup, nothing or hand; cups: full or empty, counted in order",
    )
    parser.add_argument(
        "--source", required=True, metavar="PATH",
        help="a folder of images or a video; --mode cups needs a video",
    )
    parser.add_argument(
        "--every", type=positive_int, default=None, metavar="N",
        help="with --mode frames on a video (required there): sample every Nth frame",
    )
    parser.add_argument(
        "--labels", default=None, metavar="PATH",
        help="the labels file (default data/labels/<source name>.json)",
    )


def _open_source(source: Path, mode: str, every: int | None) -> ImageFolder | VideoFrames:
    """The frames `--source` names, or `FrameReadError` saying which arguments do not fit."""
    if mode == "cups":
        if every is not None:
            raise FrameReadError("--every only applies to --mode frames")
        if source.is_dir():
            raise FrameReadError(f"{source}: --mode cups needs a video, not a folder")
        return VideoFrames(source)
    if source.is_dir():
        if every is not None:
            raise FrameReadError(f"{source}: --every only applies to a video")
        return ImageFolder(source)
    if every is None:
        raise FrameReadError(f"{source}: a video needs --every N, to sample every Nth frame")
    return VideoFrames(source, every)


def _session(
    labels: Labels, frames: ImageFolder | VideoFrames, mode: str
) -> FrameSession | CupSession:
    if isinstance(frames, VideoFrames) and mode == "cups":
        return CupSession(labels, frames)
    return FrameSession(labels, frames)


def main(
    args: argparse.Namespace,
    *,
    make_screen: Callable[[DisplayConfig, str], Screen] = _LabelScreen,
) -> int:
    """Label until stopped; 0 stopped with every answer saved, 1 a frame or a save failed, 2 could
    not start."""
    try:
        config = load_config(Path(args.config)).config
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    source = Path(args.source)
    path = Path(args.labels) if args.labels else default_labels_path(source)
    try:
        labels = load_labels(path, source_name(source))
    except LabelsError as exc:
        print(f"labels: {exc}")
        return 2
    if args.mode == "frames" and labels.source != source_name(source):
        print(
            f"labels: {path} belongs to {labels.source!r}, not {source_name(source)!r}: "
            "frames from two sources would mix in one file, "
            "so use a separate labels file (--labels)"
        )
        return 2
    try:
        frames = _open_source(source, args.mode, args.every)
    except FrameReadError as exc:
        print(f"source: {exc}")
        return 2

    screen: Screen | None = None
    session = _session(labels, frames, args.mode)
    try:
        try:
            screen = make_screen(config.display, f"QS — label {source.name}")
        except tk.TclError as exc:
            print(f"screen: {exc}")
            return 2
        code = _label(session, screen)
        print(session.summary())
        return code
    except KeyboardInterrupt:
        print(f"stopped by the operator — {session.summary()}")
        return 1
    finally:
        frames.close()
        if screen is not None:
            screen.close()


def _label(session: FrameSession | CupSession, screen: Screen) -> int:
    """Show the frame and pass on the keys, in the order pressed, until stopped."""
    try:
        while True:
            screen.tick(session.image(), session.lines())
            pressed = screen.keys()
            for press in pressed:
                if press.keysym in _QUIT:
                    return 0
                session.press(press.keysym)
            if screen.closed:
                return 0
            if not pressed:
                time.sleep(_TICK_S)
    except FrameReadError as exc:
        print(f"frame: {exc} — stopped, labels so far kept")
        return 1
    except OSError as exc:
        print(f"save: {exc} — stopped, the labels file holds every earlier answer")
        return 1
