"""Hand labels of the first visit's footage: the labels file and two sessions with no display.

The labels file is the contract the scoring command reads, one JSON document:

    {"version": 1, "source": "<name>",
     "frames": {"<frame id>": {"label": "unit|empty_cup|nothing|hand",
                               "flavour": "white|pink|null"}},
     "cups": [{"video": "<file name>", "order": 1,
               "state": "full|full_white|full_pink|empty", "frame": 1234}]}

A frame id is an image's file name, or `<video file name>#<frame index>`. A folder's frame ids are
bare file names, so each folder of images needs its own labels file. Every change rewrites the
whole file atomically before it counts, so a crash between answers loses nothing. A file that
breaks the contract is refused and never overwritten. Nothing here writes an image.

Within one video, cup frames never decrease, and two cups may share a frame. A video's counted
span runs from its first cup's frame to its last cup's frame, and every cup that crossed the view
inside that span has a record. A labels file covers one camera position, which the scoring command
is told, so footage from different positions goes in different files.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

_VERSION = 1
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_FILE_KEYS = {"version", "source", "frames", "cups"}
_FRAME_KEYS = {"label", "flavour"}
_CUP_KEYS = {"video", "order", "state", "frame"}
_FLAVOURS = ("white", "pink")


class FrameKind(StrEnum):
    """What a labelled frame shows, spelled as the labels file spells it."""

    UNIT = "unit"
    EMPTY_CUP = "empty_cup"
    NOTHING = "nothing"
    HAND = "hand"


class CupState(StrEnum):
    """A counted cup's state, spelled as the labels file spells it."""

    FULL = "full"
    FULL_WHITE = "full_white"
    FULL_PINK = "full_pink"
    EMPTY = "empty"
    GAP = "gap"  # a chain position with no cup



_ANSWERS = {
    "u": ("unit", None),
    "w": ("unit", "white"),
    "p": ("unit", "pink"),
    "e": ("empty_cup", None),
    "n": ("nothing", None),
    "h": ("hand", None),
}
_CUP_STATES = {
    "f": CupState.FULL, "w": CupState.FULL_WHITE, "p": CupState.FULL_PINK,
    "e": CupState.EMPTY, "g": CupState.GAP,
}
_WINDOW_COLOUR = (0, 255, 255)  # BGR yellow
_CUP_STEPS = {"Left": -1, "Right": 1, "Shift-Left": -25, "Shift-Right": 25}


class LabelsError(ValueError):
    """A label or a labels file that breaks the file's contract; the file is left as it is."""


class FrameReadError(ValueError):
    """A source or one of its frames could not be read; never skipped silently."""


@dataclass(frozen=True, slots=True)
class FrameLabel:
    """One frame's answer; `flavour` is None when not visible, and on anything but a unit."""

    label: FrameKind
    flavour: str | None = None


@dataclass(frozen=True, slots=True)
class Cup:
    """One whole cup: its 1-based `order` within its video, and the frame it was recorded at."""

    video: str
    order: int
    state: CupState
    frame: int


def _frame_label(frame_id: str, label: object, flavour: object) -> FrameLabel:
    """One frame's label, checked against the contract; a flavour only ever goes on a unit."""
    if not isinstance(label, str) or label not in FrameKind:
        raise LabelsError(f"{frame_id}: unknown label {label!r}")
    if flavour is not None and (not isinstance(flavour, str) or flavour not in _FLAVOURS):
        raise LabelsError(f"{frame_id}: unknown flavour {flavour!r}")
    if flavour is not None and label != FrameKind.UNIT:
        raise LabelsError(f"{frame_id}: a flavour only goes on a unit, not on {label}")
    return FrameLabel(FrameKind(label), flavour)


def _write_atomically(path: Path, text: str) -> None:
    """`text` to a temporary file beside `path`, on the disk, then renamed over `path`: at every
    moment `path` is the whole old file or the whole new one."""
    # Not the store's or calibrate's write: this one fsyncs, as hand labels cannot be regenerated.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


@dataclass(slots=True)
class Labels:
    """What one labels file holds, and the only way it changes."""

    path: Path
    source: str
    frames: dict[str, FrameLabel] = field(default_factory=dict)
    cups: tuple[Cup, ...] = ()

    def set_frame(self, frame_id: str, label: str, flavour: str | None = None) -> None:
        """Label one frame, replacing any label it had; in memory only once it is on disk."""
        frames = {**self.frames, frame_id: _frame_label(frame_id, label, flavour)}
        self._save(frames, self.cups)
        self.frames = frames

    def cups_of(self, video: str) -> tuple[Cup, ...]:
        return tuple(cup for cup in self.cups if cup.video == video)

    def answer_cup(self, video: str, state: CupState, frame: int) -> bool:
        """Apply one answer at `frame`, in memory only once it is on disk; False if it changed
        nothing. At or after the last cup's frame it records the next cup. Before it, it replaces
        the state of the highest-order cup on that frame, if that frame has one."""
        mine = self.cups_of(video)
        if not mine or frame >= mine[-1].frame:
            cups = (*self.cups, Cup(video, len(mine) + 1, state, frame))
        else:
            here = [cup for cup in mine if cup.frame == frame]
            if not here:
                return False
            cups = tuple(replace(c, state=state) if c == here[-1] else c for c in self.cups)
        self._save(self.frames, cups)
        self.cups = cups
        return True

    def remove_last_cup(self, video: str) -> None:
        """Remove `video`'s last cup, if it has one; other videos' cups stay."""
        mine = self.cups_of(video)
        if not mine:
            return
        cups = tuple(cup for cup in self.cups if cup != mine[-1])
        self._save(self.frames, cups)
        self.cups = cups

    def _save(self, frames: dict[str, FrameLabel], cups: tuple[Cup, ...]) -> None:
        document = {
            "version": _VERSION,
            "source": self.source,
            "frames": {k: {"label": v.label, "flavour": v.flavour} for k, v in frames.items()},
            "cups": [asdict(cup) for cup in cups],
        }
        _write_atomically(self.path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def _parsed_cups(entries: list[object]) -> tuple[Cup, ...]:
    """The cups in a decoded file; as they are listed, each video's `order` must run 1, 2, 3…
    and its `frame` must never decrease."""
    cups: list[Cup] = []
    seen: Counter[str] = Counter()
    last_frame: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _CUP_KEYS:
            raise ValueError(f"cup {entry!r}: expected exactly the keys {sorted(_CUP_KEYS)}")
        video, order, state, frame = (entry[key] for key in ("video", "order", "state", "frame"))
        if not isinstance(video, str):
            raise ValueError(f"cup {entry!r}: video must be text")
        if state not in CupState:
            raise ValueError(f"cup {entry!r}: unknown state")
        if type(frame) is not int or frame < 0:
            raise ValueError(f"cup {entry!r}: frame must be a whole number from 0")
        seen[video] += 1
        if type(order) is not int or order != seen[video]:
            raise ValueError(f"cup {entry!r}: expected order {seen[video]} in {video}")
        if frame < last_frame.get(video, 0):
            raise ValueError(f"cup {entry!r}: frames must not decrease within {video}")
        last_frame[video] = frame
        cups.append(Cup(video, order, CupState(state), frame))
    return tuple(cups)


def _parsed(document: object, path: Path) -> Labels:
    """The labels in a decoded file; `ValueError` naming the first thing that breaks the shape."""
    if not isinstance(document, dict) or set(document) != _FILE_KEYS:
        raise ValueError(f"expected exactly the keys {sorted(_FILE_KEYS)}")
    version = document["version"]
    if type(version) is not int or version != _VERSION:
        raise ValueError(f"unknown version {version!r}, this tool reads {_VERSION}")
    source, entries, cups = document["source"], document["frames"], document["cups"]
    if not isinstance(source, str) or not isinstance(entries, dict) or not isinstance(cups, list):
        raise ValueError("source must be text, frames an object and cups a list")
    frames = {}
    for frame_id, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != _FRAME_KEYS:
            raise ValueError(f"{frame_id}: expected exactly the keys {sorted(_FRAME_KEYS)}")
        frames[frame_id] = _frame_label(frame_id, entry["label"], entry["flavour"])
    return Labels(path, source, frames, _parsed_cups(cups))


def source_name(source: Path) -> str:
    """What a labels file calls its source: a folder's name, or a video's name without suffix."""
    return source.name if source.is_dir() else source.stem


def default_labels_path(source: Path) -> Path:
    """Where the labels of `source` go by default, under the git-ignored data folder."""
    return Path("data/labels") / f"{source_name(source)}.json"


def load_labels(path: Path, source: str) -> Labels:
    """The labels at `path`, or a new empty set named `source` if there is no file yet. An
    existing file keeps its own source name, so several videos can share one file. A file that
    cannot be trusted raises `LabelsError`, and nothing here ever writes to it."""
    if not path.exists():
        return Labels(path, source)
    try:
        return _parsed(json.loads(path.read_text(encoding="utf-8")), path)
    except (OSError, ValueError) as exc:
        raise LabelsError(f"labels file {path} refused: {exc}") from exc


class ImageFolder:
    """The images directly in a folder, in file-name order; a frame's id is its file name."""

    def __init__(self, folder: Path) -> None:
        if not folder.is_dir():
            raise FrameReadError(f"{folder} is not a folder")
        self._folder = folder
        self.ids = tuple(
            sorted(p.name for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        )
        if not self.ids:
            raise FrameReadError(f"{folder} has no image")

    def read(self, position: int) -> np.ndarray:
        path = self._folder / self.ids[position]
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise FrameReadError(f"cannot read the image {path}")
        return pixels

    def close(self) -> None:
        """Nothing stays open between reads."""


class VideoFrames:
    """Every `every`th frame of a video, in ascending order; a frame's id is
    `<file name>#<frame index>`, and with `every` 1 a position is the frame index."""

    def __init__(self, path: Path, every: int = 1) -> None:
        self.name = path.name
        self._every = every
        self._capture = cv2.VideoCapture(str(path))
        self._next = 0  # the frame a read gives without seeking, far faster than seeking
        if not self._capture.isOpened():
            raise FrameReadError(f"cannot open the video {path}")
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            self.close()
            raise FrameReadError(f"the video {path} reports no frames")
        self.ids = tuple(f"{self.name}#{index}" for index in range(0, count, every))

    def read(self, position: int) -> np.ndarray:
        index = position * self._every
        if index != self._next and not self._capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise FrameReadError(f"cannot seek to the video frame {self.name}#{index}")
        self._next = -1  # after a failed read, the next read must seek
        ok, pixels = self._capture.read()
        if not ok:
            raise FrameReadError(f"cannot read the video frame {self.name}#{index}")
        self._next = index + 1
        return pixels

    def close(self) -> None:
        self._capture.release()


class _Session:
    """A position in a list of frames, and the image at it, read once per position."""

    def __init__(self, labels: Labels, frames: ImageFolder | VideoFrames, position: int) -> None:
        self.labels = labels
        self.frames = frames
        self.position = position
        self._shown: tuple[int, np.ndarray] | None = None

    def image(self) -> np.ndarray:
        """The current frame's pixels; `FrameReadError` if it cannot be read."""
        if self._shown is None or self._shown[0] != self.position:
            self._shown = (self.position, self.frames.read(self.position))
        return self._shown[1]

    def _step(self, frames: int) -> None:
        self.position = min(max(self.position + frames, 0), len(self.frames.ids) - 1)


class FrameSession(_Session):
    """Labelling single frames. It opens at the first unlabelled frame, or the last frame when
    all are labelled, so a last label can still be corrected."""

    def __init__(self, labels: Labels, frames: ImageFolder | VideoFrames) -> None:
        unlabelled = (i for i, id_ in enumerate(frames.ids) if id_ not in labels.frames)
        super().__init__(labels, frames, next(unlabelled, len(frames.ids) - 1))

    def press(self, keysym: str) -> None:
        """One key: an answer is saved, then it moves on. Other keys are ignored."""
        answer = _ANSWERS.get(keysym)
        if answer is not None:
            self.labels.set_frame(self.frames.ids[self.position], *answer)
            self._step(1)
        elif keysym == "Left":
            self._step(-1)
        elif keysym == "Right":
            self._step(1)

    def summary(self) -> str:
        labelled = sum(id_ in self.labels.frames for id_ in self.frames.ids)
        return f"{labelled} of {len(self.frames.ids)} frames labelled in {self.labels.path}"

    def lines(self) -> list[str]:
        """The three screen lines: where this is, how far along, and the keys."""
        ids = self.frames.ids
        frame_id = ids[self.position]
        current = self.labels.frames.get(frame_id)
        if current is None:
            label = "not labelled"
        elif current.flavour is None:
            label = current.label
        else:
            label = f"{current.label}, {current.flavour}"
        labelled = sum(id_ in self.labels.frames for id_ in ids)
        return [
            f"frame {self.position + 1} of {len(ids)}: {frame_id}",
            f"{labelled} labelled · this frame: {label}",
            "u unit · w white · p pink · e empty cup · n nothing · h hand · Left/Right · q stop",
        ]


class CupSession(_Session):
    """Counting whole cups through one video. It opens at the frame of the video's last cup."""

    def __init__(
        self, labels: Labels, video: VideoFrames, window: tuple[int, int, int, int] | None = None
    ) -> None:
        self.video = video.name
        self._window = window
        self._changed_nothing = False
        cups = labels.cups_of(video.name)
        super().__init__(labels, video, cups[-1].frame if cups else 0)

    def image(self) -> np.ndarray:
        """The frame, with the lane window drawn on a copy when there is one; the frame read
        from the video is never drawn on."""
        pixels = super().image()
        if self._window is None:
            return pixels
        x, y, w, h = self._window
        shown = pixels.copy()
        cv2.rectangle(shown, (x, y), (x + w, y + h), _WINDOW_COLOUR, 2)
        return shown

    def press(self, keysym: str) -> None:
        """One key: f, w, p, e or g answers at this frame (`Labels.answer_cup`), BackSpace removes
        this video's last cup, an arrow steps. Other keys are ignored. An answer that changed
        nothing is said on the screen until the next key."""
        self._changed_nothing = False
        if keysym in _CUP_STATES:
            changed = self.labels.answer_cup(self.video, _CUP_STATES[keysym], self.position)
            self._changed_nothing = not changed
        elif keysym == "BackSpace":
            self.labels.remove_last_cup(self.video)
        elif keysym in _CUP_STEPS:
            self._step(_CUP_STEPS[keysym])

    def _counts(self) -> str:
        states = [cup.state for cup in self.labels.cups_of(self.video)]
        white, pink = states.count(CupState.FULL_WHITE), states.count(CupState.FULL_PINK)
        full = states.count(CupState.FULL) + white + pink
        flavours = f" ({white} white, {pink} pink)" if white or pink else ""
        gaps = states.count(CupState.GAP)
        gap = f", {gaps} gap" if gaps else ""
        return f"{full} full{flavours}, {states.count(CupState.EMPTY)} empty{gap}"

    def summary(self) -> str:
        count = len(self.labels.cups_of(self.video))
        return f"{count} cups in {self.video} ({self._counts()}) in {self.labels.path}"

    def lines(self) -> list[str]:
        """The three screen lines: the frame as the file records it, the cups, and the keys."""
        cups = self.labels.cups_of(self.video)
        counted = f"{len(cups)} cups: {self._counts()}"
        if cups:
            counted += f" · last: cup {cups[-1].order}, {cups[-1].state}, frame {cups[-1].frame}"
        where = f"frame {self.position} of 0–{len(self.frames.ids) - 1}: {self.video}"
        here = [f"cup {cup.order}, {cup.state}" for cup in cups if cup.frame == self.position]
        if here:
            where += f" · this frame: {'; '.join(here)}"
        if self._changed_nothing:
            return [
                where,
                counted,
                "nothing changed: no cup on this frame, and it is before the last",
            ]
        return [
            where,
            counted,
            "f full · w white · p pink · e empty · BackSpace undo · "
            "Left/Right 1 · Shift 25 · q stop",
        ]
