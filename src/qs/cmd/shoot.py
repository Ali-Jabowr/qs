"""`qs shoot`: the guided dataset session (docs/13 §6, §8; ADR-0014).

Opens the camera through the `FrameSource` contract, shows the live view, and walks Ali through
the session's shot list one step at a time. SPACE takes the shot — captured with `capture()`,
never `preview()` (C9) — and saves it as a PNG under its rule name (docs/12 §2); a damage or
brush step is confirmed with SPACE and captures nothing. At the end it writes `session.yml` and
says to copy `raw/<session>/` to the phone now (T11). Q or closing the window stops, frames kept.
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from qs.config import DisplayConfig, StationConfig, load_config
from qs.contracts import Frame, FrameSource, TriggerEvent
from qs.errors import CameraSettingsDrift, CaptureError, ConfigError, LightingFault
from qs.screen import KeyPress, Screen
from qs.session import (
    PLAN_FILE,
    FrameLog,
    Product,
    SessionRecord,
    read_plan,
    save_frame,
    session_started,
    write_session,
)
from qs.sources.factory import SourceUnavailable, frame_source
from qs.sources.v4l2 import to_readback
from qs.splitplan import Plan, PlanError, Prompt, Shot, Step, shot_list

_TICK_S = 0.05
_GAP_MS = 300  # a press counts only this long after the previous step finished (docs/11 §3)
_SHOOT = "space"
_QUIT = ("q", "Q")


def best_before(text: str) -> date:
    """argparse type: a date as YYYY-MM-DD."""
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a date as YYYY-MM-DD") from None


def purchased_at(text: str) -> datetime:
    """argparse type: an ISO time with its UTC offset, so it is unambiguous in the record."""
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        value = None
    if value is None or value.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"{text!r}: an ISO time with its offset, e.g. 2026-09-25T09:10+03:00"
        )
    return value


@dataclass(slots=True)
class _Gate:
    """The 300 ms rule, on Tk's own clock: a press counts only if it came at least 300 ms
    after the previous accepted step finished. A held key's auto-repeat, a double press, and
    presses queued while a shot was being saved are all discarded."""

    finished_tk: int | None = None
    accepted_tk: int = 0

    def admits(self, press: KeyPress) -> bool:
        if self.finished_tk is None or press.tk_ms < self.accepted_tk:
            return True  # the first press, or Tk's 32-bit ms clock wrapped
        return press.tk_ms - self.finished_tk >= _GAP_MS

    def done(self, press: KeyPress, seen: float) -> None:
        """The step `press` started, seen at perf-counter `seen`, has finished now."""
        self.accepted_tk = press.tk_ms
        self.finished_tk = press.tk_ms + round((time.perf_counter() - seen) * 1000)


@dataclass(slots=True)
class _Progress:
    steps: tuple[Step, ...]
    plan: Plan
    index: int = 0
    frames: list[Frame] = field(default_factory=list)

    @property
    def total_shots(self) -> int:
        return sum(isinstance(step, Shot) for step in self.steps)

    def lines(self) -> list[str]:
        """The three screen lines: the instruction, the progress, the grey card."""
        step = self.steps[self.index]
        if isinstance(step, Prompt):
            instruction = f"{step.instruction} — SPACE when done"
        else:
            instruction = f"{step.instruction} — SPACE to shoot"
        order = self.plan.order
        started = {s.frame.unit for s in self.steps[: self.index + 1] if isinstance(s, Shot)}
        unit_i = sum(unit in started for unit in order)
        shot_k = len(self.frames) + (1 if isinstance(step, Shot) else 0)
        progress = f"shot {shot_k} of {self.total_shots} · unit {unit_i} of {len(order)}"
        return [instruction, progress, self._patch()]

    def _patch(self) -> str:
        if not self.frames:
            return "grey card: no frame yet"
        first, last = self.frames[0].reference_patch, self.frames[-1].reference_patch
        if last is None:
            return "grey card: not measured"
        text = f"grey card {last:.4f}"
        if first:
            text += f" · drift {(last - first) / first * 100:+.2f}% since the first frame"
        return text


def _source_for(
    make_source: Callable[[StationConfig], FrameSource], config: StationConfig
) -> FrameSource | None:
    try:
        return make_source(config)
    except SourceUnavailable as exc:
        print(f"camera: {exc}")
        return None


def main(
    args: argparse.Namespace,
    *,
    make_source: Callable[[StationConfig], FrameSource] = frame_source,
    make_screen: Callable[[DisplayConfig, str], Screen] = Screen,
) -> int:
    """Shoot the session; 0 done, 1 stopped before the end or refused, 2 could not run."""
    try:
        config = load_config(Path(args.config)).config
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    folder = Path(args.root) / args.session
    if not (folder / PLAN_FILE).is_file():
        print(f"plan: no {folder / PLAN_FILE} — run qs split-plan --session {args.session} first")
        return 2
    try:
        plan = read_plan(folder)
    except PlanError as exc:
        print(f"plan: {exc}")
        return 2
    if session_started(folder):
        # Resuming a session is the recovery ticket's; until then nothing on disk is touched.
        print(f"shoot: {folder} already has frames — refusing to shoot over a started session")
        return 1
    product = Product(
        brand=None if args.client_product else args.brand,
        flavour=args.flavour,
        shop=args.shop,
        best_before=args.best_before,
        purchased_at=args.purchased_at,
    )
    steps = shot_list(plan, config.lighting.geometry)

    source = _source_for(make_source, config)
    if source is None:
        return 2
    try:
        source.open()
    except KeyboardInterrupt:
        print("stopped by the operator while opening the camera")
        source.close()
        return 1
    except (CaptureError, CameraSettingsDrift) as exc:
        print(f"camera: {exc}")
        source.close()
        return 2

    screen: Screen | None = None
    try:
        readings = getattr(source, "readings", ())
        at_open = to_readback(readings) if readings else None
        try:
            screen = make_screen(config.display, f"QS — shoot {args.session}")
        except tk.TclError as exc:
            print(f"screen: {exc}")
            return 2
        progress = _Progress(steps, plan)
        with FrameLog(folder) as log:
            code = _shoot(source, screen, progress, folder, log)
        if code != 0:
            return code
        record = SessionRecord(
            session_id=args.session,
            geometry=config.lighting.geometry,
            product=product,
            device=config.camera.device,
            camera_at_open=at_open,
            plan=plan,
            frames=tuple(progress.frames),
        )
        try:
            path = write_session(folder, record)
        except OSError as exc:
            print(f"session: {exc}")
            return 1
        done = f"Done: {len(progress.frames)} frames and {path.name} in {folder}"
        backup = f"Copy {folder}/ to the phone NOW, before anything else (T11)"
        print(done)
        print(backup)
        _wait_for_any_key(screen, source, [done, backup, "any key closes"])
        return 0
    except KeyboardInterrupt:
        print("stopped by the operator — frames so far are kept")
        return 1
    finally:
        source.close()
        if screen is not None:
            screen.close()


def _shoot(
    source: FrameSource, screen: Screen, progress: _Progress, folder: Path, log: FrameLog
) -> int:
    """Walk the steps; 0 when every step is done, 1 if stopped."""
    gate = _Gate()
    steps = progress.steps
    announced = -1
    while progress.index < len(steps):
        current = steps[progress.index]
        if isinstance(current, Prompt) and announced != progress.index:
            print(current.instruction)  # in full: a long recipe may not fit the window
            announced = progress.index
        screen.tick(source.preview(), progress.lines())
        pressed = screen.keys()
        if screen.closed or any(press.keysym in _QUIT for press in pressed):
            print(f"stopped after {len(progress.frames)} frame(s) — kept in {folder}")
            return 1
        acted = False
        for press in pressed:
            if progress.index >= len(steps):
                break
            if press.keysym != _SHOOT or not gate.admits(press):
                continue
            seen = time.perf_counter()
            step = steps[progress.index]
            if isinstance(step, Shot):
                stopped = _take(source, step, len(progress.frames) + 1, folder, log)
                if isinstance(stopped, str):
                    print(stopped)
                    return 1
                progress.frames.append(stopped)
            gate.done(press, seen)
            progress.index += 1
            acted = True
        if not acted:
            time.sleep(_TICK_S)
    return 0


def _take(source: FrameSource, shot: Shot, seq: int, folder: Path, log: FrameLog) -> Frame | str:
    """Capture and save one shot; the frame, or the one line that stops the session.

    A refused capture stops the session with the frames so far kept. Retrying the same shot
    after a refusal is the recovery ticket's (#33); until then nothing unverified or stale can
    be saved as if it were good.
    """
    event = TriggerEvent(seq, time.perf_counter(), datetime.now(UTC), "shoot")
    try:
        frame = source.capture(event)
    except (CaptureError, CameraSettingsDrift, LightingFault) as exc:
        return f"camera: {shot.file} not taken: {exc} — stopped, frames so far kept"
    try:
        sha256 = save_frame(folder, shot.frame, frame.pixels)
    except OSError as exc:
        return f"save: {shot.file}: {exc} — stopped, frames so far kept"
    log.append(shot.frame, frame, sha256, "saved")
    return frame


def _wait_for_any_key(screen: Screen, source: FrameSource, lines: list[str]) -> None:
    """Keep the closing message on screen until a key is pressed or the window closed."""
    while True:
        screen.tick(source.preview(), lines)
        if screen.keys() or screen.closed:
            return
        time.sleep(_TICK_S)
