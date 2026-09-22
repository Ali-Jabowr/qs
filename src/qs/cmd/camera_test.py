"""`qs camera-test`: the camera lock test (docs/13-data-collection.md §4, ADR-0010).

Opens the camera with every control forced to manual, captures a frame every few seconds
while Ali switches the room light on and off, saves each frame losslessly, and judges the
run: every control held, every frame fresh and verified, and the grey card steady to 1%.
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2

from qs.config import DisplayConfig, StationConfig, load_config
from qs.contracts import CameraReadback, FrameSource, TriggerEvent
from qs.errors import CameraLost, CameraSettingsDrift, CaptureError, ConfigError
from qs.screen import Screen
from qs.sources.v4l2 import ControlReading
from qs.sources.webcam import WebcamSource

_TICK_S = 0.05
_PATCH_LIMIT = 0.01
_BY_EYE = "CHECK BY EYE banding — look at the saved frames"


@dataclass(frozen=True, slots=True)
class Sample:
    """One lock-test capture, or the reason it failed."""

    seq: int
    t0: float
    grabbed_at: float | None
    capture_ms: float | None
    reference_patch: float | None
    readback: CameraReadback | None
    error: str | None


@dataclass(frozen=True, slots=True)
class LockTestResult:
    """The verdict: passed only if every judged line is a PASS."""

    passed: bool
    lines: tuple[str, ...]


def _status(reading: ControlReading) -> str:
    if reading.reported is None:
        return "unreadable"
    return "ok" if reading.ok else "MISMATCH"


def lock_test_verdict(
    readings: tuple[ControlReading, ...],
    samples: tuple[Sample, ...],
    frames_wanted: int,
    exposure_us: int,
) -> LockTestResult:
    """Judge every criterion of the lock test; one PASS or FAIL line each, and one by eye."""
    judged: list[tuple[bool, str]] = []

    bad = [f"{r.name} {_status(r)} (set {r.wanted}, reports {r.reported})"
           for r in readings if not r.ok]
    if not readings:
        judged.append((False, "controls: none were read back"))
    elif bad:
        judged.append((False, "controls: " + "; ".join(bad)))
    else:
        judged.append((True, f"controls: all {len(readings)} read back as set"))

    failed = [s for s in samples if s.error is not None]
    frames_ok = len(samples) == frames_wanted and not failed
    detail = f"frames: {len(samples)} of {frames_wanted} captured, {len(failed)} failed"
    if failed:
        detail += f" (first: #{failed[0].seq} {failed[0].error})"
    judged.append((frames_ok, detail))

    readbacks = [s.readback for s in samples if s.readback is not None]
    unverified = sum(1 for r in readbacks if not r.verified)
    judged.append((
        bool(readbacks) and unverified == 0,
        f"read-back: {len(readbacks) - unverified} of {len(readbacks)} frames verified",
    ))

    stamped = [s for s in samples if s.grabbed_at is not None]
    stale = [s.seq for s in stamped
             if s.grabbed_at is not None and s.grabbed_at <= s.t0 + exposure_us / 1e6]
    stamps = [s.grabbed_at for s in stamped if s.grabbed_at is not None]
    increasing = all(a < b for a, b in zip(stamps, stamps[1:], strict=False))
    if not stamped:
        judged.append((False, "freshness: no frame was captured"))
    elif stale or not increasing:
        problems = []
        if stale:
            problems.append("stale frame(s) " + ", ".join(f"#{n}" for n in stale))
        if not increasing:
            problems.append("timestamps not strictly increasing")
        judged.append((False, "freshness: " + "; ".join(problems)))
    else:
        judged.append((True, f"freshness: all {len(stamped)} frames newer than their trigger"))

    patches = [s.reference_patch for s in samples if s.reference_patch is not None]
    if not patches:
        judged.append((False, "reference patch: no readings"))
    else:
        mean = sum(patches) / len(patches)
        variation = (max(patches) - min(patches)) / mean if mean > 0 else float("inf")
        judged.append((
            variation < _PATCH_LIMIT,
            f"reference patch: varies {variation * 100:.2f}% (limit {_PATCH_LIMIT * 100:g}%)",
        ))

    lines = tuple(("PASS " if ok else "FAIL ") + text for ok, text in judged)
    return LockTestResult(passed=all(ok for ok, _ in judged), lines=(*lines, _BY_EYE))


def _readings_table(readings: tuple[ControlReading, ...]) -> list[str]:
    rows = [f"{'control':<20} {'wanted':>8} {'reported':>9}  status"]
    for r in readings:
        reported = "—" if r.reported is None else str(r.reported)
        rows.append(f"{r.name:<20} {r.wanted:>8} {reported:>9}  {_status(r)}")
    return rows


def _write_report(
    path: Path,
    config: StationConfig,
    readings: tuple[ControlReading, ...],
    lines: Sequence[str],
    samples: tuple[Sample, ...] | None,
    out: Path,
) -> None:
    camera = config.camera
    size = f"{camera.format} {camera.width}x{camera.height}"
    text = [
        f"# Camera lock test — {datetime.now().date().isoformat()}",
        "",
        f"- Device: `{camera.device}`",
        f"- Format read back: {size}" if samples is not None else f"- Format wanted: {size}",
        f"- Frames: `{out}`",
        "",
        "## Controls",
        "",
        "| Control | Wanted | Reported | Status |",
        "|---|---|---|---|",
    ]
    for r in readings:
        reported = "—" if r.reported is None else str(r.reported)
        text.append(f"| {r.name} | {r.wanted} | {reported} | {_status(r)} |")
    text += ["", "## Verdict", ""]
    text += [f"- {line}" for line in lines]
    if samples is not None:
        patches = [s.reference_patch for s in samples if s.reference_patch is not None]
        text += ["", "## Reference patch", ""]
        if patches:
            mean = sum(patches) / len(patches)
            variation = (max(patches) - min(patches)) / mean if mean > 0 else float("inf")
            text += [
                f"- min {min(patches):.4f} · max {max(patches):.4f} · mean {mean:.4f} · "
                f"variation {variation * 100:.2f}%",
            ]
        else:
            text.append("- no readings")
        text += ["", "## Samples", "", "| Seq | capture_ms | reference_patch | error |",
                 "|---|---|---|---|"]
        for s in samples:
            ms = "—" if s.capture_ms is None else f"{s.capture_ms:.1f}"
            patch = "—" if s.reference_patch is None else f"{s.reference_patch:.4f}"
            text.append(f"| {s.seq} | {ms} | {patch} | {s.error or ''} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def _wait(until: float, screen: Screen | None, lines: list[str], source: FrameSource) -> bool:
    """Keep the screen alive until `until`; False if the operator quit."""
    while True:
        remaining = until - time.perf_counter()
        if screen is not None:
            screen.tick(source.preview(), lines)
            pressed = screen.keys()
            if screen.closed or any(press.keysym in ("q", "Q") for press in pressed):
                return False
        if remaining <= 0:
            return True
        time.sleep(min(_TICK_S, remaining))


def main(
    args: argparse.Namespace,
    *,
    make_source: Callable[[StationConfig], FrameSource] = WebcamSource,
    make_screen: Callable[[DisplayConfig, str], Screen] = Screen,
) -> int:
    """Run the lock test; 0 on PASS, 1 on FAIL, 2 if it could not run."""
    try:
        config = load_config(Path(args.config)).config
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    today = datetime.now().date().isoformat()
    out = Path(args.out) if args.out else Path("data/raw") / f"{today}-locktest"
    report = Path(args.report_dir) / f"{today}-lock-test.md"

    source = make_source(config)
    try:
        source.open()
    except KeyboardInterrupt:
        print("stopped by the operator while opening the camera")
        source.close()
        return 2
    except CameraSettingsDrift as exc:
        readings: tuple[ControlReading, ...] = getattr(source, "readings", ())
        print("\n".join(_readings_table(readings)))
        line = f"FAIL {exc}"
        print(line)
        _write_report(report, config, readings, [line], None, out)
        print(f"report: {report}")
        source.close()
        return 1
    except CaptureError as exc:
        print(f"camera: {exc}")
        source.close()
        return 2

    screen: Screen | None = None
    try:
        readings = getattr(source, "readings", ())
        print("\n".join(_readings_table(readings)))
        if not args.no_screen:
            try:
                screen = make_screen(config.display, "QS — camera lock test")
            except tk.TclError as exc:
                print(f"screen: {exc} — run with --no-screen")
                return 2
        out.mkdir(parents=True, exist_ok=True)
        samples: list[Sample] = []
        start = time.perf_counter()
        for seq in range(1, args.frames + 1):
            status = [
                f"Lock test — frame {seq} of {args.frames}",
                "Switch the room light on and off twice during the test",
                "Q quits",
            ]
            event = TriggerEvent(seq, 0.0, datetime.now(UTC), "camera-test")
            try:
                if not _wait(start + (seq - 1) * args.interval_s, screen, status, source):
                    print(f"stopped by the operator after {len(samples)} frame(s)")
                    break
                event = TriggerEvent(seq, time.perf_counter(), datetime.now(UTC), "camera-test")
                frame = source.capture(event)
            except KeyboardInterrupt:
                print(f"stopped by the operator after {len(samples)} frame(s)")
                break
            except (CaptureError, CameraSettingsDrift) as exc:
                samples.append(Sample(seq, event.t0, None, None, None, None, str(exc)))
                print(f"frame {seq}: {exc}")
                if isinstance(exc, CameraLost):
                    break  # every later frame would fail the same way
                continue
            path = out / f"lock_{seq:02d}.png"
            saved = cv2.imwrite(str(path), frame.pixels)
            samples.append(Sample(
                seq, event.t0, frame.grabbed_at, frame.capture_ms, frame.reference_patch,
                frame.camera, None if saved else f"could not save {path}",
            ))
        result = lock_test_verdict(
            readings, tuple(samples), args.frames, config.camera.exposure_us
        )
        lines = [f"{line} in {out}" if line == _BY_EYE else line for line in result.lines]
        _write_report(report, config, readings, lines, tuple(samples), out)
        print("\n".join(lines))
        print(f"report: {report}")
        return 0 if result.passed else 1
    finally:
        source.close()
        if screen is not None:
            screen.close()
