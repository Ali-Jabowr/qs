# CONTRACTS — P05

Signatures are binding; bodies are yours. Every public function and class has a one-line docstring.
Every module starts with a docstring and `from __future__ import annotations`.

## 1. `src/qs/sources/v4l2.py`

### 1.1 Control IDs and ioctls — the kernel's own values (checked against linux/videodev2.h and
linux/v4l2-controls.h on this laptop, 2026-09-22)

| Name in code | Value | Meaning of the value we set |
|---|---|---|
| `CID_EXPOSURE_AUTO` | `0x009A0901` | menu: `1` = manual |
| `CID_EXPOSURE_ABSOLUTE` | `0x009A0902` | 100 µs units |
| `CID_GAIN` | `0x00980913` | raw |
| `CID_AUTO_WHITE_BALANCE` | `0x0098090C` | `0` = off |
| `CID_WHITE_BALANCE_TEMPERATURE` | `0x0098091A` | kelvin |
| `CID_FOCUS_AUTO` | `0x009A090C` | `0` = off |
| `CID_FOCUS_ABSOLUTE` | `0x009A090A` | raw |
| `CID_POWER_LINE_FREQUENCY` | `0x00980918` | menu: `1` = 50 Hz, `2` = 60 Hz |
| `VIDIOC_G_CTRL` | `0xC008561B` | |
| `VIDIOC_S_CTRL` | `0xC008561C` | |

`struct v4l2_control` is `struct.pack("=Ii", cid, value)`: an unsigned 32-bit id, a signed 32-bit value.

### 1.2 Types and functions

```python
class Controls(Protocol):
    def get(self, cid: int) -> int: ...            # raises OSError if the control cannot be read
    def set(self, cid: int, value: int) -> None: ...  # raises OSError if the driver refuses it
    def close(self) -> None: ...

class V4L2Controls:                                  # implements Controls
    def __init__(self, device: str) -> None: ...     # os.open(device, os.O_RDWR); OSError → CameraLost(f"{device}: {e}")

@dataclass(frozen=True, slots=True)
class ControlReading:
    name: str             # see the table in 1.3
    wanted: int           # in config units
    reported: int | None  # read back, in config units; None if it could not be read

    @property
    def ok(self) -> bool: ...   # reported == wanted

def apply_controls(controls: Controls, camera: CameraConfig) -> tuple[ControlReading, ...]: ...
def read_controls(controls: Controls, camera: CameraConfig) -> tuple[ControlReading, ...]: ...
def check_readings(readings: tuple[ControlReading, ...]) -> None: ...
def to_readback(readings: tuple[ControlReading, ...]) -> CameraReadback: ...
```

`V4L2Controls.get` packs `(cid, 0)`, calls `fcntl.ioctl(fd, VIDIOC_G_CTRL, buf)` with a mutable
`bytearray` so the kernel writes the value back, and unpacks the value. `set` packs `(cid, value)` and
calls `VIDIOC_S_CTRL`. `close` closes the fd and is safe to call twice.

### 1.3 The order, and what is read back

`apply_controls` calls `controls.set` in exactly this order (ADR-0010: auto off first, then the
value), catching and ignoring `OSError` from any single `set` — the read-back will show it — and
then returns `read_controls(...)`:

| # | Control | Value set | `ControlReading.name` | `wanted` (config units) | `reported` conversion |
|---|---|---|---|---|---|
| 1 | `CID_POWER_LINE_FREQUENCY` | `1` if `power_line_hz == 50`, `2` if `60` | `power_line_hz` | `power_line_hz` | `1`→50, `2`→60, any other value unchanged |
| 2 | `CID_EXPOSURE_AUTO` | `1` | `exposure_auto` | `1` | raw |
| 3 | `CID_EXPOSURE_ABSOLUTE` | `exposure_us // 100` | `exposure_us` | `exposure_us` | raw × 100 |
| 4 | `CID_GAIN` | `gain` | `gain` | `gain` | raw |
| 5 | `CID_AUTO_WHITE_BALANCE` | `0` | `auto_white_balance` | `0` | raw |
| 6 | `CID_WHITE_BALANCE_TEMPERATURE` | `wb_kelvin` | `wb_kelvin` | `wb_kelvin` | raw |
| 7 | `CID_FOCUS_AUTO` | `0` | `focus_auto` | `0` | raw |
| 8 | `CID_FOCUS_ABSOLUTE` | `focus` | `focus` | `focus` | raw |

- `power_line_hz` other than 50 or 60, or `exposure_us % 100 != 0`: `apply_controls` raises
  `CameraSettingsDrift` naming the key, before any `set`.
- `read_controls` reads all eight with `controls.get`, in the same order; an `OSError` makes that
  reading's `reported` `None`. It never raises.
- `check_readings` raises `CameraSettingsDrift(f"{name}: set {wanted}, camera reports {reported}")`
  for the **first** reading whose `reported` is not `None` and differs from `wanted`. Unreadable
  readings do not raise.
- `to_readback` builds `CameraReadback(exposure_us, gain, wb_kelvin, focus, verified)` from the
  readings of those names (`None` where unreadable); `verified` is `True` only if **all eight**
  readings have a `reported` value.

## 2. `src/qs/sources/webcam.py`

```python
class CaptureLike(Protocol):                 # the slice of cv2.VideoCapture used here
    def set(self, prop: int, value: float) -> bool: ...
    def get(self, prop: int) -> float: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def isOpened(self) -> bool: ...
    def release(self) -> None: ...

def open_v4l2_capture(device: str) -> CaptureLike: ...
def measure_patch(pixels: np.ndarray, roi: tuple[int, int, int, int]) -> float: ...

class WebcamSource:                           # implements qs.contracts.FrameSource
    def __init__(
        self,
        config: StationConfig,
        *,
        open_capture: Callable[[str], CaptureLike] = open_v4l2_capture,
        open_controls: Callable[[str], Controls] = V4L2Controls,
    ) -> None: ...
    @property
    def readings(self) -> tuple[ControlReading, ...]: ...   # () before open(); set by open() even if it then raises
    def open(self) -> None: ...
    def capture(self, event: TriggerEvent) -> Frame: ...
    def preview(self) -> np.ndarray | None: ...
    def health(self) -> Health: ...
    def close(self) -> None: ...
```

**`open_v4l2_capture`** — `device` must match `^/dev/video(\d+)$`, else `CameraLost(f"{device}: not a
/dev/videoN path")`. Returns `cv2.VideoCapture(n, cv2.CAP_V4L2)`. (OpenCV 5.0's V4L2 backend does not
open a camera by path — checked on this laptop — so it opens by the number.)

**`measure_patch`** — ROI is `[x, y, w, h]`. Crop `pixels[y:y+h, x:x+w]`, convert with
`cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)`, return `float(mean) / 255.0`.

**`open()`**, in this order:
1. `cap = open_capture(camera.device)`; `not cap.isOpened()` → `CameraLost(f"{device}: cannot open")`.
2. `cap.set` `CAP_PROP_FOURCC` to `cv2.VideoWriter_fourcc(*camera.format)`, then `CAP_PROP_FRAME_WIDTH`,
   then `CAP_PROP_FRAME_HEIGHT`. Read the three back with `cap.get` (FOURCC decoded to 4 characters).
   Any mismatch → `CameraSettingsDrift(f"format: set {format} {w}x{h}, camera reports {fmt} {w2}x{h2}")`.
3. `controls = open_controls(camera.device)`; `readings = apply_controls(controls, camera)`; store them;
   `check_readings(readings)`.
4. Start the grab thread (daemon). Wait up to `capture_timeout_ms` for its first frame, else
   `CaptureError(f"no frame within {capture_timeout_ms} ms of opening")`.
If any step raises, release what was opened before re-raising.

**The grab thread** loops until `close()`: `ok, img = cap.read()`; `ts = cap.get(CAP_PROP_POS_MSEC) /
1000.0`. On success, make the array read-only (`img.setflags(write=False)`), append `(ts, img)` to a
`collections.deque(maxlen=8)` under a `threading.Condition`, reset the failure count, and notify. On
failure, count it; after **3 consecutive** failures mark the source lost, notify, and exit the loop.
`cap` is touched by this thread only, after `open()` returns.

**`capture(event)`**:
1. Lost → `CameraLost(f"{device}: camera stopped delivering frames")`.
2. `not_before = event.t0 + camera.exposure_us / 1e6`. Wait on the condition until the deque holds a
   frame with `ts > not_before`, until `event.t0 + capture_timeout_ms / 1000`. Take the **earliest** such
   frame. Timeout → `CaptureError(f"no frame newer than the trigger within {capture_timeout_ms} ms")`.
3. `ts <= 0` → `CaptureError("camera gives no kernel timestamps")`. `ts > time.perf_counter() + 1.0` →
   `CaptureError("kernel timestamp is not on the monotonic clock")`.
4. Shape other than `(height, width, 3)` or dtype other than `uint8` → `CaptureError("corrupt frame:
   <shape> <dtype>")`. `int(img.max()) - int(img.min()) <= 1` → `CaptureError("corrupt frame: flat")`.
5. `readings = read_controls(controls, camera)`; `check_readings(readings)` (drift raises).
6. Return `Frame(event, img, grabbed_at=ts, camera=to_readback(readings),
   reference_patch=measure_patch(img, lighting.reference_patch.roi),
   capture_ms=(time.perf_counter() - event.t0) * 1000)`.

**`preview()`** — the newest frame in the deque, or `None`. Never inspected (C9).
**`health()`** — `Health(ok, detail)`: `ok` if not lost, the thread is alive and the newest frame is
under 2 s old (`perf_counter() - ts`); `detail` says which of these failed, or `"streaming"`.
**`close()`** — stop the thread, join it (2 s), `cap.release()`, `controls.close()`. Safe to call twice.

## 3. `src/qs/screen.py`

```python
@dataclass(frozen=True, slots=True)
class KeyPress:
    keysym: str   # Tk keysym: "space", "q", "r", …
    tk_ms: int    # the Tk event's own timestamp, ms

def compose(
    image: np.ndarray | None, lines: Sequence[str], font_path: Path, size: tuple[int, int] = (1280, 800)
) -> PIL.Image.Image: ...

class Screen:
    def __init__(self, display: DisplayConfig, title: str) -> None: ...
    def tick(self, image: np.ndarray | None, lines: Sequence[str]) -> None: ...
    def keys(self) -> list[KeyPress]: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...
```

**`compose`** — an RGB canvas of `size`, black. The top `size[1] - 80` pixels are the image area: the
BGR image converted to RGB, scaled to fit with its aspect ratio kept (Pillow `LANCZOS`), centred. If
`image` is `None`, the text `"no camera image"` is drawn centred there instead. The bottom 80 pixels
are the text panel: the first **three** of `lines`, white, font `font_path` at 22 px, left-aligned
10 px from the left edge, one line every 24 px.

**`Screen`** — creates the Tk root with `title`, sized to 1280 × 800; binds `<KeyPress>` to append
`KeyPress(event.keysym, event.time)` to a queue, and `WM_DELETE_WINDOW` to mark itself closed.
`display.font` is resolved relative to the current working directory. `tick` composes, shows it
through `PIL.ImageTk.PhotoImage` in a `tk.Label` (keeping a reference so it is not garbage-collected),
then calls `root.update()`. It never calls `mainloop()`. `keys()` returns and empties the queue.
After `closed`, `tick` does nothing. `close()` destroys the root and is safe to call twice.

## 4. The `qs` command and `qs camera-test`

**`bin/qs`**, verbatim:

```sh
#!/bin/sh
here="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$here/src${PYTHONPATH:+:$PYTHONPATH}" exec "$here/.venv/bin/python" -m qs "$@"
```

**`src/qs/__main__.py`** — `argparse` with subcommands; `qs --help` lists them. Today there is one:
`camera-test`, dispatching to `qs.cmd.camera_test.main(args) -> int`; the process exits with its
return value.

**`qs camera-test`** options:

| Option | Default | |
|---|---|---|
| `--config PATH` | required | the station config |
| `--frames N` | `50` | frames in the test |
| `--interval-s S` | `6.0` | seconds between frames (50 × 6 s = 5 minutes) |
| `--out DIR` | `data/raw/<YYYY-MM-DD>-locktest` | where the frames go |
| `--report-dir DIR` | `reports/camera` | where the report goes |
| `--no-screen` | off | run without the window |

```python
@dataclass(frozen=True, slots=True)
class Sample:
    seq: int
    t0: float
    grabbed_at: float | None
    capture_ms: float | None
    reference_patch: float | None
    readback: CameraReadback | None
    error: str | None           # the exception text if the capture failed, else None

@dataclass(frozen=True, slots=True)
class LockTestResult:
    passed: bool
    lines: tuple[str, ...]      # one line per criterion: "PASS …" or "FAIL …"

def lock_test_verdict(
    readings: tuple[ControlReading, ...], samples: tuple[Sample, ...], frames_wanted: int, exposure_us: int
) -> LockTestResult: ...

def main(
    args: argparse.Namespace,
    *,
    make_source: Callable[[StationConfig], FrameSource] = WebcamSource,
    make_screen: Callable[[DisplayConfig, str], Screen] = Screen,
) -> int: ...
```

**`main`**:
1. `load_config`; `ConfigError` → print `config: <message>`, return 2.
2. `source = make_source(config)`; `source.open()`. `CameraLost`/`CaptureError` → print one line
   `camera: <message>`, return 2. `CameraSettingsDrift` → print the readings table and
   `FAIL <message>`, write the report with the controls section only, return 1. No traceback, ever.
3. Print the readings table: name, wanted, reported, `ok`/`MISMATCH`/`unreadable`.
4. For `seq` 1…N: until the next capture is due, keep the Screen alive with `tick(source.preview(),
   [f"Lock test — frame {seq} of {N}", "Switch the room light on and off twice during the test",
   "Q quits"])` about every 50 ms. Then capture with `TriggerEvent(seq, time.perf_counter(),
   datetime.now(UTC), "camera-test")`. Save the pixels losslessly with `cv2.imwrite(out /
   f"lock_{seq:02d}.png", frame.pixels)` and record a `Sample`. A `CaptureError` or
   `CameraSettingsDrift` becomes a `Sample` with `error` set; the test goes on. `q` or a closed window
   stops the test early. With `--no-screen` there is no window: the loop only sleeps.
5. `result = lock_test_verdict(...)`; write the report; print its lines; `source.close()`; return 0 if
   passed, else 1.

**`lock_test_verdict`** — one line per criterion, all judged (docs/13 §4):
- every reading `ok`;
- `frames_wanted` samples, none with `error`;
- every `readback.verified` is `True`;
- every `grabbed_at > t0 + exposure_us / 1e6`, and `grabbed_at` strictly increasing across samples;
- the reference patch varies by less than 1%: `(max - min) / mean < 0.01`, printed as a percentage;
- plus one line that is not judged: `CHECK BY EYE banding — look at the saved frames in <out>`.

**The report** — `<report-dir>/<YYYY-MM-DD>-lock-test.md`: the device, the format read back, the
readings table, each verdict line, the patch min/max/mean/variation, and one row per sample (seq,
capture_ms, reference_patch, error). Markdown, no images.

## 5. `pyproject.toml` and `Makefile`

- `[tool.pytest.ini_options]`: add the marker `gui: needs a display (run by make gui-check)`, and
  change `addopts` to `-m 'not env and not gui'`.
- `Makefile`: `gui-check:` runs `$(VENV)/bin/pytest -m gui`; add it to `.PHONY` and `help`.

## 6. Tests — none needs a camera

Fakes live in the test files: a `FakeControls` (a dict; records every `set` in order; can be told
to raise `OSError` for a given cid on `get` or `set`) and a `FakeCapture` (serves scripted frames
and `CAP_PROP_POS_MSEC` values; can be told to fail `read()`). No test may take longer than 2 s.

`tests/test_v4l2.py`
- `apply_controls` calls `set` in the exact order and with the exact values of §1.3.
- Exposure 10 000 µs is set as `100`; a read-back of `100` is reported as `10000`.
- Power line 50 is set as `1`; a read-back of `1` is reported as `50`; 55 Hz → `CameraSettingsDrift`.
- A `get` raising `OSError` → that `reported` is `None`, `to_readback(...).verified` is `False`.
- A `set` raising `OSError` does not stop the others.
- `check_readings` raises `CameraSettingsDrift` naming the first mismatched control; does not raise for
  an unreadable one.
- `V4L2Controls` with `fcntl.ioctl` and `os.open` monkeypatched: `get` sends `VIDIOC_G_CTRL` with
  `struct.pack("=Ii", cid, 0)` and returns the value the fake writes back; `set` sends `VIDIOC_S_CTRL`
  with `(cid, value)`.

`tests/test_webcam.py`
- `open()` sets FOURCC, width, height, then applies the controls; `readings` holds eight entries.
- The camera cannot be opened → `CameraLost`. A format read-back mismatch → `CameraSettingsDrift`.
- A control mismatch at open → `CameraSettingsDrift`, and `readings` is still set.
- `capture` skips frames stamped at or before `t0 + exposure` and returns the earliest later one.
- No fresh frame within the timeout → `CaptureError`.
- A flat frame and a wrong-shape frame → `CaptureError`.
- A timestamp of 0, and one more than 1 s in the future → `CaptureError`.
- A control changed after `open()` → `CameraSettingsDrift` on the next `capture`.
- Three consecutive `read()` failures → `CameraLost` on `capture`, and `health().ok` is `False`.
- `reference_patch` of a frame whose ROI is grey 158 is `158 / 255` (within 1e-6).
- The returned `pixels` are read-only.
- `preview()` is `None` before the first frame and an array after; `close()` twice is safe and the
  thread has stopped.
- `open_v4l2_capture("/dev/sda")` → `CameraLost`.

`tests/test_screen.py`
- `compose` returns `size`; a white image puts white at the centre of the image area; `None` still
  renders; given lines, the text panel has non-black pixels.
- `@pytest.mark.gui`: a real `Screen` — `tick` with an image; a key generated with
  `root.event_generate("<KeyPress>", keysym="space")` then `tick` → `keys()` returns one
  `KeyPress("space", …)` and is then empty; `close()` twice is safe.

`tests/test_camera_test.py`
- `lock_test_verdict`: everything good → passed; patch variation 2% → failed, the line names it; one
  stale sample → failed; a mismatched reading → failed; fewer samples than wanted → failed; a sample
  with `verified=False` → failed.
- `main` with a fake source and `--no-screen --frames 3 --interval-s 0` into `tmp_path` → writes
  `lock_01.png`…`lock_03.png` and a report, returns 0.
- `main` with the real `WebcamSource` and a config whose `camera.device` is `/dev/video99` → returns 2
  and prints one line starting `camera:`.
