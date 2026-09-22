# CONTEXT — P05

## Why the camera comes before the model

The shoot on Friday 25.09 is the one step this week that cannot be redone, and a camera that quietly
re-exposes for a dent ruins it without anyone noticing (risk R4). So before a single dataset frame:
prove the camera holds its settings. That proof is the **lock test** (`docs/13-data-collection.md`
§4, ADR-0010), which Ali runs tomorrow with `qs camera-test`. Pass → ADR-0010 is accepted. Fail →
the phone becomes the source, recorded as `unverified`.

## The rules that matter most, and why

- **C4 — never a stale frame.** `capture()` returns the first frame whose **kernel timestamp** is
  later than the trigger plus one exposure time. A frame buffered before the key press would carry
  the previous unit's image under this unit's name. The kernel timestamp (`CAP_PROP_POS_MSEC`) is on
  `CLOCK_MONOTONIC`, the same clock as Python's `time.perf_counter()` on Linux (ADR-0010). A stamp
  *in the future* therefore means the driver uses another clock, and every freshness check would
  pass by accident — so it is refused, not trusted.
- **FR-13 — read back, never assume.** Every control is switched to manual, then set, then read back.
  A value that reads back different is `CameraSettingsDrift`. A value that cannot be read at all is
  recorded as `None` and the frame's `camera.verified` is `False`: unverified, never assumed.
- **Order matters** (ADR-0010). Auto off first, then the value: an absolute value written while its
  auto control is on is ignored or refused.
- **Units** (ADR-0010). V4L2 exposure is in 100 µs units; the config holds microseconds. Power-line
  frequency is a menu: 1 = 50 Hz, 2 = 60 Hz. Everything else is the raw V4L2 value.
- **Corrupt frames are refused** (FR-12): wrong shape or dtype, or *flat* — every pixel within 1 grey
  level.
- **The reference patch is measured on every frame** (FR-14): mean luminance of the grey card's ROI,
  0–1. P05 only measures it; raising a lighting fault is the Pipeline's job (P04). The lock test's
  pass condition is that it varies by less than 1% over 5 minutes of room light switched on and off.

## Why controls go through ioctl, not OpenCV

OpenCV's `VideoCapture` is not thread-safe, and the grab thread is inside `read()` almost all the
time. Reading controls back on every `capture()` through `cap.get()` from the main thread would race
it. A second file descriptor on the same device, used only for `VIDIOC_G_CTRL`/`VIDIOC_S_CTRL`, is
safe: V4L2 controls belong to the device, not to the descriptor. It also gives exact raw V4L2 values,
with none of OpenCV's normalisation. The IDs and ioctl numbers are the kernel's own (linux/videodev2.h,
linux/v4l2-controls.h); CONTRACTS pins them so nothing is guessed.

## Why a seam for the camera and the controls

No webcam is attached while the executor works, and tests must not need one. `WebcamSource` takes
two keyword-only factories — one that opens the stream, one that opens the controls — defaulting to
the real ones. Tests pass fakes. C6 still holds: configuration comes from `StationConfig` alone; the
factories are wiring, not configuration.

## The Screen

ADR-0018: one Tk window; Pillow composes everything, `ImageTk` shows it, and the caller's loop calls
`tick()` so Tk never owns the loop. Key presses are queued with their Tk timestamps, which P09's
`KeyboardTrigger` will need for its 300 ms rule. `compose()` is a pure function so it can be tested
without a display; the Tk shell itself is tested under the `gui` marker. A display exists on this
laptop (`DISPLAY=:1`).

## What already exists

- **P01:** `.venv` with OpenCV 5.0 headless, numpy, Pillow, PyYAML; `make check` (ruff, strict mypy
  on `src/`, pytest excluding `env`).
- **P02:** `qs.contracts` (`TriggerEvent`, `CameraReadback`, `Frame`, `Health`, `FrameSource`),
  `qs.errors` (`CaptureError`, `CameraLost`, `CameraSettingsDrift`), `qs.config` (`load_config`,
  `StationConfig` with `camera`, `lighting.reference_patch`, `display`).
- `pytest` finds `qs` through `pythonpath = ["src"]`; the package is not installed into `.venv`.
  `bin/qs` sets `PYTHONPATH` the same way.
