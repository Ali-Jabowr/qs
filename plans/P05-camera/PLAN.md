# PLAN-P05: The camera — `WebcamSource`, the Screen, and `qs camera-test`

## Objective

Give the station its eyes and give Ali the camera lock test (`docs/13-data-collection.md` §4,
ADR-0010) for tomorrow. `WebcamSource` implements `FrameSource`: it sets every camera control to
manual and reads each back, runs a grab thread, and returns only a frame stamped after the trigger.
The base Tk `Screen` shows a live view and collects key presses. `qs camera-test` runs the lock test
and writes its report. No part of this plan needs a camera to be tested: the camera and its controls
are reached through two small seams that the tests replace with fakes.

## Inputs

- `CONTEXT.md`, `CONTRACTS.md` (this folder).
- `docs/11-interfaces.md` §2, §3 (C4, C6, C9), §5; `decisions/ADR-0010`, `ADR-0018`;
  `docs/13-data-collection.md` §3–§4; `docs/06-requirements.md` FR-12, FR-13, FR-14, NFR-07.
- P02 done: `src/qs/contracts.py`, `src/qs/errors.py`, `src/qs/config.py`.

## Tasks

1. **`src/qs/sources/__init__.py`** — empty.
2. **`src/qs/sources/v4l2.py`** — the `Controls` protocol, `V4L2Controls`, the control IDs,
   `ControlReading`, `apply_controls`, `read_controls`, `check_readings`, `to_readback`
   (CONTRACTS §1).
3. **`src/qs/sources/webcam.py`** — `CaptureLike`, `open_v4l2_capture`, `measure_patch` and
   `WebcamSource` (CONTRACTS §2).
4. **`src/qs/screen.py`** — `KeyPress`, `compose` and `Screen` (CONTRACTS §3).
5. **`assets/DejaVuSans.ttf`** — copied from `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`, and
   **`assets/DejaVuSans-LICENSE.txt`** copied from `/usr/share/doc/fonts-dejavu-core/copyright`.
   The config's `display.font` already names `assets/DejaVuSans.ttf`.
6. **`src/qs/__main__.py`** and **`src/qs/cmd/__init__.py`** (empty) and
   **`src/qs/cmd/camera_test.py`** — the `qs` command with one subcommand, `camera-test`
   (CONTRACTS §4).
7. **`bin/qs`** — the launcher script in CONTRACTS §4, executable (`chmod +x`).
8. **`pyproject.toml`** — add the `gui` marker and exclude it by default (CONTRACTS §5).
   **`Makefile`** — add a `gui-check` target (CONTRACTS §5). No other change to either file.
9. **Tests** — `tests/test_v4l2.py`, `tests/test_webcam.py`, `tests/test_screen.py`,
   `tests/test_camera_test.py`: every case in CONTRACTS §6.
10. **Run** every ACCEPTANCE check and write `STATUS.md` in this folder, with the **actual output**
    of each check pasted in fenced code blocks, not summarised.

## Constraints

- **No new dependencies.** OpenCV (headless), numpy, Pillow and PyYAML are locked; `tkinter`,
  `fcntl`, `struct` and `threading` are the standard library.
- **Controls go through `V4L2Controls` only** (ioctl on the device). OpenCV's `VideoCapture` is used
  for the stream alone — format, size, frames and their timestamps. Never call `cap.set`/`cap.get` for
  exposure, gain, white balance, focus or power-line frequency (CONTEXT explains why).
- **No network.** No `socket`, `http`, `urllib` or `requests` anywhere in `src/qs` (NFR-07).
- **No sudo, and do not install `v4l-utils`** — that is Ali's step. There is no webcam attached while
  you work; nothing in this plan may need one.
- **Do not open image files with your file-reading tool.** Tests make their own synthetic arrays.
- **Do not change `docs/`, `decisions/`, `config/`, or any existing `src/qs` module.** If a contract
  here contradicts P02's code or the docs, stop and report it in STATUS.md.
- Write files one at a time and keep each reasoning step short. Do not draft whole files in your
  reasoning: the output allowance per step is limited.

## Definition of Done

- `make check` green (it excludes `gui` and `env` tests); `make gui-check` green.
- Every case in CONTRACTS §6 exists as a test and passes, with no camera attached.
- `STATUS.md` complete, with pasted output; no uncommitted changes.

## Out of Scope

- `qs split-plan` and `qs shoot` — plan **P05b**.
- `KeyboardTrigger` and `ScreenAnnunciator` — P09. Raising a lighting fault from the patch reading —
  the Pipeline, P04. `PhoneSnapshotSource` — only if the lock test fails (ADR-0010).
