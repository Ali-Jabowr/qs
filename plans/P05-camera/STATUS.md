# STATUS — P05

**Status: COMPLETE**

**Executor:** Claude Code (Opus 5.5), directly, on branch `exec/P05`. Not OpenCode. On 2026-09-22 Ali
asked for the cycle to go on without delegating to OpenCode. ADR-0019 is superseded by ADR-0021.

## Tasks

| # | Task | Done |
|---|---|---|
| 1 | `src/qs/sources/__init__.py`: empty | ✓ |
| 2 | `src/qs/sources/v4l2.py`: IDs, `Controls`, `V4L2Controls`, `ControlReading`, `apply_controls`, `read_controls`, `check_readings`, `to_readback` | ✓ |
| 3 | `src/qs/sources/webcam.py`: `CaptureLike`, `open_v4l2_capture`, `measure_patch`, `WebcamSource` | ✓ |
| 4 | `src/qs/screen.py`: `KeyPress`, `compose`, `Screen` | ✓ |
| 5 | `assets/DejaVuSans.ttf`, `assets/DejaVuSans-LICENSE.txt`, copied from the system | ✓ |
| 6 | `src/qs/__main__.py`, `src/qs/cmd/__init__.py`, `src/qs/cmd/camera_test.py` | ✓ |
| 7 | `bin/qs`, verbatim, executable | ✓ |
| 8 | `pyproject.toml`: the `gui` marker, `addopts` excludes it. `Makefile`: `gui-check` added to the targets, `.PHONY` and `help` | ✓ |
| 9 | Tests: `test_v4l2.py` (15), `test_webcam.py` (19), `test_screen.py` (6 + 1 gui), `test_camera_test.py` (13) | ✓ |
| 10 | Checks run; this file | ✓ |

## Automated checks (actual output)

`make check`: the test and lint lines, then the exit code:

```
157 passed, 6 deselected in 1.37s
All checks passed!
Success: no issues found in 11 source files
make check exit=0
```

`make gui-check`:

```
====================== 1 passed, 162 deselected in 0.48s =======================
make gui-check exit=0
```

`.venv/bin/pytest tests/test_v4l2.py tests/test_webcam.py tests/test_screen.py tests/test_camera_test.py -q`:

```
53 passed, 1 deselected in 0.99s
```

(The deselected test is the `gui` one, which `make gui-check` runs. The slowest test takes 0.50 s.
It waits out the 500 ms capture timeout on purpose.)

`bin/qs --help`:

```
usage: qs [-h] COMMAND ...

QS bench station.

positional arguments:
  COMMAND
    camera-test
               the camera lock test (docs/13-data-collection.md §4)

options:
  -h, --help   show this help message and exit
```

`bin/qs camera-test --config config/station-A-bench.example.yml --frames 1 --no-screen; echo "exit=$?"`, with no webcam attached (`/dev/video*` does not exist):

```
camera: /dev/video0: cannot open
exit=2
```

`grep -rnE '^\s*(import|from) (socket|http|urllib|requests)' src/qs`: no output.

`grep -rnE 'CAP_PROP_(EXPOSURE|GAIN|WB_|AUTO_WB|AUTOFOCUS|FOCUS|AUTO_EXPOSURE)' src/qs`: no output.

`cmp assets/DejaVuSans.ttf /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`: no output, exit 0 (identical).

## Manual checks (executor's own reading)

- Control order and values: `_table()` in `v4l2.py` lists the 8 rows of CONTRACTS §1.3 in order.
  `test_apply_controls_sets_every_control_in_order_with_its_value` pins the exact `set` sequence.
- `capture()` only takes frames with `ts > t0 + exposure` (strict). It refuses `ts <= 0` and
  `ts > perf_counter() + 1`. Tests cover a frame exactly at `t0 + exposure`, one inside the
  exposure, one at 0 and one 5 s in the future.
- The lock-test frames are written with `cv2.imwrite(... .png)`, which is lossless.
- `git diff master --stat` shows no file under `docs/`, `decisions/` or `config/`, and no existing
  `src/qs` module.
- Windowed smoke run: `main()` with a fake source, 3 frames 0.3 s apart and a real Tk window on
  `DISPLAY=:1`. It printed the readings table and five PASS lines, saved `lock_01…03.png` and the
  report, and exited 0.

## Deviations (for review)

1. **The `CHECK BY EYE` line names `<out>` in `main`, not in `lock_test_verdict`.** The binding
   signature of `lock_test_verdict` does not take `out`. The verdict returns
   `CHECK BY EYE banding — look at the saved frames`, and `main` appends ` in <out>` before it
   prints the line and writes the report.
2. **A zero timestamp is refused while waiting, not after it.** A frame stamped `0` can never be
   later than `t0 + exposure`, so CONTRACTS §2 step 3 could never see it. The wait would time out
   with the wrong message. If the newest queued frame is stamped `<= 0`, `capture()` raises
   `CaptureError("camera gives no kernel timestamps")` at once. Step 3's check stays as a guard.
3. **`main` handles three cases the contract does not name.** The rule is "no traceback, ever".
   - `tk.TclError` when the Screen is created prints `screen: <message> — run with --no-screen`
     and returns 2.
   - Ctrl-C during the wait stops the test as `Q` does, and the verdict is still written.
   - The source is closed on both early-return paths. Closing twice is safe.
4. **`qs.__main__.run()` sets OpenCV's log level to ERROR.** Without it, OpenCV prints two
   `[ WARN ]` lines to stderr beside the one `camera:` line when no camera is present.
5. **The report's "format read back" is the config's format.** It is printed only after a
   successful `open()`, which checks the read-back equals the config. `FrameSource` exposes no
   read-back of its own. On the drift path the report says "Format wanted" instead.
6. **`test_webcam.py` imports `FakeControls` from `test_v4l2.py`.** This avoids two copies of the
   fake. Both are test files, as CONTRACTS §6 asks.

## Blockers

None. The lock test itself is Ali's step: the last manual check in ACCEPTANCE, with the webcam in
the rig. It decides ADR-0010, not P05.
