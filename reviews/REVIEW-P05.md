# REVIEW-P05: the camera, the Screen and `qs camera-test`

**Verdict: PASS**, after one small round. Branch `exec/P05`.

**Who did what:**
- **Executor:** Claude Code (ADR-0021, the first plan run without OpenCode).
- **Review round 1:** a fresh Claude agent that had not written the code. It re-ran every check and
  probed edge cases in throwaway scripts. Verdict: FAIL-small.
- **Round 2:** Claude Code fixed the findings and re-ran every check.

## Round 1 findings

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | should-fix | `close()` released the `VideoCapture` while the grab thread could still be inside `read()`. OpenCV's V4L2 read blocks for up to 10 s by default, and a release during a read is undefined behaviour. A probe reproduced it. | Fixed. The read timeout is 1 s, and `close()` never releases a capture that is still being read. |
| 2 | should-fix (plan) | The first frame had 500 ms (`capture_timeout_ms`), but UVC webcams often need 0.5–1.5 s. Wednesday's run could have ended with `camera: no frame within 500 ms of opening`. | Contract amended: the wait is `max(capture_timeout_ms, 3000)`. |
| 3 | nit | After an unplug, the lock test kept looping for 5 minutes. | Fixed. It stops at `CameraLost`. |
| 4 | nit | Ctrl-C outside the wait printed a traceback. | Fixed. |
| 5 | nit | A `cv2.error` in the grab thread made later captures time out with the wrong message. | Fixed. The source is marked lost. |

**Declared deviations.** All six are accepted: STATUS.md, "Deviations". The undeclared ones the
reviewer found (`VideoWriter.fourcc`, an extra `report:` line, and others) are harmless.

## Checks re-run after round 2

`make check` exits 0, with 160 passed, ruff clean and strict mypy clean. `make gui-check` passes 1
test, and the 4 P05 test files pass 56. `bin/qs camera-test` with no webcam prints
`camera: /dev/video0: cannot open` and exits 2. Both greps find nothing. The font is identical. No
change under `docs/`, `decisions/` (except ADR-0021, this review's own record) or `config/`.

## ACCEPTANCE

| Check | |
|---|---|
| `make check`, `make gui-check`, P05 tests | ✓ |
| `bin/qs --help` lists `camera-test` | ✓ |
| No webcam → one `camera:` line, exit 2, no traceback | ✓ |
| No network imports, no OpenCV control properties | ✓ |
| Font identical | ✓ |
| Control order and values match §1.3 row for row | ✓ (reviewer and test) |
| `capture()` can never return a stale frame or one on another clock | ✓ (reviewer probe) |
| The frames are PNG | ✓ |
| No out-of-scope file changed | ✓ |
| **Ali: the lock test with the webcam in the rig** | open. It decides ADR-0010. |

## For Ali on Wednesday: the one thing the code cannot guard

uvcvideo answers some control reads from its **cache**. So "every control read back as set" can be
true while the camera's firmware quietly re-exposes. The real guard against a false PASS is the
**reference-patch criterion (< 1%)**. **Actually switch the room light on and off, twice, during the
5 minutes.** Otherwise the test proves nothing about the lock.

Run from the repo root, after `sudo apt install v4l-utils` and with the webcam plugged in:

```sh
v4l2-ctl --list-devices            # the webcam's /dev/videoN
cp config/station-A-bench.example.yml config/station-A-bench.yml   # set camera.device, the ROI of the grey card
bin/qs camera-test --config config/station-A-bench.yml
```

`OPENCV_LOG_LEVEL=WARNING bin/qs camera-test …` shows OpenCV's own warnings, for diagnosing a failed
run.
