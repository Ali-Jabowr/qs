# ACCEPTANCE-P05

## Automated Checks

- [ ] `make check` — 0 failures (the existing tests still pass); ruff 0 errors; mypy strict 0 errors on `src/`
- [ ] `make gui-check` — every `gui` test passes (a display is present on this laptop)
- [ ] `.venv/bin/pytest tests/test_v4l2.py tests/test_webcam.py tests/test_screen.py tests/test_camera_test.py -q`
      — every case in CONTRACTS §6 present and passing, with no camera attached
- [ ] `bin/qs --help` — lists `camera-test`
- [ ] `bin/qs camera-test --config config/station-A-bench.example.yml --frames 1 --no-screen; echo "exit=$?"`
      — with no webcam attached: one line starting `camera:` that names `/dev/video0`, then `exit=2`; no traceback
- [ ] `grep -rnE '^\s*(import|from) (socket|http|urllib|requests)' src/qs` — prints nothing
- [ ] `grep -rnE 'CAP_PROP_(EXPOSURE|GAIN|WB_|AUTO_WB|AUTOFOCUS|FOCUS|AUTO_EXPOSURE)' src/qs` — prints nothing
      (controls go through ioctl only)
- [ ] `cmp assets/DejaVuSans.ttf /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` — identical

## Manual Checks

- [ ] The control order and values in `apply_controls` match CONTRACTS §1.3 row for row.
- [ ] `capture()` can never return a frame stamped at or before `t0 + exposure`, nor one on another clock.
- [ ] The saved lock-test frames are PNG (lossless), not JPEG.
- [ ] No file under `docs/`, `decisions/` or `config/`, and no existing `src/qs` module, was modified.
- [ ] **Ali, with the webcam in the rig (not the executor):** the lock test itself — `bin/qs camera-test
      --config config/station-A-bench.yml`, room light switched twice; PASS accepts ADR-0010.

## Verdict

PASS only if every box is checked, except the last, which is Ali's and decides ADR-0010, not P05.
