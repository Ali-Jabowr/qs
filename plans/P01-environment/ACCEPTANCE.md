# ACCEPTANCE-P01

## Automated Checks

- [ ] `make env` — completes; `pip check` reports no broken requirements
- [ ] `.venv/bin/pip list --format=freeze | grep -ci opencv` — prints `1`, and that line is `opencv-python-headless`
- [ ] `make env-check` — every `env` test passes, including the offline load and its negative twin
- [ ] `make check` — 0 failures; the 44 existing tests still pass; ruff 0 errors; mypy 0 errors
- [ ] `grep -c -- '--hash=sha256' requirements.txt` — greater than 0, and every requirement line has a hash
- [ ] `git check-ignore .venv wheelhouse .hf .torch` — all four ignored
- [ ] `make env-offline` into a fresh temporary venv — completes with no network access to an index

## Manual Checks

- [ ] `reports/env/<date>-environment.md` exists, is generated, and answers R17 and R18 in plain words.
- [ ] The torch CUDA build and the driver agree: the report shows `is_available() = True` on the RTX 3050.
- [ ] `THIRD_PARTY_NOTICES.md` covers every installed component named in `docs/15-model.md` §11, and
      carries the backbone-weights caveat unchanged.
- [ ] No file under `docs/` or `decisions/` was modified.
- [ ] STATUS.md states whether `v4l2-ctl` is present (Ali installs it; the executor does not).

## Verdict

PASS only if every box is checked. If R17 or R18 contradicts the design, the verdict is still
decided on these checks — the contradiction is a finding for the planner, not a failure of P01.
