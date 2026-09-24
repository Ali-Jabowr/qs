# REVIEW-P05b, round 1: shoot tools (split-plan, shoot, shot list, import-phone, pilot) and the #35 prep

**Verdict: FIX.** 0 blocker · 2 major · 4 minor · 0 nit.

Branch `mvp/bench`, diff `master...mvp/bench -- src tests config` (29f022a … 9dce580, plus 43719b1).
Reviewer: a fresh Claude agent that did not write the code. No repo file was modified.

## What was run

- `make check`: exit 0. 337 passed, 6 deselected. ruff clean, strict mypy clean, 17 diagrams render.
- An end-to-end CLI run in `/tmp/claude-1000/…/scratchpad/e2e/` with Pillow JPEGs carrying EXIF
  (the test fixture's `write_photo`), 64x48, on a tmp phone config:
  - `split-plan --size lean` → 0. `--shot-list --geometry G2` → 0, 265 rows. Run again → 0, "identical".
  - `import-phone --check` (10 photos) → PASS, PASS, 0.
  - `--check --session` on 40 photos → `40 of 265 photos mapped — next: #41 u037 good 120°`, 0.
  - A full import of 265 photos → 0: 265 PNGs, `frames.csv`, `import.csv`, `phone/`, and a
    `session.yml` with the §8 keys and `verified: false`.
  - Pilot: `--kind pilot --size standard` → `--shot-list` (126 rows, lamp, damage and CHECK rows
    in the right places) → `--check --session` → import → 0, `kind: pilot`, `units.pilot`.
  - A DNG in the folder → exit 2 naming ADR-0022. A junk `.jpg` → `FAIL … unreadable`, 1.
  - Probes for F1–F5 below.
- Greps: there is no `socket`/`http`/`urllib`/`requests` import in `src/qs`, and no image file in
  the diff. `config/station-A-bench.yml` is #35's, which is in scope for that ticket.

## Spec: does each ticket's acceptance hold?

| Ticket | Acceptance | Proved by a test | Note |
|---|---|---|---|
| #31 split-plan | ✓ | ✓ (`test_names`, `test_split_plan`, 200 seeds) | |
| #32 shoot | ✓ | ✓ (`test_shoot`, fakes) | bench-pending, as the ticket says |
| #52 shot list | ✓ | ✓ (`test_shot_list`, 50 seeds) | see F1: the list's geometry is not tied to the import's |
| #53 import-phone | ✓ | ✓ (`test_import_phone`) | the EXIF lock fails closed: a missing tag, WB auto, zoom, off-grid exposure and size are all refused before any write. See F2–F5 |
| #34 pilot | ✓ | ✓ (`test_pilot`) | |
| #35 prep | ✓ | ✓ (`test_camera_test`, `test_config`, `test_webcam`) | |

**Same layout as `qs shoot`: yes.** It uses the same `save_frame` (exclusive create), `FrameLog`
with `FRAMES_COLUMNS`, and `write_session` with the same §8 keys. It adds `phone/` and `import.csv`,
as specified.

**CLAUDE.md §6: honoured.** There is no network import. No image is in git. `--from` is only
read. No write replaces a file: PNG, CSV, `session.yml`, `phone/` and `rejected/` are all
opened `x`. Fail-closed refusals exit 1 or 2, and nothing gets a PASS verdict by default.

## Findings

### F1 · major: the printed list's geometry and the import's geometry are never compared, so a whole session can be named under the wrong geometry

- **Where:** `src/qs/cmd/import_phone.py:275` and `:337`, which use `config.lighting.geometry`,
  and `src/qs/cmd/split_plan.py` `_shot_list`, which takes `--geometry` from the command line.
- **Scenario (reproduced):** the plan and `shots-G3.md` were printed with `--geometry G3`, and the
  phone config says `lighting.geometry: G2`. `qs import-phone` exits 0 and names all 265 frames
  `…_G2_…`, and `session.yml` says `geometries: [G2]`. Nothing warns. #51 item 10 makes G3 a likely
  choice: with one lamp in the house, the tie rule gives G3. The bench config that a phone config
  would be copied from says G2. `raw/` is never edited afterwards, so P08 builds the frames into
  the G2 dataset.
- **Fix:** for a dataset plan, the import (and `--check --session`) refuses unless
  `shots-<config geometry>.md` exists in the session. The refusal names the lists that do exist.
  Add a test with `shots-G3.md` and a G2 config that expects exit 1 and nothing written.

### F2 · major: after `--broken`, `--redo K` counts in the adjusted list, not in the printed rows, so frames are silently misnamed

- **Where:** `src/qs/phone.py:371–400` (`map_photos` numbers the shots of `shot.steps` after
  substitution) and `src/qs/cmd/import_phone.py:279`.
- **Scenario (computed on lean, seed 20260925):** `u045` (test) breaks after 2 photos, and spare
  `u053` takes its place. Printed row 200 is `u049_dmg_G2_r1`. In the adjusted list that shot is
  #202, and #200 is `u049_good_G2_r2`. Ali ticks `redo` on paper row 200 and passes `--redo 200`.
  The count still matches, so the import exits 0. But the real good r2 photo goes to `rejected/`,
  the first **damaged** photo is saved as `u049_good_G2_r2.png`, and the spoiled photo becomes
  `u049_dmg_G2_r0.png`. The paper list is the only numbering Ali has: the header says "tick `redo`
  on its row", and it has no rows for the spare. The pilot has the same problem, because
  `discard_broken` drops rows from the list.
- **Fix:** have `--redo` take the frame's file name (`--redo u049_dmg_G2_r1`), which cannot
  shift. Or map K through the printed `shot_list`/`pilot_shot_list` numbering, and refuse a
  numeric `--redo` beyond the first broken unit's position. Add a test with `--broken` plus a
  later `--redo` that checks the names.

### F3 · minor: byte-identical duplicate photos are imported as different frames

- **Where:** `src/qs/phone.py:78` (`list_photos`) and `:200` (`in_order`).
- **Scenario (reproduced):** Ali copies over USB at a CHECK row, then copies everything again at
  the end. The file manager's "keep both" creates `IMG_0010 (1).jpg`, which has the same bytes and
  the same EXIF time. `--check --session` counts it: `41 of 265 photos mapped — next: #42`, which is
  one ahead. In the final import, a duplicate plus one missing photo passes the count check. The
  run exits 0 with `u026_good_G2_r0.png` and `…_r1.png` identical, and about 190 frames after it
  are named one shot off. The only signal is a gap WARNING on another unit.
- **Fix:** refuse (exit 1, naming both files) any two photos with the same SHA-256. Also refuse
  any two with the same capture time including sub-seconds when sub-seconds are present.

### F4 · minor: an import that fails part-way leaves a session that no command can finish, with no instructions

- **Where:** `src/qs/cmd/import_phone.py:332–334` (and `:347` for `write_session`).
- **Scenario (reproduced, with `ENOSPC` injected at the 100th PNG):** the run prints `stopped
  part-way; what was written is kept`, exits 1, and leaves 99 PNGs, `frames.csv`, `import.csv` and
  `phone/`. A re-run is then refused: `already has frames — refusing to import over a started
  session`. The only way out is to delete files in `raw/<session>/` by hand, which is the thing
  §6 and docs/13 §6 forbid, and no message says which files. PNGs from 12 MP phone frames make a
  full disk a realistic cause.
- **Fix:** write into a staging folder, `raw/<session>/.import/`, and move everything into place
  only after `session.yml` is written. A stale staging folder can then be refused with one line
  that names it for deletion. At the least, print the exact paths to remove.

### F5 · minor: Ctrl-C during `qs import-phone` prints a traceback

- **Where:** `src/qs/cmd/import_phone.py:218` (`main` has no `KeyboardInterrupt` handler).
  `camera-test` and `shoot` both catch it and print one line.
- **Scenario (reproduced with an injected `KeyboardInterrupt` in the write pass):** a Python
  traceback, and a partial session as in F4. A 265-photo, 12 MP import takes minutes, so an
  interrupt on the day is plausible.
- **Fix:** catch `KeyboardInterrupt` in `main`, print `import: stopped by the operator — …`
  (with F4's recovery line), and return 1.

### F6 · minor: a flaky test compares a float clock reading for exact equality

- **Where:** `tests/test_webcam.py:193` (`frame.grabbed_at == base + 0.020`) and `:267`
  (`== base + 0.02`).
- **Scenario:** `FakeCapture.get(CAP_PROP_POS_MSEC)` returns `ts * 1000.0`, and
  `WebcamSource` divides by `1000.0` (`src/qs/sources/webcam.py:182`). `(x*1000)/1000 != x` for
  about 2.3% of `perf_counter()` values (measured over 10⁵ samples), so each of the two asserts
  fails about 1 run in 44. The observed failure was off by about 1e-11.
- **Fix:** `assert frame.grabbed_at == pytest.approx(base + 0.020, abs=1e-9)` in both places.
  The ordering assertions do not depend on exact float identity.

## For Ali, not findings

- There is no phone config yet (`camera.type: phone`, the phone's `width`/`height`, both ROIs in
  its pixels, `exposure_us: 20000` per #51 item 6, and **`lighting.geometry` equal to the list
  you print**, see F1). P05b may not touch `config/`, so it is yours before the day-zero test.
- Copy the phone folder once, into an empty `data/phone/<session>/`. Never merge copies (F3).

## Verdict

**FIX.** Fix F1 and F2 before the phone dataset shoot: each can silently mislabel frames that
cannot be reshot. F3–F6 are small and belong in the same round.

## Fixes (round 1)

Each finding was fixed test-first as its own commit on `mvp/bench`. **Round 2 has not run yet**: an independent reviewer still has to confirm PASS.

| Finding | Fix | Commit |
|---|---|---|
| F1 | Refuse a dataset import unless the printed list's geometry matches the config's | 78da152 |
| F2 | `--redo K` means row K of the printed list; a spare's photo is redone by name | 43d7e30 |
| F3 | Refuse a photo copied twice, matched by bytes or by sub-second capture time | 55d37df |
| F4 | Stage the import in `.import/` and move it in whole, `session.yml` last | f9cabf3 |
| F5 | Ctrl-C in `qs import-phone` prints one line saying what to do | 626d852 |
| F6 | Compare the webcam's kernel timestamp to 1 ns, not bit for bit | 2f7aca0 |

---

# REVIEW-P05b, round 2

**Verdict: PASS.** 0 blocker · 0 major · 0 minor · 1 nit.

Diff `2f7aca0^..626d852 -- src tests`. Reviewer: the Claude Code orchestrator of 2026-09-23 (a fresh
session that did not write the fixes). `make check` on a clean worktree of 0fbbe6e: exit 0,
356 passed, 6 deselected, ruff and strict mypy clean, 17 diagrams render.

| Finding | Fixed? | How it was checked |
|---|---|---|
| F1 | ✓ | `_printed_list` runs for every dataset plan before the redo, the read and `--check --session`. It refuses no list, the wrong list, or two lists, and each refusal names the fix. The pilot is exempt because it carries its own geometries. Test: `test_a_config_geometry_that_is_not_the_printed_lists_refuses_and_writes_nothing` |
| F2 | ✓ | `--redo K` goes through `_printed_rows`, which regenerates the list as printed, before any substitution. So row K is the paper's row K. A row whose unit broke first, or a row past the end, is refused. A spare's photo is redone by its frame name. `map_photos` now matches by file, so it cannot shift. Tests cover the round-1 scenario (broken unit plus a later redo), the spare by name, and the mid-session "next: #n" |
| F3 | ✓ | `refuse_duplicates` runs before `in_order`: SHA-256 of every file, then capture time to the sub-second, and only when SubSecTimeOriginal is present. `_taken` folds the sub-seconds into the datetime, so two burst photos in one second are not refused |
| F4 | ✓ | All writes go to `.import/`, and `write_session` writes there too. `commit_import` moves files in with link-then-unlink (never over a name), moves dirs by rename, and moves `session.yml` last. A leftover `.import/` is refused before `session_started`, with `_recovery` naming what to delete or move. The reported `session.yml` is printed by name only, so the staging path never leaks. Tests: ENOSPC while writing, a stop while moving in, and a finished import that leaves no staging |
| F5 | ✓ | `main` catches Ctrl-C and prints one line. Inside `_import` it gives the recovery line, and during `_report` it says the import is complete. Tests cover the import and `--check` |
| F6 | ✓ | `pytest.approx(…, abs=1e-9)` in both asserts |

## Nit

- **N1 · `src/qs/session.py` `commit_import`:** a stop between `os.link` and `item.unlink` leaves
  one file in both `.import/` and the session. `_recovery` then says "Move everything in .import/",
  and that one file collides. The window is one syscall, and the collision is a refusal, not an
  overwrite, so it fails safe. The recovery text could also say "skip any file already in
  place". Left as is.

## Verdict

**PASS.** P05b is done apart from #33 (webcam only, last).
