# REVIEW-P09, round 1: the station — operator view, ScreenAnnunciator, KeyboardTrigger, `qs station`, the run report

**Verdict: PASS.** Round 1 found no blocking defect: 1 minor and 2 nits. C7 holds on every path probed. The screen shows green only for a PASS verdict of the unit whose number it carries. A key press makes at most one event, and every event gets one record. The one minor is a measured draw cost. It should be fixed before the live NFR-01 run, but it does not endanger fail-safe.

Spec #84, questions #85 and #94, tickets #86 (ee4b85d), #87 (d213b7c), #88 (7d33710), merge b0e718c, #89 (425e5fb), #90 (5055599), #91 (330b6ed), merge f3638c9, all on `mvp/bench`.
Reviewer: an independent Claude agent that wrote none of this code. Probes and mutations ran on throwaway copies under the job's scratch directory. Nothing in the repo was edited except this file.

## Findings

| # | Sev | Where | Failure | Fix |
|---|---|---|---|---|
| m1 | minor | `src/qs/operator_view.py:210` (`_paste_fitted`, LANCZOS), called from `:258` and `:264` on every refresh, `src/qs/station.py:347-356` | **Drawing costs more than docs/10's "Draw and show 50 ms", and the loop runs at about 15 Hz, not 33 Hz.**<br><br>**Measured:** `render()` at the example config's 1920×1080 takes **~60 ms**: live view plus verdict image, idle laptop, Pillow only, without Tk's `PhotoImage` or `update()`. The verdict image alone takes ~28 ms, and no pictures ~3 ms. `_refresh` redraws both pictures on every iteration.<br><br>So a press can wait up to ~65 ms before it is polled, and `show()` adds ~30 ms plus Tk. By estimate the budget of ~430 ms becomes ~520 ms. NFR-01 still has room, and it is measured conservatively (see below), so this cannot make NFR-01 look better. It is the thing most likely to eat the margin on the real run (#89 and #91 flagged it too) | Downscale the live picture with `cv2.resize(..., INTER_AREA)` to the box before Pillow (measured 0.9 ms instead of ~32 ms). Cache the fitted verdict image per `Shown` in `ScreenAnnunciator`, since it changes only on `show()` |
| n1 | nit (design) | `src/qs/keyboard.py:92` (`t0 = after_s`), `src/qs/sources/webcam.py` `capture()` | **`t0` is the end of the pump before the press, not the press.** This is spec #84 rule 5, chosen so that `total_ms` never understates. But `WebcamSource.capture` also uses `t0` as C4's "not before" bound, so it can take a frame grabbed up to one loop iteration **before** the physical press. That is ~30–65 ms, and at ~5 fps a frame lands in that window on roughly one press in three.<br><br>A moti cannot be swapped in that time, so a previous unit's frame cannot come through. C4's wording ("never a buffered frame from before the trigger") holds only up to that slack. Not verified on hardware | No change for the MVP. If it matters later, give `KeyPress` the clock at `_on_key` too, and pass the later bound to `capture` (a P02 `TriggerEvent` change) |
| n2 | nit | `src/qs/screen_annunciator.py:155` | A НЕТ РЕШЕНИЯ without a frame (`camera`, `capture`, `camera_settings`, `busy`) shows `№ —`. Two such verdicts in a row look identical, so user story 18 ("tell a new verdict from the previous one") does not hold for them. They are amber, so fail-safe is not affected. It follows the ticket, because `show(frame, verdict)` does not carry the event | Later: pass the seq through (for example `frame=None, seq=`), or keep the last seq + 1 |

## Priority 1: C7 and fail-safe on screen

Each item was checked by reading the code, by a probe, or by a mutation (on a copy).

- **Green only for PASS.**
  - `label_colour` returns `GREEN` only for `Label.PASS`, and everything else not DEFECT is amber (`operator_view.py`). The band and the panel squares both use it.
  - **Mutation:** making the fallback green fails 35 tests, including `test_a_verdict_is_never_shown_green_unless_it_passed` and the lighting-band test.
- **A failed `show()`.**
  - It presents plain amber, then sets `last_shown` to НЕТ РЕШЕНИЯ with that seq, then re-raises. The Pipeline counts it (F12), and the record is still written.
  - **Probe:** PASS on seq 1, then seq 2's render made to fail. The band was amber from that show on, and **no refresh after it drew green**. The record says pass.
  - **Mutation:** removing the `_last` fallback fails `test_a_failed_show_never_brings_the_previous_verdict_back`.
  - The newest panel square keeps the true label (PASS, green). That matches the record, and the spec asks for it.
- **A failed `refresh()`.** Plain amber, re-raised, counted as `screen_errors`, and the loop goes on. The next refresh redraws from the stored `_last`, which is always this run's last verdict or the amber fallback.
- **A failed `present()`.** If Tk itself cannot present, no amber can be drawn either. But the same failure also stops `update()` from delivering keys, so no new event can be made under a frozen screen. It is self-limiting, and not a finding.
- **FAULT.** The whole canvas is amber, with no band, no live view and no panel. `show()` during FAULT renders the FAULT screen (AC7). A FAULT is always entered through a NO_VERDICT `show()`, so after a reconnect to READY the band is amber, never the pre-fault green.
- **Stop and close.**
  - Window close → `request_stop()` → P04's stop (drain, `finish_run`), then `screen.close()`. SIGINT takes the same path.
  - While storage drains (up to 10 s), the undestroyed window still shows the last image. That image is the last unit's own verdict.
- **A press during INSPECTING or the 300 ms hold-off.** It is discarded with no record. By design (A3, docs/16 §2) the previous verdict stays up, labelled `№ k`, until the next event.
  - The hold-off ends ~0.3 s plus one iteration after the verdict appears. That is shorter than a unit swap, so a press discarded there can only be a second press on the same unit.
  - While the next unit is inspected (≤ ~0.6 s), the previous unit's ГОДЕН is visible. That is the design, and the `№` is the only cue. **Bench-pending:** Ali should judge it at the rig.
- **C5.**
  - `poll()` returns at most one event per call, and `keys()` drains, so no press is delivered twice.
  - Every returned event goes straight to `handle()`, which always returns and writes one record.
  - A press with `after_s` below the bound can never reach the Pipeline's `busy` rule: the bound is always later than `_last_finished`.
  - Records equal events in AC3 and AC6 (20 frames, real Pipeline, held key, no `busy`).

## Priority 2: the live station (#90)

- **Refusals.** Exit 2, one line, before any sink: `--screen` without `--replay`; a trigger other than keyboard; a stub inspector; `SourceUnavailable`, which covers the phone, with the `--replay <dir> --screen` command named (#85); `TclError` (ADR-0018 line).
- **Camera.**
  - `CameraLost` at open → FAULT(camera) screen, with P04's backoff reconnect and the live view back.
  - `CameraSettingsDrift` at open → exit 2, mismatch printed, screen closed. On a reconnect → the Pipeline stops, the loop breaks, exit 2.
- **REVIEW-P04 n3: settled.**
  - A `WebcamSource` whose `readings` from `open()` are not all readable raises `CAMERA_SETTINGS` from the first READY refresh, for the whole run. This is done in the runner, with no Pipeline change.
  - **Mutation:** disabling it fails `test_settings_that_cannot_be_read_at_open_start_with_the_banner`.
- **Close.** Window close in FAULT → clean stop, `ended_utc` set, exit 1 (AC8).
- **C9.** A preview reaches only `refresh()`'s live box. AC2 checks the inspector's frames, the stored JPEGs and every `last_shown`.
- **NFR-07.** No `socket`, `http`, `urllib` or `requests` in `src/qs`, and no `imshow` or `namedWindow`.
- **Identity.** No face, person or identity code. The report and counters hold no name.
- **Scope.** Nothing under `docs/`, `decisions/` or `config/`, and no P02 contracts, Pipeline, store or sink, changed between c7ce2d3 and f3638c9.
  - One P04 case was removed from `test_station_cmd.py`: "appearance without `--replay` is refused". That invocation is now the live station, so the removal is justified.

## Priority 3: the run report and NFR-01 (#91)

- **Press to verdict on screen.**
  - `t0` is the earliest the press can have been made. `total_ms` is taken in `_finish` after `show()`, whose `present()` has pumped Tk.
  - It includes the wait for the next poll and the whole draw, and it can only overstate.
  - Screen, trigger and Pipeline share one clock, and the test factory asserts it.
  - **Mutation:** `t0 = last_pump_end` fails 3 keyboard tests. The 200-event end-to-end test did not catch it, because its 400–450 ms window is wide.
- **The NFR-01 line.**
  - p95 is `numpy.percentile` (linear) over inspections, against a fixed 1000 ms, never `budget_ms`.
  - PASS or FAIL only when all of these hold: live, `appearance`, `torch-cuda` (`torch-cuda` without CUDA is refused at load, so it cannot silently run on the CPU), n ≥ 200, no FAULT.
  - Otherwise "not measured", with the first reason. Replay is always "not measured" (#85).
- **Inspections.** An inspection is PASS, DEFECT, or NO_VERDICT `placement` or `inference`, as in P04 and F8. A slow capture timeout is left out of the p95 by that definition, but three in a row are a FAULT, which voids the run.
- **The report file.** Numbers and IDs only, with no image path. It is opened `"x"`, so it is never overwritten (exit 1 if it would be). Headless replay writes none and creates no directory.
- **NFR-09.** The protocol is in #91's body and is Ali's to run live. Nothing in code records a tester.

## Priority 4: #94, rows with no source marker in `qs.db`

**Judgment: acceptable for the bench MVP. Close it with a P03 migration before any client-facing archive.**

- **Confirmed by probe.** A **headless** `qs station --replay` run has no `source_id` in `run.counters` even after a clean stop (`station.py:248` wraps the sink only with a screen). So its rows can never be told from live rows by the database. A killed screen replay loses the marker too. This is broader than the question states.
- **Why the risk is low now.**
  - The MVP's numbers do not come from `qs.db` rows: NFR-01 comes from the run report, which names the mode, and the eval numbers from `qs evaluate` on golden sets.
  - Phone replays also carry a different `config_hash` from webcam runs.
  - The residual case is a webcam-config replay mixed with live rows in one evidence root.
- **The fix.** A run has one source, so the cheapest sound fix is a `run.source_id TEXT NOT NULL` written when the run row is inserted, which survives a kill. A per-row column is not needed. In the meantime, the runner could add `source_id` to headless replay's counters with no Pipeline change.

## make check

`make check` (run by this review, at f3638c9): **1074 passed, 16 deselected** (94 s). Diagrams: 17 render. ruff: clean. mypy: 57 files, no issues. Exit 0. `test_webcam.py` did not flake this run.

The P09 files alone: 165 passed in 11 s. The slowest test is the 200-event NFR-01 run at 1.07 s, and every other one is under 1 s. The `gui` tests were not re-run: they need Ali's display, and the implementer reports them passing.

## Acceptance check → test

| Ticket | Acceptance check | Test | Proves it? |
|---|---|---|---|
| #86 | AC1 size, read-only inputs, deterministic | `test_operator_view.py::test_render_is_a_1280_by_800_rgb_image_in_every_mode`, `test_render_leaves_read_only_inputs_unchanged_and_is_deterministic` | yes |
| #86 | AC2 band colours; AC3 no green on any NO_VERDICT (C7) | `test_the_band_takes_the_last_verdicts_colour`, `test_no_verdict_never_draws_a_green_pixel` (7 reasons × 2 modes) | yes (mutation-checked) |
| #86 | AC4 word size, config words | `test_the_three_words_are_at_least_80_px_tall_and_fit_the_width`, `test_the_busy_word_fits_…`, `test_the_words_come_from_the_config` | yes |
| #86 | AC5 every string | `test_status_line`, `test_reason_line`, `test_fault_line`, `test_banner_line`, `test_panel_line`, `test_label_word` | yes |
| #86 | AC6 FAULT all amber, no blue, no green | `test_the_fault_screen_is_all_amber_with_nothing_else`, `test_the_three_fault_reasons_give_three_different_screens` | yes |
| #86 | AC7–AC11 live, verdict image, banner, panel, status bar | `test_the_live_view_is_drawn_only_when_ready` … `test_the_status_bar_differs_between_starting_and_ready` | yes |
| #87 | AC1–AC4 PASS, heatmap window, geometry detail, order | `test_screen_annunciator.py` (5 tests) | yes |
| #87 | AC5 NO_VERDICT amber, no image for frameless reasons | `test_no_verdict_is_amber_with_its_reason_and_never_green`, `test_a_no_verdict_without_a_frame_has_no_seq_and_no_image` | yes |
| #87 | AC6 C9; AC7 FAULT; AC8 recent; AC10 close | named in #87's comment | yes |
| #87 | AC9 drawing failure → all amber | `test_a_drawing_failure_leaves_a_plain_amber_screen_and_still_counts`, `test_a_failed_show_never_brings_the_previous_verdict_back` | yes (mutation-checked) |
| #87 | AC11 real `present()` | `test_screen.py` (`gui`) | not re-run here |
| #88 | AC1–AC10 the 300 ms rule, held key, batch, armed, limit, source_id | `test_keyboard.py` (11) | yes (the `t0` and hold-off mutations fail them) |
| #88 | AC11 real stamp; AC12 old screen tests | `test_screen.py` | AC12 yes; AC11 `gui`, not re-run |
| #89 | AC1 5 records, clean stop, `keys_discarded` | `test_five_presses_make_five_replay_records_and_a_clean_stop` | **partly.** `source_id` is checked on the records handed to the sink and in `run.counters`, not per row (#94) |
| #89 | AC2–AC5 bands, discarded presses, after the last frame, window close, lost rows | `test_station_screen.py` (5) | yes |
| #89 | AC6 no `busy` with a held key | `test_a_held_key_never_makes_a_busy_record` | yes |
| #89 | AC7 FAULT(inference); AC8 lighting banner | `test_three_inference_failures_put_up_the_fault_screen_and_it_stays`, `test_a_frame_out_of_the_lighting_band_is_amber_and_raises_the_banner` | yes |
| #89 | AC9 refusals; AC10 sleep; AC11 ЗАПУСК then close | `test_refusals_exit_2_…`, `test_each_screen_iteration_sleeps_out_the_rest_of_30_ms`, `test_headless_replay_never_sleeps`, `test_a_failed_start_up_shows_zapusk_then_closes_the_window` | yes |
| #90 | AC1 three live records | `test_three_presses_make_three_live_records` | yes (with the #94 caveat) |
| #90 | AC2 C9 | `test_a_preview_frame_is_shown_live_and_never_inspected_recorded_or_shown` | yes |
| #90 | AC3 camera missing → FAULT → back | `test_a_camera_not_found_at_start_shows_the_fault_screen_until_it_is_back` | yes |
| #90 | AC4 drift at open | `test_a_read_back_mismatch_at_open_stops_with_exit_2` | yes |
| #90 | AC5 and n3 `CAMERA_SETTINGS` | `test_an_unverified_read_back_raises_the_camera_settings_banner_for_good`, `test_settings_that_cannot_be_read_at_open_start_with_the_banner`, `test_settings_all_read_back_at_open_raise_no_banner` | yes (mutation-checked) |
| #90 | AC6 preview raises; AC7 refusals; AC8 close in FAULT | `test_station_live.py` (3) | yes |
| #91 | AC1 replay report, NFR-01 last line | `test_a_screen_replay_leaves_a_report_whose_nfr01_line_ends_the_summary` | yes |
| #91 | AC2 percentiles, inspection set | `test_station_report.py` (5) | yes |
| #91 | AC3 NFR-01 decisions | `test_nfr01_*` (8), `test_the_limit_is_nfr01s_not_the_budget` | yes |
| #91 | AC4 200 live presses → PASS 400–450 ms | `test_200_live_presses_on_the_gpu_config_pass_nfr01` | yes. Its window cannot catch an understated `t0`; the unit tests do |
| #91 | AC5 headless writes nothing | `test_headless_replay_writes_no_report` | yes |
| #91 | AC6 no image, never overwritten | `test_the_report_has_no_image_and_is_never_overwritten`, `test_the_report_holds_no_image` | yes |
| #91 | AC7 report counters = `run.counters` | `test_the_reports_counters_are_the_runs` | yes |
