# REVIEW-P04, round 1: pipeline, state machine, replay

**Verdict: PASS** after the r1 fixes (see "Fixes (r1)"). Round 1 found 0 blocker · 0 major ·
3 minor · 3 nit.

Spec #68, tickets #69 (339f66b), #70 (4705947), #71 (2c06b58), #72 (31d7d83), #73 (e281221),
#74 (d9d5d98, 5814f27, fa71159, f750147) and #75 (282bfc5, 97b3ea5, ff43dee) on `mvp/bench`.
The reviewer is an independent Claude agent that wrote none of the code. The findings were
confirmed with throwaway probes outside the repo. The fixes are small: a few lines each in
`pipeline.py`, plus one test per finding.

**No PASS path reachable with today's Inspectors and sources ignores a failed or missing check.**
The step-6 guard, `derive_label`, the non-finite checks, the provenance checks and the
encode-failure path all end in NO_VERDICT(`inference`). An exception inside `inspect()` never
becomes a verdict. In FAULT, every trigger is a NO_VERDICT with the fault's reason. m2 and m3
are latent holes in that backstop. m1 is a replay determinism defect, and it fails safe.

## Findings

### m1 · minor · A FAULT during a replay lasts for the rest of the run, and wall-clock time decides how much of the run it takes

`src/qs/pipeline.py:314`, `:317-328` (the backoff runs on `self._clock()`), with `src/qs/station.py:152-166` (`Station.run`).

- **Scenario.** A replay directory holds three consecutive frames that `ReplaySource` refuses:
  flat, the wrong size, or a `frames.csv` hash mismatch. The third one puts the station in
  FAULT(capture). The first reopen is due 0.5 s of `perf_counter` time later, but replay polls
  events as fast as it can. A FAULT event takes about 25 µs (no capture and no image), so every
  remaining frame becomes NO_VERDICT(`capture`) before any reopen is attempted.
- **Probe.** 3 flat frames, then 20 000 good ones, through `qs station --replay` →
  `capture 20003`, `reconnect attempts 0`, `final state FAULT(capture)`, exit 1. With 5, 50,
  400 and 3000 good frames after the bad ones, the result is the same.
- **Why it matters.** The count depends on disk speed and not on the pixels, which breaks
  "same verdicts every time" (NFR-03/08, #68 user story 25) for P08 and for the demo fallback.
- **Why the tests miss it.** #73's `test_corrupt_frames_are_capture_and_three_in_a_row_a_fault`
  shows "recovery by reopen" only because it runs on `FakeClock(step=0.01)`. The real command
  has no such clock.
- **Fix.** When the source is a `ReplaySource`, make recovery deterministic. Either the
  backoff counts events instead of seconds, or a replay source that reopens cleanly returns to
  READY on the next tick. Add a test through `main` on the real clock.

### m2 · minor · The C8 guard accepts a Verdict whose `checks` can be iterated only once

`src/qs/pipeline.py:487-504` (`_guard`), `src/qs/contracts.py:174-193` (`derive_label`).

- **Scenario.** An Inspector returns `Verdict(PASS, checks=<a generator yielding geometry
  pass and appearance FAIL>)`. `derive_label` uses up the generator building `names`. Its
  `any(passed is not True …)` then runs over an empty generator and returns PASS. The guard's
  loop over non-finite values sees nothing, so the verdict is accepted.
- **Probe.** The record is `label=pass` with `checks=[]`: a silent pass with a failed check
  behind it.
- **Reach.** The current Inspectors build a tuple and strict mypy types `checks` as a tuple,
  so nothing reaches this today. But the guard exists to catch a buggy Inspector.
- **Fix.** `_guard` refuses when `checks` is not a `tuple` or any element is not a
  `CheckResult`. Alternatively, `derive_label` converts `checks` with `tuple(checks)` first.
  Add one parametrised test case.

### m3 · minor · `handle()` has no catch-all, so a bad Frame from a source escapes with no record

`src/qs/pipeline.py:384-398` (`handle`), `:411` (`frame.camera.verified` runs outside any `try`).

- **Scenario.** A source returns a `Frame` whose `camera` is `None` or another bad value.
  `handle()` raises `AttributeError`.
- **Probe.** 0 records, and the state is left at `INSPECTING`. The command's loop then ends
  in a traceback, which breaks C5 (one record per trigger) and the one-line-error rule.
- **Reach.** Only a source that breaks its contract triggers this. `_capture`'s own comment
  says such a source must become NO_VERDICT, not a crash.
- **Fix.** Wrap steps 2–7 of `handle` in `except Exception` that returns NO_VERDICT: `capture`
  before a frame is accepted, `inference` after. That makes C5 hold whatever a collaborator
  does.

### Nits

- **n1:** `_guard` (`pipeline.py:487`) accepts a PASS or DEFECT that carries a `reason`. The
  row is stored as `pass` with reason `lighting`, and the summary adds it to "no_verdict by
  reason" (`_finish`, `:587`). Fix: refuse `reason is not None` on a decided label.
- **n2:** The Pipeline never compares `frame.grabbed_at` with `event.t0`, as the #75 comment
  notes. C4 belongs to the source by design (docs/11 §3), so this is not a defect. A one-line
  `grabbed_at < t0 → capture` would add defence in depth.
- **n3:** docs/16 §2 start-up step 3 ("a value that cannot be read at all: start, with
  `CAMERA_SETTINGS`") takes effect at the first unverified frame, not at start. This makes no
  difference in replay. It is for P09 to settle with the live `WebcamSource`.

## Checked and found correct

- **Wayfinder defaults.** #62 (A1) is built as the relative band
  `expect·(1−band) ≤ patch ≤ expect·(1+band)`. The boundary is included, and `None` or NaN is
  out of band. #63 (A2) is built as one streak per step. Capture is reset only by an accepted
  frame (and by a reopen). Inference is reset only by an accepted and encoded verdict.
  `busy`, `lighting`, `camera_settings` and `placement` do not touch either streak.
- **States.**
  - Start-up order is check_config → `inspector.load` → sink/run → `open()` → preview patch.
    No evidence is written before `load` succeeds.
  - Drift on `open()` → exit 2 with the mismatch printed. Lost or capture failure → FAULT(camera).
  - Backoff is 0.5, 1.5, 3.5, 7.5, 15.5, 25.5, 35.5 s. Drift on reconnect → STOPPED, exit 2.
  - FAULT(inference) is never retried.
  - Stop runs `finish_run`, then `sink.close()`, then `source.close()`. It is idempotent.
    SIGINT is a stop request, and the previous handler is restored.
- **Exit codes.** 0 needs written = events, lost = rejected = pending = sink_errors = 0, and a
  final state of READY.
- **NFR-07 and docs/17 §6.** No network import in `src/qs`: the AST test, and a grep that
  finds none, not even under `sources/`. The offline subprocess test has a working audit hook.
  Tests 2 and 6 are shown able to fail. Bundle mismatch, validation-only and scorer-refusal all
  exit 2 before any evidence root exists. The append-only triggers raise on all four statements.
- **Scope.** No P04 commit touches `docs/`, `decisions/`, `config/`, `contracts.py`,
  `config.py` or `errors.py`.
- **Test times.** 102 P04 tests in 30.6 s. The slowest are #75 case 3 at 10.2 s (≤ 12 s
  allowed) and case 4 at 3.1 s. Everything else is ≤ 1.5 s.

## `make check`

`make check` exits 2, and **P04 is not the cause.**

- Project state, diagrams: pass.
- Tests: **752 passed, 14 deselected** in 61 s.
- `ruff`: 2 errors, both in the uncommitted `tests/test_latency.py`.
- `mypy`: 1 error, in the uncommitted `src/qs/cmd/latency.py`.

Both files belong to #50, which is in progress in parallel. With those two files excluded, ruff
passes and mypy passes on 42 files.

## Acceptance check → test

"Proves it" means the test would fail if the behaviour broke.

| Ticket · check | Test | Proves it |
|---|---|---|
| #69 5 PNGs → exit 0, 5 pass rows, provenance, JPEG hash, `ended_utc` | `test_station_cmd::test_five_frames_become_five_passing_records` | yes |
| #69 scripted `defect:appearance` → heatmap, `failed_on` | `::test_a_scripted_appearance_defect_has_its_heatmap` | yes |
| #69 name order, repeatable, different run IDs | `::test_replay_is_in_name_order_and_repeatable` | yes (only with no FAULT: see m1) |
| #69 `frames.csv` read-back, or `None` / unverified; patch measured from the pixels | `::test_frames_csv_gives_the_camera_values_and_its_absence_none`, `test_replay::test_a_good_frame_carries_…` | yes |
| #69 `CaptureError`: hash, size, flat, not PNG (and grey, duplicate row) | `test_replay::test_capture_refuses_a_frame_it_cannot_trust` | yes |
| #69 guard: mislabel, nonfinite, other hash/threshold/model | `test_pipeline::test_a_verdict_that_hides_…`, `::test_a_verdict_with_other_provenance_…` | yes (not the iterable case: m2) |
| #69 refusals → exit 2, one line, no evidence | `test_station_cmd::test_refusals_exit_2_…` | yes |
| #69 summary; one annunciator call per event | `::test_summary_and_one_annunciator_call_per_event` | yes |
| #69 `ReplayTrigger` never blocks, `exhausted`, aware UTC | `test_replay::test_the_trigger_gives_one_event_per_file_…` | yes |
| #70 one test per docs/11 §7 row | `test_pipeline_failures::test_a_capture_failure_…`, `::test_an_inspection_failure_…` | yes |
| #70 lighting boundary 0.62 × 1.05, just above, clear, `None` | `::test_the_lighting_band_includes_its_boundary`, `::test_the_inspector_is_never_called_out_of_band` | yes |
| #70 drift, then unverifiable | `::test_drift_then_an_unverifiable_camera` | yes |
| #70 heatmap encoder raises | `::test_a_heatmap_that_cannot_be_encoded_…` | yes |
| #70 annunciator raises × 10; sink raises | `::test_a_failing_annunciator_and_a_failing_sink_…` | yes |
| #70 NO_VERDICT provenance | `::test_every_pipeline_made_no_verdict_…` | yes |
| #70 mixed 20 events through `SqliteSink` | `::test_a_mixed_run_through_the_real_sink` | yes |
| #70 placement/lighting change no state | `::test_placement_and_lighting_change_…` | yes |
| #71 start-up refusals | `test_station_states::test_an_inspector_that_will_not_load_…`, `::test_a_readback_mismatch_at_start_…`, `::test_an_unknown_breach_action_…` | yes |
| #71 camera lost → backoff times → READY | `::test_camera_lost_at_start_reconnects_on_the_backoff`, `::test_camera_lost_mid_run_is_a_fault` | yes |
| #71 capture 2 + good + 3 → FAULT on the 6th, reopen | `::test_three_capture_errors_in_a_row_…` | yes |
| #71 inference × 3 → FAULT, never retried; A2 mix | `::test_three_inference_failures_…`, `::test_only_inference_events_count_…` | yes |
| #71 mismatch on reconnect → STOPPED | `::test_a_readback_mismatch_on_reconnect_…` | yes (Pipeline level; the command's exit 2 on this path is not tested through `main`) |
| #71 `request_stop`, SIGINT | `::test_request_stop_mid_replay_…`, `::test_sigint_stops_the_command_cleanly` | yes |
| #71 ends in FAULT(inference) → exit 1 | `::test_a_replay_ending_in_fault_inference_exits_1` | yes |
| #72 STORAGE | `test_pipeline_status::test_storage_is_raised_…` | yes |
| #72 LATENCY exactly 10% / above / cleared | `::test_latency_is_raised_above_ten_percent_…` | yes |
| #72 p50/p95 by hand | `::test_latency_percentiles_by_hand` | yes |
| #72 busy during an inspection / while STARTING | `::test_a_unit_trigger_during_an_inspection_is_busy`, `::test_a_trigger_while_starting_is_busy` | yes |
| #72 run counters in the DB and the summary; `camera_verified` | `::test_the_run_counters_in_the_database_…`, `::test_camera_verified_in_the_counters` | yes |
| #73 camera lost at 50 | `test_fault_injection::test_camera_lost_at_event_50_…` | yes |
| #73 read-only 50–149 | `::test_a_read_only_evidence_directory_…` | yes |
| #73 corrupt 20/40/60, then 100–102 → FAULT → reopen | `::test_corrupt_frames_are_capture_…` | on the fake clock only (m1) |
| #73 non-finite 30–32; placement every 10th; all in one run | `::test_non_finite_scores_…`, `::test_placement_faults_…`, `::test_all_of_it_in_one_run` | yes |
| #74 security 1–6, and each shown able to fail | `test_security.py` (11 tests; the `env` variant is deselected here) | yes |
| #74 appearance replay 4 pass / 1 defect / 1 placement | `test_station_appearance::test_appearance_replay_gives_decided_verdicts_…` | yes |
| #74 scorer `ConfigError` → exit 2, no CPU fallback | `test_security::test_a_scorer_factory_that_refuses_…`, `::test_torch_cuda_without_cuda_…` | yes |
| #75 cases 1–8 | `test_fault_injection_more::test_1_…` to `::test_8_…` | yes. None slipped. Case 3 asserts written + lost = events, and case 4 asserts a gapless 1…m, as the #75 comment explains |

## `bench-pending`

As #74 says: `qs station --replay` on Friday's frames with Friday's bundle, with its verdicts
compared against P08's `qs evaluate` on the same frames.

## Verdict (round 1)

**FIX.** Fix m1 to m3, each with its test, then re-run `make check` once #50's lint is clean.
The nits are optional. No re-review is needed if the fixes stay inside `pipeline.py` and its
tests.

## Fixes (r1)

Commits `9c0f825` (m2, m3, n1) and `4f706ef` (m1). Each new test failed before its fix and
passes after it, except the `not-a-CheckResult` case of m2, which the old path already
caught; it stays as a regression guard. The fixes touch only `src/qs/pipeline.py`,
`src/qs/station.py` and their tests. There are no `contracts.py`, `docs/`, `decisions/` or
`config/` changes.

| Finding | Outcome | Where | Test |
|---|---|---|---|
| m1 | **Fixed.** A replay's events are files, not units arriving in time. `Station.run` now ticks a replay on `Pipeline.reconnect_due`: one reopen attempt per tick while in a camera or capture FAULT, with no wait on wall time. Recovery is a function of the frames alone. A reopen that keeps failing still consumes one event per attempt, so the run always ends. Live triggers still tick on the clock. The reopen goes through `ReplaySource.open`, and every frame still goes through capture's checks, so nothing new can become a pass. | `src/qs/pipeline.py:205` (`reconnect_due`), `src/qs/station.py:163-169` | `tests/test_fault_injection.py::test_three_bad_frames_on_the_real_clock_recover_the_same_way_every_time`: `main` on `time.perf_counter`, 3 bad frames then 197 good ones, run twice. Seq 4–200 pass, 1 fault, 1 reconnect, READY, exit 0, and both runs agree record for record. Before the fix, 200 × capture, exit 1. |
| m2 | **Fixed.** `_guard` refuses `checks` that is not a `tuple` or that holds anything but a `CheckResult` → NO_VERDICT(`inference`). No contracts change was needed. | `src/qs/pipeline.py:530-535` | `tests/test_pipeline.py::test_checks_that_are_not_a_tuple_of_check_results_are_refused[generator-holding-a-failed-check \| list \| not-a-CheckResult]` |
| m3 | **Fixed.** `handle()` wraps steps 2–7 in `except Exception`. The result is one NO_VERDICT record with no frame attached: `capture` before step 3 accepted the frame, `inference` after. Each counts toward its streak, so three in a row are FAULT. The error is logged as one line with no traceback, and the state leaves INSPECTING. Step 3 now reads the read-back before it counts the frame as accepted. | `src/qs/pipeline.py:412-418`, `:431-433`, `:467-475` (`_unexpected`) | `tests/test_pipeline.py::test_a_frame_that_breaks_its_contract_is_one_no_verdict_record`: `camera=None` → one `capture` record and READY; a good frame then passes; three bad frames in a row → FAULT(capture); 5 events → 5 records, 5 shown. |
| n1 | **Fixed.** `_guard` refuses a PASS or DEFECT whose `reason` is not None → NO_VERDICT(`inference`). | `src/qs/pipeline.py:526-527` | `tests/test_pipeline.py::test_a_decided_verdict_that_carries_a_reason_is_refused[pass \| defect]` |
| n2 | **Left as is.** C4 belongs to the source by design (docs/11 §3). | — | — |
| n3 | **Left as is.** It belongs to P09, with the live `WebcamSource`. | — | — |

`make check`: **green**, with 759 passed and 14 deselected. ruff is clean, and mypy passes on 42
files. Under full-suite load, `test_fault_injection.py::test_a_read_only_evidence_directory_for_events_50_to_149`
failed once (`held` 2, not ≥ 100). It passed alone three times, and it passed in the full
`make check` run. The run never enters FAULT, so the m1 change is not on its path. It is a
sink-timing flake, like the known one in `test_webcam.py`.

**Verdict: PASS.**

## Flaky tests (r1b)

Three P04 fault-injection tests failed now and then under load and passed when run alone. In
each case the test's synchronisation was at fault, not the product, and only the tests changed:
`tests/test_fault_injection.py`, `tests/test_fault_injection_more.py` and two helpers in
`tests/station_fixtures.py` (`wait_for`, `every_queued_record_held`). Each fix waits on
something it can observe, with a 5 s or 8 s deadline, instead of relying on timing. Each test
still asserts every line it asserted before.

**#75 case 4, `test_4_a_process_killed_mid_run_leaves_evidence_that_verifies`**
- **Cause.** There were two causes, not just a late `send_signal`.
  - The stub decides all 600 replay events in about 0.2 s. After that, the child's lifetime
    is only the time the sink takes to drain them, a few seconds. So the child ends by itself
    soon after it starts.
  - The poller could not see the rows being written. The writer commits back to back and holds
    the database lock almost all the time, so the poller's `SELECT COUNT(*)` got "database is
    locked" on most tries. In a probe with no load, 5 of the 6 polls made after the database
    existed were locked out. The 6th saw 600 rows, after the run had ended.
  - As a result, the poller often first saw 100 rows only once the writer went idle, which is
    when the run had already ended. The kill then landed after `run.ended_utc` was written (5
    of the 9 failures), or after the child had already exited 0 (4 of 9).
- **Fix.**
  - The test now tracks progress with SQLite's file change counter: header bytes 24–27, which
    rise by one for each commit in rollback-journal mode. It reads them as plain bytes, with no
    lock, so the writer cannot starve it.
  - When the counter shows at least 100 rows (with one to spare), the test sends `SIGSTOP` and
    waits with `waitid(WSTOPPED | WNOWAIT)` until the child is frozen.
  - Then it checks that the counter is still below the run's end, and sends `SIGKILL` while
    the child is still frozen. That is the same as a kill at the instant the child stopped.
  - The kill therefore always lands mid-run. The post-mortem checks are unchanged: integrity,
    gapless `1…m` with `m ≥ 100`, `ended_utc` NULL, `verify`, orphans, and file hashes.
- **Before → after:** 11/20 → 20/20 with 4 CPU burners, and 14/20 → 20/20 with 24 burners.

**#75 case 1, `test_1_enospc_for_events_50_to_149_holds_every_row`, and #73's `test_a_read_only_evidence_directory_for_events_50_to_149`**
- **Cause.** Both failed the same way: `held` was 99 when it had to be ≥ 100.
  - A record counts as `held` only if the sink's worker made a failed attempt after that record
    was queued.
  - The hook ended the outage inside `show(149)`, straight after the last records of the outage
    were queued. A 100-event outage on the fake clock lasts a few milliseconds of wall time.
    Under load, the worker thread was not scheduled for another retry before storage came back.
  - The earlier "`held` 2" failure was the same thing, with a worker that was starved for
    longer.
- **Fix.**
  - The hook now only asks for the outage to end. The next tick, which runs outside
    `handle()`, first waits until `held` covers `pending`: the sink has failed on every record
    it still has queued.
  - Only then does the tick restore storage and wait for the backlog to drain. This models an
    outage that outlasts one retry, as a real one does. The NFR-01 timing assertion on
    `handle()` is untouched.
  - The same change fixes the same latent race in `test_all_of_it_in_one_run` (`held ≥ 50`)
    and in #75 case 2. Case 2's `held ≥ 250`, which had 10 rows of slack for this race, is now
    exact: `held == 260`.
- **Before → after, case 1:** 20/20 → 20/20 with 4 burners, and 19/20 → 20/20 with 24 burners.
- **Before → after, read-only:** 20/20 → 20/20 with 4 burners, and 18/20 → 20/20 with 24
  burners. The old read-only test also went 19/20 with 14 burners. With 4 burners on this
  12-core laptop, the old versions of these two tests did not fail.

`make check`: **green**. 786 passed and 14 deselected; ruff is clean; mypy passes on 45 files.
