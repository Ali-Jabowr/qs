# REVIEW-P08, round 1: datasets, golden set, calibration, the INDICATIVE report

**Verdict: FIX.** Round 1 found 1 major, 2 minor and 2 nits. The major is on the path #78 tells Ali to take, so it could hit on Saturday.
The split holds, and so do the threshold rule, the statistics and the drift, pairs and latency defaults.

Spec #76, tickets #80 (6e35138), #81 (e1b61b0), #82 (7179b6a) and #83 (f90c605, dc39814, c7ce2d3) on `mvp/bench`.
Reviewer: an independent Claude agent that wrote none of this code. Probes were throwaway scripts under the job's scratch
directory. Nothing in the repo was edited except this file.

## Findings

| # | Sev | Where | Failure | Fix |
|---|---|---|---|---|
| F1 | **major** | `src/qs/cmd/evaluate.py:281`, `src/qs/eval/units.py:231,247-257` | **The failure.** The 5% target is not reachable, but 10% is (A2). #78 names this as Ali's way out: `--far 10`. With lean sizes (10 calib units, so at 5% a = 0), **one** geometry-flagged calib unit is enough to get here.<br><br>**Verified by probe:** one oval calib unit, then `qs calibrate --far 10` (exit 0), then `qs evaluate`.<br><br>(a) The report and CSV are written. Then `_summary` formats `at5.choice.value` = `None` and raises `TypeError`, **a traceback**. The contract is "one line, never a traceback", and the exit code comes from the exception, not from the stop lines.<br>(b) The stop line reads "dent recall at the 5% target is **0 of 10 (0.0%)**". That is false: the row says "not reachable, no recall". `_recall` counts `flagged(None)` as not caught, and `early_warning` reads that count.<br><br>**No test covers it.** The only not-reachable test builds a `TargetResult` by hand with `far_combined=None`, and never goes through `main()` | `_summary`: print the 5% row as "not reachable (A2)" when `choice.value is None`. `early_warning`: when the 5% target is not reachable, give its own stop line: "dent recall at 5% not measurable: geometry alone flags g of n". Keep exit 1, and don't state a recall. Add one `main()` test on an oval calib unit calibrated at `--far 10` |
| F2 | minor | `src/qs/eval/golden.py:458-483` | **Dataset copies are not re-hashed at write time.** The planned SHA-256 is never compared with the bytes actually written.<br><br>**Verified by probe:**<br>- A train frame edited between `plan_build` and `write_build` lands in `train/good` silently, and its bytes differ from `frames.csv`.<br>- A calib frame edited in that window: the golden verify refuses, but **the dataset tree is already renamed into place**. That is a half-built result, against "a refused build leaves nothing half-built".<br><br>**Impact is low.** The window is seconds on a single-user laptop. `qs evaluate` would also refuse later: the golden set stays correct, and a changed calib copy breaks the 1e-6 threshold recomputation | In `_write_tree`, compare `sha256(bytes)` with `planned.sha256` for every copy before it is written. Stage both trees, verify both, then rename both |
| F3 | minor | `src/qs/eval/golden.py:201-216`, `src/qs/cmd/evaluate.py:204-220` | **Completeness is checked against `session.yml`'s own lists, not the plan's.** The build refuses a missing frame of a listed unit. It does not refuse a missing listed unit.<br><br>**Verified by probe:** a `session.yml` with `placement: []` and one control removed, with those frames deleted, **builds** with 0 placement frames and 2 controls. `qs evaluate` then prints placement **PASS** vacuously (no rows means no bad rows).<br><br>`write_session` always writes the full lists, so this needs a hand-edited `session.yml`. Still, the spec's "refuse a missing … control pass or placement frame" isn't enforced at the unit level | Require `units.placement` = `x01…x10` and `units.control` = `plan.control`. Require every plan train/calib/test unit to be in its split, in `discarded`, or replaced by a spare. In `qs evaluate`, make a set with no placement rows a FAIL |
| N1 | nit | `src/qs/cmd/evaluate.py:134-170` | The calibration record is trusted without re-checking that its canonical hash matches `threshold.id`'s suffix, so a hand-edited record passes. The threshold itself is recomputed from the golden calib scores, so a wrong threshold still can't slip through | Recompute `canonical_hash(record)[:6]` and compare it with the id |
| N2 | nit | `src/qs/cmd/evaluate.py:223-230, 396-397` | `_report_path` only checks whether the `.md` exists. If a `.md` was deleted and its `.csv` kept, the `.csv` is overwritten. The `.csv` is also opened with plain `write_text`, not exclusive create | Check both suffixes, and open with `"x"` |

Fail-safe is not affected. No finding can make a unit PASS or make a rate look better: F1 errs toward a stop, F2 is caught at evaluate, and F3 needs a hand edit.

## The priority checks

- **The split is sacred.**
  - **Splits and prefixes.** The six splits are pairwise disjoint and prefixes are checked. A train, calib or test unit must come from the plan's same split or its spares. Control and spare units must come from the plan's own lists.
  - **Frames.** Only `saved` frames are hash-checked. `discarded`-status frames belong to discarded units only (`phone.py:435`), and they go nowhere.
  - **Golden contents.** The golden set is built only from calib, test, control and placement, so it never holds a train frame.
  - **`verify_golden`** runs on the temporary tree before the rename and again in `qs evaluate`. It checks:
    - the hash file against the manifest bytes;
    - the exact file set;
    - every frame's hash;
    - every name against its row;
    - roles against prefixes;
    - one role per unit;
    - no `p` unit.
  - **E1 at evaluate.** `qs evaluate` also checks E1 against the bundle, on both units and image hashes. So even a hand-made "golden set" can't score a training frame.
  - **No overwrite.** A frozen set is never overwritten: identical content is "already built", anything else is refused, and frames are 0444.
  - **What the hash can't catch.** Rewriting `manifest.csv` and `manifest.sha256` together can't be detected by the set itself. The hash is printed by `qs build` and in every report header, and it has to be written down (as #80 says).
- **Threshold.**
  - **The rule.** `choose_threshold` takes the (a − g + 1)-th highest score among the units geometry passed. That index is always in range for targets under 100.
  - **Ties** only lower k.
  - **Direction.** `flagged = score > threshold` matches the inspector's `score <= limit`.
  - **The written value.** It is the float32 score's float64. **Probe:** 20,004 float32 values all round-trip exactly through `yaml.safe_dump`/`safe_load`. So under #92's exact comparison, the highest calib unit passes.
  - **The id.** The `threshold_id` suffix is the canonical hash of a record with no clock in it, so it is reproducible (tested). It depends on the `--dataset` path as typed, which is deterministic.
  - **No overwrite.** `--out` must not exist. The output is checked by `load_config` before `os.replace`, and the input is only read.
  - **The degenerate-range refusal is sound.** It refuses exactly `area_px[0] >= area_px[1]`, which is the loader's own refusal. That happens only when σₙ = 0 and every unit's mean area is the same integer. Nothing is widened.
- **Statistics.**
  - `stats.py` is a byte-for-byte copy of `tools/bounds.py`'s two functions: one-sided, alpha 0.05. Recall gets the lower bound and false alarms get the upper bound. They are tested against the reference loaded by path.
  - Counts are per unit, and a unit is caught if any rotation is БРАК.
  - An unscored unit is missed for recall and left out of the false-alarm denominator, with the count shown.
  - **INDICATIVE** is in the title, along with E6 and "not Gate A and not a client claim".
- **#79 and #93 defaults, ADR-0014.** All followed:
  - shift = the range over the scored passes, the largest across the controls;
  - gap = the median dent paired difference;
  - drift when shift ≥ gap, gap ≤ 0, or either can't be measured;
  - pairs are damaged − good, with `—` when a state is unscored;
  - latency covers only the frames that reached the model, with placement faults counted apart.
- **One scoring path.** `score_frames` goes through P04's `Pipeline` with an in-memory sink, and checks that it gets one record per frame.

## make check

`make check` (run by this review): **908 passed, 14 deselected** (81.7 s). Diagrams: 17 render. ruff: clean. mypy: 53 files, no issues. Exit 0.

## Acceptance check → test

| Ticket | Acceptance check | Test | Proves it? |
|---|---|---|---|
| #80 | lean session: counts per folder, byte copies | `test_build.py::test_the_lean_session_builds_the_dataset_and_the_golden_set` | yes |
| #80 | golden contents, nine columns, hash recomputed | `test_the_golden_set_holds_the_manifest_and_its_hash`, `test_the_manifest_rows_recompute_from_the_files` | yes |
| #80 | raw session untouched; copies read-only | `test_the_raw_session_is_untouched_and_the_copies_are_read_only` | yes |
| #80 | second build is "already built"; another product | `test_a_second_build_is_already_built_and_changes_nothing`, `test_the_same_session_under_another_product_builds_a_second_dataset` | yes |
| #80 | the nine refusals, nothing created | `test_a_unit_in_two_splits_is_refused` … `test_a_golden_target_that_exists_with_a_different_manifest_is_refused` | yes. A missing unit is not covered (F3) |
| #80 | broken unit and spare; phone session | `test_a_session_with_a_broken_unit_builds_the_spares_split`, `test_a_phone_session_builds_…` | yes |
| #80 | verifier refusals (byte, manifest, removed, added, row) | `test_the_verifier_refuses_*` (5) | yes |
| #80 | exit 2; `--help` | `test_a_missing_session_folder_or_session_yml_cannot_run`, `test_the_cli_lists_build` | yes |
| #81 | `choose_threshold`: hand-worked cases, ties | `test_threshold.py` (8) | yes |
| #81 | happy path: config differs only in model and threshold; input unchanged | `test_the_synthetic_dataset_calibrates`, `test_the_tolerances_equal_derive_tolerances_on_the_same_frames` | yes |
| #81 | id format; suffix recomputed; same bytes twice | `test_the_threshold_id_and_its_record_are_reproducible` | yes |
| #81 | `score_frames` = direct inspect; NO_VERDICT(lighting) | `test_score_frames_agrees_with_a_direct_inspect`, `test_score_frames_returns_the_pipelines_no_verdict` | yes |
| #81 | the eight refusals, nothing written | `test_a_bundle_whose_units_include_a_calib_unit_is_refused` … `test_a_phone_dataset_with_a_webcam_config_is_refused` | yes |
| #81 | oval unit at far 10 | `test_the_same_oval_calib_unit_passes_at_far_10` | yes. It stops at calibrate; evaluate on that config crashes (F1) |
| #81 | exit 2 for a bad bundle; `--far 3`; `--help` | `test_a_tampered_bundle_cannot_run`, `test_a_validation_only_bundle_cannot_run`, `test_far_3_is_refused_by_argparse`, `test_the_cli_lists_calibrate` | yes |
| #82 | bounds = `tools/bounds.py` | `test_stats.py` | yes |
| #82 | aggregation rules | `test_units.py` (first 6) | yes |
| #82 | hand-counted operating table; 1% and 5% share one threshold | `test_evaluate.py::test_the_operating_table_is_the_hand_counted_one` | yes |
| #82 | header, INDICATIVE, E6, phone | `test_the_synthetic_set_evaluates`, `test_a_phone_set_names_the_phone` | yes |
| #82 | placement PASS / FAIL | `test_the_placement_line_fails_on_a_frame_that_localizes` | yes. The vacuous PASS is not covered (F3) |
| #82 | early warning at 6/10 and at 7/10 | `test_the_early_warning_fires_when_four_dents_are_missed`, `…_does_not_fire_at_seventy_percent` | yes |
| #82 | the six refusals | `test_a_changed_frame_is_refused` … `test_a_webcam_config_on_a_phone_set_is_refused` | yes |
| #82 | CSV rows; `-2`; no images | `test_the_csv_has_one_row_per_unit_and_state`, `test_a_second_run_the_same_day_writes_minus_2` | yes |
| #82 | exit 2; `--help` | `test_an_unloadable_config_cannot_run`, `test_torch_cuda_without_cuda_cannot_run`, `test_a_missing_golden_set_cannot_run`, `test_the_cli_lists_evaluate` | yes |
| #82 | not-reachable row (A2) | `test_a_not_reachable_target_says_so_and_has_no_rates`, `test_units.py::test_a_not_reachable_target_has_no_threshold_and_no_labels` | **partly.** Both are hand-built, and neither goes through `main()` (F1) |
| #83 | controls 3 × 3; the drift cases | `test_the_controls_table_is_three_by_three_with_no_drift`, `test_a_blotted_control_at_pass_3_drifts`, 6 hand-built drift tests | yes |
| #83 | pairs and medians | `test_pairs_are_damaged_minus_good_with_the_median_per_class` | yes |
| #83 | latency; the report named | `test_latency_is_*` (2), `test_a_cpu_latency_report_of_this_model_is_named` | yes |
| #83 | changed units; another golden set ignored | `test_changed_units_list_exactly_the_unit_that_moved`, `test_a_csv_from_another_golden_set_is_ignored` | yes |
| #83 | NFR-08 end to end; golden and raw untouched; nothing in the repo | `test_eval_chain.py::test_build_train_calibrate_evaluate_twice_reproduces_the_report` | yes |
