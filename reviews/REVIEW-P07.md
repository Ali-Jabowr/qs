# REVIEW-P07, round 1: Inspector, learned half (PatchCore, bundle, train, pilot-report, mvtec-check, AppearanceInspector, latency)

**Verdict: FIX.** 1 blocker · 1 major · 4 minor · 1 nit. The blocker and the major are both in `qs latency` (#50).
The safety path (`AppearanceInspector`, the window and mask, the bundle and the weights check) holds.

Spec #42, tickets #45 (e64e145), #46 (ee24670, 9fbbb7a, d2e4fe1), #47 (f624df0), #48 (211425f), #49 (524dacf, db3a484,
3753dc3) and #50 (f5af6bd) on `mvp/bench`. Reviewer: an independent Claude agent that wrote none of this code. Probes were
throwaway scripts under the job's scratch directory. Nothing in the repo was edited except this file.

## Findings

| # | Sev | Where | Failure | Fix |
|---|---|---|---|---|
| F1 | **blocker** | `src/qs/cmd/latency.py:166` | The Gate B line and the GPU bench line are both computed from `p95 = _percentile(backbone, 95)`, the **backbone** p95. ADR-0017 and `docs/15` §10 judge the **p95 inspection time** × 3. **Verified by probe:** a scorer reports backbone 100 ms and search 120 ms, and the whole `inspect()` really takes 226 ms. The report says `p95 × 3 = 300.0 ms ≤ 400 ms → PASS`, when the right figure is 680 ms, a FAIL. Search is the larger share on the real model (#46 measured 40.7 ms backbone against 126.3 ms search), so Saturday's S3.5 evidence would be a false PASS on a gate that ADR-0017 says "errs towards failing". The tests lock the error in: `test_the_gate_b_line_passes_at_p95_130` and `…_fails_at_p95_140` script the backbone times | Use the p95 of `inspect_ms` for the Gate B line and the N100 range. For the bench line, use the p95 of backbone + search per call (or `inspect_ms`, and say which). Rewrite the two gate tests so that the inspection time is controlled: a scorer that sleeps, or a patched `perf_counter` |
| F2 | major | `src/qs/cmd/latency.py:323-324` | The warm-up verdict is not excluded from the backbone and search samples. The scorer calls are 0 = `load`'s warm-up, 1 = the replay warm-up verdict, then 2…timed+1 = the timed verdicts. `backbone[1:timed + 1]` keeps call 1 and drops the last timed call. **Verified by probe:** a warm-up verdict scripted at 5000 ms with every other call at 10 ms gives a backbone p99 of 4550.9 ms in the report. The test helper repeats the same off-by-one (`tests/test_latency.py:38,119-120`, "the timed verdicts are calls 1 … count"), so `test_the_percentiles_are_numpy_linear` agrees with the bug | Slice `[2:timed + 2]`, or better, record each verdict's `Scored` timings beside its `inspect_ms` so the three samples can't drift apart. Fix the test helper to match |
| F3 | minor | `src/qs/inspector/patchcore.py:26-29` | Only `HF_HOME` is forced. `huggingface_hub` resolves its cache from `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`, and only then `HF_HOME/hub` (constants.py:169-181). If either variable is set in the shell, timm loads its weights from that cache while `load_backbone` hash-checked `.hf/hub`. **Verified by probe:** with `HF_HUB_CACHE=/…/othercache`, the hash check reads `QS/.hf/hub` and hf_hub loads from `othercache`. Neither variable is set on Ali's shell today, so this is a latent ADR-0020 hole, not a live one | Also set `os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")` and pop `HUGGINGFACE_HUB_CACHE` before the imports. Or hand timm the verified file (`pretrained_cfg_overlay={"file": path}`) so the checked file and the loaded file are the same by construction |
| F4 | minor | `tests/test_latency.py`, `tests/test_appearance_env.py` | Two #50 acceptance checks have no test. (a) "Scripted inspection times → p50/p95/p99 equal `numpy.percentile`": only the backbone and search rows are checked, and the inspection row, the one that matters after F1, is never asserted. (b) The `env` case "a real bundle, `--cpu --count 200` completes, p95 stated in the closing comment" doesn't exist, and the #50 closing comment gives no p95 | Control `inspect_ms` in the test (see F1) and assert the inspection row. Add the `env` case to `test_appearance_env.py` and post the p95 |
| F5 | minor | `src/qs/cmd/latency.py:318` | `device_label` comes from the `--cpu` flag, not from the backend. A config whose `inspector.backend` is `torch-cpu`, run without `--cpu`, runs on the CPU but gets a `-cuda.md` report with the bench line. Torch isn't pinned to 4 threads and there is no Gate B line. Found by reading the code; no probe | Take the label from `loaded.config.inspector.backend` after the override. Or refuse a `torch-cpu` config without `--cpu`, with one line telling Ali to use `--cpu` |
| F6 | minor | `src/qs/cmd/pilot_report.py:324-326, 497-512` | (a) A geometry with no localized `dmg` frame, or a session with no damaged units, makes `separation` raise "fold 0: no damaged score". That is caught as `ValueError` → exit 2, so **no report is written for any geometry**, including ones with good numbers. On Friday this would cost the whole report because of one bad geometry. (b) A non-finite map raises `InferenceError`, which subclasses `QSError` and not `RuntimeError`, and no handler catches it, so it prints a traceback against "no traceback". Found by reading the code | Catch the no-damaged-score case per geometry: mark that geometry "no separation: no damaged unit localized" and leave it out of the ranking. Add `InferenceError` to the one-line handlers |
| N1 | nit | `src/qs/inspector/pilot.py:54` | With zero spread and a **negative** gap, separation is `0.0`. That ranks above a geometry whose gap is less negative but has some spread (e.g. −1.0). Real PatchCore scores almost never have zero spread | Return `-inf` when `gap < 0` |

Fail-safe is not affected by any finding: F1 and F2 corrupt a report, and none of them can make a unit PASS.

## Fail-safe (C8) and the priority checks

- **AppearanceInspector can't return PASS without both checks.**
  - `load` sets `_loaded = None` first, so a failed load leaves `inspect` raising `InferenceError`.
  - Every bundle refusal is a `ConfigError`, and so is `torch-cuda` without CUDA, with no fallback.
  - The label comes from `derive_label((geometry, appearance), {"geometry", "appearance"})`, and both checks are always built.
  - A NaN or ±inf anywhere in the map raises before the score, after the float32 cast, so an overflow is caught too. A non-finite `pred_score` also raises.
  - Any other scorer exception propagates unwrapped.
  - A wrongly shaped map (e.g. `(1, H, W)`) fails the broadcast in `score_map` with `ValueError`. That is an exception, never a score.
  - The threshold can't be NaN, inf or ≤ 0, because the config loader refuses it (`config.py:511`).
  - The config binds `threshold.set_on.{model, backend, localize_threshold, mask_dilate_px}` to the inspector, so a threshold from another bundle is refused at config load.
  - The station never passes `validation=True` (`station.py:82`). Only `qs mvtec-check` loads with it.
- **Comparison.** `passed = score <= limit`, in the right direction, with the float32 score compared as a float. An at-threshold float32 rounding lands on DEFECT, the fail-safe side (#92 is open for P08).
- **Masking (ADR-0009).**
  - `cut_window` and `score_map` use the same `x0 = ⌊c + 0.5⌋ − w // 2` box.
  - The mask is 0 outside the frame and dilated with an elliptical `2·d + 1` kernel.
  - `Silhouette.mask` is the one filled contour, so dust blobs aren't in it.
  - The dust-at-reach test pins the edge exactly (score 5.0 at `d`, 1.0 one pixel further).
- **Bundle integrity.**
  - The `model_id` hash is checked before the YAML is parsed.
  - The keys are strict, and the backbone tag, backbone hash, layers, neighbours and normalisation are pinned.
  - `validation_only` must equal `data_source != own`.
  - The bank loads with `torch.load(weights_only=True)`.
  - The weights hash is checked before `PatchcoreModel` is built.
  - `anomalib==2.6.2` is pinned in `requirements.in` and `requirements.txt`.
  - Anomalib is imported only in `patchcore.py`, and that is tested.
  - There are no `socket`/`http`/`urllib`/`requests`/`huggingface_hub` imports in `src/qs`.
  - No command module imports torch at import time, so `--cpu`'s `CUDA_VISIBLE_DEVICES=""` really does precede torch. Checked: `import qs.cmd.latency, …mvtec_check, …train, …pilot_report` leaves `torch` out of `sys.modules`.
  - See F3 for the cache-variable gap.
- **`qs latency` (#50).**
  - The looping and the placement exclusion are right. The count is 2 per pass, and a full pass with no verdict stops with exit 1.
  - The `--cpu` override uses `dataclasses.replace` in memory, and the config bytes are unchanged (tested).
  - The CPU report carries no score, threshold or label.
  - The Gate B arithmetic is right in itself (`3·p95 ≤ 400`, N100 range `2·p95 … 3·p95`, fallbacks on FAIL, the "not a Gate B result" line under 200), but it is applied to the wrong sample (F1), and that sample includes the warm-up (F2).
- **pilot-report (#47).**
  - Leave-one-out is honest: each fold fits only on the other good units' `good` frames, and damaged units and `dmg` frames never enter a fit.
  - The gap, spread (n − 1), `⌈2 × median⌉`, the tie-to-two-decimals rule and the G2/G3 rule match assumption A1 (#43).
  - No pilot model is written.
- **mvtec-check (#48).** It follows A2 (#44): `bottle`, both AUROCs ≥ 0.98, and the whole image as the window with a mask of ones. It hides CUDA before torch is imported. A reused bundle loads with `validation=True`, and the refusal without it is really called and recorded. If an own-data bundle is passed, the second load succeeds and the command exits 2. That is safe.

## make check

`make check`: **751 passed, 1 failed, 14 deselected** (69 s). The run then stopped before the diagrams and lint steps.

- **The failure** is `tests/test_fault_injection_more.py::test_1_enospc_for_events_50_to_149_holds_every_row` (`99 >= 100`). That is P04 #75, not P07. It ran under load while other processes were busy, and it passed 3 out of 3 times when run alone. It is a timing flake for the P04 review, not a P07 defect.
- **Run separately:**
  - `ruff check src tests`: clean.
  - `mypy src`: 42 files, no issues.
  - `tools/check_diagrams.py`: 17 diagrams render.
- **The P07 test files alone:** 137 passed in 10.7 s. The slowest test takes 1.11 s (under 2 s).
- **`make env-check`** was not re-run by this review (GPU). The ticket comments report 12 passed.

## Acceptance check → test

| Ticket | Acceptance check | Test | Proves it? |
|---|---|---|---|
| #45 | centred box `(560,140,800,800)`, centroid within 0.5 px | `test_window.py::test_a_disc_at_the_frames_centre_gives_the_centred_window` | yes |
| #45 | 4 edges, black pad, in-frame part equals the frame | `…::test_a_disc_100_px_from_an_edge_gives_a_black_padded_window` | yes |
| #45 | one scale (r 150 against r 200) · pure-red normalisation | `…::test_every_unit_is_seen_at_one_scale`, `…::test_to_input_normalises_a_pure_red_window` | yes |
| #45 | constant map · dust at dilate reach · NaN/±inf | `…::test_a_constant_map_scores_c_on_the_dilated_silhouette`, `…::test_dust_just_outside_the_dilated_silhouette_cannot_raise_the_alarm`, `…::test_a_non_finite_map_value_raises_before_any_score` | yes |
| #45 | PNG levels 0/64/128/255/255, bad thresholds · overlay · purity · < 20 ms | `test_heatmap.py` (8), `test_window.py` purity and `test_the_preprocess_chain_fits_its_budget` | yes |
| #46 | train → exit 0, model_id, two files · exact keys · hash suffix, `-v2-` | `test_train.py::test_train_writes_a_bundle_…`, `…::test_the_bundle_records_…`, `…::test_a_second_run_is_v2_…`; `test_bundle.py` | yes |
| #46 | refusals exit 1 (p/c/x, dmg, 2 geometries, bad name, no unit), nothing left behind | `test_train.py::test_a_frame_that_does_not_belong_in_train_good_is_refused`, `…::test_a_frame_with_no_unit_is_refused` | yes |
| #46 | exit 2: config, CUDA "no CPU fallback", weights missing or another hash, before build | `…::test_an_invalid_config_cannot_run`, `…::test_torch_cuda_without_cuda_never_falls_back`, `…::test_weights_missing_or_another_hash_cannot_run`; `test_patchcore.py::test_load_backbone_refuses_*_before_building` | yes |
| #46 | `load_bundle` refusals, validation-only needs `validation=True` | `test_bundle.py::test_load_bundle_refuses_*`, `…::test_one_changed_byte_…`, `…::test_a_validation_only_bundle_needs_validation_mode` | yes |
| #46 | pin = manifest · anomalib in one module | `test_patchcore.py::test_the_pinned_hash_is_the_manifest_s_racm_in1k_blob`, `…::test_anomalib_is_imported_in_exactly_one_module`, `…::test_no_module_imports_a_network_library` | yes |
| #46 | env: tag and hash, `torch.equal` banks, seed 43 differs, < 3.5 GiB, one score | `test_patchcore_env.py` (5) | not re-run here |
| #47 | all 11 checks | `test_pilot_report.py` (21, listed in #47) | yes. F6 is a path the tests don't take |
| #48 | all 8 checks | `test_mvtec_check.py` (listed in #48) | yes |
| #49 | PASS verdict in full · at and above the threshold · r = 210 geometry DEFECT · empty floor | `test_appearance.py::test_a_good_disc_…`, `…::test_a_score_equal_…`, `…::test_a_score_above_…`, `…::test_the_float32_score_…`, `…::test_an_oversized_disc_…`, `…::test_an_empty_floor_…` | yes |
| #49 | NaN → `InferenceError`, `RuntimeError` unwrapped · `load` refusals · warm-up once · purity with `open` patched | `…::test_a_nan_map_…`, `…::test_a_non_finite_pred_score_…`, `…::test_any_other_scorer_exception_…`, `…::test_load_refuses_*`, `…::test_a_validation_only_bundle_…`, `…::test_torch_cuda_without_cuda_…`, `…::test_load_warms_up_once_…`, `…::test_inspect_is_pure` | yes |
| #49 | env: NFR-03 to 1e-6, the blot above every good disc, median inspect_ms | `test_appearance_env.py` (2) and the #49 comment (26.0 ms) | not re-run here |
| #50 | 20 timed, looping, empties excluded and counted | `test_latency.py::test_twenty_inspections_loop_and_exclude_the_empty_frames` | yes |
| #50 | percentiles equal `numpy.percentile` | `…::test_the_percentiles_are_numpy_linear` | **no**: the inspection row is untested, and backbone/search share F2's off-by-one (F4a) |
| #50 | CPU 130 → PASS, 140 → FAIL, N100 range, no score/label | `…::test_the_gate_b_line_*` | **no**: they prove the backbone-p95 gate (F1) |
| #50 | `--cpu`: env before the scorer, `torch-cpu`, config bytes unchanged | `…::test_the_cpu_override_is_observed_through_the_seam` | yes, through the seam. The 4-thread pin is inside `_hide_cuda`, which the seam replaces |
| #50 | GPU bench line, no Gate B · `--count 50` marked · exit 1 and 2 | `…::test_the_gpu_run_…`, `…::test_below_200_…`, `…::test_only_empty_frames_exit_1`, `…::test_a_missing_directory_exits_2`, `…::test_a_bad_config_exits_2` | yes, but the bench line is on the backbone (F1) |
| #50 | env `--cpu --count 200` on a real bundle | none | **missing** (F4b) |
| all | `--help` lists train, pilot-report, mvtec-check, latency | `test_*::test_help_lists_*` | yes |

## bench-pending

As the tickets say:

- S1 on the downloaded `bottle` (GPU, then `--device cpu --bundle`).
- The real pilot on Fri 25.09.
- `qs train` on the real `train/good`, `AppearanceInspector` under P08, and the S3.5 Gate B run on Sat 26.09.

The Gate B run must not be taken as evidence until F1 and F2 are fixed.

## Verdict

**FIX.** F1 and F2 before Saturday's Gate B proxy run. F3–F6 are cheap and belong in the same round. #45–#49 need no change for fail-safe.
