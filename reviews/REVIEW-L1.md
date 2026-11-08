# REVIEW — L1 (#100–#105 + line lighting), `9868508...c7577d2` on `l1/line-score`

2026-11-06 · `/code-review`, two independent Claude reviewers (standards, spec). Findings below are the reviewers' claims, not verified by a test.

**Triage (Ali, 2026-11-08):**
- Spec #1 fixed (25b7461) · #2 kept (option A) · #3 fixed (11463ec) · #4 fixed (d7232e0, `l1/review-fixes`) · #5 deferred until #106 shows white and pink differ · #6 own ticket · #7 finish on `l1/line-example-config` (needs the unlock of `tests/test_line_inspector.py:254`) · #8 deferred, revisit if labelling for #106 hurts.
- Standards #9 fixed (d7f6dc7) · #1 fixed for the zone-in-frame check (951065b); the `direction`/`fps`/`line` re-checks stay, as they narrow optionals in components built without `load_config` · #2–8 and #10 later, in a refactor-only PR.

## Standards

**Hard violations:** none. Every commit has `Assisted-by`; no swallowed errors; refactor 7d98912 has no behaviour change; no loops or ifs in test bodies.

**Near-hard:** d95a095 and 7d98912 have no `Refs:` trailer; d95a095 changes FR-14 behaviour with no ticket (Ali's decision of 06.11 covers it).

Judgement calls (smells), most severe first:
1. Duplicated Code. `_validate_line` repeats the zone-in-frame check of `_validate`'s `rois` loop (`config.py:558-564` vs `618-625`). "Required" checks are also repeated: `direction` in `config.py:587` and `cup_trigger.py:107`; `fps` in `config.py:589` and `line_replay.py:212`; the `line` block in `config.py:404` and `inspector/line.py:65`.
2. Primitive Obsession. Cup states (`"empty"`, `"full_white"`, `"full_pink"`) and `"hand"` are bare strings, matched in `line_score.py`, `labels.py` and `score_line.py`.
3. Mysterious Name. There are two `Cup` classes: `qs.labels.Cup` (a hand-counted record) and `qs.inspector.cup.Cup` (a found rim).
4. Divergent Change. `labels.py` mixes four things: the file contract, the atomic writer, the frame readers and the UI sessions. As a result, `sources/line_replay.py` imports `VideoFrames` from the labelling tool.
5. Repeated Switches. `inspector_type == "line"` branches appear in `config.py:391-407`, and `trigger.type == "cup"` checks appear in `cmd/station.py`.
6. Middle Man. `_Files` (`line_replay.py:63-80`) only delegates, and fakes a `TriggerEvent` to pick a file. The two subclasses are constructor-only.
7. Duplicated Code. The poll → handle → advance loop is written three times: `score_line.py:64-98`, `station.py:285-296` and `test_line_station.py:153-160`.
8. Data Clump. `CupScore.white` and `.pink` are fixed fields.
9. `assert` narrows `fps` at runtime in `score_line.py:84`; `python -O` strips it.
10. Subcommand registration is inconsistent: `label` is registered inline in `__main__.py`, `score-line` through `add_parser`.

## Spec

All seven specs (#99–#105) were read.

Implemented, but the implementation looks wrong:
1. **#103.** A cup lost from view can be judged on its neighbour. A "lost" cup fires 0.25 s after it was last seen (`cup_trigger.py:78-81`). The inspector judges whichever cup is wholly in the zone in the captured frame (`inspector/line.py:76`). If the next cup is already in the zone, the lost cup gets that cup's verdict instead of NO_VERDICT(placement), and the next cup gets two verdicts. No test covers this.
2. **#103 — kept by Ali, 06.11 (option A): a duplicate is visible in `qs score-line`, a swallowed cup is not; measure on the footage.** A stopped chain can give a second verdict. A fired track is dropped after 0.25 s unseen, so glare hiding a stopped cup's rim for that long makes it a new track that fires again.
3. **#105.** A never-seen cup can take the next cup's record when W is about half the cup spacing and the label-to-record offset is systematic. Suggested fix: print the median offset, and warn when it exceeds W/2.

Missing or partial:
4. **#105.** Hand frames are counted only in videos that have counted cups.
5. **#105.** Only false БРАК is split by flavour. The never-seen and seen-but-not-judged counts are not.
6. **#104.** `--screen` is refused with a cup trigger, so the screen's ГОДЕН expiry is never exercised under a cup trigger.
7. **#103/#105.** `config/station-line.example.yml` still has `trigger: replay`, with no `direction` or `fps`. The new commands refuse it.
8. **#101.** A cup recorded at the wrong frame can only be fixed by undoing back to it, which removes every later cup.

Not asked for: `unpaired` (harmless). The rule that a line station lists exactly `[empty_cup]` must be loosened for #107.

Spec reviewer found no problem in: lighting null on line vs bench (FR-14); per-station checks with the bench default; placement wording; PASS-only expiry.

---

# REVIEW — review fixes + #109, `618740f...712e2fc` on `l1/line-appearance`

2026-11-08 · `/code-review`, two independent Claude reviewers (standards, spec). No hard violations; spec found nothing missing and no path to ГОДЕН without both checks.

**Fixed (5dfdfed, 09c3991):** standards #1 (one public copy of `real_scorer`/`real_cuda_available`, in `qs.inspector.appearance`), #3 (`_line` cleared on load), #4 (`LoadedModel` instead of a tuple); spec #4 (a non-finite map is an `InferenceError`, test guarded by a reverted mutant).

**Decided (Ali, 2026-11-08):** the mask stays `cup.inside` dilated by `mask_dilate_px` (12 px, as on the bench and as `qs train` saw it); measure the rim-glare false alarms in #110 before changing it. The approval of the two reworded `tests/test_config.py` tests goes in the PR text, not an amended commit.

**Left:** repeated constructor parameters across the two inspectors (data clump if a third appears); 951065b has no `Refs:`; d7f6dc7 does not mention the new `fps` property. `_cuda_available` is still written again in `cmd/mvtec_check.py`, `cmd/pilot_report.py` and `cmd/train.py`.

---

# REVIEW — #112, #116, #117 (2026-11-08, Claude orchestrating while Ali was away)

Each branch had one independent Claude reviewer, plus a short re-review of the fix commits.

**#112 → PR #121** (`434f907..1bd0fde`). The first review found two silent ways training data could still count as held-out: a config fps different from training's changes the gap in frames, and a relabelled file moves the train block. Both are fixed: the training fps is recorded as `line_fps` and a mismatch is refused (1bd0fde), and a train block that differs from the bundle's trained frames is refused (2553f07). The re-review gave PASS WITH NITS; the nits are #120.

**#116** (`l1/find-cup-size-reason`). The first review said CHANGES NEEDED: refactor 7677f04 made `find_cup` ~66% slower, because it built the outline and edge share before the size filter. 9b6c9d8 restored size-first filtering (median 38.4 ms on master, 37.8 ms on the branch, synthetic 1080p). The re-review passed. Left: a no-cup frame traces twice on the station (~+90 ms, PlacementFault frames only). A remounted camera can take the моти outline for the rim, filed as risk #119.

**#117 → PR #122** (`434f907..2c7e4b0`). PASS WITH NITS, no behaviour change found. Smell #7 was dropped: the two loops differ.

**Process slip:** the first #116 reviewer timed `find_cup` on real line frames under `data/`. They stayed on this machine, but this breaks the no-client-images rule. Reviewer prompts now forbid `data/`.
