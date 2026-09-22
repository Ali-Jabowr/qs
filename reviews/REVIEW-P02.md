# Review — P02 contracts and the station config

**Date:** 2026-09-22 · **Branch:** `exec/P02`, commit `799e9d0` against `master` (`cabd182`) ·
**Executor:** OpenCode, GLM-5.3 via OpenRouter (ADR-0019), run by Ali · **Reviewer:** Claude Code,
with two independent sub-reviews (Standards, Spec) · **Decision:** Ali

## Verdict — round 1

**FAIL, small: fix on this branch** (`STEPS.md` §4). Everything the plan asked for is there and
correct, and every check passes. But the spec I wrote left a way to **fail open**: the config
accepts a NaN or infinite threshold, which would pass every unit's appearance check. The spec is
amended (`CONTRACTS.md`, marked "amended at review"). The executor applies the rework below on this
branch, then round 2 re-checks it.

## Acceptance — the reviewer's own runs

| Check | Result |
|---|---|
| `make check` | exit 0; `91 passed, 5 deselected` (44 existing + 47 new); `17 diagram(s) render.`; ruff `All checks passed!`; mypy strict `Success: no issues found in 4 source files` |
| `.venv/bin/pytest tests/test_contracts.py tests/test_config.py -q` | `47 passed` |
| `PYTHONPATH=src … load_config(…example.yml).config_hash` | `cd1b5c35ec61a26fd5c58b64ce941591797ca4d5d9f1aaec362f6dc044b425e7` — 64 hex |
| `grep -nE '^(import\|from) ' src/qs/contracts.py` | `__future__`, `dataclasses`, `datetime`, `enum`, `typing`, `numpy`. `StationConfig` comes from `qs.config` under `TYPE_CHECKING` only |
| `docs/`, `decisions/` | untouched; `pyproject.toml` changed by exactly the one planned line |

Manual checks:
- Field names and order match `docs/11` §2. The Spec reviewer compared all seven records by script.
- `derive_label` never returns NO_VERDICT and never PASS with a check missing. The Spec reviewer tried every mix of three check names.
- Every validation message names its key.

**What the reviewer probed** (`load_config` on a modified example):

| Input | Result |
|---|---|
| `threshold.value` = NaN | **loads** |
| `threshold.value` = infinity | **loads** |
| `threshold.value` = −1 | **loads** (fails safe: every unit a DEFECT) |
| `axis_ratio_min`, `solidity_min`, `band`, `expect` = NaN | refused, key named |
| `display.labels.pass` = `"  "` | **loads** |

## Standards

*Sub-review, lightly cleaned.*

**Hard: the config can make every unit pass** (CLAUDE.md §6, "Fail to НЕТ РЕШЕНИЯ, never to
ГОДЕН"). `config.py:173-178` and `_validate` accept `threshold.value: .inf` and `.nan`, which were confirmed by
running the loader. The spec never asked for this check, so it's a gap in the plan and not an executor breach. → Rework 1.

**Documented-standard breaches, soft (both declared):**
- AGENTS.md says "If a task is ambiguous, stop and report". Deviations 2 (a degenerate ROI is refused
  under rule 4, `config.py:434`) and 3 (ints widened to floats) were decided rather than stopped on.
  Both decisions are sound, so they are accepted.
- STATUS.md:5 says "One forced deviation" and then lists four.

**Judgement calls:**
- `config.py:469-476`: `if not text` accepts whitespace, which breaks I6 (NO_VERDICT must look different
  from PASS). → Rework 2.
- `contracts.py:73-77`: `Verdict` accepts `label=PASS` with a failed check. Not in the spec; a note
  for the planner.
- `contracts.py:190`: `not check.passed` is truthiness. → Rework 3.

**Tests:**
- `test_contracts.py:111-113` tests the standard library's Enum.
- `test_contracts.py:33-40`: `REASONED_ERRORS` is a hand-written list. → Rework 4.
- `test_contracts.py:190-192` only checks that `detail` is stored.
- Two `derive_label` cases are missing. → Rework 3.
- `test_config.py:164-166` `_refused` matches the key as a substring, so the rule 3 and rule 4 tests
  cannot tell which rule fired.
- `test_config.py:375-387` duplicates `:197-215`, and `tmp_path` at `:171` is unused.

**Smells (judgement):**
- Duplicated Code: `_roi` and `_pair` (`config.py:186-209`).
- Duplicated knowledge: each `_parse_*` names its keys twice.
- Primitive Obsession: `Verdict.reason: str` and `PlacementFault.detail: str` should be closed sets.
- Repo convention: `config.py` and `errors.py` lack the module docstring and
  `from __future__ import annotations` every existing module has.

## Spec

*Sub-review, lightly cleaned. Verdict on spec: PASS; every item P02 asked for is present.*

**Verified complete:**
- Every case in CONTRACTS §4 has a real test.
- All seven records match `docs/11` §2 in names, types and order, checked by script.
- With comments stripped, the example YAML is identical to `docs/12`:305-353.
- All nine §3.3 rules raise `ConfigError` naming the key.

**Partial:**
- Rule 4 is tested only for `lighting.reference_patch.roi`; `inspector.localize.roi` is enforced but
  untested. → Rework 5.
- Bool-where-int is enforced but untested.
- CONTRACTS:62 asks for "class attribute `reason: str`"; `errors.py` writes `reason = "capture"`,
  unannotated. → Rework 4.

**Scope creep:** none that matters. The extra tests, the wrapping of `OSError`/`YAMLError` in
`ConfigError`, and "exit code 2" (docs/16:60) are all defensible.

**Implemented, but check these:**
- `derive_label` accepts a truthy non-bool: `passed=1` or `"no"` gives PASS. → Rework 3.
- Loader probes, all refused with the key named:
  - missing key: `camera.focus`
  - unknown key: `camera.zoom`
  - wrong type: `'1920'`, `1920.0`, `trigger: keyboard`
  - bool-where-int: `width: true`, `true` in a ROI, `expect: true`, `calib_units: yes`
- `canonical_hash` matches CONTRACTS:83-84 exactly. It is stable, independent of key order at any
  depth, and turns dates inside lists into ISO strings.

**Gaps in the spec itself:**
- `docs/12`:361-362 says a threshold is never used "with a different model, backend **or
  localisation**", but rule 6 leaves out localisation. → Rework 1.
- A NaN or infinite `threshold.value` loads. → Rework 1.
- `exposure_us: 0` or a negative value gets past rule 2. → Rework 1.
- A key written twice in the YAML loads; the last value wins. → Rework 1.

**Declared deviations:**
1. `pass_` for the YAML key `pass`: justified, since it's a Python keyword and messages still say
   `display.labels.pass`.
2. A degenerate ROI is refused: justified.
3. Ints widened to floats: fine. The stated reason is wrong (`1.0` is representable). Note that
   `expect: 1` and `expect: 1.0` load as the same config but hash differently.
4. Four extra tests: fine.

**Summary.** Standards: one hard finding (the NaN/infinite threshold fails open) and two soft
declared breaches, plus test and smell notes. Spec: the plan is complete, two tests are partial, and
four gaps sit in the spec itself; the worst is the missing localisation check in rule 6.

## Rework for the executor (round 2)

`CONTRACTS.md` is amended; the executor implements the amendments on `exec/P02`.

1. **Validation:**
   - rule 2: `exposure_us <= 0` is refused;
   - rule 6: also compare `set_on.localize_threshold` with `inspector.localize.threshold`, and
     `set_on.mask_dilate_px` with `inspector.mask_dilate_px`;
   - rule 10 (new): `threshold.value` must be finite and above 0;
   - rule 11 (new): a key repeated within one YAML mapping is refused, naming the key.
2. Rule 9: a whitespace-only label is refused.
3. `derive_label`: `passed is not True` → DEFECT. Tests for "failed + missing → DEFECT" and for
   non-bool `passed` → DEFECT.
4. `errors.py`: annotate `reason: str`. The reason test walks `QSError`'s subclasses instead of a
   hand-written list.
5. Tests: rule 4 for `inspector.localize.roi`; rule 6 for each of its four comparisons; rule 10 for
   NaN, infinity, 0 and −1; rule 11. Add a module docstring and `from __future__ import annotations`
   to `config.py` and `errors.py`, as every existing module has.
6. STATUS.md: add a "Round 2" section with the actual output of every ACCEPTANCE check. Commit.

Not required (notes for later plans): the `_roi`/`_pair` duplication, closed types for
`Verdict.reason` and `PlacementFault.detail`, `Verdict` accepting PASS with a failed check, and the
1-vs-1.0 hash difference.

## Cost (ADR-0019)

P02 round 1: **$0.52**, 28 model steps, 88.7% of input from cache. That includes one turn that spent
its whole 32K output allowance reasoning and wrote nothing (`finish=length`); Ali steered it on.
Spent so far this week: **$3.18 of $20**.

## Verdict — round 2: PASS

Re-run by the reviewer on the round-2 tree: `make check` exit 0, **104 passed** (60 in the two P02
files), ruff and strict mypy clean; the example's hash unchanged (`cd1b5c35…25e7`). Probed directly:
`threshold.value` NaN, infinity, 0 and −1 all refused; `set_on.localize_threshold` and
`set_on.mask_dilate_px` mismatches refused; `exposure_us: 0` refused; a whitespace label refused; a
repeated `gain:` key refused as `camera.gain`; `derive_label` gives DEFECT for `passed=1` and
`passed="no"` and for failed-plus-missing. Only the six allowed files changed.

The executor's one reading of the amended contract is accepted: rule 11 composes the YAML once with
`SafeLoader` to find repeated keys, then loads with `safe_load` as before; it compares key text, so
`1` and `'1'` count as a repeat, which fails closed.

Round 2 cost **$0.57**; P02 in total **$1.08**. Spent this week: **$3.74 of $20**.
