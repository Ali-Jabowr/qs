# Review — P01 environment

**Date:** 2026-09-22 · **Branch:** `exec/P01`, one commit `7570765` against `master` (`ae94885`) ·
**Executor:** OpenCode, GLM-5.3 via OpenRouter (ADR-0019) · **Reviewer:** Claude Code, with two
independent sub-reviews (Standards, Spec) · **Decision:** Ali

## Verdict

**PASS — merge.** I re-ran every automated check myself; all pass, and the offline install was
re-run with the network actually cut, which is stronger than the check asks. Two boxes pass on
intent rather than on their letter, both because of how the planner worded them (below). Nothing
found blocks P02. One real design finding (R17) goes to the planner before P07.

## Acceptance — the reviewer's own runs

| Check | Result, re-run 2026-09-22 |
|---|---|
| `make env` | exit 0; `pip check`: `No broken requirements found.` |
| `pip list \| grep -ci opencv` | `1` — `opencv-python-headless==5.0.0.93` |
| `make env-check` | `5 passed, 44 deselected, 3 warnings in 7.53s` |
| `make check` | exit 0; `44 passed, 5 deselected`; `17 diagram(s) render.`; ruff `All checks passed!`; mypy `Success: no issues found in 1 source file` |
| Hashes | `grep -c -- '--hash=sha256' requirements.txt` → `1895`. A script over both locks: 95 + 20 pins, **none without a hash** |
| `git check-ignore .venv wheelhouse .hf .torch` | prints `.venv`, `wheelhouse`, `.hf` — **not `.torch`** (see below) |
| `make env-offline` into a fresh venv | `unshare -rn make env-offline VENV=<tmp>` — no network at all: exit 0, 110 packages, `pip check` clean, 65 s |

**Two boxes passed on intent:**

- **`.torch`.** The directory does not exist (R17: the backbone went to `.hf` only), and
  `.gitignore`'s `.torch/` matches only an existing directory. With the directory created,
  `git check-ignore .torch` prints it. The rule is right; the check assumed the directory would
  exist. STATUS.md's "all four printed" is wrong as stated.
- **"No file under `docs/` or `decisions/` was modified."** `decisions/README.md` and ADR-0019 are
  in the commit, with `CLAUDE.md`, `README.md` and `opencode.json`. They are the planner's
  uncommitted files from earlier today, swept in by "commit everything". I compared them line by line
  with what I wrote: the executor changed nothing. STATUS.md declares this (Deviation 7).

## Manual checks

- [x] The environment report is generated and answers R18 plainly. It answers R17 as a fact ("received
  by `.hf`", source `timm/wide_resnet50_2.racm_in1k`), but does not say that this contradicts the
  design. That comparison is in STATUS.md.
- [x] `is_available() = True` — RTX 3050 Laptop GPU, 3.7 GiB; torch 2.14.0 built with CUDA 13.0,
  driver 580.173.02.
- [x] `THIRD_PARTY_NOTICES.md` covers the §11 components and quotes the caveat word for word.
  **Its weights row is inaccurate:** it says "None stated" and links torchvision's docs. The
  installed weights are `timm/wide_resnet50_2.racm_in1k`, whose Hugging Face model card says
  **Apache-2.0** (the card for `tv_in1k` says BSD-3-Clause; both checked 2026-09-22). `resnet18` is
  listed but not installed. The planner fixes this row with the R17 decision.
- [x] STATUS.md states that `v4l2-ctl` is absent (Ali's step).

## Standards

*Sub-review, lightly cleaned. There are no hard violations; the nearest is how STATUS.md reports results.*

1. **`plans/P01-environment/STATUS.md:61-69`: near-hard.** The DoD asks for "actual output pasted
   into STATUS.md", and AGENTS.md asks for "Test results (actual output)". STATUS.md gives a
   summary table instead ("exit 0", "all four printed"). Every row was re-run for this review.
2. **`tools/bounds.py:91`, `tools/render.py:183`: soft.** These edit files outside the plan
   (AGENTS.md: "Do not modify files outside the plan's scope"). The plan's own ruff gate over
   `tools` forced them. Declared as Deviation 5; no behaviour changes.
3. **`src/qs/__init__.py`: soft.** The PLAN keeps `src/qs` code out of scope. The empty file was
   needed because the contract asks for a bare `mypy` that exits 0 on an empty `src/`, and mypy
   exits 2 there. The contract contradicts itself; this is the planner's to fix.
4. **`requirements*.txt:1-7`: soft.** The generated headers were edited by hand (Deviation 2). The
   next `pip-compile` run will silently overwrite them.

Baseline smells (judgement calls, in the tools, not the product):
- **Duplicated code** between `tools/prefetch_weights.py` and `tools/env_report.py`:
  - `sha256_of` is written twice;
  - two cache walks, with `hub_repo_ids` already drifting between them;
  - the `.hf`/`.torch` branching is written twice;
  - the report re-hashes the caches instead of reading `weights-manifest.json`.
- **Dead code** in `env_report.py:131-132, 142`.
- **Mysterious name:** `r17_answer(hf, torchc)`.
- **`Makefile`:**
  - line 1 still says "Everything here runs offline";
  - `env` and `env-offline` repeat the same recipe;
  - `make test` runs the system `python3`, while `check` runs the venv.
- **`tests/test_env.py`:**
  - `:67-73` patches with `raising=False`, so a renamed library internal would be patched as a new attribute and the test would pass for the wrong reason;
  - `:110` accepts `(OSError, ValueError, RuntimeError)`, broad enough to pass on an unrelated failure.

## Spec

*Sub-review, lightly cleaned, with one correction. The contracted files match the spec exactly: `pyproject.toml`, both `.in` files, the four Makefile targets and the lock policy (no GUI OpenCV, no extras, no custom index). So does the caveat paragraph.*

**Missing or partial**
- The `.torch` claim in STATUS.md is wrong as stated (see Acceptance).
- The output is not pasted (see Standards 1).
- R17 in the report:
  - it does not mark which of the nine hashed files is the backbone (the 275 MB blob);
  - it judges `.hf` from a listing after the build, not a before-and-after comparison.
  - *Correction:* the sub-review also said the report never names `racm_in1k`. It does, at line 37.
- The weights row in `THIRD_PARTY_NOTICES.md` (see Manual checks).

**Beyond the plan**
- `src/qs/__init__.py` and the two one-line tool fixes (see Standards 2–3).
- `reports/env/weights-manifest.json` is not in CONTRACTS. It's a harmless addition.
- DejaVu Sans is listed in the notices but is a system font that P01 did not install.

**Implemented, but looks wrong**
- `Makefile:9` `VENV ?= .venv` applies to every target; the spec says to use `.venv/bin/...`
  explicitly. The effect is the same unless `VENV` is overridden.
- The offline test blocks Python sockets only, so native code such as hf-xet could bypass it. The
  real barrier is `HF_HUB_OFFLINE=1`. This meets the letter of the spec.
- The negative twin's broad `pytest.raises` (see Standards).

**Declared deviations:** all seven are justified. `--allow-unsafe` was forced, because torch needs
`setuptools` under `--require-hashes`; note that it also pins `pip` in the dev lock. The
pip-tools `--no-index` header bug that STATUS.md claims is unverified.

**Summary:** Standards has 4 documented-standard findings plus smells; the worst is STATUS.md summarising
instead of pasting the output. Spec has 4 partial items, 3 scope items and 3 doubtful
implementations; the worst is the inaccurate weights row in `THIRD_PARTY_NOTICES.md`.

## Findings for the planner

1. **R17: the backbone is `racm_in1k`, not `tv_in1k`.** `docs/15-model.md:109` predicted the
   original torchvision weights and said "confirm at the environment step". Timm 1.0.29's default
   for `wide_resnet50_2` is the RACM re-training, and Anomalib 2.6.2 inherits it. This needs a
   decision before P07: **keep `racm_in1k`** (Anomalib's own default, Apache-2.0 card) or **pin
   `tv_in1k`** (the design's assumption, BSD-3-Clause card). Either way, P07 must name the full tag
   so a timm upgrade cannot swap the weights silently. Both are ImageNet-trained, so the §11 caveat
   and OQ-10 stand. `docs/15` §5 and §11 and the notices row follow the decision.
2. **P02's plan must expect `src/qs/__init__.py` to exist,** and its contract must not ask mypy to
   pass on an empty `src/`.
3. **Later plans' ACCEPTANCE** should repeat "paste the actual output"; P01 showed it gets summarised.
4. **When a plan next touches `tests/test_env.py`:** `raising=True`, and a specific exception in the
   negative twin.
5. **HF telemetry.** The prefetch wrote `.hf/.agent_harnesses.json`, which STATUS.md attributes to the HF
   client's telemetry. The station is offline at runtime (`HF_HUB_OFFLINE=1`), but plans that run the
   HF client online should also set `HF_HUB_DISABLE_TELEMETRY=1`.

## ADR-0019 — the measurement it asked for

| | P01 |
|---|---|
| Cost, from OpenCode's session records | **$2.66** (159 model steps) |
| Input tokens: from cache / not cached | 12.66M / 0.46M — **96.5% cached** |
| Output + reasoning | 27K + 87K |

**The rule fired.** P01 cost more than the $1.50 threshold. Caching works, so the $35–45 worst case is
off the table. 147 of the 159 steps were spent coaxing `pip-compile` through a link that fell to
~90 KiB/s, which is P01's environment, not the model. Per ADR-0019 as written, P02 onward runs on
GLM-5.3-Flash, and P04 and P07 may stay on GLM-5.3 if the budget covers them. $17.34 of the $20
remains.

## Decision (STEPS.md §4) — Ali, 2026-09-22

- **PASS.** Ali merges `exec/P01` into `master` himself, marks P01's tasks done in `project.yml` with
  the merge commit as evidence, and runs `python3 tools/render.py`.
- **Executor model:** GLM-5.3 for P02, P04 and P07; GLM-5.3-Flash for the rest. Recorded as an amendment
  to ADR-0019.
- **Backbone:** keep `racm_in1k`, named by full tag and checked against the P01 hash (ADR-0020). Finding 1
  is closed. `docs/15` §5 and §11, `THIRD_PARTY_NOTICES.md` and risk R11 are corrected to match.
