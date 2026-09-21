# PLAN-P01: Environment — pinned, offline-capable, and checked

## Objective

A reproducible Python 3.12 environment in `.venv/` that trains on the laptop's RTX 3050, uses
headless OpenCV and Python's own Tk, loads its pretrained backbone with the network off, and passes
`make check` with pytest, ruff and mypy. It also answers two questions the design could not settle
without installing Anomalib: where Anomalib 2.6.2 downloads the backbone from (review item R17),
and which class accepts the `pre_processor` and `post_processor` arguments (R18).

## Inputs

- `CONTEXT.md` (this folder) — why each pin and each check exists.
- `CONTRACTS.md` (this folder) — the exact files, targets and report fields to produce.
- `docs/15-model.md` §5, §11; `decisions/ADR-0015`, `ADR-0016`, `ADR-0018`.
- The existing `Makefile`, `tools/`, `tests/` (44 passing tests), `.gitignore`.

## Tasks

1. **`pyproject.toml`** — create it with the `[project]`, `[tool.ruff]`, `[tool.mypy]` and
   `[tool.pytest.ini_options]` sections specified in CONTRACTS.md §1. No package code yet; P02
   adds `src/qs`.
2. **`requirements.in` and `requirements-dev.in`** — the direct dependencies listed in CONTRACTS.md
   §2, with the constraints given there and no others.
3. **Lock files** — `requirements.txt` and `requirements-dev.txt`, compiled with
   `pip-compile --generate-hashes` so every package is pinned to an exact version and hash.
   Record the pip-tools version used in a comment at the top of each.
4. **`.venv/`** — created from the system `python3` (3.12.3), installed with
   `pip install --require-hashes -r requirements.txt -r requirements-dev.txt`.
5. **`wheelhouse/`** — every locked wheel downloaded with `pip download` into `wheelhouse/`, so a
   reinstall needs no network (`make env-offline` installs from it).
6. **`tools/prefetch_weights.py`** — with `HF_HOME=.hf` and `TORCH_HOME=.torch`, build Anomalib's
   PatchCore torch model with `wide_resnet50_2`, layers `layer2` and `layer3`, pretrained — which
   downloads the backbone — then record which of the two caches received files and the SHA-256 of
   every file received.
7. **`tools/env_report.py`** — writes `reports/env/<date>-environment.md` with every field listed in
   CONTRACTS.md §4, including the R17 and R18 answers.
8. **`tests/test_env.py`** — the environment tests in CONTRACTS.md §5, marked `env`, including the
   offline backbone load with socket connections blocked.
9. **`Makefile`** — add `env`, `env-offline`, `env-check` and `lint` targets, and extend `check`, as
   CONTRACTS.md §3 specifies. `make check` must not need a GPU or the network; `make env-check` may.
10. **`THIRD_PARTY_NOTICES.md`** — one entry per component in `docs/15-model.md` §11 that is
    installed, with its licence and source URL, and the backbone-weights caveat word for word.
11. **Run** `make env`, `tools/prefetch_weights.py`, `tools/env_report.py`, `make env-check`,
    `make check`, and write `STATUS.md` in this folder.

## Constraints

- **Do not install `opencv-python`** (the GUI build), and do not install both OpenCV builds. Anomalib
  requires `opencv-python-headless`; the screen is Tk (ADR-0018).
- **Do not pin a torch build by hand** or add a custom index URL. Let `pip-compile` resolve
  `torch` and `torchvision` from PyPI within the constraints in CONTRACTS.md §2, then pin exactly what
  it resolved. If the resolved torch cannot see the GPU, stop and report — do not experiment with
  indexes.
- **Do not install Anomalib extras** (`[cu126]`, `[openvino]`, …). They add packages the design does
  not use (`torchcodec`, `nncf`, `onnx`).
- **Do not use `sudo`.** `v4l-utils` is Ali's manual step; note in STATUS.md whether it is present.
- **Do not download MVTec AD.** That is P07.
- **Do not edit any file under `docs/` or `decisions/`.** If the R17 or R18 answer contradicts the
  design, report it in STATUS.md; the planner corrects the design.
- **Do not commit** `.venv/`, `wheelhouse/`, `.hf/`, `.torch/` — they are already git-ignored; verify it.

## Definition of Done

- Every automated check in `ACCEPTANCE.md` passes, with its actual output pasted into `STATUS.md`.
- `reports/env/<date>-environment.md` exists and states the R17 and R18 answers.
- `STATUS.md` is complete; no uncommitted changes.

## Out of Scope

- Any code under `src/qs` (P02).
- Camera access, V4L2 controls, the rig (P05).
- MVTec AD, training, latency measurement (P07).
- OpenVINO (product scope, FR-57).
