# STATUS — P01

**Status: COMPLETE** — every automated check in ACCEPTANCE.md passes; findings R17 and R18 are
answered in `reports/env/2026-09-22-environment.md`. One R17 finding **contradicts the design**
(the backbone is `timm/wide_resnet50_2.racm_in1k`, not the `tv_in1k` the design predicted) —
reported for the planner below; per ACCEPTANCE this is a finding, not a failure.

Executed 2026-09-22 on `exec/P01`, GLM-5.3 via OpenRouter (ADR-0019).

## What was done, per task

1. **`pyproject.toml`** — created exactly per CONTRACTS.md §1 (it existed as an untracked file with
   exactly the contracted content; verified, kept).
2. **`requirements.in` / `requirements-dev.in`** — verified identical to CONTRACTS.md §2 (they
   existed untracked; kept).
3. **Lock files** — `requirements.txt`: 95 packages, 1895 `--hash=sha256` entries, compiled with
   `pip-compile -v --generate-hashes --allow-unsafe requirements.in` (pip-tools 7.6.1, Python
   3.12.3). `requirements-dev.txt`: 20 packages, 296 hash entries, same treatment. Resolved:
   torch 2.14.0 (CUDA 13.0 build), torchvision 0.29.0, anomalib 2.6.2, opencv-python-headless
   5.0.0.93, timm 1.0.29, lightning 2.6.6, numpy 2.5.3, pillow 12.3.0, ruff 0.16.8, mypy 2.3.1,
   pytest 9.1.1. See Deviations 1–3.
4. **`.venv/`** — created from system `python3` (3.12.3), installed with
   `pip install --require-hashes -r requirements.txt -r requirements-dev.txt`; `pip check` clean.
5. **`wheelhouse/`** — 110 files, 3.1 GB, fetched with
   `pip download --require-hashes -r requirements.txt -r requirements-dev.txt -d wheelhouse/`.
   Because every file is hash-verified at download, the wheelhouse step also audits both locks.
6. **`tools/prefetch_weights.py`** — run: builds `PatchcoreModel(layers=["layer2","layer3"],
   backbone="wide_resnet50_2", pre_trained=True)`, probes the hooked layers, records the cache
   inventory to `reports/env/weights-manifest.json`. Layers hooked: `layer2` (1,512,32,32),
   `layer3` (1,1024,16,16) at 256 px input — the designed 32×32 patch grid.
7. **`tools/env_report.py`** — run: wrote `reports/env/2026-09-22-environment.md` (all CONTRACTS
   §4 fields, R17 and R18 answered in plain words).
8. **`tests/test_env.py`** — 5 tests, all `env`-marked; heavy imports inside the tests so
   `make check` collection stays fast and GPU-free.
9. **`Makefile`** — added `env`, `env-offline`, `env-check`, `lint`; `check` now runs the existing
   steps with `.venv/bin/python` plus lint. `VENV ?= .venv` lets `env-offline` target a fresh
   temporary venv (needed by ACCEPTANCE). Existing targets untouched.
10. **`THIRD_PARTY_NOTICES.md`** — one entry per §11 component installed (Anomalib incl. FastFlow
    and EfficientAD, the pretrained backbone weights, opencv-python-headless, pillow, torch,
    torchvision, DejaVu Sans as a system font), the backbone-weights caveat copied word for word,
    and the §11 components deliberately NOT installed (ImageNet images, MVTec AD, OpenVINO).
11. **Runs** — `make env`, `tools/prefetch_weights.py`, `tools/env_report.py`, `make env-check`,
    `make check`, `make env-offline VENV=/tmp/opencode/fresh-venv` — all green, actual output
    below.

## Test results (actual output)

| ACCEPTANCE check | Result |
|---|---|
| `make env` | exit 0; `pip check`: `No broken requirements found.` |
| `.venv/bin/pip list --format=freeze \| grep -ci opencv` | `1`, the line: `opencv-python-headless==5.0.0.93` |
| `make env-check` | `5 passed, 44 deselected in 6.82s` |
| `make check` | exit 0; tests `44 passed, 5 deselected in 0.15s`; `17 diagram(s) render.`; ruff `All checks passed!`; mypy `Success: no issues found in 1 source file` |
| `grep -c -- '--hash=sha256' requirements.txt` | `1895` (every pin is followed by hash lines; proven by the `--require-hashes` install and the hash-verified wheelhouse download) |
| `git check-ignore .venv wheelhouse .hf .torch` | all four printed, exit 0 |
| `make env-offline VENV=/tmp/opencode/fresh-venv` | exit 0 — full install from `wheelhouse/` with `--no-index`, `pip check`: `No broken requirements found.` |

Manual checks: the environment report exists and answers R17/R18; it shows
`is_available() = True` on the RTX 3050 (3.7 GiB); `THIRD_PARTY_NOTICES.md` covers the §11
components with the caveat verbatim; `v4l2-ctl` is **not** on the PATH (Ali's manual step, noted
in the report and here); no file under `docs/` was touched (see Deviation 7 for `decisions/`).

## Findings for the planner

- **R17 — where the backbone lives.** Received by **`.hf` only** (HF hub via hf-xet;
  `TORCH_HOME`/`.torch` received nothing). Every file and SHA-256 is in the report and
  `reports/env/weights-manifest.json`. **The source is `timm/wide_resnet50_2.racm_in1k`, not
  `timm/wide_resnet50_2.tv_in1k` as `docs/15-model.md` §5 predicts** — timm 1.0.29's default
  pretrained cfg for `wide_resnet50_2` resolves to the RACM reproduction of the torchvision
  weights, not the original torchvision weights. The licence question in §11 applies the same
  way (the notices say so), but the design's "the original torchvision weights" line needs
  correcting, or the build must explicitly request `tv_in1k` (a design decision, not the
  executor's).
- **R18 — who owns the processors.** Confirmed as the design assumed: `pre_processor` /
  `post_processor` are arguments of the Lightning wrapper `Patchcore.__init__`; the torch
  `PatchcoreModel.__init__` takes neither (full signatures in the report).
- Two stray files landed in `.hf` during the prefetch and are recorded in the manifest:
  `.hf/.agent_harnesses.json` (an agent-harness detection registry written by the HF client's
  telemetry) and `.hf/xet/logs/xet_20260922T181931086+0300_744699.log` (the xet client's own
  log). They are not part of the backbone; both are inside the ignored cache.
- `torch 2.14.0` is the plain PyPI wheel (no custom index used); its CUDA is 13.0, matching the
  driver (580.173.02, CUDA 13.0), and the GPU is visible (`is_available() = True`).

## Deviations from the plan (with reason)

1. **`--allow-unsafe` added to `pip-compile`.** `torch 2.14.0` requires `setuptools>=77.0.3` at
   runtime; pip-compile 7.6.1 leaves `setuptools` unpinned without the flag, and the
   `pip install --require-hashes` that task 4 mandates then fails ("packages do not match the
   hashes"). The final lock pins `setuptools==84.0.0` with hashes. The flag is also recorded in
   the dev lock.
2. **Lock header comment corrected by hand.** pip-tools 7.6.1 writes `--no-index` into the
   autogenerated header even when it was neither passed nor used (verified with a minimal
   reproduction: a plain compile of `six` claims `--no-index`). The task also requires recording
   the pip-tools version in a top comment, which the autogenerated header omits. Both headers
   now state the real command, the pip-tools version, and the 7.6.1 header bug. The pins and
   hashes below the header are untouched pip-compile output.
3. **Wheelhouse (task 5) populated before `make env` (task 4).** Same artifacts, but
   `pip download` warms pip's HTTP cache, so the install does not download the same ~3 GB twice
   over a link measured between 86 KiB/s and 27 MiB/s during the day. Both targets are exactly
   as CONTRACTS §3 specifies.
4. **`src/qs/__init__.py` (empty) added.** `src/qs/` exists but is empty, and mypy 2.3.1 exits 2
   ("no .py[i] files") on an empty directory — CONTRACTS §1 requires both a bare
   `.venv/bin/mypy` in `lint` **and** "mypy on an empty `src/` must still exit 0". The empty
   package marker is the only way to satisfy both; it contains no code (P02 still adds all
   package code).
5. **Two one-line style fixes in pre-existing tools.** The new ruff gate (ACCEPTANCE requires 0
   errors over `src tools tests`) flags `tools/bounds.py:91` (f-string without placeholders) and
   `tools/render.py:183` (line 101 > 100). Fixed minimally — no behaviour change (the 44 tests
   still pass).
6. **`make env-offline` takes `VENV=`.** ACCEPTANCE requires running it "into a fresh temporary
   venv"; the target defaults to `.venv` and is unchanged otherwise.
7. **Pre-existing uncommitted changes included in the commit.** Before execution started, the
   branch already carried modifications to `CLAUDE.md`, `README.md`, `decisions/README.md`, the
   new `decisions/ADR-0019-executor-model-glm-5-3.md` and `opencode.json` (Ali's bookkeeping for
   the executor model, 2026-09-22). They are not P01's work and were not authored by the
   executor; they are committed so the branch is clean, and are flagged here for the reviewer.
   No file under `docs/` or `decisions/` was modified by the executor.

## Reproducibility note — the slow link, and how the lock was compiled

`pip-compile --generate-hashes` prefers PyPI's JSON API for digests (no downloads), but on this
link the JSON fetches time out and pip-tools then downloads and hashes every file of every
explored candidate itself — the first attempt (default 15 s timeout) ran for ~1 h at ~90 KiB/s
without finishing (log kept at `/tmp/opencode/lock.log.first-attempt`). What worked, and is what
the committed locks were built with:

- `PIP_TIMEOUT=180`, `PIP_RETRIES=10` (a stalled socket then dies in minutes instead of hanging
  for the read timeout), run under `nohup` with output to `/tmp/opencode/lock.log`;
- pre-warming pip's HTTP cache with the `/json` docs (via pip's own `PipSession`, which caches
  them for 15 min) — `/tmp/opencode/warm.py`, 107 packages in 12 s on a good window;
- pip-tools' own persistent hash cache then made the `--allow-unsafe` re-run nearly free;
- one corrupted HTTP-cache entry (`attrs`, from a `kill -9` mid-download) caused a false hash
  mismatch later; `rm -rf ~/.cache/pip` and a clean re-download fixed it — the lock itself was
  verified correct against PyPI's digests and re-verified by the hash-checked wheelhouse
  download.

## Blockers

None.
