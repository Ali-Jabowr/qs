# CONTEXT — P01

## The machine

One laptop does training and the demo (confirmed by Ali, 2026-09-21):

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, **4 GB** — driver 580.173.02, CUDA 13.0 |
| CPU | Intel Core i5-11400H, 6 cores / 12 threads |
| RAM | 15 GB, often ~10 GB in use by other programs — keep batch sizes small |
| Disk | ~1.2 TB free |
| Python | system `python3` 3.12.3; `python3-tk` installed, Tk 8.6; a desktop session (`DISPLAY=:1`) |
| Not yet installed | OpenCV, PyTorch, Anomalib, ruff, mypy, `v4l-utils`; no webcam plugged in |

## Why each decision

- **Anomalib 2.6.2, pinned exactly** (ADR-0015). Released 2026-09-11; minor versions arrive about
  monthly and 2.0 changed the core API. Its PyPI metadata requires `opencv-python-headless>=4.5.3.56`,
  `lightning>=2.6,!=2.6.2,!=2.6.3`, `timm`, `torchmetrics>=1.8.2`; torch comes only through extras,
  which we do not use — we name `torch>=2.6` and `torchvision>=0.21` ourselves.
- **Headless OpenCV only** (ADR-0018). The headless build has no window functions; the operator
  screen is Tk. Two OpenCV builds in one environment overwrite each other's files.
- **Hashes and a wheelhouse** (`17-security.md` T6; ADR-0016). Setup is online; everything after it
  must work offline. Access to download sites from Russia was not verified blocked, and was not
  verified open either — the wheelhouse is the insurance.
- **Weights cached, then loaded with the network off** (ADR-0016). The station never downloads at
  runtime. Research suggests Anomalib loads its backbone through timm, which fetches from the Hugging
  Face Hub into `HF_HOME` (`timm/wide_resnet50_2.tv_in1k`), not into `TORCH_HOME`. Unconfirmed —
  that is R17.
- **R18.** The design uses Anomalib's torch model `PatchcoreModel` directly. The independent review
  said `pre_processor` / `post_processor` are arguments of the Lightning wrapper `Patchcore`, not of
  `PatchcoreModel`. Unconfirmed — record the real signatures.
- **ruff and mypy** are required by NFR-10 and are not installed.

## Repo state

- `tests/` holds 44 passing tests for `tools/` (project state, bounds). `make check` today runs the
  project-state validator, pytest and the diagram renderer (which uses `npx` and may download the
  Mermaid CLI on first run — leave that target as it is).
- `.gitignore` already ignores `.venv/`, `wheelhouse/`, `.hf/`, `.torch/`, `data/*`, `evidence/`,
  `models/`.
- `src/` is empty. P02 creates `src/qs`.

## Where to look

`docs/15-model.md` §5 (PatchCore), §11 (licences); `docs/17-security.md` §5–§6;
`decisions/ADR-0015`, `ADR-0016`, `ADR-0018`; `docs/06-requirements.md` NFR-10 to NFR-12.
