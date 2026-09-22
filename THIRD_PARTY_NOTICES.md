# Third-party notices — what is installed, and under what terms

Generated for the P01 environment: one entry per component named in
`docs/15-model.md` §11 that is installed on the bench, with its licence, its
caveat and its source. Exact versions are those locked in `requirements.txt`
and `requirements-dev.txt` — the locks are the complete inventory; this file
covers the §11 items.

| Component | Installed as | Licence | Caveat | Source |
|---|---|---|---|---|
| Anomalib, incl. PatchCore and PaDiM | anomalib 2.6.2 | Apache-2.0 (Intel) | PatchCore and PaDiM are Intel's own code, not third-party | [LICENSE](https://github.com/open-edge-platform/anomalib/blob/main/LICENSE) |
| Anomalib FastFlow | inside the anomalib 2.6.2 wheel (installed, unused by the station) | Apache-2.0 (third party) | Keep the attribution | [third-party-programs.txt](https://github.com/open-edge-platform/anomalib/blob/main/third-party-programs.txt) |
| Anomalib EfficientAD | inside the anomalib 2.6.2 wheel (installed, unused by the station) | Apache-2.0 code | Commercial use **unclear** — downloads extra pretrained teacher weights with no licence found, and trains on Imagenette. Excluded as a fallback model by the design | [docs](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/efficient_ad.html) |
| Pretrained `wide_resnet50_2.racm_in1k` weights | the backbone cache `.hf/` — `timm/wide_resnet50_2.racm_in1k` (see `reports/env/2026-09-22-environment.md`, R17; ADR-0020) | Apache-2.0, declared on the model card | Commercial use **unresolved** — see the backbone-weights caveat below, word for word from `docs/15-model.md` §11 | [model card](https://huggingface.co/timm/wide_resnet50_2.racm_in1k) |
| OpenCV ≥ 4.5 | opencv-python-headless 5.0.0.93 — the one OpenCV distribution installed | Apache-2.0 | Its wheels bundle FFmpeg (LGPL-2.1) — an obligation only if we redistribute the wheel | [opencv-python](https://github.com/opencv/opencv-python) |
| Pillow | pillow 12.3.0 | MIT-CMU | Keep the notice | [LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| PyTorch | torch 2.14.0, torchvision 0.29.0 | BSD-3-Clause | — | [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| DejaVu Sans | system fonts (`fc-list` finds 23 DejaVu faces; not installed by the venv) | Bitstream Vera; changes public domain | Not sold alone; keep the notice | [licence](https://dejavu-fonts.github.io/License.html) |

Named in §11 but **not installed here**: ImageNet images (only the pretrained
weights are fetched, never the dataset), MVTec AD (P07, and validation-only
under R8), OpenVINO (product scope, FR-57).

**What is settled and what is not about the backbone weights.** Settled: the ImageNet images are non-commercial. torchvision attaches no licence to its weights; timm's model cards do declare one (Apache-2.0 for `racm_in1k`, corrected 2026-09-22, ADR-0020). Contested: whether the dataset's terms reach the weights, which a publisher's licence does not decide. One position holds that weights contain no images and may not be copyrightable at all ([arXiv 2412.07066](https://arxiv.org/pdf/2412.07066)); the other says to assume the dataset's terms apply and to take advice. These backbones ship in many commercial products — which is practice, not a ruling.

The consequences drawn in §11 stand unchanged: the MVP ships nothing; before
anything ships to a client, the backbone question goes to a lawyer;
EfficientAD stays excluded; MVTec AD stays validation-only.
