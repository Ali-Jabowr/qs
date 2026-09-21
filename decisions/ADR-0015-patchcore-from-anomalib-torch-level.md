# ADR-0015: PatchCore comes from Anomalib 2.6.2, pinned, used at the torch-model level only

**Status:** Accepted · 2026-09-21 · Ali (design Step 7)

## Context

Station A's appearance check is PatchCore (ADR-0002, cited). The facts, researched 2026-09-21:

- **Anomalib 2.6.2** (2026-09-11) is the current stable release, Apache-2.0, supporting Python
  3.10–3.12. The laptop has Python 3.12.3. The 2.0 rework changed the core API, and minor versions
  have come roughly monthly since.
- Its `Folder` datamodule splits by **image** by default, which breaks unit-level splits (E1).
- Its default post-processor **normalises and clamps** scores against its own threshold; its default
  pre-processor resizes to 256 × 256; its exporter **bakes both into** the ONNX / OpenVINO graph.
- Its greedy coreset and random projection draw on **global, unseeded** random generators.
- Its torch model (`PatchcoreModel`) returns raw scores and maps with none of the above.

Our design already owns localisation, the window, the resize, masking, scoring and calibration
(`15-model.md` §2–§7). The only thing we need from a library is the PatchCore core itself.

## Decision

Depend on `anomalib==2.6.2`, pinned exactly, and use **only** `PatchcoreModel`, with
`pre_processor=False` and `post_processor=False`, fed tensors that our own preprocessing produced.
Seed `torch` and `numpy` globally immediately before coreset subsampling. Import Anomalib in exactly
one module, so a replacement touches one file.

## Alternatives considered

- **Anomalib's full stack** — `Engine`, `Folder`, its processors and exporter. The least glue code, but
  it splits by image, rescales scores, and bakes hidden steps into the exported model: three silent
  ways to break the evaluation protocol. Rejected.
- **Our own PatchCore, ~150–200 lines on torchvision.** Full control, no Lightning, exact
  determinism. But a from-scratch implementation written by the executor this week carries a risk of
  subtle errors — patch aggregation, re-weighting — with nothing to check it against. **Kept in
  reserve**: built only if the pinned API gets in the way, and accepted only once it matches
  Anomalib's scores on the same MVTec images.

## Consequences

- A tested implementation of the one component we do not want to get wrong this week.
- Heavy transitive dependencies — Lightning, timm — installed but barely used. A local wheel cache
  is kept from the first install (`15-model.md` §11, access from Russia).
- We depend on an internal-ish class that could change. The exact pin and the one-module rule
  contain that.
- S1 on MVTec AD verifies the environment and our glue together, before our own data can be blamed.
- The licence question about ImageNet-pretrained weights (§11) is unchanged by this choice; every
  PatchCore variant carries it.
