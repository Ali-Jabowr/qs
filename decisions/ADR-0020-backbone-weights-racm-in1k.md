# ADR-0020: PatchCore's backbone weights are timm's `wide_resnet50_2.racm_in1k`, named by full tag

**Status:** Accepted · 2026-09-22 · Ali, on review of P01 (`reviews/REVIEW-P01.md`). Settles the
"confirm at the environment step" in `docs/15-model.md` §5.

## Context

`docs/15-model.md` §5 predicted that Anomalib would load `timm/wide_resnet50_2.tv_in1k`, the original
torchvision weights, and asked P01 to confirm it. P01 found otherwise (R17): timm 1.0.29 resolves
`wide_resnet50_2` to its default pretrained tag, **`racm_in1k`**, timm's own re-training of the same
network on ImageNet-1k. Anomalib 2.6.2 inherits this default. The weights sit in `.hf/` and are
hashed in `reports/env/weights-manifest.json`. The 275,835,296-byte blob has SHA-256
`03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000`.

The two Hugging Face model cards, checked 2026-09-22: `racm_in1k` declares **Apache-2.0**;
`tv_in1k` declares **BSD-3-Clause**. Both networks were trained on ImageNet, whose images are
non-commercial.

## Decision

- PatchCore uses **`wide_resnet50_2.racm_in1k`**.
- P07 passes the **full tag**, not the bare architecture name, and verifies the loaded file's
  SHA-256 against the P01 manifest. A timm upgrade that changes the default then fails loudly
  instead of swapping the weights.

## Alternatives considered

- **`tv_in1k`**, as the design assumed. It is what the PatchCore paper used. It needs an explicit tag,
  a second download on a slow link, and a departure from Anomalib's own default. Rejected: nothing
  in the MVP depends on matching the paper's backbone exactly.
- **The bare name `wide_resnet50_2`.** It works today, and it is exactly how this surprise
  happened. Rejected.

## Consequences

- `docs/15-model.md` §5 and §11 and `THIRD_PARTY_NOTICES.md` name `racm_in1k` and its declared
  licence.
- **The licence question does not close.** The publisher now declares a licence on the weights,
  which corrects §11's "neither torchvision nor timm attaches a licence". Whether ImageNet's terms
  reach the weights is still contested, and it stays with the lawyer line (OQ-10, risk R11). The
  consequences in §11 are unchanged.
- The MVTec bring-up in P07 compares against Anomalib's results, which were produced with whatever
  timm default was current at the time. A small gap there is not, by itself, a finding.
