# ADR-0016: The station runs only what it was given — offline, hashed, and never on validation-only models

**Status:** Accepted · 2026-09-21 · Ali (design Step 8)

## Context

Three things can make the station run something nobody decided it should:

- **A runtime download.** Libraries fetch pretrained weights on first use. On the bench that is
  silent; at the plant there is no route, so it fails the first time the box boots. Either way it is
  a network call NFR-07 forbids, and weights that could change underneath a calibrated threshold.
- **The wrong bundle.** A bundle rebuilt, copied or swapped without its config changing.
- **A model trained on MVTec AD.** The pipeline is brought up on MVTec, whose licence is
  non-commercial (R8). A bundle trained on it is a working model that must never ship. Today, only a
  person's memory prevents that.

## Decision

1. **Setup is online; running is offline.** Every wheel and every pretrained weight is fetched once
   at setup, hashed, and cached in the repo's ignored `wheelhouse/`, `.torch/` and `.hf/` — Anomalib loads its backbone
   through timm, which fetches from the Hugging Face Hub (confirmed at the environment step). The
   station runs with `TORCH_HOME` and `HF_HOME` pointing at the caches and `HF_HUB_OFFLINE=1`, and a
   test runs it with networking disabled.
2. **Bundles carry provenance.** `bundle.yml` records the hash of every training image and the source
   of the data: `own` or `external:<name>`. The station checks the bundle's hash against the
   config's `model_id` at start-up.
3. **Validation-only is enforced in code.** A bundle with any `external:` source is tagged
   `validation-only`, and the station refuses it unless started in validation mode.

## Alternatives considered

- **A written rule** — "never ship an MVTec model". Rejected: the failure is exactly the kind a tired
  person makes once.
- **Keeping MVTec on a different machine.** It separates the data, but not a copied bundle. Rejected
  as insufficient on its own; the directory separation is kept as well.

## Consequences

- The first install needs the internet; every run after it does not.
- A local cache of pinned wheels and weights is also the insurance against download blocks from
  Russia (`15-model.md` §11).
- A small amount of code — one tag check and one hash check — replaces a rule that relied on
  memory.
