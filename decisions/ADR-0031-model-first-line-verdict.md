# ADR-0031: On the line, trained models find the cup and judge it; warmth only guards

**Status:** Accepted · 2026-11-12 · Ali. Amends [ADR-0029](ADR-0029-cup-rim-under-own-backlight-warmth-verdict.md)
§3, §4, §5 and §9; keeps §1, §2, §6, §7 and §8. Plan and reasons: the line v1 design review of 2026-11-12.

## Context

ADR-0029 made the empty-cup verdict a measurement: median warmth inside the rim. QS is meant to
be a vision-model product that grows from the empty cup to dents, shape and surface defects, and a
hand-tuned colour rule does not grow that way. PatchCore already runs on the line beside the
empty-cup check (`LineInspector`, `qs train` line mode) but has never been trained on line footage.

The rim finder traces a rim on 94% of stopped frames and 40% of moving ones on the first visit's
footage. A second visit will record the line from a fixed mount, which gives the footage a trained
cup detector needs.

## Decision

1. **PatchCore is the empty-cup verdict.** It is trained on full cups only, so an empty cup is an
   anomaly. In v1 it judges empty cups only; its score is kept for later surface-defect work.
2. **It scores every frame of the burst (at least 3), and the median score decides.** One blurred
   or glare-hit frame cannot flip the verdict (same rule as ADR-0029 §4).
3. **The threshold is set on held-out full cups at a false-alarm cap,** and the empty-cup catch rate
   is measured at that threshold. The client's 95% is read as "95% of empty cups caught, with no
   more than N% of full cups flagged"; N is agreed with the client after the second visit.
4. **Warmth stays as a guard only.** When it disagrees with PatchCore the verdict is НЕТ РЕШЕНИЯ;
   it can withhold ГОДЕН but never grant it.
5. **A trained detector finds the cup:** torchvision's FasterRCNN-MobileNetV3 (BSD code, already
   pinned, no new dependency), fine-tuned on boxes from the second visit. The rim finder pre-draws
   those boxes, which are corrected in a local labelling tool, and stays as the fallback locator.
   The detector follows the triggered cup across its burst. This replaces ADR-0029 §3's fixed-radius
   ring matcher and comes before the photo sensor in ADR-0029 §9.
6. **Data:** the home rig proves the chain and supplies the 300+ empty-cup passes of ADR-0029 §8;
   the second visit records at least 500 full-cup passes and about 50 staged empty cups, and both
   models are retrained on them before anything is shown as advisory.
7. **v1 runs on the laptop's GPU.** The N100 comes after its latency is measured.

## Alternatives considered

- **Warmth as the verdict (ADR-0029).** Separates every labelled cup, but does not extend to other
  defects. Kept as the guard.
- **A supervised full/empty classifier.** Sharper on empty cups, but needs hundreds of labelled
  empties and only knows what it was taught. Possible later, compared on the same labels.
- **YOLOX-Tiny through Intel's Geti library.** Apache code and weights and a maintained OpenVINO
  path, but a large new framework. Revisited only if the N100 is too slow.
- **Ultralytics YOLO.** AGPL-3.0; ruled out for a closed product.

## Consequences

- **Two models in the hot path.** Latency on the laptop must be measured before the visit.
- **A model trained at home will likely false-alarm on the real channel;** only visit-trained models
  count for the 95%.
- **Pretrained weights carry dataset terms** (COCO for the detector, ImageNet for PatchCore's
  backbone); both go to the lawyer line under OQ-10.
- **Config:** `blob_share_min` goes (as in ADR-0029 §5); a PatchCore threshold and a detector model
  are set per mount.
