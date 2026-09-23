# Decision log

Architecture Decision Records, Nygard format: Title, Status, Context, Decision, Consequences.
Numbers are never reused. A superseded ADR stays, with its status changed and a pointer forward.

| ADR | Decision | Status | Date |
|---|---|---|---|
| [0001](ADR-0001-smartbot-closed-qs-replaces-it.md) | SmartBot Industrial is closed; QS replaces it | Accepted (backfilled) | 2026-09-01 |
| [0002](ADR-0002-anomaly-detection-not-supervised.md) | Anomaly detection, not supervised detection | Accepted (backfilled) | 2026-09-01 |
| [0003](ADR-0003-marketplace-dimensioning-not-v1.md) | Marketplace parcel dimensioning is not V1 — not the geometry check | Accepted (backfilled) | 2026-09-01 |
| [0004](ADR-0004-marking-station-ships-first.md) | Marking station ships first; appearance is the R&D | Accepted (backfilled) | 2026-09-02 |
| [0005](ADR-0005-gate-a-restated.md) | Gate A restated so it can tell a working system from luck | Accepted (backfilled) | 2026-09-17 |
| [0006](ADR-0006-dates-replanned.md) | The schedule is re-planned after fifteen idle days | Accepted (backfilled) | 2026-09-17 |
| [0007](ADR-0007-mvp-is-demo-grade.md) | The MVP is demo-grade; Gate A stays a separate claim | Accepted | 2026-09-21 |
| [0008](ADR-0008-geometry-check-in-code.md) | Size and shape are measured in code, not left to the anomaly model | Accepted | 2026-09-21 |
| [0009](ADR-0009-inspector-localizes-then-judges.md) | Station A's Inspector localizes the unit, then judges the unit — never the frame | Accepted | 2026-09-21 |
| [0010](ADR-0010-bench-camera-usb-webcam.md) | Bench camera: USB webcam over V4L2; phone as fallback | Proposed — lock test; amended by 0022 | 2026-09-21 |
| [0011](ADR-0011-operator-screen-opencv-pillow.md) | Operator screen: one OpenCV window, text by Pillow | Superseded in part by 0018 | 2026-09-21 |
| [0012](ADR-0012-storage-failure-verdict-wins.md) | When storage fails, the verdict wins and the record outlives the image | Accepted | 2026-09-21 |
| [0013](ADR-0013-pilot-geometries-one-frame.md) | One frame serves both checks; the pilot compares three front-lit geometries | Accepted — winner after the pilot | 2026-09-21 |
| [0014](ADR-0014-paired-shooting.md) | Test units shot good, then damaged, minutes apart, in random order | Accepted | 2026-09-21 |
| [0015](ADR-0015-patchcore-from-anomalib-torch-level.md) | PatchCore from Anomalib 2.6.2, pinned, torch-model level only | Accepted | 2026-09-21 |
| [0016](ADR-0016-station-runs-only-what-it-was-given.md) | The station runs only what it was given: offline, hashed, never validation-only models | Accepted | 2026-09-21 |
| [0017](ADR-0017-gate-b-proxy.md) | Gate B judged on a conservative N100 estimate until an N100 exists | Accepted | 2026-09-21 |
| [0018](ADR-0018-operator-screen-tk.md) | Operator screen is a Tk window; OpenCV headless everywhere | Accepted | 2026-09-21 |
| [0019](ADR-0019-executor-model-glm-5-3.md) | The executor runs GLM-5.3 through OpenRouter, only on hosts that do not train on prompts | Superseded by 0021 | 2026-09-22 |
| [0020](ADR-0020-backbone-weights-racm-in1k.md) | PatchCore's backbone weights are timm's `wide_resnet50_2.racm_in1k`, named by full tag | Accepted | 2026-09-22 |
| [0021](ADR-0021-executor-is-claude-code.md) | Claude Code executes the plans; OpenCode is withdrawn | Superseded by 0023 | 2026-09-22 |
| [0022](ADR-0022-bench-capture-by-phone.md) | Bench capture by phone until the webcam arrives; the lock test becomes an EXIF read-back | Accepted | 2026-09-23 |
| [0023](ADR-0023-opencode-implements-tickets.md) | OpenCode implements the tickets again; Claude Code reviews and commits | Accepted | 2026-09-23 |

New decisions take the next free number.
