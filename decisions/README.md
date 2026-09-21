# Decision log

Architecture Decision Records, Nygard format: Title, Status, Context, Decision, Consequences.
Numbers are never reused. A superseded ADR stays, with its status changed and a pointer forward.

| ADR | Decision | Status | Date |
|---|---|---|---|
| 0001 | *reserved — backfill of a decision taken before 2026-09-17 (E0.7)* | — | — |
| 0002 | Anomaly detection, not supervised detection — cited in `14-eval-protocol.md` §4 | to write (E0.7) | 2026-09-01 |
| 0003 | *reserved — backfill (E0.7)* | — | — |
| 0004 | *reserved — backfill (E0.7)* | — | — |
| 0005 | Gate A restated: 30 dents, calibration split, FAR as a point estimate — cited in `14-eval-protocol.md` §5 and `project.yml` | to write (E0.7) | 2026-09-17 |
| 0006 | Dates re-planned after 15 idle days — cited in `project.yml` | to write (E0.7) | 2026-09-17 |
| [0007](ADR-0007-mvp-is-demo-grade.md) | The MVP is demo-grade; Gate A stays a separate claim | Accepted | 2026-09-21 |
| [0008](ADR-0008-geometry-check-in-code.md) | Size and shape are measured in code, not left to the anomaly model | Accepted | 2026-09-21 |
| [0009](ADR-0009-inspector-localizes-then-judges.md) | Station A's Inspector localizes the unit, then judges the unit — never the frame | Accepted | 2026-09-21 |
| [0010](ADR-0010-bench-camera-usb-webcam.md) | Bench camera: USB webcam over V4L2; phone as fallback | Proposed — lock test | 2026-09-21 |
| [0011](ADR-0011-operator-screen-opencv-pillow.md) | Operator screen: one OpenCV window, text by Pillow | Accepted | 2026-09-21 |
| [0012](ADR-0012-storage-failure-verdict-wins.md) | When storage fails, the verdict wins and the record outlives the image | Accepted | 2026-09-21 |

New decisions take the next free number.
