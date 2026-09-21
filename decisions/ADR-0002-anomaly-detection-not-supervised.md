# ADR-0002: Anomaly detection, not supervised detection

**Status:** Accepted · 2026-09-01 · Ali · backfilled 2026-09-21 from CLAUDE.md §7

## Context

A food line produces thousands of good units and very few defective ones, and the defect list is
open-ended — the next defect may be one nobody has seen. A supervised detector (YOLO, RF-DETR) needs
labelled examples of every class it should find.

## Decision

Station A's appearance check is unsupervised anomaly detection — the PatchCore family, trained on good
units only, flagging what differs from them.

## Alternatives considered

- **Supervised detection.** Needs hundreds of labelled defects per class that do not exist, and
  misses classes it was never shown. The best-known option, YOLO, also carries an AGPL licence trap.
  Rejected.

## Consequences

- Training needs good units only — cheap to collect, and the evidence store collects them forever.
- The threshold becomes the key setting, so it is versioned as configuration (`12-data-model.md` §4).
- Per-model licences had to be checked before anything ships. Done in `15-model.md` §11; the library
  choice is ADR-0015.
