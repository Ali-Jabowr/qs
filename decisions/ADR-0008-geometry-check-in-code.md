# ADR-0008: Size and shape are measured in code, not left to the anomaly model

**Status:** Accepted · 2026-09-21 · Ali (Gate 1)

## Context

Two of the client's four defect types — shape deviation and dimension deviation — are global
properties of the unit. PatchCore compares small patches with a memory of small patches. A моти
10% smaller than normal is made of normal-looking patches. If preprocessing also resized each
unit to a fixed size, the size information would be gone before the model saw it.

Size and roundness have a right answer. The engineering rule is a deterministic backbone with
learned components only where there is no right answer.

## Decision

Station A runs two checks on every unit:

- **Geometry** — deterministic. From the unit's silhouette: area, equivalent diameter and
  roundness, compared with tolerances derived from good units. Pass/fail against a golden
  reference, in pixels, not millimetres.
- **Appearance** — PatchCore anomaly score against the threshold.

A unit is DEFECT if either check fails, and the verdict records which one. The geometry check uses
the same segmentation that the placement check (FR-25) already needs.

## Alternatives considered

- **PatchCore alone.** Rejected: it is weak exactly where the client asked for two of their four
  defect types.
- **A learned size model.** Rejected: it would learn a quantity that a contour measures exactly.

## Consequences

- The client's shape and size defects are covered in the MVP, and are the easiest thing to
  demonstrate: put a small моти under the camera.
- Tolerances in pixels hold only while camera height and lens stay fixed. A rig change means new
  tolerances, which the config version records.
- Millimetres later (OQ-1) mean adding a pixel-to-mm calibration. The code path is unchanged.
- The Verdict must carry per-check results, not a single score — a contract change for Step 4.
