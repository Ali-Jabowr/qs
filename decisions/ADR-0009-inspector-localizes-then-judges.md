# ADR-0009: Station A's Inspector localizes the unit, then judges the unit — never the frame

**Status:** Accepted · 2026-09-21 · Ali (design Step 3)

## Context

The placement check (FR-25) and the geometry check (FR-26, ADR-0008) both need the unit's
silhouette. PatchCore is sensitive to where the object sits in the image, and anything that differs
from the training images scores as anomalous — including starch dust on the background, which is
constant at a моти line. Station M will need to localize something different: a date code, not a
round unit.

## Decision

1. **Localization is inside the Inspector.** The Inspector segments the unit from the background,
   and raises a placement fault when there is no unit or it touches the edge of the inspection
   area. The Pipeline never learns what a unit looks like.
2. **PatchCore sees a fixed-size window centred on the unit's centroid, resized by one constant
   factor.** Every unit is translated to the centre; no unit is rescaled on its own.
3. **The anomaly map is masked to the silhouette, dilated by a margin, before the maximum is
   taken.** Background pixels cannot raise an alarm.
4. **The verdict is DEFECT if any check fails**, and carries every check's result.

## Alternatives considered

- **Localization as a Pipeline stage.** Shared by both stations, but it puts product knowledge in
  the core that is supposed to be product-free. Rejected.
- **The full frame to PatchCore.** Position variation and background dust become anomalies, and
  most pixels are spent on black paper. Rejected.
- **Crop to each unit's bounding box and resize.** Per-unit scale erases size and changes texture
  scale from unit to unit. Rejected.
- **No mask on the anomaly map.** Simpler, and a steady source of false alarms from dust. Rejected.

## Consequences

- The Pipeline stays unchanged when Station M arrives.
- Background dust cannot cause a false alarm.
- A crumb or object next to the unit is ignored. That is consistent with the foreign-object scope:
  surface-visible, on the unit, best effort.
- Rotation still varies. A моти is roughly round, so this is minor; training includes rotations.
- `Verdict` needs per-check results — a contract change, made at Step 4.
