# ADR-0014: Test units are shot good, then damaged, minutes apart, in random order

**Status:** Accepted · 2026-09-21 · Ali (design Step 6)

## Context

The old plan bought separate good and damaged units and shot all the damaged ones after the
good ones. With моти that invites the failure R12 describes. A soft, starch-dusted unit changes
over a session — drying, dust settling, softening. If every damaged image is taken an hour after
every good image, a model can separate them by *when* they were shot, and the recall it reports is
partly a clock.

Моти also cost several times more than пряники, and the budget is open.

## Decision

- **Paired.** Every test unit is shot good (3 rotations), damaged outside the rig by a standard
  recipe, and shot again (3 rotations) straight away. Its good images go to `test/good` and its
  damaged images to its defect class.
- **Random order.** Units are assigned to splits by seed before shooting and shot in a random
  order, so time is not tied to any split.
- **Controls.** Three units, in no split, are shot at the start, middle and end. Their scores
  show whether the session drifted.

## Alternatives considered

- **Separate good and damaged units, shot in blocks.** The old plan. More units, and damage tied
  to time. Rejected.
- **Damage everything first, then shoot.** It removes the time gap between classes, but damaged
  units age while the rest are handled, and there is no good image of a damaged unit to compare
  with. Rejected.

## Consequences

- Units needed: MVP 83, where the same split sizes unpaired need 113; Gate A 133, where the old plan
  needed 170.
- `test/good` and each defect class share units. That is fine: one sample measures false alarms,
  the other recall, and no unit crosses from train or calib into test (E1 still holds).
- Each test unit gives a paired score difference, damaged minus good — the direct size of a defect's
  effect, free of unit-to-unit variation. It goes in the report as a diagnostic.
- Handling doubles for test units, so gloves and the spatula matter more (§2 of the data
  collection doc).
