# ADR-0013: One frame serves both checks, so the pilot compares three front-lit geometries

**Status:** Accepted · 2026-09-21 · Ali (design Step 6). The winning geometry is recorded here
after the pilot.

## Context

`04-build-plan.md` planned to compare low-angle, diffuse and backlight. Since then Station A has
two checks on every unit (ADR-0008): geometry needs a clean silhouette, and appearance needs
surface relief. A backlit frame is a perfect silhouette with no surface at all — it cannot show a
dent. Using it would take two frames per trigger under two lights, switched between them.

Low-angle light shows relief best, but on a domed моти it can leave the far side in shadow. On a
black floor, a shadowed edge disappears into the background and the silhouette shrinks.

## Decision

One frame per trigger serves both checks. The pilot (`13-data-collection.md` §5) compares:

- **G1** — low-angle, one side
- **G2** — low-angle, two opposite sides
- **G3** — diffuse

The choice is made in order: first a clean silhouette (area varies under 2% between rotations of
the same unit), then the widest separation between damaged and good scores. A tie goes to G2.
The pilot's units are discarded afterwards.

## Alternatives considered

- **Keep backlight in the comparison.** It cannot see a dent, so it cannot be the one geometry.
  Rejected.
- **Two frames per trigger** — a backlit one for geometry, a low-angle one for appearance. The best
  of both, but it needs switchable lights and doubles capture time and failure modes. Rejected for
  the MVP; it is the upgrade path if no front-lit geometry passes the silhouette test.

## Consequences

- One capture per unit, which keeps the latency budget and the contracts unchanged.
- G2 needs two similar lamps.
- If all three fail the silhouette test, the MVP falls back to the two-frame design. That is a
  real delay, and the pilot finds it out on day one rather than day five.
