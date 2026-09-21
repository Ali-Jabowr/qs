# ADR-0007: The MVP is a demo-grade bench build; Gate A stays a separate claim

**Status:** Accepted · 2026-09-21 · Ali (Gate 0)

## Context

The week of 2026-09-21 is for delivering the MVP; client work is deferred. The product is моти
(owner call, 2026-09-20): perishable, soft, possibly frozen.

The full evaluation protocol (`docs/14-eval-protocol.md`) needs 170 units, three lighting
geometries × three rotations (~1530 frames) and 70 damaged units, so that Gate A's recall figure
carries a defensible confidence bound. The shooting alone is about ten hours. Squeezed into one
week, the corners that get cut are the ones the number's validity rests on: locked camera
settings, splits by unit, and good and damaged units shot in the same session.

## Decision

The MVP runs the real pipeline end to end on the bench, on one flavour of store-bought моти.

- A short pilot of the three lighting geometries on about ten units picks one. The dataset is shot
  in that geometry only.
- Numbers from the MVP are reported as **indicative** and are never presented as the Gate A result
  or quoted to the client as accuracy.
- Gate A keeps its full protocol, due 2026-10-13, on freshly bought units that no MVP step has seen.

## Alternatives considered

- **Gate-grade MVP** — the full protocol this week. Rejected: it fits only if nothing goes wrong,
  and the likely failure is a quietly invalid number rather than a visible delay.

## Consequences

- The week has room for one re-shoot.
- Gate A's holdout is uncontaminated by anything tuned during the MVP.
- The MVP produces no accuracy claim that can be put in front of the client.
- Product is bought twice: once for the MVP, once for Gate A.
- A geometry chosen on ten units may not be the best one; Gate A re-tests it.
