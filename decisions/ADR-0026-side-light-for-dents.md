# ADR-0026: The bench is lit from one side (G1), because top light hides dents

**Status:** Accepted · 2026-10-09 · Ali. Records the pilot's outcome for [ADR-0013](ADR-0013-pilot-geometries-one-frame.md)
and replaces the lighting pilot of `docs/13` §5 for the bench.

## Context

The webcam passed the lock test inside a PULUZ USB light box, which lights the floor evenly from
above (G3, ADR-0010). Dents are the first defect the client named. Before ordering the dataset's
моти, a dent check on the 5 моти at hand asked whether the camera can see one.

**Dent check, 2026-10-09, 5 моти, each shot good then damaged** (`data/raw/2026-10-09-dentcheck/`):

| Unit | Damage | Top light (G3) | Side light (G1) |
|---|---|---|---|
| m3 pale yellow | dent, gentle | not visible | **visible**: a dimple with a shadow |
| m3 | dent, pressed deeper | faint shading; Ali: "barely" by eye | — |
| m1 pink | dent | visible only as starch pushed off | **clearly visible**: hollow, ridges, shadow |
| m4 brown | dent | not visible | **visible**: a dark hollow |
| m2 brown | tear | visible: filling in the slit | — |
| m0 yellow | squashed | visible; area +44% | — |

**Second round, 5 fresh моти of mixed flavours, all under side light, each good then damaged**
(`data/raw/2026-10-09-dentcheck-side/`):

| Unit | Damage | Visible? | Through what |
|---|---|---|---|
| m1 pink-orange | dent, firm | **yes** | hollow with a shadow; area +14% |
| m2 light pink | three dents | **yes** | starch pushed off; the deepest with a shadow; area +11% |
| m3 pale yellow | dent, **gentle** | **no** | area +0.05%: the unit did not spread, the starch stayed |
| m3 | dent, firm, the same unit | **yes** | a 1.5 cm crater, shadowed wall; area +10% |
| m4 brownish | tear | **yes** | the slit's shadow; solidity 0.951 → 0.872 |
| m5 caramel | squash | **yes** | axis ratio 0.858 → 0.744, solidity 0.940 → 0.873 |

Side light shows every firm dent, tear and squash. A gentle dent that neither moves the starch nor
spreads the unit stays invisible from above, to the camera and, by Ali's eye, nearly so.

Under even top light a shallow hollow gets as much light as its surroundings and casts no shadow.
A dent then shows only if it disturbs the starch or spreads the unit. A lamp low at one side puts
the hollow's far wall in shadow.

## Decision

Light the bench with **one lamp low at one side, about 20–30° above the mat, the box's top light
off** — geometry G1. The formal three-way pilot is not run: the light box cannot make G2, G3 is
shown above to hide dents, and the 5 моти were the pilot.

Camera settings refitted for G1 on a pale моти: exposure 2 ms, **gain 25** (clipping 0.02%),
reference patch expect 0.48, **localize threshold 90** — the mat's woven texture lights up to 70
(p99.9) under grazing light, a pale unit's darkest 5% sit at 110. The side lamp does not flicker:
paper brightness std 0.31 of 255 over 40 frames at 2 ms, no banding.

## Alternatives considered

- **Keep G3 and accept dents as best effort.** Loses the client's first-named defect. Rejected.
- **Run the full pilot (`docs/13` §5).** It needs 6 more моти to compare a geometry the box cannot
  make. The dent check answered the question it exists for. Rejected.
- **Two frames per unit, top and side light.** Better coverage, but two lights switched per trigger
  (ADR-0013 rejected that for backlight). Kept for later if side light misses dents at the line.

## Consequences

- **The lock test must be run again** under G1 before the dataset shoot: the 2026-10-09 pass was
  under the box's top light. The 8-second check above already varied 0.84% against the 1% limit.
- **The lamp is fixed and marked**: same place, height and angle for every frame of the dataset.
  Paper and mat are taped: both moved when units were placed.
- **The dataset is one pale flavour.** Both brown units lost part of their outline on the black mat
  (solidity 0.76 and 0.87 when good).
- **Side light also sharpens normal texture** — starch grains, creases — so good units vary more
  to PatchCore. Train on the full lean count.
- **Starch dust** left on the mat glitters under side light; wipe the mat between units.
- **What we can promise is "visible dents", not every dent.** Which dents cause the client's
  refunds is a question for the client, beside the 95% (OQ list). The dataset's ~3 mm recipe sits at
  the edge of visibility, which keeps the test honest.
- Thawed моти are soft: a firm press spreads them 10–14%, so a calibrated geometry check also
  catches many dents.
- On the line the light must again be low and to one side. That shapes the factory light tunnel.
