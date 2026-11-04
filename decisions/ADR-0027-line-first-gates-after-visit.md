# ADR-0027: Line work goes first; Gate A loses its date and no longer blocks the visit stages

**Status:** Accepted · 2026-11-04 · Ali. Amends Gate C in `project.yml`; the Gate A protocol of
[ADR-0005](ADR-0005-gate-a-restated.md) and `docs/14-eval-protocol.md` is unchanged.

## Context

The plan was written before anyone saw the line: bench first, Gate A, then a site visit with the
demo in hand. The visit happened on 14–16.10 without a model or a demo, and it changed the problem
(`docs/mentor/2026-10-19-report.md`):

- The product stands in clear cups on a chain over polished steel. Today's `localize` finds the unit
  in 0 of 2 883 line frames, so no defect check can run there at all.
- An empty cup is a defect the plan does not contain.
- The bench is not available for now, and Gate A needs it and 133 fresh моти.

Read literally, the plan then stops. The second visit needs a rigid overhead mount and a glare
diffuser. Gate C forbids buying them until Gate A passes, Gate A waits for the bench, and Gate C also
wants defective samples and an answer to OQ-1, neither of which the first visit produced.

## Decision

1. **Line work goes first.** Stage L1 builds the cup-first localizer and the empty-cup check. Stage L2
   is the second visit. The bench stages S2, S3 and GA are blocked on bench access and keep their
   tasks.
2. **Gate A has no date until the bench is back.** Its protocol does not change. A date nobody can
   meet would only teach the warnings to be ignored.
3. **Gates A and B no longer block a stage.** S5 ran without them, and L2 is not bench work.
4. **Gate C gets one exception:** a rigid camera mount and a glare diffuser for the second visit,
   once the food-safety requirements for equipment over open product are known. Everything else, the
   industrial camera and lens above all, stays behind the gate.

## Alternatives considered

- **Keep Gate C strict and do the bench first.** Protects the rule, but stops line work for as long
  as the bench is away, and Gate A says nothing about glare on steel. Rejected.
- **Drop Gate C.** The expensive purchases still depend on OQ-1 and on line evidence. Rejected.
- **Keep Gate A at 15.11.** Eleven days, no bench, 133 units. Rejected as a date we know is false.

## Consequences

- **No bench number will exist for a while.** Line results stay INDICATIVE and say nothing about
  dents: dents and tears cannot be told apart reliably on the footage from the first visit.
- **The first visit's footage came from a temporary, angled camera**, and the second visit plans an
  overhead one. L1's score on that footage shows whether the method works, not what it will score on
  the line, so it gets light tuning only (risk R13).
- **FR-25 must change in L1:** today "no unit" ends in NO_VERDICT(`placement`), which would report an
  empty cup as a placement problem, not as a defect.
- The mount and diffuser sit over open product, so the exception depends on food-safety sign-off
  (OQ-6) and a purchase can still be refused after it is made if that is skipped (R7).
