# ADR-0028: On the line the unit is the cup, and v1 triggers on seeing it

**Status:** Accepted · 2026-11-04 · Ali. Amends FR-25 and contract rule C5; builds on
[ADR-0027](ADR-0027-line-first-gates-after-visit.md). The bench keeps its behaviour. §3 and §5 amended
by [ADR-0029](ADR-0029-cup-rim-under-own-backlight-warmth-verdict.md) (2026-11-10).

## Context

The design was written for one моти on a mat: an operator pressed a key, and a frame with no моти
meant the unit was badly placed, so it ended in NO_VERDICT(`placement`). On the line the same logic
reports an empty cup, the factory's main defect, as a placement problem and asks the operator to place
the item in the frame, though nobody places anything there.

The line also has no key press. Something must say that a cup has arrived. A spike on the first
visit's footage (19.10) found the cup rim in 1 379 of 1 882 frames. That count is per frame, was not
checked against labels, and comes from a temporary, angled camera. A cup is in view for many frames,
so the rate per cup is unknown and may be much better.

## Decision

1. **On the line the unit is one cup with whatever is in it.** Every trigger leads to one verdict about
   that cup. An empty cup is a defective unit. On the bench the unit stays one моти.
2. **v1 triggers in software, on the cup finder seeing a cup.** The trigger stays behind `TriggerSource`,
   so the photo sensor (FR-16) can replace it without touching the rest of the system.
3. **A cup is judged once, on the first frame showing all of it in a fixed inspection zone.** A cup seen
   in view that never gets such a frame ends in NO_VERDICT(`placement`).
4. **A missed cup, never seen in any frame, has no event and no record.** C5 now says so. ГОДЕН expires
   from the screen so that it cannot stand for a cup nobody judged.
5. **The empty-cup check is measured, not learned.** It is a new check beside appearance, and it
   reports the cup as empty in words. White моти are scored separately. If it cannot separate them from
   glare, a small classifier on labelled cups replaces it.
6. **The line station runs the empty-cup and appearance checks.** Geometry stays off until overhead
   footage shows it works there. Until then ГОДЕН on the line means the cup holds a моти and nothing
   looks unusual, and it promises nothing about size or shape.
7. **No hand detection in v1.** The normal checks decide, and labelling counts how often a hand appears.

## Alternatives considered

- **The моти as the unit, with the empty cup a separate event.** Stays closer to today's code, but puts
  the main defect outside the verdict, the evidence record and the scoring. Rejected.
- **The existing photo sensor as the trigger.** Independent of the cup finder, so a missed rim would give
  a visible НЕТ РЕШЕНИЯ and no cup could vanish. It is the better design, and it needs wiring and the
  client's agreement, and cannot be tested until the second visit. Deferred, not rejected.
- **A strip across the chain that fires on any change in the picture.** Also independent of the cup
  finder, but glare and hands may fire it and a clear cup may not. Kept as the fallback if the sensor
  proves unusable.
- **A rhythm check on the 21 cm cup spacing,** showing НЕТ РЕШЕНИЯ when the next cup is overdue. It makes
  misses visible, but needs to know the chain is moving and may false-alarm on chain positions meant to
  be empty. Decided by the measured miss rate, not now.
- **PatchCore alone for the empty cup.** It flagged 80 of 81 in the spike, but it says only "unusual", and
  needs a model for each camera position before any empty cup can be caught. Rejected as the only check;
  it still runs beside the measured one.

## Consequences

- **A missed empty cup goes to packing with nothing on the screen and no record.** This is the price of
  decision 2, and the reason it is a v1 choice. The operator must keep watching the cups until a rhythm
  check or the sensor exists, and the count on the report may not match the count on the line.
- **The share of empty cups caught, per cup, is L1's main number.** The finder caps it: an empty cup it
  never sees cannot get БРАК. Measuring it needs whole cups counted through the video, not only single
  frames labelled.
- **Two of the client's four defect types, shape and size, are not measured on the line for now.**
- **The checks a station runs must be set per station,** because the bench keeps geometry.
- **Code still to change in L1:** the inspector chain, `qs train` (old localizer, bench file names), the
  screen's verdict expiry, and the `TriggerSource` docstring that repeats the old C5.
- **Line numbers stay INDICATIVE** and say nothing about dents or the 95% (R13).
