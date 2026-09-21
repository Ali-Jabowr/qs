# ADR-0012: When storage fails, the verdict wins, and the record outlives the image

**Status:** Accepted · 2026-09-21 · Ali (design Step 5)

## Context

Two requirements collide when the disk fills or the evidence directory becomes unwritable:

- **I9 / NFR-05** — persistence never blocks a verdict. A full disk degrades the station; it does
  not stop the line.
- **I2 / NFR-06** — every inspection writes a record; none is lost.

A bounded queue must eventually either block (breaking I9) or drop (breaking I2). An unbounded one
exhausts memory at ~500 KB per image. The two things being stored differ in size by three orders
of magnitude: a row is ~400 B, an image ~500 KB.

## Decision

1. **The operator always gets the verdict on time.** The sink never blocks the pipeline.
2. **Rows are held in memory while the process lives** and written when storage returns. At ~400 B
   each, a day's production fits in a few megabytes.
3. **Images are shed past a high-water mark** of 200 queued images: PASS images first, then the
   rest. A shed image leaves its row with `image_status = 'shed'`.
4. **Nothing is lost silently.** `STORAGE` is on the screen for the whole outage, and held and shed
   counts go into the run record.

## Alternatives considered

- **Block the pipeline until the disk returns.** Satisfies I2 and stops the line. Rejected — the
  failure the client will never forgive.
- **Drop whole records past a limit.** Simple, and it erases exactly the traceability the archive is
  sold for. Rejected.
- **Spill to a second location.** On a single-disk edge box it fails in the same event; on the bench
  it is complexity for a case that does not occur. Rejected for the MVP.

## Consequences

- A storage outage can leave PASS units without an image, never without a record. The gap is
  visible in the data.
- DEFECT images, the scarcest data in the project, are the last thing shed.
- If the process dies during an outage, the held rows are lost. That is the one remaining loss path;
  it is counted at the next start from the `seq` gap.
- The schema needs `image_status` and a nullable image path — also required for NO_VERDICT records
  that have no image at all (camera lost, busy).
