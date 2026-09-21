# ADR-0005: Gate A is restated so it can tell a working system from luck

**Status:** Accepted · 2026-09-17 · backfilled 2026-09-21 from commit 6159d5b. Revised for моти and
paired shooting on 2026-09-21 (ADR-0014).

## Context

The Gate A of 2026-09-02 asked for 90% dent recall on 5 dents. 90% of 5 means 5 of 5, and 5 of 5 is
consistent with a true catch rate of 54.9%. On the false-alarm side, 10 good units resolve a rate in
steps of 10%, so "at most 5%" was not measurable. And a threshold chosen on the same units whose
false-alarm rate is then reported returns the target by construction.

## Decision

- **30 dented units.** The smallest number at which an observed 90% (27 of 30) clears a 75% lower
  bound.
- **The threshold is set on a calibration split** and the false-alarm rate measured on separate
  units.
- **The false-alarm figure is a point estimate** with its upper bound, not a confidence claim — a
  5% claim at 95% confidence needs 59 good units with none flagged.
- Every bound is computed by `tools/bounds.py`, not asserted.

## Consequences

- More units: 170 in the 2026-09-17 plan; 133 after paired shooting (ADR-0014), with 70 good test
  units instead of 40.
- The gate's definition lives in `14-eval-protocol.md` §6 and nowhere else.
