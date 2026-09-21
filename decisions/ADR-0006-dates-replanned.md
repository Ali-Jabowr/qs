# ADR-0006: The schedule is re-planned after fifteen idle days

**Status:** Accepted · 2026-09-17 · backfilled 2026-09-21 from `project.yml`

## Context

The build plan of 2026-09-02 put Stage 0 on 2 to 7 September and Gate A on 22 September. Nothing was
started in the fifteen days that followed.

## Decision

Retire the old dates rather than report them as slipping. Stage 0 moves to 17 to 22 September; Gate A
and Gate B to 13 October.

## Consequences

- Dates in `project.yml` are the only dates; nothing else states one.
- Gate A stays due 13 October under the demo-grade MVP (ADR-0007).
- Schedule honesty is a rule: a missed date is re-planned in the open, not quietly stretched.
