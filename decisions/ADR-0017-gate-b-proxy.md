# ADR-0017: Gate B is judged on a conservative N100 estimate until an N100 exists

**Status:** Accepted · 2026-09-21 · Ali (Gate 2, review finding R10)

## Context

Gate B asked for ≤ 700 ms at p95 "on CPU as the Intel N100 proxy". There is no N100 (Gate C), and
the laptop CPU is estimated 2–3× faster (`15-model.md` §10). As written, a laptop result of 650 ms
would pass while the N100 took 1.3–2 s. The gate also did not say whether the camera was included.

## Decision

Until an N100 exists, Gate B is: on replay, camera excluded, laptop CPU pinned to 4 threads, at least
200 consecutive inspections — **p95 inspection time × 3 ≤ 400 ms**, the architecture's inference
budget. The real gate, ≤ 700 ms end to end on an N100 with the factory camera, runs after Gate C.

## Alternatives considered

- **Keep the gate as written.** Passes on a lower bound. Rejected.
- **Defer Gate B until an N100 exists.** Honest, but it removes the one signal that could move the
  model choice — coreset, resolution, backbone — before the factory rather than at it. Rejected.

## Consequences

- The gate errs towards failing: × 3 is the top of the estimated range.
- The default PatchCore configuration may well fail it. The fallback order in `15-model.md` §10 is
  then run on the golden set this month, not discovered at installation.
