# REVIEW-P06, round 1: Inspector, deterministic half (localize, geometry, tolerances)

**Verdict: PASS.** 0 blocker · 0 major · 0 minor · 1 nit.

Spec #36, tickets #39 (19b493d), #40 (27d9ac1) and #41 (063a06b) on `mvp/bench`. The tickets were
implemented by OpenCode (glm-5.3-flash, ADR-0023). Reviewer: the Claude Code orchestrator, which wrote
none of the code. Each ticket was reviewed on its own before it was committed. This file records those
reviews and the plan-level checks.

## Per-ticket reviews (before each commit)

| Ticket | Findings sent back | Result |
|---|---|---|
| #39 localize | (1) `CHAIN_APPROX_SIMPLE` reduces a straight-sided outline to 4 points, so #40's `fitEllipse` would turn a square or squashed unit into NO_VERDICT(inference) instead of a geometry verdict. Now `CHAIN_APPROX_NONE`, with a full-boundary test on a square. (2) The "one pixel closer" test didn't prove its premise. It now asserts gap = margin − 1 | fixed in one round |
| #40 geometry | `min(axes)/max(axes)` raised a bare `ZeroDivisionError` on a degenerate fit. It now raises `InferenceError` (C8), with a monkeypatched test | fixed in one round |
| #41 tolerances | none. Otsu on the pooled histogram was checked against `cv2.threshold(THRESH_OTSU)`: 300 of 300 probe images agree. The hand-worked example uses literal arithmetic, not the code's formula | — |

Accepted readings of ambiguous ticket text: `failed_on` joins with `", "` (the same as
`derive_label`), and `GeometryMeasures.measures` is a property. Two refusals were added in #41
beyond the ticket's list, both fail-loud: a one-bin pooled histogram, and a unit whose mean area
is ≤ 0.

## Plan-level checks

- `make check`: 424 passed, 6 deselected; ruff and strict mypy clean (23 files); 17 diagrams.
- The P06 tests (68) run in 1.96 s in total. The slowest is 0.40 s (the 60-frame round trip).
- No `socket`/`http`/`urllib`/`requests` import in `src/qs/inspector`. There is no clock, file or
  image I/O in the three modules, so they are pure (C1, C2).
- No change under `docs/`, `config/`, or to `config.py`, `contracts.py` or `errors.py`. P06 added only
  `src/qs/inspector/*` and its tests, plus `tests/conftest.py`.
- Consistent across tickets: `localize` keeps a pixel when it is `> threshold` (`THRESH_BINARY`), and
  `_otsu` returns t with the split [0..t] | [t+1..255]. So a suggested value can be typed straight
  into `inspector.localize.threshold` with the same meaning. #40's timing: localize + measure + check
  takes a 3.4 ms median on 1920 × 1080, against the 20 ms budget.
- What P07 needs from here exists as the spec promised: `Silhouette.mask` (frame-sized, read-only),
  `centroid` and `unit_box` in frame coordinates, and `measure`/`check_geometry` →
  `CheckResult`.

## Nit

- **N1:** `_BLUR_KSIZE = (5, 5)` is defined in both `localize.py` and `tolerances.py`. The threshold
  suggestion is only right if its blur is the Localizer's blur. Importing the one constant would tie
  them together. Left as is: both follow docs/15 §2's 5 × 5, and a test would catch a drift only
  indirectly.

## bench-pending

As the tickets say: A-04 and the 2% area spread on real моти via `qs pilot-report` (P07), and the
tolerances and the x01–x10 placement frames via `qs calibrate` / `qs evaluate` (P08).

## Verdict

**PASS.** P06 is done. Next: P07 #45 → #47, which `qs pilot-report` needs before the shoot.
