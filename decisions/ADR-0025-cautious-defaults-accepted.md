# ADR-0025: The cautious defaults of the build's twenty open questions are accepted

**Status:** Accepted · 2026-09-24 · Ali.

## Context

While specifying P03–P09, every gap the docs left open became a `wayfinder` + `question` issue. Each
one stated a cautious default, and the tickets built that default until Ali answered. By
2026-09-24 all of P03–P09 was built and had passed its review, with twenty such questions still
open. Ali was given each question with a recommendation (accept, in every case) and answered:

> "defaults accepted, go ahead"

## Decision

The default stated in each issue below is the decision. The code already implements it, so nothing
changes in `src/`. Each issue is closed with a note pointing here.

| Issue | Plan | Decision in one line |
|---|---|---|
| #28 | P05b | A spoiled shot moves to `rejected/`, never deleted; a broken unit is discarded and the next spare takes its place |
| #29 | P05b | `raw/<session>/frames.csv` holds the per-frame metadata beside `session.yml` (docs/12 §2) |
| #30 | P05b | Product fields are given to the shoot tool before the first frame; only `notes` is hand-written, before the backup copy |
| #37 | P06 | The localize threshold is suggested once, by Otsu on the pooled pilot histogram; Ali types it into the config |
| #38 | P06 | The pilot's 2% criterion is (max − min) / mean of `area_px` over rotations, for every good unit |
| #43 | P07 | The pilot's separation is pessimistic (the worst damaged against the worst good); the spread is the sample σ; ties are to 2 decimals; the window is ⌈2 × median diameter⌉ |
| #44 | P07 | The MVTec check runs on `bottle`, and passes at image AUROC ≥ 0.98; validation mode is an argument, never a config key |
| #51 | P05b | Phone shoot: Xiaomi, Pro mode, JPEG, 12 MP, taped flat, 2 s timer, 1/50 s, fixed white balance; every cloud and messenger sync off |
| #55 | P03 | A record the store can never write is rejected and counted; `STORAGE` stays up; it is never rewritten into another label |
| #56 | P03 | Shedding order: PASS, then НЕТ РЕШЕНИЯ, then DEFECT; oldest first within each |
| #57 | P03 | At a stop during an outage, the station waits 10 s *without progress*, not 10 s in total |
| #62 | P04 | The reference-patch band is relative: \|patch − expect\| ≤ band × expect |
| #63 | P04 | One streak per step; events that never reach a step do not touch its streak |
| #77 | P08 | `qs calibrate` writes a new file (`--out`, must not exist) and never overwrites; `version` is unchanged |
| #78 | P08 | If geometry alone flags past the target, calibrate refuses; nothing widens automatically; `--far 10` is the escape |
| #79 | P08 | Drift when a control's shift ≥ the median paired dent difference, or when the gap is ≤ 0; the report says so and exits 1 |
| #85 | P09 | Until the webcam: key-paced replay on screen; NFR-01 and NFR-09 are not claimed on replay |
| #92 | P07 | score ≤ threshold is decided exactly in float64; calibrate writes the score's exact repr |
| #93 | P08 | Evaluate's latency covers only frames that reached the model; an unmeasurable gap or shift counts as drift |
| #94 | P09 | For the MVP, the run-level `source_id` marker is enough |

## Alternatives

Each issue lists its own alternatives. None was chosen.

## Consequences

- **docs/12 §2** gains the `frames.csv` line (#29).
- **Follow-up after the MVP week (#94):** a `run.source_id` column, written at run start so that it
  survives a kill, before any client-facing archive. This is a P03 schema change.
- **#78 will probably bite on Sat 26.09.** With 10 calibration units, a 5% target allows no flagged
  unit, so docs/18 Part G gives the `--far 10` step.
