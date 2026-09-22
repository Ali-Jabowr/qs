# Roadmap — the bench MVP in ten plans

**Written 2026-09-21, after Gate 2 (GO).** One plan = one module or one feature. The executor works
one plan at a time; each is reviewed (`reviews/REVIEW-<id>.md`) before the next begins. Plans are
written **one ahead** of execution, so each can learn from the last. Status of every plan lives in
`project.yml`, never here.

## The plans

| Plan | What it builds | Needs | Design |
|---|---|---|---|
| P01 | Environment: pinned venv, CUDA check, headless OpenCV, Tk, weights cached offline, `make check` with ruff and mypy, licence notices — and answers to review items R17 and R18 | — | `15` §5, §11; ADR-0015, 0016, 0018 |
| P02 | `src/qs` package: contracts (values, protocols, exceptions, the C8 label rule) and the typed station config with its canonical hash | P01 | `11`; `12` §4 |
| **P05** | Camera: `WebcamSource` (settings order, read-back by ioctl, grab thread, kernel timestamps, frame validation), the base Tk `Screen`, `qs camera-test` — the lock test | P02 | `11` §3, §5; ADR-0010, 0018; `13` §3–§4 |
| P05b | Shoot tools: `qs split-plan` (units to splits by seed, damage classes, shooting order) and `qs shoot` (the guided session, dataset names, `session.yml`) | P05 | `12` §2; `13` §5–§8; `14` §9; ADR-0014 |
| P06 | Inspector, deterministic half: localize, placement rules, geometry measures, tolerance derivation | P02 | `15` §2–§4; ADR-0008, 0009 |
| P07 | Inspector, learned half: PatchCore via Anomalib, window, mask, score, heatmap, model bundle, `qs train`, `qs pilot-report`, MVTec check, Gate B proxy timing | P06 | `15` §5–§10; ADR-0013, 0015, 0017 |
| P03 | Evidence store: schema, `SqliteSink` — files first, hold and shed, stop behaviour | P02 | `12` §3; ADR-0012 |
| P04 | Pipeline, state machine, replay, stub inspector, fault-injection and security tests, headless `qs station --replay` | P03 | `11` §6–§7; `16`; `17` §6 |
| P08 | `qs build` (datasets and golden set), `qs calibrate`, `qs evaluate` — the INDICATIVE report | P04, P07 | `14` §4–§9; `15` §7 |
| P09 | The station: `ScreenAnnunciator`, `KeyboardTrigger`, live `qs station`; NFR-01 measured; the non-author test | P05, P08 | `11` §3; `16` §5; ADR-0018 |

**Executor** (ADR-0021): Claude Code implements every plan from P05 on, and one fresh Claude agent
reviews it. P01 and P02 were executed by OpenCode (ADR-0019, superseded).

P05 was split in two on 2026-09-22, so each half fits one executor session: the camera first (the
lock test needs it on Wednesday), then the shoot tools (Friday's shoot needs them).

P03 and P04 are numbered by the original skeleton task (E0.8) but run after P07: the shoot is the
one step that cannot be redone, so everything it needs comes first.

## The week

| Day | Executor | Ali, at the bench |
|---|---|---|
| Tue 22.09 | P01, P02 | `sudo apt install v4l-utils`; build the rig (`13` §3); plug in the webcam |
| Wed 23.09 | P05, P05b | Camera lock test (`13` §4) with `bin/qs camera-test` — ADR-0010 accepted or the phone takes over |
| Thu 24.09 | P06, P07 | Decide standard or lean at the shop price (OQ-8) |
| Fri 25.09 | P03 | Buy моти in the morning; pilot (`13` §5); choose the geometry; dataset shoot (`13` §6); back up to the phone |
| Sat 26.09 | P04, P08 | Read the INDICATIVE report; early-warning check (`14` §9) |
| Sun 27.09 | P09 | Demo end to end; hand it to someone who has never seen it (NFR-09) |

**What slips first:** P09's polish, then P04's lesser fault-injection cases. **What never slips:**
the order P05 → P05b → shoot → P08. A shoot done without the tools, or a report built on an unfrozen set,
is the one mistake this week cannot absorb.

**Before the shoot, walk through `13-data-collection.md` together** — Ali asked for this on
2026-09-21.

## After the week

Gate A (13.10): the full protocol on fresh units (ADR-0005, 0007). Gate B proxy (ADR-0017) is
measured by P07 and judged at the same date. Site visit, marking station and export to OpenVINO
remain product scope.
