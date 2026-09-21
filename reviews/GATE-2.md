# Gate 2 — design review

**Date:** 2026-09-21 · **Scope:** the bench MVP (ADR-0007) · **Reviewers:** Claude Code (author),
an independent reviewer with no part in writing the design (§3), Ali (decision, §5)

*Is the technical plan sound, secure and complete enough to start implementation?*

---

## 1. Deliverables required at Gate 2

| Required | Where | Status |
|---|---|---|
| Design doc — context, goals, non-goals, design, APIs, data, alternatives, cross-cutting | `10`, `11`, `12`, `15`, `16`, `17` | Present |
| ADRs for every decision | `decisions/` 0001–0016 | Present — 0001–0006 backfilled today (E0.7) |
| C4 — context, container, component | `10` §2, §4, §6; the Inspector in `10` §4 and `15` §1 | Present |
| One sequence diagram per critical flow | Inspect a unit — `10` §5. Train, calibrate, evaluate — `15` §7 (added at this review) | Present |
| State diagrams, with failure states | The station and the record — `16` §2, §4 | Present |
| Contracts for everything crossing a boundary | Values and protocols `11`; schema and config `12`; `session.yml` `13` §8; `bundle.yml` `15` §8 (fields listed, schema pinned in its plan) | Present, one pinned at planning |
| Threat model and data handling | `17` | Present |
| Evaluation embedded — golden set, held-out split, audit trail | `14`, `13` | Present |

## 2. Traceability — every MVP requirement to its design and its check

The Phase 4 matrix is this table with a result column added.

| Req | Designed in | Verified by |
|---|---|---|
| FR-11 one event per press, gapless | `11` §3 `poll()`, C5; `16` F9–F10 | Test: auto-repeat, busy, gapless `seq` |
| FR-12 one fresh frame | `11` C4, C9; `10` §5 threads; ADR-0010 timestamps | Test: no frame after `t0`; stale frame never inspected |
| FR-13 settings applied and read back | `11` `CameraReadback`; ADR-0010; `16` start-up step 3, F4 | Test: fake camera mismatch → STOPPED; lock test (demo) |
| FR-14 reference patch | `10` §3; `11` §5; `16` F5 | Test: darkened patch |
| FR-15 replay | `11` §6 | Test: replay run |
| FR-21 exactly one label | `11` §2, C8; `12` CHECK | Test |
| FR-22 appearance check | `15` §5–§7; ADR-0015 | Test: stub inspector; integration on the golden set |
| FR-23 heatmap on DEFECT | `15` §6 | Test |
| FR-24 NO_VERDICT with reason, never PASS | `11` §7; `16` §3; C7 | Test: fault-injection suite, `16` §7 |
| FR-25 placement | `15` §2 | Test: placement set in the golden set, `14` §9 |
| FR-26 geometry | `15` §3; ADR-0008 | Test: synthetic circle, ellipse, notched masks; deformed recall in the report |
| FR-31 live view | `11` `preview()`; ADR-0011 | Demo |
| FR-32 colour and word, amber NO_VERDICT | `16` §5; `12` `display`; ADR-0011 | Test: label-to-colour mapping; demo |
| FR-33 heatmap or failed measure on screen | `15` §6; `16` §5 | Demo |
| FR-34 session panel | `16` §6 | Demo |
| FR-41 the record | `12` §3 | Test: schema |
| FR-42 append-only | `12` triggers; `17` T2 | Test |
| FR-43 batch = run on the bench | `12` §3 | Test |
| FR-51 shoot mode, PNG, `session.yml` | `13` §6, §8; `12` §2 | Test: names and manifest |
| FR-52 splitter by unit | `12` §2; `14` E1; `13` §6 | Test: overlap refused |
| FR-53 train | `15` §5, §8 | Test; report |
| FR-54 calibrate | `15` §3, §7 | Test |
| FR-55 evaluation report | `14` §7, §9 | Test; report |
| FR-56 golden set reproducible | `14` §9; `15` §7 flow | Test: regenerate, identical numbers |
| FR-61 one command, live or replay | `11` §3 main loop; `10` §6 | Demo — command names pinned in the plans |
| FR-62 no network | `17` T1, T5; ADR-0016 | Test: offline run; import scan |
| NFR-01 ≤ 1.0 s p95 | `10` bench budget, ~430 ms | Report |
| NFR-02 CPU latency | `15` §10 | Report |
| NFR-03 determinism | `15` §9 | Test |
| NFR-04 fault tolerance | `16` §7 | Test |
| NFR-05 persistence never blocks | `16` F11; ADR-0012 | Test |
| NFR-06 no lost record | `16` §4 | Test |
| NFR-07 on-prem | `17` §6 | Test |
| NFR-08 reproducibility | `14` §9 | Test |
| NFR-09 non-author use | `06` §4 | Demo |
| NFR-10 `make check` green | `Makefile` | Test — ruff and mypy are not installed yet (finding A3) |
| NFR-11 licences recorded | `15` §11 | Inspection — a notices file is pinned in a plan |
| NFR-12 pinned environment | `15` §11; ADR-0016 | Demo: fresh clone |

**Every MVP requirement has a design and a check.** None is orphaned.

## 3. Findings

### 3.1 The author's own

| | Severity | Finding | Resolution |
|---|---|---|---|
| A1 | Major | ADR-0002, 0005 and 0006 were cited as foundations but never written | Fixed at this review: 0001–0006 backfilled |
| A2 | Major | The offline flow had no sequence diagram | Fixed: `15` §7 |
| A3 | Minor | `make check` has no ruff or mypy, and neither is installed, so NFR-10 cannot pass yet | The environment plan installs them and extends `make check` |
| A4 | Minor | Command names (FR-61) and the `bundle.yml` schema exist only as descriptions | Pinned in the plans — the executor must not choose them |
| A5 | Minor | CLAUDE.md, README, `project.yml` and docs 02–04 still describe пряники, the RTX 3060, 400 mm and "no code written" | Step 10 |

### 3.2 The independent reviewer's

A reviewer with no part in writing the design read CLAUDE.md, docs 05–17 and every ADR. Its verdict:
**conditional pass**. It checked the arithmetic — every bound, unit total, memory size and latency
sum — and found it correct. Every finding below was verified against the files before being fixed.

| | Sev. | Finding | Resolution |
|---|---|---|---|
| R1 | **Blocker** | Shoot filenames collide: good and damaged, control passes, pilot geometries | Fixed — `<unit>_<state>_<geometry>_r<rot>[_pass<k>]`, unit prefixes u/c/p/x (`12` §2) |
| R2 | **Blocker** | `class Inspector(Protocol)` line lost in an edit; `Health`, `StationConfig`, `InspectionRecord` undefined | Fixed — restored and defined (`11` §2, §3). The lost line was the author's own editing error |
| R3 | Major | Geometry measures named three ways | Fixed — area, axis ratio, solidity everywhere; diameter reported only |
| R4 | Major | Tolerances derived in two places | Fixed — at calibration, from train ∪ calib (`14` §4, `15` §3) |
| R5 | Major | C8 let a PASS stand with a check missing; C8 and the error table disagreed | Fixed — PASS only if every configured check passed; any exception in `inspect()` is NO_VERDICT |
| R6 | Major | NO_VERDICT(busy) unreachable on the bench; the debounce untestable | Fixed — key presses are discarded and counted, not recorded; busy only for unit triggers (`11` §3, `16` F9–F10) |
| R7 | Major | FAULT had one exit and one reason for three causes | Fixed — FAULT(camera / capture / inference), each with its own exit and message |
| R8 | Major | Unreadable camera settings handled two ways | Fixed — unreadable: inspect, `camera_verified = 0`; readable and different: NO_VERDICT |
| R9 | Major | Gain never set; `gain_db` wrong for UVC | Fixed — gain set and lock-tested, raw units |
| R10 | Major | Gate B passes on a laptop lower bound | Decided — proxy gate, p95 × 3 ≤ 400 ms (ADR-0017) |
| R11 | Major | Latency inside a report required to be reproducible | Fixed — latency in its own section, excluded from NFR-08 |
| R12 | Major | Corrupt frames had no design | Fixed — F15, capture validation, a test |
| R13 | Major | Golden set undefined; NO_VERDICT counting; lean set under 200 frames | Fixed — layout and manifest (`12` §2), counting rule (`14` §9), replay loops to 200 |
| R14 | Major | `model_id` without hash; two sources for window size; no `data_source`; hash-only disjointness | Fixed — full `model_id` in config, bundle authoritative, unit IDs and `data_source` in the bundle |
| R15 | Major | Sink queue could reach ~2.9 GB of raw pixels | Fixed — encoded before queuing, ~110 MB at high-water; heatmap in score units, window coordinates |
| R16 | Major | Lost tail rows claimed to show as a `seq` gap | Fixed — they cannot; the stop path waits 10 s and prints the count (ADR-0012) |
| R17 | Major (verify) | Backbone likely fetched via timm and the Hugging Face Hub, not `TORCH_HOME` | Adopted — `.hf/` cached and ignored, `HF_HUB_OFFLINE=1`; confirmed at the environment step |
| R18 | Major (verify) | Processor flags belong to the Lightning wrapper, not `PatchcoreModel` | Adopted — the torch model has none; confirmed at the environment step |
| R19 | Minor | "OK" label; 75 vs 55 GB a month; bound-table headers | Fixed |
| R20 | Minor | ADR-0010 overstated what was verified; t0 vs t0 + exposure | Fixed |
| R21 | Minor | "Gap recorded" vs gapless; batch = session or run; brand; `low_angle`; table count; `camera_verified` with no frame | Fixed |
| R22 | Minor | `threshold_id` not unique; calibration context not recorded | Fixed — hash suffix and `set_on` in the config |
| R23 | Minor | The lean pilot left no good unit to score | Fixed — leave-one-out |
| R24 | Minor | I4 vs the streaming camera; "armed"; cuDNN determinism; canonical config hash | Fixed |
| R25 | Near-miss | A legal conclusion ("research use") stated as fact | Fixed — moved to the lawyer line |
| R26 | Near-miss | A backup stick and a second lamp may mean buying something | Decided — nothing bought; backup to the phone over USB; one lamp means G1 and G3 only |

After the fixes: the schema re-tested (constraints and append-only), the config parsed, 17 diagrams
render, the existing 44 tests pass, and a sweep for the old terms comes back empty.

## 4. Open items that do not block implementation

| Item | Needed by | Owner |
|---|---|---|
| Unit count, standard or lean | The shop | Ali |
| A backup drive | The first shoot | Ali |
| Lock-test result — ADR-0010 | First build day | Ali, with the webcam |
| Pilot geometry — ADR-0013 | After the pilot | Ali |
| Backbone-weight licence, hands in frame under 152-ФЗ | Before anything ships to the client | Lawyer line |
| OQ-1 … OQ-6 | Product scope | Deferred by decision |

## 5. Verdict

**GO — Ali, 2026-09-21.** Implementation (Phase 3) starts on the bench MVP.

Conditions, each already a task in the plans:

1. **Confirm at the environment step** how Anomalib 2.6.2 loads the backbone (timm and the Hugging
   Face Hub, R17) and where the processor flags live (R18). If either differs from the design, the
   design is corrected before code depends on it.
2. **Bring project state up to date** — `project.yml`, CLAUDE.md, README, and superseded banners on
   docs 02–04 (A5). Step 10.
3. **Pin in the plans** what the design only describes: command names, the `StationConfig` fields,
   the `bundle.yml` schema (A4).

Two decisions taken at this gate: Gate B is judged on a conservative N100 estimate (ADR-0017), and
nothing is bought — raw images are backed up to the phone over USB (R26).
