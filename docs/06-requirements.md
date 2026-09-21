# 06 · Requirements — Phase 1

**Written 2026-09-21 · Status: Gate 1 GO (Ali, 2026-09-21) · Owner of the "what": Ali**

What the bench MVP must do, stated so each line can be checked, plus the product requirements the
MVP design must not block. Built on `05-concept.md` (Gate 0 GO) and `decisions/ADR-0007`.

**Reading the tables**

- **ID** — stable and never reused. Gaps are left deliberately so related requirements can be
  inserted without renumbering.
- **Scope** — **MVP**: built and verified this week. **Product**: not built now; the design must
  leave room for it.
- **Basis** — where the requirement comes from: `fact` (the client said it — `Qn`, or the owner
  call of 2026-09-20), `invariant` (`10-architecture.md` §7), `design` (docs 10–14), `A-n` (an
  assumption, §6), `OQ-n` (open).
- **Check** — **T** automated test · **R** generated report or measurement · **D** demonstration ·
  **I** inspection of code or docs. Phase 4's traceability matrix is this column filled in.

---

## 1. Stakeholders and roles

| Who | Role in the MVP | Role in the product |
|---|---|---|
| **Ali** | Product owner and builder: shoots, builds the rig, approves every gate | Same, plus the calibration visit |
| **Claude Code** | Architect, planner, reviewer. Does not implement | Same |
| **OpenCode** | Executor: implements approved plans, writes STATUS.md | Same |
| **Non-author tester** | Someone who has never seen the demo uses it unaided (NFR-09) | — |
| Factory owner | Sponsor. Source of the product fact (моти) | Buyer |
| Operator at the station | — | The user. Acts on the verdict |
| QA / technologist | — | Reads batch traceability |
| Food-safety officer | — | Can veto the install (OQ-6) |
| Student friend | Not involved | Grants, legal, documents |

## 2. Functional requirements

### 2.1 Trigger and capture

| ID | Requirement | Scope | Basis | Check |
|---|---|---|---|---|
| FR-11 | Each operator key press produces exactly one inspection event. Event sequence numbers are gapless within a run. | MVP | invariant I4 · design C5 | T |
| FR-12 | Each event captures exactly one frame from the configured camera (USB webcam or Android phone). A stale or partial frame is never returned; a failed capture ends in NO_VERDICT (FR-24). | MVP | design C4 | T, D |
| FR-13 | Capture applies manual exposure, white balance and focus from the config and records the values the camera reports back. A value that cannot be read back is recorded as `unverified`, never assumed. | MVP | R4 · design 11 §5 | T, D |
| FR-14 | The mean luminance of the fixed reference patch is measured on every frame and recorded. Outside the configured band the result is NO_VERDICT(`lighting`). | MVP | design 10 §3 | T |
| FR-15 | The same pipeline runs from a directory of recorded images and their metadata in place of camera and trigger (replay). | MVP | design 11 §6 | T |
| FR-16 | A photoelectric trigger replaces the key press. | Product | invariant I4 | — |
| FR-17 | A GigE industrial camera (Hikrobot) replaces the USB or phone camera. | Product | invariant I8 | — |

### 2.2 Inspection

| ID | Requirement | Scope | Basis | Check |
|---|---|---|---|---|
| FR-21 | Every event ends in exactly one label: PASS, DEFECT or NO_VERDICT. | MVP | design 11 §2 | T |
| FR-22 | **Appearance check.** A PatchCore anomaly score is compared with the threshold; above it is DEFECT. The verdict carries the score, the threshold applied, `model_id`, `threshold_id` and `config_hash`. | MVP | CLAUDE.md §7 (ADR-0002) | T |
| FR-23 | Every appearance DEFECT carries a heatmap that locates the anomaly. | MVP | build plan S4.3 | T, D |
| FR-24 | Capture failure, lighting fault, placement fault and inference error each end in NO_VERDICT with a stated reason. NO_VERDICT is never shown or recorded as PASS. | MVP | invariant I6 · design C7 | T |
| FR-25 | **Placement.** A frame with no unit, or with the unit not wholly inside the inspection area, ends in NO_VERDICT(`placement`) — not DEFECT. | MVP | new: a stranger at the demo will place units anywhere | T, D |
| FR-26 | **Geometry check.** The unit's silhouette area, equivalent diameter and roundness are compared with tolerances derived from good units; outside them is DEFECT(`geometry`). Pass/fail against a golden reference, in pixels — not millimetres. Shares the segmentation step of FR-25. | MVP (ADR-0008) | fact: client defect list (shape and dimension deviation) · OQ-1 | T, R |
| FR-27 | Millimetre measurement with a stated tolerance. | Product | OQ-1 | — |
| FR-28 | Marking check on the other conveyor. | Product | fact Q21 | — |

Why FR-26 exists: two of the client's four defect types — shape deviation and dimension deviation
— are global properties of the unit. PatchCore compares small patches with small patches; a моти
10% smaller than normal is made of perfectly normal-looking patches, and if preprocessing crops
and resizes each unit to a fixed size, the size information is gone before the model sees it.
Size and roundness have a right answer and are cheap to measure, so they belong in deterministic
code, not in the model.

### 2.3 Operator display

| ID | Requirement | Scope | Basis | Check |
|---|---|---|---|---|
| FR-31 | A live camera view is on screen whenever the station is armed. | MVP | build plan S4.1 | D |
| FR-32 | After each event the verdict is shown as a colour and a word readable at 3 m: PASS green, DEFECT red, NO_VERDICT amber — never green. Words in Russian: **ГОДЕН / БРАК / НЕТ РЕШЕНИЯ**, set in config. | MVP | build plan S4.2 · invariant I6 · language policy (`project.yml`) | T, D |
| FR-33 | A DEFECT shows the heatmap over the unit image, or the failed geometry measurement. | MVP | build plan S4.3 | D |
| FR-34 | A session panel shows counts per label and the last 10 verdicts. | MVP | build plan S4.4 | D |
| FR-35 | A green/red lamp at the station. | Product | design 11 §4 | — |
| FR-36 | Operator override: the operator marks a verdict wrong; stored as a separate row. | Product | design 12 §3 | — |

### 2.4 Evidence

| ID | Requirement | Scope | Basis | Check |
|---|---|---|---|---|
| FR-41 | Every event, NO_VERDICT included, writes one record: image, heatmap if any, label, reason, score, threshold, `model_id`, `threshold_id`, `config_hash`, camera values read back, reference patch, timings, image SHA-256. | MVP | invariant I2 · design 12 §3 | T |
| FR-42 | The application never updates or deletes a record. | MVP | design 12 §3 | T, I |
| FR-43 | A batch ID on every record. In the MVP the session ID stands in for it. | Product (MVP stand-in) | fact: photo archive per batch | T |
| FR-44 | Retention: pass images 400 days; records, defect and override images forever. | Product | design 12 §3 | — |
| FR-45 | Batch traceability report for QA. | Product | design 10 §2 | — |

### 2.5 Dataset, training and evaluation (offline)

| ID | Requirement | Scope | Basis | Check |
|---|---|---|---|---|
| FR-51 | A **shoot mode** saves each frame as `u<unit>_r<rotation>.jpg`, with a per-session `session.yml`: camera values read back, lighting geometry, reference patch reading, product and flavour, purchase date, shoot start and end time. | MVP | design 12 §2 · A-08 | T, I |
| FR-52 | The splitter assigns **units, never images**, to train / calib / test, deterministically from a seed, and refuses to run if any unit falls in two splits. | MVP | design E1 | T |
| FR-53 | One command trains PatchCore on `train/good` only and writes a model bundle whose `model_id` includes a hash of the weights. | MVP | design 14 §4 | T, R |
| FR-54 | One command sets the threshold at a target false-alarm rate on `calib/good` and writes a new `threshold_id`. Geometry tolerances (FR-26) are derived from good units in the same step; the method is fixed in `15-model.md`. | MVP | design 12 §4 · 14 §4 | T |
| FR-55 | One command produces the evaluation report of `14-eval-protocol.md` §7: recall per defect class counted per unit, with lower bounds; false-alarm rate with upper bound; at 1 / 5 / 10% targets; latency p50 / p95 / p99; the invocation and the golden-set hash. MVP reports are stamped **INDICATIVE**. | MVP | design 14 · ADR-0007 | T, R |
| FR-56 | The evaluation set is frozen as a golden set with a content hash; re-running the report on it reproduces every number. | MVP | Gate A5 | T |
| FR-57 | Export to ONNX / OpenVINO for the Intel N100. | Product | CLAUDE.md §8 | — |

### 2.6 Running it

| ID | Requirement | Scope | Basis | Check |
|---|---|---|---|---|
| FR-61 | One command starts the station from a config file; the same command with `--replay <dir>` runs it from disk. | MVP | build plan S4.5 | D |
| FR-62 | The station runs with no network connection and makes no network calls. | MVP | invariant I1 | T, I |
| FR-63 | A signed update bundle, installed by hand at the calibration visit. | Product | design 10 §6 | — |

## 3. Non-functional requirements

| ID | Quality | Requirement and metric | Scope | Check |
|---|---|---|---|---|
| NFR-01 | Responsiveness | Trigger to verdict on screen **≤ 1.0 s at p95** on the demo machine — the RTX 3060 laptop — over ≥ 200 consecutive events. | MVP | R |
| NFR-02 | CPU latency | Trigger to verdict on CPU only, pinned to 4 threads, p50 / p95 / p99 over ≥ 200 events. The laptop CPU is faster per core than an N100, so this is a lower bound on N100 latency. MVP: measured and reported. Gate B: ≤ 700 ms at p95. | MVP (measure) · Gate B (pass) | R |
| NFR-03 | Determinism | The same frame, model and threshold give the same score, equal to 1e-6, on the same machine. | MVP | T |
| NFR-04 | Fault tolerance | 0 crashes over a 200-event run with injected faults — camera unplugged, evidence directory read-only, corrupt frame. Each fault ends in NO_VERDICT and the station stays armed. | MVP | T |
| NFR-05 | Persistence never blocks | With the evidence directory unwritable, verdicts still meet NFR-01; failed writes are counted and shown. | MVP | T |
| NFR-06 | No lost record | Records written equals events triggered, over a 200-event run. | MVP | T |
| NFR-07 | On-prem | 0 connections to any host other than the station's own camera, verified with the internet disabled. No socket or HTTP use in `src/qs` except inside a `FrameSource` implementation. | MVP | T, I |
| NFR-08 | Reproducibility | The evaluation report regenerated from the golden set by one command gives identical numbers. | MVP | T |
| NFR-09 | Usability | A person who has never seen it starts it and checks 5 units without help. | MVP | D |
| NFR-10 | Code quality | `make check` green: pytest 0 failures, ruff 0 errors, mypy 0 errors on `src/`. | MVP | T |
| NFR-11 | Licences | Every dependency and model that could ship has a recorded licence compatible with commercial use. Nothing trained on MVTec AD ships. | MVP (recorded) | I |
| NFR-12 | Environment | Python version and every dependency pinned in a lock file; a fresh clone reaches `make check` green by documented commands. | MVP | D |
| NFR-13 | Sourceability | Product hardware can be re-ordered in Russia — Intel and Hikrobot. | Product | I |

## 4. MVP success metrics — the exit criteria

The MVP is done when all of these hold:

1. Every requirement scoped **MVP** above passes its check.
2. On real моти in the rig: key press → verdict and heatmap on screen.
3. An INDICATIVE evaluation report exists, generated by one command from a frozen golden set.
4. CPU latency is measured and reported (the input to Gate B).
5. The non-author test (NFR-09) has been passed.

Detection numbers are **reported, not required** (ADR-0007). One early-warning line: if
indicative recall on dented units is **below 70% at the 5% false-alarm target**, stop and fix
the lighting before building more of the demo — the Gate A rule, applied early.

## 5. Out of scope

As `05-concept.md` §5: anything at the factory, the Intel N100 and OpenVINO, the Hikrobot camera,
the photoelectric trigger, a second flavour, the marking station, millimetres, foreign objects,
PLC or actuator, networking, user accounts, cloud.

## 6. Assumptions

| ID | Assumption | If wrong | Checked by, when |
|---|---|---|---|
| A-01 | The client's моти is chilled or room temperature, not frozen. | Frost and melting during a shoot; the protocol changes. | The owner, next contact. |
| A-02 | Store-bought моти of one flavour stand in well enough for the client's to prove the method. | The bench result transfers weakly. | S6 retrain on theirs. |
| A-03 | Deliberate damage — pressed dents, tears, squashing — resembles real defects. | Recall looks better at the bench than on the line (E6). | Real defective samples, S5. |
| A-04 | A моти separates cleanly from a matte black background in a top view. | FR-25 and FR-26 need a different background or light. | Lighting pilot, Step 6. |
| A-05 | At least one owned camera holds exposure, white balance and focus locked. | R4: the dataset is quietly ruined. | Camera test, first build day. |
| A-06 | PatchCore fits ~400 ms on CPU at a usable resolution. | Gate B fails; smaller coreset, lower resolution, PaDiM (R5). | S1 / S3 latency. |
| A-07 | The client's defects on моти are dents, tears with filling showing, deformation, size deviation and surface debris. | Wrong defect classes shot at the bench. | The owner, next contact. |
| A-08 | Within one shooting session (≤ 2 h) a unit's appearance does not change enough to matter. | The model learns elapsed time instead of damage (R12). | Control units re-shot at the start and end of each session — Step 6. |

## 7. Dependencies

| ID | Dependency | Needed for | Status |
|---|---|---|---|
| D-01 | The RTX 3060 machine with working CUDA | Training | S0 — unverified |
| D-02 | Anomalib, PyTorch and OpenCV versions compatible with one pinned Python | Everything | S0 |
| D-03 | An owned camera passing the settings-lock test | FR-12, FR-13 | A-05 |
| D-04 | моти bought | The dataset | Budget **OPEN** |
| D-05 | Rig materials: box, matte black paper, desk lamp, tracing paper, a grey card | Shooting | Household, ~0 ₽ |
| D-06 | OpenCode, with a capable model | Executing plans | Installed; model **OPEN** |

## 8. Open

| | Question | Blocks | Status |
|---|---|---|---|
| OQ-1 … OQ-6 | The client questions in `project.yml` | Product scope only | Deferred 2026-09-21 |
| new | Budget ceiling for bench product | D-04, how many units | Open — Ali |
| new | Flavours, sizes, real defect types (A-07) | Product; not the MVP | Open — owner |
| new | Which model OpenCode runs | How much each plan must pin | Open — Ali |

New questions go into `project.yml` at Step 10.

## 9. Gate 1 — requirements review

*Do we know exactly what to build and why?*

- **The MVP: yes.** Every MVP requirement has a basis and a check. FR-26 is in scope (ADR-0008).
- **The product: hold.** OQ-1 … OQ-6 stay open by decision; nothing product-scoped is built.

**Recommendation: GO** on the MVP requirements, with FR-26 in scope.

**Decision: GO** · Ali · 2026-09-21. FR-26 included. The laptop is both the training and the demo machine.
