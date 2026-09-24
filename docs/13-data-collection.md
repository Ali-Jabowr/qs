# 13 · Data collection — the rig, the pilot, the shoot

**Audience:** Ali at the bench, and whoever re-shoots it for Gate A. English, per the language policy.
**Status:** design, written 2026-09-21 (design Step 6).
**Requirements:** FR-51, FR-52, A-04, A-05, A-08 in [`06-requirements.md`](06-requirements.md).
**Decisions:** ADR-0007 (demo-grade), ADR-0010 (webcam), ADR-0013 (pilot geometries), ADR-0014
(paired shooting). How the images are measured is [`14-eval-protocol.md`](14-eval-protocol.md).

Every rule below exists because the dataset can be ruined quietly, and a ruined dataset still
trains a model and still produces a number. The number is just wrong.

---

## 1. The product

| | Rule | Why |
|---|---|---|
| Flavour | **One**, the palest on the shelf — white, cream or pink | A pale unit on a black floor gives a clean silhouette (A-04) |
| Source | One shop, one batch — same best-before date | No between-batch variation this week |
| When | Bought on the morning of the shoot | Perishable (A-08) |
| Whose | The client's own моти, if it is in retail | Real product, no site visit (A-02) |
| Record | Brand, flavour, shop, purchase time, best-before, storage temperature | Goes into `session.yml` |

**How many.** Units are numbered, never images (`14-eval-protocol.md` §2).

| Use | Standard | Lean — if the budget is short |
|---|---:|---:|
| Lighting pilot (discarded after) | 10 | 6 |
| Control units | 3 | 3 |
| `train/good` | 20 | 18 |
| `calib/good` | 15 | 10 |
| Test — shot good, then damaged | 30 | 24 |
| Spares, for breakage | 5 | 3 |
| **Total** | **83** | **64** |

Test units by damage: standard 12 dent · 9 tear · 9 deformed; lean 10 · 7 · 7.

**Which column: lean — decided by Ali, 2026-09-23 (OQ-8).** The standard column stays for
reference. Both columns keep every rule below; lean widens the bounds in `14-eval-protocol.md` §9.

## 2. Two things that make fake defects

- **Fingerprints.** A touch leaves a mark in the starch that the model will call an anomaly —
  correctly. Wear food gloves. Move units with a spatula. Never touch the top surface.
- **Condensation.** A chilled unit brought into a warm room fogs with moisture for the first
  10–20 minutes, and fog looks like a defect. Let every unit reach room temperature for 30 minutes,
  **still in its packaging** so it does not dry out, before the first frame.

## 3. The rig

Household parts, nothing bought (Gate C).

| Part | How | Why |
|---|---|---|
| Box | Cardboard, about 40 × 30 × 30 cm | The shroud |
| Inside | Matte black paper or card on floor and walls. Not felt — it sheds fibres | Ambient light out; black background for the silhouette |
| Camera | USB webcam through a hole in the lid, pointing straight down. Height marked | Pixel tolerances hold only at one height (ADR-0008) |
| Height | The lowest at which the webcam focuses sharply on a unit's top, with the placement area and grey card in view | Most pixels on the unit |
| Grey card | Mid-grey matte patch in a corner of the floor, inside the view, outside the placement area | Lighting drift on every frame (FR-14) |
| Placement marks | Four small dots **outside** the inspection area | A mark inside it is a white line on black — the Localizer would find it |
| Lamps | Two identical LED desk lamps, if the house has them | G2 needs two (§5). No purchase approved (2026-09-21): if the house has only one, the pilot compares G1 and G3 |
| Front | A black cloth over the opening while shooting | Room light out |

**Exposure is a multiple of 10 ms.** Mains lamps in Russia flicker at 100 Hz (twice the 50 Hz
supply). At an exposure that is not a whole number of flicker periods, frames come out in bands
of varying brightness. Set 10 ms or 20 ms, set the camera's power-line frequency to 50 Hz, and run
LED lamps at full brightness — a dimmed LED flickers by PWM at a frequency of its own.

**After the pilot, the camera and lamps do not move.** Every tolerance and threshold is tied to
this geometry; the config version records it.

## 4. Day zero — the camera lock test

ADR-0010. The rig closed, the webcam in place:

1. Set exposure, **gain**, white balance and focus to manual from the config; read each back. Gain
   left on automatic stays wherever auto-exposure last put it, and differs between sessions.
2. Capture 50 frames over 5 minutes, switching the room light on and off twice.
3. **Pass:** every read-back equals the value set, the grey card reading varies by less than 1%,
   and no frame shows horizontal bands. Read-back alone is not enough — the driver can report a
   value the sensor is not using; the grey card is what proves it.

Pass: ADR-0010 becomes Accepted. Fail: the phone becomes the source, recorded as `unverified`.

## 5. Session 1 — the lighting pilot

ADR-0013. One frame must serve both checks — the silhouette for geometry and the surface for
appearance — so the three candidates are:

| | Geometry | How |
|---|---|---|
| G1 | Low-angle, one side | One lamp 20–30° above the floor, from the left |
| G2 | Low-angle, two sides | Two lamps at the same angle, opposite each other |
| G3 | Diffuse | The lamp aimed at white card or through a tracing-paper tent |

**Units:** 10 — 6 kept good, 2 to dent, 2 to tear (lean: 4 · 1 · 1), numbered `p01`…

**Order:** all 10 units good under G1, then G2, then G3, three rotations each. Then damage 4, and
shoot those 4 under G1, G2, G3. The lamps move three times, not thirty.

**Choosing, in this order:**

1. **Silhouette.** On every good unit, the area measured by the Localizer varies less than 2%
   between rotations. A unit does not change size when turned; shadows do. A geometry that fails
   here fails the geometry check and is out.
2. **Separation.** A quick PatchCore per geometry, **leave one out**: trained on every good pilot
   unit but one, scoring that one and the damaged units, repeated for each good unit — so the lean
   pilot works too. The winner has the widest gap between the lowest damaged score and the highest
   held-out good score, relative to the spread of the held-out good scores.
3. **The sentence.** Which lighting shows a dent best, and why — written down (S2.10).

A tie goes to G2: two-sided light gives the steadier silhouette. With only one lamp in the house,
the pilot runs G1 and G3 and the tie goes to G3. **The pilot units are then
thrown away.** They helped make a decision, so they can never help measure it.

## 6. Session 2 — the MVP dataset

ADR-0014. One session, under two hours (A-08).

**Before the first frame**

- **A backup drive exists** — an external drive or USB stick, 32 GB is plenty (PNG frames run
  ~3 MB; the MVP shoot is ~1.5 GB). **No purchase approved (Ali, 2026-09-21).** Until a stick exists,
  the backup is the **Android phone over USB** — a second device, no network involved.
- The splitter assigns unit numbers to splits by seed **before** shooting. Every unit's split is
  known in advance, so the shooting order can be random.
- Each unit sits in a numbered cell of a tray — an egg box, a muffin tray. Units cannot be labelled;
  cells can.
- Gloves on. Units at room temperature for 30 minutes.
- Grey card reading recorded.

**The shoot**

Every frame is named by the rule in `12-data-model.md` §2 — the shoot tool writes the names.

1. **Controls first.** The 3 control units `c01`–`c03`, 3 rotations each (`_pass1`).
2. **Then every other unit in the random order the shoot tool prints.** 3 rotations — 0°, 120°, 240°,
   by eye.
3. **A test unit is shot good, then damaged immediately.** Take it out, damage it on a board outside
   the rig by its recipe (§7), put it back, shoot 3 rotations again. Its good and damaged images
   are minutes apart, not hours.
4. **Controls again** at the middle and the end (`_pass2`, `_pass3`).
5. **Placement frames**, last: 5 with no unit in view (`x01`–`x05`, state `empty`) and 5 with a
   unit touching the edge of the inspection area (`x06`–`x10`, state `edge`).
6. **Brush the floor** every 10 units. Starch dust is masked out of the anomaly map (ADR-0009), but
   a drift of it near the unit disturbs the silhouette.

**After**

- `raw/<session>/` is copied untouched and never edited again — **and copied to the backup drive
  before anything else happens** (`17-security.md` T11).
- The splitter builds `datasets/mochi-<flavour>/<geometry>/`; it refuses to run if any unit falls
  in two splits.
- The evaluation set is frozen as `golden/mvp-<date>/`, with a content hash (FR-56).
- The control units' scores, start against end, go into the report. A shift larger than the gap
  between good and dented units means the session drifted, and the numbers mean nothing (E9).

## 7. Damage recipes

Standard, so they can be repeated at Gate A and compared.

| Class | Recipe | Caught mainly by |
|---|---|---|
| `dent` | A rounded tip about 10 mm across — the end of a marker cap — pressed straight down about 3 mm, once, off-centre on the top | Appearance |
| `tear` | A clean knife slit 5–10 mm through the skin on the top or upper side, opened slightly so the filling shows | Appearance |
| `deformed` | Flattened: pressed from above with a flat plate until about a third of the height is gone. Or oval: squeezed from two sides. Roughly half each, recorded per unit | Geometry |

Not shot in the MVP: **size**. A normal unit cannot be made smaller without deforming it. If a
smaller моти of the same flavour is on sale, 5 of them make a diagnostic class — reported, not
gated. **Foreign objects**: out of scope (`05-concept.md` §5).

## 8. `session.yml`

One per session, written by the shoot tool, completed by hand where marked:

```yaml
session_id: 2026-09-23-mvp
kind: dataset                  # pilot | dataset | gate_a
geometries: [G2]               # the pilot lists all three: [G1, G2, G3]
product: { brand: "…", flavour: "…", shop: "…", best_before: 2026-09-26 }   # by hand; brand → client_product: true if theirs (17 §5)
purchased_at: 2026-09-23T09:10+03:00                                       # by hand
started_at: 2026-09-23T11:02+03:00
ended_at:   2026-09-23T12:41+03:00
camera: { device: /dev/video0, exposure_us: 10000, gain: 0, wb_kelvin: 4500, focus: 30, verified: true }
reference_patch: { start: 0.621, end: 0.618 }
split_seed: 20260923
units: { control: [c01, c02, c03], train: […], calib: […], test: […], placement: [x01 … x10] }
damage: { u031: dent, u044: deformed-oval, … }
notes: ""                      # by hand — anything odd
```

## 9. Gate A

The same procedure, larger: 30 train, 30 calib, 70 test shot good then damaged (30 dent · 20 tear
· 20 deformed), 3 controls — 133 units, bought fresh, in the geometry the pilot chose.
`14-eval-protocol.md` §3 gives the reasons for each number.
