# 18 · Shoot and train: the step-by-step guide

**For:** Ali, at the bench, with the phone. Follow it top to bottom and tick each box.
**Written:** 2026-09-23. **Size:** lean (OQ-8, decided 2026-09-23). **Camera:** the phone (ADR-0022).
**The why behind every rule:** [`13-data-collection.md`](13-data-collection.md). This file is the *how*.

> **The one idea to hold on to.** The model learns only what a **good** моти looks like. It is
> never shown a defect while it learns. The damaged units are there to *test* it: we show them to
> the trained model and see whether it notices. So the photos of good units have to be clean and
> consistent. Anything odd in them (a fingerprint, fog, a shadow that moves) becomes part of what
> the model thinks "good" looks like.

---

## The timeline

| When | What | Needs code? |
|---|---|---|
| **Today, Wed 23.09** | Part A (the phone), Part B (the rig), Part C (the day-zero test) | Ready now |
| **Thu 24.09** | Part D: the **lighting pilot**, 6 моти | Ready now |
| Thu–Fri | Claude builds P06 (#39–#41) and P07 up to `qs pilot-report` (#45–#47) | Being built |
| **After that** | Part E: read the pilot report and **choose the lighting** | #47 |
| **Next morning (Fri 25.09 at the earliest)** | Part F: buy fresh моти, **the dataset shoot** | Ready now |
| **Right after the shoot** | Part G: build the datasets, **train**, calibrate, evaluate | #46, #80–#83 |

**When does the shooting start?** The day-zero test today, the pilot tomorrow. **When can the
model be trained?** Minutes after the dataset shoot, once `qs train` (#46) and `qs build` (#80)
exist. Training PatchCore on 54 photos takes minutes on the laptop's GPU; the waiting is for the
code. These dates hold only if Claude usage allows the build to continue on Thursday.

---

## Part A: set up the phone (once, about 20 minutes)

**Use the Xiaomi, not the iPhone.** The stock Xiaomi camera has a **Pro** mode that locks
shutter, ISO, white balance and focus, and it saves JPEG. The stock iPhone camera can't lock
shutter or ISO, saves HEIC by default, and always applies its own processing (Smart HDR, Deep
Fusion). Getting around that needs a paid app. Whichever phone you start with, keep it for the
whole project: one camera per dataset (ADR-0022).

- [ ] **A1. Cut it off from the cloud.**
  - Gallery → Settings → **Cloud sync off** (Xiaomi Cloud).
  - Google Photos → **Backup off**.
  - Turn off auto-save in Telegram and WhatsApp.
  - While shooting, put the phone in **airplane mode**.
  - Why: no image leaves the building (NFR-07). It stays off for as long as any QS photo is on
    the phone.
- [ ] **A2. Open Camera → ⋮ / Settings** and set:
  - **Watermark: off.** Xiaomi stamps "Shot on Xiaomi" into the picture, and the model would learn
    that too.
  - **AI camera / AI scene: off.**
  - **Save location info: off.**
  - **Picture quality: High.**
  - **HEIF/HEIC: off**, if the option exists. We need JPEG.
  - **Aspect ratio: 4:3**, and **not** the 50/108/200 MP "Ultra HD" mode.
- [ ] **A3. Switch to Pro mode.** Then:
  - **Lens: the main 1× lens.** Not the ultra-wide, no zoom.
  - **S (shutter): 1/100 s.** If it's too dark at the ISO below, use 1/50 s. Nothing else: the
    lamps flicker at 100 Hz, and any other shutter speed puts dark stripes across the photo.
  - **ISO: fixed**, the lowest value that gives a clearly lit photo. Try 100, 200, 400. Never "Auto".
  - **WB: a fixed setting.** A preset such as "Daylight", or a K value such as 4500 K. Never "Auto".
  - **Focus: MF (manual).** Set it on the top of a моти in the rig (step B5).
  - **RAW: off.** JPEG only (the default, #51).
- [ ] **A4. Write down** the phone's model, and the shutter, ISO, WB and focus you chose. You'll
      need them for the config (C1).

**Don't:** touch the screen to focus, pinch to zoom, switch lenses, or leave Pro mode during a
session. Any change makes the import refuse the session.

---

## Part B: build the rig (about 40 minutes)

- [ ] **B1. The box.** Cardboard, about 40 × 30 × 30 cm. Line the floor and walls with **matte
      black paper or card**, not felt (felt sheds fibres).
- [ ] **B2. A hole in the lid** for the phone's main lens, pointing straight down. The phone lies
      flat on the lid and is **taped down**. Mark its outline with a pen so it can go back to the
      exact spot.
- [ ] **B3. The grey card.** A mid-grey matte patch (grey card, or matte grey paper) in **one
      corner** of the floor. It must be in the photo but away from where the моти sits.
- [ ] **B4. Placement marks.** Four small dots **outside** the area where the моти sits. Never
      inside it: a mark there looks like part of the моти.
- [ ] **B5. The height.** As low as possible while the phone still focuses sharply on the top of a
      моти **and** the grey card and the whole placement area are in the photo. Most pixels on the
      моти is the goal.
- [ ] **B6. The lamps.** LED desk lamps at **full brightness**; a dimmed LED flickers. **Count
      them now:** two identical lamps means the pilot compares G1, G2 and G3. With one lamp it
      compares G1 and G3.
- [ ] **B7. A black cloth** over the front opening while shooting.

**After the pilot, the phone and lamps do not move again** until the dataset shoot is done.

---

## Part C: the day-zero test (today, 15 minutes)

It proves the phone really holds its settings. It replaces the webcam lock test (#25).

- [ ] **C1. Make the phone config.** In the repo, copy the bench config:
      `cp config/station-A-bench.yml config/station-A-phone.yml`, then edit it:
  - `camera.type: phone`
  - `camera.device: "phone:Xiaomi <your model>"`
  - `camera.width` and `camera.height`: the photo's pixel size. Gallery → a photo → Details shows
    it, e.g. 4080 × 3072.
  - `camera.exposure_us`: `10000` for 1/100 s, `20000` for 1/50 s
  - `camera.gain`: your ISO
  - `lighting.reference_patch.roi` and `inspector.localize.roi`: see C3
- [ ] **C2. Take 10 photos over 2 minutes.** Rig closed, cloth down, one моти in place. While you
      shoot, switch the **room** light off and on twice.
- [ ] **C3. Fit the two boxes** (`[x, y, width, height]` in pixels) on one of these photos.
      Open it in an image viewer that shows the pointer's pixel position. GIMP shows it at the
      bottom left.
  - `reference_patch.roi`: a box **inside** the grey card only.
  - `localize.roi`: the placement area, **not** overlapping the grey card.
- [ ] **C4. Copy the photos to the laptop over the USB cable.** On the phone choose "File
      transfer", then copy `DCIM/Camera` to, e.g., `~/qs-phone/day0/`. Never by messenger, e-mail
      or cloud.
- [ ] **C5. Run the check:**
      ```
      bin/qs import-phone --config config/station-A-phone.yml --from ~/qs-phone/day0 --check
      ```
  - **Pass:** every photo has the same exposure, ISO, WB mode and focal length, and the grey card
    varies by less than 1%.
  - **Fail:** it names the photo and the field. Fix that setting on the phone and repeat C2–C5.

---

## Part D: the lighting pilot (tomorrow, about 1 hour)

It decides which lighting the dataset will use. **Lean: 6 моти**, numbered `p01`–`p06`.
Of these, 4 stay good, 1 gets a dent and 1 gets a tear. The pilot units are **thrown away**
afterwards and never used again.

- [ ] **D1. Buy 6 моти.** One flavour, the palest on the shelf. One shop, one best-before date.
      Keep the receipt details: brand, shop, time, best-before.
- [ ] **D2. Let them reach room temperature: 30 minutes, still in the packaging.** A cold моти
      fogs up, and fog looks like a defect.
- [ ] **D3. Plan the session and print the list:**
      ```
      bin/qs split-plan --session 2026-09-24-pilot --kind pilot --size lean          # two lamps
      bin/qs split-plan --session 2026-09-24-pilot --kind pilot --size lean --geometries G1,G3   # one lamp
      bin/qs split-plan --session 2026-09-24-pilot --shot-list
      ```
      Print the `shots-*.md` it writes, or keep it on the laptop screen. **One line = one photo, in
      order.** The bold lines tell you when to move the lamps and when to damage a unit.
- [ ] **D4. Put on food gloves. Move units with a spatula, and never touch the top.** Every
      fingerprint is a "defect" in the starch.
- [ ] **D5. Shoot, line by line.** Tick each line on the paper as you take it.
  - **G1:** one lamp from the left, 20–30° above the floor.
  - **G2:** two lamps, same angle, from opposite sides.
  - **G3:** the lamp aimed at white card, or through tracing paper, for soft light.
  - **Rotations:** 0°, 120°, 240°, turned by eye with the spatula.
  - **Damage** exactly by the recipe on the list. **Dent:** a marker-cap end pressed about 3 mm
    straight down, off-centre. **Tear:** a knife slit of 5–10 mm, opened slightly.
- [ ] **D6. A spoiled photo** (blurred, hand in view, unit moved): **don't delete it on the phone.**
      Retake it immediately and write "redo" next to that line on the paper. The import sorts it
      out (`--redo K`).
- [ ] **D7. A unit breaks:** write down after how many of its photos it broke (`--broken p03:2`).
      In the pilot it is simply dropped.
- [ ] **D8. Copy to the laptop over USB** (`~/qs-phone/pilot/`), then import:
      ```
      bin/qs import-phone --config config/station-A-phone.yml --from ~/qs-phone/pilot \
        --session 2026-09-24-pilot --flavour <flavour> --shop "<shop>" --brand "<brand>" \
        --best-before 2026-09-27 --purchased-at 2026-09-24T09:10+03:00 [--redo K] [--broken pNN:N]
      ```
      If the brand is the client's own моти, write `--client-product` instead of `--brand`.
- [ ] **D9. Back up** the new `raw/2026-09-24-pilot/` folder to the phone over USB (the backup
      copy), **before anything else**.

---

## Part E: choose the lighting (when #47 is built)

- [ ] **E1.** Run `qs pilot-report` on the pilot session:
      ```
      bin/qs pilot-report --config config/station-A-phone.yml --session data/raw/2026-09-24-pilot
      ```
      The report lands in `reports/pilot/2026-09-24-pilot.md`, with the threshold histogram
      beside it as `…-histogram.csv`. Exit 0 means a lighting is recommended; exit 1 means none
      passed (the report then names the two-frame fallback, ADR-0013). It also prints a
      suggested window size (`window_px`): write it down for G2. It gives two numbers per lighting:
  - **Silhouette:** does the measured area of a good unit stay within 2% when it's turned? A
    lighting that fails this is **out**, because its shadows would fool the size check.
  - **Separation:** how far apart the damaged units' scores are from the good units' scores.
    Wider is better.
- [ ] **E2. Pick the winner** by those two, in that order. A tie goes to G2, or G3 with one lamp.
      Write one sentence: which lighting shows a dent best, and why.
- [ ] **E3.** Set `lighting.geometry` in `config/station-A-phone.yml` to the winner and commit it.
      **From now on the phone and lamps don't move.**

---

## Part F: the dataset shoot (next morning, under 2 hours)

**Lean: 58 fresh моти.**
- 3 control units
- 18 for training (good only)
- 10 for calibration (good only)
- 24 for testing, each shot good and then damaged: 10 dent, 7 tear, 7 deformed
- 3 spares

- [ ] **F1. Buy 58 моти that morning.** Same flavour, one shop, one best-before date. 30 minutes
      at room temperature in the packaging.
- [ ] **F2. An egg box or muffin tray with numbered cells.** You can't label a моти, but you can
      label a cell.
- [ ] **F3. Plan and print:**
      ```
      bin/qs split-plan --session 2026-09-25-mvp --size lean
      bin/qs split-plan --session 2026-09-25-mvp --shot-list --geometry <winner>
      ```
      The splitter decides by seed which unit goes where, **before** shooting, and prints a random
      order.
- [ ] **F4. Mid-shoot check,** as often as you like. Copy the photos so far and run:
      `bin/qs import-phone --config config/station-A-phone.yml --from <dir> --session 2026-09-25-mvp --check`
      It tells you where you are on the list and whether the lock still holds.
- [ ] **F5. Shoot in the printed order:**
  1. **Controls first:** `c01`–`c03`, 3 rotations.
  2. **Every other unit** in the printed order, 3 rotations each.
  3. **A test unit:** shot good, then taken out, damaged on a board **outside** the rig by its
     recipe, put back and shot again straight away. Minutes apart, not hours.
     **Deformed:** press with a flat plate until about a third of the height is gone, or squeeze
     it oval from two sides.
  4. **Controls again** in the middle and at the end, where the list says.
  5. **Brush the floor** every 10 units. The list reminds you.
  6. **Placement photos last:** 5 with no unit, and 5 with a unit touching the edge of the area.
- [ ] **F6.** Handle spoiled photos and broken units as in D6 and D7. A broken unit's place is
      taken by the next spare (`--broken u012:2`).
- [ ] **F7. Copy over USB, import** (the same command as D8, with the new session), and **back up
      at once.**

**Don't:** reuse pilot units, mix phones, shoot on two different days, move the lamps, or edit
anything in `raw/` by hand. `raw/` is copied once and never touched again.

---

## Part G: build, train, calibrate, evaluate (after the shoot)

Run these in order, from the repo root. `<session>` is the folder `qs import-phone` wrote under
`data/raw/`. Each command refuses with a one-line reason rather than writing half a result.

- [ ] **G1. Build the datasets and freeze the test set** with `qs build` (#80):
      ```
      bin/qs build --session data/raw/<session> --product mochi
      ```
      It sorts the photos into `data/datasets/mochi/<geometry>/` (`train`, `calib`, `test`) by
      the plan, and freezes the test set under `data/golden/mvp-<date>/` with a content hash. It
      prints the counts and the **golden hash: write the hash down beside the day's notes** before
      training. **Nothing in a frozen set is ever changed or re-shot.** A report built on an
      unfrozen set is worthless. A second run on the same session says "already built" and
      changes nothing.
- [ ] **G2. Train** with `qs train` (#46):
      ```
      bin/qs train --config config/station-A-phone.yml --dataset data/datasets/mochi/<winner> \
        --window-px <N from E1> --product mochi
      ```
      `<winner>` is the geometry from E3, e.g. `G2`. The defaults are `--seed 42`,
      `--coreset 0.1` and `--input-px 256`; leave them. PatchCore reads the **18 training units × 3 rotations
      = 54 good photos**. From each photo it keeps a "memory" of small patches of what good
      starch looks like. It runs in minutes on the laptop's GPU. The output is a **model bundle**
      (the weights plus a record of exactly what they were trained on). **Note the `model_id` it
      prints**; G3 needs it.
- [ ] **G3. Calibrate** with `qs calibrate` (#81). On the 10 calibration units, which the model
      never saw, it sets:
  - the anomaly **threshold**: how unusual a patch must be to count as a defect
  - the **size and shape tolerances**: how far area and roundness may vary on a good unit
      ```
      bin/qs calibrate --config config/station-A-phone.yml \
        --dataset data/datasets/mochi/<winner> --model <model_id> \
        --out config/station-A-phone-cal.yml
      ```
      It writes a **new** config (`--out` must not exist; the input is only read), with a new
      `threshold_id`, and a calibration record under `reports/calib/`.
      **If it refuses because geometry alone flags too many units** (#78: likely with 10 units,
      because 5% allows none), run it again with `--far 10` and a new `--out` name. The report
      states which target it was set at. Nothing widens the tolerances automatically.
- [ ] **G4. Evaluate** with `qs evaluate` (#82, #83):
      ```
      bin/qs evaluate --config config/station-A-phone-cal.yml --golden data/golden/mvp-<date>
      ```
      It runs every test photo (good and damaged) through the trained model and writes the
      **INDICATIVE report** and its CSV under `reports/eval/`: how many damaged units it caught,
      how many good units it wrongly flagged, the paired good-versus-damaged differences, the
      placement check, the latency, and whether the controls drifted during the session.
      INDICATIVE means demo-grade, not the 95% acceptance claim. That claim comes at Gate A on
      fresh units (ADR-0007). **Exit 1 with "the session drifted" at the top** means the numbers
      mean nothing (see below).
- [ ] **G5. The demo on screen** (P09, until the webcam arrives, #85):
      ```
      bin/qs station --config config/station-A-phone-cal.yml \
        --replay data/golden/mvp-<date>/frames --screen
      ```
      Each SPACE press shows the next photo's verdict: ГОДЕН green, БРАК red, НЕТ РЕШЕНИЯ
      amber. Closing the window is a clean stop, and the run report goes to `reports/station/`.
      Its latency is indicative only: NFR-01 is measured live, on the webcam.
- [ ] **G6. Timing for Gate B** (S3.5) with `qs latency` (#50), on the trained bundle:
      ```
      bin/qs latency --config config/station-A-phone-cal.yml \
        --frames data/golden/mvp-<date>/frames --cpu
      ```
      On the CPU it prints the Gate B line: the whole inspection's p95 × 3 against 400 ms.

**How to tell it worked:**
- The dents and tears score clearly above the good units.
- The deformed ones fail the shape check.
- The control units score the same at the start and at the end of the session.

If the controls drifted by more than the gap between good and dented, the session drifted and
the numbers mean nothing. Then the shoot is redone.
