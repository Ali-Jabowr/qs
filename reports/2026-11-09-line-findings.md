# Line footage findings — 09.11.2026

INDICATIVE: one visit's footage from a temporary, angled C920 (R13), labelled by Ali. Numbers only;
no image left the machine. Issue: #106.

## What was done

- Ali labelled the first 5 minutes of `moti_line_moscow.mp4` (cut at
  `data/footage/l1/moti_line_moscow_0-5min.mp4`): 159 whole cups (112 pink, 43 white, 4 empty) and
  301 single frames (no hands). Plus `moti_line.mp4`: 2 cups, a staged empty-then-filled clip.
- `qs score-line` ran with a config scaled to 720p (`config/station-line-moscow.yml`).
- The losses were diagnosed, a rim-free check was tried, and it was run over all 29 minutes; Ali
  checked its 17 flags in minutes 5–29.

## What works, what does not

| Part | Result | Verdict |
|---|---|---|
| Labelling tool, `qs score-line` pipeline | Ran end to end on real footage | **Works** |
| Rim finder (cup by its plastic edge) | Rim traced on 23 of 153 full cups at the labelled frame; standing cups yes, moving cups mostly no | **Fails on this footage**: the thin clear edge blurs into the glare |
| Empty check `blob_share` (inside the rim) | Full cups median 0.10, empty 0.26: inverted | **Wrong assumption**: it expects floor visible around the моти, but a моти fills the cup |
| Station as a whole (rim + `blob_share`) | Empty caught 1 of 6 (labels before correction), false БРАК 27 of 153, 48% НЕТ РЕШЕНИЯ | **Not usable** |
| Rim-free "моти present" (pinkness in a fixed 160×160 window on the chain path) | Minutes 0–5: full cups 153 of 155 right; every empty and every gap below the cut | **Works on this footage** |
| Rim-free "empty cup vs no cup" (light in the window) | Minutes 5–29: 10 of 17 flags right (~60%); gaps called empty | **Not reliable**: a clear empty cup looks like bare chain |

## Facts about the line (new, from the footage)

- Line speed while running: ~56 cups/min (beat ~32 frames at 30 fps).
- The line stands still more than half the time: it moves in 23 007 of 52 134 frames (44%).
- Slots on the chain can pass with no cup.
- Empty cups pass this camera's spot and are **filled later** (Ali, from the clips). So an empty cup
  here is not yet a defect; the defect is an empty cup at the last point before packing.
- Empty cups at this spot: 4 of 159 in minutes 0–5; ~15 flagged of ~550 in minutes 5–29 (~60% of
  flags right).

## Before the second visit (local, no client input needed)

1. **Rim-free "моти present" check in `src/qs`**, test-first on synthetic frames: a fixed window
   on the chain path, a pinkness/lightness measure, one verdict per passing cup. Replaces the
   rim finder and `blob_share` on the line (new ADR superseding the relevant part of ADR-0028).
2. **Cup-presence input** in the station contract: a cup event from the photo sensor (read-only),
   with replay support, so "sensor says cup + no моти" = empty cup.
3. **Close L1**: link #106 from `project.yml`, commit the 720p config and this report; open
   tickets #120, #118, #111; #107 (PatchCore on the line) code only, no training on this footage.
4. **Mount prototype** (3D printer available): camera, light and diffuser in one clamp-on piece,
   adjustable height and angle, no drilling. Printed plastic is for the prototype; the final
   material waits on OQ-6, since porous or chipping plastic over open food may be refused.
5. **Home rehearsal**: C920 on the prototype mount, clear cups and store-bought моти moved past
   it; trigger, check, operator screen and recording run together, so the visit is for mounting,
   recording and tuning, not debugging.
6. **Visit kit and checklist**: station records cup-triggered from the fixed camera; settings
   for a short fixed exposure; how to log the sensor; how to label on site.
7. **Client**: OQ-6 (food-safety sign-off for equipment over open product), the visit date,
   where the last filling step is, and the line's dimensions for the mount (channel width, free
   height above the chain, where a clamp can hold).

## On the visit

1. **Camera after the last filling step, right before packing**, fixed overhead, short fixed
   exposure, glare diffuser (L2.1).
2. **Tap the photo sensor** read-only and record its signal with the video (L2.2).
3. **Collect empties on purpose**: ask the operator to send ~50 empty cups through, plus gaps.
4. **Time the line and count**: speed, stops, empties per shift, how empties are fixed today.
5. **Record several hundred good cups** from the fixed camera for PatchCore (L2.3).
6. **Run the station on the line, advisory only**, and label its verdicts on site (L2.4).

## What cannot be done before the visit

- Prove any accuracy number; 4 empties and one temporary camera cannot carry a 95% claim.
- Tune positions, sizes or exposure for the real mount.
- Separate an empty cup from a missing cup by camera alone on this footage.
