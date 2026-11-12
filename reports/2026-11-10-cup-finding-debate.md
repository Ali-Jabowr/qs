# Finding cups on the line — design debate, 10.11.2026

Two Claude engineers (Opus 5.5) debated how to find clear cups and judge empty vs full with a camera
only, after Ali ruled out a hardware sensor and a slot clock. This is the joint recommendation. It
is **not decided** until Ali approves it. Debate rounds: session scratchpad `debate-r1.md`,
`debate-r2.md`. Research behind it: `reports/Camera only cup presence detection.md`.

## The evidence that settled the direction

T0, run on the visit footage (numbers only, no image viewed; rim size range 60–700 px):

| | Frames sampled | Rim traced |
|---|---|---|
| Chain stopped (29 min, every 10th frame) | 2 918 | **2 744 (94%)** |
| Chain moving | 2 296 | **925 (40%)** |
| Labelled full cups, at the labelled frame | 153 | 22 |
| Labelled full cups, any frame within ±4 | 153 | **101 (66%)** |
| Labelled empty cups, any frame within ±4 | 4 | 4 |

- **Motion blur is the main reason the rim finder failed**, not the method. Sharp frames → found.
- **Using several frames per cup** already lifts moving cups from 22 to 101 of 153.
- Caveat: "rim traced" may include моти outlines taken for rims (risk #119), so T0 says almost
  nothing about **empty** cups. Only the home tests can.

## The plan

**Optics (L2.7): make the polished steel a backlight.**
- A shroud with a white matte inside, a plastic diffuser (acrylic or PC, no glass) with a ~20 mm hole
  for the lens, and a warm DC LED (≤3000 K) behind it. The flat steel mirrors the bright diffuser,
  so the clear cup lip shows as a **dark ring** on a bright floor: the textbook backlight for clear
  plastic, delivered by the mirror with nothing under the chain.
- Exposure short enough for ≤1 px blur (expected 1–2 ms). White balance, focus and gain locked.
- The lens hole mirrors back as a ~10 mm dark spot at a fixed pixel: mask it, keep it off the zone
  centre.
- **New ADR:** our own DC light in a shroud exempts the "exposure must be a multiple of 10 ms" rule
  (`src/qs/config.py:604`), as long as room light is ≤10% of the image.
- Polariser film: not in the first test (it costs 6–10× the light); kept for a 10-minute check if
  the backlight fails.

**Finding the cup (L2.6, replaces the sensor task):** keep today's rim finder and tracker.
- Size gate ±5% of the rim measured on the mount.
- Reject rings that stay still while the chain moves (our own light's reflections).
- A fixed-radius ring matcher (finds a rim from ≥60% of its arc) **only if T2 misses cups**.

**Judging (L2.5):** median warmth (R−B) over the inner 50% of the rim, lens spot masked, over ≥3
in-zone frames. Too few frames or too much spread → НЕТ РЕШЕНИЯ. Threshold per mount. Delete
`blob_share`. Amends ADR-0028 §3 (was: first frame wholly in the zone).

**Self-checks:**
- Fixed-window warmth beside the finder: a моти passes but no cup was tracked → НЕТ РЕШЕНИЯ, and
  the screen shows rolling full-cup recall. It counts моти, not time.
- An edge-energy guard for **empty** cups, built only if T2 shows empties at ≥2× the gap level.

## Tests (L2.8), in order

| Test | Setup | Pass |
|---|---|---|
| T1 static, one evening | Clean and starch-dusted steel, U-channel mock-up; empty, white, pink cups and a gap | Room light ≤10%; empty-cup lip edge ≥5× noise on ≥60% of the rim; white моти vs empty warmth margin ≥10; lens spot ≤15 mm |
| T2 moving | Tray at ≥20 cm/s | **0 misses in ≥300 empty passes**; ≥99% of full cups found; 0 false cups in ≥100 gaps; 0 empty → ГОДЕН |
| T3 visit | ≥50 staged empties plus gaps, self-checks on | Recall reported with its confidence interval; advisory mode only after T2 passed |

Why 300: zero misses in 50 empties proves only ~94% recall (95% bound); ≥99% needs ~300.

## Where the two engineers disagreed, and how it was settled

| Point | Settled as | Why |
|---|---|---|
| "Blur is the cause" | Confirmed by T0 | 94% vs 40% |
| Polariser first vs backlight first | Backlight first | Light budget: polariser costs 6–10× at a 1–2 ms exposure |
| Max vs median warmth | Median, inner 50% | Max picks the one frame where something warm leaks in, failing toward ГОДЕН |
| 99% gate on ~50 empties | 300 at home, CI on the visit | Statistics |
| Beam splitter for the lens hole | No: mask instead | Glass over open food, and it halves the light twice |
| Live guard for empties | Conditional on T2 data | Cheap if the signal is there; pointless if not |
| Any empty → ГОДЕН on the visit | Pauses advisory mode | A missed empty is the one failure the station exists to prevent |
| If T2 fails even with the ring matcher | Bring the photo sensor back to Ali | Industry solves this with a sensor; more model capacity on a signal the optics never captured will not |

## Risks still open

- A tray at home is not the channel: links, side walls and starch differ. Hence T3 and the advisory pause.
- White моти warmth under our own warm LED is unmeasured (T1).
- The channel profile and link finish are unknown: ask the client.
- Purchases need Ali's approval: warm DC LED, acrylic diffuser, polariser film (kept in the kit).
