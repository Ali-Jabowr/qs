# ADR-0029: The line finds the cup by its rim under our own backlight and judges it by warmth

**Status:** Accepted · 2026-11-10 · Ali. Amends [ADR-0028](ADR-0028-line-unit-cup-software-trigger.md)
§3 and §5; keeps §1, §2, §4, §6 and §7. §3, §4, §5 and §9 amended by
[ADR-0031](ADR-0031-model-first-line-verdict.md). Light rule in [ADR-0030](ADR-0030-own-dc-light-exempts-flicker-rule.md).
Evidence: `reports/2026-11-09-line-findings.md`, `reports/Camera only cup presence detection.md`,
`reports/2026-11-10-cup-finding-debate.md`.

## Context

On the first visit's footage (temporary, angled C920, long auto exposure) the line station was not
usable: the rim finder traced 22 of 153 labelled full cups at their labelled frame, and the empty
check `blob_share` was inverted, because a моти fills its cup and leaves no floor to see.

Measured on the same footage, numbers only:

- The rim finder traces a rim in **94%** of frames while the chain is stopped (2 744 of 2 918
  sampled) and **40%** while it moves (925 of 2 296). Blur, not the method, is the main cause.
- On any frame within ±4 of the labelled one, a rim is found for 101 of 153 full cups.
- Warmth, the mean of R − B, separates every labelled cup: empty −7.9 to −7.1 (4 cups), white моти
  11.3 to 47.3 (43), pink моти 23.5 to 57.8 (112). Lightness does not: the faintest white моти is
  darker than an empty cup.
- A rim-free window cannot tell an empty cup from a gap: 10 of 17 "no моти" flags were right.

Empty cups also pass the old camera spot and are filled later, so an empty cup is a defect only
after the last filling step.

No hardware sensor goes into v1, and counting chain slots from motion was rejected: the station
has to find the cups themselves.

## Decision

1. **A line station is mounted after the last filling step, right before packing.** Upstream of
   it an empty cup is not a defect, and the station is in the wrong place.
2. **The cup is found by today's rim finder, on sharp frames from our own optics.** A shroud with a
   white inside, an opal plastic diffuser with a hole for the lens, and a warm DC LED behind it
   (≤ 3000 K): the polished steel mirrors the lit diffuser, so a clear cup's lip shows as a dark
   ring on a bright floor. Exposure is short enough for ≤ 1 px of motion blur. White balance, focus
   and gain are locked. The lens hole's own reflection is masked.
3. **The finder is made strict:** a rim must be within ±5% of the size measured on the mount, and
   a ring that stays still while the chain moves is a reflection, not a cup. A fixed-radius ring
   matcher, which finds a rim from 60% or more of its arc, is built only if the moving test (T2)
   misses cups.
4. **The verdict is the median warmth over the inner half of the rim, across at least 3 in-zone
   frames** (replaces ADR-0028 §3, "the first frame wholly in the zone"). Fewer frames, or too much
   spread between them, ends in НЕТ РЕШЕНИЯ. The threshold is set for each mount. A median, not a
   maximum, because one frame where something warm leaks in must not turn an empty cup into ГОДЕН.
5. **`blob_share` is deleted** (replaces ADR-0028 §5's measure; the check stays measured, not
   learned). PatchCore keeps running beside it as the appearance check.
6. **A gap gets no verdict**, as before: no cup found, no unit.
7. **Self-check:** a fixed window keeps measuring warmth beside the finder. A моти passing with no
   tracked cup is recorded as НЕТ РЕШЕНИЯ, and the screen shows rolling recall on full cups. A
   matching guard for empty cups is built only if T2 shows empty cups at twice the gap level or
   more in that window.
8. **Advisory mode needs proof first.** At home: 0 missed in at least 300 empty-cup passes, 99% or
   more of full cups found, 0 false cups in at least 100 gaps, and 0 empty cups judged ГОДЕН. On the
   visit, recall is reported with its confidence interval, and any empty cup judged ГОДЕН pauses
   advisory mode.
9. **If T2 fails even with the ring matcher, the photo sensor is brought back to Ali** with that
   evidence, before any trained model is added to a signal the optics never captured.

## Alternatives considered

- **Rim-free warmth window alone.** Catches full cups, but calls every gap an empty cup. Rejected
  as the verdict; kept as the self-check in decision 7.
- **Counting chain slots from motion (a slot clock).** Every slot gets a verdict and nothing
  vanishes, but gaps raise false alarms and the count can drift with stops, hands and dust. Rejected
  by Ali.
- **A hardware sensor (tap the existing one, or our own).** What industry uses; out of v1 by Ali's
  decision. Reopened only under decision 9.
- **Crossed polarisers first.** The cheapest untested lead on paper, but they cost 6–10× the light,
  which fights a 1–2 ms exposure. Kept in the kit as a backup test.
- **A trained cup detector (YOLO-class).** No footage from the fixed camera exists to train it, and
  it is a new dependency. Fallback only.
- **A beam splitter so the lens hole leaves no mark.** Glass over open food, and it halves the light
  twice. Rejected; the spot is masked instead.

## Consequences

- **A missed cup is still silent** (ADR-0028 §4). An operator who reacts only to БРАК treats silence
  as ГОДЕН, so a missed empty cup is a ГОДЕН-direction failure in practice. Decision 8's gate and
  pause exist for that reason.
- **Zero misses in 50 empties proves only about 94% recall** (95% bound). Claiming 99% needs about
  300, so that number can only be shown at home.
- **The mount carries its own light** and must be agreed under OQ-6 (equipment over open food).
- **Config:** `blob_share_min` goes; the rim size, the lens-hole mask and the warmth threshold are
  set per mount.
- **The home rig is not the channel.** Links, side walls and starch differ, so the visit can still
  overturn a home pass.
