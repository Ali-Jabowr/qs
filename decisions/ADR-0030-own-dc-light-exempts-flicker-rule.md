# ADR-0030: A line station lit by our own DC light may use an exposure shorter than 10 ms

**Status:** Accepted · 2026-11-10 · Ali. Goes with [ADR-0029](ADR-0029-cup-rim-under-own-backlight-warmth-verdict.md).

## Context

`src/qs/config.py` refuses any exposure that is not a multiple of 10 ms when `power_line_hz: 50`,
so that lamps flickering at 100 Hz light every frame the same. That rule is right under room
light. On the line it forbids the sharp frames ADR-0029 needs: the chain moves about 20 cm/s, so a
10 ms exposure smears about 2 mm, and the rim finder fails on moving cups for that reason. Under
flickering room light a 1 ms exposure would also change brightness and warmth from frame to frame.

## Decision

A line station whose own light is a DC-driven LED, with no PWM dimmer, inside a shroud may use an
exposure that is not a multiple of 10 ms. It must show that room light makes up **10% or less** of
the image: a frame with our light off, compared with one with it on, at the same settings. The
config states this light explicitly; without it, the 10 ms rule stays. The bench keeps the rule.

## Consequences

- The light-off / light-on check is part of setting up each mount and of the home test T1.
- A PWM dimmer or a mains LED driver with ripple brings the flicker back; the light's driver is
  part of the station's hardware list.
- The station cannot switch its light off by itself, so it cannot re-check the 10% while running.
  A changed shroud, a new lamp nearby or daylight on the line means the check is done again.
