"""`KeyboardTrigger`: the bench `TriggerSource` (docs/11 §3–§4, #88).

One press of `trigger.key` makes one inspection event, and `seq` has no gaps (FR-11, C5). A
press made during an inspection, within 300 ms of one finishing, or by a held key's
auto-repeat is discarded and counted. It makes no event and no record (F10, docs/16 §7).

The 300 ms rule runs on the application clock (P09 assumption A2). Each press carries
`after_s`, the screen clock when the pump before the one that delivered it ended, which is
the earliest moment it can have been made. At the first `poll()` the bound is the screen's
`last_pump_end`. At the first `poll()` after one that returned an event, it is
`last_pump_end + 0.300`: that pump ran after the event was handled, so the bound is never
less than 300 ms after the event finished. A press is eligible when `after_s ≥ bound`. Of
one poll's eligible presses, the earliest by Tk timestamp becomes the event, and every
other press of the key is discarded. The rule can only ever discard more, never less.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from qs.config import StationConfig
from qs.contracts import TriggerEvent
from qs.screen import ScreenLike

_HOLD_OFF_S = 0.300


def _utc_now() -> datetime:
    return datetime.now(UTC)


class KeyboardTrigger:
    """Polled once per main-loop iteration; never blocks and never sleeps."""

    def __init__(
        self,
        config: StationConfig,
        screen: ScreenLike,
        *,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], datetime] = _utc_now,
        armed: Callable[[], bool] = lambda: True,
        source_id: str | None = None,
        limit: int | None = None,
    ) -> None:
        # `clock` must be the screen's clock: `t0` is a press's `after_s`, read on it.
        self._clock = clock
        self._key = config.trigger.key
        self._screen = screen
        self._wall_clock = wall_clock
        self._armed = armed
        self._source_id = source_id if source_id is not None else f"keyboard:{self._key}"
        self._limit = limit
        self._seq = 0
        self._discarded = 0
        self._not_before: float | None = None
        self._returned_event = False

    @property
    def discarded(self) -> int:
        """Presses of the key that made no event."""
        return self._discarded

    @property
    def exhausted(self) -> bool:
        """True once `seq` reached `limit`; never without a limit."""
        return self._limit is not None and self._seq >= self._limit

    def poll(self) -> TriggerEvent | None:
        """The event for the first eligible press of the key, or `None`."""
        delivered = self._screen.keys()
        presses = [press for press in delivered if press.keysym == self._key]
        if self._not_before is None:
            self._not_before = self._screen.last_pump_end
        elif self._returned_event:
            self._not_before = self._screen.last_pump_end + _HOLD_OFF_S
        self._returned_event = False
        if not self._armed() or self.exhausted:
            self._discarded += len(presses)
            return None
        eligible = [press for press in presses if press.after_s >= self._not_before]
        if not eligible:
            self._discarded += len(presses)
            return None
        chosen = min(eligible, key=lambda press: press.tk_ms)
        self._discarded += len(presses) - 1
        self._seq += 1
        self._returned_event = True
        return TriggerEvent(
            seq=self._seq, t0=chosen.after_s, wall=self._wall_clock(), source_id=self._source_id
        )

    def close(self) -> None:
        """Nothing to release; safe to call twice."""
