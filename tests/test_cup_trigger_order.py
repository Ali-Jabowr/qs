"""The order `CupTrigger.poll` returns events in when one frame makes several (#134): cups first
wholly in the zone, then cups that left never wholly in it, then untracked passages. Each event is
`(never_in_zone, untracked)`.

The pending counts are set directly: no drawn frame makes all three kinds at once.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from qs.config import LineConfig
from qs.cup_trigger import CupTrigger

LINE = LineConfig(
    zone=(20, 30, 240, 180), rim_size_px=(105, 105), blob_share_min=0.3, direction="right", fps=8.0
)
ZONE, LOST, UNTRACKED = (False, False), (True, False), (False, True)

CASES = [
    pytest.param((2, 1, 1), [ZONE, ZONE, LOST, UNTRACKED], id="one of each kind and a second cup"),
    pytest.param((0, 2, 1), [LOST, LOST, UNTRACKED], id="lost cups before the untracked"),
    pytest.param((1, 0, 2), [ZONE, UNTRACKED, UNTRACKED], id="untracked after the cup"),
    pytest.param((0, 0, 2), [UNTRACKED, UNTRACKED], id="untracked alone"),
    pytest.param((1, 1, 0), [ZONE, LOST], id="no untracked: as before"),
]


class _NoFrames:
    def preview(self) -> np.ndarray | None:
        return None


@pytest.mark.parametrize(("pending", "order"), CASES)
def test_events_come_in_zone_then_lost_then_untracked(
    pending: tuple[int, int, int], order: list[tuple[bool, bool]]
) -> None:
    in_zone, lost, untracked = pending
    trigger = CupTrigger(
        LINE, _NoFrames(), source_id="t", clock=lambda: 0.0,  # type: ignore[arg-type]
        wall=lambda: datetime(2026, 11, 10, tzinfo=UTC), ended=lambda: False,
    )
    trigger._add(in_zone, lost)
    trigger._pending += untracked
    trigger._pending_untracked += untracked

    events = [trigger.poll() for _ in order]

    assert [(e.never_in_zone, e.untracked) for e in events if e is not None] == order
    assert [e.seq for e in events if e is not None] == list(range(1, len(order) + 1))
    assert trigger.poll() is None
