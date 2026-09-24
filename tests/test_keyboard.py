"""KeyboardTrigger (#88): one SPACE press, one event; the 300 ms rule on the app clock (A2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from conftest import FakeScreen
from qs.config import load_config
from qs.contracts import TriggerEvent
from qs.keyboard import KeyboardTrigger
from qs.screen import KeyPress

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "station-A-bench.example.yml").config


def space(after_s: float, tk_ms: int = 5000) -> KeyPress:
    return KeyPress("space", tk_ms, after_s)


def trigger(screen: FakeScreen, **options: object) -> KeyboardTrigger:
    return KeyboardTrigger(CONFIG, screen, **options)  # type: ignore[arg-type]


def poll_with(t: KeyboardTrigger, screen: FakeScreen, presses: list[KeyPress],
              pump_end: float | None = None) -> TriggerEvent | None:
    """One loop iteration: the pump ends at `pump_end` and delivered `presses`."""
    if pump_end is not None:
        screen.last_pump_end = pump_end
    screen.script.append(list(presses))
    return t.poll()


def first_event(screen: FakeScreen, t: KeyboardTrigger) -> TriggerEvent:
    """Start at 1.00, take event 1 from a press made after it."""
    assert poll_with(t, screen, [], 1.00) is None
    event = poll_with(t, screen, [space(1.00)], 1.05)
    assert event is not None
    return event


# AC1 ------------------------------------------------------------------------------------


def test_one_press_is_one_event() -> None:
    screen = FakeScreen()
    t = trigger(screen)
    assert poll_with(t, screen, [], 1.00) is None
    event = poll_with(t, screen, [space(1.00, 5000)], 1.05)
    assert event is not None
    assert event.seq == 1
    assert event.t0 == 1.00
    assert event.source_id == "keyboard:space"
    assert event.wall.tzinfo is not None
    assert event.wall.utcoffset() == timedelta(0)
    assert t.discarded == 0


# AC2 ------------------------------------------------------------------------------------


def test_a_press_made_before_the_first_poll_is_discarded() -> None:
    screen = FakeScreen()
    t = trigger(screen)
    assert poll_with(t, screen, [space(0.50)], 1.00) is None
    assert t.discarded == 1


# AC3 ------------------------------------------------------------------------------------


def test_presses_within_300_ms_of_the_finish_are_discarded() -> None:
    screen = FakeScreen()
    t = trigger(screen)
    first_event(screen, t)
    assert poll_with(t, screen, [space(2.00)], 2.00) is None
    assert poll_with(t, screen, [space(2.10)], 2.15) is None
    assert poll_with(t, screen, [space(2.299)], 2.32) is None
    assert t.discarded == 3
    event = poll_with(t, screen, [space(2.30)], 2.35)
    assert event is not None
    assert event.seq == 2
    assert event.t0 == 2.30


# AC4 ------------------------------------------------------------------------------------


def test_presses_made_during_the_inspection_are_discarded() -> None:
    screen = FakeScreen()
    t = trigger(screen)
    first_event(screen, t)
    during = [space(1.05, 5100), space(1.05, 5200), space(1.05, 5300)]
    assert poll_with(t, screen, during, 2.00) is None
    assert t.discarded == 3


# AC5 ------------------------------------------------------------------------------------


def test_a_key_held_for_two_seconds_never_floods() -> None:
    screen = FakeScreen()
    t = trigger(screen)
    repeat_s, iteration_s, inspection_s, held_s = 0.033, 0.030, 0.40, 2.0
    press_times = [0.10 + i * repeat_s for i in range(int(held_s / repeat_s))]
    now, previous_pump_end = 0.0, 0.0
    delivered = 0
    events: list[TriggerEvent] = []
    finish_bounds: list[float] = []  # F(k): the pump end of the first poll after event k
    waiting_for_bound = False
    while now < 0.10 + held_s + 1.0:
        batch = [KeyPress("space", round(p * 1000), previous_pump_end)
                 for p in press_times if previous_pump_end < p <= now]
        delivered += len(batch)
        event = poll_with(t, screen, batch, now)
        if waiting_for_bound:
            finish_bounds.append(now)
            waiting_for_bound = False
        previous_pump_end = now
        if event is not None:
            events.append(event)
            waiting_for_bound = True
            now += inspection_s
        now += iteration_s
    assert len(events) >= 2
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert len(events) + t.discarded == delivered == len(press_times)
    for k in range(len(events) - 1):
        assert events[k + 1].t0 >= finish_bounds[k] + 0.300


# AC6 ------------------------------------------------------------------------------------


def test_the_earliest_press_of_a_batch_wins() -> None:
    screen = FakeScreen()
    t = trigger(screen)
    assert poll_with(t, screen, [], 1.00) is None
    batch = [space(1.00, 7010), KeyPress("space", 7000, 1.01), space(1.00, 7020)]
    event = poll_with(t, screen, batch, 1.05)
    assert event is not None
    assert event.t0 == 1.01  # the 7000 press
    assert t.discarded == 2


# AC7 ------------------------------------------------------------------------------------


def test_other_keys_are_ignored_and_not_counted() -> None:
    screen = FakeScreen()
    t = trigger(screen)
    assert poll_with(t, screen, [], 1.00) is None
    others = [KeyPress("a", 5000, 1.00), KeyPress("Escape", 5001, 1.00)]
    assert poll_with(t, screen, others, 1.05) is None
    assert t.discarded == 0


# AC8 ------------------------------------------------------------------------------------


def test_presses_while_not_armed_are_discarded_and_seq_stays_gapless() -> None:
    screen = FakeScreen()
    armed = [False]
    t = trigger(screen, armed=lambda: armed[0])
    assert poll_with(t, screen, [], 1.00) is None
    assert poll_with(t, screen, [space(1.00), space(1.00, 5001)], 1.05) is None
    assert t.discarded == 2
    armed[0] = True
    event = poll_with(t, screen, [space(1.05)], 1.10)
    assert event is not None
    assert event.seq == 1


# AC9 ------------------------------------------------------------------------------------


def test_a_limit_exhausts_the_trigger() -> None:
    screen = FakeScreen()
    t = trigger(screen, limit=3)
    now = 1.0
    assert poll_with(t, screen, [], now) is None
    for seq in (1, 2, 3):
        assert t.exhausted is False
        now += 1.0
        event = poll_with(t, screen, [space(now)], now + 0.05)
        assert event is not None and event.seq == seq
        now += 0.5
        assert poll_with(t, screen, [], now) is None  # the pump after the event
    assert t.exhausted is True
    before = t.discarded
    assert poll_with(t, screen, [space(now + 1.0)], now + 1.05) is None
    assert t.discarded == before + 1


def test_without_a_limit_the_trigger_never_exhausts() -> None:
    screen = FakeScreen()
    t = trigger(screen)
    now = 1.0
    assert poll_with(t, screen, [], now) is None
    for seq in range(1, 51):
        now += 1.0
        event = poll_with(t, screen, [space(now)], now + 0.05)
        assert event is not None and event.seq == seq
        now += 0.5
        assert poll_with(t, screen, [], now) is None
    assert t.exhausted is False


# AC10 -----------------------------------------------------------------------------------


def test_the_given_source_id_is_carried() -> None:
    screen = FakeScreen()
    wall = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)
    t = trigger(screen, source_id="replay:session-01", wall_clock=lambda: wall)
    event = first_event(screen, t)
    assert event.source_id == "replay:session-01"
    assert event.wall == wall
    t.close()
    t.close()
