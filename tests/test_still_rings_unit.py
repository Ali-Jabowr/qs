"""`StillRings` drives the still-ring filter of the cup trigger (#133). These are the failures of an
earlier version, as literal frame sequences. A frame is the (along, cross) positions of its rims;
the expected value is, per frame, whether each rim is followed. Frames are 0.125 s apart (8 fps),
`still_px` is 3, the across tolerance 25 and the largest step of a cup in one frame 43.

The default is to follow: a rim is left out only when it has been seen at the same place on 3
frames while the chain moved, and a rim not seen for `LOST_AFTER_S` is forgotten.
"""

from __future__ import annotations

from qs.cup_trigger import LOST_AFTER_S
from qs.still_rings import StillRings

DT = 0.125
Frame = list[tuple[float, float]]


def _run(frames: list[Frame]) -> list[list[bool]]:
    rings = StillRings(3.0, 25.0, 43.0, LOST_AFTER_S)
    return [rings.update(points, i * DT) for i, points in enumerate(frames)]


def test_a_moving_cup_found_on_every_other_frame_is_always_followed() -> None:
    frames: list[Frame] = [[(0, 120)], [], [(40, 120)], [], [(80, 120)], [], [(120, 120)]]

    assert _run(frames) == [[True], [], [True], [], [True], [], [True]]


def test_a_cup_found_on_every_other_frame_beside_a_ring_is_always_followed() -> None:
    frames: list[Frame] = [[(60, 190), (0, 120)], [(60, 190)], [(60, 190), (40, 120)],
                           [(60, 190)], [(60, 190), (80, 120)]]

    assert _run(frames) == [[True, True], [True], [True, True], [True], [True, True]]


def test_cups_standing_in_a_stop_are_followed_when_one_misses_a_frame() -> None:
    frames: list[Frame] = [
        [(100, 120), (300, 120)],
        [(100, 120), (300, 120)],
        [(100, 120), (300, 120)],
        [(300, 120)],
        [(100, 120)],            # B missed, A back: B must not match A 200 px behind it
        [(100, 120), (300, 120)],
        [(100, 120), (300, 120)],
        [(100, 120), (300, 120)],
    ]

    assert _run(frames) == [[True, True], [True, True], [True, True], [True], [True],
                            [True, True], [True, True], [True, True]]


def test_a_cup_whose_rim_wobbles_4_px_across_the_travel_is_followed() -> None:
    frames: list[Frame] = [[(60, 190), (20 * i, 120 + (4 if i % 2 else 0))] for i in range(8)]

    assert [flags[1] for flags in _run(frames)] == [True] * 8


def test_a_ring_seen_at_the_same_place_on_3_moving_frames_is_left_out_after_them() -> None:
    frames: list[Frame] = [[(60, 190), (10 + 40 * i, 120)] for i in range(6)]

    assert _run(frames) == [
        [True, True], [True, True], [True, True], [False, True], [False, True], [False, True],
    ]


def test_a_ring_is_followed_while_the_chain_is_stopped() -> None:
    frames: list[Frame] = [[(60, 190), (130, 120)]] * 6

    assert _run(frames) == [[True, True]] * 6


def test_a_left_out_ring_that_drifts_a_pixel_or_two_a_frame_stays_out() -> None:
    frames: list[Frame] = [[(60 + 2 * max(0, i - 4), 190), (10 + 40 * i, 120)] for i in range(9)]

    assert [flags[0] for flags in _run(frames)] == [True, True, True, False, False,
                                                    False, False, False, False]


def test_a_left_out_ring_is_forgotten_after_it_has_not_been_seen_for_a_while() -> None:
    moving: list[Frame] = [[(60, 190), (10 + 40 * i, 120)] for i in range(4)]
    frames: list[Frame] = [*moving, [], [], [], [(60, 190)], [(60, 190)], [(60, 190)]]

    assert _run(frames)[4:] == [[], [], [], [True], [True], [True]]


def test_a_ring_missed_for_one_frame_is_still_the_same_ring() -> None:
    frames: list[Frame] = [
        [(60, 190), (10, 120)], [(60, 190), (50, 120)], [(90, 120)], [(60, 190), (130, 120)],
        [(60, 190), (170, 120)], [(60, 190), (210, 120)],
    ]

    assert _run(frames) == [[True, True], [True, True], [True], [True, True],
                            [False, True], [False, True]]
