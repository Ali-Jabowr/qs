"""The operator screen states the `untracked` reason in words, not as ГОДЕН (#134)."""

from __future__ import annotations

import numpy as np

from qs.operator_view import AMBER, GREEN, reason_line
from test_screen_annunciator import frame, has_colour, make, no_verdict, uniform

WORDS = "стаканчик не распознан — проверьте вручную"


def test_the_reason_line_says_a_cup_was_not_recognised_and_must_be_checked() -> None:
    assert reason_line("untracked") == WORDS


def test_an_untracked_record_is_amber_with_the_reason_in_words_and_no_green() -> None:
    annunciator, screen = make()

    annunciator.show(frame(uniform((30, 200, 10))), no_verdict("untracked"))

    presented = screen.presented[-1]
    shown = annunciator.last_shown
    assert presented.getpixel((4, 40)) == AMBER
    assert not has_colour(np.asarray(presented), GREEN)
    assert shown is not None
    assert shown.detail == WORDS
