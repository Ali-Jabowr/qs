"""A line station may add the appearance check to its empty-cup check (#109).

`empty_cup` stays required: an empty cup is never judged only by a model trained on full cups.
"""

from __future__ import annotations

from pathlib import Path

from qs.config import load_config
from test_config import CUP_LINE_KEYS, LINE_BLOCK, _line_mutated, _mutated, _refused


def test_a_line_station_may_list_empty_cup_and_appearance(tmp_path: Path) -> None:
    head = "  type: line\n  checks: [empty_cup, appearance]\n"
    path = _line_mutated(tmp_path, head + LINE_BLOCK + CUP_LINE_KEYS)

    assert load_config(path).config.inspector.checks == frozenset({"empty_cup", "appearance"})


def test_a_line_station_may_not_list_appearance_without_empty_cup(tmp_path: Path) -> None:
    head = "  type: line\n  checks: [appearance]\n"
    _refused(
        _mutated(tmp_path, "  type: appearance\n", head + LINE_BLOCK),
        "inspector.checks: a line inspector must list ['empty_cup'] or "
        "['appearance', 'empty_cup']",
    )
