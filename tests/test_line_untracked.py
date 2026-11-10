"""A warm passage with no tracked cup is recorded as НЕТ РЕШЕНИЯ(`untracked`) (#134, ADR-0029 §7).

Footage: three frames of steel, a warm disc crossing with no rim drawn, a pink cup crossing, and
steel. The pink cup is tracked and judged on its burst as in `test_line_warmth_score`; the disc
is seen only by the window.
"""

from __future__ import annotations

from pathlib import Path

from qs.cmd.score_line import play_video
from qs.config import load_config
from qs.contracts import InspectionRecord
from untracked_fixtures import VIDEO, config, cup_crossing, disc_crossing, steel, write_video


def _play(root: Path, frames: list, **line: object) -> list[InspectionRecord]:
    video = write_video(root / VIDEO, frames)
    records, _ = play_video(load_config(config(root, **line)), video)
    return records


def _outcome(record: InspectionRecord) -> tuple[str, str | None]:
    return record.verdict.label.value, record.verdict.reason


def test_a_warm_disc_with_no_rim_gives_one_untracked_record(tmp_path: Path) -> None:
    records = _play(tmp_path, steel(3) + disc_crossing() + steel(3))

    assert [_outcome(r) for r in records] == [("no_verdict", "untracked")]


def test_a_tracked_full_cup_gives_no_untracked_record(tmp_path: Path) -> None:
    records = _play(tmp_path, steel(3) + cup_crossing("pink") + steel(3))

    assert [_outcome(r) for r in records] == [("pass", None)]


def test_a_disc_and_a_cup_each_get_their_own_record_in_order(tmp_path: Path) -> None:
    records = _play(tmp_path, steel(3) + disc_crossing() + cup_crossing("pink") + steel(3))

    assert [_outcome(r) for r in records] == [("no_verdict", "untracked"), ("pass", None)]


def test_a_tracked_empty_cup_does_not_excuse_the_disc_that_follows(tmp_path: Path) -> None:
    records = _play(tmp_path, steel(3) + cup_crossing("empty") + disc_crossing() + steel(3))

    assert [_outcome(r) for r in records] == [("defect", None), ("no_verdict", "untracked")]


def test_without_the_window_the_disc_is_not_seen(tmp_path: Path) -> None:
    records = _play(tmp_path, steel(3) + disc_crossing() + steel(3), untracked_window=None)

    assert records == []


def test_the_untracked_record_has_no_checks(tmp_path: Path) -> None:
    (record,) = _play(tmp_path, steel(3) + disc_crossing() + steel(3))

    assert (record.event.seq, record.verdict.checks) == (1, ())
