"""`qs train-lane-b --states`: fit on the passes of the listed states only (#175).

`_setup` of the train-lane-b tests gives T-1.mkv with full@50, empty@82, full_pink@114, gap@146.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FakePatchCore
from qs.__main__ import build_parser
from qs.cmd import train_lane_b
from test_train_lane_b import _setup


def _train(tmp_path: Path, fake: FakePatchCore, *flags: str) -> int:
    paths = _setup(tmp_path)
    args = build_parser().parse_args([
        "train-lane-b", "--lane", str(paths["lane"]), "--labels", str(paths["labels"]),
        "--footage", str(paths["footage"]), "--split", str(paths["split"]),
        "--product", "mochi-lane", "--input-px", "32", "--models", str(tmp_path / "models"),
        *flags,
    ])
    return train_lane_b.main(args, make_patchcore=lambda device: fake, cuda_available=lambda: True)


def _fitted_files(tmp_path: Path) -> str:
    (bundle,) = (tmp_path / "models").iterdir()
    return (bundle / "bundle.yml").read_text(encoding="utf-8")


def test_states_keeps_only_the_passes_of_the_listed_states(tmp_path: Path) -> None:
    fake = FakePatchCore()
    assert _train(tmp_path, fake, "--states", "full_pink") == 0
    assert len(fake.fits[0][0]) == 1
    assert "T-1.mkv#114" in _fitted_files(tmp_path)
    assert "T-1.mkv#50" not in _fitted_files(tmp_path)


def test_a_state_with_no_pass_in_the_clips_is_listed_without_error(tmp_path: Path) -> None:
    fake = FakePatchCore()
    assert _train(tmp_path, fake, "--states", "full_pink,full_white") == 0
    assert "T-1.mkv#114" in _fitted_files(tmp_path)
    assert len(fake.fits[0][0]) == 1


def test_a_single_other_state_selects_its_own_pass(tmp_path: Path) -> None:
    fake = FakePatchCore()
    assert _train(tmp_path, fake, "--states", "full") == 0
    assert "T-1.mkv#50" in _fitted_files(tmp_path)
    assert "T-1.mkv#114" not in _fitted_files(tmp_path)


def test_states_left_out_fit_all_three_full_states(tmp_path: Path) -> None:
    fake = FakePatchCore()
    assert _train(tmp_path, fake) == 0
    assert len(fake.fits[0][0]) == 2


def test_a_state_that_is_not_a_full_state_is_refused_with_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        _train(tmp_path, FakePatchCore(), "--states", "empty")
    assert stop.value.code == 2
    assert "'empty'" in capsys.readouterr().err
