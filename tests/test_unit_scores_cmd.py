"""`qs unit-scores`: the off-colour scores of a block's full passes, written to a CSV (#175).

The rig is the drawn calibrate clip of the unit-scores tests; the model bundles are real files
holding a tiny bank, and the scorer is a stand-in.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from qs.__main__ import build_parser
from qs.cmd import unit_scores
from qs.inspector.bundle import BundleInfo, ImageRecord, write_bundle
from test_lane_unit_scores import CentreScorer, make_rig


def _bundle(models: Path) -> str:
    info = BundleInfo(32, 276, 0.1, 42, "cpu", "own", (ImageRecord("T-1.mkv#5", "T-1.mkv", "0" * 64),))
    return write_bundle(models, "mochi-lane", np.zeros((2, 3), dtype=np.float32), info)


def _run(
    tmp_path: Path, paths: dict[str, Path], *flags: str, block: str = "calibrate"
) -> int:
    args = build_parser().parse_args([
        "unit-scores", "--lane", str(paths["lane"]), "--unit", str(paths["unit"]),
        "--labels", str(paths["labels"]), "--footage", str(paths["footage"]),
        "--split", str(paths["split"]), "--block", block,
        "--out", str(tmp_path / "scores.csv"), "--models", str(tmp_path / "models"),
        "--device", "cpu", *flags,
    ])
    return unit_scores.main(args, make_scorer=lambda device, bank: CentreScorer(),
                            cuda_available=lambda: False)


def test_the_calibrate_block_is_scored_into_the_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_rig(tmp_path)
    model_id = _bundle(tmp_path / "models")
    assert _run(tmp_path, paths, "--patchcore", f"full_pink={model_id}") == 0
    table = list(csv.DictReader((tmp_path / "scores.csv").open(encoding="utf-8")))
    assert [(r["state"], r["frame"], r["patch_score"]) for r in table] == [
        ("full_pink", "50", "7.5"), ("full_white", "114", ""), ("full", "146", "")]
    assert table[2]["fault"] == "no unit"
    assert capsys.readouterr().out == "scored 3 passes of 1 clips; 1 without a cut-out\nno unit: 1\n"


def test_the_test_block_cannot_be_asked_for(tmp_path: Path) -> None:
    paths = make_rig(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _run(tmp_path, paths, block="test")
    assert stop.value.code == 2


def test_the_test_clips_are_not_read(tmp_path: Path) -> None:
    paths = make_rig(tmp_path)
    (paths["footage"] / "E-1.mkv").write_bytes(b"changed after the split")
    assert _run(tmp_path, paths) == 0


@pytest.mark.parametrize("entry", ["empty=x", "full_pink", "full_pink=patchcore-mochi-lane-v1-000000000000"])
def test_a_bad_patchcore_entry_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: str
) -> None:
    paths = make_rig(tmp_path)
    assert _run(tmp_path, paths, "--patchcore", entry) == 2
    assert not (tmp_path / "scores.csv").exists()
    assert entry.split("=")[0] in capsys.readouterr().out


def test_a_block_with_no_full_pass_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_rig(tmp_path)
    paths["labels"].write_text(
        '{"version": 1, "source": "L", "frames": {}, "cups": ['
        '{"video": "C-1.mkv", "order": 1, "state": "empty", "frame": 82}]}')
    assert _run(tmp_path, paths) == 1
    assert capsys.readouterr().out.endswith("refused: no full pass in the calibrate clips\n")
