"""`qs freeze`: the hashes and the commit of what a scoring will rest on (#160)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from lane_rig import HEAD, MODEL_ID, fake_git, make_rig, run_freeze
from qs.lane.freeze import Freeze, load_freeze

MODEL_SHA = "a2d42c4aa884e21216cbb8da4c7ba2fcf9b6033b2331666e666145c24caf7a38"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_writes_the_hashes_and_the_commit(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    assert run_freeze(rig) == 0
    assert load_freeze(rig.freeze) == Freeze(
        git_commit=HEAD,
        model_id=MODEL_ID,
        model_sha256=MODEL_SHA,
        threshold_sha256=_sha(rig.threshold),
        lane_sha256=_sha(rig.lane),
        split_sha256=_sha(rig.split),
        patchcore=None,
        created_utc="2026-11-16T12:00:00+00:00",
    )


def test_freeze_file_has_the_contract_keys(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    run_freeze(rig, "--patchcore", "patchcore-mochi-v1-0123456789ab", "--patchcore-threshold",
               "0.4")
    data = yaml.safe_load(rig.freeze.read_text())
    assert sorted(data) == ["created_utc", "git_commit", "lane_bundle", "lane_sha256",
                            "patchcore", "split_sha256", "threshold_sha256", "version"]
    assert data["version"] == 1
    assert data["lane_bundle"] == {"model_id": MODEL_ID, "model_sha256": MODEL_SHA}
    assert data["patchcore"] == {"model_id": "patchcore-mochi-v1-0123456789ab", "threshold": 0.4}


def test_freeze_on_a_dirty_tree_is_refused_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = make_rig(tmp_path)
    code = run_freeze(rig, git=fake_git(dirty=" M src/qs/lane/passes.py\n"))
    out = capsys.readouterr().out
    assert code == 2
    assert len(out.splitlines()) == 1
    assert "uncommitted" in out
    assert not rig.freeze.exists()


def test_freeze_does_not_overwrite_a_freeze_file(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    rig.freeze.write_text("old freeze")
    assert run_freeze(rig) == 2
    assert rig.freeze.read_text() == "old freeze"


@pytest.mark.parametrize(
    "extra",
    [
        ("--patchcore", "patchcore-mochi-v1-0123456789ab"),
        ("--patchcore-threshold", "0.4"),
        ("--patchcore", "patchcore-mochi-v1-0123456789ab", "--patchcore-threshold", "nan"),
    ],
)
def test_the_patchcore_model_and_threshold_come_together_and_finite(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    rig = make_rig(tmp_path)
    assert run_freeze(rig, *extra) == 2
    assert not rig.freeze.exists()


def test_a_threshold_made_for_another_model_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = make_rig(tmp_path)
    rig.threshold.write_text(rig.threshold.read_text().replace(MODEL_ID, "lane-mochi-v2"))
    assert run_freeze(rig) == 2
    assert "lane-mochi-v2" in capsys.readouterr().out
    assert not rig.freeze.exists()


def test_a_model_file_that_differs_from_its_bundle_record_is_refused(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    rig.model_file.write_bytes(b"tampered")
    assert run_freeze(rig) == 2
    assert not rig.freeze.exists()
