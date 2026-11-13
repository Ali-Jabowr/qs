"""`qs lane-split`: each recorded clip gets one role, and its hash is pinned (#154)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qs.__main__ import build_parser
from qs.cmd import lane_split
from qs.errors import ConfigError
from qs.lane.split import Split, file_sha256, load_split, require_roles, verify

CLIPS = {"T-1": b"train bytes", "C-1": b"calibrate bytes", "E-1": b"test bytes"}

GOOD = ("--train", "T-1", "--calibrate", "C-1", "--test", "E-1")


def _footage(root: Path, drifted: dict[str, bool] | None = None) -> Path:
    footage = root / "footage"
    footage.mkdir()
    for name, data in CLIPS.items():
        video = footage / f"{name}.mkv"
        video.write_bytes(data)
        sidecar = {"version": 1, "clip": name, "sha256": file_sha256(video),
                   "drifted": (drifted or {}).get(name, False)}
        (footage / f"{name}.json").write_text(json.dumps(sidecar))
    return footage


def _run(footage: Path, out: Path, *roles: str) -> int:
    return lane_split.main(build_parser().parse_args(
        ["lane-split", "--footage", str(footage), *roles, "--out", str(out)]))


def test_three_clips_get_their_roles(tmp_path: Path) -> None:
    footage = _footage(tmp_path)
    out = tmp_path / "split.yml"
    assert _run(footage, out, *GOOD) == 0
    split = load_split(out)
    assert {v: r for v, (r, _) in split.clips.items()} == {
        "T-1.mkv": "train", "C-1.mkv": "calibrate", "E-1.mkv": "test"}
    assert split.clips["T-1.mkv"][1] == file_sha256(footage / "T-1.mkv")
    assert split.videos("calibrate") == ("C-1.mkv",)


def test_a_clip_in_two_roles_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    footage = _footage(tmp_path)
    code = _run(footage, tmp_path / "split.yml", "--train", "T-1", "--calibrate", "C-1",
                "--test", "T-1")
    assert code == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        "lane-split: T-1 is given two roles: train, test"]


def test_a_missing_sidecar_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    footage = _footage(tmp_path)
    (footage / "C-1.json").unlink()
    assert _run(footage, tmp_path / "split.yml", *GOOD) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "C-1.json" in out[0]


def test_changed_bytes_are_refused_with_both_hashes(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    footage = _footage(tmp_path)
    recorded = file_sha256(footage / "T-1.mkv")[:8]
    (footage / "T-1.mkv").write_bytes(b"other bytes")
    now = file_sha256(footage / "T-1.mkv")[:8]
    assert _run(footage, tmp_path / "split.yml", *GOOD) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "T-1.mkv" in out[0] and recorded in out[0] and now in out[0]


def test_a_drifted_clip_is_refused_unless_allowed(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    footage = _footage(tmp_path, drifted={"E-1": True})
    assert _run(footage, tmp_path / "split.yml", *GOOD) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "E-1" in out[0] and "drifted" in out[0]
    assert _run(footage, tmp_path / "split.yml", *GOOD, "--allow-drifted") == 0


@pytest.mark.parametrize("roles", [
    ("--train", "T-1", "--test", "E-1"),
    ("--train", "T-1", "--calibrate", "C-1"),
])
def test_every_role_needs_a_clip(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                 roles: tuple[str, ...]) -> None:
    footage = _footage(tmp_path)
    assert _run(footage, tmp_path / "split.yml", *roles) == 2
    assert "every role needs at least one clip" in capsys.readouterr().out


def _split() -> Split:
    return Split({"T-1.mkv": ("train", "a" * 64), "E-1.mkv": ("test", "b" * 64)})


def test_require_roles_names_the_video_and_its_role() -> None:
    with pytest.raises(ConfigError, match=r"E-1\.mkv.*test"):
        require_roles(_split(), ["T-1.mkv", "E-1.mkv"], {"train", "calibrate"})


def test_require_roles_accepts_allowed_videos() -> None:
    require_roles(_split(), ["T-1.mkv"], {"train", "calibrate"})


def test_role_of_an_unknown_clip_names_it() -> None:
    with pytest.raises(ConfigError, match=r"X\.mkv"):
        _split().role_of("X.mkv")


def test_verify_names_a_clip_whose_bytes_changed(tmp_path: Path) -> None:
    footage = _footage(tmp_path)
    out = tmp_path / "split.yml"
    assert _run(footage, out, *GOOD) == 0
    split = load_split(out)
    verify(split, footage, file_sha256)
    (footage / "C-1.mkv").write_bytes(b"changed")
    with pytest.raises(ConfigError, match=r"C-1\.mkv"):
        verify(split, footage, file_sha256)
