"""`qs lane-split`: each recorded clip gets one role, and its hash is pinned (#154)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from gapped_clip import make_gapped_clip
from qs.__main__ import build_parser, run
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


# Review of PR #162: literal hashes, strict sidecars, load_split refusals, the real entry point.

TRAIN_SHA = "f37203b993fa504d44b6983b7271a90db8e9ea194b415be36abc765aff9c8a1b"


def test_the_pinned_hash_is_the_sha256_of_the_bytes(tmp_path: Path) -> None:
    footage = tmp_path / "footage"
    footage.mkdir()
    for name, data in CLIPS.items():
        (footage / f"{name}.mkv").write_bytes(data)
        sidecar = {"version": 1, "clip": name, "drifted": False,
                   "sha256": TRAIN_SHA if name == "T-1" else hashlib.sha256(data).hexdigest()}
        (footage / f"{name}.json").write_text(json.dumps(sidecar))
    out = tmp_path / "split.yml"
    assert _run(footage, out, *GOOD) == 0
    assert load_split(out).clips["T-1.mkv"] == ("train", TRAIN_SHA)


@pytest.mark.parametrize("drifted", [{}, {"drifted": "no"}, {"drifted": 0}])
def test_a_sidecar_without_a_boolean_drifted_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], drifted: dict[str, object]) -> None:
    footage = _footage(tmp_path)
    sidecar = {"version": 1, "clip": "E-1", "sha256": file_sha256(footage / "E-1.mkv"), **drifted}
    (footage / "E-1.json").write_text(json.dumps(sidecar))
    assert _run(footage, tmp_path / "split.yml", *GOOD) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "E-1.json" in out[0]


@pytest.mark.parametrize("text, word", [
    ("{version: 1, clips: {T-1.mkv: {role: boss, sha256: abc}}}", "boss"),
    ("{version: 2, clips: {}}", "version"),
    ("- a\n- b\n", "not a split file"),
])
def test_load_split_refuses_a_bad_file(tmp_path: Path, text: str, word: str) -> None:
    path = tmp_path / "split.yml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=word):
        load_split(path)


def test_the_real_entry_point_runs_lane_split(tmp_path: Path) -> None:
    footage = _footage(tmp_path)
    out = tmp_path / "split.yml"
    code = run(["lane-split", "--footage", str(footage), *GOOD, "--out", str(out)])
    assert code == 0
    assert load_split(out).videos("test") == ("E-1.mkv",)


def test_the_same_clip_twice_in_one_role_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    footage = _footage(tmp_path)
    code = _run(footage, tmp_path / "split.yml", "--train", "T-1", "T-1", "--calibrate", "C-1",
                "--test", "E-1")
    assert code == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        "lane-split: T-1 is listed twice under train"]


def test_an_unreadable_sidecar_is_not_blamed_on_out(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    footage = _footage(tmp_path)
    (footage / "C-1.json").chmod(0)
    assert _run(footage, tmp_path / "split.yml", *GOOD) == 2
    out = capsys.readouterr().out
    assert "C-1.json" in out
    assert "cannot be written" not in out


def test_an_unwritable_out_is_reported_as_such(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    footage = _footage(tmp_path)
    assert _run(footage, tmp_path / "missing-dir" / "split.yml", *GOOD) == 2
    assert "cannot be written" in capsys.readouterr().out


def test_an_existing_out_is_not_overwritten(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    footage = _footage(tmp_path)
    out = tmp_path / "split.yml"
    out.write_text("keep me")
    assert _run(footage, out, *GOOD) == 2
    assert "already exists" in capsys.readouterr().out
    assert out.read_text() == "keep me"


def test_a_clip_with_frames_missing_is_refused_naming_the_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    footage = _footage(tmp_path)
    video = make_gapped_clip(footage, "T-1-gapped")
    shutil.move(str(video), footage / "T-1.mkv")
    sidecar = json.loads((footage / "T-1.json").read_text())
    (footage / "T-1.json").write_text(
        json.dumps({**sidecar, "sha256": file_sha256(footage / "T-1.mkv")}))
    out = tmp_path / "split.yml"
    assert _run(footage, out, *GOOD) == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        "lane-split: T-1.mkv decodes 110 frames but its container says 120: "
        "frame numbers are not reliable; run qs clip-fix"]
    assert not out.exists()


def test_a_clip_that_cannot_be_opened_is_named_as_not_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    footage = _footage(tmp_path)
    assert _run(footage, tmp_path / "split.yml", *GOOD) == 0
    assert capsys.readouterr().out.strip().splitlines() == [
        "lane-split: T-1.mkv: cannot be opened, frame count not checked",
        "lane-split: C-1.mkv: cannot be opened, frame count not checked",
        "lane-split: E-1.mkv: cannot be opened, frame count not checked",
        f"lane-split: 3 clips written to {tmp_path / 'split.yml'}",
    ]
