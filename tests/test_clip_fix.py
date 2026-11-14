"""`qs clip-fix`: renumber a clip's frames without re-encoding (#173)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gapped_clip import make_gapped_clip, make_off_rate_clip
from qs.__main__ import run
from qs.cmd import clip_fix
from qs.lane.split import ClipCounts

SIDECAR = {
    "version": 1,
    "clip": "T-1",
    "video": "T-1.mkv",
    "fps": 30,
    "width": 1280,
    "started_utc": "2026-11-14T10:00:00.000Z",
    "stopped_utc": "2026-11-14T10:01:00.000Z",
    "drifted": False,
    "sha256": "aaaa",
}


class Rig:
    """The ffmpeg process and the frame count, faked, and what they saw."""

    def __init__(self, tmp_path: Path, missing: int = 10) -> None:
        self.video = tmp_path / "T-1.mkv"
        self.video.write_bytes(b"original")
        self.sidecar = tmp_path / "T-1.json"
        self.sidecar.write_text(json.dumps(SIDECAR))
        self.missing = missing
        self.argv: list[list[str]] = []

    def ffmpeg(self, argv: list[str]) -> int:
        self.argv.append(argv)
        Path(argv[-1]).write_bytes(b"fixed")
        return 0

    def counts(self, video: Path) -> ClipCounts:
        if video.name.endswith("-fixed.mkv"):
            return ClipCounts(110, 110, 0)
        return ClipCounts(110, 120 if self.missing else 110, self.missing)

    def run(self) -> int:
        return clip_fix.main(
            argparse_namespace(self.video),
            run_ffmpeg=self.ffmpeg,
            count=self.counts,
            file_hash=lambda path: "bbbb",
        )


def argparse_namespace(video: Path) -> argparse.Namespace:
    return argparse.Namespace(video=str(video))


def test_ffmpeg_gets_exactly_the_copy_and_setts_arguments(tmp_path):
    rig = Rig(tmp_path)
    assert rig.run() == 0
    assert rig.argv == [[
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(tmp_path / "T-1.mkv"),
        "-c", "copy", "-bsf:v", "setts=ts=N/(30*TB)", str(tmp_path / "T-1-fixed.mkv"),
    ]]


def test_the_sidecar_keeps_every_field_and_adds_three(tmp_path):
    rig = Rig(tmp_path, missing=7)
    rig.run()
    written = json.loads((tmp_path / "T-1-fixed.json").read_text())
    assert written == {**SIDECAR, "sha256": "bbbb", "fixed_from": "aaaa", "missing_frames": 7}


def test_the_original_clip_and_sidecar_are_untouched(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    assert rig.video.read_bytes() == b"original"
    assert json.loads(rig.sidecar.read_text()) == SIDECAR


@pytest.mark.parametrize("existing", ["T-1-fixed.mkv", "T-1-fixed.json"])
def test_an_existing_output_is_refused_and_nothing_is_written(tmp_path, capsys, existing):
    rig = Rig(tmp_path)
    (tmp_path / existing).write_bytes(b"keep me")
    assert rig.run() == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        f"clip-fix: {tmp_path / existing} already exists; nothing overwritten"
    ]
    assert rig.argv == []
    assert (tmp_path / existing).read_bytes() == b"keep me"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["T-1.mkv", "T-1.json", existing]
    )


def test_a_clip_without_missing_frames_is_refused(tmp_path, capsys):
    rig = Rig(tmp_path, missing=0)
    assert rig.run() == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        f"clip-fix: {tmp_path / 'T-1.mkv'} has no missing frames and its counts agree; "
        "nothing to fix"
    ]
    assert rig.argv == []


def test_a_clip_without_a_sidecar_is_refused(tmp_path, capsys):
    rig = Rig(tmp_path)
    rig.sidecar.unlink()
    assert rig.run() == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        f"clip-fix: the sidecar {tmp_path / 'T-1.json'} is missing"
    ]


def test_a_failing_ffmpeg_writes_nothing(tmp_path, capsys):
    rig = Rig(tmp_path)
    code = clip_fix.main(
        argparse_namespace(rig.video),
        run_ffmpeg=lambda argv: 1,
        count=lambda video: ClipCounts(110, 120, 10),
        file_hash=lambda path: "bbbb",
    )
    assert code == 2
    assert capsys.readouterr().out.strip().splitlines() == ["clip-fix: ffmpeg exited with code 1"]
    assert not (tmp_path / "T-1-fixed.json").exists()


def decode_in_order(video: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    frames = []
    ok, frame = capture.read()
    while ok:
        frames.append(frame)
        ok, frame = capture.read()
    capture.release()
    return frames


def decode_by_seeking(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    assert ok
    return frame


def container_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def test_a_real_clip_with_ten_frames_removed_is_fixed(tmp_path, capsys):
    video = make_gapped_clip(tmp_path, "T-1")
    (tmp_path / "T-1.json").write_text(json.dumps({**SIDECAR, "sha256": "aaaa"}))
    assert (len(decode_in_order(video)), container_count(video)) == (110, 120)
    assert run(["clip-fix", str(video)]) == 0
    fixed = tmp_path / "T-1-fixed.mkv"
    assert (len(decode_in_order(fixed)), container_count(fixed)) == (110, 110)
    assert np.array_equal(decode_by_seeking(fixed, 60), decode_in_order(fixed)[60])
    written = json.loads((tmp_path / "T-1-fixed.json").read_text())
    assert (written["missing_frames"], written["fixed_from"]) == (10, "aaaa")


def rig_with_counts(tmp_path: Path, counts: dict[str, ClipCounts]) -> tuple[Rig, int]:
    """A rig whose frame counter answers by file name: the original and the fixed output."""
    rig = Rig(tmp_path)
    code = clip_fix.main(
        argparse_namespace(rig.video),
        run_ffmpeg=rig.ffmpeg,
        count=lambda video: counts[video.name],
        file_hash=lambda path: "bbbb",
    )
    return rig, code


def test_a_clip_whose_container_count_is_off_but_misses_no_frame_is_fixed(tmp_path):
    rig, code = rig_with_counts(tmp_path, {
        "T-1.mkv": ClipCounts(91, 92, 0), "T-1-fixed.mkv": ClipCounts(91, 91, 0)})
    assert code == 0
    written = json.loads((tmp_path / "T-1-fixed.json").read_text())
    assert written["missing_frames"] == 0
    assert len(rig.argv) == 1


def test_a_clip_with_agreeing_counts_and_no_gap_has_nothing_to_fix(tmp_path, capsys):
    rig, code = rig_with_counts(tmp_path, {"T-1.mkv": ClipCounts(91, 91, 0)})
    assert code == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        f"clip-fix: {tmp_path / 'T-1.mkv'} has no missing frames and its counts agree; "
        "nothing to fix"
    ]
    assert rig.argv == []


def test_an_output_whose_counts_still_disagree_is_deleted(tmp_path, capsys):
    _, code = rig_with_counts(tmp_path, {
        "T-1.mkv": ClipCounts(110, 120, 10), "T-1-fixed.mkv": ClipCounts(110, 112, 0)})
    assert code == 2
    assert capsys.readouterr().out.strip().splitlines() == [
        f"clip-fix: {tmp_path / 'T-1-fixed.mkv'} decodes 110 frames but its container says "
        "112 after the fix; output deleted"
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["T-1.json", "T-1.mkv"]


def test_a_real_clip_at_a_true_rate_of_30_2_with_nothing_removed_is_fixed(tmp_path):
    video = make_off_rate_clip(tmp_path, "T-1")
    (tmp_path / "T-1.json").write_text(json.dumps({**SIDECAR, "sha256": "aaaa"}))
    assert (len(decode_in_order(video)), container_count(video)) == (91, 90)
    assert run(["clip-fix", str(video)]) == 0
    fixed = tmp_path / "T-1-fixed.mkv"
    assert (len(decode_in_order(fixed)), container_count(fixed)) == (91, 91)
    assert json.loads((tmp_path / "T-1-fixed.json").read_text())["missing_frames"] == 0
