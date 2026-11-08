"""`qs label`: the labels file and the two labelling sessions, on synthetic images only."""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from qs.__main__ import run
from qs.cmd.label import main, named_key
from qs.labels import (
    CupSession,
    FrameLabel,
    FrameReadError,
    FrameSession,
    ImageFolder,
    LabelsError,
    VideoFrames,
    default_labels_path,
    load_labels,
    source_name,
)
from qs.screen import KeyPress

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"


def draw_images(folder: Path, *names: str) -> Path:
    """Flat grey images drawn here, one per name, in a new folder."""
    folder.mkdir()
    for name in names:
        cv2.imwrite(str(folder / name), np.full((24, 32, 3), 128, np.uint8))
    return folder


def frame_session(root: Path, *names: str) -> FrameSession:
    folder = root / "shoot"
    if not folder.exists():
        draw_images(folder, *names)
    return FrameSession(load_labels(root / "labels.json", "shoot"), ImageFolder(folder))


def on_disk(root: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    return document


def test_an_answer_is_on_disk_at_once(tmp_path: Path) -> None:
    session = frame_session(tmp_path, "a.png", "b.png")
    session.press("w")
    assert (tmp_path / "labels.json").is_file()
    assert on_disk(tmp_path) == {
        "version": 1,
        "source": "shoot",
        "frames": {"a.png": {"label": "unit", "flavour": "white"}},
        "cups": [],
    }


@pytest.mark.parametrize(
    ("key", "label", "flavour"),
    [
        ("u", "unit", None),
        ("w", "unit", "white"),
        ("p", "unit", "pink"),
        ("e", "empty_cup", None),
        ("n", "nothing", None),
        ("h", "hand", None),
    ],
)
def test_each_answer_key_means_its_label(
    tmp_path: Path, key: str, label: str, flavour: str | None
) -> None:
    session = frame_session(tmp_path, "a.png")
    session.press(key)
    assert session.labels.frames.get("a.png") == FrameLabel(label, flavour)


def test_a_flavour_on_anything_but_a_unit_is_refused_and_not_saved(tmp_path: Path) -> None:
    labels = load_labels(tmp_path / "labels.json", "shoot")
    with pytest.raises(LabelsError, match="flavour"):
        labels.set_frame("a.png", "empty_cup", "white")
    assert not (tmp_path / "labels.json").exists()


def test_an_answer_moves_on_to_the_next_frame_in_name_order(tmp_path: Path) -> None:
    session = frame_session(tmp_path, "b.png", "a.png", "c.png")
    session.press("n")
    session.press("h")
    assert session.labels.frames == {
        "a.png": FrameLabel("nothing", None),
        "b.png": FrameLabel("hand", None),
    }


def test_reopening_resumes_at_the_first_unlabelled_frame(tmp_path: Path) -> None:
    first = frame_session(tmp_path, "a.png", "b.png", "c.png")
    first.press("n")
    first.press("h")
    frame_session(tmp_path).press("e")
    assert on_disk(tmp_path)["frames"] == {
        "a.png": {"label": "nothing", "flavour": None},
        "b.png": {"label": "hand", "flavour": None},
        "c.png": {"label": "empty_cup", "flavour": None},
    }


def test_going_back_and_answering_replaces_the_label(tmp_path: Path) -> None:
    session = frame_session(tmp_path, "a.png", "b.png")
    session.press("w")
    session.press("Left")
    session.press("e")
    assert on_disk(tmp_path)["frames"] == {"a.png": {"label": "empty_cup", "flavour": None}}


def test_going_forward_answers_nothing_and_resume_finds_the_skipped_frame(tmp_path: Path) -> None:
    session = frame_session(tmp_path, "a.png", "b.png")
    session.press("Right")
    session.press("n")
    assert on_disk(tmp_path)["frames"] == {"b.png": {"label": "nothing", "flavour": None}}
    frame_session(tmp_path).press("h")
    assert on_disk(tmp_path)["frames"] == {
        "a.png": {"label": "hand", "flavour": None},
        "b.png": {"label": "nothing", "flavour": None},
    }


def test_with_every_frame_labelled_reopening_shows_the_last_for_correction(tmp_path: Path) -> None:
    session = frame_session(tmp_path, "a.png", "b.png")
    session.press("n")
    session.press("n")
    frame_session(tmp_path).press("h")
    assert on_disk(tmp_path)["frames"] == {
        "a.png": {"label": "nothing", "flavour": None},
        "b.png": {"label": "hand", "flavour": None},
    }


def test_arrows_stop_at_both_ends(tmp_path: Path) -> None:
    session = frame_session(tmp_path, "a.png", "b.png")
    session.press("Left")
    session.press("n")
    session.press("Right")
    session.press("Right")
    session.press("h")
    assert on_disk(tmp_path)["frames"] == {
        "a.png": {"label": "nothing", "flavour": None},
        "b.png": {"label": "hand", "flavour": None},
    }


def test_the_screen_lines_show_position_count_and_current_label(tmp_path: Path) -> None:
    session = frame_session(tmp_path, "a.png", "b.png", "c.png")
    session.press("w")
    session.press("Left")
    assert session.lines()[:2] == ["frame 1 of 3: a.png", "1 labelled · this frame: unit, white"]
    session.press("Right")
    assert session.lines()[:2] == ["frame 2 of 3: b.png", "1 labelled · this frame: not labelled"]


def test_a_crash_during_a_save_leaves_the_previous_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(source: object, target: object) -> None:
        raise OSError("disk gone")

    session = frame_session(tmp_path, "a.png", "b.png")
    session.press("n")
    before = (tmp_path / "labels.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr("qs.labels.os.replace", crash)
        with pytest.raises(OSError, match="disk gone"):
            session.press("h")
    assert (tmp_path / "labels.json").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.json", "shoot"]
    assert session.labels.frames == {"a.png": FrameLabel("nothing", None)}
    frame_session(tmp_path).press("e")
    assert on_disk(tmp_path)["frames"] == {
        "a.png": {"label": "nothing", "flavour": None},
        "b.png": {"label": "empty_cup", "flavour": None},
    }


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param('{"version": 1, "source": "s", "frames": {', id="cut-off"),
        pytest.param("[1, 2]", id="not-an-object"),
        pytest.param('{"version": 2, "source": "s", "frames": {}, "cups": []}', id="new-version"),
        pytest.param(
            '{"version": "1", "source": "s", "frames": {}, "cups": []}', id="text-version"
        ),
        pytest.param('{"source": "s", "frames": {}, "cups": []}', id="no-version"),
        pytest.param('{"version": 1, "frames": {}, "cups": []}', id="no-source"),
        pytest.param('{"version": 1, "source": "s", "frames": {}}', id="no-cups"),
        pytest.param(
            '{"version": 1, "source": "s", "frames": {}, "cups": [], "extra": 1}', id="extra-key"
        ),
        pytest.param(
            '{"version": 1, "source": "s", "cups": [], '
            '"frames": {"a.png": {"label": "maybe", "flavour": null}}}',
            id="unknown-label",
        ),
        pytest.param(
            '{"version": 1, "source": "s", "cups": [], '
            '"frames": {"a.png": {"label": "unit", "flavour": "green"}}}',
            id="unknown-flavour",
        ),
        pytest.param(
            '{"version": 1, "source": "s", "cups": [], '
            '"frames": {"a.png": {"label": "hand", "flavour": "white"}}}',
            id="flavour-off-a-unit",
        ),
        pytest.param(
            '{"version": 1, "source": "s", "cups": [], "frames": {"a.png": {"label": "unit"}}}',
            id="no-flavour-key",
        ),
    ],
)
def test_a_bad_labels_file_is_refused_and_left_untouched(tmp_path: Path, text: str) -> None:
    path = tmp_path / "labels.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LabelsError, match="labels.json"):
        load_labels(path, "shoot")
    assert path.read_text(encoding="utf-8") == text


def draw_video(path: Path, count: int, fourcc: str = "MJPG") -> Path:
    """A video drawn here: frame k is black with a white bar at x = 10 k, a mark that survives
    lossy compression, so a read shows which frame it really got."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*fourcc), 30.0, (320, 48))
    assert writer.isOpened()
    for k in range(count):
        pixels = np.zeros((48, 320, 3), np.uint8)
        pixels[:, 10 * k : 10 * k + 10] = 255
        writer.write(pixels)
    writer.release()
    return path


def frame_shown(pixels: np.ndarray) -> int:
    """Which frame of `draw_video` these pixels are: from the middle of the white bar."""
    bright = np.flatnonzero(pixels.mean(axis=(0, 2)) > 128)
    return int(bright.mean()) // 10


def test_a_video_is_sampled_every_nth_frame_in_ascending_order(tmp_path: Path) -> None:
    frames = VideoFrames(draw_video(tmp_path / "v.avi", 7), every=3)
    assert frames.ids == ("v.avi#0", "v.avi#3", "v.avi#6")


@pytest.mark.parametrize(
    ("name", "fourcc"),
    [
        pytest.param("v.avi", "MJPG", id="every-frame-a-keyframe"),
        pytest.param("v.mp4", "mp4v", id="frames-depend-on-earlier-ones"),
    ],
)
def test_a_video_frame_is_read_by_position_in_any_order(
    tmp_path: Path, name: str, fourcc: str
) -> None:
    frames = VideoFrames(draw_video(tmp_path / name, 30, fourcc), every=2)
    assert frame_shown(frames.read(10)) == 20
    assert frame_shown(frames.read(3)) == 6
    assert frame_shown(frames.read(4)) == 8


def test_an_image_is_read_by_position_in_name_order(tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    cv2.imwrite(str(folder / "b.png"), np.full((24, 32, 3), 200, np.uint8))
    cv2.imwrite(str(folder / "a.png"), np.full((24, 32, 3), 50, np.uint8))
    assert ImageFolder(folder).read(1).mean() == 200.0


def test_only_images_count_as_frames_of_a_folder(tmp_path: Path) -> None:
    folder = draw_images(tmp_path / "shoot", "b.jpg", "a.png")
    (folder / "notes.txt").write_text("not a frame", encoding="utf-8")
    assert ImageFolder(folder).ids == ("a.png", "b.jpg")


def test_an_unreadable_image_is_an_error_naming_it(tmp_path: Path) -> None:
    folder = draw_images(tmp_path / "shoot", "a.png")
    (folder / "broken.png").write_bytes(b"not an image")
    with pytest.raises(FrameReadError, match=r"broken\.png"):
        ImageFolder(folder).read(1)


def test_an_unreadable_video_frame_is_an_error_naming_it(tmp_path: Path) -> None:
    frames = VideoFrames(draw_video(tmp_path / "v.avi", 3))
    with pytest.raises(FrameReadError, match=r"v\.avi#9"):
        frames.read(9)


def test_after_a_failed_read_the_next_read_still_gets_its_own_frame(tmp_path: Path) -> None:
    frames = VideoFrames(draw_video(tmp_path / "v.avi", 3))
    assert frame_shown(frames.read(0)) == 0
    with pytest.raises(FrameReadError):
        frames.read(9)
    assert frame_shown(frames.read(1)) == 1


def test_a_file_that_is_no_video_is_an_error_naming_it(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(FrameReadError, match=r"broken\.mp4"):
        VideoFrames(path)


def test_a_folder_with_no_image_is_an_error_naming_it(tmp_path: Path) -> None:
    with pytest.raises(FrameReadError, match="shoot"):
        ImageFolder(draw_images(tmp_path / "shoot"))


def test_the_session_shows_the_frame_at_its_position(tmp_path: Path) -> None:
    session = FrameSession(
        load_labels(tmp_path / "labels.json", "v"), VideoFrames(draw_video(tmp_path / "v.avi", 3))
    )
    assert frame_shown(session.image()) == 0
    session.press("n")
    assert frame_shown(session.image()) == 1


def cup_session(root: Path, name: str = "a.avi") -> CupSession:
    """Cups on the video `name`, 30 frames long, drawn the first time it is asked for."""
    path = root / name
    return CupSession(
        load_labels(root / "labels.json", "a"),
        VideoFrames(path if path.exists() else draw_video(path, 30)),
    )


def test_cups_are_recorded_in_order_at_their_frames_and_on_disk_at_once(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("Right")
    session.press("Right")
    session.press("f")
    session.press("Shift-Right")
    session.press("Left")
    session.press("e")
    assert (tmp_path / "labels.json").is_file()
    assert on_disk(tmp_path)["cups"] == [
        {"video": "a.avi", "order": 1, "state": "full", "frame": 2},
        {"video": "a.avi", "order": 2, "state": "empty", "frame": 26},
    ]


def test_big_steps_stop_at_both_ends_of_the_video(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("Shift-Left")
    session.press("f")
    session.press("Shift-Right")
    session.press("Shift-Right")
    session.press("f")
    assert [cup["frame"] for cup in on_disk(tmp_path)["cups"]] == [0, 29]


def test_two_videos_in_one_file_keep_separate_order(tmp_path: Path) -> None:
    cup_session(tmp_path, "a.avi").press("f")
    cup_session(tmp_path, "b.avi").press("e")
    cup_session(tmp_path, "a.avi").press("e")
    assert on_disk(tmp_path)["cups"] == [
        {"video": "a.avi", "order": 1, "state": "full", "frame": 0},
        {"video": "b.avi", "order": 1, "state": "empty", "frame": 0},
        {"video": "a.avi", "order": 2, "state": "empty", "frame": 0},
    ]


def test_backspace_removes_only_the_last_cup_of_this_video(tmp_path: Path) -> None:
    a = cup_session(tmp_path, "a.avi")
    a.press("f")
    a.press("e")
    b = cup_session(tmp_path, "b.avi")
    b.press("f")
    a = cup_session(tmp_path, "a.avi")
    a.press("BackSpace")
    assert on_disk(tmp_path)["cups"] == [
        {"video": "a.avi", "order": 1, "state": "full", "frame": 0},
        {"video": "b.avi", "order": 1, "state": "full", "frame": 0},
    ]
    a.press("e")
    assert on_disk(tmp_path)["cups"][-1] == {
        "video": "a.avi", "order": 2, "state": "empty", "frame": 0,
    }


def test_backspace_with_no_cup_in_this_video_changes_nothing(tmp_path: Path) -> None:
    cup_session(tmp_path, "a.avi").press("f")
    before = (tmp_path / "labels.json").read_bytes()
    cup_session(tmp_path, "b.avi").press("BackSpace")
    assert (tmp_path / "labels.json").read_bytes() == before


def test_reopening_a_video_resumes_at_the_frame_of_its_last_cup(tmp_path: Path) -> None:
    first = cup_session(tmp_path, "a.avi")
    first.press("Shift-Right")
    first.press("f")
    first.press("Right")
    cup_session(tmp_path, "b.avi").press("f")
    assert cup_session(tmp_path, "a.avi").position == 25
    assert cup_session(tmp_path, "b.avi").position == 0


def test_a_new_video_opens_at_its_first_frame(tmp_path: Path) -> None:
    assert cup_session(tmp_path).position == 0


def test_a_crash_during_a_cup_save_leaves_the_previous_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(source: object, target: object) -> None:
        raise OSError("disk gone")

    session = cup_session(tmp_path)
    session.press("f")
    before = (tmp_path / "labels.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr("qs.labels.os.replace", crash)
        with pytest.raises(OSError, match="disk gone"):
            session.press("e")
    assert (tmp_path / "labels.json").read_bytes() == before
    session.press("e")
    assert [cup["order"] for cup in on_disk(tmp_path)["cups"]] == [1, 2]


def test_frame_labels_and_cups_share_a_file_and_survive_each_other(tmp_path: Path) -> None:
    frames = VideoFrames(draw_video(tmp_path / "a.avi", 30), every=10)
    FrameSession(load_labels(tmp_path / "labels.json", "a"), frames).press("w")
    cup_session(tmp_path).press("f")
    FrameSession(load_labels(tmp_path / "labels.json", "a"), frames).press("n")
    assert on_disk(tmp_path) == {
        "version": 1,
        "source": "a",
        "frames": {
            "a.avi#0": {"label": "unit", "flavour": "white"},
            "a.avi#10": {"label": "nothing", "flavour": None},
        },
        "cups": [{"video": "a.avi", "order": 1, "state": "full", "frame": 0}],
    }


def test_the_cup_screen_lines_show_the_frame_and_the_counts(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("Right")
    session.press("f")
    session.press("Right")
    session.press("e")
    session.press("Right")
    assert session.lines()[:2] == [
        "frame 3 of 0–29: a.avi",
        "2 cups: 1 full, 1 empty · last: cup 2, empty, frame 2",
    ]


def test_the_default_labels_path_is_under_the_ignored_data_folder(tmp_path: Path) -> None:
    folder = draw_images(tmp_path / "photos.day1", "a.png")
    assert default_labels_path(folder) == Path("data/labels/photos.day1.json")
    assert default_labels_path(draw_video(tmp_path / "visit.avi", 1)) == Path(
        "data/labels/visit.json"
    )


def test_a_source_name_is_the_folder_name_or_the_video_stem(tmp_path: Path) -> None:
    folder = draw_images(tmp_path / "photos.day1", "a.png")
    assert source_name(folder) == "photos.day1"
    assert source_name(draw_video(tmp_path / "v.avi", 1)) == "v"


def test_the_summaries_are_counts_and_the_labels_path(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("f")
    session.press("e")
    assert session.summary() == f"2 cups in a.avi (1 full, 1 empty) in {tmp_path}/labels.json"


class FakeScreen:
    """The window: records every tick's lines; `keys()` plays a script, then closes."""

    def __init__(self, script: list[list[KeyPress]]) -> None:
        self.script = deque(script)
        self.lines: list[list[str]] = []
        self.closed = False

    def tick(self, image: np.ndarray | None, lines: list[str]) -> None:
        self.lines.append(list(lines))

    def keys(self) -> list[KeyPress]:
        if not self.script:
            self.closed = True  # the operator closes the window when the script runs out
            return []
        return self.script.popleft()

    def close(self) -> None:
        self.closed = True


def run_label(screen: FakeScreen, source: Path, labels: Path | None, **changes: object) -> int:
    values: dict[str, object] = dict(
        config=str(EXAMPLE), mode="frames", source=str(source), every=None, labels=labels
    )
    values.update(changes)
    return main(argparse.Namespace(**values), make_screen=lambda display, title: screen)


def key(*keysyms: str) -> list[KeyPress]:
    """The keys of one pump of the window."""
    return [KeyPress(keysym, 1000) for keysym in keysyms]


def test_main_labels_a_folder_through_the_window_and_writes_no_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = draw_images(tmp_path / "shoot", "a.png", "b.png", "c.png")
    screen = FakeScreen([key("w"), key("Left"), key("e", "n")])
    assert run_label(screen, folder, tmp_path / "labels.json") == 0
    assert on_disk(tmp_path)["frames"] == {
        "a.png": {"label": "empty_cup", "flavour": None},
        "b.png": {"label": "nothing", "flavour": None},
    }
    assert screen.lines[0][0] == "frame 1 of 3: a.png"
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [
        "a.png", "b.png", "c.png", "labels.json",
    ]
    assert capsys.readouterr().out == f"2 of 3 frames labelled in {tmp_path}/labels.json\n"


def test_main_counts_cups_through_the_window(tmp_path: Path) -> None:
    video = draw_video(tmp_path / "v.avi", 30)
    screen = FakeScreen([key("Right", "f"), key("Shift-Right", "e"), key("BackSpace", "f")])
    assert run_label(screen, video, tmp_path / "labels.json", mode="cups") == 0
    assert on_disk(tmp_path)["cups"] == [
        {"video": "v.avi", "order": 1, "state": "full", "frame": 1},
        {"video": "v.avi", "order": 2, "state": "full", "frame": 26},
    ]


def test_main_samples_a_video_with_every(tmp_path: Path) -> None:
    video = draw_video(tmp_path / "v.avi", 7)
    screen = FakeScreen([key("u", "h")])
    assert run_label(screen, video, tmp_path / "labels.json", every=3) == 0
    assert on_disk(tmp_path)["frames"] == {
        "v.avi#0": {"label": "unit", "flavour": None},
        "v.avi#3": {"label": "hand", "flavour": None},
    }


def test_q_stops_and_keeps_the_answers_pressed_before_it(tmp_path: Path) -> None:
    folder = draw_images(tmp_path / "shoot", "a.png", "b.png", "c.png")
    screen = FakeScreen([key("n", "q", "h")])
    assert run_label(screen, folder, tmp_path / "labels.json") == 0
    assert on_disk(tmp_path)["frames"] == {"a.png": {"label": "nothing", "flavour": None}}


def test_without_labels_the_file_goes_under_data_labels_of_the_working_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = draw_images(tmp_path / "shoot", "a.png")
    assert run_label(FakeScreen([key("n")]), folder, None) == 0
    assert (tmp_path / "data" / "labels" / "shoot.json").is_file()


def test_main_refuses_a_corrupt_labels_file_and_opens_no_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = draw_images(tmp_path / "shoot", "a.png")
    (tmp_path / "labels.json").write_text("{cut off", encoding="utf-8")
    screen = FakeScreen([key("n")])
    assert run_label(screen, folder, tmp_path / "labels.json") == 2
    assert screen.lines == []
    assert (tmp_path / "labels.json").read_text(encoding="utf-8") == "{cut off"
    assert "labels.json refused" in capsys.readouterr().out


def test_an_unreadable_frame_stops_main_naming_it_with_earlier_labels_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = draw_images(tmp_path / "shoot", "a.png")
    (folder / "b.png").write_bytes(b"not an image")
    screen = FakeScreen([key("n")])
    assert run_label(screen, folder, tmp_path / "labels.json") == 1
    assert on_disk(tmp_path)["frames"] == {"a.png": {"label": "nothing", "flavour": None}}
    assert "b.png" in capsys.readouterr().out


def test_a_failed_save_stops_main_with_the_previous_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def crash(source: object, target: object) -> None:
        raise OSError("disk gone")

    folder = draw_images(tmp_path / "shoot", "a.png", "b.png")
    (tmp_path / "labels.json").write_text(
        '{"version": 1, "source": "s", "cups": [], '
        '"frames": {"a.png": {"label": "hand", "flavour": null}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr("qs.labels.os.replace", crash)
    assert run_label(FakeScreen([key("n")]), folder, tmp_path / "labels.json") == 1
    assert on_disk(tmp_path)["frames"] == {"a.png": {"label": "hand", "flavour": None}}
    assert "disk gone" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("mode", "every", "is_folder", "says"),
    [
        pytest.param("frames", None, False, "--every", id="video-without-every"),
        pytest.param("frames", 2, True, "--every", id="folder-with-every"),
        pytest.param("cups", 2, False, "--every", id="cups-with-every"),
        pytest.param("cups", None, True, "needs a video", id="cups-on-a-folder"),
    ],
)
def test_arguments_that_do_not_fit_the_source_are_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    every: int | None,
    is_folder: bool,
    says: str,
) -> None:
    draw_images(tmp_path / "shoot", "a.png")
    draw_video(tmp_path / "v.avi", 3)
    source = tmp_path / ("shoot" if is_folder else "v.avi")
    screen = FakeScreen([])
    assert run_label(screen, source, tmp_path / "labels.json", mode=mode, every=every) == 2
    assert screen.lines == []
    assert says in capsys.readouterr().out
    assert not (tmp_path / "labels.json").exists()


def test_a_missing_source_is_refused_before_any_window(tmp_path: Path) -> None:
    labels = str(tmp_path / "labels.json")
    code = run(
        ["label", "--config", str(EXAMPLE), "--mode", "frames", "--every", "2",
         "--source", str(tmp_path / "missing.mp4"), "--labels", labels]
    )
    assert code == 2


@pytest.mark.parametrize(
    ("keysym", "state", "name"),
    [
        ("Right", 0x0, "Right"),
        ("Right", 0x1, "Shift-Right"),
        ("Left", 0x1, "Shift-Left"),
        ("f", 0x1, "f"),
        ("Right", 0x10, "Right"),
    ],
)
def test_only_a_shifted_arrow_gets_its_own_key_name(keysym: str, state: int, name: str) -> None:
    assert named_key(keysym, state) == name


@pytest.mark.parametrize(
    "cups",
    [
        pytest.param('[{"video": "a", "order": 1, "state": "half", "frame": 0}]', id="state"),
        pytest.param('[{"video": "a", "order": 2, "state": "full", "frame": 0}]', id="order-gap"),
        pytest.param('[{"video": "a", "order": 0, "state": "full", "frame": 0}]', id="order-zero"),
        pytest.param('[{"video": "a", "order": 1, "state": "full", "frame": -1}]', id="negative"),
        pytest.param('[{"video": "a", "order": 1, "state": "full", "frame": 1.5}]', id="float"),
        pytest.param('[{"video": "a", "order": 1, "state": "full", "frame": true}]', id="bool"),
        pytest.param('[{"video": "a", "order": 1, "state": "full"}]', id="no-frame"),
        pytest.param('[{"video": 1, "order": 1, "state": "full", "frame": 0}]', id="video-type"),
        pytest.param("[1]", id="not-an-object"),
    ],
)
def test_a_bad_cup_in_the_file_is_refused_and_left_untouched(tmp_path: Path, cups: str) -> None:
    path = tmp_path / "labels.json"
    text = f'{{"version": 1, "source": "s", "frames": {{}}, "cups": {cups}}}'
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LabelsError, match="labels.json"):
        load_labels(path, "shoot")
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize(
    ("keysym", "state"),
    [("f", "full"), ("w", "full_white"), ("p", "full_pink"), ("e", "empty")],
)
def test_each_cup_key_records_its_state(tmp_path: Path, keysym: str, state: str) -> None:
    cup_session(tmp_path).press(keysym)
    assert (tmp_path / "labels.json").is_file()
    assert on_disk(tmp_path)["cups"] == [
        {"video": "a.avi", "order": 1, "state": state, "frame": 0},
    ]


def test_the_loader_accepts_the_flavoured_cup_states(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        '{"version": 1, "source": "s", "frames": {}, "cups": ['
        '{"video": "a", "order": 1, "state": "full_white", "frame": 0}, '
        '{"video": "a", "order": 2, "state": "full_pink", "frame": 1}]}',
        encoding="utf-8",
    )
    assert [cup.state for cup in load_labels(path, "s").cups] == ["full_white", "full_pink"]


def test_the_cup_screen_and_summary_show_white_and_pink(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("f")
    session.press("Right")
    session.press("w")
    session.press("Right")
    session.press("p")
    session.press("Right")
    session.press("e")
    assert session.lines()[1:] == [
        "4 cups: 3 full (1 white, 1 pink), 1 empty · last: cup 4, empty, frame 3",
        "f full · w white · p pink · e empty · BackSpace undo · Left/Right 1 · Shift 25 · q stop",
    ]
    assert session.summary() == (
        f"4 cups in a.avi (3 full (1 white, 1 pink), 1 empty) in {tmp_path}/labels.json"
    )


def test_an_answer_before_the_last_cup_replaces_the_cup_on_that_frame(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("Right")
    session.press("f")
    session.press("Right")
    session.press("f")
    session.press("Left")
    session.press("p")
    assert on_disk(tmp_path)["cups"] == [
        {"video": "a.avi", "order": 1, "state": "full_pink", "frame": 1},
        {"video": "a.avi", "order": 2, "state": "full", "frame": 2},
    ]


def test_a_replacement_where_cups_share_a_frame_takes_the_highest_order(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("f")
    session.press("f")
    session.press("Right")
    session.press("f")
    session.press("Left")
    session.press("e")
    assert on_disk(tmp_path)["cups"] == [
        {"video": "a.avi", "order": 1, "state": "full", "frame": 0},
        {"video": "a.avi", "order": 2, "state": "empty", "frame": 0},
        {"video": "a.avi", "order": 3, "state": "full", "frame": 1},
    ]


def test_an_answer_on_the_last_cups_frame_records_the_next_cup(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("Right")
    session.press("f")
    session.press("e")
    assert on_disk(tmp_path)["cups"] == [
        {"video": "a.avi", "order": 1, "state": "full", "frame": 1},
        {"video": "a.avi", "order": 2, "state": "empty", "frame": 1},
    ]


def test_a_replacement_looks_only_at_this_video(tmp_path: Path) -> None:
    other = cup_session(tmp_path, "b.avi")
    other.press("f")
    session = cup_session(tmp_path, "a.avi")
    session.press("Right")
    session.press("f")
    session.press("Left")
    session.press("e")
    assert on_disk(tmp_path)["cups"] == [
        {"video": "b.avi", "order": 1, "state": "full", "frame": 0},
        {"video": "a.avi", "order": 1, "state": "full", "frame": 1},
    ]


def test_an_answer_before_the_last_cup_on_a_frame_with_no_cup_changes_nothing(
    tmp_path: Path,
) -> None:
    session = cup_session(tmp_path)
    session.press("Right")
    session.press("f")
    session.press("Left")
    before = (tmp_path / "labels.json").read_bytes()
    session.press("e")
    assert (tmp_path / "labels.json").read_bytes() == before
    assert session.labels.cups_of("a.avi")[-1].order == 1


def test_the_screen_says_when_an_answer_changed_nothing_until_the_next_key(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("Right")
    session.press("f")
    session.press("Left")
    session.press("e")
    assert session.lines()[2] == "nothing changed: no cup on this frame, and it is before the last"
    assert session.lines()[2] == "nothing changed: no cup on this frame, and it is before the last"
    session.press("Right")
    assert session.lines()[2] == (
        "f full · w white · p pink · e empty · BackSpace undo · Left/Right 1 · Shift 25 · q stop"
    )


def test_the_screen_shows_the_cup_on_the_current_frame(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("Right")
    session.press("f")
    session.press("Right")
    session.press("w")
    session.press("Left")
    assert session.lines()[0] == "frame 1 of 0–29: a.avi · this frame: cup 1, full"
    session.press("Left")
    assert session.lines()[0] == "frame 0 of 0–29: a.avi"


def test_the_screen_shows_every_cup_sharing_the_current_frame(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("f")
    session.press("w")
    assert session.lines()[0] == (
        "frame 0 of 0–29: a.avi · this frame: cup 1, full; cup 2, full_white"
    )


def test_a_failed_replacement_leaves_file_and_memory_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(source: object, target: object) -> None:
        raise OSError("disk gone")

    session = cup_session(tmp_path)
    session.press("f")
    session.press("Right")
    session.press("f")
    session.press("Left")
    before = (tmp_path / "labels.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr("qs.labels.os.replace", crash)
        with pytest.raises(OSError, match="disk gone"):
            session.press("e")
    assert (tmp_path / "labels.json").read_bytes() == before
    assert session.labels.cups_of("a.avi")[0].state == "full"
    session.press("e")
    assert on_disk(tmp_path)["cups"] == [
        {"video": "a.avi", "order": 1, "state": "empty", "frame": 0},
        {"video": "a.avi", "order": 2, "state": "full", "frame": 1},
    ]


def test_a_cup_frame_that_decreases_within_a_video_is_refused_and_left_untouched(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.json"
    text = (
        '{"version": 1, "source": "s", "frames": {}, "cups": ['
        '{"video": "a", "order": 1, "state": "full", "frame": 5}, '
        '{"video": "a", "order": 2, "state": "full", "frame": 4}]}'
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LabelsError, match="labels.json"):
        load_labels(path, "shoot")
    assert path.read_text(encoding="utf-8") == text


def test_cups_may_share_a_frame_and_each_video_has_its_own_frames(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        '{"version": 1, "source": "s", "frames": {}, "cups": ['
        '{"video": "a", "order": 1, "state": "full", "frame": 5}, '
        '{"video": "a", "order": 2, "state": "full", "frame": 5}, '
        '{"video": "b", "order": 1, "state": "full", "frame": 1}]}',
        encoding="utf-8",
    )
    assert [cup.frame for cup in load_labels(path, "s").cups] == [5, 5, 1]


def test_main_shows_that_an_answer_changed_nothing(tmp_path: Path) -> None:
    video = draw_video(tmp_path / "v.avi", 30)
    screen = FakeScreen([key("Right", "f", "Left", "e")])
    assert run_label(screen, video, tmp_path / "labels.json", mode="cups") == 0
    assert screen.lines[-1][2] == "nothing changed: no cup on this frame, and it is before the last"
    assert on_disk(tmp_path)["cups"] == [
        {"video": "v.avi", "order": 1, "state": "full", "frame": 1},
    ]
