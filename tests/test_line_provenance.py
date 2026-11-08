"""The line bundle records how it was split, and `qs line-alarms` checks it (#112).

Footage, labels and config are those of `tests/test_line_alarms.py`: one 10 fps video of 40 frames,
all labelled unit; with a 0.2 s gap the split trains on frames 0–27.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
import yaml

from conftest import FakePatchCore
from qs.__main__ import build_parser
from qs.cmd import line_alarms, train
from qs.errors import ConfigError
from qs.inspector.bundle import LineRecord, load_bundle
from test_bundle import _rehashed
from test_cup import drawn_frame
from test_line_alarms import _config, _footage, _labels


def _train(root: Path) -> Path:
    """Train on all 40 frames with a 0.2 s gap; the bundle's directory."""
    models = root / "models"
    args = build_parser().parse_args([
        "train", "--config", str(_config(root)), "--labels", str(_labels(root)),
        "--footage", str(_footage(root)), "--models", str(models), "--gap-s", "0.2",
        "--window-px", "40", "--input-px", "32", "--product", "mochi-top",
    ])
    assert train.main(args, make_patchcore=lambda device: FakePatchCore(),
                      cuda_available=lambda: True) == 0
    (bundle,) = models.iterdir()
    return bundle


def _alarms(root: Path, model_id: str, gap_s: str = "0.2") -> int:
    """`qs line-alarms` on the footage and labels `_train` wrote."""
    args = build_parser().parse_args([
        "line-alarms", "--config", str(root / "line.yml"), "--labels", str(root / "top.json"),
        "--footage", str(root / "footage"), "--models", str(root / "models"),
        "--model", model_id, "--gap-s", gap_s,
    ])
    return line_alarms.main(args, make_scorer=lambda device, bank: FakePatchCore().scorer(bank),
                            cuda_available=lambda: True)


def _rewrite(video: Path, pink: tuple[int, ...]) -> None:
    """Write the 40-frame video again under the same name, pink at the frames `pink` names."""
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (320, 240))
    assert writer.isOpened()
    for index in range(40):
        writer.write(drawn_frame("pink" if index in pink else "pale"))
    writer.release()


def test_a_line_bundle_records_the_gap_and_the_labels_file_it_was_split_with(
    tmp_path: Path,
) -> None:
    bundle = _train(tmp_path)

    data = yaml.safe_load((bundle / "bundle.yml").read_text(encoding="utf-8"))
    assert data.get("line") == {"gap_s": 0.2, "labels": "top.json"}


def test_the_loaded_bundle_carries_its_split(tmp_path: Path) -> None:
    bundle = _train(tmp_path)

    assert load_bundle(bundle.parent, bundle.name).line == LineRecord(0.2, "top.json")


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ({"gap_s": -1, "labels": "top.json"}, "line.gap_s: -1 is not a finite number at or"),
        ({"gap_s": True, "labels": "top.json"}, "line.gap_s: True is not a finite number"),
        ({"gap_s": 0.2, "labels": ""}, "line.labels: '' is not a file name"),
        ({"gap_s": 0.2}, "line: missing key 'labels'"),
    ],
)
def test_a_malformed_split_record_is_refused(
    tmp_path: Path, line: dict[str, object], match: str
) -> None:
    bundle = _train(tmp_path)
    edited = _rehashed(bundle.parent, bundle.name, {"line": line})

    with pytest.raises(ConfigError, match=match):
        load_bundle(bundle.parent, edited)


def test_a_gap_other_than_the_bundles_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A 0.3 s gap holds out frames 31–33 and 37–39: none trained on, yet the blocks move.
    bundle = _train(tmp_path)
    capsys.readouterr()

    assert _alarms(tmp_path, bundle.name, gap_s="0.3") == 1
    assert capsys.readouterr().out.strip() == (
        "refused: --gap-s 0.3 is not the 0.2 s the model was split with; use the gap it was "
        "trained with"
    )


def test_a_bundle_made_before_the_split_was_recorded_warns_and_still_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _train(tmp_path)
    older = _rehashed(bundle.parent, bundle.name, drop="line")
    capsys.readouterr()

    assert _alarms(tmp_path, older, gap_s="0.3") == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "WARNING: the bundle records no gap_s (made before #112), so --gap-s is not checked "
        "against training"
    )


def test_footage_whose_trained_frame_differs_from_the_bundles_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The same file name, but frame 3 — a training frame — now shows a pink моти.
    bundle = _train(tmp_path)
    _rewrite(tmp_path / "footage" / "line.avi", pink=(3, 36, 37))
    capsys.readouterr()

    assert _alarms(tmp_path, bundle.name) == 1
    assert capsys.readouterr().out.strip() == (
        "refused: line.avi#3: its pixels differ from the frame the model was trained on "
        "(SHA-256); re-encoded or renamed footage also does this — retrain on the footage used"
    )


def test_a_bundle_trained_on_image_files_not_video_frames_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _train(tmp_path)
    image = {"file": "u001_good_G2_r0.png", "unit": "line.avi", "sha256": "a" * 64}
    bench = _rehashed(bundle.parent, bundle.name, {"images": [image]})
    capsys.readouterr()

    assert _alarms(tmp_path, bench) == 1
    assert capsys.readouterr().out.strip() == (
        "refused: the bundle's training image 'u001_good_G2_r0.png' is not a video frame "
        "(<video file name>#<index>)"
    )


def test_labels_whose_training_block_is_not_the_bundles_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Relabelled with frame 39 gone: the span is 39 frames, so the split trains on 0–26, holds
    # out 29–32 and 35–38. No held-out frame was trained on, yet frame 29 sits two frames from
    # trained frame 27 and may show the same cup.
    bundle = _train(tmp_path)
    _labels(tmp_path, 39)
    capsys.readouterr()

    assert _alarms(tmp_path, bundle.name) == 1
    assert capsys.readouterr().out.strip() == (
        "refused: top.json: its training block is not the frames the model was trained on, so "
        "the labels differ from training; retrain, or use the labels it was trained with"
    )
