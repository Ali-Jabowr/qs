"""`qs line-alarms`: the line model's false alarms on held-out good frames (#110).

The drawn video, at 10 fps, 40 frames, every one labelled unit. With a 0.2 s gap (2 frames) the
time split of `qs.eval.line_split` trains on 0–27, drops 28–29, calibrates on 30–33, drops 34–35
and tests on 36–39. Every frame shows the same full pale cup at the same place, so the pale frames
are identical pixels and score the same, except frames 36 and 37, whose моти is pink. The model
is trained by `qs train` in the test with the numpy `FakePatchCore`, whose score is the distance
to the training windows' colours: a pink window scores far above a pale one. With 4 calibrate
frames and a 5% target no calibrate frame may lie above the threshold, so it is the pale score;
the two pink test frames are the false alarms: 2 of 4, and the one-sided 95% Clopper-Pearson
upper bound for 2 of 4 is 90.2%.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from conftest import FakePatchCore
from qs.__main__ import build_parser
from qs.cmd import line_alarms, train
from qs.inspector.model import Scorer
from station_fixtures import config_mapping
from test_appearance import ConstantScorer, Factory
from test_cup import drawn_frame

FPS = 10.0
VIDEO = "line.avi"
FRAMES = 40
PINK = (36, 37)


def _footage(root: Path) -> Path:
    footage = root / "footage"
    footage.mkdir()
    writer = cv2.VideoWriter(str(footage / VIDEO), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (320, 240))
    assert writer.isOpened()
    for index in range(FRAMES):
        writer.write(drawn_frame("pink" if index in PINK else "pale"))
    writer.release()
    return footage


def _labels(root: Path, count: int = FRAMES, name: str = "top.json") -> Path:
    frames = {f"{VIDEO}#{i}": {"label": "unit", "flavour": "white"} for i in range(count)}
    path = root / name
    path.write_text(json.dumps({"version": 1, "source": "line", "frames": frames, "cups": []}),
                    encoding="utf-8")
    return path


def _config(root: Path) -> Path:
    data = config_mapping(root, inspector_type="line")
    data["trigger"]["type"] = "cup"
    data["inspector"]["checks"] = ["empty_cup"]
    data["inspector"]["line"] = {
        "zone": [20, 0, 240, 240],
        "rim_axis_px": [50, 140],
        "blob_share_min": 0.3,
        "direction": "right",
        "fps": FPS,
    }
    path = root / "line.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _alarms(
    root: Path,
    scored_labels: int = FRAMES,
    make_scorer: Callable[[str, np.ndarray], Scorer] = lambda device, bank: FakePatchCore().scorer(
        bank
    ),
) -> int:
    """Train on all frames' labels, then count false alarms with the first `scored_labels`."""
    config, labels, footage = _config(root), _labels(root), _footage(root)
    models = root / "models"
    common = ["--config", str(config), "--footage", str(footage), "--models", str(models),
              "--gap-s", "0.2"]
    trained = train.main(
        build_parser().parse_args(
            ["train", *common, "--labels", str(labels), "--window-px", "40", "--input-px", "32",
             "--product", "mochi-top"]
        ),
        make_patchcore=lambda device: FakePatchCore(),
        cuda_available=lambda: True,
    )
    assert trained == 0
    (bundle,) = models.iterdir()
    scored = _labels(root, scored_labels, "scored.json")
    return line_alarms.main(
        build_parser().parse_args(
            ["line-alarms", *common, "--labels", str(scored), "--model", bundle.name]
        ),
        make_scorer=make_scorer,
        cuda_available=lambda: True,
    )


def test_false_alarms_are_counted_on_the_test_block_at_the_calibrate_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _alarms(tmp_path)

    out = capsys.readouterr().out
    assert code == 0
    assert "- threshold set on 4 calibrate frames at a 5% target, 0 allowed above it" in out
    assert "- false alarms on test frames: 2 of 4 = 50.0%, upper bound 90.2%" in out


def test_the_report_is_stamped_indicative_and_says_what_it_does_not_measure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _alarms(tmp_path) == 0

    out = capsys.readouterr().out
    assert "# INDICATIVE — false alarms of the line model on held-out good frames" in out
    assert ("Recall on real defects is unmeasured: no labelled defect exists, so this report "
            "counts false alarms only.") in out
    assert ("Frames, not cups: neighbouring frames often show the same cup, so they are not "
            "independent and the bound is tighter than the truth.") in out


def test_a_held_out_frame_the_model_was_trained_on_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Frames 0–19 only: the split calibrates on 16 and tests on 19, both trained on (0–27).
    assert _alarms(tmp_path, scored_labels=20) == 1

    assert capsys.readouterr().out.splitlines()[-1] == (
        "refused: line.avi#16: the model was trained on this held-out frame; "
        "use the labels and --gap-s it was trained with"
    )


@pytest.mark.parametrize("far", ["0", "100", "-5"])
def test_a_false_alarm_target_outside_1_5_10_is_refused_by_the_parser(far: str) -> None:
    # The same targets as `qs calibrate`: 100 would index past the scores, -5 would pick the
    # lowest calibrate score as the threshold.
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(
            ["line-alarms", "--config", "c.yml", "--labels", "l.json", "--footage", "f",
             "--model", "m", "--far", far]
        )

    assert raised.value.code == 2


def test_a_model_scoring_not_a_number_could_not_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _alarms(tmp_path, make_scorer=Factory(ConstantScorer(1.0, pred_score=float("nan"))))

    assert code == 2
    assert capsys.readouterr().out.splitlines()[-1] == (
        "line-alarms: the model's pred_score nan is not finite"
    )
