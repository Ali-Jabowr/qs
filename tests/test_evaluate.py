"""`qs evaluate` end to end on the synthetic lean session (docs/14 §4, §7, §9; ticket #82).

The dataset and golden set are built with #80's `plan_build` + `write_build`, the bundle is
written with P07's `write_bundle` naming the train units, and the calibration is #81's
`qs calibrate` — all reused unchanged. The scorer is a fake whose map is high on dark
pixels the window border cannot reach — a blot inside the unit — so a good unit scores
exactly 0.0, a blotted one 1.0, and every rate has a hand-countable expected value: dent
and tear recall caught by appearance, deformed caught by geometry, no false alarms. The
blot stays inside the silhouette (its grey is above the localize threshold), so the
geometry check passes on dented units and only the appearance check catches them.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from conftest import (
    BENCH_CAMERA,
    BLOT_RADIUS,
    BUILD_SESSION,
    CARD_BOX,
    FLOOR_GREY,
    PHONE_CAMERA,
    SMALL_H,
    SMALL_W,
    ShotRecord,
    bench_product,
)
from qs.__main__ import build_parser
from qs.cmd import calibrate, evaluate
from qs.config import LabelsConfig
from qs.contracts import CheckResult, Label
from qs.eval.golden import plan_build, write_build
from qs.eval.report import Report, RoleCount
from qs.eval.report import markdown as _markdown
from qs.eval.scoring import FrameResult
from qs.eval.units import (
    Aggregated,
    Pair,
    class_medians,
    control_passes,
    drift,
    latency,
    pairs,
)
from qs.inspector.model import Scored, Scorer
from qs.session import FrameLog, SessionRecord, save_frame, write_plan, write_session
from qs.splitplan import Plan, Shot, make_plan, shot_list
from test_calibrate import PLAN, PLAN_SEED, PRODUCT, write_train_bundle
from test_units import _result, _set

BLOT = 100  # the blot every scorer in this module flags (grey, above localize's 70)
DARK = -0.2  # the standard scorer's dark threshold in the normalised grey
WARN_DARK = 0.5  # the second scorer's: it also flags the LIGHT blot, never the LIGHTER one
LIGHT = 140  # a blot only the WARN_DARK scorer flags
LIGHTER = 170  # a blot no scorer in this module flags


_DOTS = [(dx, dy) for dy in range(-11, 12) for dx in range(-20, 21)
         if dx * dx + dy * dy <= 400][:60]


def draw_frame(
    unit: str, state: str, rotation: int, damage: str | None, blot: int = BLOT,
    radius: int = 26, dot: tuple[int, int] | None = None,
) -> np.ndarray:
    """One small synthetic frame: a pale disc on a black floor; a dent or tear as a blot on
    the disc (inside the silhouette); deformed-flat flattened, deformed-oval squeezed;
    `empty` a bare floor with the grey card; `edge` a disc over the inspection area's edge.
    `dot` marks the unit with a faint 1-px spot inside the disc — real photographs of
    different units never share bytes, and E1 compares the bundle's image hashes with the
    golden set's, so no two units' frames here may either."""
    paint_bgr = np.full((SMALL_H, SMALL_W, 3), FLOOR_GREY, dtype=np.uint8)
    paint_bgr[CARD_BOX[1]:CARD_BOX[1] + CARD_BOX[3],
              CARD_BOX[0]:CARD_BOX[0] + CARD_BOX[2]] = 158  # the card reads 0.62
    if state == "empty":
        return paint_bgr
    if state == "edge":
        cv2.circle(paint_bgr, (272, 120), 26, (200, 200, 200), -1)
        return paint_bgr
    centre = (150 + 6 * rotation, 116 - 4 * rotation)
    if state == "dmg" and damage == "deformed-flat":
        cv2.ellipse(paint_bgr, centre, (30, 15), 0, 0, 360, (200, 200, 200), -1)
    elif state == "dmg" and damage == "deformed-oval":
        cv2.ellipse(paint_bgr, centre, (22, 26), 0, 0, 360, (200, 200, 200), -1)
    else:
        cv2.circle(paint_bgr, centre, radius, (200, 200, 200), -1)
        if state == "dmg":
            cv2.circle(paint_bgr, (centre[0] + 7, centre[1] - 5), BLOT_RADIUS,
                       (blot, blot, blot), -1)
    if dot is not None:
        cv2.circle(paint_bgr, (centre[0] + dot[0], centre[1] + dot[1]), 1,
                   (190, 190, 190), -1)
    return paint_bgr


class BlotScorer:
    """The fake scorer: the map is 1 on dark pixels the window border cannot reach — a
    blot inside the unit — and a flat 0.01 elsewhere; the floor rim is border-connected
    and never scores. `dark` is the threshold in the normalised grey. The baseline keeps
    every score above 0, which the config loader requires of a threshold."""

    def __init__(self, bank: np.ndarray, dark: float = DARK) -> None:
        self.bank = bank
        self.dark = dark

    def score(self, window_input: np.ndarray) -> Scored:
        grey = window_input.mean(axis=0)  # the frames are grey: all three channels equal
        dark = (grey < self.dark).astype(np.uint8)
        count, labels = cv2.connectedComponents(dark, connectivity=4)
        border = set(labels[0].tolist()) | set(labels[-1].tolist())
        border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
        border.discard(0)
        inside = dark.astype(bool) & ~np.isin(labels, sorted(border))
        anomaly_map = inside.astype(np.float32)
        anomaly_map += np.float32(0.01)
        return Scored(anomaly_map, float(anomaly_map.max()), backbone_ms=1.0, search_ms=2.0)


def make_scorer(device: str, bank: np.ndarray, dark: float = DARK) -> Scorer:
    """The P07 seam with the blot scorer; `dark` selects which blots are flagged."""
    return BlotScorer(bank, dark)


def write_eval_session(
    root: Path,
    *,
    device: str = "/dev/video0",
    camera: object = BENCH_CAMERA,
    blot_of: object = None,
    oval_unit: str | None = None,
) -> tuple[Plan, Path]:
    """The lean dataset session as P05b writes it, drawn by this module: every good disc
    r26 except one train unit's r25 (the one variation that keeps the derived area range
    from being a single pixel), dents and tears as blots inside the silhouette.
    `oval_unit`'s good frames are ovals instead — (26, 34), (26, 34), (17, 52), as
    test_calibrate's `make_oval` — so geometry alone flags it."""
    plan = make_plan("lean", PLAN_SEED)
    folder = root / BUILD_SESSION
    write_plan(folder, plan, "sheet\n")
    record = ShotRecord(camera=camera)  # type: ignore[arg-type]
    index = {unit: i for i, unit in enumerate(
        (*plan.train, *plan.calib, *plan.test, *plan.control))}
    with FrameLog(folder) as log:
        for step in shot_list(plan, "G2"):
            if not isinstance(step, Shot):
                continue
            frame = step.frame
            damage = plan.damage.get(frame.unit)
            blot = BLOT
            if blot_of is not None and frame.state == "dmg" and damage in ("dent", "tear"):
                blot = blot_of(frame.unit)  # type: ignore[operator]
            if frame.state == "good" and frame.unit == plan.train[0]:
                pixels = draw_frame(frame.unit, frame.state, frame.rotation, damage,
                                    radius=25, dot=_DOTS[index[frame.unit]])
            elif frame.state == "good" and frame.unit == oval_unit:
                pixels = draw_frame(frame.unit, "empty", frame.rotation, None)
                axes = ((26, 34), (26, 34), (17, 52))[frame.rotation]
                cv2.ellipse(pixels, (150 + 6 * frame.rotation, 116 - 4 * frame.rotation),
                            axes, 0, 0, 360, (200, 200, 200), -1)
            else:
                pixels = draw_frame(frame.unit, frame.state, frame.rotation, damage, blot,
                                    dot=_DOTS[index[frame.unit]]
                                    if frame.unit in index else None)
            sha = save_frame(folder, frame, pixels)
            log.append_row(step.file, frame, record, None, None, sha, "saved")
    write_session(
        folder,
        SessionRecord(session_id=BUILD_SESSION, geometry="G2", product=bench_product(),
                      device=device, camera_at_open=camera, plan=plan, frames=(record,)),  # type: ignore[arg-type]
        by="qs import-phone",
    )
    return plan, folder


def run_calibrate(
    config: Path, dataset: Path, model_id: str, models: Path, out: Path, report_dir: Path,
    *, dark: float = DARK, far: int = 5,
) -> int:
    """`qs calibrate` through `main()` with the blot scorer and a fake CUDA probe."""
    args = build_parser().parse_args([
        "calibrate", "--config", str(config), "--dataset", str(dataset),
        "--model", model_id, "--out", str(out), "--models", str(models),
        "--report-dir", str(report_dir), "--far", str(far),
    ])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = calibrate.main(args, make_scorer=lambda d, b: make_scorer(d, b, dark),
                              cuda_available=lambda: True)
    assert "Traceback" not in buffer.getvalue(), buffer.getvalue()
    return code


def build_env(
    root: Path, config: Path, *, device: str = "/dev/video0", camera: object = BENCH_CAMERA,
    blot_of: object = None, oval_unit: str | None = None, far: int = 5,
) -> SimpleNamespace:
    """Session → build → bundle → calibrated config and record: everything #82 needs. The
    input config is the example's small-frame rewrite with the example's placeholder
    product replaced by the dataset's own."""
    text = config.read_text(encoding="utf-8").replace("product: mochi-<flavour>-v1",
                                                      f"product: {PRODUCT}")
    assert "product: mochi-vanilla" in text
    input_config = root / "input.yml"
    input_config.write_text(text, encoding="utf-8")
    _, session = write_eval_session(root, device=device, camera=camera, blot_of=blot_of,
                                    oval_unit=oval_unit)
    data = root / "data"
    build = plan_build(session, data, PRODUCT, None)
    write_build(build)
    models = data / "models"
    model_id = write_train_bundle(models, build.dataset.root)
    out = data / "station.yml"
    assert run_calibrate(input_config, build.dataset.root, model_id, models, out,
                         data / "reports" / "calib", far=far) == 0
    from qs.config import load_config

    threshold = load_config(out).config.threshold
    return SimpleNamespace(
        session=session, dataset=build.dataset.root, golden=build.golden.root,
        golden_hash=build.golden_hash, models=models, model_id=model_id, config=out,
        threshold_id=threshold.id,
        record=data / "reports" / "calib" / f"{threshold.id}.yml",
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory, bench_config: Path) -> SimpleNamespace:
    """The built, calibrated synthetic set, built once for the module."""
    return build_env(tmp_path_factory.mktemp("eval-env"), bench_config)


def fresh_reports(tmp_path: Path, env: SimpleNamespace) -> Path:
    """A report directory of this test's own, with the calibration record beside it —
    `qs evaluate` reads the record from the calib directory next to the report dir."""
    calib = tmp_path / "reports" / "calib"
    calib.mkdir(parents=True, exist_ok=True)
    shutil.copy(env.record, calib / env.record.name)
    return tmp_path / "reports" / "eval"


def run(
    tmp_path: Path, env: SimpleNamespace, *extra: str, config: Path | None = None,
    golden: Path | None = None, models: Path | None = None, dark: float = DARK,
    report_dir: Path | None = None, cuda: bool = True,
) -> tuple[int, str]:
    """`qs evaluate` through `main()`, its output captured."""
    directory = report_dir if report_dir is not None else fresh_reports(tmp_path, env)
    args = build_parser().parse_args([
        "evaluate", "--config", str(config or env.config),
        "--golden", str(golden or env.golden), "--models", str(models or env.models),
        "--report-dir", str(directory), *extra,
    ])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = evaluate.main(args, make_scorer=lambda d, b: make_scorer(d, b, dark),
                             cuda_available=lambda: cuda)
    return code, buffer.getvalue()


def report_of(directory: Path) -> Path:
    reports = sorted(directory.glob("*.md"))
    assert len(reports) == 1, sorted(directory.rglob("*"))
    return reports[0]


def refused(tmp_path: Path, env: SimpleNamespace, *parts: object,
            **kwargs: object) -> str:
    """One refusal: exit 1, every `part` in the output, no traceback, nothing written."""
    code, out = run(tmp_path, env, **kwargs)  # type: ignore[arg-type]
    assert code == 1, out
    for part in parts:
        assert str(part) in out, (part, out)
    assert "Traceback" not in out
    assert not any((tmp_path / "reports" / "eval").glob("*"))
    return out


@pytest.fixture(scope="module")
def first_report(env: SimpleNamespace, tmp_path_factory: pytest.TempPathFactory
                 ) -> SimpleNamespace:
    """The first evaluation of the synthetic set, run once for the module."""
    tmp = tmp_path_factory.mktemp("first-report")
    code, out = run(tmp, env)
    assert code == 0, out
    directory = tmp / "reports" / "eval"
    path = report_of(directory)
    return SimpleNamespace(
        code=code, out=out, path=path, directory=directory, text=path.read_text("utf-8"),
        csv=path.with_suffix(".csv").read_text("utf-8"),
    )


def _caught(text: str, which: str) -> list[str]:
    """The units the caught line attributes to `which` in the 5% target's row."""
    line = next(line for line in text.splitlines() if line.startswith("- caught — "))
    segment = line.split(f"{which}: ")[1].split(" · ")[0]
    return [] if segment == "—" else segment.split(", ")


def test_the_synthetic_set_evaluates(first_report: SimpleNamespace, env: SimpleNamespace
                                     ) -> None:
    """Exit 0; INDICATIVE in the title; the header carries the invocation, the golden hash,
    the model, the threshold and the config hash; the E6 sentence is there."""
    text = first_report.text
    assert first_report.path.name == f"{date.today().isoformat()}-mochi-vanilla-G2.md"
    assert text.startswith("# INDICATIVE — mochi-vanilla, G2, golden mvp-2026-09-25\n")
    assert f"- invocation: `qs evaluate --config {env.config} --golden {env.golden}" in text
    golden_hash = (env.golden / "manifest.sha256").read_text(encoding="utf-8").strip()
    assert f"- golden sha256: `{golden_hash}`" in text
    assert f"- model_id: `{env.model_id}`" in text
    assert f"- threshold_id: `{env.threshold_id}`" in text
    from qs.config import load_config

    assert f"- config_hash: `{load_config(env.config).config_hash}`" in text
    assert "this measures моти we broke on purpose" in text
    assert "not Gate A and not a client claim" in text
    assert "- source camera: the webcam (`/dev/video0`)" in text
    assert "## Stop lines" in text and "- none." in text
    assert "- calib: 10 units, 30 frames" in text
    assert "- test: 24 units, 144 frames" in text
    assert "- control: 3 units, 27 frames" in text
    assert "- placement: 10 units, 10 frames" in text
    assert "the roles are disjoint: every unit is in exactly one role" in text
    assert "- PASS — every placement frame came back НЕТ РЕШЕНИЯ(placement)." in text


def test_the_operating_table_is_the_hand_counted_one(first_report: SimpleNamespace) -> None:
    """Dent and tear recall caught by appearance, deformed by geometry, each with its
    bound; the FAR row with its upper bound; 1% and 5% share one threshold and say the
    target is below 1/n on the 10 calibration units."""
    text = first_report.text
    for target in (1, 5, 10):
        block = text.split(f"### far {target}% target")[1].split("###")[0]
        assert "dent 10/10 = 100.0%, lower bound 74.1%" in block, block
        assert "tear 7/7 = 100.0%, lower bound 65.2%" in block
        assert "deformed 7/7 = 100.0%, lower bound 65.2%" in block
        assert "combined 0/24 = 0.0%, upper bound 11.7%" in block
        assert "geometry-only 0/24, upper bound 11.7%" in block
        assert "appearance-only 0/24, upper bound 11.7%" in block
        assert "- good units flagged — none." in block
    assert text.count("the target is below 1/n on 10 calibration units") == 2
    first = text.split("### far 1% target")[1].split("###")[0]
    fifth = text.split("### far 5% target")[1].split("###")[0]
    threshold = "- threshold: 0.010000"
    assert threshold in first and threshold in fifth
    dents = sorted(u for u, cls in PLAN.damage.items() if cls == "dent")
    tears = sorted(u for u, cls in PLAN.damage.items() if cls == "tear")
    deformed = sorted(u for u, cls in PLAN.damage.items() if cls.startswith("deformed"))
    assert _caught(text, "appearance") == sorted(dents + tears)
    assert _caught(text, "geometry") == deformed
    assert _caught(text, "both") == [] and _caught(text, "missed") == []


def test_the_unscored_section_says_none(first_report: SimpleNamespace) -> None:
    assert "- none. Every unit and state was scored." in first_report.text


def test_the_csv_has_one_row_per_unit_and_state(first_report: SimpleNamespace,
                                                env: SimpleNamespace) -> None:
    """The columns in order; calib, test (both states) and control rows; no placement
    rows; a dent unit's damaged row carries its score, its geometry result and its label."""
    lines = first_report.csv.splitlines()
    assert lines[0] == ("golden_sha256,model_id,threshold_id,unit,role,class,state,"
                        "rotations_scored,score,geometry_failed,label_1,label_5,label_10")
    rows = [line.split(",") for line in lines[1:]]
    assert len(rows) == 10 + 24 * 2 + 3
    assert not any(row[3].startswith("x") for row in rows)
    golden_hash = (env.golden / "manifest.sha256").read_text(encoding="utf-8").strip()
    dent = sorted(u for u, cls in PLAN.damage.items() if cls == "dent")[0]
    [row] = [r for r in rows if r[3] == dent and r[6] == "dmg"]
    assert (row[0], row[1], row[2]) == (golden_hash, env.model_id, env.threshold_id)
    assert (row[4], row[5], row[6], row[7]) == ("test", "dent", "dmg", "3")
    assert row[8].startswith("1.0") and row[9] == "false"
    assert (row[10], row[11], row[12]) == ("БРАК", "БРАК", "БРАК")
    [good] = [r for r in rows if r[3] == dent and r[6] == "good"]
    assert abs(float(good[8]) - 0.01) < 1e-6 and good[9] == "false"
    assert (good[10], good[11], good[12]) == ("ГОДЕН", "ГОДЕН", "ГОДЕН")
    calib = next(r for r in rows if r[4] == "calib")
    assert calib[6] == "good" and calib[5] == "" and calib[7] == "3"
    control = next(r for r in rows if r[4] == "control")
    assert control[6] == "good" and control[7] == "9"  # three passes, three rotations


def test_a_csv_left_without_its_report_is_never_overwritten(
    tmp_path: Path, env: SimpleNamespace
) -> None:
    """A `<stem>.csv` whose `.md` was deleted still takes the name: the new report is
    `-2`, and the kept CSV is byte-identical (review P08 N2)."""
    directory = fresh_reports(tmp_path, env)
    directory.mkdir(parents=True)
    stem = f"{date.today().isoformat()}-{PRODUCT}-G2"
    kept = directory / f"{stem}.csv"
    kept.write_text("an orphaned per-unit CSV\n", encoding="utf-8")
    code, out = run(tmp_path, env, report_dir=directory)
    assert code == 0, out
    assert kept.read_text(encoding="utf-8") == "an orphaned per-unit CSV\n"
    assert not (directory / f"{stem}.md").exists()
    assert (directory / f"{stem}-2.md").is_file()
    assert (directory / f"{stem}-2.csv").is_file()


def test_a_second_run_the_same_day_writes_minus_2(tmp_path: Path, env: SimpleNamespace
                                                  ) -> None:
    """The first report is never overwritten; the second gets `-2`; and no file under the
    report directory is an image (NFR-07)."""
    directory = fresh_reports(tmp_path, env)
    assert run(tmp_path, env, report_dir=directory)[0] == 0
    first = report_of(directory)
    first_bytes = first.read_bytes()
    assert run(tmp_path, env, report_dir=directory)[0] == 0
    second = directory / f"{first.stem}-2.md"
    assert second.is_file() and second.with_suffix(".csv").is_file()
    assert first.read_bytes() == first_bytes
    images = [p for p in (tmp_path / "reports").rglob("*")
              if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]
    assert images == []


@pytest.fixture(scope="module")
def phone_env(tmp_path_factory: pytest.TempPathFactory, bench_config: Path
              ) -> SimpleNamespace:
    """The same set as the phone shot it (ADR-0022), calibrated with a phone config."""
    config = tmp_path_factory.mktemp("phone-config") / "phone.yml"
    config.write_text(
        bench_config.read_text(encoding="utf-8").replace("type: webcam", "type: phone"),
        encoding="utf-8",
    )
    return build_env(tmp_path_factory.mktemp("phone-env"), config,
                     device="phone:Xiaomi Redmi Note 12", camera=PHONE_CAMERA)


def test_a_phone_set_names_the_phone(tmp_path: Path, phone_env: SimpleNamespace) -> None:
    """The source camera is the phone, with the carry-over warning (ADR-0022)."""
    code, out = run(tmp_path, phone_env)
    assert code == 0, out
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    assert "- source camera: the phone (`phone:Xiaomi Redmi Note 12`)" in text
    assert "its numbers do not carry over to the webcam" in text


def test_a_webcam_config_on_a_phone_set_is_refused(tmp_path: Path, env: SimpleNamespace,
                                                   phone_env: SimpleNamespace) -> None:
    refused(tmp_path, env, "the phone", "camera.type is webcam", "ADR-0022",
            golden=phone_env.golden)


def _relog(session: Path, file: str) -> None:
    """Rewrite one frame and its `frames.csv` hash, so the build takes the new bytes."""
    import csv

    path = session / "frames.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    sha = hashlib.sha256((session / file).read_bytes()).hexdigest()
    for row in rows[1:]:
        if row[0] == file:
            row[15] = sha
    os.chmod(path, 0o644)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_the_placement_line_fails_on_a_frame_that_localizes(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """A placement frame that localizes instead of refusing: the line is FAIL naming the
    frame and what it got, the report is still written, and the exit is 1."""
    session = tmp_path / BUILD_SESSION
    shutil.copytree(env.session, session)
    file = "x01_empty_G2_r0.png"
    os.chmod(session / file, 0o644)
    assert cv2.imwrite(str(session / file), draw_frame("u001", "good", 0, None))
    _relog(session, file)
    build = plan_build(session, tmp_path / "data", PRODUCT, None)
    write_build(build)
    code, out = run(tmp_path, env, golden=build.golden.root)
    assert code == 1, out
    assert "Traceback" not in out
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    placement = text.split("## Placement")[1]
    assert "FAIL" in placement and file in placement and "came back pass" in placement


def test_a_golden_set_with_no_placement_frame_fails_placement(
    tmp_path: Path, env: SimpleNamespace
) -> None:
    """A hand-made golden set with every placement row and frame removed (and its hash
    made to match): no placement frame is no evidence, so the line is FAIL, never a
    vacuous PASS, and the exit is 1 (review P08 F3)."""
    import csv

    golden = tmp_path / "golden"
    shutil.copytree(env.golden, golden)
    with (golden / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    kept = [rows[0]] + [row for row in rows[1:] if row[2] != "placement"]
    assert len(kept) == len(rows) - 10
    for row in rows[1:]:
        if row[2] == "placement":
            (golden / "frames" / row[0]).unlink()
    manifest = golden / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(kept)
    (golden / "manifest.sha256").write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    code, out = run(tmp_path, env, golden=golden)
    assert code == 1, out
    assert "Traceback" not in out
    assert "placement: FAIL" in out
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    placement = text.split("## Placement")[1].split("\n## ")[0]
    assert "FAIL" in placement and "no placement frame" in placement
    assert "- PASS" not in placement


# --- the early warning (docs/06 §4) ----------------------------------------------------------

@pytest.fixture(scope="module")
def warn_env(tmp_path_factory: pytest.TempPathFactory, bench_config: Path
             ) -> SimpleNamespace:
    """The same set with six dents the standard scorer catches, one only the wider
    threshold catches, and three no scorer in this module catches."""
    dents = sorted(u for u, cls in PLAN.damage.items() if cls == "dent")

    def blot_of(unit: str) -> int:
        if unit not in dents:
            return BLOT
        i = dents.index(unit)
        return BLOT if i < 6 else LIGHT if i == 6 else LIGHTER

    return build_env(tmp_path_factory.mktemp("warn-env"), bench_config, blot_of=blot_of)


def test_the_early_warning_fires_when_four_dents_are_missed(
    tmp_path: Path, warn_env: SimpleNamespace
) -> None:
    """6 of 10 dents at the 5% target is 60%, below 70%: the stop line is printed and at
    the top of the report, and the exit is 1."""
    code, out = run(tmp_path, warn_env)
    assert code == 1, out
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if "EARLY WARNING" in line]
    assert len(lines) == 1
    line = lines[0]
    assert line in out or line[2:] in out  # printed as well as written
    assert "dent recall at the 5% target is 6 of 10 (60.0%), below 70%" in line
    assert "fix the lighting before building more of the demo" in line
    assert text.index(line) < text.index("## The set")  # at the top, under the title
    assert "dent 6/10 = 60.0%" in text


def test_the_early_warning_does_not_fire_at_seventy_percent(
    tmp_path: Path, warn_env: SimpleNamespace
) -> None:
    """7 of 10 dents is exactly 70% — not below it: no stop line, exit 0."""
    code, out = run(tmp_path, warn_env, dark=WARN_DARK)
    assert code == 0, out
    assert "EARLY WARNING" not in out
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    assert "EARLY WARNING" not in text
    assert "- none." in text.split("## Stop lines")[1].split("##")[0]
    assert "dent 7/10 = 70.0%, lower bound" in text


# --- A2 through main(): 5% not reachable, 10% reachable (#78, review F1) -------------------


@pytest.fixture(scope="module")
def oval_env(tmp_path_factory: pytest.TempPathFactory, bench_config: Path
             ) -> SimpleNamespace:
    """The same set with one oval calib unit, calibrated at `--far 10` — #78's way out."""
    return build_env(tmp_path_factory.mktemp("oval-env"), bench_config,
                     oval_unit=PLAN.calib[0], far=10)


def test_an_oval_calib_unit_at_far_10_evaluates_with_5_percent_not_reachable(
    tmp_path: Path, oval_env: SimpleNamespace
) -> None:
    """One geometry-flagged calib unit: the 1% and 5% rows are not reachable, the 10% row
    is. No traceback; the 5% summary line and the early warning say "not reachable" /
    "not measurable" and never state a recall; the report is written and the exit is 1."""
    code, out = run(tmp_path, oval_env)
    assert "Traceback" not in out and "Error" not in out, out
    assert code == 1, out
    assert "far 5%: not reachable" in out, out
    assert "geometry alone flags 1 of 10 calibration units" in out
    warning = next(line for line in out.splitlines() if "EARLY WARNING" in line)
    assert "not measurable" in warning, warning
    assert "of 10 (" not in warning and "%)" not in warning, warning
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    assert warning in text
    at5 = text.split("### far 5% target")[1].split("###")[0]
    assert "not reachable" in at5 and "- recall per unit" not in at5
    at10 = text.split("### far 10% target")[1].split("\n## ")[0]
    assert "- threshold: 0.010000" in at10 and "dent 10/10" in at10


# --- the refusals: exit 1, nothing written ---------------------------------------------------


def test_a_changed_frame_is_refused(tmp_path: Path, env: SimpleNamespace) -> None:
    golden = tmp_path / "golden"
    shutil.copytree(env.golden, golden)
    file = sorted((golden / "frames").glob("*.png"))[0]
    os.chmod(file, 0o644)
    raw = bytearray(file.read_bytes())
    raw[100] ^= 0xFF
    file.write_bytes(bytes(raw))
    refused(tmp_path, env, file.name, "unfrozen set", golden=golden)


def test_a_bundle_with_a_golden_test_unit_is_refused(
    tmp_path: Path, env: SimpleNamespace
) -> None:
    from qs.inspector.bundle import BundleInfo, ImageRecord, write_bundle
    from qs.names import parse_name

    test_file = sorted(row["file"] for row in
                       ({"file": p.name} for p in (env.golden / "frames").glob("*.png"))
                       if parse_name(row["file"]).unit.startswith("u")
                       and parse_name(row["file"]).state == "dmg")[0]
    images = [
        ImageRecord(path.name, parse_name(path.name).unit,
                    hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted((env.dataset / "train" / "good").glob("*.png"))
    ]
    frame = env.golden / "frames" / test_file
    images.append(ImageRecord(test_file, parse_name(test_file).unit,
                              hashlib.sha256(frame.read_bytes()).hexdigest()))
    info = BundleInfo(input_px=32, window_px=64, coreset_ratio=0.1, seed=42, device="cuda",
                      data_source="own", images=tuple(images))
    models = tmp_path / "models"
    model = write_bundle(models, PRODUCT, np.full((4, 3), 0.5, np.float32), info)
    config = tmp_path / "naming-the-bundle.yml"
    config.write_text(env.config.read_text(encoding="utf-8").replace(env.model_id, model),
                      encoding="utf-8")
    refused(tmp_path, env, parse_name(test_file).unit, "a golden unit", "E1",
            config=config, models=models)


def test_a_threshold_off_by_a_cent_is_refused(tmp_path: Path, env: SimpleNamespace) -> None:
    """The recomputation disagrees: the config's threshold is not what this set gives."""
    config = tmp_path / "off.yml"
    text = env.config.read_text(encoding="utf-8")
    assert "value: 0.009999999776482582" in text
    config.write_text(text.replace("value: 0.009999999776482582", "value: 0.02"),
                      encoding="utf-8")
    refused(tmp_path, env, "the threshold recomputed on this set's calib units is 0.010000",
            "the scores changed", config=config)


def test_a_missing_calibration_record_is_refused(tmp_path: Path, env: SimpleNamespace
                                                 ) -> None:
    directory = tmp_path / "reports" / "eval"
    code, out = run(tmp_path, env, report_dir=directory)
    assert code == 1, out
    assert "no calibration record" in out and "never used without its record" in out
    assert "Traceback" not in out
    assert not directory.exists()


def test_a_hand_edited_calibration_record_is_refused(tmp_path: Path, env: SimpleNamespace
                                                     ) -> None:
    """The record's canonical hash must give the threshold_id's suffix: a record edited
    after calibration is refused before anything is written (review P08 N1)."""
    import yaml

    fresh_reports(tmp_path, env)
    record = tmp_path / "reports" / "calib" / env.record.name
    data = yaml.safe_load(record.read_text(encoding="utf-8"))
    data["a"] = data["a"] + 1
    record.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                      encoding="utf-8")
    refused(tmp_path, env, env.threshold_id[-6:], "record was edited",
            report_dir=tmp_path / "reports" / "eval")


def test_a_no_verdict_calib_frame_is_refused(tmp_path: Path, env: SimpleNamespace) -> None:
    """A calib frame out of the lighting band: the threshold would be set on fewer units
    than it claims — refused before anything is written."""
    session = tmp_path / BUILD_SESSION
    shutil.copytree(env.session, session)
    unit = sorted(PLAN.calib)[0]
    file = f"{unit}_good_G2_r0.png"
    path = session / file
    pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert pixels is not None
    x, y, w, h = CARD_BOX
    pixels[y:y + h, x:x + w] = 100  # 0.39, outside 0.62 ± 5 %
    os.chmod(path, 0o644)
    assert cv2.imwrite(str(path), pixels)
    _relog(session, file)
    build = plan_build(session, tmp_path / "data", PRODUCT, None)
    write_build(build)
    refused(tmp_path, env, file, "НЕТ РЕШЕНИЯ (lighting)",
            "never set on fewer units", golden=build.golden.root)


# --- could not run: exit 2, one line, no traceback -------------------------------------------


def _one_line(code: int, out: str, tmp_path: Path) -> None:
    assert code == 2, out
    assert len(out.strip().splitlines()) == 1, out
    assert "Traceback" not in out
    assert not any((tmp_path / "reports").rglob("*.md"))


def test_an_unloadable_config_cannot_run(tmp_path: Path, env: SimpleNamespace) -> None:
    config = tmp_path / "garbage.yml"
    config.write_text("station_id: 7\n", encoding="utf-8")
    code, out = run(tmp_path, env, config=config)
    _one_line(code, out, tmp_path)
    assert "config:" in out


def test_torch_cuda_without_cuda_cannot_run(tmp_path: Path, env: SimpleNamespace) -> None:
    code, out = run(tmp_path, env, cuda=False)
    _one_line(code, out, tmp_path)
    assert "CUDA is not available" in out


def test_a_missing_golden_set_cannot_run(tmp_path: Path, env: SimpleNamespace) -> None:
    code, out = run(tmp_path, env, golden=tmp_path / "nope")
    _one_line(code, out, tmp_path)
    assert "nope" in out


# --- the CLI ---------------------------------------------------------------------------------


def test_the_cli_lists_evaluate() -> None:
    parser = build_parser()
    assert "evaluate" in parser.format_help()
    parsed = parser.parse_args(["evaluate", "--config", "c", "--golden", "g"])
    assert (parsed.models, parsed.report_dir) == ("models", "reports/eval")
    parsed = parser.parse_args(["evaluate", "--config", "c", "--golden", "g",
                                "--models", "m2", "--report-dir", "r2"])
    assert (parsed.models, parsed.report_dir) == ("m2", "r2")


# --- A2 at the report: a target geometry alone cannot meet -----------------------------------


def test_a_not_reachable_target_says_so_and_has_no_rates() -> None:
    """A2/#78: the row says not reachable, with no recall and no false-alarm rate."""
    from qs.config import LabelsConfig
    from qs.eval.report import Report, RoleCount
    from qs.eval.threshold import ThresholdChoice
    from qs.eval.units import TargetResult

    choice = ThresholdChoice(value=None, n=10, a=0, g=1, k=1, below_one_over_n=True,
                             flagged=("u004",))
    target = TargetResult(
        target=1, choice=choice, recall=(),
        far_combined=None,  # type: ignore[arg-type]
        far_geometry=None, far_appearance=None, far_unscored=0,
        caught=(), flagged_good=(),
    )
    text = Report(
        product="mochi-vanilla", geometry="G2", golden_name="mvp-2026-09-25",
        golden_hash="0" * 64, model_id="m", threshold_id="t", config_hash="c",
        invocation="qs evaluate", camera_device="/dev/video0", camera_is_phone=False,
        counts=(RoleCount("calib", 10, 30, ()),),
        targets=(target,), units=(), placement_ok=True, placement_bad=(),
        early_warning=None,
        labels=LabelsConfig("ГОДЕН", "БРАК", "НЕТ РЕШЕНИЯ"),
    )
    markdown = _markdown(text)
    assert "not reachable: geometry alone flags 1 of 10 calibration units, the target " \
        "allows 0 (A2, #78)" in markdown
    block = markdown.split("### far 1% target")[1].split("\n## ")[0]
    assert "- recall per unit" not in block and "- false alarms" not in block


# --- ticket #83 on hand-built results: controls, drift, pairs, latency ----------------------

LABELS = LabelsConfig("ГОДЕН", "БРАК", "НЕТ РЕШЕНИЯ")
MEASURED = CheckResult("geometry", True, None, None,
                       (("area_px", 2000.0), ("diameter_px", 50.0), ("axis_ratio", 0.98),
                        ("solidity", 0.99)))


def _control(unit: str, pass_: int, scores: tuple[float | None, ...],
             areas: tuple[float, ...] = (2000.0, 2000.0, 2000.0)) -> list[FrameResult]:
    """One control pass: a score per rotation, `None` for НЕТ РЕШЕНИЯ(lighting)."""
    out = []
    for r, (score, area) in enumerate(zip(scores, areas, strict=True)):
        file = f"{unit}_good_G2_r{r}_pass{pass_}.png"
        if score is None:
            out.append(_result(file, label=Label.NO_VERDICT, score=None, geometry=None,
                               reason="lighting"))
        else:
            geometry = CheckResult("geometry", True, None, None,
                                   (("area_px", area), ("diameter_px", 50.0),
                                    ("axis_ratio", 0.98), ("solidity", 0.99)))
            out.append(_result(file, score=score, geometry=geometry))
    return out


def _steady_controls() -> list[tuple[str, str, str, str, list[FrameResult]]]:
    """Three controls whose pass scores are 0.25, 0.375, 0.5 — a shift of exactly 0.25
    (binary fractions, so every comparison below is exact)."""
    return [(unit, "control", "good", "",
             _control(unit, p, (0.125, 0.125 * (p + 1), 0.125)))
            for unit in ("c01", "c02", "c03") for p in (1, 2, 3)]


def _dents(*diffs: float) -> tuple[Pair, ...]:
    return tuple(Pair(f"u{i + 1:03d}", "dent", 0.0, d) for i, d in enumerate(diffs))


def _report(**fields: object) -> str:
    base: dict[str, object] = dict(
        product="mochi-vanilla", geometry="G2", golden_name="mvp-2026-09-25",
        golden_hash="0" * 64, model_id="m", threshold_id="t", config_hash="c",
        invocation="qs evaluate", camera_device="/dev/video0", camera_is_phone=False,
        counts=(RoleCount("calib", 10, 30, ()),), targets=(), units=(), placement_ok=True,
        placement_bad=(), early_warning=None, labels=LABELS,
    )
    base.update(fields)
    return _markdown(Report(**base))  # type: ignore[arg-type]


def test_a_control_pass_all_no_verdict_is_shown_and_left_out_of_the_range() -> None:
    """c03's pass 3 came back НЕТ РЕШЕНИЯ on every rotation: the table says so, and the
    shift is the range over passes 1 and 2 only (A3, #79)."""
    entries = [e for e in _steady_controls() if e[0] != "c03"]
    entries += [("c03", "control", "good", "", _control("c03", 1, (0.25, 0.25, 0.25))),
                ("c03", "control", "good", "", _control("c03", 2, (0.75, 0.5, 0.25))),
                ("c03", "control", "good", "", _control("c03", 3, (None, None, None)))]
    controls = control_passes(*_set(*entries))
    assert [(c.unit, c.pass_) for c in controls] == [
        (u, p) for u in ("c01", "c02", "c03") for p in (1, 2, 3)]
    last = controls[-1]
    assert (last.scored, last.score, last.measures, last.reasons) == (0, None, (), ("lighting",))
    result = drift(controls, _dents(1.0, 1.0, 1.0))
    c03 = next(s for s in result.controls if s.unit == "c03")
    assert c03.passes_scored == 2 and c03.shift == 0.5  # 0.75 − 0.25, pass 3 left out
    assert result.shift_unit == "c03" and result.gap == pytest.approx(1.0)
    assert not result.drifted and result.line() is None
    text = _report(controls=controls, drift=result)
    table = text.split("## Controls")[1].split("\n## ")[0]
    assert "| c03 | pass3 | 0 | НЕТ РЕШЕНИЯ (lighting) | — | — | — | — | — |" in table
    assert "| c03 | pass2 | 3 | 0.750000 | passed | 2000.0 | 50.00 | 0.9800 | 0.9900 |" in table
    assert "the session drifted" not in text


def test_the_measures_are_the_mean_over_the_scored_rotations() -> None:
    controls = control_passes(*_set(
        ("c01", "control", "good", "", _control("c01", 1, (0.1, 0.2, None),
                                                 areas=(2000.0, 2100.0, 0.0)))))
    assert dict(controls[0].measures)["area_px"] == 2050.0
    assert controls[0].scored == 2 and controls[0].score == 0.2


def test_a_control_with_one_scored_pass_is_drift() -> None:
    entries = [e for e in _steady_controls() if e[0] != "c02"]
    entries += [("c02", "control", "good", "", _control("c02", 1, (0.1, 0.1, 0.1)))]
    entries += [("c02", "control", "good", "", _control("c02", p, (None, None, None)))
                for p in (2, 3)]
    result = drift(control_passes(*_set(*entries)), _dents(1.0))
    assert result.drifted
    line = result.line()
    assert line is not None and line.startswith(
        "the session drifted: the numbers mean nothing (E9)")
    assert "c02's shift is not measurable" in line


@pytest.mark.parametrize("gap", [0.0, -0.05])
def test_a_gap_at_or_below_zero_is_drift(gap: float) -> None:
    result = drift(control_passes(*_set(*_steady_controls())), _dents(gap, gap, gap))
    assert result.drifted and result.gap == pytest.approx(gap)
    assert "is not above 0" in (result.line() or "")


def test_a_shift_at_the_gap_is_drift_and_below_it_is_not() -> None:
    """Steady controls shift by 0.25: a median gap of 0.25 drifts (shift ≥ gap), 0.375
    does not; the line names the control and both numbers."""
    controls = control_passes(*_set(*_steady_controls()))
    below = drift(controls, _dents(0.25, 0.375, 0.5))
    assert (below.shift, below.gap, below.gap_units, below.drifted) == (0.25, 0.375, 3, False)
    at = drift(controls, _dents(0.125, 0.25, 0.25, 0.5))
    assert (at.shift, at.gap, at.drifted) == (0.25, 0.25, True)
    assert "the shift 0.250000 (c01) is not below the gap 0.250000" in (at.line() or "")


def test_no_dent_scored_in_both_states_is_drift() -> None:
    result = drift(control_passes(*_set(*_steady_controls())),
                   (Pair("u001", "dent", None, 1.0), Pair("u002", "tear", 0.1, 1.0)))
    assert result.drifted and result.gap is None
    assert "the gap is not measurable" in (result.line() or "")


def _agg(unit: str, state: str, cls: str, score: float | None) -> Aggregated:
    return Aggregated(unit, "test", state, cls, 0 if score is None else 3, score,
                      None if score is None else False, () if score is not None
                      else ("lighting",))


def test_pairs_are_damaged_minus_good_with_the_median_per_class() -> None:
    units = (
        Aggregated("u001", "calib", "good", "", 3, 0.5, False, ()),
        _agg("u010", "good", "dent", 0.10), _agg("u010", "dmg", "dent", 0.90),
        _agg("u011", "good", "dent", 0.20), _agg("u011", "dmg", "dent", 0.50),
        _agg("u012", "good", "dent", 0.10), _agg("u012", "dmg", "dent", None),
        _agg("u013", "good", "dent", 0.30), _agg("u013", "dmg", "dent", 0.40),
        _agg("u020", "good", "tear", None), _agg("u020", "dmg", "tear", 1.00),
        _agg("u030", "good", "deformed", 0.10), _agg("u030", "dmg", "deformed", 0.10),
        _agg("u031", "good", "deformed", 0.20), _agg("u031", "dmg", "deformed", 0.60),
    )
    paired = pairs(units)
    assert [p.unit for p in paired] == ["u010", "u011", "u012", "u013", "u020", "u030",
                                        "u031"]
    diffs = {p.unit: p.difference for p in paired}
    assert diffs["u010"] == pytest.approx(0.80) and diffs["u011"] == pytest.approx(0.30)
    assert diffs["u013"] == pytest.approx(0.10) and diffs["u031"] == pytest.approx(0.40)
    assert diffs["u012"] is None and diffs["u020"] is None
    medians = {cls: (m, n) for cls, m, n in class_medians(paired)}
    assert medians["dent"][0] == pytest.approx(0.30) and medians["dent"][1] == 3
    assert medians["tear"] == (None, 0)
    assert medians["deformed"][0] == pytest.approx(0.20) and medians["deformed"][1] == 2
    text = _report(pairs=paired)
    section = text.split("## Pairs")[1].split("\n## ")[0]
    assert "| u010 | dent | 0.100000 | 0.900000 | 0.800000 |" in section
    assert "| u012 | dent | 0.100000 | — | — |" in section
    assert "| u020 | tear | — | 1.000000 | — |" in section
    assert "dent 0.300000 (3 units)" in section
    assert "tear — (0 units)" in section
    assert "deformed 0.200000 (2 units)" in section


def test_latency_is_the_percentiles_of_the_frames_that_reached_the_model() -> None:
    """Verdicts and НЕТ РЕШЕНИЯ(inference) count; a placement fault (refused before the
    model, as `qs latency` leaves it out) is counted apart; one decided before the
    Inspector (lighting) is neither. p50/p95/p99 by numpy's linear method, by hand."""
    def timed(file: str, inspect: float, total: float, **kwargs: object) -> FrameResult:
        base = _result(file, **kwargs)  # type: ignore[arg-type]
        return FrameResult(base.file, base.name, base.label, base.reason, base.geometry,
                           base.score, base.pred_score, inspect, total)

    results = [timed(f"u001_good_G2_r{r}.png", 10.0 * (r + 1), 20.0 * (r + 1))
               for r in range(3)]
    results.append(timed("u001_dmg_G2_r0.png", 40.0, 80.0, label=Label.NO_VERDICT,
                         score=None, geometry=None, reason="inference"))
    results.append(timed("x01_empty_G2_r0.png", 1.0, 2.0, label=Label.NO_VERDICT,
                         score=None, geometry=None, reason="placement"))
    results.append(timed("u002_good_G2_r0.png", 0.0, 500.0, label=Label.NO_VERDICT,
                         score=None, geometry=None, reason="lighting"))
    got = latency(results)
    assert (got.count, got.placement) == (4, 1)
    # inspect 10, 20, 30, 40: p50 = 25; p95 = 30 + 0.85 × 10 = 38.5; p99 = 30 + 0.97 × 10
    assert got.inspect_ms == pytest.approx((25.0, 38.5, 39.7))
    assert got.total_ms == pytest.approx((50.0, 77.0, 79.4))
    assert latency(results[-2:]) == type(got)(0, None, None, 1)


# --- ticket #83 through `qs evaluate`: controls, drift, latency, changed units --------------


def test_the_controls_table_is_three_by_three_with_no_drift(first_report: SimpleNamespace
                                                           ) -> None:
    """c01–c03 × pass1–3, each with its score and measures; steady controls, no drift line."""
    table = first_report.text.split("## Controls (E9)")[1].split("\n## ")[0]
    rows = [line for line in table.splitlines() if line.startswith("| c0")]
    assert [tuple(line.split(" | ")[:2]) for line in rows] == [
        (f"| c0{u}", f"pass{p}") for u in (1, 2, 3) for p in (1, 2, 3)]
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[2] == "3" and cells[3] == "0.010000" and cells[4] == "passed"
        assert all(cell not in ("", "—") for cell in cells[5:9])
    assert "- drift: none — the largest shift 0.000000" in table
    assert "gap: the median paired difference of the dent units, 1.000000 over 10 units" \
        in table
    assert "the session drifted" not in first_report.text


def test_a_blotted_control_at_pass_3_drifts(tmp_path: Path, env: SimpleNamespace) -> None:
    """c02's pass-3 frames carry a blot the fake scorer flags: its shift (1.0) reaches the
    dent gap (1.0), so the drift line is at the top, printed, and the exit is 1 — with the
    report still written."""
    session = tmp_path / BUILD_SESSION
    shutil.copytree(env.session, session)
    for rotation in range(3):
        file = f"c02_good_G2_r{rotation}_pass3.png"
        pixels = cv2.imread(str(session / file), cv2.IMREAD_COLOR)
        assert pixels is not None
        centre = (150 + 6 * rotation, 116 - 4 * rotation)
        cv2.circle(pixels, (centre[0] + 7, centre[1] - 5), BLOT_RADIUS, (BLOT, BLOT, BLOT), -1)
        os.chmod(session / file, 0o644)
        assert cv2.imwrite(str(session / file), pixels)
        _relog(session, file)
    build = plan_build(session, tmp_path / "data", PRODUCT, None)
    write_build(build)
    code, out = run(tmp_path, env, golden=build.golden.root)
    assert code == 1, out
    assert "Traceback" not in out
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    [line] = [line for line in text.splitlines() if line.startswith("- the session drifted")]
    assert "the numbers mean nothing (E9)" in line
    assert "the shift 1.000000 (c02) is not below the gap 1.000000" in line
    assert text.index(line) < text.index("## The set")  # at the top, under the title
    assert line[2:] in out
    assert "| c02 | pass3 | 3 | 1.010000 | passed |" in text
    assert "- c02: shift 1.000000 over 3 scored passes" in text


def test_latency_is_computed_on_the_runs_own_results(
    tmp_path: Path, env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The p50/p95/p99 in the report are numpy's, over the frames the run itself scored that
    reached the model — the 10 placement frames left out and counted; with no
    `qs latency --cpu` report the command is named."""
    seen: list[tuple[FrameResult, ...]] = []
    original = evaluate.score_frames

    def recording(*args: object, **kwargs: object) -> tuple[FrameResult, ...]:
        results = original(*args, **kwargs)  # type: ignore[arg-type]
        seen.append(results)
        return results

    monkeypatch.setattr(evaluate, "score_frames", recording)
    code, out = run(tmp_path, env)
    assert code == 0, out
    [results] = seen
    reached = [r for r in results if r.label is not Label.NO_VERDICT]
    assert (len(results), len(reached)) == (211, 201)  # every frame but the 10 placement
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    section = text.split("## Latency — measured, not reproducible")[1]
    assert ("- backend `torch-cuda`, 201 frames that reached the model (10 placement-fault "
            "frames left out: they never reach the model, as in `qs latency`)") in section
    for name, values in (("inspect_ms", [r.inspect_ms for r in reached]),
                         ("total_ms", [r.total_ms for r in reached])):
        expected = " | ".join(f"{float(np.percentile(values, q)):.1f}" for q in (50, 95, 99))
        assert f"| {name} | {expected} |" in section
    assert (f"not measured yet — `bin/qs latency --config {env.config} "
            f"--frames {env.golden / 'frames'} --cpu`") in section
    assert text.rstrip().endswith("- nothing is gated here.")  # latency is the last section


def test_a_cpu_latency_report_of_this_model_is_named(tmp_path: Path, env: SimpleNamespace
                                                     ) -> None:
    """The newest `reports/latency/*-<model_id>-cpu.md` is named; a CUDA report and another
    model's CPU report are not."""
    latency_dir = tmp_path / "reports" / "latency"
    latency_dir.mkdir(parents=True)
    for name in (f"2026-09-26-{env.model_id}-cpu.md", f"2026-09-27-{env.model_id}-cpu.md",
                 f"2026-09-28-{env.model_id}-cuda.md", "2026-09-29-mochi-other-cpu.md"):
        (latency_dir / name).write_text("# Latency\n", encoding="utf-8")
    code, out = run(tmp_path, env)
    assert code == 0, out
    text = report_of(tmp_path / "reports" / "eval").read_text(encoding="utf-8")
    newest = latency_dir / f"2026-09-27-{env.model_id}-cpu.md"
    assert f"- Gate B proxy (ADR-0017): `{newest}`" in text
    assert "not measured yet" not in text


def _changed(text: str) -> str:
    return text.split("## Changed units")[1].split("\n## ")[0].strip()


def test_changed_units_list_exactly_the_unit_that_moved(
    tmp_path: Path, warn_env: SimpleNamespace
) -> None:
    """Run 1 catches the light dent (the wider scorer), run 2 does not: run 2 lists exactly
    that unit's damaged state at 1%, 5% and 10%, БРАК → ГОДЕН with both scores, under
    run 1's model, threshold and file; run 3 finds the newest earlier run (run 2's) and
    says no unit changed."""
    directory = fresh_reports(tmp_path, warn_env)
    code, out = run(tmp_path, warn_env, dark=WARN_DARK, report_dir=directory)
    assert code == 0, out
    first = report_of(directory)
    assert _changed(first.read_text(encoding="utf-8")) == \
        "- no previous run on this golden set."
    run(tmp_path, warn_env, report_dir=directory)
    second = directory / f"{first.stem}-2.md"
    moved = sorted(u for u, cls in PLAN.damage.items() if cls == "dent")[6]
    expected = "\n".join(
        [f"- since `{first.stem}.csv` (model_id `{warn_env.model_id}`, threshold_id "
         f"`{warn_env.threshold_id}`):"]
        + [f"  - {moved} dmg at {target}%: БРАК → ГОДЕН, score 1.010000 → 0.010000"
           for target in (1, 5, 10)])
    assert _changed(second.read_text(encoding="utf-8")) == expected
    run(tmp_path, warn_env, report_dir=directory)
    third = directory / f"{first.stem}-3.md"
    assert _changed(third.read_text(encoding="utf-8")) == (
        f"- no unit changed since `{first.stem}-2.csv` (model_id `{warn_env.model_id}`, "
        f"threshold_id `{warn_env.threshold_id}`).")


def test_a_csv_from_another_golden_set_is_ignored(
    tmp_path: Path, env: SimpleNamespace, first_report: SimpleNamespace
) -> None:
    """A per-unit CSV with another golden hash — every label flipped — is not a previous
    run on this set."""
    directory = fresh_reports(tmp_path, env)
    directory.mkdir(parents=True)
    other = first_report.csv.replace(env.golden_hash, "f" * 64).replace("ГОДЕН", "БРАК")
    assert "f" * 64 in other
    (directory / "2026-09-20-mochi-vanilla-G2.csv").write_text(other, encoding="utf-8")
    code, out = run(tmp_path, env, report_dir=directory)
    assert code == 0, out
    report = next(p for p in directory.glob("*.md"))
    assert _changed(report.read_text(encoding="utf-8")) == \
        "- no previous run on this golden set."
