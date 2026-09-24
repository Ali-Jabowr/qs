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
from qs.eval.golden import plan_build, write_build
from qs.eval.report import markdown as _markdown
from qs.inspector.model import Scored, Scorer
from qs.session import FrameLog, SessionRecord, save_frame, write_plan, write_session
from qs.splitplan import Plan, Shot, make_plan, shot_list
from test_calibrate import PLAN, PLAN_SEED, PRODUCT, write_train_bundle

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
) -> tuple[Plan, Path]:
    """The lean dataset session as P05b writes it, drawn by this module: every good disc
    r26 except one train unit's r25 (the one variation that keeps the derived area range
    from being a single pixel), dents and tears as blots inside the silhouette."""
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
    *, dark: float = DARK,
) -> int:
    """`qs calibrate` through `main()` with the blot scorer and a fake CUDA probe."""
    args = build_parser().parse_args([
        "calibrate", "--config", str(config), "--dataset", str(dataset),
        "--model", model_id, "--out", str(out), "--models", str(models),
        "--report-dir", str(report_dir),
    ])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = calibrate.main(args, make_scorer=lambda d, b: make_scorer(d, b, dark),
                              cuda_available=lambda: True)
    assert "Traceback" not in buffer.getvalue(), buffer.getvalue()
    return code


def build_env(
    root: Path, config: Path, *, device: str = "/dev/video0", camera: object = BENCH_CAMERA,
    blot_of: object = None,
) -> SimpleNamespace:
    """Session → build → bundle → calibrated config and record: everything #82 needs. The
    input config is the example's small-frame rewrite with the example's placeholder
    product replaced by the dataset's own."""
    text = config.read_text(encoding="utf-8").replace("product: mochi-<flavour>-v1",
                                                      f"product: {PRODUCT}")
    assert "product: mochi-vanilla" in text
    input_config = root / "input.yml"
    input_config.write_text(text, encoding="utf-8")
    _, session = write_eval_session(root, device=device, camera=camera, blot_of=blot_of)
    data = root / "data"
    build = plan_build(session, data, PRODUCT, None)
    write_build(build)
    models = data / "models"
    model_id = write_train_bundle(models, build.dataset.root)
    out = data / "station.yml"
    assert run_calibrate(input_config, build.dataset.root, model_id, models, out,
                         data / "reports" / "calib") == 0
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
