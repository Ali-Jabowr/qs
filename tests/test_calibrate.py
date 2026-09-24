"""`qs calibrate` end to end on the synthetic lean session (docs/15 §7; ticket #81).

The dataset is built once from `tests/conftest.py`'s lean session with #80's `qs build`
(`plan_build` + `write_build`, reused unchanged). The bundle is written with P07's
`write_bundle` naming the dataset's train units, and the scorer is the conftest fake — the
P07 seam. The oval calib unit is made by rewriting its three good frames in a copy of the
dataset: two rotations at half-axes (26, 34), one at (17, 52) — the same area, a far smaller
axis ratio, so its worst rotation falls below the derived `axis_ratio_min` while every area
stays inside the derived range.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from appearance_fixtures import frame_of
from conftest import (
    BENCH_CAMERA,
    BUILD_SESSION,
    CARD_BOX,
    FLOOR_GREY,
    SMALL_H,
    SMALL_W,
    FakeScorer,
    Paint,
    ShotRecord,
    bench_product,
    draw_lean_frame,
)
from qs.__main__ import build_parser
from qs.cmd import calibrate
from qs.config import canonical_hash, load_config
from qs.contracts import CHECK_APPEARANCE, CHECK_GEOMETRY, Label
from qs.eval import calibrate as cal
from qs.eval.golden import plan_build, write_build
from qs.eval.scoring import score_frames
from qs.inspector.appearance import AppearanceInspector
from qs.inspector.bundle import BundleInfo, ImageRecord, write_bundle
from qs.inspector.geometry import measure
from qs.inspector.localize import localize
from qs.inspector.model import Scorer
from qs.inspector.tolerances import derive_tolerances
from qs.names import parse_name
from qs.session import FrameLog, SessionRecord, save_frame, write_plan, write_session
from qs.splitplan import Plan, Shot, make_plan, shot_list

PRODUCT = "mochi-vanilla"
PLAN_SEED = 20260925

ID_RE = re.compile(r"^cal-(\d{4}-\d{2}-\d{2})-far(\d)-([0-9a-f]{6})$")


def write_varied_session(root: Path) -> tuple[Plan, Path]:
    """The lean session with one train unit's good discs one pixel smaller (radius 25).

    The untouched lean session's frames are pixel-identical, so its σₙ is 0 and the derived
    area range is a single pixel — refused (see `test_a_degenerate_derivation_is_refused`).
    One unit at radius 25 is the only variation; everything else is drawn by the conftest
    drawer and written with P05b's own writers.
    """
    plan = make_plan("lean", PLAN_SEED)
    folder = root / BUILD_SESSION
    write_plan(folder, plan, "sheet\n")
    record = ShotRecord()
    with FrameLog(folder) as log:
        for step in shot_list(plan, "G2"):
            if not isinstance(step, Shot):
                continue
            frame = step.frame
            pixels = draw_lean_frame(frame.unit, frame.state, frame.rotation,
                                     plan.damage.get(frame.unit))
            if frame.state == "good" and frame.unit == plan.train[0]:
                pixels = np.full((SMALL_H, SMALL_W, 3), FLOOR_GREY, dtype=np.uint8)
                paint = Paint()
                paint.blob(pixels, CARD_BOX, 158)
                paint.disc(pixels, (150 + 6 * frame.rotation, 116 - 4 * frame.rotation), 25)
            sha = save_frame(folder, frame, pixels)
            log.append_row(step.file, frame, record, None, None, sha, "saved")
    write_session(
        folder,
        SessionRecord(
            session_id=BUILD_SESSION, geometry="G2", product=bench_product(),
            device="/dev/video0", camera_at_open=BENCH_CAMERA, plan=plan, frames=(record,),
        ),
        by="qs import-phone",
    )
    return plan, folder


def make_scorer(device: str, bank: np.ndarray) -> Scorer:
    """The P07 seam with the conftest fake: the map is each pixel's distance to the bank."""
    return FakeScorer(bank)


def write_train_bundle(models: Path, dataset: Path, *, extra: str | None = None) -> str:
    """A bundle naming exactly the dataset's train units — or one calib unit too (`extra`)."""
    images = [
        ImageRecord(path.name, parse_name(path.name).unit,
                    hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted((dataset / "train" / "good").glob("*.png"))
    ]
    if extra is not None:
        path = dataset / "calib" / "good" / extra
        images.append(ImageRecord(path.name, parse_name(path.name).unit,
                                  hashlib.sha256(path.read_bytes()).hexdigest()))
    info = BundleInfo(input_px=32, window_px=64, coreset_ratio=0.1, seed=42, device="cuda",
                      data_source="own", images=tuple(images))
    bank = np.full((4, 3), 0.5, dtype=np.float32)
    return write_bundle(models, PRODUCT, bank, info)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory, bench_config: Path) -> SimpleNamespace:
    """The built dataset (the varied session), a bundle on its train units, and the config."""
    _, session = write_varied_session(tmp_path_factory.mktemp("varied-session"))
    data = tmp_path_factory.mktemp("calib-data")
    build = plan_build(session, data, PRODUCT, None)
    write_build(build)
    dataset = build.dataset.root
    models = data / "models"
    return SimpleNamespace(
        dataset=dataset, models=models, config=bench_config,
        model_id=write_train_bundle(models, dataset),
    )


def copy_dataset(env: SimpleNamespace, tmp_path: Path) -> Path:
    """The dataset copied into this test's `tmp_path`, for the tests that tamper."""
    target = tmp_path / "dataset"
    return shutil.copytree(env.dataset, target)


def rewrite(path: Path, pixels: np.ndarray) -> None:
    """Rewrite one read-only dataset frame in place."""
    os.chmod(path, 0o644)
    assert cv2.imwrite(str(path), pixels)


def oval_frame(rotation: int, axes: tuple[int, int]) -> np.ndarray:
    """The lean frame's disc replaced by an oval of the given half-axes."""
    paint = Paint()
    pixels = np.full((SMALL_H, SMALL_W, 3), FLOOR_GREY, dtype=np.uint8)
    paint.blob(pixels, CARD_BOX, 158)
    paint.ellipse(pixels, (150 + 6 * rotation, 116 - 4 * rotation), axes, 0)
    return pixels


def bare_frame() -> np.ndarray:
    """The lean frame's floor and grey card, with no unit at all."""
    paint = Paint()
    pixels = np.full((SMALL_H, SMALL_W, 3), FLOOR_GREY, dtype=np.uint8)
    paint.blob(pixels, CARD_BOX, 158)
    return pixels


def make_oval(dataset: Path, unit: str) -> None:
    """One calib unit's three good frames rewritten as ovals: (26, 34), (26, 34), (17, 52)."""
    for rotation, axes in ((0, (26, 34)), (1, (26, 34)), (2, (17, 52))):
        rewrite(dataset / "calib" / "good" / f"{unit}_good_G2_r{rotation}.png",
                oval_frame(rotation, axes))


def run(
    tmp_path: Path, env: SimpleNamespace, *extra: str, dataset: Path | None = None,
    config: Path | None = None, model: str | None = None,
) -> int:
    """`qs calibrate` through `main()` with the fake scorer and a fake CUDA probe."""
    args = build_parser().parse_args([
        "calibrate",
        "--config", str(config or env.config),
        "--dataset", str(dataset or env.dataset),
        "--model", model or env.model_id,
        "--out", str(tmp_path / "out" / "station.yml"),
        "--models", str(env.models),
        "--report-dir", str(tmp_path / "reports" / "calib"),
        *extra,
    ])
    return calibrate.main(args, make_scorer=make_scorer, cuda_available=lambda: True)


def refused(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str],
    *parts: object, **kwargs: object,
) -> str:
    """One refusal: exit 1, every `part` in the output, no traceback, nothing written."""
    code = run(tmp_path, env, **kwargs)
    out = capsys.readouterr().out
    assert code == 1, out
    for part in parts:
        assert str(part) in out, (part, out)
    assert "Traceback" not in out
    assert not (tmp_path / "out" / "station.yml").exists()
    assert not (tmp_path / "reports").exists() or not any((tmp_path / "reports").rglob("*.yml"))
    return out


# --- the happy path --------------------------------------------------------------------------


def test_the_synthetic_dataset_calibrates(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 0; the new config loads; it differs from the input only in `inspector.model` and
    `threshold`; the input is byte-identical afterwards."""
    before = env.config.read_bytes()
    code = run(tmp_path, env)
    out = capsys.readouterr().out
    assert code == 0, out
    out_path = tmp_path / "out" / "station.yml"
    loaded = load_config(out_path)  # the station's own loader takes it
    after = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    original = yaml.safe_load(before.decode("utf-8"))
    assert set(after) == set(original)  # key order kept, nothing added or dropped
    for key in original:
        if key in ("inspector", "threshold"):
            continue
        assert after[key] == original[key], key
    assert after["inspector"]["model"] == env.model_id
    assert {k: v for k, v in after["inspector"].items() if k != "model"} == {
        k: v for k, v in original["inspector"].items() if k != "model"
    }
    assert after["version"] == original["version"]
    assert env.config.read_bytes() == before  # the input file is byte-identical
    header = out_path.read_text(encoding="utf-8").splitlines()[0:3]
    joined = "\n".join(header)
    assert "qs calibrate" in joined and str(env.config) in joined
    assert load_config(env.config).config_hash in joined  # the input's config_hash
    threshold = loaded.config.threshold
    assert threshold.policy == "far_5pct_on_calib"
    assert threshold.set_on.model == env.model_id
    assert threshold.set_on.backend == loaded.config.inspector.backend
    assert threshold.set_on.calib_units == 10  # the lean plan's calib split
    assert threshold.set_on.localize_threshold == loaded.config.inspector.localize.threshold
    assert threshold.set_on.mask_dilate_px == loaded.config.inspector.mask_dilate_px
    record_path = tmp_path / "reports" / "calib" / f"{threshold.id}.yml"
    assert f"threshold_id: {threshold.id}" in out
    assert str(record_path) in out
    assert "the highest calibration score: 0 of 10 flagged" in out
    assert "below 1/n" in out
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    scores = [unit["score"] for unit in record["units"].values()]
    assert threshold.value == max(scores)  # a = 0: the threshold is the highest score
    assert record["n"] == 10 and record["a"] == 0 and record["g"] == 0
    assert record["flagged"] == []
    assert record["model"] == env.model_id and record["target"] == 5
    assert record["config_hash"] == load_config(env.config).config_hash
    assert record["dataset"] == str(env.dataset)
    assert (record["localize_threshold"], record["mask_dilate_px"]) == (
        loaded.config.inspector.localize.threshold, loaded.config.inspector.mask_dilate_px)
    assert "Traceback" not in out


def test_the_tolerances_equal_derive_tolerances_on_the_same_frames(
    tmp_path: Path, env: SimpleNamespace
) -> None:
    """The written tolerances are what P06's `derive_tolerances` gives on train ∪ calib."""
    code = run(tmp_path, env)
    assert code == 0
    loaded = load_config(tmp_path / "out" / "station.yml")
    per_unit: dict[str, list[object]] = {}
    for split in ("train", "calib"):
        for path in sorted((env.dataset / split / "good").glob("*.png")):
            pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert pixels is not None
            silhouette = localize(pixels, loaded.config.inspector.localize)
            per_unit.setdefault(parse_name(path.name).unit, []).append(measure(silhouette))
    direct = derive_tolerances(per_unit)  # type: ignore[arg-type]
    assert loaded.config.threshold.geometry == direct.config
    record = yaml.safe_load(
        (tmp_path / "reports" / "calib" / f"{loaded.config.threshold.id}.yml").read_text(
            encoding="utf-8")
    )
    assert record["tolerances"]["area_px"] == list(direct.config.area_px)
    assert record["tolerances"]["axis_ratio_min"] == direct.config.axis_ratio_min
    assert record["tolerances"]["solidity_min"] == direct.config.solidity_min
    assert record["tolerances"]["derivation"]["units"] == direct.units
    assert record["tolerances"]["derivation"]["frames"] == direct.frames


def test_the_threshold_id_and_its_record_are_reproducible(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cal-YYYY-MM-DD-far5-<6 hex>`; the suffix recomputes from the record file; two runs
    give the same suffix and the same record bytes."""
    assert run(tmp_path, env) == 0
    first = load_config(tmp_path / "out" / "station.yml").config.threshold
    assert ID_RE.fullmatch(first.id)
    record_file = tmp_path / "reports" / "calib" / f"{first.id}.yml"
    record = yaml.safe_load(record_file.read_text(encoding="utf-8"))
    suffix = canonical_hash(record)[:6]
    assert first.id[-6:] == suffix
    first_bytes = record_file.read_bytes()
    other = tmp_path / "out-2"
    other.mkdir()
    assert run(tmp_path, env, "--out", str(other / "station.yml")) == 0
    out = capsys.readouterr().out
    second = load_config(other / "station.yml").config.threshold
    assert second.id == first.id  # the same day, the same record, the same suffix
    assert record_file.read_bytes() == first_bytes
    assert out.count(f"threshold_id: {first.id}") == 2


# --- score_frames: the one scoring path ------------------------------------------------------


def _candidate(env: SimpleNamespace) -> tuple[AppearanceInspector, object, str, list[Path]]:
    """The candidate config and a loaded `AppearanceInspector`, as the command builds them."""
    config = load_config(env.config).config
    per_unit: dict[str, list[object]] = {}
    for split in ("train", "calib"):
        for path in sorted((env.dataset / split / "good").glob("*.png")):
            pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert pixels is not None
            silhouette = localize(pixels, config.inspector.localize)
            per_unit.setdefault(parse_name(path.name).unit, []).append(measure(silhouette))
    tolerances = derive_tolerances(per_unit)  # type: ignore[arg-type]
    raw = yaml.safe_load(env.config.read_text(encoding="utf-8"))
    candidate = cal.candidate_config(config, env.model_id, tolerances.config)
    block = cal.threshold_block(
        cal.CANDIDATE_ID, cal.CANDIDATE_VALUE, model=env.model_id, config=config,
        geometry=tolerances.config, calib_units=10, target=5,
    )
    config_hash = canonical_hash(cal.with_threshold(raw, env.model_id, block))
    inspector = AppearanceInspector(config_hash, models_root=env.models,
                                    make_scorer=make_scorer, cuda_available=lambda: True)
    inspector.load(candidate)
    files = sorted((env.dataset / "calib" / "good").glob("*.png"))[:5]
    return inspector, candidate, config_hash, files


def test_score_frames_agrees_with_a_direct_inspect(env: SimpleNamespace) -> None:
    """5 synthetic frames: the same appearance values as `AppearanceInspector.inspect` on
    the same pixels, one `FrameResult` per file, in order."""
    inspector, candidate, config_hash, files = _candidate(env)
    results = score_frames(candidate, config_hash, files, inspector=inspector)  # type: ignore[arg-type]
    assert [result.file for result in results] == files
    assert len(results) == 5
    for result in results:
        pixels = cv2.imread(str(result.file), cv2.IMREAD_COLOR)
        assert pixels is not None
        frame = frame_of(pixels)
        direct = inspector.inspect(frame)
        appearance = next(c for c in direct.checks if c.name == CHECK_APPEARANCE)
        geometry = next(c for c in direct.checks if c.name == CHECK_GEOMETRY)
        assert result.score == appearance.value
        assert result.pred_score == dict(appearance.measures)["pred_score"]
        assert result.geometry == geometry
        assert result.label == direct.label
        assert result.name == parse_name(result.file.name)
        assert result.inspect_ms >= 0.0 and result.total_ms >= 0.0


def test_score_frames_returns_the_pipelines_no_verdict(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    """A frame whose reference patch is out of band comes back НЕТ РЕШЕНИЯ(lighting)."""
    inspector, candidate, config_hash, files = _candidate(env)
    pixels = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    assert pixels is not None
    dark = pixels.copy()
    x, y, w, h = CARD_BOX
    dark[y : y + h, x : x + w] = 100  # 0.39, outside 0.62 ± 5 %
    directory = tmp_path / "dark"
    directory.mkdir()
    path = directory / files[0].name
    assert cv2.imwrite(str(path), dark)
    [result] = score_frames(candidate, config_hash, [path], inspector=inspector)  # type: ignore[arg-type]
    assert result.label is Label.NO_VERDICT
    assert result.reason == "lighting"
    assert result.geometry is None and result.score is None and result.pred_score is None


# --- the refusals ----------------------------------------------------------------------------


PLAN = make_plan("lean", PLAN_SEED)


def test_a_bundle_whose_units_include_a_calib_unit_is_refused(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    unit = PLAN.calib[0]
    model = write_train_bundle(env.models, env.dataset, extra=f"{unit}_good_G2_r0.png")
    out = refused(tmp_path, env, capsys, unit, "calibration unit", model=model)
    assert "E1" in out


def test_a_bundle_trained_on_other_units_is_refused(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    images = [
        ImageRecord(path.name, parse_name(path.name).unit,
                    hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted((env.dataset / "train" / "good").glob("*.png"))
        if parse_name(path.name).unit != PLAN.train[0]
    ]
    images.append(ImageRecord("u099_good_G2_r0.png", "u099", "0" * 64))
    info = BundleInfo(input_px=32, window_px=64, coreset_ratio=0.1, seed=42, device="cuda",
                      data_source="own", images=tuple(images))
    model = write_bundle(env.models, PRODUCT, np.full((4, 3), 0.5, np.float32), info)
    refused(tmp_path, env, capsys, PLAN.train[0], "u099", "not the dataset's train/good units",
            model=model)


def test_a_calib_frame_out_of_the_lighting_band_is_refused(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = copy_dataset(env, tmp_path)
    file = f"{PLAN.calib[0]}_good_G2_r0.png"
    pixels = cv2.imread(str(dataset / "calib" / "good" / file), cv2.IMREAD_COLOR)
    assert pixels is not None
    x, y, w, h = CARD_BOX
    pixels[y : y + h, x : x + w] = 100
    rewrite(dataset / "calib" / "good" / file, pixels)
    refused(tmp_path, env, capsys, file, "lighting", dataset=dataset)


def test_a_train_frame_with_no_unit_is_refused(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = copy_dataset(env, tmp_path)
    file = f"{PLAN.train[1]}_good_G2_r0.png"
    rewrite(dataset / "train" / "good" / file, bare_frame())
    refused(tmp_path, env, capsys, file, "placement", dataset=dataset)


def test_an_oval_calib_unit_is_refused_at_far_5(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = copy_dataset(env, tmp_path)
    unit = PLAN.calib[0]
    make_oval(dataset, unit)
    out = refused(tmp_path, env, capsys, unit, "r2", "axis_ratio<min", "below the minimum",
                  dataset=dataset)
    assert "geometry alone flags 1 of 10 units" in out
    assert "nothing widened" in out and "tolerances: area_px [" in out


def test_the_same_oval_calib_unit_passes_at_far_10(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = copy_dataset(env, tmp_path)
    unit = PLAN.calib[0]
    make_oval(dataset, unit)
    assert run(tmp_path, env, "--far", "10", dataset=dataset) == 0
    out = capsys.readouterr().out
    loaded = load_config(tmp_path / "out" / "station.yml")
    threshold = loaded.config.threshold
    assert threshold.policy == "far_10pct_on_calib"
    assert threshold.set_on.calib_units == 10
    record = yaml.safe_load(
        (tmp_path / "reports" / "calib" / f"{threshold.id}.yml").read_text(encoding="utf-8")
    )
    assert (record["n"], record["a"], record["g"]) == (10, 1, 1)
    assert record["flagged"] == [unit]
    others = [u["score"] for key, u in record["units"].items() if key != unit]
    assert threshold.value == max(others)  # the highest score among the other 9
    assert "the highest calibration score: 0 of 10 flagged" in out


def test_an_existing_out_is_refused(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "out" / "station.yml"
    out_path.parent.mkdir()
    out_path.write_bytes(b"kept")
    code = run(tmp_path, env)
    out = capsys.readouterr().out
    assert code == 1 and "exists" in out and "Traceback" not in out
    assert out_path.read_bytes() == b"kept"
    assert not (tmp_path / "reports").exists()


def test_a_g1_config_against_a_g2_dataset_is_refused(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "g1.yml"
    config.write_text(
        env.config.read_text(encoding="utf-8").replace("geometry: G2", "geometry: G1"),
        encoding="utf-8",
    )
    refused(tmp_path, env, capsys, "G1", "G2", config=config)


def test_a_phone_dataset_with_a_webcam_config_is_refused(
    tmp_path: Path, env: SimpleNamespace, phone_session: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build = plan_build(phone_session, tmp_path / "pdata", PRODUCT, None)
    write_build(build)
    refused(tmp_path, env, capsys, "phone", "ADR-0022", dataset=build.dataset.root)


def test_a_degenerate_derivation_is_refused(
    tmp_path: Path, env: SimpleNamespace, lean_session: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The untouched lean session: pixel-identical frames, σₙ = 0, a single-pixel area range
    the config loader refuses. Nothing widens it; the calibration is refused (design gap)."""
    build = plan_build(lean_session, tmp_path / "data", PRODUCT, None)
    write_build(build)
    refused(tmp_path, env, capsys, "degenerate", "single-pixel", dataset=build.dataset.root)


# --- could not run, the CLI ------------------------------------------------------------------


def test_a_tampered_bundle_cannot_run(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    models = tmp_path / "models"
    shutil.copytree(env.models / env.model_id, models / env.model_id)
    bank = models / env.model_id / "memory_bank.pt"
    raw = bytearray(bank.read_bytes())
    raw[20] ^= 0xFF
    os.chmod(bank, 0o644)
    bank.write_bytes(bytes(raw))
    code = _run_with_models(tmp_path, env, models)
    out = capsys.readouterr().out
    assert code == 2 and "changed after it was written" in out
    assert len(out.strip().splitlines()) == 1 and "Traceback" not in out
    assert not (tmp_path / "out" / "station.yml").exists()


def _run_with_models(tmp_path: Path, env: SimpleNamespace, models: Path) -> int:
    args = build_parser().parse_args([
        "calibrate", "--config", str(env.config), "--dataset", str(env.dataset),
        "--model", env.model_id, "--out", str(tmp_path / "out" / "station.yml"),
        "--models", str(models), "--report-dir", str(tmp_path / "reports" / "calib"),
    ])
    return calibrate.main(args, make_scorer=make_scorer, cuda_available=lambda: True)


def test_a_validation_only_bundle_cannot_run(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    images = tuple(
        ImageRecord(f"u{u:03d}_good_G2_r0.png", f"u{u:03d}", f"{u:02d}".ljust(64, "b"))
        for u in (1, 2, 3)
    )
    info = BundleInfo(input_px=32, window_px=64, coreset_ratio=0.1, seed=42, device="cuda",
                      data_source="external:mvtec", images=images)
    model = write_bundle(env.models, PRODUCT, np.full((4, 3), 0.5, np.float32), info)
    code = run(tmp_path, env, model=model)
    out = capsys.readouterr().out
    assert code == 2 and "validation-only" in out
    assert len(out.strip().splitlines()) == 1 and "Traceback" not in out
    assert not (tmp_path / "out" / "station.yml").exists()


def test_a_missing_dataset_folder_cannot_run(
    tmp_path: Path, env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(tmp_path, env, dataset=tmp_path / "nope")
    out = capsys.readouterr().out
    assert code == 2 and "nope" in out and "Traceback" not in out
    assert not (tmp_path / "out" / "station.yml").exists()


def test_far_3_is_refused_by_argparse() -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args([
            "calibrate", "--config", "c", "--dataset", "d", "--model", "m", "--out", "o",
            "--far", "3",
        ])
    assert caught.value.code == 2


def test_the_cli_lists_calibrate() -> None:
    parser = build_parser()
    assert "calibrate" in parser.format_help()
    parsed = parser.parse_args([
        "calibrate", "--config", "c", "--dataset", "d", "--model", "m", "--out", "o",
    ])
    assert (parsed.far, parsed.models, parsed.report_dir) == (5, "models", "reports/calib")
    parsed = parser.parse_args([
        "calibrate", "--config", "c", "--dataset", "d", "--model", "m", "--out", "o",
        "--far", "10", "--models", "m2", "--report-dir", "r2",
    ])
    assert (parsed.far, parsed.models, parsed.report_dir) == (10, "m2", "r2")
