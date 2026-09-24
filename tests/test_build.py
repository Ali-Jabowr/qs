"""`qs build`: one raw session → the dataset tree and the frozen golden set (docs/12 §2,
docs/13 §6; FR-52, FR-56; ticket #80).

The synthetic lean session of `tests/conftest.py` drives every test: written with P05b's own
planner and writers at a small frame size, 265 frames under G2, seed 20260925.
"""

import argparse
import csv
import hashlib
import os
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import (
    BENCH_CAMERA,
    FLOOR_GREY,
    SMALL_H,
    SMALL_W,
    ShotRecord,
    bench_product,
)
from qs.__main__ import build_parser
from qs.cmd.build import main
from qs.eval.golden import BuildRefused, GoldenRefused, plan_build, verify_golden, write_build
from qs.names import FrameName, format_name, parse_name
from qs.session import FrameLog, SessionRecord, save_frame, write_plan, write_session
from qs.splitplan import Plan, make_pilot_plan, make_plan

PRODUCT = "mochi-vanilla"
GOLDEN = "mvp-2026-09-25"  # mvp-<date of started_at>
PLAN_SEED = 20260925
MANIFEST_COLUMNS = [
    "file", "unit", "role", "state", "class", "geometry", "rotation", "pass", "sha256",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(folder: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under `folder`: its bytes and its mtime_ns, keyed by relative path."""
    return {str(p.relative_to(folder)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def args(session: Path | str, data: Path | str, **changes: object) -> argparse.Namespace:
    values: dict[str, object] = dict(
        session=str(session), product=PRODUCT, data=str(data), name=None,
    )
    values.update(changes)
    return argparse.Namespace(**values)


def dataset_of(data: Path, product: str = PRODUCT) -> Path:
    return data / "datasets" / product / "G2"


def golden_of(data: Path, name: str = GOLDEN) -> Path:
    return data / "golden" / name


def class_of(plan: Plan, unit: str) -> str:
    """The dataset folder of a test unit's damage: the variant stays in `session.yml`."""
    return plan.damage[unit].split("-")[0]


def planned_dataset(plan: Plan) -> dict[str, list[str]]:
    """Every file of the dataset tree, per split folder, from the plan's lean sizes."""
    return {
        "train/good": [f"{u}_good_G2_r{r}.png" for u in plan.train for r in range(3)],
        "calib/good": [f"{u}_good_G2_r{r}.png" for u in plan.calib for r in range(3)],
        "test/good": [f"{u}_good_G2_r{r}.png" for u in plan.test for r in range(3)],
        **{
            f"test/{cls}": [f"{u}_dmg_G2_r{r}.png" for u in plan.test
                            if class_of(plan, u) == cls for r in range(3)]
            for cls in ("dent", "tear", "deformed")
        },
    }


def planned_golden(plan: Plan) -> list[str]:
    """Every frame of the golden set: calib, test (both states), control, placement."""
    files = [f"{u}_good_G2_r{r}.png" for u in plan.calib for r in range(3)]
    files += [f"{u}_{s}_G2_r{r}.png" for u in plan.test for s in ("good", "dmg")
              for r in range(3)]
    files += [f"c0{c}_good_G2_r{r}_pass{p}.png" for c in (1, 2, 3)
              for r in range(3) for p in (1, 2, 3)]
    files += [f"x0{n}_empty_G2_r0.png" for n in range(1, 6)]
    files += [f"x{n:02d}_edge_G2_r0.png" for n in range(6, 11)]
    return files


# --- the whole build ------------------------------------------------------------------------


def test_the_lean_session_builds_the_dataset_and_the_golden_set(
    tmp_path: Path, lean_session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = make_plan("lean", PLAN_SEED)
    data = tmp_path / "data"
    assert main(args(lean_session, data)) == 0
    dataset = dataset_of(data)
    for split, files in planned_dataset(plan).items():
        got = sorted(p.name for p in (dataset / split).iterdir())
        assert got == sorted(files), split
        for file in files:
            assert (dataset / split / file).read_bytes() == (lean_session / file).read_bytes()
    golden = golden_of(data)
    frames = sorted(p.name for p in (golden / "frames").iterdir())
    assert frames == sorted(planned_golden(plan))
    for file in frames:
        assert (golden / "frames" / file).read_bytes() == (lean_session / file).read_bytes()
    assert (golden / "session.yml").read_bytes() == (lean_session / "session.yml").read_bytes()
    assert (dataset / "session.yml").read_bytes() == (lean_session / "session.yml").read_bytes()
    out = capsys.readouterr().out
    for line in (
        "train 18 units / 54 frames", "calib 10 units / 30 frames",
        "test 24 units / 144 frames", "control 3 units / 27 frames", "placement 10 frames",
        f"dataset: {dataset}", f"golden: {golden}", "golden sha256: ",
        f"qs train --dataset {dataset}", f"qs calibrate --dataset {dataset}",
        f"qs evaluate --golden {golden}",
    ):
        assert line in out, line


# --- the golden set's contents --------------------------------------------------------------


def test_the_golden_set_holds_the_manifest_and_its_hash(
    tmp_path: Path, lean_session: Path
) -> None:
    plan = make_plan("lean", PLAN_SEED)
    data = tmp_path / "data"
    assert main(args(lean_session, data)) == 0
    frames = sorted(p.name for p in (golden_of(data) / "frames").iterdir())
    wanted = planned_golden(plan)
    assert Counter(f.split("_")[0] for f in frames) == {
        **{u: 3 for u in plan.calib}, **{u: 6 for u in plan.test},
        **{c: 9 for c in plan.control}, **{x: 1 for x in
                                           [f"x{n:02d}" for n in range(1, 11)]},
    }
    assert not any(f.split("_")[0] in plan.train for f in frames)  # never a train frame
    assert frames == sorted(wanted)
    with (golden_of(data) / "manifest.csv").open(encoding="utf-8", newline="") as manifest:
        rows = list(csv.DictReader(manifest))
    assert list(rows[0]) == MANIFEST_COLUMNS
    assert [row["file"] for row in rows] == frames  # one row per frame, in name order
    roles = {**{u: "calib" for u in plan.calib}, **{u: "test" for u in plan.test},
             **{c: "control" for c in plan.control},
             **{f"x{n:02d}": "placement" for n in range(1, 11)}}
    for row in rows:
        unit = row["file"].split("_")[0]
        name = parse_name(row["file"])
        assert row["unit"] == unit and row["role"] == roles[unit]
        assert row["geometry"] == "G2" and row["rotation"] == f"r{name.rotation}"
        assert row["state"] == name.state
        if roles[unit] == "control":
            assert row["pass"] == row["file"].split("_")[4][: -len(".png")]
        else:
            assert row["pass"] == ""
        # class set on both states of a test unit, empty otherwise
        assert row["class"] == (class_of(plan, unit) if roles[unit] == "test" else "")
        assert row["sha256"] == sha(golden_of(data) / "frames" / row["file"])
    manifest_sha = (golden_of(data) / "manifest.sha256").read_text(encoding="utf-8")
    assert manifest_sha == sha(golden_of(data) / "manifest.csv") + "\n"


def test_the_manifest_rows_recompute_from_the_files(tmp_path: Path, lean_session: Path) -> None:
    """The hash and every row recomputed from the files on disk, not from the builder."""
    plan = make_plan("lean", PLAN_SEED)
    data = tmp_path / "data"
    assert main(args(lean_session, data)) == 0
    golden = golden_of(data)
    with (golden / "manifest.csv").open(encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    for row in rows:
        assert row["sha256"] == sha(golden / "frames" / row["file"])
    assert (golden / "manifest.sha256").read_text(encoding="utf-8").strip() == (
        hashlib.sha256((golden / "manifest.csv").read_bytes()).hexdigest()
    )
    assert len(rows) == len(planned_golden(plan))


# --- the raw session, the modes, idempotence ------------------------------------------------


def test_the_raw_session_is_untouched_and_the_copies_are_read_only(
    tmp_path: Path, lean_session: Path
) -> None:
    data = tmp_path / "data"
    before = snapshot(lean_session)
    assert main(args(lean_session, data)) == 0
    assert snapshot(lean_session) == before  # byte-identical, every file hashed
    for tree in (dataset_of(data), golden_of(data)):
        for path in tree.rglob("*.png"):
            assert path.stat().st_mode & 0o777 == 0o444, path


def test_a_second_build_is_already_built_and_changes_nothing(
    tmp_path: Path, lean_session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    assert main(args(lean_session, data)) == 0
    before = snapshot(data)
    assert main(args(lean_session, data)) == 0
    assert "already built" in capsys.readouterr().out
    assert snapshot(data) == before  # no file's mtime changed


def test_the_same_session_under_another_product_builds_a_second_dataset(
    tmp_path: Path, lean_session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    assert main(args(lean_session, data)) == 0
    golden_before = snapshot(golden_of(data))
    assert main(args(lean_session, data, product="mochi-mango")) == 0
    out = capsys.readouterr().out
    assert dataset_of(data, "mochi-mango").is_dir()
    assert "already built" in out and str(golden_of(data)) in out
    assert snapshot(golden_of(data)) == golden_before  # the same golden set, untouched


# --- refusals: exit 1, the cause named, nothing written -------------------------------------


def write_pilot_session(root: Path) -> Path:
    """A lighting-pilot session, written with P05b's own writers (docs/13 §5)."""
    plan = make_pilot_plan("lean", PLAN_SEED)
    folder = root / "2026-09-25-pilot"
    write_plan(folder, plan, "sheet\n")
    record = ShotRecord()
    with FrameLog(folder) as log:
        name = FrameName("p01", "good", "G1", 0, None)
        sha = save_frame(folder, name, np.full((SMALL_H, SMALL_W, 3), FLOOR_GREY, np.uint8))
        log.append_row(format_name(name), name, record, None, None, sha, "saved")
    write_session(
        folder,
        SessionRecord(
            session_id="2026-09-25-pilot", geometry="G1", product=bench_product(),
            device="/dev/video0", camera_at_open=BENCH_CAMERA, plan=plan, frames=(record,),
        ),
        by="qs import-phone",
    )
    return folder


def refused(
    capsys: pytest.CaptureFixture[str], session: Path, data: Path, *parts: object,
    **changes: object,
) -> str:
    """One refusal: exit 1, every `part` named in the output, no directory under `--data`."""
    assert main(args(session, data, **changes)) == 1
    out = capsys.readouterr().out
    for part in parts:
        assert str(part) in out, (part, out)
    assert "Traceback" not in out
    assert not data.exists() or not any(data.iterdir())
    return out


def edit_session_yml(session: Path, change: Callable[[dict[str, object]], None]) -> None:
    """`session.yml` parsed, changed, written back — the tamper the refusals catch."""
    data = yaml.safe_load((session / "session.yml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    change(data)
    (session / "session.yml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )


def drop_row(session: Path, file: str) -> None:
    """Drop one frame's row from `frames.csv`, as if the frame was never logged."""
    with (session / "frames.csv").open(encoding="utf-8", newline="") as log:
        reader = csv.DictReader(log)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    kept = [row for row in rows if row["file"] != file]
    assert len(kept) == len(rows) - 1
    with (session / "frames.csv").open("w", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(log, columns)
        writer.writeheader()
        writer.writerows(kept)


def relog_sha(session: Path, file: str) -> None:
    """Correct one row's `sha256` to the file's current bytes, as a re-saved frame would be."""
    with (session / "frames.csv").open(encoding="utf-8", newline="") as log:
        reader = csv.DictReader(log)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row["file"] == file:
            row["sha256"] = sha(session / file)
    with (session / "frames.csv").open("w", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(log, columns)
        writer.writeheader()
        writer.writerows(rows)


def flip_a_byte(session: Path, file: str) -> None:
    """One byte of a frame changed after the shoot; a copy's frame is read-only."""
    path = session / file
    if path.exists():
        os.chmod(path, 0o644)
    raw = bytearray(path.read_bytes())
    raw[10] ^= 0xFF
    path.write_bytes(bytes(raw))


def test_a_unit_in_two_splits_is_refused(
    tmp_path: Path, session_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = make_plan("lean", PLAN_SEED)
    unit = plan.test[0]
    edit_session_yml(session_copy, lambda data: data["units"]["train"].append(unit))  # type: ignore[index]
    refused(capsys, session_copy, tmp_path / "data", unit, "train", "test")


def test_a_control_unit_in_train_is_refused(
    tmp_path: Path, session_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edit_session_yml(session_copy, lambda data: data["units"]["train"].append("c09"))  # type: ignore[index]
    refused(capsys, session_copy, tmp_path / "data", "c09", "train", "c")


def test_a_frame_whose_bytes_changed_after_the_shoot_is_refused(
    tmp_path: Path, session_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = make_plan("lean", PLAN_SEED)
    file = f"{plan.calib[0]}_good_G2_r0.png"
    flip_a_byte(session_copy, file)
    refused(capsys, session_copy, tmp_path / "data", file, "SHA-256")


def test_an_extra_png_is_refused(
    tmp_path: Path, session_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (session_copy / "u099_good_G2_r0.png").write_bytes(b"not a png")
    refused(capsys, session_copy, tmp_path / "data", "u099_good_G2_r0.png", "frames.csv")


def test_a_missing_dmg_rotation_is_refused(
    tmp_path: Path, session_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = make_plan("lean", PLAN_SEED)
    unit = plan.test[0]
    file = f"{unit}_dmg_G2_r2.png"
    (session_copy / file).unlink()
    drop_row(session_copy, file)
    refused(capsys, session_copy, tmp_path / "data", unit, "dmg", "r2")


def test_a_missing_control_pass_is_refused(
    tmp_path: Path, session_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for rotation in range(3):
        file = f"c02_good_G2_r{rotation}_pass2.png"
        (session_copy / file).unlink()
        drop_row(session_copy, file)
    refused(capsys, session_copy, tmp_path / "data", "c02", "pass2")


def test_a_pilot_session_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = write_pilot_session(tmp_path / "raw")
    refused(capsys, folder, tmp_path / "data", "pilot", "docs/13 §5")


def test_a_session_unit_not_in_the_plans_split_or_spares_is_refused(
    tmp_path: Path, session_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = make_plan("lean", PLAN_SEED)
    unit = plan.calib[0]

    def move(data: dict[str, object]) -> None:
        units = data["units"]
        assert isinstance(units, dict)
        units["calib"].remove(unit)  # type: ignore[union-attr]
        units["train"].append(unit)  # type: ignore[union-attr]

    edit_session_yml(session_copy, move)
    refused(capsys, session_copy, tmp_path / "data", unit, "train")


def test_a_golden_target_that_exists_with_a_different_manifest_is_refused(
    tmp_path: Path, session_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    assert main(args(session_copy, data)) == 0
    before = snapshot(data)
    file = "c01_good_G2_r0_pass1.png"  # a control frame: the golden set holds it, the
    flip_a_byte(session_copy, file)  # dataset does not, so only the manifest would differ
    relog_sha(session_copy, file)
    assert main(args(session_copy, data)) == 1
    out = capsys.readouterr().out
    assert "golden" in out and "different content" in out
    assert "Traceback" not in out
    assert snapshot(data) == before  # nothing written over the frozen set


# --- a broken unit, a phone session, the verifier -------------------------------------------


def test_a_session_with_a_broken_unit_builds_the_spares_split(
    tmp_path: Path, broken_session: Path
) -> None:
    plan = make_plan("lean", PLAN_SEED)
    broken, spare = plan.test[0], plan.spare[0]
    data = tmp_path / "data"
    assert main(args(broken_session, data)) == 0
    dataset, golden = dataset_of(data), golden_of(data)
    everywhere = list(dataset.rglob("*.png")) + list((golden / "frames").iterdir())
    held = {p.name for p in everywhere}
    for rotation in range(3):  # the discarded unit's frames are in neither tree
        assert f"{broken}_good_G2_r{rotation}.png" not in held
    assert not any(p.name.startswith(f"{broken}_") for p in everywhere)
    for state in ("good", "dmg"):  # the spare is in the split it took over, with its damage
        for rotation in range(3):
            assert f"{spare}_{state}_G2_r{rotation}.png" in held
    counts = {str(p.relative_to(dataset)): 1 for p in dataset.rglob("*.png")}
    assert Counter("/".join(k.split("/")[:2]) for k in counts) == {
        "train/good": 54, "calib/good": 30, "test/good": 72,
        "test/dent": 30, "test/tear": 21, "test/deformed": 21,
    }  # the counts stay the plan's
    assert len(list((golden / "frames").iterdir())) == 211
    with (golden / "manifest.csv").open(encoding="utf-8", newline="") as manifest:
        rows = list(csv.DictReader(manifest))
    assert not any(row["unit"] == broken for row in rows)
    assert [row["class"] for row in rows if row["unit"] == spare] == [
        class_of(plan, broken) for _ in range(6)
    ]


def test_a_phone_session_builds_the_same_way_and_the_golden_keeps_its_session_yml(
    tmp_path: Path, phone_session: Path
) -> None:
    data = tmp_path / "data"
    assert main(args(phone_session, data)) == 0
    session_yml = yaml.safe_load((phone_session / "session.yml").read_text(encoding="utf-8"))
    assert str(session_yml["camera"]["device"]).startswith("phone:")  # type: ignore[union-attr]
    assert session_yml["camera"]["verified"] is False  # type: ignore[index]
    golden_copy = golden_of(data) / "session.yml"
    assert golden_copy.read_bytes() == (phone_session / "session.yml").read_bytes()


def test_the_builder_verifies_the_golden_set_it_just_wrote(
    tmp_path: Path, lean_session: Path
) -> None:
    """The builder calls `verify_golden` on what it wrote — a good set verifies quietly."""
    data = tmp_path / "data"
    assert main(args(lean_session, data)) == 0
    assert verify_golden(golden_of(data))


# --- a frame changed between planning and writing (review P08 F2) ---------------------------


def nothing_built(data: Path) -> None:
    """No file anywhere under `data`, and neither target nor a staging directory left."""
    assert not [p for p in data.rglob("*") if p.is_file()], sorted(data.rglob("*"))
    assert not dataset_of(data).exists() and not golden_of(data).exists()
    assert not [p for p in data.rglob(".*")], sorted(data.rglob(".*"))


@pytest.mark.parametrize("split", ["train", "calib"])
def test_a_frame_changed_after_planning_is_refused_and_nothing_is_built(
    tmp_path: Path, session_copy: Path, split: str
) -> None:
    """Each copy is re-hashed against its planned SHA-256 as it is written: a train frame
    (dataset only) or a calib frame (both trees) edited between `plan_build` and
    `write_build` is refused naming the file, and neither tree is left behind."""
    plan = make_plan("lean", PLAN_SEED)
    unit = plan.train[0] if split == "train" else plan.calib[0]
    file = f"{unit}_good_G2_r0.png"
    data = tmp_path / "data"
    build = plan_build(session_copy, data, PRODUCT, None)
    flip_a_byte(session_copy, file)
    with pytest.raises(BuildRefused) as raised:
        write_build(build)
    assert file in str(raised.value) and "SHA-256" in str(raised.value)
    nothing_built(data)


def test_a_golden_set_that_fails_verification_leaves_no_dataset(
    tmp_path: Path, session_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both trees are staged and verified before either is renamed into place: a golden
    set that fails its verification leaves no dataset tree behind."""
    from qs.eval import golden as golden_module

    data = tmp_path / "data"
    build = plan_build(session_copy, data, PRODUCT, None)

    def failing(path: Path) -> list[dict[str, str]]:
        raise GoldenRefused("probe: the staged golden set does not verify")

    monkeypatch.setattr(golden_module, "verify_golden", failing)
    with pytest.raises(GoldenRefused):
        write_build(build)
    nothing_built(data)


def refuses(
    golden: Path, tamper: Callable[[Path], None], *parts: object,
) -> None:
    """One verifier refusal on a copy of a built golden set: the cause named."""
    copy = golden.parent / (golden.name + "-copy")
    shutil.copytree(golden, copy)
    tamper(copy)
    with pytest.raises(GoldenRefused) as raised:
        verify_golden(copy)
    for part in parts:
        assert str(part) in str(raised.value), (part, str(raised.value))
    shutil.rmtree(copy)


def test_the_verifier_refuses_a_changed_frame_byte(
    tmp_path: Path, lean_session: Path,
) -> None:
    assert main(args(lean_session, tmp_path / "data")) == 0
    file = "c01_good_G2_r0_pass1.png"

    def tamper(copy: Path) -> None:
        flip_a_byte(copy / "frames", file)

    refuses(golden_of(tmp_path / "data"), tamper, file, "hashes to")


def test_the_verifier_refuses_an_edited_manifest(tmp_path: Path, lean_session: Path) -> None:
    assert main(args(lean_session, tmp_path / "data")) == 0

    def tamper(copy: Path) -> None:
        text = (copy / "manifest.csv").read_text(encoding="utf-8")
        (copy / "manifest.csv").write_text(text.replace("dent", "tear", 1),
                                           encoding="utf-8")

    refuses(golden_of(tmp_path / "data"), tamper, "manifest.sha256", "edited")


def test_the_verifier_refuses_a_removed_frame(tmp_path: Path, lean_session: Path) -> None:
    assert main(args(lean_session, tmp_path / "data")) == 0
    file = "x01_empty_G2_r0.png"

    def tamper(copy: Path) -> None:
        (copy / "frames" / file).unlink()

    refuses(golden_of(tmp_path / "data"), tamper, file, "not in frames/")


def test_the_verifier_refuses_an_added_frame(tmp_path: Path, lean_session: Path) -> None:
    assert main(args(lean_session, tmp_path / "data")) == 0

    def tamper(copy: Path) -> None:
        shutil.copyfile(copy / "frames" / "x01_empty_G2_r0.png",
                        copy / "frames" / "x11_empty_G2_r0.png")

    refuses(golden_of(tmp_path / "data"), tamper, "x11_empty_G2_r0.png", "not in the manifest")


def test_the_verifier_refuses_a_row_whose_unit_disagrees_with_its_file_name(
    tmp_path: Path, lean_session: Path,
) -> None:
    assert main(args(lean_session, tmp_path / "data")) == 0

    def tamper(copy: Path) -> None:
        text = (copy / "manifest.csv").read_text(encoding="utf-8")
        row = "c01_good_G2_r0_pass1.png,c01,control,good,,G2,r0,pass1,"
        moved = row.replace("c01,control", "c02,control", 1)
        assert row in text
        text = text.replace(row, moved, 1)
        (copy / "manifest.csv").write_text(text, encoding="utf-8")
        (copy / "manifest.sha256").write_text(
            sha(copy / "manifest.csv") + "\n", encoding="utf-8"
        )

    refuses(golden_of(tmp_path / "data"), tamper, "c01_good_G2_r0_pass1.png", "c01", "c02")


# --- could not run, the CLI -----------------------------------------------------------------


def test_a_missing_session_folder_or_session_yml_cannot_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    assert main(args(tmp_path / "nope", data)) == 2
    [line] = capsys.readouterr().out.strip().splitlines()
    assert "nope" in line and "Traceback" not in line
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(args(empty, data)) == 2
    [line] = capsys.readouterr().out.strip().splitlines()
    assert "session.yml" in line and "Traceback" not in line
    assert not data.exists()


def test_the_cli_lists_build() -> None:
    parser = build_parser()
    assert "build" in parser.format_help()
    parsed = parser.parse_args([
        "build", "--session", "data/raw/2026-09-25-mvp", "--product", "mochi-vanilla",
        "--data", "data", "--name", "mvp-2026-09-25",
    ])
    assert parsed.product == "mochi-vanilla" and parsed.name == "mvp-2026-09-25"
    assert parsed.data == "data"
    defaults = parser.parse_args(["build", "--session", "s", "--product", "mochi"])
    assert defaults.data == "data" and defaults.name is None
    for bad in (["--product", "Mochi"], ["--product", "mochi--v"], ["--name", "MVP-1"],
                ["--name", "mvp/1"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["build", "--session", "s", *bad])
