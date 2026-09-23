"""`qs import-phone`: phone photos into the session, and the phone's lock test (ADR-0022).

Fixture photos are small JPEGs written by Pillow with EXIF into `tmp_path`; no real photo is
in git. The grey card sits in one 16x16 JPEG block, saved 4:4:4, so its reading is exact.
"""

import argparse
import csv
import errno
import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image, TiffImagePlugin

import qs.cmd.import_phone
import qs.session
from qs.cmd.import_phone import main, redo_list
from qs.cmd.split_plan import shots_text
from qs.phone import decode
from qs.session import write_plan, write_shot_list
from qs.splitplan import Plan, Shot, make_plan, shot_list

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"
SESSION = "2026-09-25-mvp"
W, H = 64, 48
CARD = (slice(0, 16), slice(48, 64))  # rows, columns of the grey card: ROI [48, 0, 16, 16]
LEAN = 265
SESSION_KEYS = [
    "session_id", "kind", "geometries", "product", "purchased_at", "started_at", "ended_at",
    "camera", "reference_patch", "split_seed", "units", "damage", "notes",
]  # docs/13 §8, in its order
START = datetime(2026, 9, 25, 10, 0, 0)


@dataclass(frozen=True)
class Tags:
    """One photo's EXIF; None leaves the tag out."""

    exposure: tuple[int, int] | None = (1, 50)
    iso: int | None = 100
    wb: int | None = 1
    focal: tuple[int, int] | None = (47, 10)
    zoom: tuple[int, int] | None = None
    make: str | None = "Xiaomi"
    model: str | None = "Redmi Note 12"
    orientation: int | None = 1
    offset: str | None = "+03:00"
    size: tuple[int, int] = (W, H)
    subsec: str | None = None


LOCKED = Tags()


def write_photo(path: Path, n: int, taken: datetime, tags: Tags = LOCKED,
                card: int = 128) -> None:
    """A random frame, the grey card at `card`, as the phone would store it with `tags`."""
    width, height = tags.size
    shown = np.random.default_rng(n).integers(0, 256, (height, width, 3), dtype=np.uint8)
    shown[CARD] = card
    image = Image.fromarray(shown)
    if tags.orientation == 6:  # stored a quarter turn back; the viewer turns it 90° clockwise
        image = image.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    sub = exif.get_ifd(0x8769)
    for ifd, tag, value in (
        (exif, 0x010F, tags.make), (exif, 0x0110, tags.model), (exif, 0x0112, tags.orientation),
        (sub, 0x8827, tags.iso), (sub, 0xA403, tags.wb), (sub, 0x9011, tags.offset),
        (sub, 0x9291, tags.subsec),
    ):
        if value is not None:
            ifd[tag] = value
    for tag, pair in ((0x829A, tags.exposure), (0x920A, tags.focal), (0xA404, tags.zoom)):
        if pair is not None:
            sub[tag] = TiffImagePlugin.IFDRational(*pair)
    sub[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
    image.save(path, "JPEG", exif=exif, quality=95, subsampling=0)


def shoot(folder: Path, count: int, tags: dict[int, Tags] | None = None, every: Tags = LOCKED,
          cards: dict[int, int] | None = None, gaps: dict[int, int] | None = None) -> Path:
    """`count` photos IMG_0001.jpg … 5 s apart; `tags`, `cards` and `gaps` (extra seconds
    before a photo) change single photos, by number from 1."""
    folder.mkdir(parents=True, exist_ok=True)
    taken = START
    for n in range(1, count + 1):
        taken += timedelta(seconds=5 + (gaps or {}).get(n, 0))
        write_photo(folder / f"IMG_{n:04d}.jpg", n, taken, (tags or {}).get(n, every),
                    (cards or {}).get(n, 128))
    return folder


def phone_config(tmp_path: Path, camera_type: str = "phone") -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    for old, new in (
        ("type: webcam", f"type: {camera_type}"), ("width: 1920", f"width: {W}"),
        ("height: 1080", f"height: {H}"), ("exposure_us: 10000", "exposure_us: 20000"),
        ("[1780, 40, 100, 100]", "[48, 0, 16, 16]"), ("[360, 0, 1200, 1080]", "[0, 0, 48, 48]"),
    ):
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / f"{camera_type}.yml"
    path.write_text(text, encoding="utf-8")
    return path


def plan_session(root: Path, printed: tuple[str, ...] = ("G2",)) -> Plan:
    """The lean plan, and the photo list printed for each geometry in `printed`."""
    plan = make_plan("lean", 20260925)
    write_plan(root / SESSION, plan, "sheet\n")
    for geometry in printed:
        write_shot_list(root / SESSION, geometry, shots_text(SESSION, plan, geometry))
    return plan


def args(root: Path, photos: Path, **changes: object) -> argparse.Namespace:
    values: dict[str, object] = dict(
        config=str(phone_config(root)),
        source=str(photos),
        session=SESSION,
        check=False,
        flavour="клубника",
        shop="Магнит, Рабоче-Крестьянская",
        best_before=date(2026, 9, 28),
        purchased_at=datetime(2026, 9, 25, 9, 10, tzinfo=timezone(timedelta(hours=3))),
        brand="Мочи-Дом",
        client_product=False,
        redo=[],
        broken=[],
        utc_offset=None,
        root=str(root / "raw"),
    )
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    plan_session(tmp_path / "raw")
    return tmp_path


def session(root: Path) -> Path:
    return root / "raw" / SESSION


def snapshot(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as log:
        return list(csv.DictReader(log))


def session_yml(root: Path) -> dict[str, object]:
    data = yaml.safe_load((session(root) / "session.yml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shots(plan: Plan) -> list[Shot]:
    return [step for step in shot_list(plan, "G2") if isinstance(step, Shot)]


def pngs(root: Path) -> list[str]:
    return sorted(p.name for p in session(root).glob("*.png"))


# --- the full import -----------------------------------------------------------------------


def test_a_full_lean_session_writes_every_frame_the_originals_and_the_manifest(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN)
    before = snapshot(photos)
    assert main(args(root, photos)) == 0
    wanted = [shot.file for shot in shots(plan_session(root / "again"))]
    assert len(wanted) == LEAN and pngs(root) == sorted(wanted)
    folder = session(root)
    for n, file in enumerate(wanted, start=1):
        saved = cv2.imread(str(folder / file), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(saved, decode(photos / f"IMG_{n:04d}.jpg"))
    assert snapshot(folder / "phone") == before  # every original, byte for byte
    manifest = rows(folder / "import.csv")
    assert list(manifest[0]) == [
        "original", "original_sha256", "exif_time", "frame_file", "png_sha256", "status"]
    assert len(manifest) == LEAN
    for n, (row, file) in enumerate(zip(manifest, wanted, strict=True), start=1):
        assert row["original"] == f"IMG_{n:04d}.jpg"
        assert row["original_sha256"] == sha(photos / row["original"])
        assert row["frame_file"] == file and row["png_sha256"] == sha(folder / file)
        assert row["status"] == "saved"
    assert snapshot(photos) == before  # --from is only read
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert "Copy" in last and SESSION in last and "backup" in last


def test_the_session_record_has_the_docs_keys_and_the_phones_values(root: Path) -> None:
    plan = plan_session(root / "again")
    assert main(args(root, shoot(root / "phone", LEAN))) == 0
    data = session_yml(root)
    assert list(data) == SESSION_KEYS
    assert data["camera"] == {"device": "phone:Xiaomi Redmi Note 12", "exposure_us": 20000,
                              "gain": 100, "wb_kelvin": None, "focus": None, "verified": False}
    assert data["split_seed"] == plan.seed == 20260925
    units = data["units"]
    assert isinstance(units, dict)
    for name in ("control", "train", "calib", "test", "spare"):
        assert units[name] == list(getattr(plan, name))
    assert data["damage"] == dict(plan.damage)
    assert data["geometries"] == ["G2"]
    assert data["started_at"] == datetime(2026, 9, 25, 7, 0, 5, tzinfo=UTC)  # 10:00:05+03
    assert data["started_at"] <= data["ended_at"]  # type: ignore[operator]
    assert data["reference_patch"] == {"start": pytest.approx(128 / 255, abs=0.002),
                                       "end": pytest.approx(128 / 255, abs=0.002)}


def test_the_frame_log_has_one_row_per_png_without_kernel_times(root: Path) -> None:
    assert main(args(root, shoot(root / "phone", LEAN))) == 0
    folder = session(root)
    log = rows(folder / "frames.csv")
    assert sorted(row["file"] for row in log) == pngs(root)
    for row in log:
        assert row["grabbed_at"] == row["capture_ms"] == ""
        assert row["wb_kelvin"] == row["focus"] == ""
        assert (row["exposure_us"], row["gain"], row["verified"]) == ("20000", "100", "false")
        assert row["status"] == "saved"
        assert row["sha256"] == sha(folder / row["file"])
    assert log[0]["wall_utc"] == "2026-09-25T07:00:05+00:00"


# --- the lock, fail closed -----------------------------------------------------------------


@pytest.mark.parametrize(("odd", "field"), [
    (Tags(exposure=(1, 100)), "exposure time"),
    (Tags(iso=200), "ISO"),
    (Tags(wb=0), "white balance"),
    (Tags(focal=(60, 10)), "focal length"),
    (Tags(model="Redmi Note 13"), "model"),
    (Tags(orientation=3), "orientation"),
    (Tags(size=(80, 48)), "frame size"),
    (Tags(zoom=(2, 1)), "digital zoom"),
    (Tags(iso=None), "ISO"),
])
def test_one_photo_off_the_lock_refuses_the_import_and_writes_nothing(
    root: Path, odd: Tags, field: str, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN, tags={7: odd})
    before = snapshot(session(root))
    assert main(args(root, photos)) == 1
    out = capsys.readouterr().out
    assert any(field in line and "IMG_0007.jpg" in line for line in out.splitlines()), out
    assert snapshot(session(root)) == before  # still only the plan


@pytest.mark.parametrize(("every", "field"), [
    (Tags(wb=0), "white balance: must be manual"),
    (Tags(exposure=(1, 60)), "exposure time: must be a whole multiple of 10 ms"),
])
def test_a_setting_wrong_on_every_photo_refuses_the_import(
    root: Path, every: Tags, field: str, capsys: pytest.CaptureFixture[str]
) -> None:
    before = snapshot(session(root))
    assert main(args(root, shoot(root / "phone", LEAN, every=every))) == 1
    assert field in capsys.readouterr().out
    assert snapshot(session(root)) == before


def test_no_offset_in_the_exif_needs_utc_offset(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN, every=Tags(offset=None))
    assert main(args(root, photos)) == 1
    assert "--utc-offset" in capsys.readouterr().out and not (session(root) / "phone").exists()
    assert main(args(root, photos, utc_offset=timezone(timedelta(hours=3)))) == 0
    assert session_yml(root)["started_at"] == datetime(2026, 9, 25, 7, 0, 5, tzinfo=UTC)


# --- the count, redo and broken units ------------------------------------------------------


@pytest.mark.parametrize("count", [LEAN - 1, LEAN + 1])
def test_a_photo_too_few_or_too_many_refuses_with_expected_and_found(
    root: Path, count: int, capsys: pytest.CaptureFixture[str]
) -> None:
    before = snapshot(session(root))
    assert main(args(root, shoot(root / "phone", count))) == 1
    out = capsys.readouterr().out
    assert f"expected {LEAN} photos" in out and f"found {count}" in out
    assert snapshot(session(root)) == before


def test_a_redo_moves_the_spoiled_photo_to_rejected(root: Path) -> None:
    plan = plan_session(root / "again")
    photos = shoot(root / "phone", LEAN + 1)
    assert main(args(root, photos, redo=[[40]])) == 0
    folder = session(root)
    shot40 = shots(plan)[39].file
    assert sorted(p.name for p in (folder / "rejected").iterdir()) == [shot40]
    assert len(pngs(root)) == LEAN
    log = rows(folder / "frames.csv")
    assert [row["status"] for row in log[38:41]] == ["saved", "rejected", "saved"]
    assert [row["file"] for row in log[39:41]] == [f"rejected/{shot40}", shot40]
    # the retake, photo 41, is shot 40; the spoiled photo 40 is kept as it was
    assert np.array_equal(cv2.imread(str(folder / shot40), cv2.IMREAD_UNCHANGED),
                          decode(photos / "IMG_0041.jpg"))
    assert np.array_equal(cv2.imread(str(folder / "rejected" / shot40), cv2.IMREAD_UNCHANGED),
                          decode(photos / "IMG_0040.jpg"))
    assert len(list((folder / "phone").iterdir())) == LEAN + 1
    assert [row["status"] for row in rows(folder / "import.csv")].count("rejected") == 1
    assert "discarded" not in session_yml(root)


def test_a_second_redo_of_the_same_shot_keeps_both_spoiled_photos(root: Path) -> None:
    plan = plan_session(root / "again")
    assert main(args(root, shoot(root / "phone", LEAN + 2), redo=[[40], [40]])) == 0
    stem = shots(plan)[39].file[: -len(".png")]
    assert sorted(p.name for p in (session(root) / "rejected").iterdir()) == [
        f"{stem}.png", f"{stem}_2.png"]


def test_a_train_unit_broken_mid_unit_is_replaced_by_the_first_spare(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = plan_session(root / "again")
    unit, spare = plan.train[3], plan.spare[0]
    # the unit's 3 photos become 2, and the spare adds its own 3
    assert main(args(root, shoot(root / "phone", LEAN + 2), broken=[(unit, 2)])) == 0
    data = session_yml(root)
    units = data["units"]
    assert isinstance(units, dict)
    assert units["train"] == [spare if u == unit else u for u in plan.train]
    assert units["spare"] == list(plan.spare[1:])
    assert data["discarded"] == {unit: [f"{unit}_good_G2_r0.png", f"{unit}_good_G2_r1.png"]}
    assert list(data) == [*SESSION_KEYS[:-1], "discarded", "notes"]
    names = pngs(root)
    assert f"{unit}_good_G2_r2.png" not in names
    assert {f"{spare}_good_G2_r{d}.png" for d in (0, 1, 2)} <= set(names)
    log = {row["file"]: row["status"] for row in rows(session(root) / "frames.csv")}
    assert log[f"{unit}_good_G2_r0.png"] == log[f"{unit}_good_G2_r1.png"] == "discarded"
    assert f"{spare} took its place in train" in capsys.readouterr().out


def test_a_test_unit_broken_after_its_good_frames_is_discarded_whole(root: Path) -> None:
    plan = plan_session(root / "again")
    unit, spare = plan.test[0], plan.spare[0]
    # 6 photos (good, damage, dmg) become 3, and the spare adds 6
    assert main(args(root, shoot(root / "phone", LEAN + 3), broken=[(unit, 3)])) == 0
    data = session_yml(root)
    damage = data["damage"]
    assert isinstance(damage, dict)
    assert damage[spare] == plan.damage[unit] and unit not in damage
    assert data["discarded"] == {unit: [f"{unit}_good_G2_r{d}.png" for d in (0, 1, 2)]}
    names = set(pngs(root))
    assert {f"{spare}_{s}_G2_r{d}.png" for s in ("good", "dmg") for d in (0, 1, 2)} <= names
    assert not any(name.startswith(f"{unit}_dmg") for name in names)


def test_with_no_spare_left_the_split_is_one_short_and_says_so(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = plan_session(root / "again")
    four = [unit for unit in plan.order if unit in plan.train][:4]
    broken = [(unit, 1) for unit in four]  # 4 breaks, 3 spares in a lean plan
    # 4 units lose 2 photos each; 3 spares add 3 each
    assert main(args(root, shoot(root / "phone", LEAN - 8 + 9), broken=broken)) == 0
    units = session_yml(root)["units"]
    assert isinstance(units, dict)
    assert len(units["train"]) == len(plan.train) - 1 and units["spare"] == []
    assert f"{four[3]} broke and no spare was left — train is one short" in (
        capsys.readouterr().out)


def broken_first_test_unit(plan: Plan) -> tuple[str, str, int]:
    """The first test unit in the order, broken after 2 of its 6 photos: the unit, the spare
    that takes its place, and the unit's first row on the printed list (from 1)."""
    unit = next(u for u in plan.order if u in plan.damage)
    row = next(n for n, s in enumerate(shots(plan), start=1) if s.frame.unit == unit)
    return unit, plan.spare[0], row


def test_redo_k_after_a_broken_unit_is_row_k_of_the_printed_list(root: Path) -> None:
    plan = plan_session(root / "again")
    unit, _, row = broken_first_test_unit(plan)
    # a later test unit's dmg r1: after the broken unit, the paper and the as-shot list differ
    later = next(u for u in plan.order[plan.order.index(unit) + 1:] if u in plan.damage)
    k = next(n for n, s in enumerate(shots(plan), start=1)
             if s.file == f"{later}_dmg_G2_r1.png")
    assert k > row + 6
    # 6 photos become 2, the spare adds 6, and the redo 1: before row k, 2 more photos
    photos = shoot(root / "phone", LEAN + 3)
    assert main(args(root, photos, broken=[(unit, 2)], redo=[[k]])) == 0
    folder, file = session(root), f"{later}_dmg_G2_r1.png"
    assert sorted(p.name for p in (folder / "rejected").iterdir()) == [file]
    assert np.array_equal(cv2.imread(str(folder / "rejected" / file), cv2.IMREAD_UNCHANGED),
                          decode(photos / f"IMG_{k + 2:04d}.jpg"))
    assert np.array_equal(cv2.imread(str(folder / file), cv2.IMREAD_UNCHANGED),
                          decode(photos / f"IMG_{k + 3:04d}.jpg"))
    # every frame of the later unit is the photo the paper says it is, shifted by 2
    for n, shot in enumerate(shots(plan), start=1):
        if shot.frame.unit == later and n < k:
            assert np.array_equal(
                cv2.imread(str(folder / shot.file), cv2.IMREAD_UNCHANGED),
                decode(photos / f"IMG_{n + 2:04d}.jpg"))


def test_a_spares_photo_has_no_printed_row_and_is_redone_by_its_frame_name(root: Path) -> None:
    plan = plan_session(root / "again")
    unit, spare, row = broken_first_test_unit(plan)
    photos = shoot(root / "phone", LEAN + 3)
    # the broken unit's 2 photos are rows row, row+1; the spare's good r0, r1 follow them
    assert main(args(root, photos, broken=[(unit, 2)],
                     redo=[redo_list(f"{spare}_good_G2_r1")])) == 0
    folder, file = session(root), f"{spare}_good_G2_r1.png"
    assert sorted(p.name for p in (folder / "rejected").iterdir()) == [file]
    assert np.array_equal(cv2.imread(str(folder / file), cv2.IMREAD_UNCHANGED),
                          decode(photos / f"IMG_{row + 4:04d}.jpg"))


@pytest.mark.parametrize("redo", ["row", "u099_good_G2_r0", "beyond"])
def test_a_redo_of_a_shot_that_was_never_taken_refuses_and_writes_nothing(
    root: Path, redo: str, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = plan_session(root / "again")
    unit, _, row = broken_first_test_unit(plan)
    # row+2 is the broken unit's good r2: printed, but never shot
    given = {"row": str(row + 2), "beyond": str(LEAN + 1)}.get(redo, redo)
    before = snapshot(session(root))
    assert main(args(root, shoot(root / "phone", LEAN + 3), broken=[(unit, 2)],
                     redo=[redo_list(given)])) == 1
    out = capsys.readouterr().out
    assert f"--redo {given}" in out and "nothing written" in out
    if redo == "row":
        assert f"{unit}_good_G2_r2.png" in out
    assert snapshot(session(root)) == before


def test_the_mid_session_check_names_the_next_photo_by_its_printed_row(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = plan_session(root / "again")
    unit, spare, row = broken_first_test_unit(plan)
    # everything before the unit, its 2 photos, and 5 of the spare's 6
    photos = shoot(root / "phone", row - 1 + 2 + 5)
    capsys.readouterr()
    assert main(check_args(root, photos, SESSION, broken=[(unit, 2)])) == 0
    out = capsys.readouterr().out
    assert f"next: {spare} dmg 240° — a spare's photo, not on the printed list" in out
    photos = shoot(root / "more", row - 1 + 2 + 6)
    assert main(check_args(root, photos, SESSION, broken=[(unit, 2)])) == 0
    following = shots(plan)[row + 5].frame  # printed row row+6, the unit after the broken one
    assert f"next: #{row + 6} {following.unit} {following.state}" in capsys.readouterr().out


def test_a_photo_copied_twice_refuses_naming_both_and_writes_nothing(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN - 1)  # one photo short, and one copied twice
    (photos / "IMG_0010 (1).jpg").write_bytes((photos / "IMG_0010.jpg").read_bytes())
    before = snapshot(session(root))
    assert main(args(root, photos)) == 1
    out = capsys.readouterr().out
    assert "IMG_0010 (1).jpg" in out and "IMG_0010.jpg" in out and "nothing written" in out
    assert snapshot(session(root)) == before
    assert main(check_args(root, photos, SESSION)) == 1
    assert "IMG_0010 (1).jpg" in capsys.readouterr().out


@pytest.mark.parametrize(("subsec", "code"), [("123", 1), (None, 0)])
def test_two_photos_taken_at_the_same_sub_second_refuse_but_the_same_second_does_not(
    root: Path, subsec: str | None, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    # photo 11 is stamped with photo 10's time; with sub-seconds that cannot be two shots
    photos = shoot(root / "phone", LEAN, every=Tags(subsec=subsec), gaps={11: -5})
    assert main(args(root, photos)) == code
    if code:
        out = capsys.readouterr().out
        assert "IMG_0010.jpg" in out and "IMG_0011.jpg" in out and "sub-second" in out
        assert not (session(root) / "phone").exists()


def fail_at(monkeypatch: pytest.MonkeyPatch, module: object, name: str, nth: int) -> None:
    """`module.name` raises ENOSPC on its nth call, and works before it."""
    real = getattr(module, name)
    calls = iter(range(1, 10**6))

    def failing(*a: object, **k: object) -> object:
        if next(calls) == nth:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(*a, **k)

    monkeypatch.setattr(module, name, failing)


def test_a_disk_full_part_way_adds_nothing_to_the_session_and_names_what_to_delete(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN)
    folder = session(root)
    planned = snapshot(folder)
    fail_at(monkeypatch, qs.session, "_write_png", 100)
    assert main(args(root, photos)) == 1
    staging = folder / ".import"
    out = capsys.readouterr().out
    assert "No space left" in out and f"delete {staging}" in out
    assert {k: v for k, v in snapshot(folder).items() if not k.startswith(".import")} == planned
    monkeypatch.undo()
    stale = snapshot(folder)
    assert main(args(root, photos)) == 1  # a stale staging folder refuses, and names it
    assert f"delete {staging}" in capsys.readouterr().out
    assert snapshot(folder) == stale  # not one staged frame deleted or overwritten
    for path in sorted(staging.rglob("*"), reverse=True):  # Ali deletes it, as told
        path.rmdir() if path.is_dir() else path.unlink()
    staging.rmdir()
    assert main(args(root, photos)) == 0
    assert len(pngs(root)) == LEAN and not staging.exists()


def test_an_import_stopped_while_moving_in_says_how_to_finish_it_by_moving(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN)
    folder = session(root)
    fail_at(monkeypatch, qs.session.os, "link", 100)
    assert main(args(root, photos)) == 1
    staging = folder / ".import"
    assert "Move everything in" in capsys.readouterr().out
    monkeypatch.undo()
    assert 0 < len(pngs(root)) < LEAN and not (folder / "session.yml").exists()
    stale = snapshot(folder)
    assert main(args(root, photos)) == 1
    assert f"Move everything in {staging}" in capsys.readouterr().out
    assert snapshot(folder) == stale
    for path in staging.iterdir():  # Ali moves them, as told: none is in the session yet
        assert not (folder / path.name).exists()
        path.rename(folder / path.name)
    staging.rmdir()
    assert len(pngs(root)) == LEAN and "session_id" in session_yml(root)


def test_a_finished_import_leaves_no_staging_folder(root: Path) -> None:
    assert main(args(root, shoot(root / "phone", LEAN))) == 0
    assert not (session(root) / ".import").exists()
    assert os.listdir(session(root))  # the session holds the import itself


@pytest.mark.parametrize("stage", ["reading", "writing", "moving in"])
def test_ctrl_c_prints_one_line_with_what_to_do_and_no_traceback(
    root: Path, stage: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN)
    folder = session(root)
    target, name = {"reading": (qs.cmd.import_phone, "decode"),
                    "writing": (qs.session, "_write_png"),
                    "moving in": (qs.session.os, "link")}[stage]
    real, calls = getattr(target, name), iter(range(1, 10**6))

    def interrupted(*a: object, **k: object) -> object:
        if next(calls) == 100:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(target, name, interrupted)
    assert main(args(root, photos)) == 1  # returned, not raised: no traceback
    [line] = capsys.readouterr().out.strip().splitlines()
    assert line.startswith("import: stopped by the operator — ")
    assert {"reading": f"nothing was written to {folder}",
            "writing": f"delete {folder / '.import'}",
            "moving in": f"Move everything in {folder / '.import'}"}[stage] in line


def test_ctrl_c_during_the_check_says_nothing_was_written(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", 40)

    def interrupted(path: Path) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(qs.cmd.import_phone, "decode", interrupted)
    before = snapshot(root / "raw")
    assert main(check_args(root, photos, SESSION)) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == (
        "check: stopped by the operator — nothing was written")
    assert snapshot(root / "raw") == before


# --- order, formats, config ----------------------------------------------------------------


def test_time_order_against_file_name_order_refuses_naming_both(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN)
    (photos / "IMG_0010.jpg").rename(photos / "IMG_0010x.jpg")
    (photos / "IMG_0011.jpg").rename(photos / "IMG_0010.jpg")
    (photos / "IMG_0010x.jpg").rename(photos / "IMG_0011.jpg")
    before = snapshot(session(root))
    assert main(args(root, photos)) == 1
    out = capsys.readouterr().out
    assert "IMG_0010.jpg" in out and "IMG_0011.jpg" in out
    assert snapshot(session(root)) == before


def test_a_heic_file_in_the_folder_cannot_run(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", 3)
    (photos / "IMG_0004.HEIC").write_bytes(b"not read")
    assert main(args(root, photos)) == 2
    [line] = capsys.readouterr().out.strip().splitlines()
    assert "IMG_0004.HEIC" in line and "ADR-0022" in line


def test_a_webcam_config_cannot_run_and_names_adr_0022(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = phone_config(root, "webcam")
    assert main(args(root, shoot(root / "phone", 3), config=str(config))) == 2
    [line] = capsys.readouterr().out.strip().splitlines()
    assert "ADR-0022" in line and "webcam" in line


def test_the_exif_orientation_is_applied_exactly_once(root: Path) -> None:
    plan = plan_session(root / "again")
    photos = shoot(root / "phone", LEAN, every=Tags(orientation=6, size=(W, H)))
    assert main(args(root, photos)) == 0
    with Image.open(photos / "IMG_0001.jpg") as image:
        assert image.size == (H, W)  # stored on its side
        stored = np.asarray(image.convert("RGB"))[:, :, ::-1]
    saved = cv2.imread(str(session(root) / shots(plan)[0].file), cv2.IMREAD_UNCHANGED)
    assert saved.shape == (H, W, 3)
    assert np.array_equal(saved, np.rot90(stored, k=-1))  # one quarter turn clockwise


def test_a_test_unit_whose_damage_gap_is_not_its_widest_is_warned_about(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = plan_session(root / "again")
    unit = plan.test[0]
    first = next(n for n, s in enumerate(shots(plan), start=1) if s.frame.unit == unit)
    assert main(args(root, shoot(root / "phone", LEAN, gaps={first + 1: 300}))) == 0
    warnings = [line for line in capsys.readouterr().out.splitlines()
                if line.startswith("WARNING")]
    assert len(warnings) == 1 and unit in warnings[0]


def test_a_second_import_into_the_same_session_is_refused_untouched(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(root / "phone", LEAN)
    assert main(args(root, photos)) == 0
    before = snapshot(session(root))
    capsys.readouterr()
    assert main(args(root, photos)) == 1
    assert "already" in capsys.readouterr().out
    assert snapshot(session(root)) == before


def test_a_session_without_a_plan_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(tmp_path, shoot(tmp_path / "phone", 3))) == 1
    assert "qs split-plan" in capsys.readouterr().out
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize(("printed", "named"), [
    (("G3",), "shots-G3.md"),  # printed under G3, the config says G2
    ((), "no photo list"),
    (("G2", "G3"), "shots-G3.md"),  # which one is in Ali's hand is a guess
])
def test_a_config_geometry_that_is_not_the_printed_lists_refuses_and_writes_nothing(
    tmp_path: Path, printed: tuple[str, ...], named: str, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_session(tmp_path / "raw", printed)
    photos = shoot(tmp_path / "phone", LEAN)
    before = snapshot(tmp_path / "raw")
    assert main(args(tmp_path, photos)) == 1
    out = capsys.readouterr().out
    assert "lighting.geometry G2" in out and named in out
    assert snapshot(tmp_path / "raw") == before
    assert main(check_args(tmp_path, photos, SESSION)) == 1
    assert named in capsys.readouterr().out
    assert snapshot(tmp_path / "raw") == before


# --- --check ---------------------------------------------------------------------------------


def check_args(root: Path, photos: Path, session_id: str | None = None,
               **changes: object) -> argparse.Namespace:
    return args(root, photos, check=True, session=session_id, flavour=None, shop=None,
                best_before=None, purchased_at=None, brand=None, **changes)


def test_the_day_zero_check_passes_on_10_locked_photos_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(tmp_path / "phone", 10, gaps={n: 7 for n in range(2, 11)})  # 2 minutes
    phone_config(tmp_path)
    before = snapshot(tmp_path)
    assert main(check_args(tmp_path, photos)) == 0
    out = capsys.readouterr().out
    assert "PASS EXIF lock: 10 photos" in out and "PASS reference patch" in out
    assert snapshot(tmp_path) == before


def test_the_day_zero_check_fails_on_a_card_2_percent_brighter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(tmp_path / "phone", 10, cards={6: 131})  # 128 → 131, +2.3%
    phone_config(tmp_path)
    before = snapshot(tmp_path)
    assert main(check_args(tmp_path, photos)) == 1
    out = capsys.readouterr().out
    assert "PASS EXIF lock" in out and "FAIL reference patch" in out
    assert snapshot(tmp_path) == before


def test_the_day_zero_check_fails_on_a_photo_off_the_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photos = shoot(tmp_path / "phone", 10, tags={4: Tags(iso=400)})
    phone_config(tmp_path)
    assert main(check_args(tmp_path, photos)) == 1
    assert "FAIL ISO" in capsys.readouterr().out


def test_the_mid_session_check_maps_the_photos_so_far_and_names_the_next(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = plan_session(root / "again")
    photos = shoot(root / "phone", 40)
    phone_config(root)
    before = snapshot(root)
    assert main(check_args(root, photos, SESSION)) == 0
    out = capsys.readouterr().out
    following = shots(plan)[40].frame
    assert f"40 of {LEAN} photos mapped — next: #41 {following.unit} {following.state}" in out
    assert "drift" in out
    assert snapshot(root) == before


# --- the command line ----------------------------------------------------------------------


def test_the_cli_lists_import_phone_and_parses_its_options(
    capsys: pytest.CaptureFixture[str]
) -> None:
    from qs.__main__ import build_parser

    parser = build_parser()
    assert "import-phone" in parser.format_help()
    parsed = parser.parse_args([
        "import-phone", "--config", "c.yml", "--from", "data/phone/x", "--session", SESSION,
        "--flavour", "манго", "--shop", "Ашан", "--best-before", "2026-09-28",
        "--purchased-at", "2026-09-25T09:10+03:00", "--brand", "X", "--redo", "40,41",
        "--redo", "40", "--redo", "u049_dmg_G2_r1,u053_good_G2_r0.png", "--broken", "u012:2",
        "--utc-offset", "+03:00",
    ])
    assert parsed.source == "data/phone/x"
    assert parsed.redo == [[40, 41], [40], ["u049_dmg_G2_r1.png", "u053_good_G2_r0.png"]]
    assert parsed.broken == [("u012", 2)]
    assert parsed.utc_offset == timezone(timedelta(hours=3))
    assert parsed.check is False and parsed.root == "data/raw"
    for bad in (["--redo", "0"], ["--redo", "u049_dmg"], ["--redo", "40,u049"],
                ["--broken", "u012"], ["--utc-offset", "3"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["import-phone", "--config", "c", "--from", "d", *bad])


def test_an_import_without_the_product_cannot_run(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(root, shoot(root / "phone", 3), brand=None)) == 2
    assert "--client-product" in capsys.readouterr().out

