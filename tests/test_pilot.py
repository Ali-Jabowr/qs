"""The lighting pilot (docs/13 §5, ADR-0013) on the phone path (ADR-0022): the plan, its shot
list, the printed list and the import."""

import argparse
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from qs.__main__ import build_parser, run
from qs.cmd import import_phone, split_plan
from qs.names import parse_name
from qs.session import read_session_plan
from qs.splitplan import Prompt, Shot, make_pilot_plan, pilot_shot_list
from test_import_phone import LOCKED, SESSION_KEYS, Tags, rows, shoot
from test_import_phone import args as import_args

G123 = ("G1", "G2", "G3")
PILOT = "2026-09-24-pilot"


def plan_args(root: Path, size: str = "standard", seed: int | None = 20260924,
              geometries: tuple[str, ...] | None = None) -> argparse.Namespace:
    return argparse.Namespace(session=PILOT, size=size, seed=seed, root=str(root),
                              shot_list=False, geometry=None, kind="pilot",
                              geometries=geometries)


def list_args(root: Path, geometry: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(session=PILOT, size=None, seed=None, root=str(root),
                              shot_list=True, geometry=geometry, kind="dataset",
                              geometries=None)


# --- the plan ------------------------------------------------------------------------------


@pytest.mark.parametrize(("size", "units", "dent", "tear"),
                         [("standard", 10, 2, 2), ("lean", 6, 1, 1)])
def test_pilot_plan_counts_over_50_seeds(size: str, units: int, dent: int, tear: int) -> None:
    for seed in range(50):
        plan = make_pilot_plan(size, seed)
        assert plan.units == tuple(f"p{n:02d}" for n in range(1, units + 1))
        assert Counter(plan.damage.values()) == {"dent": dent, "tear": tear}
        assert set(plan.damage) < set(plan.units)
        assert len(plan.good) == units - dent - tear
        assert sorted(plan.order) == list(plan.units)
        assert plan.geometries == G123


def test_the_same_seed_gives_the_same_pilot_plan_and_seeds_differ() -> None:
    assert make_pilot_plan("standard", 20260924) == make_pilot_plan("standard", 20260924)
    plans = {(tuple(make_pilot_plan("standard", s).damage), make_pilot_plan("standard", s).order)
             for s in range(20)}
    assert len(plans) > 1


# --- the shot list -------------------------------------------------------------------------


def photos(plan_steps: tuple[Shot | Prompt, ...]) -> list[Shot]:
    return [step for step in plan_steps if isinstance(step, Shot)]


@pytest.mark.parametrize(("size", "geometries", "total"), [
    ("standard", G123, 126), ("lean", G123, 72), ("standard", ("G1", "G3"), 84),
])
def test_pilot_shot_totals_and_every_name_parses_and_is_unique(
    size: str, geometries: tuple[str, ...], total: int
) -> None:
    for seed in range(20):
        shots = photos(pilot_shot_list(make_pilot_plan(size, seed, geometries)))
        assert len(shots) == total
        files = [shot.file for shot in shots]
        assert len(set(files)) == total
        for shot in shots:
            assert parse_name(shot.file) == shot.frame
            assert shot.file.startswith("p") and shot.frame.pass_ is None


@pytest.mark.parametrize("geometries", [G123, ("G1", "G3")])
def test_every_good_shot_by_geometry_comes_before_any_dmg_and_the_lamps_move_per_geometry(
    geometries: tuple[str, ...]
) -> None:
    for seed in range(30):
        plan = make_pilot_plan("standard", seed, geometries)
        steps = pilot_shot_list(plan)
        states = [(s.frame.state, s.frame.geometry) for s in photos(steps)]
        good = [g for state, g in states if state == "good"]
        dmg = [g for state, g in states if state == "dmg"]
        assert states == [("good", g) for g in good] + [("dmg", g) for g in dmg]
        per = 3 * len(plan.units)
        assert good == [g for g in geometries for _ in range(per)]
        assert dmg == [g for g in geometries for _ in range(3 * len(plan.damage))]
        lamps = [s for s in steps if isinstance(s, Prompt) and s.kind == "lamps"]
        assert len(lamps) == 2 * len(geometries)
        for step, following in zip(steps, steps[1:], strict=False):
            if isinstance(step, Prompt) and step.kind == "lamps":
                assert isinstance(following, Shot)
                assert following.frame.geometry in step.instruction


def test_units_follow_the_seeded_order_three_rotations_each_and_only_the_damaged_are_dmg() -> None:
    plan = make_pilot_plan("standard", 20260924)
    shots = photos(pilot_shot_list(plan))
    first = shots[: 3 * len(plan.units)]
    assert [s.frame.unit for s in first[::3]] == list(plan.order)
    assert [s.frame.rotation for s in first] == [0, 1, 2] * len(plan.units)
    dmg = [s for s in shots if s.frame.state == "dmg"][: 3 * len(plan.damage)]
    assert [s.frame.unit for s in dmg[::3]] == [u for u in plan.order if u in plan.damage]


def test_one_damage_step_names_each_damaged_unit_with_its_recipe() -> None:
    plan = make_pilot_plan("standard", 7)
    steps = pilot_shot_list(plan)
    damage = [s for s in steps if isinstance(s, Prompt) and s.kind == "damage"]
    assert len(damage) == 1
    at = steps.index(damage[0])
    assert all(isinstance(s, Shot) and s.frame.state == "good" for s in steps[:at]
               if isinstance(s, Shot))
    for unit, kind in plan.damage.items():
        assert unit in damage[0].instruction and kind in damage[0].instruction
    assert "pressed straight down about 3 mm" in damage[0].instruction
    assert "filling shows" in damage[0].instruction


# --- qs split-plan --kind pilot ------------------------------------------------------------


def test_split_plan_kind_pilot_writes_a_pilot_plan_that_reads_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert split_plan.main(plan_args(tmp_path)) == 0
    folder = tmp_path / PILOT
    assert read_session_plan(folder) == make_pilot_plan("standard", 20260924)
    assert "kind: pilot" in (folder / "plan.yml").read_text(encoding="utf-8")
    sheet = (folder / "plan.md").read_text(encoding="utf-8")
    out = capsys.readouterr().out
    plan = make_pilot_plan("standard", 20260924)
    for unit in plan.units:
        assert unit in sheet and unit in out
    for unit, kind in plan.damage.items():
        assert any(unit in line and kind in line for line in sheet.splitlines())
    assert "G1, G2, G3" in sheet


def test_one_lamp_in_the_house_plans_g1_and_g3(tmp_path: Path) -> None:
    assert split_plan.main(plan_args(tmp_path, "lean", 3, ("G1", "G3"))) == 0
    plan = read_session_plan(tmp_path / PILOT)
    assert plan == make_pilot_plan("lean", 3, ("G1", "G3"))


@pytest.mark.parametrize("given", ["G2", "G1,G2", "G3,G1", "G1,G2,G3,G3", "G4"])
def test_other_geometries_are_refused_with_exit_2_naming_the_allowed_lists(
    tmp_path: Path, given: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        run(["split-plan", "--session", PILOT, "--kind", "pilot", "--size", "standard",
             "--geometries", given, "--root", str(tmp_path)])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "G1,G2,G3" in err and "G1,G3" in err and "Traceback" not in err
    assert not (tmp_path / PILOT).exists()


def test_geometries_belong_to_the_pilot_and_the_default_kind_is_the_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parsed = build_parser().parse_args(
        ["split-plan", "--session", PILOT, "--size", "lean"])
    assert parsed.kind == "dataset" and parsed.geometries is None
    args = plan_args(tmp_path, geometries=G123)
    args.kind = "dataset"
    assert split_plan.main(args) == 2
    assert "--geometries" in capsys.readouterr().out
    assert not (tmp_path / PILOT).exists()


# --- the printed pilot list ----------------------------------------------------------------

NUMBERED = re.compile(r"\| (\d+) \| +\| +\| (.*) \| (\w+) \| (\d+)° \| `([^`]+)` \|$")


def printed(root: Path, size: str = "standard", seed: int = 20260924,
            geometries: tuple[str, ...] | None = None) -> list[str]:
    assert split_plan.main(plan_args(root, size, seed, geometries)) == 0
    assert split_plan.main(list_args(root)) == 0
    return (root / PILOT / "shots-pilot.md").read_text(encoding="utf-8").splitlines()


def test_the_printed_pilot_list_has_126_rows_and_row_k_is_shot_k(tmp_path: Path) -> None:
    lines = printed(tmp_path)
    found = [NUMBERED.fullmatch(line) for line in lines]
    rows = [m for m in found if m is not None]
    assert [int(m[1]) for m in rows] == list(range(1, 127))
    shots = photos(pilot_shot_list(make_pilot_plan("standard", 20260924)))
    assert [m[5] for m in rows] == [shot.file for shot in shots]
    assert rows[0][2] == f"{shots[0].frame.unit} · cell {int(shots[0].frame.unit[1:])}"
    head = "\n".join(lines[:8])
    for fact in (PILOT, "seed 20260924", "pilot", "G1, G2, G3", "126 photos"):
        assert fact in head
    assert "never delete a photo on the phone" in head.lower()


def test_lamp_and_damage_rows_are_bold_and_one_check_row_follows_the_first_geometry(
    tmp_path: Path,
) -> None:
    lines = printed(tmp_path)
    steps = [line for line in lines if line.startswith("| |")]
    lamps = [line for line in steps if "Lamps to" in line]
    assert len(lamps) == 6 and all("**" in line for line in lamps)
    assert [line.split("Lamps to ")[1][:2] for line in lamps] == [*G123, *G123]
    [damage] = [line for line in steps if "damage each" in line]
    assert "**" in damage
    [check] = [i for i, line in enumerate(lines) if "CHECK" in line]
    last_g1_good = max(i for i, line in enumerate(lines) if "_good_G1_r" in line)
    assert check == last_g1_good + 1
    assert "qs import-phone --check" in lines[check] and "30 photos" in lines[check]


def test_the_pilot_list_needs_no_geometry_and_refuses_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    printed(tmp_path, "lean", 1, ("G1", "G3"))
    text = (tmp_path / PILOT / "shots-pilot.md").read_text(encoding="utf-8")
    assert "**48 photos**" in text
    capsys.readouterr()
    assert split_plan.main(list_args(tmp_path, geometry="G2")) == 2
    [line] = capsys.readouterr().out.strip().splitlines()
    assert "--geometry" in line and "G1, G3" in line
    assert not (tmp_path / PILOT / "shots-G2.md").exists()


def test_the_pilot_list_takes_kind_pilot_but_a_dataset_plan_refuses_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert split_plan.main(plan_args(tmp_path / "p")) == 0
    args = list_args(tmp_path / "p")
    args.kind = "pilot"
    assert split_plan.main(args) == 0
    assert (tmp_path / "p" / PILOT / "shots-pilot.md").is_file()
    dataset = plan_args(tmp_path / "d", "lean")
    dataset.kind = "dataset"
    assert split_plan.main(dataset) == 0
    args = list_args(tmp_path / "d", geometry="G2")
    args.kind = "pilot"
    capsys.readouterr()
    assert split_plan.main(args) == 2
    assert "dataset plan" in capsys.readouterr().out
    assert not list((tmp_path / "d" / PILOT).glob("shots-*.md"))


# --- qs import-phone on a pilot plan -------------------------------------------------------


def pilot_import(root: Path, size: str = "lean", seed: int = 20260924,
                 geometries: tuple[str, ...] | None = None) -> list[Shot]:
    """Plan the pilot under `root/raw`; its shots, in order."""
    assert split_plan.main(plan_args(root / "raw", size, seed, geometries)) == 0
    return photos(pilot_shot_list(make_pilot_plan(size, seed, geometries or G123)))


def pilot_args(root: Path, photo_dir: Path, **changes: object) -> argparse.Namespace:
    return import_args(root, photo_dir, session=PILOT, **changes)


def pilot_yml(root: Path) -> dict[str, object]:
    data = yaml.safe_load((root / "raw" / PILOT / "session.yml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def session_pngs(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "raw" / PILOT).glob("*.png"))


def test_a_full_pilot_import_writes_the_lists_files_and_a_pilot_session_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shots = pilot_import(tmp_path)
    assert len(shots) == 72
    assert import_phone.main(pilot_args(tmp_path, shoot(tmp_path / "phone", 72))) == 0
    assert session_pngs(tmp_path) == sorted(shot.file for shot in shots)
    log = rows(tmp_path / "raw" / PILOT / "frames.csv")
    assert [row["file"] for row in log] == [shot.file for shot in shots]
    data = pilot_yml(tmp_path)
    assert list(data) == SESSION_KEYS
    plan = make_pilot_plan("lean", 20260924)
    assert data["kind"] == "pilot"
    assert data["geometries"] == list(G123)
    assert data["units"] == {"pilot": list(plan.units)}
    assert data["damage"] == plan.damage
    assert data["split_seed"] == 20260924
    assert "Copy" in capsys.readouterr().out.strip().splitlines()[-1]


def test_the_plans_geometries_win_over_the_configs(tmp_path: Path) -> None:
    shots = pilot_import(tmp_path, "lean", 5, ("G1", "G3"))  # the config says G2
    assert import_phone.main(pilot_args(tmp_path, shoot(tmp_path / "phone", len(shots)))) == 0
    assert pilot_yml(tmp_path)["geometries"] == ["G1", "G3"]
    assert not any("_G2_" in name for name in session_pngs(tmp_path))


def test_the_exif_lock_holds_across_every_geometry_of_the_pilot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shots = pilot_import(tmp_path)
    k = next(i for i, shot in enumerate(shots, 1) if shot.frame.geometry == "G3")
    odd = {k: Tags(exposure=(1, 100))}  # a new exposure for the diffuse light: refused
    assert import_phone.main(pilot_args(tmp_path, shoot(tmp_path / "phone", 72, tags=odd))) == 1
    assert "exposure time" in capsys.readouterr().out
    assert session_pngs(tmp_path) == []
    assert LOCKED.exposure == (1, 50)


def test_a_redo_in_the_pilot_keeps_the_spoiled_photo(tmp_path: Path) -> None:
    shots = pilot_import(tmp_path)
    assert import_phone.main(pilot_args(tmp_path, shoot(tmp_path / "phone", 73),
                                        redo=[[20]])) == 0
    rejected = tmp_path / "raw" / PILOT / "rejected"
    assert sorted(p.name for p in rejected.iterdir()) == [shots[19].file]
    assert len(session_pngs(tmp_path)) == 72


def test_a_broken_pilot_unit_is_discarded_because_the_pilot_has_no_spares(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = make_pilot_plan("lean", 20260924)
    shots = pilot_import(tmp_path)
    unit = next(u for u in plan.order if u in plan.damage)  # a damaged unit: 18 photos
    # broke after 4 photos: its 3 G1 good and its first G2 good; its 14 others are never shot
    assert import_phone.main(pilot_args(tmp_path, shoot(tmp_path / "phone", 72 - 14),
                                        broken=[(unit, 4)])) == 0
    taken = [shot.file for shot in shots if shot.frame.unit == unit][:4]
    data = pilot_yml(tmp_path)
    assert data["discarded"] == {unit: taken}
    assert data["units"] == {"pilot": [u for u in plan.units if u != unit]}
    assert data["damage"] == {u: d for u, d in plan.damage.items() if u != unit}
    assert list(data) == [*SESSION_KEYS[:-1], "discarded", "notes"]
    log = {row["file"]: row["status"] for row in rows(tmp_path / "raw" / PILOT / "frames.csv")}
    assert [log[file] for file in taken] == ["discarded"] * 4
    assert not any(name.startswith(f"{unit}_dmg") for name in session_pngs(tmp_path))
    out = capsys.readouterr().out
    assert f"{unit} broke" in out and "discarded" in out and "no spares" in out


def test_broken_takes_a_pilot_unit_on_the_command_line(tmp_path: Path) -> None:
    parsed = build_parser().parse_args(
        ["import-phone", "--config", "c.yml", "--from", "d", "--check", "--session", PILOT,
         "--broken", "p03:4"])
    assert parsed.broken == [("p03", 4)]
    assert import_phone.broken_unit("u012:2") == ("u012", 2)


def test_a_dataset_unit_on_a_pilot_plan_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot_import(tmp_path)
    assert import_phone.main(pilot_args(tmp_path, shoot(tmp_path / "phone", 72),
                                        broken=[("u001", 2)])) == 2
    assert "u001" in capsys.readouterr().out
    assert session_pngs(tmp_path) == []


def test_where_am_i_on_a_pilot_names_the_next_lamp_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot_import(tmp_path)
    args = pilot_args(tmp_path, shoot(tmp_path / "phone", 18), check=True, flavour=None,
                      shop=None, best_before=None, purchased_at=None, brand=None)
    capsys.readouterr()
    assert import_phone.main(args) == 0
    out = capsys.readouterr().out
    assert "18 of 72 photos mapped" in out and "first: Lamps to G2" in out
    assert not (tmp_path / "raw" / PILOT / "frames.csv").exists()
