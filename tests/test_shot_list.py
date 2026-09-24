"""`qs split-plan --shot-list`: the printed, numbered photo list for the phone shoot (ADR-0022)."""

import argparse
import re
from pathlib import Path

import pytest

from qs.__main__ import build_parser
from qs.cmd.split_plan import main
from qs.splitplan import Shot, make_plan, shot_list

SESSION = "2026-09-25-mvp"
NUMBERED = re.compile(r"\| (\d+) \| +\| +\| (.*) \| (\w+) \| (\d+)° \| `([^`]+)` \|$")
# 9 control frames per pass × 3 + 3 per good unit + 6 per test unit + 10 placement
PHOTOS = {"lean": 265, "standard": 322}


def plan_args(root: Path, size: str = "lean", seed: int = 20260925) -> argparse.Namespace:
    return argparse.Namespace(session=SESSION, size=size, seed=seed, root=str(root),
                              shot_list=False, geometry=None)


def list_args(root: Path, geometry: str | None = "G2") -> argparse.Namespace:
    return argparse.Namespace(session=SESSION, size=None, seed=None, root=str(root),
                              shot_list=True, geometry=geometry)


def planned(root: Path, size: str = "lean", seed: int = 20260925) -> Path:
    assert main(plan_args(root, size, seed)) == 0
    return root / SESSION


def rows(text: str) -> list[tuple[int, str, str, int, str]]:
    """The numbered rows: (number, unit column, state, degrees, file name)."""
    found = []
    for line in text.splitlines():
        match = NUMBERED.fullmatch(line)
        if match is not None:
            found.append((int(match[1]), match[2], match[3], int(match[4]), match[5]))
    return found


def written(root: Path, geometry: str = "G2") -> str:
    return (root / SESSION / f"shots-{geometry}.md").read_text(encoding="utf-8")


# --- the list ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["lean", "standard"])
def test_one_numbered_row_per_photo_in_shot_list_order_over_50_seeds(
    tmp_path: Path, size: str
) -> None:
    for seed in range(50):
        root = tmp_path / str(seed)
        planned(root, size, seed)
        assert main(list_args(root)) == 0
        found = rows(written(root))
        assert [n for n, *_ in found] == list(range(1, PHOTOS[size] + 1)), seed
        expected = [s.file for s in shot_list(make_plan(size, seed), "G2")
                    if isinstance(s, Shot)]
        assert [file for *_, file in found] == expected, seed


def test_each_row_names_the_unit_its_cell_state_and_rotation(tmp_path: Path) -> None:
    planned(tmp_path)
    assert main(list_args(tmp_path, "G3")) == 0
    found = rows(written(tmp_path, "G3"))
    assert found[0] == (1, "c01 · control", "good", 0, "c01_good_G3_r0_pass1.png")
    unit = make_plan("lean", 20260925).order[0]
    n, what, state, degrees, file = found[9]
    assert (n, state, degrees, file) == (10, "good", 0, f"{unit}_good_G3_r0.png")
    assert what == f"{unit} · cell {int(unit[1:])}"
    assert [r[3] for r in found[9:12]] == [0, 120, 240]
    assert found[-1][1:3] == ("x10 · placement, a unit touching the edge", "edge")


def test_each_damage_row_sits_between_the_good_and_dmg_rows_with_its_recipe(
    tmp_path: Path,
) -> None:
    planned(tmp_path, "standard", 11)
    plan = make_plan("standard", 11)
    assert main(list_args(tmp_path)) == 0
    lines = written(tmp_path).splitlines()
    assert "deformed-oval" in plan.damage.values()
    for unit, damage in plan.damage.items():
        good = [i for i, line in enumerate(lines) if f"`{unit}_good_G2_r" in line]
        dmg = [i for i, line in enumerate(lines) if f"`{unit}_dmg_G2_r" in line]
        assert len(good) == 3 and len(dmg) == 3
        between = lines[good[-1] + 1: dmg[0]]
        assert len(between) == 1, unit
        [row] = between
        assert NUMBERED.fullmatch(row) is None and "**" in row
        assert unit in row and damage in row
        if damage == "deformed-oval":
            assert "squeezed from two sides" in row
        elif damage == "dent":
            assert "pressed straight down about 3 mm" in row


def test_a_check_row_follows_the_last_pass1_and_the_last_pass2_photo(tmp_path: Path) -> None:
    planned(tmp_path)
    assert main(list_args(tmp_path)) == 0
    lines = written(tmp_path).splitlines()
    checks = [i for i, line in enumerate(lines) if "CHECK" in line]
    assert len(checks) == 2
    for pass_, at in zip((1, 2), checks, strict=True):
        last = max(i for i, line in enumerate(lines) if f"_pass{pass_}.png`" in line)
        assert at == last + 1
        assert "USB" in lines[at] and "qs import-phone --check" in lines[at]
        assert NUMBERED.fullmatch(lines[at]) is None


def test_a_brush_row_follows_every_tenth_non_control_unit(tmp_path: Path) -> None:
    planned(tmp_path, "lean", 5)
    assert main(list_args(tmp_path)) == 0
    units_done, last_unit = 0, None
    brushed_at: list[int] = []
    for line in written(tmp_path).splitlines():
        if "brush" in line.lower() and NUMBERED.fullmatch(line) is None:
            brushed_at.append(units_done)
        match = NUMBERED.fullmatch(line)
        if match is not None and match[5].startswith("u") and match[5][:4] != last_unit:
            units_done, last_unit = units_done + 1, match[5][:4]
    assert brushed_at == [10, 20, 30, 40, 50]


def test_the_header_gives_the_session_the_total_and_the_never_delete_rule(
    tmp_path: Path,
) -> None:
    planned(tmp_path, "standard", 77)
    assert main(list_args(tmp_path, "G1")) == 0
    head = "\n".join(written(tmp_path, "G1").splitlines()[:8])
    for fact in (SESSION, "seed 77", "standard", "G1", "322 photos"):
        assert fact in head
    assert "never delete a photo on the phone" in head.lower()
    assert "tick `redo`" in head


# --- the file, written once -----------------------------------------------------------


def test_a_second_run_leaves_the_same_file_and_another_geometry_a_second_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = planned(tmp_path)
    capsys.readouterr()
    assert main(list_args(tmp_path)) == 0
    path = folder / "shots-G2.md"
    assert str(path) in capsys.readouterr().out
    before, stamp = path.read_bytes(), path.stat().st_mtime_ns
    assert main(list_args(tmp_path)) == 0
    [line] = capsys.readouterr().out.strip().splitlines()
    assert "already" in line
    assert path.read_bytes() == before and path.stat().st_mtime_ns == stamp
    assert main(list_args(tmp_path, "G3")) == 0
    assert (folder / "shots-G3.md").is_file() and path.read_bytes() == before


def test_a_tampered_list_is_refused_and_left_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = planned(tmp_path)
    assert main(list_args(tmp_path)) == 0
    path = folder / "shots-G2.md"
    tampered = path.read_text(encoding="utf-8").replace("| 1 |", "| 0 |", 1)
    path.write_text(tampered, encoding="utf-8")
    capsys.readouterr()
    assert main(list_args(tmp_path)) == 1
    assert path.read_text(encoding="utf-8") == tampered
    [line] = capsys.readouterr().out.strip().splitlines()
    assert "differs" in line


def test_no_plan_exits_2_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(list_args(tmp_path)) == 2
    out = capsys.readouterr()
    [line] = out.out.strip().splitlines()
    assert "qs split-plan" in line and "Traceback" not in out.out + out.err
    assert not (tmp_path / SESSION).exists()


def test_the_list_plans_nothing_new(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = planned(tmp_path)
    before = (folder / "plan.yml").read_bytes()
    args = list_args(tmp_path)
    args.size = "lean"
    assert main(args) == 2
    assert "--size" in capsys.readouterr().out
    assert main(list_args(tmp_path, geometry=None)) == 2
    assert (folder / "plan.yml").read_bytes() == before
    assert not list(folder.glob("shots-*.md"))


def test_the_help_shows_the_shot_list_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args(["split-plan", "--help"])
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    assert "--shot-list" in out and "--geometry" in out


def test_planning_without_size_still_needs_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = plan_args(tmp_path)
    args.size = None
    assert main(args) == 2
    assert "--size" in capsys.readouterr().out
