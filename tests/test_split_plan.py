"""`qs split-plan`: the seeded dataset plan (docs/13 §1, §6; ADR-0014; FR-52)."""

import argparse
import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from qs.__main__ import build_parser, run
from qs.cmd.split_plan import main
from qs.names import FrameName, parse_name
from qs.splitplan import Plan, PlanError, Prompt, Shot, make_plan, shot_list

SIZES = ("standard", "lean")

# docs/13 §1 and docs/14 §9: train · calib · test · (dent, tear, deformed) · control · spare
COUNTS = {
    "standard": (20, 15, 30, (12, 9, 9), 3, 5),
    "lean": (18, 10, 24, (10, 7, 7), 3, 3),
}
# deformed variants, roughly half each: (flat, oval)
VARIANTS = {"standard": (5, 4), "lean": (4, 3)}
# 9 control frames per pass × 3 + 3 per good unit + 6 per test unit + 10 placement
FRAMES = {"standard": 322, "lean": 265}


def shots(plan: Plan, geometry: str = "G2") -> list[Shot]:
    return [step for step in shot_list(plan, geometry) if isinstance(step, Shot)]


def frame(step: Shot | Prompt) -> FrameName | None:
    return step.frame if isinstance(step, Shot) else None


# --- the plan --------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_the_split_sizes_are_those_of_the_docs(size: str) -> None:
    plan = make_plan(size, 20260925)
    train, calib, test, (dent, tear, deformed), control, spare = COUNTS[size]
    assert (len(plan.train), len(plan.calib), len(plan.test)) == (train, calib, test)
    assert plan.control == ("c01", "c02", "c03") and control == 3
    assert len(plan.spare) == spare
    classes = Counter(damage.split("-")[0] for damage in plan.damage.values())
    assert classes == {"dent": dent, "tear": tear, "deformed": deformed}
    variants = Counter(d for d in plan.damage.values() if d.startswith("deformed"))
    assert (variants["deformed-flat"], variants["deformed-oval"]) == VARIANTS[size]


@pytest.mark.parametrize("size", SIZES)
def test_units_are_numbered_split_units_first_then_spares(size: str) -> None:
    plan = make_plan(size, 7)
    split = len(plan.train) + len(plan.calib) + len(plan.test)
    assert sorted((*plan.train, *plan.calib, *plan.test)) == [
        f"u{n:03d}" for n in range(1, split + 1)
    ]
    assert plan.spare == tuple(f"u{n:03d}" for n in range(split + 1, split + len(plan.spare) + 1))


@pytest.mark.parametrize("size", SIZES)
def test_over_200_seeds_no_unit_is_in_two_splits_and_the_order_is_complete(size: str) -> None:
    for seed in range(200):
        plan = make_plan(size, seed)
        groups = [set(plan.train), set(plan.calib), set(plan.test), set(plan.control),
                  set(plan.spare)]
        assert sum(len(g) for g in groups) == len(set().union(*groups)), seed
        assert Counter(plan.order) == Counter((*plan.train, *plan.calib, *plan.test)), seed
        assert not set(plan.spare) & set(plan.order), seed
        assert set(plan.damage) == set(plan.test), seed


@pytest.mark.parametrize("size", SIZES)
def test_the_same_seed_gives_the_same_plan_and_another_seed_another_order(size: str) -> None:
    assert make_plan(size, 42) == make_plan(size, 42)
    assert make_plan(size, 42).order != make_plan(size, 43).order


def test_a_plan_with_a_unit_in_two_splits_is_refused() -> None:
    plan = make_plan("lean", 1)
    moved = plan.calib[0]
    with pytest.raises(PlanError, match=moved):
        dataclasses.replace(plan, train=(*plan.train, moved))
    with pytest.raises(PlanError, match=plan.spare[0]):
        dataclasses.replace(plan, train=(*plan.train[:-1], plan.spare[0]))


def test_a_plan_whose_order_misses_or_repeats_a_unit_is_refused() -> None:
    plan = make_plan("lean", 1)
    with pytest.raises(PlanError, match="order"):
        dataclasses.replace(plan, order=plan.order[:-1])
    with pytest.raises(PlanError, match="order"):
        dataclasses.replace(plan, order=(*plan.order[:-1], plan.order[0]))
    with pytest.raises(PlanError, match="order"):
        dataclasses.replace(plan, order=(*plan.order, plan.spare[0]))


def test_an_unknown_size_is_refused() -> None:
    with pytest.raises(PlanError, match="huge"):
        make_plan("huge", 1)


# --- the shot list -------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_shot_totals_and_every_name_parses_and_is_unique(size: str) -> None:
    plan = make_plan(size, 20260925)
    names = [shot.file for shot in shots(plan)]
    assert len(names) == FRAMES[size]
    assert len(set(names)) == len(names)
    assert all(parse_name(name) == shot.frame for name, shot in zip(names, shots(plan),
                                                                     strict=True))
    assert {shot.frame.geometry for shot in shots(plan)} == {"G2"}


def test_the_list_opens_with_the_controls_first_pass() -> None:
    first = [shot.file for shot in shots(make_plan("standard", 3))[:9]]
    assert first == [f"c0{c}_good_G2_r{r}_pass1.png" for c in (1, 2, 3) for r in (0, 1, 2)]


@pytest.mark.parametrize("size", SIZES)
def test_the_list_closes_with_pass3_then_placement(size: str) -> None:
    last = [shot.file for shot in shots(make_plan(size, 3), "G1")[-19:]]
    assert last[:9] == [f"c0{c}_good_G1_r{r}_pass3.png" for c in (1, 2, 3) for r in (0, 1, 2)]
    assert last[9:] == [f"x0{n}_empty_G1_r0.png" for n in range(1, 6)] + [
        f"x{n:02d}_edge_G1_r0.png" for n in range(6, 11)
    ]


@pytest.mark.parametrize("size", SIZES)
def test_units_are_shot_in_the_plan_order_with_pass2_after_half(size: str) -> None:
    for seed in range(50):
        plan = make_plan(size, seed)
        seen: list[str] = []
        pass2_after: int | None = None
        for shot in shots(plan):
            unit = shot.frame.unit
            if shot.frame.pass_ == 2 and pass2_after is None:
                pass2_after = len(seen)
            if unit.startswith("u") and unit not in seen:
                seen.append(unit)
        assert tuple(seen) == plan.order, seed
        assert pass2_after == len(plan.order) // 2, seed


@pytest.mark.parametrize("size", SIZES)
def test_each_test_unit_is_damaged_straight_after_its_good_frames(size: str) -> None:
    for seed in range(50):
        plan = make_plan(size, seed)
        steps = shot_list(plan, "G2")
        for unit in plan.test:
            start = next(i for i, s in enumerate(steps) if (f := frame(s)) and f.unit == unit)
            run_of_steps = steps[start:start + 7]
            good = [frame(s) for s in run_of_steps[:3]]
            assert good == [FrameName(unit, "good", "G2", r, None) for r in (0, 1, 2)], seed
            damage = run_of_steps[3]
            assert isinstance(damage, Prompt) and damage.kind == "damage", seed
            assert damage.unit == unit, seed
            dmg = [frame(s) for s in run_of_steps[4:7]]
            assert dmg == [FrameName(unit, "dmg", "G2", r, None) for r in (0, 1, 2)], seed
        good_only = set(plan.train) | set(plan.calib)
        assert not any(isinstance(s, Prompt) and s.unit in good_only for s in steps)
        assert not any((f := frame(s)) and f.state == "dmg" and f.unit in good_only
                       for s in steps)


def test_the_damage_step_carries_the_recipe_of_docs_13_section_7() -> None:
    plan = make_plan("standard", 11)
    recipes = {p.unit: p.instruction for p in shot_list(plan, "G2")
               if isinstance(p, Prompt) and p.kind == "damage"}
    for unit, damage in plan.damage.items():
        text = recipes[unit]
        assert unit in text
        if damage == "dent":
            assert "pressed straight down about 3 mm" in text
        elif damage == "tear":
            assert "knife slit 5–10 mm" in text
        elif damage == "deformed-flat":
            assert "flat plate until about a third of the height is gone" in text
        else:
            assert "squeezed from two sides" in text


@pytest.mark.parametrize("size", SIZES)
def test_a_floor_brush_step_comes_after_every_tenth_unit(size: str) -> None:
    plan = make_plan(size, 5)
    units_done = 0
    brushed_at: list[int] = []
    last_unit: str | None = None
    for step in shot_list(plan, "G2"):
        if isinstance(step, Prompt) and step.kind == "brush":
            brushed_at.append(units_done)
        f = frame(step)
        if f is not None and f.unit.startswith("u") and f.unit != last_unit:
            units_done += 1
            last_unit = f.unit
    assert brushed_at == list(range(10, len(plan.order) + 1, 10))
    assert brushed_at  # the list has at least one brush step


# --- qs split-plan ----------------------------------------------------------------------


def args(tmp_path: Path, size: str = "lean", seed: int | None = 20260925,
         session: str = "2026-09-25-mvp") -> argparse.Namespace:
    return argparse.Namespace(session=session, size=size, seed=seed, root=str(tmp_path))


def test_main_writes_the_plan_and_the_printable_sheet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(tmp_path)) == 0
    folder = tmp_path / "2026-09-25-mvp"
    assert (folder / "plan.yml").is_file()
    sheet = (folder / "plan.md").read_text(encoding="utf-8")
    plan = make_plan("lean", 20260925)
    out = capsys.readouterr().out
    assert "seed 20260925" in out
    for unit in (*plan.train, *plan.calib, *plan.test, *plan.spare, *plan.control):
        assert unit in sheet and unit in out
    first_test = plan.test[0]
    assert any(first_test in line and plan.damage[first_test] in line
               for line in out.splitlines())


def test_the_plan_file_reads_back_as_the_same_plan(tmp_path: Path) -> None:
    from qs.session import read_plan

    assert main(args(tmp_path, size="standard", seed=9)) == 0
    assert read_plan(tmp_path / "2026-09-25-mvp") == make_plan("standard", 9)


def test_the_same_seed_writes_a_byte_identical_plan_file(tmp_path: Path) -> None:
    assert main(args(tmp_path / "a")) == 0
    assert main(args(tmp_path / "b")) == 0
    a = (tmp_path / "a" / "2026-09-25-mvp" / "plan.yml").read_bytes()
    b = (tmp_path / "b" / "2026-09-25-mvp" / "plan.yml").read_bytes()
    assert a == b


def test_a_second_run_refuses_and_leaves_the_plan_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(tmp_path, seed=1)) == 0
    path = tmp_path / "2026-09-25-mvp" / "plan.yml"
    before = path.read_bytes()
    capsys.readouterr()
    assert main(args(tmp_path, seed=2)) == 1
    assert path.read_bytes() == before
    [line] = capsys.readouterr().out.strip().splitlines()
    assert line.startswith("plan:") and "exists" in line


def test_the_seed_defaults_to_the_session_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from qs.session import read_plan

    assert main(args(tmp_path, seed=None, session="2026-09-25-mvp")) == 0
    assert read_plan(tmp_path / "2026-09-25-mvp").seed == 20260925
    assert "seed 20260925" in capsys.readouterr().out


def test_a_tampered_plan_file_with_a_unit_in_two_splits_is_refused(tmp_path: Path) -> None:
    from qs.session import read_plan

    assert main(args(tmp_path, seed=4)) == 0
    folder = tmp_path / "2026-09-25-mvp"
    plan = read_plan(folder)
    text = (folder / "plan.yml").read_text(encoding="utf-8")
    moved = plan.calib[0]
    text = text.replace(f"- {plan.train[0]}\n", f"- {moved}\n", 1)
    (folder / "plan.yml").write_text(text, encoding="utf-8")
    with pytest.raises(PlanError, match=moved):
        read_plan(folder)


def test_the_cli_lists_split_plan_and_refuses_an_unknown_size(
    capsys: pytest.CaptureFixture[str]
) -> None:
    assert "split-plan" in build_parser().format_help()
    with pytest.raises(SystemExit) as stopped:
        run(["split-plan", "--session", "2026-09-25-mvp", "--size", "huge"])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'huge'" in err and "Traceback" not in err


@pytest.mark.parametrize("session", ["../escape", "a/b", "", ".hidden"])
def test_a_session_id_that_is_not_a_plain_folder_name_is_refused(
    session: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        run(["split-plan", "--session", session, "--size", "lean"])
    assert stopped.value.code == 2
