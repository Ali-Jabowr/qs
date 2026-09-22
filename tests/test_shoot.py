"""`qs shoot`: the guided dataset session (docs/13 §6, §8; docs/12 §2; ADR-0014; C9)."""

import argparse
import csv
import hashlib
from collections import deque
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from qs.cmd.shoot import main
from qs.config import StationConfig
from qs.contracts import CameraReadback, Frame, Health, TriggerEvent
from qs.screen import KeyPress
from qs.session import write_plan
from qs.sources.v4l2 import ControlReading
from qs.splitplan import Plan, Prompt, Shot, make_plan, shot_list

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"
SESSION = "2026-09-25-mvp"
SEED = 20260925
GOOD_READINGS = (
    ControlReading("power_line_hz", 50, 50),
    ControlReading("exposure_auto", 1, 1),
    ControlReading("exposure_us", 10000, 10000),
    ControlReading("gain", 0, 0),
    ControlReading("auto_white_balance", 0, 0),
    ControlReading("wb_kelvin", 4500, 4500),
    ControlReading("focus_auto", 0, 0),
    ControlReading("focus", 30, 30),
)
VERIFIED = CameraReadback(10000, 0, 4500, 30, verified=True)
SESSION_KEYS = [
    "session_id", "kind", "geometries", "product", "purchased_at", "started_at", "ended_at",
    "camera", "reference_patch", "split_seed", "units", "damage", "notes",
]  # docs/13 §8, in its order
FRAMES_COLUMNS = [
    "file", "unit", "state", "geometry", "rotation", "pass", "wall_utc", "grabbed_at",
    "capture_ms", "reference_patch", "exposure_us", "gain", "wb_kelvin", "focus", "verified",
    "sha256", "status",
]  # #29, plus status


class FakeSource:
    """A camera: every capture a new random image, the live view a constant grey one."""

    def __init__(self, config: StationConfig, unverified_at: int | None = None) -> None:
        self.readings = GOOD_READINGS
        self.captured: list[Frame] = []
        self.previews = 0
        self.closed = 0
        self.unverified_at = unverified_at
        self.preview_pixels = np.full((24, 32, 3), 128, np.uint8)

    def open(self) -> None:
        pass

    def capture(self, event: TriggerEvent) -> Frame:
        n = len(self.captured) + 1
        pixels = np.random.default_rng(n).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        camera = VERIFIED
        if n == self.unverified_at:
            camera = CameraReadback(10000, None, 4500, 30, verified=False)
        frame = Frame(event, pixels, event.t0 + 0.03, camera, 0.62 + n / 10000, 30.0)
        self.captured.append(frame)
        return frame

    def preview(self) -> np.ndarray | None:
        self.previews += 1
        return self.preview_pixels

    def health(self) -> Health:
        return Health(True, "streaming")

    def close(self) -> None:
        self.closed += 1


class FakeScreen:
    """The window: records every tick's lines; `keys()` plays a script, then closes."""

    def __init__(self, script: list[list[KeyPress]]) -> None:
        self.script = deque(script)
        self.lines: list[list[str]] = []
        self.closed = False

    def tick(self, image: np.ndarray | None, lines: list[str]) -> None:
        self.lines.append(list(lines))

    def keys(self) -> list[KeyPress]:
        if not self.script:
            self.closed = True  # the operator closes the window when the script runs out
            return []
        return self.script.popleft()

    def close(self) -> None:
        self.closed = True


def spaces(n: int, start: int = 1000, gap: int = 400) -> list[list[KeyPress]]:
    return [[KeyPress("space", start + i * gap)] for i in range(n)]


def plan_session(root: Path, size: str = "lean", seed: int = SEED) -> Plan:
    plan = make_plan(size, seed)
    write_plan(root / SESSION, plan, "sheet\n")
    return plan


def args(root: Path, config: Path = EXAMPLE, **changes: object) -> argparse.Namespace:
    values: dict[str, object] = dict(
        config=str(config),
        session=SESSION,
        flavour="клубника",
        shop="Магнит, Рабоче-Крестьянская",
        best_before=date(2026, 9, 28),
        purchased_at=datetime(2026, 9, 25, 9, 10, tzinfo=timezone(timedelta(hours=3))),
        brand="Мочи-Дом",
        client_product=False,
        root=str(root),
    )
    values.update(changes)
    return argparse.Namespace(**values)


def run(
    root: Path, script: list[list[KeyPress]], **changes: object
) -> tuple[int, FakeSource, FakeScreen]:
    made: list[FakeSource] = []
    screen = FakeScreen(script)
    unverified_at = changes.pop("unverified_at", None)

    def make_source(config: StationConfig) -> FakeSource:
        made.append(FakeSource(config, unverified_at))  # type: ignore[arg-type]
        return made[-1]

    code = main(args(root, **changes), make_source=make_source,
                make_screen=lambda display, title: screen)
    return code, made[0], screen


def pngs(root: Path) -> list[str]:
    return sorted(p.name for p in (root / SESSION).glob("*.png"))


def frame_rows(root: Path) -> list[dict[str, str]]:
    with (root / SESSION / "frames.csv").open(encoding="utf-8", newline="") as log:
        return list(csv.DictReader(log))


def session_yml(root: Path) -> dict[str, object]:
    data = yaml.safe_load((root / SESSION / "session.yml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


# --- the full session ------------------------------------------------------------------


def test_a_full_lean_session_writes_every_frame_of_the_shot_list_losslessly(
    tmp_path: Path,
) -> None:
    plan = plan_session(tmp_path)
    steps = shot_list(plan, "G2")
    code, source, _ = run(tmp_path, spaces(len(steps)))
    assert code == 0
    wanted = sorted(step.file for step in steps if isinstance(step, Shot))
    assert len(wanted) == 265
    assert pngs(tmp_path) == wanted
    rows = frame_rows(tmp_path)
    assert [row["file"] for row in rows] == [s.file for s in steps if isinstance(s, Shot)]
    for row, frame in zip(rows, source.captured, strict=True):
        saved = cv2.imread(str(tmp_path / SESSION / row["file"]), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(saved, frame.pixels)


def test_every_saved_frame_came_from_capture_as_a_shoot_event(tmp_path: Path) -> None:
    plan = plan_session(tmp_path)
    steps = shot_list(plan, "G2")
    code, source, _ = run(tmp_path, spaces(len(steps)))
    assert code == 0
    assert source.previews > 0  # the live view ran …
    rows = frame_rows(tmp_path)
    assert len(source.captured) == len(rows) == 265  # … but every file is a capture()
    for row in rows:
        saved = cv2.imread(str(tmp_path / SESSION / row["file"]), cv2.IMREAD_UNCHANGED)
        assert not np.array_equal(saved, source.preview_pixels)
    assert {frame.event.source_id for frame in source.captured} == {"shoot"}
    assert [frame.event.seq for frame in source.captured] == list(range(1, 266))


def test_the_session_record_has_the_docs_keys_and_the_plans_values(tmp_path: Path) -> None:
    plan = plan_session(tmp_path)
    steps = shot_list(plan, "G2")
    code, source, _ = run(tmp_path, spaces(len(steps)))
    assert code == 0
    data = session_yml(tmp_path)
    assert list(data) == SESSION_KEYS
    planned = yaml.safe_load((tmp_path / SESSION / "plan.yml").read_text(encoding="utf-8"))
    assert data["session_id"] == SESSION and data["kind"] == "dataset"
    assert data["geometries"] == ["G2"]  # the example config's lighting.geometry
    assert data["split_seed"] == planned["seed"] == SEED
    assert data["damage"] == planned["damage"]
    units = data["units"]
    assert isinstance(units, dict)
    assert {k: units[k] for k in planned["units"]} == planned["units"]
    assert units["placement"] == [f"x{n:02d}" for n in range(1, 11)]
    assert data["product"] == {
        "brand": "Мочи-Дом", "flavour": "клубника", "shop": "Магнит, Рабоче-Крестьянская",
        "best_before": date(2026, 9, 28),
    }
    assert data["purchased_at"] == datetime(2026, 9, 25, 9, 10,
                                            tzinfo=timezone(timedelta(hours=3)))
    assert data["camera"] == {"device": "/dev/video0", "exposure_us": 10000, "gain": 0,
                              "wb_kelvin": 4500, "focus": 30, "verified": True}
    first, last = source.captured[0], source.captured[-1]
    assert data["reference_patch"] == {"start": first.reference_patch,
                                       "end": last.reference_patch}
    assert first.reference_patch != last.reference_patch
    started, ended = data["started_at"], data["ended_at"]
    assert isinstance(started, datetime) and isinstance(ended, datetime)
    assert started == first.event.wall and ended == last.event.wall
    assert started <= ended and started.tzinfo is not None
    assert data["notes"] == ""


def test_a_client_product_is_recorded_without_a_brand(tmp_path: Path) -> None:
    plan = plan_session(tmp_path)
    code, _, _ = run(tmp_path, spaces(len(shot_list(plan, "G2"))),
                     brand=None, client_product=True)
    assert code == 0
    product = session_yml(tmp_path)["product"]
    assert isinstance(product, dict)
    assert "brand" not in product and product["client_product"] is True
    assert product["flavour"] == "клубника"


def test_one_unverified_frame_makes_the_session_camera_unverified(tmp_path: Path) -> None:
    plan = plan_session(tmp_path)
    code, _, _ = run(tmp_path, spaces(len(shot_list(plan, "G2"))), unverified_at=100)
    assert code == 0
    camera = session_yml(tmp_path)["camera"]
    assert isinstance(camera, dict) and camera["verified"] is False
    rows = frame_rows(tmp_path)
    assert [row["verified"] for row in rows].count("false") == 1
    assert rows[99]["verified"] == "false" and rows[99]["gain"] == ""


def test_the_frame_log_has_one_row_per_png_and_each_hash_matches(tmp_path: Path) -> None:
    plan = plan_session(tmp_path)
    code, source, _ = run(tmp_path, spaces(len(shot_list(plan, "G2"))))
    assert code == 0
    with (tmp_path / SESSION / "frames.csv").open(encoding="utf-8", newline="") as log:
        assert next(csv.reader(log)) == FRAMES_COLUMNS
    rows = frame_rows(tmp_path)
    assert sorted(row["file"] for row in rows) == pngs(tmp_path)
    for row, frame in zip(rows, source.captured, strict=True):
        digest = hashlib.sha256((tmp_path / SESSION / row["file"]).read_bytes()).hexdigest()
        assert row["sha256"] == digest
        assert row["status"] == "saved"
        assert row["wall_utc"] == frame.event.wall.isoformat()
        assert float(row["reference_patch"]) == frame.reference_patch
    first = rows[0]
    assert (first["file"], first["unit"], first["state"], first["geometry"], first["rotation"],
            first["pass"]) == ("c01_good_G2_r0_pass1.png", "c01", "good", "G2", "0", "1")
    assert (first["exposure_us"], first["gain"], first["wb_kelvin"], first["focus"],
            first["verified"]) == ("10000", "0", "4500", "30", "true")
    assert rows[-1]["pass"] == ""  # a placement frame has no pass


# --- the keys ----------------------------------------------------------------------------


def quit_at(tk_ms: int) -> list[KeyPress]:
    return [KeyPress("q", tk_ms)]


def test_two_presses_100_ms_apart_take_one_shot_and_400_ms_apart_two(tmp_path: Path) -> None:
    plan_session(tmp_path / "a")
    code, source, _ = run(tmp_path / "a", [[KeyPress("space", 1000)], [KeyPress("space", 1100)],
                                           [quit_at(5000)[0]]])
    assert code == 1 and len(source.captured) == 1
    assert pngs(tmp_path / "a") == ["c01_good_G2_r0_pass1.png"]

    plan_session(tmp_path / "b")
    code, source, _ = run(tmp_path / "b", [[KeyPress("space", 1000)], [KeyPress("space", 1400)],
                                           [quit_at(5000)[0]]])
    assert code == 1 and len(source.captured) == 2
    assert pngs(tmp_path / "b") == ["c01_good_G2_r0_pass1.png", "c01_good_G2_r1_pass1.png"]


def test_a_held_key_in_one_batch_takes_one_shot(tmp_path: Path) -> None:
    plan_session(tmp_path)
    held = [KeyPress("space", 1000 + 33 * i) for i in range(8)]  # auto-repeat, ~30 Hz
    code, source, _ = run(tmp_path, [held, quit_at(5000)])
    assert code == 1 and len(source.captured) == 1


def first_prompt(steps: tuple[Shot | Prompt, ...], kind: str, damage: str | None = None,
                 plan: Plan | None = None) -> int:
    for i, step in enumerate(steps):
        if not isinstance(step, Prompt) or step.kind != kind:
            continue
        if damage is None or (plan is not None and plan.damage[str(step.unit)] == damage):
            return i
    raise AssertionError(f"no {kind} prompt")


@pytest.mark.parametrize("kind", ["damage", "brush"])
def test_a_prompt_step_is_confirmed_with_space_and_saves_no_file(
    tmp_path: Path, kind: str
) -> None:
    plan = plan_session(tmp_path)
    steps = shot_list(plan, "G2")
    at = first_prompt(steps, kind)
    before = sum(isinstance(s, Shot) for s in steps[:at])
    # every step up to and including the prompt, then Q
    code, source, screen = run(tmp_path, [*spaces(at + 1), quit_at(10**6)])
    assert code == 1
    assert len(pngs(tmp_path)) == len(source.captured) == before
    shown = [lines for lines in screen.lines if str(steps[at].instruction) in lines[0]]
    assert shown and all(line.endswith("SPACE when done") for line, *_ in shown)


def test_the_damage_step_shows_the_units_recipe(tmp_path: Path) -> None:
    plan = plan_session(tmp_path)
    steps = shot_list(plan, "G2")
    at = first_prompt(steps, "damage", "deformed-oval", plan)
    unit = steps[at].unit if isinstance(steps[at], Prompt) else None
    code, _, screen = run(tmp_path, [*spaces(at + 1), quit_at(10**6)])
    assert code == 1
    shown = [lines[0] for lines in screen.lines if f"Take {unit} out" in lines[0]]
    assert shown and "Oval: squeezed from two sides" in shown[0]


def test_the_screen_shows_the_instruction_the_progress_and_the_grey_card(
    tmp_path: Path,
) -> None:
    plan = plan_session(tmp_path)
    code, source, screen = run(tmp_path, [*spaces(2), quit_at(10**6)])
    assert code == 1
    before, after_one, after_two = screen.lines[0], screen.lines[1], screen.lines[2]
    assert before == ["c01 · control, pass 1 · 0° · good — SPACE to shoot",
                      f"shot 1 of 265 · unit 0 of {len(plan.order)}", "grey card: no frame yet"]
    assert after_one[1].startswith("shot 2 of 265")
    assert after_one[2] == "grey card 0.6201 · drift +0.00% since the first frame"
    assert after_two[2] == "grey card 0.6202 · drift +0.02% since the first frame"


def test_q_halfway_stops_with_the_frames_kept_and_no_session_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = plan_session(tmp_path)
    steps = shot_list(plan, "G2")
    half = len(steps) // 2
    code, source, _ = run(tmp_path, [*spaces(half), quit_at(10**6)])
    assert code == 1
    assert len(pngs(tmp_path)) == len(source.captured) == len(frame_rows(tmp_path)) > 100
    assert not (tmp_path / SESSION / "session.yml").exists()
    assert capsys.readouterr().out.splitlines()[-1].startswith(
        f"stopped after {len(source.captured)} frame(s)"
    )


def test_closing_the_window_stops_like_q(tmp_path: Path) -> None:
    plan_session(tmp_path)
    code, source, _ = run(tmp_path, spaces(5))  # the script runs out: the window is closed
    assert code == 1 and len(pngs(tmp_path)) == 5
    assert not (tmp_path / SESSION / "session.yml").exists()


def test_the_end_says_to_copy_the_session_to_the_phone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = plan_session(tmp_path)
    code, _, screen = run(tmp_path, spaces(len(shot_list(plan, "G2"))))
    assert code == 0
    folder = tmp_path / SESSION
    assert f"Copy {folder}/ to the phone NOW, before anything else (T11)" in screen.lines[-1]
    assert capsys.readouterr().out.splitlines()[-1] == (
        f"Copy {folder}/ to the phone NOW, before anything else (T11)"
    )


def test_a_started_session_is_refused_and_left_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_session(tmp_path)
    assert run(tmp_path, [*spaces(3), quit_at(10**6)])[0] == 1
    before = {p.name: p.read_bytes() for p in (tmp_path / SESSION).iterdir()}
    capsys.readouterr()
    opened: list[FakeSource] = []
    code = main(args(tmp_path), make_source=lambda c: opened.append(FakeSource(c)),
                make_screen=lambda d, t: FakeScreen(spaces(3)))
    assert code == 1 and opened == []  # refused before the camera is touched
    assert {p.name: p.read_bytes() for p in (tmp_path / SESSION).iterdir()} == before
    [line] = capsys.readouterr().out.strip().splitlines()
    assert line.startswith("shoot:") and "already has frames" in line


def test_a_file_already_on_disk_is_never_overwritten(tmp_path: Path) -> None:
    from qs.names import parse_name
    from qs.session import save_frame

    pixels = np.zeros((4, 4, 3), np.uint8)
    name = parse_name("u001_good_G2_r0.png")
    save_frame(tmp_path, name, pixels)
    before = (tmp_path / "u001_good_G2_r0.png").read_bytes()
    with pytest.raises(FileExistsError):
        save_frame(tmp_path, name, np.full((4, 4, 3), 255, np.uint8))
    assert (tmp_path / "u001_good_G2_r0.png").read_bytes() == before


# --- could not run -----------------------------------------------------------------------


def config_with(tmp_path: Path, old: str, new: str) -> Path:
    path = tmp_path / "station.yml"
    path.write_text(EXAMPLE.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    return path


def test_a_phone_camera_type_names_adr_0010_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_session(tmp_path)
    config = config_with(tmp_path, "type: webcam", "type: phone")
    assert main(args(tmp_path, config), make_screen=lambda d, t: FakeScreen([])) == 2
    [line] = capsys.readouterr().out.strip().splitlines()
    assert line.startswith("camera:") and "ADR-0010" in line and "phone" in line
    assert pngs(tmp_path) == []


def test_no_plan_says_to_run_split_plan_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[FakeSource] = []
    code = main(args(tmp_path), make_source=lambda c: opened.append(FakeSource(c)),
                make_screen=lambda d, t: FakeScreen([]))
    assert code == 2 and opened == []
    [line] = capsys.readouterr().out.strip().splitlines()
    assert line.startswith("plan:") and "qs split-plan" in line


def test_a_missing_camera_prints_one_camera_line_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_session(tmp_path)
    config = config_with(tmp_path, "/dev/video0", "/dev/video99")
    assert main(args(tmp_path, config), make_screen=lambda d, t: FakeScreen([])) == 2
    out = capsys.readouterr().out
    [line] = out.strip().splitlines()
    assert line.startswith("camera:") and "/dev/video99" in line
    assert "Traceback" not in out
    assert sorted(p.name for p in (tmp_path / SESSION).iterdir()) == ["plan.md", "plan.yml"]


def test_a_bad_config_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan_session(tmp_path)
    assert main(args(tmp_path, tmp_path / "missing.yml")) == 2
    assert capsys.readouterr().out.startswith("config:")


def test_no_display_says_so_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import tkinter as tk

    plan_session(tmp_path)
    made: list[FakeSource] = []

    def no_display(display: object, title: str) -> FakeScreen:
        raise tk.TclError("no display name and no $DISPLAY environment variable")

    code = main(args(tmp_path), make_source=lambda c: made.append(FakeSource(c)) or made[-1],
                make_screen=no_display)
    assert code == 2 and made[0].closed == 1
    assert capsys.readouterr().out.splitlines()[-1].startswith("screen: no display")


# --- the command line ----------------------------------------------------------------------


def test_the_cli_lists_shoot_and_parses_the_product_up_front() -> None:
    from qs.__main__ import build_parser

    parser = build_parser()
    assert "shoot" in parser.format_help()
    parsed = parser.parse_args([
        "shoot", "--config", "c.yml", "--session", SESSION, "--flavour", "манго",
        "--shop", "Ашан", "--best-before", "2026-09-28",
        "--purchased-at", "2026-09-25T09:10+03:00", "--client-product",
    ])
    assert parsed.client_product is True and parsed.brand is None
    assert parsed.best_before == date(2026, 9, 28)
    assert parsed.purchased_at == datetime(2026, 9, 25, 6, 10, tzinfo=UTC)
    assert parsed.root == "data/raw"


@pytest.mark.parametrize("bad", [
    [],  # neither --brand nor --client-product
    ["--brand", "X", "--client-product"],  # both
    ["--brand", "X", "--purchased-at", "2026-09-25T09:10"],  # no offset
    ["--brand", "X", "--best-before", "28.09.2026"],
])
def test_the_cli_refuses_an_incomplete_or_ambiguous_product(
    bad: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    from qs.__main__ import run as qs

    base = ["shoot", "--config", "c.yml", "--session", SESSION, "--flavour", "манго",
            "--shop", "Ашан", "--best-before", "2026-09-28",
            "--purchased-at", "2026-09-25T09:10+03:00"]
    with pytest.raises(SystemExit) as stopped:
        qs([*base, *bad])
    assert stopped.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
