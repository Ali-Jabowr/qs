"""`qs pilot-report` through `main()` with the blot fake, on a synthetic lean pilot session.

Six units (4 good + 2 damaged) × 3 rotations × G1·G2·G3 of pale discs on a 640×480 black
floor, written once per module with `qs.session.save_frame` under the name rule, against a
station config cut down from the example. G1's good discs vary in radius across rotations and
so fail the 2% silhouette criterion; the damaged discs carry a dark blot that `BlotPatchCore`
scores high. The heavy run happens once; the variant sessions symlink the same frames.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import EXAMPLE, BlotPatchCore
from qs.__main__ import build_parser
from qs.cmd import pilot_report
from qs.contracts import CameraReadback, TriggerEvent
from qs.inspector.model import PatchCore, load_backbone
from qs.inspector.pilot import rank, separation, suggest_window
from qs.names import FrameName
from qs.session import Product, SessionRecord, save_frame, write_session
from qs.splitplan import PilotPlan, Substitution, discard_broken, make_pilot_plan

SESSION = "2026-09-25-pilot"
GEOMETRIES = ("G1", "G2", "G3")
FLOOR_GREY = 20
GOOD_GREYS = (197, 199, 201, 203)
UNIT_GREY = 200
BLOT_GREY = {"G2": 10, "G3": 0}
BLOT_RADIUS = 25
ROTATION_RADII = (83, 85, 87)
ROI_PX = 560 * 480
CAMERA = CameraReadback(10000, 0, 4500, 30, True)


@dataclass(frozen=True, slots=True)
class _Shot:
    """One recorded frame, enough for `write_session`: event, reference patch, camera."""

    event: TriggerEvent = TriggerEvent(1, 0.0, datetime(2026, 9, 25, 10, 0, tzinfo=UTC), "test")
    reference_patch: float | None = 0.62
    camera: CameraReadback = CAMERA


def _product() -> Product:
    return Product(
        brand="Mochi-Dom",
        flavour="strawberry",
        shop="Magnit",
        best_before=date(2026, 9, 28),
        purchased_at=datetime(2026, 9, 25, 9, 10, tzinfo=UTC),
    )


def _record(plan: PilotPlan, shot: Substitution | None) -> SessionRecord:
    return SessionRecord(
        session_id=SESSION,
        geometry="G2",
        product=_product(),
        device="/dev/video0",
        camera_at_open=CAMERA,
        plan=plan,
        frames=(_Shot(),),
        substitution=shot,
    )


def _station_config(directory: Path) -> Path:
    """The example config cut down to a 640×480 frame, so the session runs in seconds."""
    text = EXAMPLE.read_text(encoding="utf-8")
    for old, new in (
        ("width: 1920", "width: 640"),
        ("height: 1080", "height: 480"),
        ("[1780, 40, 100, 100]", "[600, 40, 16, 16]"),
        ("[360, 0, 1200, 1080]", "[40, 0, 560, 480]"),
    ):
        assert old in text
        text = text.replace(old, new)
    path = directory / "station.yml"
    path.write_text(text, encoding="utf-8")
    return path


def _disc_frame(grey: int, radius: int, rotation: int, blot: int | None) -> np.ndarray:
    pixels = np.full((480, 640, 3), FLOOR_GREY, dtype=np.uint8)
    centre = (320 + 3 * rotation, 240 - 2 * rotation)
    cv2.circle(pixels, centre, radius, (grey, grey, grey), -1)
    if blot is not None:
        cv2.circle(pixels, centre, BLOT_RADIUS, (blot, blot, blot), -1)
    return pixels


def _two_unit_frame() -> np.ndarray:
    pixels = np.full((480, 640, 3), FLOOR_GREY, dtype=np.uint8)
    cv2.circle(pixels, (230, 240), 85, (UNIT_GREY, UNIT_GREY, UNIT_GREY), -1)
    cv2.circle(pixels, (410, 240), 85, (UNIT_GREY, UNIT_GREY, UNIT_GREY), -1)
    return pixels


def _fill_session(folder: Path, plan: PilotPlan) -> None:
    """Every frame of the lean pilot: all units good in 3 rotations, then the damaged ones."""
    greys = dict(zip(plan.good, GOOD_GREYS, strict=True))
    for geometry in GEOMETRIES:
        for unit in plan.units:
            for rotation in range(3):
                save_frame(
                    folder,
                    FrameName(unit, "good", geometry, rotation, None),
                    _disc_frame(
                        greys.get(unit, UNIT_GREY),
                        ROTATION_RADII[rotation] if geometry == "G1" and unit in greys else 85,
                        rotation,
                        None,
                    ),
                )
                if unit in plan.damage:
                    blot = 0 if geometry == "G1" else BLOT_GREY[geometry]
                    save_frame(
                        folder,
                        FrameName(unit, "dmg", geometry, rotation, None),
                        _disc_frame(UNIT_GREY, 85, rotation, blot),
                    )


def _run(
    tmp_path: Path,
    config: Path,
    folder: Path,
    *,
    model: Callable[[str], PatchCore] | None = None,
    cuda: bool = True,
) -> tuple[int, str, BlotPatchCore, Path]:
    """One `pilot-report` run with a fresh blot fake; code, output, fake and report dir."""
    fake = BlotPatchCore()
    report_dir = tmp_path / "reports"
    args = build_parser().parse_args([
        "pilot-report", "--config", str(config), "--session", str(folder),
        "--report-dir", str(report_dir),
    ])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = pilot_report.main(
            args,
            make_patchcore=model or (lambda device: fake),
            cuda_available=lambda: cuda,
        )
    return code, buffer.getvalue(), fake, report_dir


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^## ", text, flags=re.MULTILINE)
    return {part.splitlines()[0]: part for part in parts[1:]}


@dataclass(frozen=True)
class PilotRun:
    """The one heavy run: the session it read, and everything it produced."""

    config: Path
    session: Path
    plan: PilotPlan
    fake: BlotPatchCore
    code: int
    out: str
    report_dir: Path

    @property
    def good_units(self) -> tuple[str, ...]:
        return self.plan.good

    @property
    def damaged(self) -> tuple[str, ...]:
        return tuple(self.plan.damage)

    @property
    def report(self) -> str:
        return (self.report_dir / f"{SESSION}.md").read_text(encoding="utf-8")

    @property
    def csv_path(self) -> Path:
        return self.report_dir / f"{SESSION}-histogram.csv"

    @property
    def sections(self) -> dict[str, str]:
        return _sections(self.report)


def _like_main(
    tmp_path: Path,
    run: PilotRun,
    shot: Substitution | None,
    replace: dict[str, np.ndarray],
    drop: set[str] = frozenset(),
) -> Path:
    """A session folder with the main run's frames (some replaced, some dropped) and its
    own `session.yml`."""
    folder = tmp_path / "session"
    folder.mkdir()
    for png in sorted(run.session.glob("*.png")):
        if png.name not in replace and png.name not in drop:
            (folder / png.name).symlink_to(png.resolve())
    for name, pixels in replace.items():
        cv2.imwrite(str(folder / name), pixels)
    write_session(folder, _record(run.plan, shot), by="qs import-phone")
    return folder


@pytest.fixture(scope="module")
def station(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _station_config(tmp_path_factory.mktemp("station"))


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory, station: Path) -> PilotRun:
    """The lean pilot session, built and reported once."""
    root = tmp_path_factory.mktemp("pilot")
    folder = root / "raw" / SESSION
    folder.mkdir(parents=True)
    plan = make_pilot_plan("lean", 20260925)
    _fill_session(folder, plan)
    write_session(folder, _record(plan, None), by="qs import-phone")
    fake = BlotPatchCore()
    report_dir = root / "reports"
    args = build_parser().parse_args([
        "pilot-report", "--config", str(station), "--session", str(folder),
        "--report-dir", str(report_dir),
    ])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = pilot_report.main(
            args, make_patchcore=lambda device: fake, cuda_available=lambda: True
        )
    return PilotRun(station, folder, plan, fake, code, buffer.getvalue(), report_dir)


# --- the pure arithmetic --------------------------------------------------------------------


def test_help_lists_pilot_report() -> None:
    assert "pilot-report" in build_parser().format_help()


def test_the_separation_arithmetic_on_the_hand_example() -> None:
    result = separation([1.0, 1.2, 1.1, 1.3], [[2.3], [2.1], [2.2], [2.0]])
    assert result.gap == pytest.approx(0.7, abs=1e-9)
    assert result.spread == pytest.approx(0.129099445, abs=1e-9)
    assert result.separation == pytest.approx(5.422176685, abs=1e-9)
    assert result.n == 4


def test_the_separation_refuses_one_fold_or_an_empty_fold() -> None:
    with pytest.raises(ValueError, match="at least 2 folds"):
        separation([1.0], [[2.0]])
    with pytest.raises(ValueError, match="no damaged score"):
        separation([1.0, 1.1], [[], [2.0]])


def test_the_window_suggestion_is_twice_the_median() -> None:
    assert suggest_window([380.0, 380.0, 380.0]) == (380.0, 760)
    assert suggest_window([171.0, 172.0]) == (171.5, 343)


def test_a_tie_to_two_decimals_goes_to_g2() -> None:
    assert rank({"G1": 4.201, "G2": 4.196, "G3": 4.189}, ("G1", "G2", "G3")) == [
        "G2", "G1", "G3",
    ]


def test_a_tie_goes_to_g3_when_the_session_has_one_lamp() -> None:
    assert rank({"G1": 3.101, "G3": 3.096}, ("G1", "G3")) == ["G3", "G1"]


def test_distinct_separations_rank_highest_first() -> None:
    assert rank({"G1": 1.0, "G2": 3.0, "G3": 2.0}, ("G1", "G2", "G3")) == ["G2", "G3", "G1"]


# --- the one heavy run ----------------------------------------------------------------------


def test_the_report_recommends_the_higher_separation(run: PilotRun) -> None:
    assert run.code == 0, run.out
    assert "recommendation: G3" in run.out
    sections = run.sections
    g1 = sections["G1"]
    assert "silhouette, spread < 2%: out" in g1
    for unit in run.good_units:
        assert unit in g1
    assert "of 3 rotations" not in g1
    for geometry in ("G2", "G3"):
        assert "silhouette, spread < 2%: pass" in sections[geometry]
        match = re.search(r"separation: ([\d.]+) — gap ([\d.]+), spread ([\d.]+), n (\d+)",
                          sections[geometry])
        assert match is not None
        assert int(match[4]) == len(run.good_units)
        assert math.isfinite(float(match[1]))
    ranking = sections["Ranking"]
    g2 = float(re.search(r"- G2: separation ([\d.]+)", ranking)[1])
    g3 = float(re.search(r"- G3: separation ([\d.]+)", ranking)[1])
    assert g3 > g2 > 0
    assert "Recommendation: G3." in ranking
    assert "- G1: out by silhouette" in ranking


def test_the_report_shows_the_localization_counts_and_the_threshold_suggestion(
    run: PilotRun,
) -> None:
    g1 = run.sections["G1"]
    assert "localization: 24 ok" in g1
    match = re.search(r"threshold suggestion: (\d+) — histogram: (\S+)", g1)
    assert match is not None
    assert match[2] == f"{SESSION}-histogram.csv"
    assert 0 <= int(match[1]) <= 255


def test_the_window_line_is_twice_the_median_diameter(run: PilotRun) -> None:
    for geometry in ("G2", "G3"):
        match = re.search(r"median diameter ([\d.]+) px → window_px (\d+)",
                          run.sections[geometry])
        assert match is not None
        assert int(match[2]) == math.ceil(2 * float(match[1]))


def test_the_histogram_csv_has_256_rows_per_geometry_summing_to_the_roi(run: PilotRun) -> None:
    rows = list(csv.DictReader(run.csv_path.open(encoding="utf-8", newline="")))
    assert [row["geometry"] for row in rows[::256]] == list(GEOMETRIES)
    sums: dict[str, int] = {}
    for row in rows:
        sums[row["geometry"]] = sums.get(row["geometry"], 0) + int(row["count"])
    assert sums == {geometry: 18 * ROI_PX for geometry in GEOMETRIES}


def test_the_report_directory_holds_exactly_the_two_files(run: PilotRun) -> None:
    assert sorted(p.name for p in run.report_dir.iterdir()) == [
        f"{SESSION}-histogram.csv", f"{SESSION}.md",
    ]
    assert not (run.report_dir.parent / "models").exists()


def test_leave_one_out_is_honest(run: PilotRun) -> None:
    greys = sorted(GOOD_GREYS)
    folds = len(run.good_units)
    assert len(run.fake.fits) == 2 * folds
    for index, (windows, settings) in enumerate(run.fake.fits):
        held = greys[index % folds]
        assert len(windows) == 3 * (folds - 1)
        seen = sorted(round((w.max() * 0.225 + 0.406) * 255) for w in windows)
        assert seen == sorted([g for g in greys if g != held] * 3)
        assert all(w.min() > -1.9 for w in windows)
        assert (settings.coreset_ratio, settings.seed, settings.device) == (0.1, 42, "cuda")


# --- variant sessions: the main frames, re-imported with a difference -----------------------


def test_a_discarded_unit_appears_nowhere(tmp_path: Path, run: PilotRun) -> None:
    broken = run.good_units[0]
    shot = discard_broken(run.plan, [(broken, 2)])
    on_disk = set(shot.discarded[broken])
    drop = {f"{broken}_good_{g}_r{r}.png" for g in GEOMETRIES for r in range(3)} - on_disk
    folder = _like_main(tmp_path, run, shot, {}, drop)
    code, out, fake, report_dir = _run(tmp_path, run.config, folder)
    assert code == 0, out
    text = (report_dir / f"{SESSION}.md").read_text(encoding="utf-8")
    assert broken not in text
    rows = list(csv.DictReader((report_dir / f"{SESSION}-histogram.csv")
                               .open(encoding="utf-8", newline="")))
    sums: dict[str, int] = {}
    for row in rows:
        sums[row["geometry"]] = sums.get(row["geometry"], 0) + int(row["count"])
    assert sums == {geometry: 15 * ROI_PX for geometry in GEOMETRIES}
    folds = len(run.good_units) - 1
    assert len(fake.fits) == 2 * folds
    assert all(len(windows) == 3 * (folds - 1) for windows, _ in fake.fits)


def test_a_good_frame_with_two_units_puts_the_geometry_out(
    tmp_path: Path, run: PilotRun
) -> None:
    victim = run.good_units[0]
    folder = _like_main(
        tmp_path, run, None, {f"{victim}_good_G2_r0.png": _two_unit_frame()}
    )
    code, out, _, report_dir = _run(tmp_path, run.config, folder)
    assert code == 0, out
    sections = _sections((report_dir / f"{SESSION}.md").read_text(encoding="utf-8"))
    g2 = sections["G2"]
    assert "more than one unit" in g2
    assert f"{victim}_good_G2_r0.png: more than one unit" in g2
    line = re.search(r"- silhouette[^\n]*", g2)[0]
    assert f"{victim}: 2 of 3 rotations" in line
    assert f"{victim} " not in line
    assert "silhouette, spread < 2%: out" in g2
    assert "Recommendation: G3." in sections["Ranking"]
    assert "- G2: out by silhouette" in sections["Ranking"]


def test_no_geometry_passing_names_the_two_frame_fallback(
    tmp_path: Path, run: PilotRun
) -> None:
    victim = run.good_units[0]
    replace = {
        f"{victim}_good_{geometry}_r0.png": _two_unit_frame()
        for geometry in GEOMETRIES
    }
    folder = _like_main(tmp_path, run, None, replace)
    code, out, _, report_dir = _run(tmp_path, run.config, folder)
    assert code == 1, out
    text = (report_dir / f"{SESSION}.md").read_text(encoding="utf-8")
    assert "No geometry passes the silhouette criterion." in text
    assert "ADR-0013's two-frame design" in text
    assert "no geometry passed the silhouette criterion" in out


# --- refusals -------------------------------------------------------------------------------


def test_a_dataset_session_is_refused(tmp_path: Path, run: PilotRun) -> None:
    folder = tmp_path / "session"
    folder.mkdir()
    (folder / "session.yml").write_text("kind: dataset\nunits: {}\n", encoding="utf-8")
    code, out, _, report_dir = _run(tmp_path, run.config, folder)
    lines = out.strip().splitlines()
    assert code == 1
    assert len(lines) == 1 and "not a pilot session" in lines[0]
    assert "Traceback" not in out
    assert not report_dir.exists()


def test_a_missing_session_yml_cannot_run(tmp_path: Path, run: PilotRun) -> None:
    folder = tmp_path / "session"
    folder.mkdir()
    code, out, _, _ = _run(tmp_path, run.config, folder)
    lines = out.strip().splitlines()
    assert code == 2
    assert len(lines) == 1 and lines[0].startswith("session: ")
    assert "session.yml" in lines[0] and "Traceback" not in out


def test_a_broken_config_cannot_run(tmp_path: Path, run: PilotRun) -> None:
    bad = tmp_path / "bad.yml"
    bad.write_text(
        run.config.read_text(encoding="utf-8").replace("backend:   torch-cuda",
                                                       "backend:   tpu"),
        encoding="utf-8",
    )
    code, out, _, _ = _run(tmp_path, bad, run.session)
    lines = out.strip().splitlines()
    assert code == 2
    assert len(lines) == 1 and lines[0].startswith("config: ")


def test_torch_cuda_without_cuda_cannot_run(tmp_path: Path, run: PilotRun) -> None:
    code, out, _, _ = _run(tmp_path, run.config, run.session, cuda=False)
    lines = out.strip().splitlines()
    assert code == 2
    assert len(lines) == 1 and "no CPU fallback" in lines[0]


def test_missing_weights_cannot_run(tmp_path: Path, run: PilotRun) -> None:
    built: list[str] = []

    def make(device: str) -> PatchCore:
        return load_backbone(lambda: None, lambda: built.append(device) or BlotPatchCore())

    code, out, _, _ = _run(tmp_path, run.config, run.session, model=make)
    lines = out.strip().splitlines()
    assert code == 2
    assert len(lines) == 1 and lines[0].startswith("weights: ")
    assert built == []
