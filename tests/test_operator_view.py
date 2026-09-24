"""The operator view (#86): every screen of docs/16 §5, checked by its pixels and strings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qs.config import LabelsConfig, load_config
from qs.contracts import Label
from qs.operator_view import (
    AMBER,
    BANNER,
    GREEN,
    GREY,
    RED,
    Shown,
    Status,
    View,
    banner_line,
    fault_line,
    label_word,
    panel_line,
    reason_line,
    render,
    status_line,
)

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "DejaVuSans.ttf"
EXAMPLE = ROOT / "config" / "station-A-bench.example.yml"
LABELS = load_config(EXAMPLE).config.display.labels

READY = Status("ready", None, frozenset(), None)
STARTING = Status("starting", None, frozenset(), None)
MODES = {
    "starting": STARTING,
    "ready": READY,
    "fault": Status("fault", "camera", frozenset(), None),
}


def uniform(colour: tuple[int, int, int], shape: tuple[int, int] = (120, 160)) -> np.ndarray:
    return np.full((*shape, 3), colour, dtype=np.uint8)


def shown(label: Label, reason: str | None = None, **changes: object) -> Shown:
    values: dict[str, object] = dict(seq=7, label=label, reason=reason, detail="", image=None)
    values.update(changes)
    return Shown(**values)  # type: ignore[arg-type]


def view(
    status: Status = READY,
    *,
    live: np.ndarray | None = None,
    last: Shown | None = None,
    counts: dict[Label, int] | None = None,
    recent: tuple[tuple[int | None, Label], ...] = (),
) -> View:
    return View(status=status, live=live, last=last, counts=counts or {}, recent=recent)


def pixels(v: View, labels: LabelsConfig = LABELS) -> np.ndarray:
    return np.asarray(render(v, labels, FONT))


def has_colour(image: np.ndarray, colour: tuple[int, int, int]) -> bool:
    return bool(np.any(np.all(image == np.array(colour, dtype=np.uint8), axis=-1)))


# AC1 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(MODES))
def test_render_is_a_1280_by_800_rgb_image_in_every_mode(mode: str) -> None:
    image = render(view(MODES[mode]), LABELS, FONT)
    assert image.mode == "RGB"
    assert image.size == (1280, 800)


def test_render_leaves_read_only_inputs_unchanged_and_is_deterministic() -> None:
    live = uniform((255, 0, 0))
    image = uniform((10, 200, 30))
    live.setflags(write=False)
    image.setflags(write=False)
    before = (live.copy(), image.copy())
    v = view(
        READY,
        live=live,
        last=shown(Label.DEFECT, detail="площадь 90000 px < 98000", image=image),
        counts={Label.DEFECT: 1},
        recent=((7, Label.DEFECT),),
    )
    first = render(v, LABELS, FONT).tobytes()
    second = render(v, LABELS, FONT).tobytes()
    assert first == second
    assert np.array_equal(live, before[0])
    assert np.array_equal(image, before[1])


# AC2 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("last", "colour"),
    [
        (shown(Label.PASS), GREEN),
        (shown(Label.DEFECT), RED),
        *[
            (shown(Label.NO_VERDICT, reason), AMBER)
            for reason in ("placement", "lighting", "capture", "camera_settings",
                           "inference", "camera", "busy")
        ],
        (None, GREY),
    ],
)
def test_the_band_takes_the_last_verdicts_colour(last: Shown | None, colour: tuple) -> None:
    image = render(view(READY, last=last), LABELS, FONT)
    assert image.getpixel((4, 40)) == colour
    assert image.getpixel((1275, 232)) == colour


# AC3 ------------------------------------------------------------------------------------

REASONS = ("placement", "lighting", "capture", "camera_settings", "inference", "camera", "busy")


@pytest.mark.parametrize("mode", ["ready", "starting"])
@pytest.mark.parametrize("reason", REASONS)
def test_no_verdict_never_draws_a_green_pixel(reason: str, mode: str) -> None:
    v = view(MODES[mode], last=shown(Label.NO_VERDICT, reason, detail=reason))
    assert not has_colour(pixels(v), GREEN)


# AC4 ------------------------------------------------------------------------------------


def word_box(v: View, band: tuple[int, int, int], labels: LabelsConfig = LABELS) -> tuple:
    region = pixels(v, labels)[36:186]
    ys, xs = np.nonzero(np.any(region != np.array(band, dtype=np.uint8), axis=-1))
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)


@pytest.mark.parametrize(
    ("label", "band"), [(Label.PASS, GREEN), (Label.DEFECT, RED), (Label.NO_VERDICT, AMBER)]
)
def test_the_three_words_are_at_least_80_px_tall_and_fit_the_width(
    label: Label, band: tuple[int, int, int]
) -> None:
    reason = "placement" if label is Label.NO_VERDICT else None
    height, width = word_box(view(last=shown(label, reason)), band)
    assert height >= 80
    assert width <= 1240


def test_the_busy_word_fits_and_is_at_least_50_px_tall() -> None:
    height, width = word_box(view(last=shown(Label.NO_VERDICT, "busy")), AMBER)
    assert height >= 50
    assert width <= 1240


def test_the_words_come_from_the_config(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        "{ pass: ГОДЕН, defect: БРАК, no_verdict: НЕТ РЕШЕНИЯ }",
        "{ pass: OK, defect: NG, no_verdict: '??' }",
    )
    copy = tmp_path / "station.yml"
    copy.write_text(text, encoding="utf-8")
    changed = load_config(copy).config.display.labels
    assert changed.pass_ == "OK"
    for label, reason in ((Label.PASS, None), (Label.DEFECT, None),
                          (Label.NO_VERDICT, "placement")):
        v = view(last=shown(label, reason))
        assert not np.array_equal(pixels(v)[36:236], pixels(v, changed)[36:236])


# AC5 ------------------------------------------------------------------------------------


def test_status_line() -> None:
    assert status_line(STARTING, LABELS) == "ЗАПУСК"
    assert status_line(READY, LABELS) == "ГОТОВ — положите изделие в рамку и нажмите ПРОБЕЛ"
    captioned = Status("ready", None, frozenset(), "ПОВТОР: кадр 3 из 5")
    assert status_line(captioned, LABELS) == "ПОВТОР: кадр 3 из 5"
    assert status_line(Status("fault", "camera", frozenset(), None), LABELS) == (
        "СТАНЦИЯ НЕ ПРОВЕРЯЕТ"
    )


@pytest.mark.parametrize(
    ("reason", "line"),
    [
        ("placement", "положите изделие в рамку"),
        ("lighting", "проверьте освещение"),
        ("capture", "нет кадра"),
        ("camera_settings", "настройки камеры изменились"),
        ("inference", "ошибка анализа"),
        ("camera", "камера не подключена"),
        ("busy", "изделие не проверено"),
        ("something else", "нет решения"),
        (None, "нет решения"),
    ],
)
def test_reason_line(reason: str | None, line: str) -> None:
    assert reason_line(reason) == line


@pytest.mark.parametrize(
    ("reason", "line"),
    [
        ("camera", "камера не подключена"),
        ("capture", "нет кадров"),
        ("inference", "ошибка анализа — перезапустите станцию"),
    ],
)
def test_fault_line(reason: str, line: str) -> None:
    assert fault_line(reason) == line


@pytest.mark.parametrize(
    ("warnings", "line"),
    [
        (set(), ""),
        ({"STORAGE"}, "архив не записывается"),
        ({"LIGHTING"}, "освещение вне нормы"),
        ({"CAMERA_SETTINGS"}, "настройки камеры не подтверждены"),
        ({"LATENCY"}, "медленная работа"),
        (
            {"LATENCY", "CAMERA_SETTINGS", "LIGHTING", "STORAGE"},
            "архив не записывается · освещение вне нормы · "
            "настройки камеры не подтверждены · медленная работа",
        ),
    ],
)
def test_banner_line(warnings: set[str], line: str) -> None:
    assert banner_line(frozenset(warnings)) == line


def test_panel_line() -> None:
    assert panel_line({}, LABELS) == "ГОДЕН 0 · БРАК 0 · НЕТ РЕШЕНИЯ 0"
    assert panel_line({Label.NO_VERDICT: 4, Label.PASS: 12}, LABELS) == (
        "ГОДЕН 12 · БРАК 0 · НЕТ РЕШЕНИЯ 4"
    )


@pytest.mark.parametrize(
    ("label", "reason", "word"),
    [
        (Label.PASS, None, "ГОДЕН"),
        (Label.DEFECT, None, "БРАК"),
        (Label.NO_VERDICT, "placement", "НЕТ РЕШЕНИЯ"),
        (Label.NO_VERDICT, "busy", "ИЗДЕЛИЕ НЕ ПРОВЕРЕНО"),
    ],
)
def test_label_word(label: Label, reason: str | None, word: str) -> None:
    assert label_word(label, reason, LABELS) == word


# AC6 ------------------------------------------------------------------------------------

BLUE_RGB = (0, 0, 255)


@pytest.mark.parametrize("reason", ["camera", "capture", "inference"])
def test_the_fault_screen_is_all_amber_with_nothing_else(reason: str) -> None:
    v = view(
        Status("fault", reason, frozenset({"STORAGE", "LIGHTING"}), None),
        live=uniform((255, 0, 0)),
        last=shown(Label.PASS, image=uniform(BLUE_RGB)),
        counts={Label.PASS: 12},
        recent=tuple((i, Label.PASS) for i in range(1, 11)),
    )
    image = render(v, LABELS, FONT)
    for point in ((4, 4), (1275, 4), (4, 795), (1275, 795), (640, 4), (640, 795)):
        assert image.getpixel(point) == AMBER
    array = np.asarray(image)
    assert not has_colour(array, BLUE_RGB)
    assert not has_colour(array, GREEN)


def test_the_three_fault_reasons_give_three_different_screens() -> None:
    images = {
        render(view(Status("fault", reason, frozenset(), None)), LABELS, FONT).tobytes()
        for reason in ("camera", "capture", "inference")
    }
    assert len(images) == 3


# AC7 ------------------------------------------------------------------------------------


def test_the_live_view_is_drawn_only_when_ready() -> None:
    blue_bgr = uniform((255, 0, 0))
    assert render(view(READY, live=blue_bgr), LABELS, FONT).getpixel((320, 456)) == BLUE_RGB
    assert not has_colour(pixels(view(STARTING, live=blue_bgr)), BLUE_RGB)


def test_the_replay_caption_stands_in_for_the_live_view() -> None:
    captioned = Status("ready", None, frozenset(), "ПОВТОР: кадр 3 из 5")
    left = pixels(view(captioned))[236:676, 0:640]
    assert has_colour(left, (255, 255, 255))
    assert not has_colour(left, BLUE_RGB)


# AC8 ------------------------------------------------------------------------------------


def test_the_verdict_image_fills_the_right_box() -> None:
    colour = (10, 200, 30)
    image = render(view(last=shown(Label.PASS, image=uniform(colour))), LABELS, FONT)
    assert image.getpixel((960, 456)) == colour


def test_no_verdict_image_shows_no_frame_text() -> None:
    right = pixels(view(last=shown(Label.PASS, image=None)))[236:676, 640:1280]
    assert has_colour(right, (255, 255, 255))
    assert not has_colour(right, (10, 200, 30))


# AC9 ------------------------------------------------------------------------------------


def test_no_warning_leaves_the_banner_black() -> None:
    assert pixels(view(READY))[0:36].max() == 0


def test_a_warning_fills_the_banner() -> None:
    lit = Status("ready", None, frozenset({"LIGHTING"}), None)
    assert render(view(lit), LABELS, FONT).getpixel((2, 2)) == BANNER


# AC10 -----------------------------------------------------------------------------------


def test_the_recent_squares_sit_newest_on_the_right() -> None:
    recent = ((1, Label.PASS), (2, Label.DEFECT), (3, Label.NO_VERDICT))
    image = render(view(recent=recent), LABELS, FONT)
    assert image.getpixel((1156, 708)) == GREEN
    assert image.getpixel((1204, 708)) == RED
    assert image.getpixel((1252, 708)) == AMBER


def test_only_the_newest_ten_are_drawn() -> None:
    kinds = (Label.PASS, Label.DEFECT, Label.NO_VERDICT)
    eleven = tuple((i, kinds[i % 3]) for i in range(1, 12))
    assert np.array_equal(pixels(view(recent=eleven)), pixels(view(recent=eleven[1:])))
    assert not np.array_equal(pixels(view(recent=eleven)), pixels(view(recent=eleven[:10])))


# AC11 -----------------------------------------------------------------------------------


def test_the_status_bar_differs_between_starting_and_ready() -> None:
    starting = pixels(view(STARTING))[740:800]
    ready = pixels(view(READY))[740:800]
    assert not np.array_equal(starting, ready)
    assert has_colour(starting, (255, 255, 255))
    assert has_colour(ready, (255, 255, 255))


def test_status_refuses_an_unknown_mode_or_warning() -> None:
    with pytest.raises(ValueError):
        Status("sleeping", None, frozenset(), None)
    with pytest.raises(ValueError):
        Status("ready", None, frozenset({"NOISE"}), None)
