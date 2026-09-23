"""`AppearanceInspector` with a fake scorer, on synthetic discs (docs/11 §2, docs/15 §4–§6).

A small random bank is written with `write_bundle` into `tmp_path`, the example config is
rewritten to name it, and the scorer is a numpy fake whose anomaly map is one constant
value — so the masked maximum, the score, is exactly that value in float32. CUDA is faked
available unless a test says otherwise. No GPU, no weights, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from appearance_fixtures import config_naming, frame_of
from conftest import Paint
from qs.config import LoadedConfig
from qs.contracts import CHECK_APPEARANCE, CHECK_GEOMETRY, CheckResult, Label, Verdict
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.inspector.appearance import AppearanceInspector
from qs.inspector.bundle import BundleInfo, ImageRecord, write_bundle
from qs.inspector.localize import localize
from qs.inspector.model import Scored

WINDOW_PX = 400
INPUT_PX = 64
CENTRE = (960, 540)


class ConstantScorer:
    """A scorer whose anomaly map is `value` everywhere; it records every input it is given."""

    def __init__(self, value: float = 1.0, pred_score: float = 0.75) -> None:
        self.value = value
        self.pred_score = pred_score
        self.inputs: list[np.ndarray] = []

    def score(self, window_input: np.ndarray) -> Scored:
        self.inputs.append(np.array(window_input, copy=True))
        anomaly_map = np.full(window_input.shape[1:], self.value, dtype=np.float32)
        return Scored(anomaly_map, self.pred_score, backbone_ms=1.0, search_ms=2.0)


class Factory:
    """`make_scorer`: records the device and bank it was asked for, returns one scorer."""

    def __init__(self, scorer: ConstantScorer) -> None:
        self.scorer = scorer
        self.calls: list[tuple[str, np.ndarray]] = []

    def __call__(self, device: str, bank: np.ndarray) -> ConstantScorer:
        self.calls.append((device, bank))
        return self.scorer


def _bank() -> np.ndarray:
    return np.random.default_rng(3).random((20, 8), dtype=np.float32)


def _bundle(models: Path, data_source: str = "own") -> str:
    images = tuple(
        ImageRecord(f"u{u:03d}_good_G2_r0.png", f"u{u:03d}", f"{u:02d}".ljust(64, "b"))
        for u in (1, 2, 3)
    )
    info = BundleInfo(input_px=INPUT_PX, window_px=WINDOW_PX, coreset_ratio=0.1, seed=42,
                      device="cuda", data_source=data_source, images=images)
    return write_bundle(models, "mochi", _bank(), info)


@pytest.fixture
def models(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def loaded(tmp_path: Path, models: Path) -> LoadedConfig:
    return config_naming(tmp_path, _bundle(models))


def _disc(radius: int, centre: tuple[int, int] = CENTRE) -> np.ndarray:
    paint = Paint()
    pixels = paint.floor()
    paint.disc(pixels, centre, radius)
    return pixels


def _inspector(
    loaded: LoadedConfig, models: Path, scorer: ConstantScorer, *, cuda: bool = True,
    validation: bool = False,
) -> tuple[AppearanceInspector, Factory]:
    factory = Factory(scorer)
    inspector = AppearanceInspector(
        loaded.config_hash, models, validation=validation, make_scorer=factory,
        cuda_available=lambda: cuda,
    )
    return inspector, factory


def test_a_good_disc_with_a_low_score_passes_with_both_checks(
    loaded: LoadedConfig, models: Path
) -> None:
    inspector, _ = _inspector(loaded, models, ConstantScorer(1.0, pred_score=0.75))
    inspector.load(loaded.config)
    pixels = _disc(190)
    verdict = inspector.inspect(frame_of(pixels))

    assert verdict.label is Label.PASS
    geometry, appearance = verdict.checks
    assert geometry.name == CHECK_GEOMETRY and geometry.passed is True
    assert appearance.name == CHECK_APPEARANCE and appearance.passed is True
    assert appearance.value == 1.0
    assert appearance.limit == 1.83
    assert appearance.measures == (("pred_score", 0.75),)
    assert appearance.failed_on is None

    assert verdict.heatmap is not None
    assert verdict.heatmap.dtype == np.float32
    assert verdict.heatmap.shape == (WINDOW_PX, WINDOW_PX)
    assert verdict.heatmap[WINDOW_PX // 2, WINDOW_PX // 2] == 1.0  # on the unit
    assert verdict.heatmap[0, 0] == 0.0  # the floor, masked off
    assert verdict.window_box == (760, 340, WINDOW_PX, WINDOW_PX)  # centred on (960, 540)
    assert verdict.unit_box == (770, 350, 381, 381)
    assert verdict.unit_box == localize(pixels, loaded.config.inspector.localize).unit_box

    assert verdict.model_id == inspector.model_id == loaded.config.inspector.model
    assert verdict.threshold_id == loaded.config.threshold.id
    assert verdict.config_hash == loaded.config_hash
    assert verdict.inspect_ms > 0
    assert verdict.reason is None


def _appearance_of(
    loaded: LoadedConfig, models: Path, value: float, radius: int = 190
) -> tuple[Verdict, CheckResult]:
    inspector, _ = _inspector(loaded, models, ConstantScorer(value))
    inspector.load(loaded.config)
    verdict = inspector.inspect(frame_of(_disc(radius)))
    return verdict, verdict.checks[1]


def test_a_score_equal_to_the_threshold_passes(tmp_path: Path, models: Path) -> None:
    at = float(np.float32(1.83))  # the float32 score nearest 1.83, the threshold exactly
    loaded = config_naming(
        tmp_path, _bundle(models), replace=(("value:  1.83", f"value:  {at!r}"),)
    )
    verdict, appearance = _appearance_of(loaded, models, at)
    assert appearance.value == appearance.limit == at
    assert appearance.passed is True
    assert verdict.label is Label.PASS


def test_a_score_above_the_threshold_is_a_defect(loaded: LoadedConfig, models: Path) -> None:
    verdict, appearance = _appearance_of(loaded, models, 1.8301)
    assert verdict.label is Label.DEFECT
    assert appearance.passed is False
    assert appearance.failed_on == "score>threshold"
    assert appearance.limit == 1.83
    assert verdict.checks[0].passed is True  # the geometry still ran, and passed


def test_the_float32_score_of_1_83_lies_above_a_threshold_of_1_83_and_is_not_passed(
    loaded: LoadedConfig, models: Path
) -> None:
    # 1.83 has no float32; the model's nearest, 1.8300000429…, is above the decimal line.
    verdict, appearance = _appearance_of(loaded, models, 1.83)
    assert appearance.value is not None and appearance.value > 1.83
    assert appearance.passed is False
    assert verdict.label is Label.DEFECT


def test_an_oversized_disc_is_a_geometry_defect_with_both_checks_present(
    loaded: LoadedConfig, models: Path
) -> None:
    verdict, appearance = _appearance_of(loaded, models, 0.5, radius=210)
    assert verdict.label is Label.DEFECT
    geometry = verdict.checks[0]
    assert [check.name for check in verdict.checks] == [CHECK_GEOMETRY, CHECK_APPEARANCE]
    assert geometry.passed is False
    assert geometry.failed_on == "area_px>max"
    assert appearance.passed is True
    assert appearance.value == 0.5
    assert verdict.heatmap is not None


def test_an_empty_floor_raises_no_unit_and_never_scores(
    loaded: LoadedConfig, models: Path
) -> None:
    scorer = ConstantScorer(0.5)
    inspector, _ = _inspector(loaded, models, scorer)
    inspector.load(loaded.config)
    with pytest.raises(PlacementFault) as caught:
        inspector.inspect(frame_of(Paint().floor()))
    assert caught.value.detail == "no unit"
    assert len(scorer.inputs) == 1  # the warm-up only


def test_a_nan_map_is_an_inference_error(loaded: LoadedConfig, models: Path) -> None:
    scorer = ConstantScorer(0.5)
    inspector, _ = _inspector(loaded, models, scorer)
    inspector.load(loaded.config)
    scorer.value = float("nan")
    with pytest.raises(InferenceError):
        inspector.inspect(frame_of(_disc(190)))


@pytest.mark.parametrize("pred_score", [float("nan"), float("inf")])
def test_a_non_finite_pred_score_is_an_inference_error(
    loaded: LoadedConfig, models: Path, pred_score: float
) -> None:
    scorer = ConstantScorer(0.5)
    inspector, _ = _inspector(loaded, models, scorer)
    inspector.load(loaded.config)
    scorer.pred_score = pred_score
    with pytest.raises(InferenceError, match="pred_score"):
        inspector.inspect(frame_of(_disc(190)))


class CrashingScorer(ConstantScorer):
    """Warms up, then raises what a CUDA fault would."""

    def score(self, window_input: np.ndarray) -> Scored:
        if self.inputs:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return super().score(window_input)


def test_any_other_scorer_exception_propagates_unwrapped(
    loaded: LoadedConfig, models: Path
) -> None:
    inspector, _ = _inspector(loaded, models, CrashingScorer())
    inspector.load(loaded.config)
    with pytest.raises(RuntimeError, match="illegal memory access") as caught:
        inspector.inspect(frame_of(_disc(190)))
    assert type(caught.value) is RuntimeError


def _refused(
    loaded: LoadedConfig, models: Path, match: str, *, cuda: bool = True,
    validation: bool = False,
) -> Factory:
    inspector, factory = _inspector(loaded, models, ConstantScorer(), cuda=cuda,
                                    validation=validation)
    with pytest.raises(ConfigError, match=match) as caught:
        inspector.load(loaded.config)
    assert "\n" not in str(caught.value)
    with pytest.raises(InferenceError, match="not loaded"):
        inspector.inspect(frame_of(_disc(190)))
    return factory


def test_load_refuses_a_model_id_with_no_directory(tmp_path: Path, models: Path) -> None:
    loaded = config_naming(tmp_path, "patchcore-mochi-v1-3f9a1c2b7d4e")
    factory = _refused(loaded, models, "no such bundle directory")
    assert factory.calls == []


def test_load_refuses_a_tampered_bundle(loaded: LoadedConfig, models: Path) -> None:
    yml = models / loaded.config.inspector.model / "bundle.yml"
    yml.write_text(yml.read_text(encoding="utf-8").replace("seed: 42", "seed: 43"),
                   encoding="utf-8")
    factory = _refused(loaded, models, "does not match its model_id")
    assert factory.calls == []


def test_a_validation_only_bundle_loads_only_in_validation_mode(
    tmp_path: Path, models: Path
) -> None:
    loaded = config_naming(tmp_path, _bundle(models, data_source="external:mvtec-ad"))
    _refused(loaded, models, "validation-only .* refused outside validation mode")

    inspector, _ = _inspector(loaded, models, ConstantScorer(1.0), validation=True)
    inspector.load(loaded.config)
    assert inspector.model_id == loaded.config.inspector.model
    assert inspector.inspect(frame_of(_disc(190))).label is Label.PASS


def test_torch_cuda_without_cuda_is_refused_with_no_cpu_fallback(
    loaded: LoadedConfig, models: Path
) -> None:
    assert loaded.config.inspector.backend == "torch-cuda"
    factory = _refused(loaded, models, "CUDA is not available — no CPU fallback", cuda=False)
    assert factory.calls == []  # no scorer was built, on any device


def test_the_scorer_is_built_on_the_backend_device_from_the_bundle_bank(
    tmp_path: Path, models: Path
) -> None:
    model_id = _bundle(models)
    for backend, device in (("torch-cuda", "cuda"), ("torch-cpu", "cpu")):
        loaded = config_naming(tmp_path, model_id, replace=(
            ("backend:   torch-cuda", f"backend:   {backend}"),
            ("backend: torch-cuda", f"backend: {backend}"),
        ))
        inspector, factory = _inspector(loaded, models, ConstantScorer())
        inspector.load(loaded.config)
        [(built_on, bank)] = factory.calls
        assert built_on == device
        assert np.array_equal(bank, _bank())


def test_load_warms_up_once_on_a_black_input_window(
    loaded: LoadedConfig, models: Path
) -> None:
    scorer = ConstantScorer(1.0)
    inspector, _ = _inspector(loaded, models, scorer)
    inspector.load(loaded.config)

    [warm_up] = scorer.inputs
    assert warm_up.shape == (3, INPUT_PX, INPUT_PX)
    assert warm_up.dtype == np.float32
    # black, after the ImageNet normalisation: −mean / std in each RGB channel
    for channel, (mean, std) in enumerate(zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                                             strict=True)):
        np.testing.assert_allclose(warm_up[channel], -mean / std, rtol=1e-6)

    inspector.inspect(frame_of(_disc(190)))
    assert len(scorer.inputs) == 2  # and the next call is the first inspection's


def _same_but_timing(first: Verdict, second: Verdict) -> None:
    assert first.label is second.label
    assert first.checks == second.checks
    assert first.heatmap is not None and second.heatmap is not None
    assert np.array_equal(first.heatmap, second.heatmap)
    assert (first.window_box, first.unit_box, first.model_id, first.threshold_id,
            first.config_hash, first.reason) == (second.window_box, second.unit_box,
                                                 second.model_id, second.threshold_id,
                                                 second.config_hash, second.reason)


def test_inspect_is_pure(
    loaded: LoadedConfig, models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import io
    import os

    inspector, _ = _inspector(loaded, models, ConstantScorer(1.0))
    inspector.load(loaded.config)
    pixels = _disc(190)
    before = pixels.tobytes()
    frame = frame_of(pixels)

    def no_file(*args: object, **kwargs: object) -> None:
        raise AssertionError("inspect() opened a file")

    with monkeypatch.context() as patched:
        for module, name in ((builtins, "open"), (io, "open"), (os, "open")):
            patched.setattr(module, name, no_file)
        first = inspector.inspect(frame)
        second = inspector.inspect(frame)

    assert pixels.tobytes() == before
    _same_but_timing(first, second)
    assert first.label is Label.PASS


def test_a_read_only_frame_is_inspected(loaded: LoadedConfig, models: Path) -> None:
    inspector, _ = _inspector(loaded, models, ConstantScorer(1.0))
    inspector.load(loaded.config)
    pixels = _disc(190)
    pixels.setflags(write=False)
    assert inspector.inspect(frame_of(pixels)).label is Label.PASS


def test_a_frame_between_two_others_is_judged_alone(
    loaded: LoadedConfig, models: Path
) -> None:
    inspector, _ = _inspector(loaded, models, ConstantScorer(1.0))
    inspector.load(loaded.config)
    alone = inspector.inspect(frame_of(_disc(190), seq=1))
    inspector.inspect(frame_of(_disc(210, (900, 500)), seq=2))
    again = inspector.inspect(frame_of(_disc(190), seq=3))
    _same_but_timing(alone, again)
