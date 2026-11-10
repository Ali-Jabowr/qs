"""`AppearanceInspector`: Station A's `Inspector` — both checks in one verdict (docs/11 §2).

`load(config)` loads and verifies the bundle `inspector.model` names (`load_bundle`; every
refusal is a `ConfigError`, docs/16 §2), selects the device from `inspector.backend` —
`torch-cuda` without CUDA is refused, never run on the CPU (docs/16 §3) — builds the scorer
on the bundle's bank, and warms it up with one call on a black window, so the first real
unit does not pay for loading weights onto the GPU (docs/16 §2, step 4). `window_px`,
`input_px` and the normalisation come from the bundle only (docs/12 §4); the loader has
already refused a bundle whose normalisation is not the one `window.to_input` applies.
The real scorer is `qs.inspector.patchcore`, which sets the determinism flags before the
model loads (docs/15 §9); tests pass a fake through `make_scorer`.

`inspect(frame)` runs the chain of docs/15 §2–§6 on the frame alone: localize (a
`PlacementFault` goes straight through), measure and check the geometry against
`threshold.geometry` (ADR-0008), cut the window centred on the unit, score it and mask the
map to the unit (ADR-0009), then check the score against `threshold.value` — exactly: the
float32 score, as a float, is compared with the threshold as written, so a score a float32
rounding puts above the line is never passed, and every record satisfies `passed == (value
<= limit)`. The model's own `pred_score` is logged beside it and must be finite. Every check the
station lists in `inspector.checks` always runs (geometry is left out where the list omits it,
but `localize` still runs: the window is centred on the silhouette), and the label comes from
them alone (C8). A non-finite score is an `InferenceError`; any other exception is left to the
Pipeline (docs/11 §7). Pure: no I/O, the frame is only ever read, and nothing changes between
frames (C1, C2).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qs.config import StationConfig
from qs.contracts import (
    CHECK_APPEARANCE,
    CHECK_GEOMETRY,
    CheckResult,
    Frame,
    Verdict,
    derive_label,
    only_frame,
)
from qs.errors import InferenceError
from qs.inspector.bundle import Bundle, load_bundle
from qs.inspector.geometry import check_geometry, measure
from qs.inspector.localize import localize
from qs.inspector.model import Scorer, device_for
from qs.inspector.window import cut_window, score_map, to_input


def real_scorer(device: str, bank: np.ndarray) -> Scorer:
    """The real PatchCore scorer on `device`; torch and Anomalib are imported here only."""
    from qs.inspector.patchcore import AnomalibPatchCore

    return AnomalibPatchCore(device).scorer(bank)


def real_cuda_available() -> bool:
    """Whether torch sees a CUDA device; torch is imported here only."""
    from qs.inspector.patchcore import cuda_available

    return cuda_available()


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A verified bundle and the scorer built on its bank, warmed up."""

    bundle: Bundle
    scorer: Scorer


def load_model(
    models_root: Path,
    config: StationConfig,
    *,
    validation: bool,
    make_scorer: Callable[[str, np.ndarray], Scorer],
    cuda_available: Callable[[], bool],
) -> LoadedModel:
    """The verified bundle `config.inspector.model` names and its scorer, warmed up on a black
    window."""
    bundle = load_bundle(models_root, config.inspector.model, validation=validation)
    device = device_for(config.inspector.backend, cuda_available=cuda_available())
    scorer = make_scorer(device, bundle.bank)
    black = np.zeros((bundle.window_px, bundle.window_px, 3), dtype=np.uint8)
    scorer.score(to_input(black, bundle.input_px))
    return LoadedModel(bundle, scorer)


def check_appearance(
    pixels: np.ndarray,
    centre: tuple[float, float],
    mask: np.ndarray,
    model: LoadedModel,
    config: StationConfig,
) -> tuple[CheckResult, np.ndarray, tuple[int, int, int, int]]:
    """The appearance check on the window centred on `centre`, with the map masked to the
    frame-sized `mask`: the check, the heatmap and the window's box."""
    bundle = model.bundle
    window, window_box = cut_window(pixels, centre, bundle.window_px)
    scored = model.scorer.score(to_input(window, bundle.input_px))
    heatmap, score = score_map(
        scored.anomaly_map, mask, window_box, bundle.window_px, config.inspector.mask_dilate_px
    )
    pred_score = float(scored.pred_score)
    if not math.isfinite(pred_score):
        raise InferenceError(f"the model's pred_score {pred_score} is not finite")
    limit = config.threshold.value
    passed = score <= limit
    check = CheckResult(
        name=CHECK_APPEARANCE,
        passed=passed,
        value=score,
        limit=limit,
        measures=(("pred_score", pred_score),),
        failed_on=None if passed else "score>threshold",
    )
    return check, heatmap, window_box


@dataclass(frozen=True, slots=True)
class _Loaded:
    """Everything `load` fixed: the config and the loaded model."""

    config: StationConfig
    model: LoadedModel


class AppearanceInspector:
    """The `Inspector` of Station A: geometry and PatchCore appearance (ADR-0008, ADR-0009).

    `config_hash` is the loaded config's hash (C6); `models_root` holds the bundles.
    `validation=True` admits a validation-only bundle and is passed only by `qs
    mvtec-check` and tests — never a config key, never a default (ADR-0016).
    `make_scorer(device, bank)` builds the scorer; `cuda_available` asks torch.
    """

    def __init__(
        self,
        config_hash: str,
        models_root: str | Path = "models",
        *,
        validation: bool = False,
        make_scorer: Callable[[str, np.ndarray], Scorer] = real_scorer,
        cuda_available: Callable[[], bool] = real_cuda_available,
    ) -> None:
        self._config_hash = config_hash
        self._models_root = Path(models_root)
        self._validation = validation
        self._make_scorer = make_scorer
        self._cuda_available = cuda_available
        self._loaded: _Loaded | None = None

    @property
    def model_id(self) -> str:
        """The loaded bundle's `model_id`."""
        if self._loaded is None:
            raise InferenceError("the appearance inspector was not loaded")
        return self._loaded.model.bundle.model_id

    def load(self, config: StationConfig) -> None:
        """Load, verify and warm up the bundle `config.inspector.model` names."""
        self._loaded = None
        model = load_model(
            self._models_root, config, validation=self._validation,
            make_scorer=self._make_scorer, cuda_available=self._cuda_available,
        )
        self._loaded = _Loaded(config, model)

    def inspect_burst(self, frames: Sequence[Frame]) -> Verdict:
        """A burst of 1 is `inspect`; the bench inspector judges no burst longer than 1."""
        return self.inspect(only_frame(frames))

    def inspect(self, frame: Frame) -> Verdict:
        """One `Verdict` carrying the geometry and appearance checks (C8)."""
        start = time.perf_counter()
        loaded = self._loaded
        if loaded is None:
            raise InferenceError("the appearance inspector was not loaded")
        config, bundle = loaded.config, loaded.model.bundle
        pixels = frame.pixels
        silhouette = localize(pixels, config.inspector.localize)
        listed = config.inspector.checks
        geometry = (
            check_geometry(measure(silhouette), config.threshold.geometry)
            if CHECK_GEOMETRY in listed
            else None
        )
        appearance, heatmap, window_box = check_appearance(
            pixels, silhouette.centroid, silhouette.mask, loaded.model, config
        )
        checks = (appearance,) if geometry is None else (geometry, appearance)
        return Verdict(
            label=derive_label(checks, listed),
            checks=checks,
            heatmap=heatmap,
            window_box=window_box,
            unit_box=silhouette.unit_box,
            model_id=bundle.model_id,
            threshold_id=config.threshold.id,
            config_hash=self._config_hash,
            inspect_ms=(time.perf_counter() - start) * 1000.0,
        )
