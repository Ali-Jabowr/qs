"""`StubInspector`: scripted checks and no model, for replay and the tests (docs/11 §4).

The script maps an event's `seq` to one outcome; an unscripted `seq` passes both checks.

- `pass` — geometry and appearance both pass.
- `defect:geometry` / `defect:appearance` — that check fails; `defect:appearance` carries a
  small float32 heatmap in score units.
- `placement:<detail>` — raises `PlacementFault(detail)`.
- `inference` — raises `InferenceError`. `crash` — raises `RuntimeError`.
- `nonfinite` — returns a PASS whose appearance score is NaN (the way a NaN fails open).
- `mislabel` — returns a PASS with a failed appearance check.

The last two exist to be refused by the Pipeline's guard (C8). The stub is deterministic given
the frame, which carries its event; it holds no per-frame state and does no I/O (C1, C2), and
never reads the clock: its `inspect_ms` is 0. Its `model_id` is `"stub"`, so no record of a stub
can be mistaken for a model's; `qs station` refuses it without `--replay`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from qs.config import StationConfig
from qs.contracts import CHECK_APPEARANCE, CHECK_GEOMETRY, CheckResult, Frame, Label, Verdict
from qs.errors import ConfigError, InferenceError, PlacementFault

STUB_MODEL_ID = "stub"
_HEATMAP_PX = 32
_SIMPLE = frozenset({"pass", "inference", "crash", "nonfinite", "mislabel"})
_DEFECTS = frozenset({CHECK_GEOMETRY, CHECK_APPEARANCE})


def check_outcome(outcome: str) -> str:
    """`outcome` if it is in the script vocabulary, else `ConfigError` naming it."""
    if outcome in _SIMPLE:
        return outcome
    kind, sep, detail = outcome.partition(":")
    if sep and kind == "defect" and detail in _DEFECTS:
        return outcome
    if sep and kind == "placement" and detail.strip():
        return outcome
    raise ConfigError(
        f"stub script: {outcome!r} is not an outcome — pass, defect:geometry, "
        "defect:appearance, placement:<detail>, inference, crash, nonfinite or mislabel"
    )


class StubInspector:
    """An `Inspector` that returns what its script says (C6: built from the config hash and
    configured by `load(config)`)."""

    def __init__(self, config_hash: str, script: Mapping[int, str] | None = None) -> None:
        self._config_hash = config_hash
        self._script = {int(seq): check_outcome(o) for seq, o in (script or {}).items()}
        self._config: StationConfig | None = None

    @property
    def model_id(self) -> str:
        return STUB_MODEL_ID

    def load(self, config: StationConfig) -> None:
        self._config = config

    def inspect(self, frame: Frame) -> Verdict:
        config = self._config
        if config is None:
            raise InferenceError("the stub inspector was not loaded")
        outcome = self._script.get(frame.event.seq, "pass")
        if outcome == "inference":
            raise InferenceError(f"scripted inference failure at seq {frame.event.seq}")
        if outcome == "crash":
            raise RuntimeError(f"scripted crash at seq {frame.event.seq}")
        if outcome.startswith("placement:"):
            raise PlacementFault(outcome.partition(":")[2])
        threshold = config.threshold.value
        geometry_ok = outcome != "defect:geometry"
        appearance_ok = outcome not in ("defect:appearance", "mislabel")
        score = threshold * (2.0 if not appearance_ok else 0.5)
        if outcome == "nonfinite":
            score = float("nan")
        geometry = CheckResult(
            name=CHECK_GEOMETRY,
            passed=geometry_ok,
            value=None,
            limit=None,
            measures=(("area_px", 100000.0 if geometry_ok else 50000.0), ("axis_ratio", 0.98),
                      ("solidity", 0.98)),
            failed_on=None if geometry_ok else "area_px<min",
        )
        appearance = CheckResult(
            name=CHECK_APPEARANCE,
            passed=appearance_ok,
            value=score,
            limit=threshold,
            measures=(),
            failed_on=None if appearance_ok else "score>threshold",
        )
        height, width = frame.pixels.shape[:2]
        side = min(_HEATMAP_PX, height, width)
        box = ((width - side) // 2, (height - side) // 2, side, side)
        heatmap = None
        if outcome == "defect:appearance":
            ramp = np.linspace(0.0, 2.0 * threshold, side * side, dtype=np.float32)
            heatmap = ramp.reshape(side, side)
        if outcome in ("nonfinite", "mislabel"):
            label = Label.PASS  # what the guard must refuse
        else:
            label = Label.PASS if geometry_ok and appearance_ok else Label.DEFECT
        return Verdict(
            label=label,
            checks=(geometry, appearance),
            heatmap=heatmap,
            window_box=box,
            unit_box=box,
            model_id=STUB_MODEL_ID,
            threshold_id=config.threshold.id,
            config_hash=self._config_hash,
            inspect_ms=0.0,
        )
