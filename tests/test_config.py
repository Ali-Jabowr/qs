import re
from datetime import date
from pathlib import Path

import pytest

from qs.config import (
    CameraConfig,
    DisplayConfig,
    EvidenceConfig,
    GeometryConfig,
    InspectorConfig,
    LabelsConfig,
    LatencyConfig,
    LightingConfig,
    LocalizeConfig,
    ReferencePatchConfig,
    SetOnConfig,
    StationConfig,
    ThresholdConfig,
    TriggerConfig,
    canonical_hash,
    load_config,
)
from qs.errors import ConfigError

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"

EXPECTED = StationConfig(
    version=1,
    station_id="A-bench",
    product="mochi-<flavour>-v1",
    trigger=TriggerConfig(type="keyboard", key="space"),
    camera=CameraConfig(
        type="webcam",
        device="/dev/video0",
        width=1920,
        height=1080,
        format="YUYV",
        power_line_hz=50,
        exposure_us=10000,
        gain=0,
        wb_kelvin=4500,
        focus=30,
        capture_timeout_ms=500,
    ),
    lighting=LightingConfig(
        geometry="G2",
        reference_patch=ReferencePatchConfig(
            roi=(1780, 40, 100, 100), expect=0.62, band=0.05
        ),
    ),
    inspector=InspectorConfig(
        type="appearance",
        localize=LocalizeConfig(
            roi=(360, 0, 1200, 1080),
            threshold=70,
            min_area_px=20000,
            border_margin_px=10,
        ),
        mask_dilate_px=12,
        model="patchcore-mochi-v1-3f9a1c2b7d4e",
        backend="torch-cuda",
    ),
    threshold=ThresholdConfig(
        id="cal-2026-09-25-far5-a41c07",
        value=1.83,
        policy="far_5pct_on_calib",
        set_on=SetOnConfig(
            model="patchcore-mochi-v1-3f9a1c2b7d4e",
            backend="torch-cuda",
            calib_units=15,
            localize_threshold=70,
            mask_dilate_px=12,
        ),
        geometry=GeometryConfig(
            area_px=(98000, 131000), axis_ratio_min=0.90, solidity_min=0.96
        ),
    ),
    display=DisplayConfig(
        labels=LabelsConfig(pass_="ГОДЕН", defect="БРАК", no_verdict="НЕТ РЕШЕНИЯ"),
        font="assets/DejaVuSans.ttf",
    ),
    evidence=EvidenceConfig(root="evidence/", retention_days=400),
    latency=LatencyConfig(budget_ms=1000, breach_action="record_and_continue"),
)

# The same mapping as the example, with keys reordered, comments added and
# indentation changed — it must hash the same.
REORDERED = """\
# reordered, re-indented, commented — the same mapping
latency:
    budget_ms: 1000
    breach_action: record_and_continue
version: 1
station_id: A-bench
product: mochi-<flavour>-v1
evidence:
    # where the archive lives
    root: evidence/
    retention_days: 400
display:
    font: assets/DejaVuSans.ttf
    labels:
        no_verdict: НЕТ РЕШЕНИЯ
        pass: ГОДЕН
        defect: БРАК
threshold:
    geometry:
        solidity_min: 0.96
        axis_ratio_min: 0.90
        area_px: [98000, 131000]
    set_on: { mask_dilate_px: 12, localize_threshold: 70, calib_units: 15,
              backend: torch-cuda, model: patchcore-mochi-v1-3f9a1c2b7d4e }
    policy: far_5pct_on_calib
    value: 1.83
    id: cal-2026-09-25-far5-a41c07
inspector:
    backend: torch-cuda
    model: patchcore-mochi-v1-3f9a1c2b7d4e
    mask_dilate_px: 12
    localize:
        border_margin_px: 10
        min_area_px: 20000
        threshold: 70
        roi: [360, 0, 1200, 1080]
    type: appearance
lighting:
    reference_patch:
        band: 0.05
        expect: 0.62
        roi: [1780, 40, 100, 100]
    geometry: G2
camera:
    capture_timeout_ms: 500
    focus: 30
    wb_kelvin: 4500
    gain: 0
    exposure_us: 10000
    power_line_hz: 50
    format: YUYV
    height: 1080
    width: 1920
    device: /dev/video0
    type: webcam
trigger:
    key: space
    type: keyboard
"""


def _written(tmp_path: Path, text: str, name: str = "config.yml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _mutated(tmp_path: Path, old: str, new: str) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    assert old in text, f"the example no longer contains {old!r}"
    return _written(tmp_path, text.replace(old, new))


def _refused(path: Path, key: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(key)):
        load_config(path)


def test_the_example_loads_and_round_trips(tmp_path: Path) -> None:
    loaded = load_config(EXAMPLE)
    assert loaded.config == EXPECTED
    assert loaded.path == EXAMPLE
    assert re.fullmatch(r"[0-9a-f]{64}", loaded.config_hash)
    config = loaded.config
    assert type(config.version) is int
    assert type(config.camera.power_line_hz) is int
    assert type(config.lighting.reference_patch.expect) is float
    assert type(config.threshold.value) is float
    assert type(config.lighting.reference_patch.roi) is tuple
    assert type(config.threshold.geometry.area_px) is tuple
    assert type(config.display.labels.pass_) is str
    assert type(config.threshold.set_on.calib_units) is int


def test_hash_is_stable_across_key_order_comments_and_indentation(
    tmp_path: Path,
) -> None:
    assert load_config(_written(tmp_path, REORDERED)).config_hash == load_config(
        EXAMPLE
    ).config_hash


def test_changing_one_value_changes_the_hash(tmp_path: Path) -> None:
    changed = _mutated(tmp_path, "expect: 0.62", "expect: 0.63")
    assert load_config(changed).config_hash != load_config(EXAMPLE).config_hash


def test_canonical_hash_is_order_insensitive() -> None:
    left = {"version": 1, "latency": {"budget_ms": 1000, "breach_action": "record"}}
    right = {"latency": {"breach_action": "record", "budget_ms": 1000}, "version": 1}
    assert canonical_hash(left) == canonical_hash(right)


def test_canonical_hash_converts_dates_to_iso_strings() -> None:
    assert canonical_hash({"d": date(2026, 9, 25)}) == canonical_hash(
        {"d": "2026-09-25"}
    )


def test_canonical_hash_is_64_hex_characters() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", canonical_hash({"a": 1}))


def test_rule_1_unknown_top_level_key(tmp_path: Path) -> None:
    _refused(_mutated(tmp_path, "version: 1", "version: 1\nextra_top: true"), "extra_top")


def test_rule_1_unknown_nested_key(tmp_path: Path) -> None:
    _refused(
        _mutated(tmp_path, "  focus: 30", "  focus: 30\n  zoom: 5"), "camera.zoom"
    )


def test_rule_1_missing_top_level_key(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path, "evidence:  { root: evidence/, retention_days: 400 }\n", ""
        ),
        "evidence",
    )


def test_rule_1_missing_nested_key(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path,
            "trigger:   { type: keyboard, key: space }",
            "trigger:   { type: keyboard }",
        ),
        "trigger.key",
    )


def test_rule_1_wrong_scalar_type(tmp_path: Path) -> None:
    _refused(_mutated(tmp_path, "width: 1920", "width: '1920'"), "camera.width")


def test_rule_1_wrong_roi_type(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path, "roi:    [1780, 40, 100, 100]", "roi:    [1780, 40, 100.5, 100]"
        ),
        "lighting.reference_patch.roi",
    )


def test_rule_1_value_outside_its_enumeration(tmp_path: Path) -> None:
    _refused(_mutated(tmp_path, "format: YUYV", "format: H264"), "camera.format")


def test_rule_2_exposure_not_a_multiple_of_10ms_at_50hz(tmp_path: Path) -> None:
    _refused(
        _mutated(tmp_path, "exposure_us: 10000", "exposure_us: 12345"),
        "camera.exposure_us",
    )


def test_rule_2_exposure_a_multiple_of_10ms_loads(tmp_path: Path) -> None:
    loaded = load_config(_mutated(tmp_path, "exposure_us: 10000", "exposure_us: 20000"))
    assert loaded.config.camera.exposure_us == 20000


def test_rule_3_reference_patch_overlapping_the_localize_roi(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path, "roi:    [1780, 40, 100, 100]", "roi:    [1000, 40, 100, 100]"
        ),
        "lighting.reference_patch.roi",
    )


def test_rule_4_roi_outside_the_frame(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path, "roi:    [1780, 40, 100, 100]", "roi:    [1880, 40, 100, 100]"
        ),
        "lighting.reference_patch.roi",
    )


def test_rule_5_model_not_a_full_model_id(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path,
            "model:     patchcore-mochi-v1-3f9a1c2b7d4e",
            "model:     patchcore-Mochi-v1-3f9a1c2b7d4e",
        ),
        "inspector.model",
    )


def test_rule_6_threshold_not_set_on_the_model(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path,
            "set_on: { model: patchcore-mochi-v1-3f9a1c2b7d4e,",
            "set_on: { model: patchcore-mochi-v1-ffffffffffff,",
        ),
        "threshold.set_on.model",
    )


def test_rule_6_threshold_not_set_on_the_backend(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path,
            "backend: torch-cuda, calib_units: 15",
            "backend: torch-cpu, calib_units: 15",
        ),
        "threshold.set_on.backend",
    )


def test_rule_7_area_px_min_not_below_max(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path,
            "area_px:        [98000, 131000]",
            "area_px:        [131000, 98000]",
        ),
        "threshold.geometry.area_px",
    )


def test_rule_7_axis_ratio_min_outside_unit_interval(tmp_path: Path) -> None:
    _refused(
        _mutated(tmp_path, "axis_ratio_min: 0.90", "axis_ratio_min: 1.5"),
        "threshold.geometry.axis_ratio_min",
    )


def test_rule_7_solidity_min_outside_unit_interval(tmp_path: Path) -> None:
    _refused(
        _mutated(tmp_path, "solidity_min:   0.96", "solidity_min:   0"),
        "threshold.geometry.solidity_min",
    )


def test_rule_8_band_outside_the_open_interval(tmp_path: Path) -> None:
    _refused(_mutated(tmp_path, "band:   0.05", "band:   0"), "lighting.reference_patch.band")


def test_rule_8_expect_outside_the_open_interval(tmp_path: Path) -> None:
    _refused(
        _mutated(tmp_path, "expect: 0.62", "expect: 1"), "lighting.reference_patch.expect"
    )


def test_rule_9_labels_key_missing(tmp_path: Path) -> None:
    _refused(
        _mutated(
            tmp_path,
            "labels: { pass: ГОДЕН, defect: БРАК, no_verdict: НЕТ РЕШЕНИЯ }",
            "labels: { pass: ГОДЕН, defect: БРАК }",
        ),
        "display.labels.no_verdict",
    )


def test_rule_9_label_text_empty(tmp_path: Path) -> None:
    _refused(
        _mutated(tmp_path, "defect: БРАК", "defect: ''"), "display.labels.defect"
    )


def test_no_defaults_missing_and_unknown_keys_are_both_refused(
    tmp_path: Path,
) -> None:
    without_latency = _mutated(
        tmp_path, "latency:   { budget_ms: 1000, breach_action: record_and_continue }\n", ""
    )
    _refused(without_latency, "latency")
    with_extra = _mutated(
        tmp_path, "station_id: A-bench", "station_id: A-bench\noperator: ali"
    )
    _refused(with_extra, "operator")


def test_unreadable_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no-such-file"):
        load_config(tmp_path / "no-such-file.yml")
