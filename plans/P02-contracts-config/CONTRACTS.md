# CONTRACTS — P02

## 1. `src/qs/contracts.py`

### 1.1 Values

Exactly the eight records of `docs/11-interfaces.md` §2 — `Label`, `TriggerEvent`, `CameraReadback`,
`Frame`, `CheckResult`, `Verdict`, `Health`, `InspectionRecord` — with the fields, types and order
given there. All dataclasses `frozen=True, slots=True`. `Label` is a `StrEnum`.

`Verdict.__post_init__` raises `ValueError` when `label is Label.NO_VERDICT` and `reason is None`,
and when `label is not Label.NO_VERDICT` and `checks` is empty.

### 1.2 Protocols

`TriggerSource`, `FrameSource`, `Inspector`, `Annunciator`, `Sink` — the method signatures of
`docs/11-interfaces.md` §3, with their docstrings carried over. `StationConfig` is imported from
`qs.config` under `TYPE_CHECKING` only, so `contracts` does not import the loader at runtime.

### 1.3 Constants

```python
CHECK_GEOMETRY = "geometry"
CHECK_APPEARANCE = "appearance"
REASONS = frozenset({"capture", "camera", "camera_settings", "lighting",
                     "placement", "inference", "busy"})
```

### 1.4 `derive_label`

```python
def derive_label(checks: tuple[CheckResult, ...], configured: frozenset[str]) -> Label: ...
```

- Raises `ValueError` if `configured` is empty, if two checks share a name, or if a check's name is
  not in `configured`.
- Returns `DEFECT` if any check has `passed=False`.
- Returns `PASS` only if the set of check names equals `configured` and every one passed.
- Otherwise raises `ValueError` — a completed inspection with a configured check missing is a
  programming error, not a verdict. (NO_VERDICT is decided by the Pipeline from an exception, never
  here.)

## 2. `src/qs/errors.py`

```
QSError(Exception)
├── ConfigError
├── CaptureError          reason "capture"
│   └── CameraLost        reason "camera"
├── CameraSettingsDrift   reason "camera_settings"
├── LightingFault         reason "lighting"
├── PlacementFault        reason "placement"
└── InferenceError        reason "inference"
```

Each class except `QSError` and `ConfigError` has a class attribute `reason: str` equal to one of
`REASONS`. `PlacementFault` takes a `detail: str` — `"no unit"`, `"more than one unit"`,
`"not wholly in view"` (`15-model.md` §2).

## 3. `src/qs/config.py`

### 3.1 Shape

One frozen dataclass per YAML mapping in `config/station-A-bench.example.yml`, nested the same way,
field names identical to the keys. Lists of four ints (ROIs) become `tuple[int, int, int, int]`;
`[min, max]` pairs become `tuple[int, int]`. The root is `StationConfig`.

Allowed enumerations:

| Key | Values |
|---|---|
| `trigger.type` | `keyboard`, `replay` |
| `camera.type` | `webcam`, `phone`, `replay` |
| `camera.format` | `YUYV`, `MJPG` |
| `lighting.geometry` | `G1`, `G2`, `G3` |
| `inspector.type` | `appearance`, `stub` |
| `inspector.backend` | `torch-cuda`, `torch-cpu` |

### 3.2 Loading

```python
@dataclass(frozen=True, slots=True)
class LoadedConfig:
    config: StationConfig
    config_hash: str        # 64 hex characters
    path: Path

def load_config(path: Path) -> LoadedConfig: ...
def canonical_hash(mapping: Mapping[str, Any]) -> str: ...
```

`canonical_hash` = SHA-256 hex of
`json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`.
YAML dates are converted to ISO strings before hashing. `load_config` uses `yaml.safe_load` only.

### 3.3 Validation — each a `ConfigError` naming the key

1. Unknown key, missing key, wrong type, value outside its enumeration.
2. `camera.power_line_hz == 50` and `camera.exposure_us % 10000 != 0`.
3. `lighting.reference_patch.roi` overlaps `inspector.localize.roi`.
4. Any ROI outside `camera.width` × `camera.height`.
5. `inspector.model` does not match `^patchcore-[a-z0-9]+(-[a-z0-9]+)*-v\d+-[0-9a-f]{12}$`.
6. `threshold.set_on.model != inspector.model`, or `threshold.set_on.backend != inspector.backend`.
7. `threshold.geometry.area_px` min not below max; `axis_ratio_min` or `solidity_min` outside (0, 1].
8. `lighting.reference_patch.band` not in (0, 1); `expect` not in (0, 1).
9. `display.labels` missing any of `pass`, `defect`, `no_verdict`, or any empty.

## 4. Tests

`tests/test_contracts.py`

- Every value record is immutable: assigning an attribute raises.
- `Verdict` refuses NO_VERDICT without a reason, and PASS or DEFECT with no checks.
- `derive_label`: both pass → PASS; geometry fails → DEFECT; appearance fails → DEFECT; both fail →
  DEFECT; one configured check missing → `ValueError`; an unconfigured check name → `ValueError`;
  duplicate names → `ValueError`; empty `configured` → `ValueError`.
- Every exception's `reason` is in `REASONS`, and the two sets are equal apart from `busy`.

`tests/test_config.py`

- The example loads, and every value round-trips to the right type.
- Its hash is stable: the same mapping with keys reordered, comments added and indentation changed
  gives the same hash; changing one value changes it.
- One test per validation rule in §3.3, each asserting the key named in the message.
- A missing key and an unknown key are both refused — no defaults.
