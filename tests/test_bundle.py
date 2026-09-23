"""The model bundle: `write_bundle` and `load_bundle` with a small random bank (docs/15 §8)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from qs.config import _MODEL_ID
from qs.errors import ConfigError
from qs.inspector.bundle import BundleInfo, ImageRecord, load_bundle, write_bundle

PINNED = "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000"
KEYS = {
    "backbone", "backbone_sha256", "layers", "input_px", "window_px", "normalisation",
    "coreset_ratio", "num_neighbors", "seed", "device", "versions", "data_source",
    "validation_only", "units", "images",
}


def _bank(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((40, 16), dtype=np.float32)


def _info(data_source: str = "own") -> BundleInfo:
    images = tuple(
        ImageRecord(f"u{u:03d}_good_G2_r{r}.png", f"u{u:03d}", f"{u:02d}{r}".ljust(64, "a"))
        for u in (3, 1, 2)
        for r in (0, 1, 2)
    )
    return BundleInfo(
        input_px=256, window_px=400, coreset_ratio=0.1, seed=42, device="cuda",
        data_source=data_source, images=images,
    )


def _suffix(directory: Path) -> str:
    joined = (directory / "memory_bank.pt").read_bytes() + (directory / "bundle.yml").read_bytes()
    return hashlib.sha256(joined).hexdigest()[:12]


def _rehashed(models: Path, model_id: str, edit: dict[str, object] | None = None,
              drop: str | None = None) -> str:
    """A copy of the bundle with `bundle.yml` edited, under a model_id whose hash matches it —
    the bundle a hand edit would leave, if the editor also fixed the name."""
    source = models / model_id
    data = yaml.safe_load((source / "bundle.yml").read_text(encoding="utf-8"))
    data.update(edit or {})
    if drop is not None:
        del data[drop]
    text = yaml.safe_dump(data, sort_keys=False).encode("utf-8")
    bank = (source / "memory_bank.pt").read_bytes()
    prefix = model_id.rsplit("-", 1)[0].replace("-v1", "-v9")
    new_id = f"{prefix}-{hashlib.sha256(bank + text).hexdigest()[:12]}"
    (models / new_id).mkdir()
    (models / new_id / "memory_bank.pt").write_bytes(bank)
    (models / new_id / "bundle.yml").write_bytes(text)
    return new_id


def _refused(models: Path, model_id: str, match: str, *, validation: bool = False) -> None:
    with pytest.raises(ConfigError, match=match) as caught:
        load_bundle(models, model_id, validation=validation)
    assert "\n" not in str(caught.value)


@pytest.fixture
def written(tmp_path: Path) -> tuple[Path, str]:
    models = tmp_path / "models"
    return models, write_bundle(models, "mochi", _bank(), _info())


def test_write_bundle_makes_a_hashed_model_id_and_exactly_two_files(
    written: tuple[Path, str],
) -> None:
    models, model_id = written
    assert _MODEL_ID.fullmatch(model_id)
    assert re.fullmatch(r"patchcore-mochi-v1-[0-9a-f]{12}", model_id)
    directory = models / model_id
    assert sorted(p.name for p in directory.iterdir()) == ["bundle.yml", "memory_bank.pt"]
    assert model_id.endswith(_suffix(directory))
    assert [p.name for p in models.iterdir()] == [model_id]  # no temporary left behind


def test_bundle_yml_has_exactly_the_spec_keys_and_no_time(written: tuple[Path, str]) -> None:
    models, model_id = written
    text = (models / model_id / "bundle.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert set(data) == KEYS
    assert data["backbone"] == "wide_resnet50_2.racm_in1k"
    assert data["backbone_sha256"] == PINNED
    assert data["layers"] == ["layer2", "layer3"]
    assert data["normalisation"] == {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    assert (data["input_px"], data["window_px"], data["coreset_ratio"]) == (256, 400, 0.1)
    assert (data["num_neighbors"], data["seed"], data["device"]) == (9, 42, "cuda")
    assert set(data["versions"]) == {
        "python", "torch", "torchvision", "anomalib", "timm", "numpy", "opencv"
    }
    assert data["versions"]["anomalib"] == "2.6.2"
    assert data["data_source"] == "own"
    assert data["validation_only"] is False
    assert data["units"] == ["u001", "u002", "u003"]
    assert len(data["images"]) == 9
    assert set(data["images"][0]) == {"file", "unit", "sha256"}
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}", text)


def test_the_next_bundle_of_a_product_is_v2_and_the_first_is_untouched(
    written: tuple[Path, str],
) -> None:
    models, first = written
    before = {p.name: p.read_bytes() for p in (models / first).iterdir()}
    second = write_bundle(models, "mochi", _bank(1), _info())
    other = write_bundle(models, "mochi-lean", _bank(2), _info())
    assert re.fullmatch(r"patchcore-mochi-v2-[0-9a-f]{12}", second)
    assert re.fullmatch(r"patchcore-mochi-lean-v1-[0-9a-f]{12}", other)
    assert {p.name: p.read_bytes() for p in (models / first).iterdir()} == before


def test_the_same_bank_and_info_write_the_same_bytes(tmp_path: Path) -> None:
    left = write_bundle(tmp_path / "a", "mochi", _bank(), _info())
    right = write_bundle(tmp_path / "b", "mochi", _bank(), _info())
    assert left == right


def test_write_bundle_refuses_a_bad_product(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="product"):
        write_bundle(tmp_path, "Mochi_1", _bank(), _info())


def test_load_bundle_round_trips(written: tuple[Path, str]) -> None:
    models, model_id = written
    bundle = load_bundle(models, model_id)
    assert bundle.model_id == model_id
    assert np.array_equal(bundle.bank, _bank())
    assert bundle.bank.dtype == np.float32
    assert (bundle.window_px, bundle.input_px, bundle.seed, bundle.device) == (400, 256, 42, "cuda")
    assert bundle.units == ("u001", "u002", "u003")
    assert bundle.images[0] == _info().images[0]
    assert bundle.data_source == "own"
    assert bundle.validation_only is False


def test_load_bundle_refuses_a_missing_directory_or_file(written: tuple[Path, str]) -> None:
    models, model_id = written
    missing = model_id[:-12] + "0" * 12
    _refused(models, missing, re.escape(f"{missing}: no such bundle directory"))
    (models / model_id / "memory_bank.pt").unlink()
    _refused(models, model_id, "memory_bank.pt is missing")


def test_load_bundle_refuses_a_name_that_is_not_a_model_id(written: tuple[Path, str]) -> None:
    models, _ = written
    _refused(models, "../elsewhere", "not a full model_id")


@pytest.mark.parametrize("name", ["memory_bank.pt", "bundle.yml"])
def test_one_changed_byte_in_either_file_is_a_hash_mismatch(
    written: tuple[Path, str], name: str
) -> None:
    models, model_id = written
    path = models / model_id / name
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    _refused(models, model_id, "does not match its model_id")


def test_a_missing_or_unknown_key_is_refused(written: tuple[Path, str]) -> None:
    models, model_id = written
    _refused(models, _rehashed(models, model_id, drop="seed"), "missing key 'seed'")
    _refused(models, _rehashed(models, model_id, {"trained_at": "x"}), "unknown key 'trained_at'")


def test_another_backbone_or_hash_is_refused(written: tuple[Path, str]) -> None:
    models, model_id = written
    tag = _rehashed(models, model_id, {"backbone": "wide_resnet50_2.tv_in1k"})
    _refused(models, tag, "backbone: 'wide_resnet50_2.tv_in1k' is not the pinned")
    digest = _rehashed(models, model_id, {"backbone_sha256": "0" * 64})
    _refused(models, digest, "backbone_sha256: .* is not the pinned")


def test_validation_only_inconsistent_with_data_source_is_refused(
    written: tuple[Path, str],
) -> None:
    models, model_id = written
    own = _rehashed(models, model_id, {"validation_only": True})
    _refused(models, own, "validation_only: true does not match data_source 'own'", validation=True)
    external = _rehashed(models, model_id, {"data_source": "external:mvtec-ad"})
    _refused(models, external, "validation_only: false does not match data_source")


def test_a_validation_only_bundle_needs_validation_mode(tmp_path: Path) -> None:
    models = tmp_path / "models"
    model_id = write_bundle(models, "mvtec-bottle", _bank(), _info("external:mvtec-ad"))
    data = yaml.safe_load((models / model_id / "bundle.yml").read_text(encoding="utf-8"))
    assert data["validation_only"] is True
    _refused(models, model_id, "validation-only .*ADR-0016")
    bundle = load_bundle(models, model_id, validation=True)
    assert bundle.validation_only is True
    assert bundle.data_source == "external:mvtec-ad"
