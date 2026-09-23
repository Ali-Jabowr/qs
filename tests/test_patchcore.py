"""The PatchCore seam without the model: pins, the weights check, the device, the import rules.

Nothing here imports torch, timm or Anomalib — the real module is exercised by
`tests/test_patchcore_env.py` under `make env-check`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from qs.errors import ConfigError
from qs.inspector.model import (
    BACKBONE,
    BACKBONE_SHA256,
    cached_weights,
    device_for,
    load_backbone,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "qs"
MANIFEST = ROOT / "reports" / "env" / "weights-manifest.json"


def _imports(path: Path) -> set[str]:
    """The top-level package of every import statement in one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_the_pinned_hash_is_the_manifest_s_racm_in1k_blob() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["caches"][".hf"]["hub_repo_ids"] == ["timm/wide_resnet50_2.racm_in1k"]
    blobs = [
        entry["sha256"]
        for entry in manifest["caches"][".hf"]["files"]
        if "/blobs/" in entry["path"] and entry["bytes"] == 275_835_296
    ]
    assert blobs == [BACKBONE_SHA256]
    assert BACKBONE_SHA256 == "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000"
    assert BACKBONE == "wide_resnet50_2.racm_in1k"


def test_anomalib_is_imported_in_exactly_one_module() -> None:
    importers = sorted(
        str(path.relative_to(SRC)) for path in SRC.rglob("*.py") if "anomalib" in _imports(path)
    )
    assert importers == ["inspector/patchcore.py"]


def test_no_module_imports_a_network_library() -> None:
    forbidden = {"socket", "http", "urllib", "requests"}
    offenders = {
        str(path.relative_to(SRC)): sorted(_imports(path) & forbidden)
        for path in SRC.rglob("*.py")
        if _imports(path) & forbidden
    }
    assert offenders == {}


def _hub(tmp_path: Path, content: bytes | None, revision: str = "30f73ace") -> Path:
    """A fake `.hf/hub` holding the backbone's snapshot, or only its ref when `content` is None."""
    repo = tmp_path / "hub" / "models--timm--wide_resnet50_2.racm_in1k"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    if content is not None:
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(content)
    return tmp_path / "hub"


def test_cached_weights_finds_the_snapshot_of_refs_main(tmp_path: Path) -> None:
    hub = _hub(tmp_path, b"weights")
    found = cached_weights(hub)
    assert found == hub / "models--timm--wide_resnet50_2.racm_in1k" / "snapshots" / (
        "30f73ace") / "model.safetensors"


def test_cached_weights_is_none_without_the_file_or_the_ref(tmp_path: Path) -> None:
    assert cached_weights(_hub(tmp_path / "a", None)) is None
    assert cached_weights(tmp_path / "empty" / "hub") is None


class _Spy:
    """A model builder that records whether it was ever called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return "model"


def test_load_backbone_refuses_a_missing_weights_file_before_building(tmp_path: Path) -> None:
    spy = _Spy()
    with pytest.raises(ConfigError, match=r"^weights: .*not in the local cache"):
        load_backbone(lambda: None, spy)
    with pytest.raises(ConfigError, match=r"^weights: .*not in the local cache"):
        load_backbone(lambda: tmp_path / "missing.safetensors", spy)
    assert spy.calls == 0


def test_load_backbone_refuses_another_hash_before_building(tmp_path: Path) -> None:
    other = tmp_path / "model.safetensors"
    other.write_bytes(b"not the pinned weights")
    spy = _Spy()
    with pytest.raises(ConfigError, match=r"^weights: .*" + hashlib.sha256(
            b"not the pinned weights").hexdigest()[:12]):
        load_backbone(lambda: other, spy)
    assert spy.calls == 0


def test_load_backbone_builds_once_the_hash_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = tmp_path / "model.safetensors"
    good.write_bytes(b"pinned")
    monkeypatch.setattr(
        "qs.inspector.model.BACKBONE_SHA256", hashlib.sha256(b"pinned").hexdigest()
    )
    spy = _Spy()
    assert load_backbone(lambda: good, spy) == "model"
    assert spy.calls == 1


def test_device_for_each_backend() -> None:
    assert device_for("torch-cuda", cuda_available=True) == "cuda"
    assert device_for("torch-cpu", cuda_available=True) == "cpu"
    assert device_for("torch-cpu", cuda_available=False) == "cpu"


def test_torch_cuda_without_cuda_never_falls_back() -> None:
    with pytest.raises(ConfigError, match=re.escape("no CPU fallback")):
        device_for("torch-cuda", cuda_available=False)
