"""The lane classifier's ImageNet weights load only from the local cache and only by hash."""

from pathlib import Path

import pytest

from qs.errors import ConfigError
from qs.lane import weights

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _put(torch_home: Path, data: bytes) -> Path:
    path = torch_home / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def test_pinned_value_is_a_full_hash_of_the_real_file() -> None:
    assert len(weights.RESNET18_SHA256) == 64
    assert weights.RESNET18_SHA256.startswith("f37072fd")
    assert set(weights.RESNET18_SHA256) <= set("0123456789abcdef")


def test_missing_file_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as err:
        weights.resnet18_weights(tmp_path)
    assert "resnet18-f37072fd.pth" in str(err.value)
    assert "tools/prefetch_weights.py" in str(err.value)


def test_wrong_bytes_are_refused_with_both_hashes(tmp_path: Path) -> None:
    _put(tmp_path, b"abc")
    with pytest.raises(ConfigError) as err:
        weights.resnet18_weights(tmp_path)
    assert "ba7816bf" in str(err.value)
    assert "f37072fd" in str(err.value)


def test_right_file_is_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _put(tmp_path, b"abc")
    monkeypatch.setattr(weights, "RESNET18_SHA256", ABC_SHA256)
    assert weights.resnet18_weights(tmp_path) == path
