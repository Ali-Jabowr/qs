"""The pin matches the real ImageNet file, and a wrong file is refused by the pinned hash."""

from pathlib import Path

import pytest

from qs.errors import ConfigError
from qs.lane import weights

ROOT = Path(__file__).resolve().parent.parent


def test_wrong_bytes_message_carries_the_pinned_hash(tmp_path: Path) -> None:
    path = tmp_path / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc")
    with pytest.raises(ConfigError) as err:
        weights.resnet18_weights(tmp_path)
    assert "has SHA-256 ba7816bf" in str(err.value)
    assert "not the pinned f37072fd" in str(err.value)


@pytest.mark.env
def test_pinned_file_in_the_repo_cache_loads_into_resnet18() -> None:
    import torch
    from torchvision.models import resnet18

    path = weights.resnet18_weights(ROOT / ".torch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet18()
    result = model.load_state_dict(state)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
