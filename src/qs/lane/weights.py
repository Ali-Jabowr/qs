"""Pinned ImageNet weights for the lane classifier (ResNet18).

The file is fetched once by `tools/prefetch_weights.py` and from then on only
read from the repo's own torch cache, after its SHA-256 matches the pin
(ADR-0016, ADR-0020).
"""

from pathlib import Path

from qs.errors import ConfigError
from qs.inspector.model import sha256_of

RESNET18_FILE = "resnet18-f37072fd.pth"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def resnet18_weights(torch_home: Path) -> Path:
    """The cached ResNet18 weights file under `torch_home`, refused unless its hash is the pin."""
    path = torch_home / "hub" / "checkpoints" / RESNET18_FILE
    if not path.is_file():
        raise ConfigError(
            f"weights: {RESNET18_FILE} is not in {path.parent} — run "
            "tools/prefetch_weights.py once, online (ADR-0016)"
        )
    digest = sha256_of(path)
    if digest != RESNET18_SHA256:
        raise ConfigError(
            f"weights: {path} has SHA-256 {digest[:8]}…, not the pinned "
            f"{RESNET18_SHA256[:8]}… of {RESNET18_FILE}"
        )
    return path
