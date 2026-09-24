"""PatchCore from Anomalib 2.6.2, at the torch-model level — the one module that imports it.

ADR-0015: only `PatchcoreModel` is used, fed tensors our own `window.to_input` produced;
Anomalib's Lightning wrapper, `Engine`, datamodules, pre- and post-processors and exporter
are never imported. ADR-0020: the backbone is named by its full timm tag and its cached
weights file is hash-checked before the model is built. ADR-0016: before torch, timm or
Anomalib is imported, this module points the caches at the repo's `.hf/` and `.torch/`
and sets `HF_HUB_OFFLINE=1`, so nothing is downloaded at runtime. docs/15 §9: on CUDA
the determinism flags are set before the model loads, and `CUBLAS_WORKSPACE_CONFIG` before
CUDA initialises; the coreset is seeded immediately before subsampling.
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[3]
HF_HOME = ROOT / ".hf"
HUB_DIR = HF_HOME / "hub"  # where the weights are hash-checked, and so where they must load from
TORCH_HOME = ROOT / ".torch"

# huggingface_hub takes its cache from HF_HUB_CACHE, then HUGGINGFACE_HUB_CACHE, and only then
# HF_HOME/hub: all three are pinned, or a shell variable would load weights nobody checked.
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HUB_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUB_DIR)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import huggingface_hub.constants as hf_constants  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

with warnings.catch_warnings():
    # Anomalib's package import pulls in modules we never use, which warn about their own
    # deprecations; our one-line errors should not be buried under them.
    warnings.simplefilter("ignore", FutureWarning)
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

from qs.errors import ConfigError  # noqa: E402
from qs.inspector.model import (  # noqa: E402
    BACKBONE,
    LAYERS,
    NUM_NEIGHBORS,
    FitSettings,
    Fitted,
    Scored,
    cached_weights,
    device_for,
    load_backbone,
)

BATCH = 8  # docs/15 §5: one pass in batches of 8 windows


def cuda_available() -> bool:
    """Whether torch sees a CUDA device — asked only after the environment above is set."""
    return bool(torch.cuda.is_available())


def hub_dir() -> Path:
    """The cache huggingface_hub will load the backbone from — `HUB_DIR`, the one the hash is
    checked in, and offline — or `ConfigError`. The pins above hold only if huggingface_hub
    was first imported after them; this reads what it actually took (ADR-0016, ADR-0020)."""
    cache = Path(hf_constants.HF_HUB_CACHE).resolve()
    if cache != HUB_DIR.resolve():
        raise ConfigError(
            f"weights: huggingface_hub reads its cache from {cache}, not {HUB_DIR}, where the "
            "weights are hash-checked — unset HF_HUB_CACHE and HUGGINGFACE_HUB_CACHE (ADR-0020)"
        )
    if not hf_constants.is_offline_mode():
        raise ConfigError("weights: huggingface_hub is not offline — HF_HUB_OFFLINE (NFR-07)")
    return HUB_DIR


def _seed(seed: int) -> None:
    """Seed every generator the coreset draws on: torch's, CUDA's and NumPy's global one."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AnomalibPatchCore:
    """The real model behind the seam: fit windows into a bank, score one window.

    `device` is `"cuda"` or `"cpu"` (`model.device_for`). `resolve` finds the cached
    weights file; the default is the repo's `.hf/hub`. The weights are checked before
    `PatchcoreModel` is built, and the model is built exactly as docs/15 §5 names it.
    """

    def __init__(self, device: str, *, resolve: Callable[[], Path | None] | None = None) -> None:
        backend = {"cuda": "torch-cuda", "cpu": "torch-cpu"}.get(device, device)
        self._device = torch.device(device_for(backend, cuda_available=cuda_available()))
        if self._device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        hub = hub_dir()  # before anything is read: the file checked is the file loaded
        resolver = resolve or (lambda: cached_weights(hub))
        self._model = load_backbone(
            resolver,
            lambda: PatchcoreModel(
                layers=list(LAYERS),
                backbone=BACKBONE,
                pre_trained=True,
                num_neighbors=NUM_NEIGHBORS,
            ),
        ).to(self._device)
        self._model.eval()
        self._marks: list[float] = []
        extractor = self._model.feature_extractor
        extractor.register_forward_pre_hook(lambda *_: self._mark())
        extractor.register_forward_hook(lambda *_: self._mark())

    @property
    def backbone_tag(self) -> str:
        """The timm name the backbone was built from."""
        return str(self._model.backbone)

    @property
    def pretrained_tag(self) -> str | None:
        """The pretrained tag timm resolved for the backbone's weights."""
        cfg = getattr(self._model.feature_extractor.feature_extractor, "pretrained_cfg", None)
        return None if cfg is None else cfg.get("tag")

    def _sync(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _mark(self) -> None:
        self._sync()
        self._marks.append(time.perf_counter())

    def fit(self, windows: Sequence[np.ndarray], settings: FitSettings) -> Fitted:
        """One pass over the windows in batches of 8, then the seeded coreset (docs/15 §5).

        The per-batch embeddings are kept on the CPU and only the stacked store is moved
        to the device, so the 4 GB card holds one copy of it at a time.
        """
        if settings.device != self._device.type:
            raise ValueError(f"fit on {settings.device!r}, but the model is on {self._device}")
        if not windows:
            raise ValueError("no windows to fit")
        model = self._model
        cuda = self._device.type == "cuda"
        model.memory_bank = torch.empty(0, device=self._device)
        model.embedding_store.clear()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self._device)
        chunks: list[torch.Tensor] = []
        model.train()
        try:
            with torch.no_grad():
                for start in range(0, len(windows), BATCH):
                    part = windows[start : start + BATCH]
                    batch = np.stack([np.asarray(w, dtype=np.float32) for w in part])
                    model(torch.from_numpy(batch).to(self._device))
                    chunks.extend(cast(torch.Tensor, e).cpu() for e in model.embedding_store)
                    model.embedding_store.clear()
                model.embedding_store.append(torch.cat(chunks).to(self._device))
                chunks.clear()
                _seed(settings.seed)
                with warnings.catch_warnings():
                    # 2.6.2's @deprecate warns about an `embeddings` argument we never pass.
                    warnings.filterwarnings("ignore", "Argument embeddings", DeprecationWarning)
                    model.subsample_embedding(settings.coreset_ratio)
        finally:
            model.embedding_store.clear()
            model.eval()
        bank = model.memory_bank.detach().cpu().numpy().copy()
        peak = int(torch.cuda.max_memory_allocated(self._device)) if cuda else None
        return Fitted(bank=bank, peak_gpu_bytes=peak)

    def scorer(self, bank: np.ndarray) -> _AnomalibScorer:
        """A scorer holding its own copy of `bank` on the model's device."""
        if bank.ndim != 2 or bank.shape[0] == 0:
            raise ValueError(f"a memory bank is rows × features, not {bank.shape}")
        tensor = torch.from_numpy(np.array(bank, dtype=np.float32, copy=True))
        return _AnomalibScorer(self, tensor.to(self._device))

    def _score(self, bank: torch.Tensor, window_input: np.ndarray) -> Scored:
        if window_input.ndim != 3 or window_input.shape[0] != 3:
            raise ValueError(f"a window input is 3 × H × W, not {window_input.shape}")
        x = torch.from_numpy(np.array(window_input, dtype=np.float32, copy=True))
        x = x.unsqueeze(0).to(self._device)
        model = self._model
        model.memory_bank = bank
        model.eval()
        self._marks.clear()
        with torch.inference_mode():
            self._sync()
            start = time.perf_counter()
            output = model(x)
            self._sync()
            end = time.perf_counter()
        backbone_s = self._marks[1] - self._marks[0]
        anomaly_map = output.anomaly_map[0, 0].detach().float().cpu().numpy().copy()
        return Scored(
            anomaly_map=anomaly_map,
            pred_score=float(output.pred_score.reshape(-1)[0].item()),
            backbone_ms=backbone_s * 1000.0,
            search_ms=(end - start - backbone_s) * 1000.0,
        )


class _AnomalibScorer:
    """Scores one window against a fixed bank, under `eval()` and `inference_mode()`."""

    def __init__(self, owner: AnomalibPatchCore, bank: torch.Tensor) -> None:
        self._owner = owner
        self._bank = bank

    def score(self, window_input: np.ndarray) -> Scored:
        return self._owner._score(self._bank, window_input)

