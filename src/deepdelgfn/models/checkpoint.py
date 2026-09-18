"""Configuration and compatibility helpers for DeepSets checkpoints."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch

from .deepsets import TripleDeepSet


@dataclass(frozen=True)
class DeepSetConfig:
    d_in: int
    d_hidden: int = 256
    d_rho: int = 256
    dropout: float = 0.1
    shared_phi: bool = True
    pooling: str = "mean"
    output_head: str = "linear"
    reward_bound_k: Optional[int] = None
    fourier_n_freqs: int = 0
    fourier_freq_scale: float = 1.5
    fourier_linear: bool = False
    fourier_append_raw: bool = False
    fourier_condition: str = "rho"
    fourier_threshold_center: Optional[float] = None
    fourier_threshold_scale: Optional[float] = None

    def __post_init__(self):
        if min(self.d_in, self.d_hidden, self.d_rho) <= 0:
            raise ValueError("DeepSet dimensions must be positive")
        if self.pooling not in {"mean", "sum"}:
            raise ValueError(f"Unsupported pooling: {self.pooling!r}")
        if self.output_head not in TripleDeepSet.OUTPUT_HEADS:
            raise ValueError(f"Unsupported output head: {self.output_head!r}")
        if self.fourier_condition not in {"rho", "phi"}:
            raise ValueError(f"Unsupported Fourier condition: {self.fourier_condition!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_deepset_config(ckpt: Mapping[str, Any]) -> DeepSetConfig:
    """Build config from current metadata or legacy ``ckpt['args']`` fields."""
    args = ckpt.get("args", {}) or {}
    meta = ckpt.get("model_config", {}) or {}
    source = {**args, **meta}
    return DeepSetConfig(
        d_in=int(ckpt.get("d_in", source.get("d_in", source.get("bb_fp_bits", 2048)))),
        d_hidden=int(source.get("d_hidden", source.get("hidden_dim", 256))),
        d_rho=int(source.get("d_rho", source.get("rho_dim", 256))),
        dropout=float(source.get("dropout", 0.1)),
        shared_phi=bool(source.get("shared_phi", False)),
        pooling=str(source.get("pooling", "mean")).lower(),
        output_head=str(source.get("output_head", "linear")).lower(),
        reward_bound_k=source.get("reward_bound_k", source.get("lib_size")),
        fourier_n_freqs=int(source.get("fourier_n_freqs", 0)),
        fourier_freq_scale=float(source.get("fourier_freq_scale", 1.5)),
        fourier_linear=bool(source.get("fourier_linear", False)),
        fourier_append_raw=bool(source.get("fourier_append_raw", False)),
        fourier_condition=str(source.get("fourier_condition", "rho")).lower(),
        fourier_threshold_center=source.get("fourier_threshold_center"),
        fourier_threshold_scale=source.get("fourier_threshold_scale"),
    )


def load_deepset_checkpoint(path, *, map_location="cpu", strict=True):
    """Load a legacy or current checkpoint and return ``(model, checkpoint)``."""
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError("DeepDEL checkpoint must contain 'model_state'.")
    config = make_deepset_config(ckpt)
    model = TripleDeepSet(**config.to_dict())
    try:
        model.load_state_dict(ckpt["model_state"], strict=strict)
    except RuntimeError as exc:
        raise ValueError(f"Incompatible DeepDEL checkpoint for {config}: {exc}") from exc
    model.eval()
    return model, ckpt