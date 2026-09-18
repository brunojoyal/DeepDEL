#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fourier (sinusoidal) encoding of a scalar conditioning value.

NeRF-style positional encodings map a scalar ``t`` to a vector::

    [sin(2π · t · f_0), cos(2π · t · f_0), ...,
     sin(2π · t · f_{n-1}), cos(2π · t · f_{n-1})]

where the frequencies ``f_i`` are either geometric (``freq_scale**i``) or
linear (``i + 1``).  This is used to condition the DeepDEL reward model and
GFlowNet policy on the docking-score threshold without baking in a single
hardcoded value.
"""

import numpy as np
import torch


def _frequencies(n_freqs: int, freq_scale: float, linear: bool) -> np.ndarray:
    """Return the ordered frequency vector for a Fourier encoding.

    ``linear=False`` uses geometric frequencies ``freq_scale**i`` (NeRF-style),
    which span many octaves. ``linear=True`` uses ``i+1``.
    """
    n_freqs = int(n_freqs)
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be > 0, got {n_freqs}")
    if linear:
        return np.arange(1, n_freqs + 1, dtype=np.float64)
    freq_scale = float(freq_scale)
    if freq_scale <= 0.0:
        raise ValueError(f"freq_scale must be > 0, got {freq_scale}")
    return freq_scale ** np.arange(n_freqs, dtype=np.float64)


def fourier_dim(n_freqs: int, append_raw: bool = False) -> int:
    """Return the output dimensionality of the encoding."""
    return 2 * int(n_freqs) + (1 if append_raw else 0)


def fourier_encode(
    t: float,
    *,
    n_freqs: int = 8,
    freq_scale: float = 1.5,
    linear: bool = False,
    append_raw: bool = False,
    threshold_center: float | None = None,
    threshold_scale: float | None = None,
) -> np.ndarray:
    """Encode a scalar ``t`` as a numpy Fourier feature vector.

    The raw value is not normalized, so callers should choose ``freq_scale``
    and the conditioning range to cover the expected dynamic range of ``t``.
    """
    t = float(t)
    if threshold_scale is not None:
        threshold_scale = float(threshold_scale)
        if not np.isfinite(threshold_scale) or threshold_scale <= 0.0:
            raise ValueError(f"threshold_scale must be finite and > 0, got {threshold_scale}")
        t = (t - float(threshold_center or 0.0)) / threshold_scale
    freqs = _frequencies(n_freqs, freq_scale, linear)
    angles = 2.0 * np.pi * t * freqs
    out = np.empty(2 * freqs.size, dtype=np.float32)
    out[0::2] = np.sin(angles).astype(np.float32)
    out[1::2] = np.cos(angles).astype(np.float32)
    if append_raw:
        out = np.append(out, np.array(t, dtype=np.float32))
    return out


def fourier_encode_torch(
    t: torch.Tensor,
    *,
    n_freqs: int = 8,
    freq_scale: float = 1.5,
    linear: bool = False,
    append_raw: bool = False,
    threshold_center: float | None = None,
    threshold_scale: float | None = None,
) -> torch.Tensor:
    """Encode a tensor of scalar thresholds as Fourier feature vectors.

    ``t`` may have arbitrary shape ``[...,]``; the output is
    ``[..., 2*n_freqs (+1)]``.
    """
    n_freqs = int(n_freqs)
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be > 0, got {n_freqs}")
    if threshold_scale is not None:
        threshold_scale = float(threshold_scale)
        if not np.isfinite(threshold_scale) or threshold_scale <= 0.0:
            raise ValueError(f"threshold_scale must be finite and > 0, got {threshold_scale}")
        t = (t - float(threshold_center or 0.0)) / threshold_scale

    if linear:
        freqs = torch.arange(1, n_freqs + 1, dtype=t.dtype, device=t.device)
    else:
        freq_scale = float(freq_scale)
        if freq_scale <= 0.0:
            raise ValueError(f"freq_scale must be > 0, got {freq_scale}")
        freqs = torch.tensor(
            [freq_scale ** i for i in range(n_freqs)],
            dtype=t.dtype,
            device=t.device,
        )

    t = t.unsqueeze(-1)  # [..., 1]
    angles = 2.0 * torch.pi * t * freqs  # [..., n_freqs]
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)
    out = torch.stack([sin_part, cos_part], dim=-1)  # [..., n_freqs, 2]
    out = out.reshape(*t.shape[:-1], 2 * n_freqs)  # [..., 2*n_freqs]
    if append_raw:
        out = torch.cat([out, t], dim=-1)  # t is already [..., 1]
    return out