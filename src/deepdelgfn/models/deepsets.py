"""Canonical DeepSets model for three-cycle DEL libraries."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepdelgfn.deepdel.fourier import fourier_dim, fourier_encode_torch


class Phi(nn.Module):
    """Per-building-block embedding network."""

    def __init__(self, d_in, d_hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TripleDeepSet(nn.Module):
    """Permutation-invariant regression model for DEL triples."""

    OUTPUT_HEADS = ("linear", "sigmoid_scaled", "softplus_scaled")

    def __init__(self, d_in, d_hidden=256, d_rho=256, dropout=0.1, shared_phi=True,
                 pooling="mean", output_head="linear", reward_bound_k=None,
                 fourier_n_freqs=0, fourier_freq_scale=1.5, fourier_linear=False,
                 fourier_append_raw=False, fourier_condition="rho",
                 fourier_threshold_center=None, fourier_threshold_scale=None):
        super().__init__()
        pooling = str(pooling).lower()
        output_head = str(output_head).lower()
        if pooling not in {"mean", "sum"}:
            raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")
        if output_head not in self.OUTPUT_HEADS:
            raise ValueError(f"output_head must be one of {self.OUTPUT_HEADS}, got {output_head!r}")
        if output_head != "linear" and (reward_bound_k is None or int(reward_bound_k) <= 0):
            raise ValueError(f"output_head={output_head!r} requires reward_bound_k (lib_size) > 0")
        self.output_head = output_head
        self.reward_bound_k = None if reward_bound_k is None else int(reward_bound_k)
        self.log_reward_bound = None if self.reward_bound_k is None else math.log(1.0 + self.reward_bound_k ** 3)
        self.pooling = pooling
        self.fourier_n_freqs = int(fourier_n_freqs)
        self.fourier_freq_scale = float(fourier_freq_scale)
        self.fourier_linear = bool(fourier_linear)
        self.fourier_append_raw = bool(fourier_append_raw)
        self.fourier_threshold_center = fourier_threshold_center
        self.fourier_threshold_scale = fourier_threshold_scale
        self.fourier_condition = str(fourier_condition).lower()
        if self.fourier_condition not in {"rho", "phi"}:
            raise ValueError(f"fourier_condition must be 'rho' or 'phi', got {fourier_condition!r}")
        self.fourier_dim = fourier_dim(self.fourier_n_freqs, self.fourier_append_raw) if self.fourier_n_freqs > 0 else 0
        if shared_phi:
            self.phi = Phi(d_in, d_hidden, dropout)
            self.phi2 = self.phi
            self.phi3 = self.phi
        else:
            self.phi = Phi(d_in, d_hidden, dropout)
            self.phi2 = Phi(d_in, d_hidden, dropout)
            self.phi3 = Phi(d_in, d_hidden, dropout)
        rho_in = 3 * d_hidden + (self.fourier_dim if self.fourier_condition == "rho" else 0)
        self.rho = nn.Sequential(
            nn.Linear(rho_in, d_rho), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_rho, 1)
        )

    def _pool(self, h, m):
        me = m.unsqueeze(-1)
        h_sum = (h * me).sum(1)
        if self.pooling == "sum":
            return h_sum
        return h_sum / me.sum(1).clamp(min=1.0)

    def forward_logits(self, triple, threshold: Optional[torch.Tensor] = None):
        """Return the raw scalar logit before applying the output head."""
        (X1, M1), (X2, M2), (X3, M3) = triple
        if self.fourier_n_freqs > 0 and threshold is not None:
            f = fourier_encode_torch(threshold, n_freqs=self.fourier_n_freqs,
                                     freq_scale=self.fourier_freq_scale,
                                     linear=self.fourier_linear,
                                     append_raw=self.fourier_append_raw,
                                     threshold_center=self.fourier_threshold_center,
                                     threshold_scale=self.fourier_threshold_scale)
            if self.fourier_condition == "phi":
                X1 = torch.cat([X1, f.unsqueeze(1).expand(-1, X1.shape[1], -1)], dim=-1)
                X2 = torch.cat([X2, f.unsqueeze(1).expand(-1, X2.shape[1], -1)], dim=-1)
                X3 = torch.cat([X3, f.unsqueeze(1).expand(-1, X3.shape[1], -1)], dim=-1)
        H = torch.cat([self._pool(self.phi(X1), M1), self._pool(self.phi2(X2), M2),
                       self._pool(self.phi3(X3), M3)], dim=1)
        if self.fourier_n_freqs > 0 and threshold is not None and self.fourier_condition == "rho":
            H = torch.cat([H, f], dim=1)
        return self.rho(H).squeeze(-1)

    def forward(self, triple, threshold: Optional[torch.Tensor] = None):
        z = self.forward_logits(triple, threshold)
        if self.output_head == "sigmoid_scaled":
            return self.log_reward_bound * torch.sigmoid(z)
        if self.output_head == "softplus_scaled":
            return self.log_reward_bound * F.softplus(z)
        return z