#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate logZ(t) from a threshold-conditioned GFlowNet policy checkpoint.

The checkpoint (e.g. a ``gfn_policy_pretrain.pt`` produced by
``deepdelgfn.gfn.train_gfn`` during the conditioned-pretrain phase) stores the
threshold-conditioned partition function in ``logZ_net_state`` as the two-layer
MLP ``ConditionalLogZ``::

    logZ(t) = W2 · ReLU(W1 · fourier(t) + b1) + b2

where ``fourier(t) = [sin(2π·t·1.5^i), cos(2π·t·1.5^i)]_{i=0..n_freqs-1}``
(plus an optional raw-threshold append and optional threshold center/scale
normalization).  This script reconstructs that module from the checkpoint
weights (pure torch/numpy — no pandas/rdkit needed) and evaluates it over a
dense grid of thresholds.

Usage:
  python scripts/eval_logz.py \
      --ckpt outputs/experiment2/exp2_21278656/conditioned_pretrained/seed_11/gfn_policy_pretrain.pt \
      --t-min -12 --t-max -6 --n-points 61 \
      --out-csv outputs/experiment2/exp2_21278656/logz_vs_threshold.csv \
      --out-png outputs/experiment2/exp2_21278656/logz_vs_threshold.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def fourier_encode_torch(
    t: torch.Tensor,
    *,
    n_freqs: int,
    freq_scale: float,
    linear: bool,
    append_raw: bool,
    threshold_center,
    threshold_scale,
) -> torch.Tensor:
    """Mirror of ``deepdelgfn.deepdel.fourier.fourier_encode_torch``."""
    n_freqs = int(n_freqs)
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be > 0, got {n_freqs}")
    if threshold_scale is not None:
        threshold_scale = float(threshold_scale)
        t = (t - float(threshold_center or 0.0)) / threshold_scale

    if linear:
        freqs = torch.arange(1, n_freqs + 1, dtype=t.dtype, device=t.device)
    else:
        freq_scale = float(freq_scale)
        freqs = torch.tensor(
            [freq_scale ** i for i in range(n_freqs)],
            dtype=t.dtype,
            device=t.device,
        )

    t = t.unsqueeze(-1)
    angles = 2.0 * torch.pi * t * freqs
    out = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
    out = out.reshape(*t.shape[:-1], 2 * n_freqs)
    if append_raw:
        out = torch.cat([out, t], dim=-1)
    return out


def build_logz_fn(ckpt: dict):
    """Return a callable logZ(tensor) -> tensor from a checkpoint dict."""
    meta = ckpt["meta"]
    lz = ckpt["logZ_net_state"]

    n_freqs = int(meta["gfn_fourier_n_freqs"])
    freq_scale = float(meta["gfn_fourier_freq_scale"])
    linear = bool(meta["gfn_fourier_linear"])
    append_raw = bool(meta["gfn_fourier_append_raw"])
    center = meta.get("gfn_fourier_threshold_center")
    scale = meta.get("gfn_fourier_threshold_scale")

    # The checkpoint may store either a conditioned MLP (net.*) or a scalar
    # (logZ_scalar) depending on whether strip_conditioning was used.
    if "net.0.weight" in lz:
        W1 = lz["net.0.weight"]
        b1 = lz["net.0.bias"]
        W2 = lz["net.2.weight"]
        b2 = lz["net.2.bias"]

        def logz(t: torch.Tensor) -> torch.Tensor:
            f = fourier_encode_torch(
                t,
                n_freqs=n_freqs,
                freq_scale=freq_scale,
                linear=linear,
                append_raw=append_raw,
                threshold_center=center,
                threshold_scale=scale,
            )
            h = torch.relu(f @ W1.T + b1)
            return (h @ W2.T + b2).squeeze(-1)

    else:  # scalar (non-conditioned) checkpoint
        scalar = float(lz["logZ_scalar"].detach())

        def logz(t: torch.Tensor) -> torch.Tensor:
            return torch.full(t.shape[:-1] if t.dim() > 1 else t.shape, scalar)

    return logz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=str, help="Policy checkpoint (.pt)")
    ap.add_argument("--t-min", type=float, default=-12.0)
    ap.add_argument("--t-max", type=float, default=-6.0)
    ap.add_argument("--n-points", type=int, default=61)
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--out-png", type=str, default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    logz = build_logz_fn(ck)

    ts = np.linspace(args.t_min, args.t_max, args.n_points, dtype=np.float64)
    with torch.no_grad():
        zs = logz(torch.as_tensor(ts, dtype=torch.float32)).numpy()

    print(f"{'t':>8}  {'logZ(t)':>12}")
    for t, z in zip(ts, zs):
        print(f"{t:8.3f}  {z:12.4f}")

    finite = zs[np.isfinite(zs)]
    print(
        f"\nOver [{args.t_min}, {args.t_max}] ({args.n_points} pts): "
        f"min={finite.min():.4f} max={finite.max():.4f} "
        f"mean={finite.mean():.4f} std={finite.std():.4f} "
        f"all_finite={bool(np.isfinite(zs).all())}"
    )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_csv,
            np.column_stack([ts, zs]),
            header="threshold,logZ",
            delimiter=",",
            comments="",
            fmt="%.6f",
        )
        print(f"Wrote {out_csv}")

    if args.out_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "Skipping PNG: matplotlib not installed in this environment. "
                f"The CSV was already written. Run with a matplotlib-enabled env "
                f"(e.g. `module load StdEnv/2023 scipy-stack/2025a`) to get the plot."
            )
            return

        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts, zs, marker="o", ms=3, lw=1.5)
        ax.set_xlabel("threshold t")
        ax.set_ylabel("logZ(t)")
        ax.set_title(f"Conditioned logZ — {args.ckpt}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
