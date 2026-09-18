"""Library-level reward aggregation functions that reduce a set of
per-molecule docking scores to a single scalar reward, used by both dataset
generation and the GFlowNet training loop.
"""

import numpy as np


AMPC_HITRATE_REWARD_MODES = {"ampc_pki6_hits", "ampc_pki6_proportion"}
"""Set of reward mode strings that delegate to the AmpC hit-rate model
(``ampc_pki6_hits``, ``ampc_pki6_proportion``).
"""

REWARD_MODES = ("mean", "topk_mean", "threshold", "ampc_pki6_hits", "ampc_pki6_proportion")
"""Tuple of all supported reward aggregation modes."""

DEFAULT_AMPC_PKI = 6.0
"""Default pKi threshold used by ``ampc_pki6_hits`` and ``ampc_pki6_proportion``
reward modes when no explicit ``pki`` argument is provided."""


def reward(
    values: np.ndarray,
    mode: str,
    *,
    k: int = 10,
    threshold: float = 0.0,
    alpha: float = 1.0,
    weights: np.ndarray | None = None,
    max_weight: float | None = None,
    pki: float = DEFAULT_AMPC_PKI,
) -> float:
    """Aggregate docking scores according to `mode`.

    When `weights` and `max_weight` are provided, entries with
    `weights[i] > max_weight` are excluded before aggregation.

    In ``threshold`` mode the reward is defined as
    ``1 + sum(1 / (1 + exp((value - threshold) / alpha)))``. The +1 baseline
    ensures terminal GFN rewards stay positive even when every molecule is
    filtered out by ``max_weight``.

    In ``ampc_pki6_hits`` mode the reward is the expected number of
    AmpC hits with ``pKi >= pki`` (default 6.0). When ``pki`` is not
    explicitly set the behaviour is identical to the original ``ampc_pki6_hits``.
    """
    mode = str(mode)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 1.0 if mode == "threshold" else (0.0 if mode in AMPC_HITRATE_REWARD_MODES else np.nan)
    if max_weight is not None and weights is not None:
        mask = weights <= max_weight
        if not np.any(mask):
            return 1.0 if mode == "threshold" else (0.0 if mode in AMPC_HITRATE_REWARD_MODES else 0.0)
        values = values[mask]
    if mode == "mean":
        return float(values.mean())
    if mode == "topk_mean":
        kk = min(k, values.size)
        idx = np.argpartition(values, kk - 1)[:kk]  # lower values = better (more negative kcal/mol)
        return float(values[idx].mean())
    if mode == "threshold":
        return float(1.0 + np.sum((1 + np.exp((values - threshold) / alpha)) ** (-1)))
    if mode in AMPC_HITRATE_REWARD_MODES:
        from deepdelgfn.ampc_hitrate import ampc_expected_hits, ampc_hit_proportion

        if mode == "ampc_pki6_hits":
            return float(ampc_expected_hits(values, pki=float(pki)))
        return float(ampc_hit_proportion(values, pki=float(pki)))
    raise ValueError(f"mode must be one of {REWARD_MODES}")

