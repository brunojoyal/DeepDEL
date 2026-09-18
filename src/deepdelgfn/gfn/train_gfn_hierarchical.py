#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hierarchical (H-DEL-GFlowNet) for DEL library design, following Section 3.2
of Koziarski et al. (2024) "Towards DNA-Encoded Library Generation with
GFlowNets".

State representation:
    x = [binary vector over all truncated pools (N dims)]
      ⊕ [one-hot cycle (3 dims)]
      ⊕ [one-hot cluster (max_clusters dims, fixed at 20)]
      ⊕ [cycle_picked flag (1 dim)]
      ⊕ [cluster_picked flag (1 dim)]

Action space (3-level hierarchy):
    1. Pick cycle      ∈ {1, 2, 3}      (masked: not yet full)
    2. Pick cluster    ∈ {0, ..., K_c-1} (masked: has remaining BBs for cycle)
    3. Pick building block ∈ [0, N_c)    (masked: not yet selected, in cluster)

Each environment step produces 3 log-probability terms:
    log π_total = log π(cycle|s) + log π(cluster|s,cycle) + log π(bb|s,cycle,cluster)

Trajectory Balance loss:
    L_TB = (logZ + Σ_t log π_total(a_t|s_t) - β · log R(x))²

Pool sizes default to the paper's truncated values: N1=90, N2=89, N3=197.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Reuse shared infrastructure from the deepsets GFN trainer
from deepdelgfn.gfn.train_gfn import (
    set_seed,
    load_bbs,
    load_or_create_ecfp_cache,
    smiles_to_morgan_bits,
    TripleDeepSet,
    Phi,
    AutodockProxyRewardOracle,
    MetricsRecorder,
    reward_from_triple,
    tb_loss,
    anneal_ratio,
    progress_write,
    generalized_tanimoto_distance,
    cosine_distance,
    euclidean_distance,
    build_triple_representation,
    mean_pairwise_distance,
    _parse_pipe_separated_ints,
)
from deepdelgfn.rewards import REWARD_MODES
from deepdelgfn.autodock_proxy.model import load_autodock_proxy
from deepdelgfn.utils.weights import smiles_weight_array, bb_weight_lookup, bb_sum_weights
import deepdelgfn.mols.dels as tri_mod

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# ============================================================================
# Helper: load cluster assignments
# ============================================================================

def load_clusters(path: str) -> Dict[int, Dict[int, int]]:
    """Load cluster JSON produced by scripts/precompute_clusters.py.

    Returns: {cycle: {global_bb_idx: cluster_id}}
        cycle ∈ {1, 2, 3}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cluster file not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    clusters: Dict[int, Dict[int, int]] = {}
    for cycle_str, mapping in raw.items():
        cycle = int(cycle_str)
        clusters[cycle] = {int(k): int(v) for k, v in mapping.items()}
    # Validate
    for c in (1, 2, 3):
        if c not in clusters:
            raise ValueError(f"Cluster file missing cycle {c}")
    return clusters


# ============================================================================
# Hierarchical Multi-hot Environment
# ============================================================================

class HierarchicalMultiHotEnv:
    """Binary-vector DEL environment with hierarchical action selection.

    State: x ∈ {0,1}^{N}  where N = n1 + n2 + n3 (truncated pool sizes)
    Augmented with cycle one-hot, cluster one-hot, and two binary flags.

    Actions are 3 sub-steps per BB selection:
        1) cycle ∈ {0,1,2}
        2) cluster ∈ cluster_ids for that cycle
        3) bb ∈ {global indices in that cluster}
    """

    def __init__(
        self,
        sizes: Tuple[int, int, int],
        pool_sizes: Tuple[int, int, int],
        cluster_assignments: Dict[int, Dict[int, int]],
        device: torch.device,
        allowed_per_cycle: Optional[Tuple[List[int], List[int], List[int]]] = None,
        max_clusters: int = 20,
    ):
        self.s1, self.s2, self.s3 = sizes
        self.n1, self.n2, self.n3 = pool_sizes
        self.N = self.n1 + self.n2 + self.n3
        self.device = device
        self.max_clusters = max_clusters

        # Offsets within the flat binary vector
        self.off1 = 0
        self.off2 = self.n1
        self.off3 = self.n1 + self.n2

        # Allowed BB row indices per cycle (maps local position → global BB index)
        if allowed_per_cycle is None:
            self.allowed1 = list(range(self.n1))
            self.allowed2 = list(range(self.n2))
            self.allowed3 = list(range(self.n3))
        else:
            self.allowed1 = list(allowed_per_cycle[0][:self.n1])
            self.allowed2 = list(allowed_per_cycle[1][:self.n2])
            self.allowed3 = list(allowed_per_cycle[2][:self.n3])

        if self.s1 > len(self.allowed1):
            raise ValueError(f"size1={self.s1} exceeds pool-1 size={len(self.allowed1)}")
        if self.s2 > len(self.allowed2):
            raise ValueError(f"size2={self.s2} exceeds pool-2 size={len(self.allowed2)}")
        if self.s3 > len(self.allowed3):
            raise ValueError(f"size3={self.s3} exceeds pool-3 size={len(self.allowed3)}")

        # Cluster assignments
        self.cluster_assignments = cluster_assignments

        # Build reverse maps: (cycle, global_idx) → cluster_id
        # and: (cycle, cluster_id) → list of local positions
        self.global_to_cluster: Dict[Tuple[int, int], int] = {}
        self.cluster_to_local_positions: Dict[Tuple[int, int], List[int]] = {}
        # Also track unique cluster IDs per cycle
        self.cluster_ids_per_cycle: Dict[int, List[int]] = {1: [], 2: [], 3: []}

        for cycle, allowed in ((1, self.allowed1), (2, self.allowed2), (3, self.allowed3)):
            seen_clusters: Set[int] = set()
            for local_pos, global_idx in enumerate(allowed):
                cluster_id = cluster_assignments[cycle].get(global_idx, 0)
                self.global_to_cluster[(cycle, global_idx)] = cluster_id
                key = (cycle, cluster_id)
                if key not in self.cluster_to_local_positions:
                    self.cluster_to_local_positions[key] = []
                self.cluster_to_local_positions[key].append(local_pos)
                seen_clusters.add(cluster_id)
            self.cluster_ids_per_cycle[cycle] = sorted(seen_clusters)

        # Augmented state dimensions
        self.n_cycle_onehot = 3
        self.n_cluster_onehot = max_clusters
        self.augmented_dim = self.N + self.n_cycle_onehot + self.n_cluster_onehot + 2

        self.reset()

    def reset(self):
        self.x = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0
        # Internal step tracking: None means sub-step 1 (pick cycle)
        self._cycle_picked: Optional[int] = None
        self._cluster_picked: Optional[int] = None

    def is_terminal(self) -> bool:
        return self.count1 == self.s1 and self.count2 == self.s2 and self.count3 == self.s3

    # ------ State representation ------

    def _cycle_onehot(self) -> torch.Tensor:
        oh = torch.zeros(self.n_cycle_onehot, device=self.device, dtype=torch.float32)
        if self._cycle_picked is not None:
            oh[self._cycle_picked] = 1.0
        return oh

    def _cluster_onehot(self) -> torch.Tensor:
        oh = torch.zeros(self.n_cluster_onehot, device=self.device, dtype=torch.float32)
        if self._cluster_picked is not None and self._cluster_picked < self.n_cluster_onehot:
            oh[self._cluster_picked] = 1.0
        return oh

    def state_vector(self) -> torch.Tensor:
        """Return the augmented state vector [N + 3 + max_clusters + 2]."""
        return torch.cat([
            self.x,
            self._cycle_onehot(),
            self._cluster_onehot(),
            torch.tensor(
                [1.0 if self._cycle_picked is not None else 0.0,
                 1.0 if self._cluster_picked is not None else 0.0],
                device=self.device, dtype=torch.float32,
            ),
        ], dim=0)

    # ------ Action space ------

    def valid_cycle_mask(self) -> torch.Tensor:
        """Return boolean mask [3]: which cycles can be picked (not yet full)."""
        mask = torch.zeros(3, device=self.device, dtype=torch.bool)
        if self.count1 < self.s1:
            mask[0] = True
        if self.count2 < self.s2:
            mask[1] = True
        if self.count3 < self.s3:
            mask[2] = True
        return mask

    def valid_cluster_mask(self, cycle: int) -> torch.Tensor:
        """Return boolean mask [max_clusters]: which clusters have remaining BBs."""
        mask = torch.zeros(self.max_clusters, device=self.device, dtype=torch.bool)
        # cycle is 0-indexed; all dictionaries use 1-indexed keys
        cycle1 = cycle + 1
        allowed = {1: self.allowed1, 2: self.allowed2, 3: self.allowed3}[cycle1]
        for cluster_id in self.cluster_ids_per_cycle[cycle1]:
            if cluster_id >= self.max_clusters:
                continue
            local_positions = self.cluster_to_local_positions.get((cycle1, cluster_id), [])
            # Check if any BB in this cluster is still available
            for lp in local_positions:
                if lp < len(allowed):
                    if cycle == 0:
                        bit_idx = self.off1 + lp
                    elif cycle == 1:
                        bit_idx = self.off2 + lp
                    else:
                        bit_idx = self.off3 + lp
                    if self.x[bit_idx] == 0:
                        mask[cluster_id] = True
                        break
        return mask

    def valid_bb_mask(self, cycle: int, cluster_id: int) -> torch.Tensor:
        """Return boolean mask [N]: which BBs can be picked (in cluster, not selected)."""
        mask = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        # cycle is 0-indexed; all dictionaries use 1-indexed keys
        cycle1 = cycle + 1
        allowed = {1: self.allowed1, 2: self.allowed2, 3: self.allowed3}[cycle1]
        local_positions = self.cluster_to_local_positions.get((cycle1, cluster_id), [])
        for lp in local_positions:
            if lp >= len(allowed):
                continue
            if cycle == 0:
                bit_idx = self.off1 + lp
            elif cycle == 1:
                bit_idx = self.off2 + lp
            else:
                bit_idx = self.off3 + lp
            if self.x[bit_idx] == 0:
                mask[bit_idx] = True
        return mask

    # ------ Step ------

    def step_cycle(self, cycle: int):
        """Set the current cycle for this selection step.

        Accepts 0-indexed cycle (0, 1, 2) and stores as-is.
        cluster/BB dictionaries use 1-indexed keys, so lookups convert.
        """
        self._cycle_picked = cycle
        self._cluster_picked = None

    def step_cluster(self, cluster_id: int):
        """Set the current cluster for this selection step."""
        self._cluster_picked = cluster_id

    def step_bb(self, bb_idx: int):
        """Flip bit bb_idx from 0 to 1 and advance counts."""
        if self.x[bb_idx] != 0:
            raise ValueError(f"Invalid BB action: bit {bb_idx} is already 1.")
        self.x[bb_idx] = 1.0
        if bb_idx < self.off2:
            self.count1 += 1
        elif bb_idx < self.off3:
            self.count2 += 1
        else:
            self.count3 += 1
        # Reset sub-step tracking for next BB selection
        self._cycle_picked = None
        self._cluster_picked = None

    # ------ Terminal ------

    def terminal_sets(self) -> Tuple[List[int], List[int], List[int]]:
        """Map positions back to original BB row indices for reward computation."""
        B1, B2, B3 = [], [], []
        for lp in range(self.n1):
            if self.x[self.off1 + lp] == 1:
                B1.append(self.allowed1[lp])
        for lp in range(self.n2):
            if self.x[self.off2 + lp] == 1:
                B2.append(self.allowed2[lp])
        for lp in range(self.n3):
            if self.x[self.off3 + lp] == 1:
                B3.append(self.allowed3[lp])
        B1.sort(); B2.sort(); B3.sort()
        return B1, B2, B3

    @property
    def cycle_picked(self) -> Optional[int]:
        return self._cycle_picked

    @property
    def cluster_picked(self) -> Optional[int]:
        return self._cluster_picked


# ============================================================================
# Hierarchical Policy (shared encoder + three heads)
# ============================================================================

class HierarchicalPolicy(nn.Module):
    """Hierarchical MLP policy with shared state encoder and three action heads.

    Architecture:
        shared_encoder: [augmented_dim] → hidden → ... → hidden  (n_layers)
        cycle_head: Linear(hidden, 3)         → cycle logits
        cluster_head: Linear(hidden, max_clusters) → cluster logits
        bb_head: Linear(hidden, N)            → BB logits (masked per cluster)
    """

    def __init__(
        self,
        state_dim: int,
        N: int,
        max_clusters: int = 20,
        hidden_dim: int = 512,
        n_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.N = N
        self.max_clusters = max_clusters

        # Shared encoder
        layers: List[nn.Module] = []
        in_dim = state_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Three heads
        self.cycle_head = nn.Linear(hidden_dim, 3)
        self.cluster_head = nn.Linear(hidden_dim, max_clusters)
        self.bb_head = nn.Linear(hidden_dim, N)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """state: [B, state_dim] → hidden: [B, hidden_dim]"""
        return self.encoder(state)

    def forward_cycle(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, hidden_dim] → cycle_logits: [B, 3]"""
        return self.cycle_head(hidden)

    def forward_cluster(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, hidden_dim] → cluster_logits: [B, max_clusters]"""
        return self.cluster_head(hidden)

    def forward_bb(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, hidden_dim] → bb_logits: [B, N]"""
        return self.bb_head(hidden)


# ============================================================================
# Rollout
# ============================================================================

def rollout_trajectory_hierarchical(
    env: HierarchicalMultiHotEnv,
    policy: HierarchicalPolicy,
    rng: np.random.Generator,
    device: torch.device,
    epsilon: float = 0.05,
) -> Dict:
    """Roll out a single trajectory using the hierarchical action space.

    Returns:
        actions: list of (cycle, cluster, bb_idx) tuples
        logps: list of 3-tuples of torch.Tensor (logπ_cycle, logπ_cluster, logπ_bb)
        terminal: (B1, B2, B3) lists
    """
    env.reset()
    actions: List[Tuple[int, int, int]] = []
    logps: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    while not env.is_terminal():
        s = env.state_vector().unsqueeze(0)  # [1, augmented_dim]
        hidden = policy.encode(s.float())  # [1, hidden_dim]

        # --- Sub-step 1: pick cycle ---
        mask_cycle = env.valid_cycle_mask().unsqueeze(0)  # [1, 3]
        cycle_logits = policy.forward_cycle(hidden).squeeze(0)  # [3]
        cycle_logits = cycle_logits.masked_fill(~mask_cycle, float("-inf"))

        valid_cycle_logits = cycle_logits[mask_cycle]
        if not torch.isfinite(valid_cycle_logits).all():
            # Fallback uniform
            n_valid = int(mask_cycle.sum().item())
            if torch.rand((), device=device).item() < epsilon:
                valid_idx = torch.where(mask_cycle)[0]
                cycle = valid_idx[torch.randint(n_valid, (1,), device=device)].item()
            else:
                log_probs = mask_cycle.float() / n_valid
                cycle = torch.multinomial(log_probs, 1).item()
            logp_cycle = torch.log(mask_cycle.float()[cycle] / n_valid)
        else:
            n_valid = int(mask_cycle.sum().item())
            if torch.rand((), device=device).item() < epsilon:
                valid_idx = torch.where(mask_cycle)[0]
                cycle = valid_idx[torch.randint(n_valid, (1,), device=device)].item()
            else:
                dist = torch.distributions.Categorical(logits=cycle_logits)
                cycle = dist.sample().item()
            logp_cycle = cycle_logits[cycle] - torch.logsumexp(cycle_logits[mask_cycle], dim=-1)

        env.step_cycle(cycle)

        # --- Sub-step 2: pick cluster ---
        s2 = env.state_vector().unsqueeze(0)  # [1, augmented_dim] — now has cycle one-hot
        hidden2 = policy.encode(s2.float())  # [1, hidden_dim]

        mask_cluster = env.valid_cluster_mask(cycle).unsqueeze(0)  # [1, max_clusters]
        cluster_logits = policy.forward_cluster(hidden2).squeeze(0)  # [max_clusters]
        cluster_logits = cluster_logits.masked_fill(~mask_cluster, float("-inf"))

        valid_cluster_logits = cluster_logits[mask_cluster]
        if not torch.isfinite(valid_cluster_logits).all() or int(mask_cluster.sum().item()) == 0:
            n_valid = int(mask_cluster.sum().item())
            if n_valid == 0:
                # No clusters available — shouldn't happen if cycle was valid
                # but handle gracefully: reset and continue
                env._cycle_picked = None
                continue
            if torch.rand((), device=device).item() < epsilon:
                valid_idx = torch.where(mask_cluster)[0]
                cluster_id = valid_idx[torch.randint(n_valid, (1,), device=device)].item()
            else:
                log_probs = mask_cluster.float() / n_valid
                cluster_id = torch.multinomial(log_probs, 1).item()
            logp_cluster = torch.log(mask_cluster.float()[cluster_id] / n_valid)
        else:
            n_valid = int(mask_cluster.sum().item())
            if torch.rand((), device=device).item() < epsilon:
                valid_idx = torch.where(mask_cluster)[0]
                cluster_id = valid_idx[torch.randint(n_valid, (1,), device=device)].item()
            else:
                dist = torch.distributions.Categorical(logits=cluster_logits)
                cluster_id = dist.sample().item()
            logp_cluster = cluster_logits[cluster_id] - torch.logsumexp(
                cluster_logits[mask_cluster], dim=-1
            )

        env.step_cluster(cluster_id)

        # --- Sub-step 3: pick BB ---
        s3 = env.state_vector().unsqueeze(0)  # [1, augmented_dim]
        hidden3 = policy.encode(s3.float())  # [1, hidden_dim]

        mask_bb = env.valid_bb_mask(cycle, cluster_id).unsqueeze(0)  # [1, N]
        bb_logits = policy.forward_bb(hidden3).squeeze(0)  # [N]
        bb_logits = bb_logits.masked_fill(~mask_bb, float("-inf"))

        valid_bb_logits = bb_logits[mask_bb]
        if not torch.isfinite(valid_bb_logits).all() or int(mask_bb.sum().item()) == 0:
            n_valid = int(mask_bb.sum().item())
            if n_valid == 0:
                env._cycle_picked = None
                env._cluster_picked = None
                continue
            if torch.rand((), device=device).item() < epsilon:
                valid_idx = torch.where(mask_bb)[0]
                bb_idx = valid_idx[torch.randint(n_valid, (1,), device=device)].item()
            else:
                log_probs = mask_bb.float() / n_valid
                bb_idx = torch.multinomial(log_probs, 1).item()
            logp_bb = torch.log(mask_bb.float()[bb_idx] / n_valid)
        else:
            n_valid = int(mask_bb.sum().item())
            if torch.rand((), device=device).item() < epsilon:
                valid_idx = torch.where(mask_bb)[0]
                bb_idx = valid_idx[torch.randint(n_valid, (1,), device=device)].item()
            else:
                dist = torch.distributions.Categorical(logits=bb_logits)
                bb_idx = dist.sample().item()
            logp_bb = bb_logits[bb_idx] - torch.logsumexp(bb_logits[mask_bb], dim=-1)

        env.step_bb(bb_idx)

        actions.append((cycle, cluster_id, bb_idx))
        logps.append((logp_cycle, logp_cluster, logp_bb))

    B1, B2, B3 = env.terminal_sets()
    # Flatten logps: each step contributed 3 log-probs
    flattened_logps: List[torch.Tensor] = []
    for lp_triple in logps:
        flattened_logps.extend(lp_triple)
    return {
        "actions": actions,
        "logps": flattened_logps,
        "terminal": (B1, B2, B3),
    }


def _sample_action_with_epsilon(
    logits: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    epsilon: float,
) -> Tuple[int, torch.Tensor]:
    """Sample from masked logits with epsilon-greedy exploration.

    Args:
        logits: [dim] logits (masked positions set to -inf)
        mask: [dim] boolean mask of valid actions
        device: torch device
        epsilon: probability of random action

    Returns:
        (action_idx, log_prob) tuple
    """
    valid_indices = torch.where(mask)[0]
    n_valid = len(valid_indices)

    if torch.rand((), device=device).item() < epsilon:
        action = valid_indices[torch.randint(n_valid, (1,), device=device)].item()
    else:
        action = torch.distributions.Categorical(logits=logits).sample().item()
    logp = logits[action] - torch.logsumexp(logits[mask], dim=-1)
    return action, logp


def _sample_uniform(
    mask: torch.Tensor,
    device: torch.device,
) -> Tuple[int, torch.Tensor]:
    """Sample uniformly from mask (fallback for non-finite logits)."""
    valid_indices = torch.where(mask)[0]
    n_valid = len(valid_indices)
    action = valid_indices[torch.multinomial(mask.float() / n_valid, 1)].item()
    logp = torch.log(mask.float()[action] / n_valid)
    return action, logp


def rollout_trajectories_batched_hierarchical(
    *,
    sizes: Tuple[int, int, int],
    pool_sizes: Tuple[int, int, int],
    cluster_assignments: Dict[int, Dict[int, int]],
    policy: HierarchicalPolicy,
    rng: np.random.Generator,
    batch_trajectories: int,
    device: torch.device,
    epsilon: float = 0.05,
    allowed_per_cycle: Optional[Tuple[List[int], List[int], List[int]]] = None,
    max_clusters: int = 20,
) -> List[Dict]:
    """Batched rollout for hierarchical GFN with robust sub-step grouping.

    Environments may be at different sub-steps if dead-end fallbacks occur
    (e.g., cluster selection finds no valid BBs and resets).  To keep policy
    evaluations batched, we partition active environments into up to three
    groups (cycle-picking, cluster-picking, BB-picking) and process each
    group with the appropriate head.

    This replaces the previous implementation that sampled all active
    environments using the sub-step of ``envs[active[0]]``, which silently
    produced wrong log-probabilities when environments were desynchronised.
    """
    envs = [
        HierarchicalMultiHotEnv(
            sizes, pool_sizes, cluster_assignments, device,
            allowed_per_cycle=allowed_per_cycle, max_clusters=max_clusters,
        )
        for _ in range(batch_trajectories)
    ]
    for e in envs:
        e.reset()

    done = [False] * batch_trajectories
    actions_taken: List[List[Tuple[int, int, int]]] = [[] for _ in range(batch_trajectories)]
    logps_taken: List[List[torch.Tensor]] = [[] for _ in range(batch_trajectories)]

    T = sum(sizes)

    # Main loop — 3*T sub-steps is enough for any trajectory
    for _step in range(3 * T):
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break

        # ---- Partition active environments by their current sub-step ----
        group_cycle: List[int] = []    # need to pick cycle
        group_cluster: List[int] = []  # need to pick cluster
        group_bb: List[int] = []       # need to pick BB

        for gi in active:
            env = envs[gi]
            if env.cycle_picked is None:
                group_cycle.append(gi)
            elif env.cluster_picked is None:
                group_cluster.append(gi)
            else:
                group_bb.append(gi)

        # ---- Process cycle-picking group ----
        if group_cycle:
            g_indices = group_cycle
            states = torch.stack([envs[gi].state_vector() for gi in g_indices], dim=0)
            hidden = policy.encode(states.float())
            cycle_logits_all = policy.forward_cycle(hidden)  # [B, 3]
            masks_cycle = torch.stack([envs[gi].valid_cycle_mask() for gi in g_indices], dim=0)
            cycle_logits_all = cycle_logits_all.masked_fill(~masks_cycle, float("-inf"))

            for bi, gi in enumerate(g_indices):
                env = envs[gi]
                mask = masks_cycle[bi]
                valid_indices = torch.where(mask)[0]
                n_valid = len(valid_indices)
                if n_valid == 0:
                    done[gi] = True
                    continue

                traj_logits = cycle_logits_all[bi]
                valid_logits = traj_logits[mask]
                if torch.isfinite(valid_logits).all():
                    cycle, logp = _sample_action_with_epsilon(traj_logits, mask, device, epsilon)
                else:
                    cycle, logp = _sample_uniform(mask, device)

                env.step_cycle(cycle)
                logps_taken[gi].append(logp)

        # ---- Process cluster-picking group ----
        if group_cluster:
            g_indices = group_cluster
            states = torch.stack([envs[gi].state_vector() for gi in g_indices], dim=0)
            hidden = policy.encode(states.float())
            cluster_logits_all = policy.forward_cluster(hidden)  # [B, max_clusters]

            masks_cluster = []
            for gi in g_indices:
                env = envs[gi]
                cycle = env.cycle_picked
                if cycle is None:
                    masks_cluster.append(torch.zeros(policy.max_clusters, device=device, dtype=torch.bool))
                else:
                    masks_cluster.append(env.valid_cluster_mask(cycle))
            masks_cluster = torch.stack(masks_cluster, dim=0)
            cluster_logits_all = cluster_logits_all.masked_fill(~masks_cluster, float("-inf"))

            for bi, gi in enumerate(g_indices):
                env = envs[gi]
                mask = masks_cluster[bi]
                valid_indices = torch.where(mask)[0]
                n_valid = len(valid_indices)
                if n_valid == 0:
                    # No valid clusters — reset cycle and retry
                    env._cycle_picked = None
                    continue

                traj_logits = cluster_logits_all[bi]
                valid_logits = traj_logits[mask]
                if torch.isfinite(valid_logits).all():
                    cluster_id, logp = _sample_action_with_epsilon(traj_logits, mask, device, epsilon)
                else:
                    cluster_id, logp = _sample_uniform(mask, device)

                env.step_cluster(cluster_id)
                logps_taken[gi].append(logp)

        # ---- Process BB-picking group ----
        if group_bb:
            g_indices = group_bb
            states = torch.stack([envs[gi].state_vector() for gi in g_indices], dim=0)
            hidden = policy.encode(states.float())
            bb_logits_all = policy.forward_bb(hidden)  # [B, N]

            masks_bb = []
            for gi in g_indices:
                env = envs[gi]
                cycle = env.cycle_picked
                cluster_id = env.cluster_picked
                if cycle is None or cluster_id is None:
                    masks_bb.append(torch.zeros(policy.N, device=device, dtype=torch.bool))
                else:
                    masks_bb.append(env.valid_bb_mask(cycle, cluster_id))
            masks_bb = torch.stack(masks_bb, dim=0)
            bb_logits_all = bb_logits_all.masked_fill(~masks_bb, float("-inf"))

            for bi, gi in enumerate(g_indices):
                env = envs[gi]
                mask = masks_bb[bi]
                valid_indices = torch.where(mask)[0]
                n_valid = len(valid_indices)
                if n_valid == 0:
                    env._cycle_picked = None
                    env._cluster_picked = None
                    continue

                traj_logits = bb_logits_all[bi]
                valid_logits = traj_logits[mask]
                if torch.isfinite(valid_logits).all():
                    bb_idx, logp = _sample_action_with_epsilon(traj_logits, mask, device, epsilon)
                else:
                    bb_idx, logp = _sample_uniform(mask, device)

                cycle = env.cycle_picked
                cluster_id = env.cluster_picked
                env.step_bb(bb_idx)
                logps_taken[gi].append(logp)
                actions_taken[gi].append((cycle, cluster_id, bb_idx))
                done[gi] = env.is_terminal()

    out = []
    for gi in range(batch_trajectories):
        env = envs[gi]
        B1, B2, B3 = env.terminal_sets()
        out.append({
            "actions": actions_taken[gi],
            "logps": logps_taken[gi],
            "terminal": (B1, B2, B3),
        })
    return out


# ============================================================================
# Main training
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        "Hierarchical (H-DEL-GFlowNet) for DEL library design with Trajectory Balance"
    )
    ap.add_argument("--bbs", default="data/bbs.csv",
                    help="bbs.csv with 'SMILES' (optional 'Name','pool')")
    ap.add_argument("--clusters", type=str, default="data/clusters.json",
                    help="Precomputed cluster assignments JSON from precompute_clusters.py.")
    ap.add_argument("--size1", type=int, default=25, help="Target |B1|")
    ap.add_argument("--size2", type=int, default=25, help="Target |B2|")
    ap.add_argument("--size3", type=int, default=40, help="Target |B3|")
    ap.add_argument("--deepdel", default="models/deepdel.pt",
                    help="Path to DeepDel model (for reward)")

    # Pool limiting — default to paper sizes
    ap.add_argument("--bb-pool-size", type=int, default=None,
                    help="Limit all three pools to the same first N BBs (CSV order). If None, use all.")

    # Features
    ap.add_argument("--bb-fp-bits", type=int, default=2048)
    ap.add_argument("--bb-fp-radius", type=int, default=2)

    # Policy architecture
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-clusters", type=int, default=20,
                    help="Maximum number of clusters across all cycles (paper: 20).")

    # Tempered target
    ap.add_argument("--beta", type=float, default=50.0)
    ap.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)

    # Forbidden BBs
    ap.add_argument("--forbidden-bb1", default=None)
    ap.add_argument("--forbidden-bb2", default=None)
    ap.add_argument("--forbidden-bb3", default=None)

    # Training
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-trajectories", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--logz-lr", type=float, default=1)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Logging / outputs
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--topm", type=int, default=100)
    ap.add_argument("--topm-distance", type=str, default="tanimoto",
                    choices=["tanimoto", "cosine", "euclidean"])

    # Reward source
    ap.add_argument(
        "--gfn-reward-source", choices=["deepdel", "autodock_proxy"], default="deepdel")
    ap.add_argument("--autodock-model", type=str, default=None)
    ap.add_argument("--autodock-proxy-reward", type=str, default="threshold",
                    choices=REWARD_MODES)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--autodock-proxy-batch-size", type=int, default=4096)
    ap.add_argument("--autodock-proxy-device", type=str, default=None)
    ap.add_argument("--max-weight", type=float, default=None)
    ap.add_argument("--weight-source", choices=["smiles", "bb_sum"], default="bb_sum")
    ap.add_argument("--reaction-mode", type=str, default="amide_sulfonamide",
                    choices=["amide_sulfonamide", "amide_amide", "amide_amide_legacy"])
    ap.add_argument("--append-encountered-dataset", type=str, default=None)
    ap.add_argument("--encountered-out", type=str, default=None)

    # CPU / cache
    ap.add_argument("--cpu-threads", type=int, default=None)
    ap.add_argument("--ecfp-cache-dir", type=str, default=None)

    # Resume
    ap.add_argument("--policy-ckpt", type=str, default=None)

    # Logging controls
    ap.add_argument("--log-metrics", type=int, default=10)
    ap.add_argument("--no-svg", type=int, default=1)
    ap.add_argument("--progress", type=int, default=1)
    ap.add_argument("--batched-rollouts", action="store_true", default=True)
    ap.add_argument("--model_is_logprobs", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    # Handle backward-compat temperature alias
    if args.temperature is not None and args.beta == 1.0:
        args.beta = float(args.temperature)
        print(f"[Compat] Using --temperature={args.temperature} as beta.")

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    # CPU thread control
    if device.type == "cpu":
        if args.cpu_threads is not None:
            torch.set_num_threads(max(1, int(args.cpu_threads)))
        elif "SLURM_CPUS_PER_TASK" in os.environ:
            try:
                torch.set_num_threads(
                    max(1, min(32, int(os.environ["SLURM_CPUS_PER_TASK"]) // 4)))
            except Exception:
                pass

    # Enable TF32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        try:
            n_gpu = torch.cuda.device_count()
            cur = torch.cuda.current_device()
            print(f"[CUDA] visible devices = {n_gpu} (using cuda:{cur} = "
                  f"{torch.cuda.get_device_name(cur)})")
            if n_gpu > 1:
                print(f"[CUDA] WARNING: single-GPU script; {n_gpu-1} device(s) will sit idle.")
        except Exception as e:
            print(f"[CUDA] WARN: could not query CUDA devices: {e}")

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    recorder = MetricsRecorder(args.outdir, ema_alpha=0.98) if args.log_metrics else None

    # ---- Load BBs ----
    df_bbs = load_bbs(args.bbs)
    smiles = df_bbs["SMILES"].tolist()
    N_full = len(smiles)
    id_map = df_bbs["ID"].to_numpy() if "ID" in df_bbs.columns else None

    # ---- Load cluster assignments ----
    cluster_assignments = load_clusters(args.clusters)
    print(f"[Clusters] Loaded cluster assignments from {args.clusters}")
    for c in (1, 2, 3):
        n_bb = len(cluster_assignments[c])
        n_cl = len(set(cluster_assignments[c].values()))
        print(f"  Cycle {c}: {n_bb} BBs → {n_cl} clusters")

    # ---- Autodock proxy reward oracle (optional) ----
    autodock_reward_oracle: Optional[AutodockProxyRewardOracle] = None
    if args.gfn_reward_source == "autodock_proxy":
        if not args.autodock_model:
            raise ValueError("--autodock-model is required when --gfn-reward-source=autodock_proxy")
        if args.autodock_proxy_reward == "threshold" and args.threshold is None:
            raise ValueError("--threshold is required when --autodock-proxy-reward=threshold")
        autodock_reward_oracle = AutodockProxyRewardOracle(
            bbs_df=df_bbs,
            id_map=id_map,
            autodock_model=args.autodock_model,
            reward_mode=args.autodock_proxy_reward,
            threshold=args.threshold,
            alpha=args.alpha,
            k=args.k,
            batch_size=args.autodock_proxy_batch_size,
            device=args.autodock_proxy_device or args.device,
            reaction_mode=args.reaction_mode,
            max_weight=args.max_weight,
            weight_source=args.weight_source,
        )

    # ---- Pool-aware action universe ----
    sizes = (args.size1, args.size2, args.size3)
    if any(s <= 0 for s in sizes):
        raise ValueError("All sizes must be positive.")

    pool_arr = df_bbs["pool"].to_numpy() if "pool" in df_bbs.columns else np.zeros(N_full, dtype=int)
    allowed_lists: Optional[List[List[int]]] = None
    if np.any(pool_arr != 0):
        idx1 = np.where(pool_arr == 1)[0].tolist()
        idx2 = np.where(pool_arr == 2)[0].tolist()
        idx3 = np.where(pool_arr == 3)[0].tolist()
        if not (idx1 and idx2 and idx3):
            raise ValueError(
                f"Pool column present in {args.bbs}, but at least one pool is empty: "
                f"|pool1|={len(idx1)}, |pool2|={len(idx2)}, |pool3|={len(idx3)}."
            )
        allowed_lists = [idx1, idx2, idx3]
        print(f"[Pools] Pool-aware: |pool1|={len(idx1)}, |pool2|={len(idx2)}, |pool3|={len(idx3)}")
    else:
        print(f"[Pools] No pool column; sampling from full BB universe (N={N_full}).")

    # ---- Forbidden BBs ----
    id_to_idx: Optional[Dict[int, int]] = None
    if id_map is not None:
        id_to_idx = {int(bb_id): int(i) for i, bb_id in enumerate(id_map.tolist())}

    forbidden_idx_per_cycle: Dict[int, Set[int]] = {1: set(), 2: set(), 3: set()}
    forbidden_ids_raw = {
        1: _parse_pipe_separated_ints(args.forbidden_bb1),
        2: _parse_pipe_separated_ints(args.forbidden_bb2),
        3: _parse_pipe_separated_ints(args.forbidden_bb3),
    }
    for cycle, raw_ids in forbidden_ids_raw.items():
        if not raw_ids:
            continue
        if id_to_idx is None:
            raise ValueError("Forbidden BB flags require the BB CSV to include an ID column.")
        found: Set[int] = set()
        missing: List[int] = []
        for bb_id in raw_ids:
            idx = id_to_idx.get(int(bb_id))
            if idx is None:
                missing.append(int(bb_id))
                continue
            forbidden_idx_per_cycle[cycle].add(int(idx))
            found.add(int(bb_id))
        if found:
            print(f"[Forbidden] Pool {cycle}: excluding {len(found)} BB(s).")
        if missing:
            print(f"[Forbidden] Pool {cycle}: {len(missing)} ID(s) not found: {sorted(set(missing))}")

    if any(forbidden_idx_per_cycle[c] for c in (1, 2, 3)):
        if allowed_lists is None:
            allowed_lists = [list(range(N_full)), list(range(N_full)), list(range(N_full))]
        for cycle in (1, 2, 3):
            forb = forbidden_idx_per_cycle[cycle]
            if not forb:
                continue
            before = len(allowed_lists[cycle - 1])
            allowed_lists[cycle - 1] = [idx for idx in allowed_lists[cycle - 1] if idx not in forb]
            removed = before - len(allowed_lists[cycle - 1])
            print(f"[Forbidden] Pool {cycle}: removed {removed} BB(s); "
                  f"remaining={len(allowed_lists[cycle - 1])}.")
        for cycle in (1, 2, 3):
            remaining = len(allowed_lists[cycle - 1])
            need = sizes[cycle - 1]
            if remaining < need:
                raise ValueError(
                    f"After forbidden BBs, pool {cycle} only has {remaining} candidate(s) "
                    f"but size{cycle}={need}."
                )

    # ---- Apply pool size limit (--bb-pool-size) ----
    if args.bb_pool_size is not None:
        bb_pool_size = int(args.bb_pool_size)
        if N_full < bb_pool_size:
            raise ValueError(
                f"bbs.csv has {N_full} rows but --bb-pool-size={bb_pool_size} "
                f"requires at least {bb_pool_size}."
            )
        if allowed_lists is None:
            allowed_lists = [
                list(range(0, bb_pool_size)),
                list(range(0, bb_pool_size)),
                list(range(0, bb_pool_size)),
            ]
            print(f"[Pools] No pool column — all three pools share the same first {bb_pool_size} BBs: "
                  f"pool1=0:{bb_pool_size}, pool2=0:{bb_pool_size}, pool3=0:{bb_pool_size}")
        else:
            for cycle in (1, 2, 3):
                before = len(allowed_lists[cycle - 1])
                allowed_lists[cycle - 1] = allowed_lists[cycle - 1][:bb_pool_size]
                after = len(allowed_lists[cycle - 1])
                need = sizes[cycle - 1]
                if after < need:
                    raise ValueError(
                        f"--bb-pool-size={bb_pool_size} leaves only {after} candidate(s) "
                        f"in pool {cycle} but size{cycle}={need}."
                    )
                print(f"[Pools] Truncated pool {cycle}: {before} -> {after} (bb_pool_size={bb_pool_size})")

    if allowed_lists is not None:
        allowed_per_cycle: Optional[Tuple[List[int], List[int], List[int]]] = (
            allowed_lists[0], allowed_lists[1], allowed_lists[2],
        )
        pool_sizes = (len(allowed_lists[0]), len(allowed_lists[1]), len(allowed_lists[2]))
    else:
        allowed_per_cycle = None
        pool_sizes = (N_full, N_full, N_full)

    N_total = sum(pool_sizes)
    print(f"[Hierarchical] State dimension: N = {N_total} (pool sizes: {pool_sizes}), "
          f"augmented = {N_total + 3 + args.max_clusters + 2}")

    # ---- ECFP cache for diversity diagnostics ----
    cache_dir = args.ecfp_cache_dir or os.path.join(args.outdir, "cache")
    bb_fp_bits = int(args.bb_fp_bits)
    bb_ecfp = load_or_create_ecfp_cache(
        smiles,
        n_bits=bb_fp_bits,
        radius=args.bb_fp_radius,
        cache_path=os.path.join(cache_dir, f"ecfp{bb_fp_bits}_r{args.bb_fp_radius}.npz"),
        progress_desc=f"ECFP({bb_fp_bits}) for diversity",
        progress_enabled=False,
    )
    bb_ecfp_table = torch.from_numpy(bb_ecfp).to(device)

    # ---- Load frozen TripleDeepSet checkpoint ----
    ckpt = torch.load(args.deepdel, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    d_in = ckpt_args.get("bb_fp_bits", 2048)
    d_h = ckpt_args.get("hidden_dim", 256)
    d_rho = ckpt_args.get("rho_dim", 256)
    dropout_val = ckpt_args.get("dropout", 0.1)
    shared_phi = ckpt_args.get("shared_phi", False)
    pooling = ckpt_args.get("pooling", "mean")
    output_head = ckpt_args.get("output_head", "linear")
    lib_size = ckpt_args.get("lib_size", None)
    log_target = bool(ckpt_args.get("log_target", False))

    triple_model = TripleDeepSet(
        d_in=d_in, d_hidden=d_h, d_rho=d_rho, dropout=dropout_val,
        shared_phi=shared_phi, pooling=pooling,
        output_head=output_head, reward_bound_k=lib_size,
    )
    triple_model.load_state_dict(ckpt["model_state"])
    print(f"[Info] Loaded DeepDEL TripleDeepSet pooling={pooling} output_head={output_head}")
    triple_model.to(device).eval()
    for p in triple_model.parameters():
        p.requires_grad_(False)

    # Build model-input fingerprints for reward computation
    print(f"Building model-input fingerprints for reward (d_in={d_in}, radius={args.bb_fp_radius}) ...")
    X_in = load_or_create_ecfp_cache(
        smiles,
        n_bits=int(d_in),
        radius=args.bb_fp_radius,
        cache_path=os.path.join(cache_dir, f"ecfp{int(d_in)}_r{args.bb_fp_radius}.npz"),
        progress_desc=f"ECFP({d_in}) for reward",
        progress_enabled=False,
    )
    X_all = torch.from_numpy(X_in).to(device)

    # ---- Policy ----
    augmented_dim = N_total + 3 + args.max_clusters + 2
    policy = HierarchicalPolicy(
        state_dim=augmented_dim,
        N=N_total,
        max_clusters=args.max_clusters,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    logZ = nn.Parameter(torch.tensor(0.0, device=device))

    # ---- Optional resume ----
    if args.policy_ckpt is not None and os.path.exists(args.policy_ckpt):
        try:
            pck = torch.load(args.policy_ckpt, map_location="cpu")
            pck_state_repr = pck.get("state_repr", pck.get("meta", {}).get("state_repr"))
            if pck_state_repr is not None and str(pck_state_repr) != "hierarchical":
                raise ValueError(
                    f"Checkpoint state_repr mismatch: checkpoint={pck_state_repr!r}, "
                    f"expected='hierarchical'."
                )
            state = pck.get("policy_state", None)
            if state is not None:
                missing, unexpected = policy.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[Resume] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            policy = policy.to(device)
            if "logZ" in pck:
                with torch.no_grad():
                    logZ.data = torch.tensor(float(pck["logZ"]), device=device)
                print(f"[Resume] logZ set to {logZ.item():.4f}")
            print(f"[Resume] Loaded checkpoint from {args.policy_ckpt}")
        except Exception as e:
            raise RuntimeError(f"[Resume] ERROR loading {args.policy_ckpt}: {e}") from e
    elif args.policy_ckpt is not None:
        print(f"[Resume] No existing checkpoint at {args.policy_ckpt}; starting from scratch.")

    # ---- Optimizer ----
    opt = torch.optim.Adam(
        [
            {"params": policy.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [logZ], "lr": args.logz_lr},
        ]
    )
    n_policy_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Params] policy={n_policy_params:,} logZ=1 total_trainable={n_policy_params+1:,}")

    # ---- Training loop ----
    T = args.size1 + args.size2 + args.size3
    print(f"[Info] Steps per trajectory = {T} (3 sub-steps each = {3*T} total), "
          f"N_state_augmented = {augmented_dim}")
    if device.type == "cuda":
        print(f"[CUDA] AMP enabled: {use_amp} | "
              f"TF32: matmul={torch.backends.cuda.matmul.allow_tf32}")

    ema_loss = None
    rng = np.random.default_rng(args.seed if args.seed is not None else None)

    rewards_seen: List[float] = []
    yhats_seen: List[float] = []
    topm_records: List[Dict[str, object]] = []
    encountered_rows: List[Dict[str, object]] = []

    for step in range(1, args.steps + 1):
        batch_logps_sum = []
        batch_beta_logR = []
        batch_rewards_this_step: List[float] = []
        terminals_in_batch = 0

        if args.batched_rollouts:
            trajs = rollout_trajectories_batched_hierarchical(
                sizes=sizes,
                pool_sizes=pool_sizes,
                cluster_assignments=cluster_assignments,
                policy=policy,
                rng=rng,
                batch_trajectories=args.batch_trajectories,
                device=device,
                epsilon=args.epsilon,
                allowed_per_cycle=allowed_per_cycle,
                max_clusters=args.max_clusters,
            )
        else:
            trajs = []
            for _ in range(args.batch_trajectories):
                env = HierarchicalMultiHotEnv(
                    sizes, pool_sizes, cluster_assignments, device,
                    allowed_per_cycle=allowed_per_cycle,
                    max_clusters=args.max_clusters,
                )
                traj = rollout_trajectory_hierarchical(
                    env, policy, rng, device, epsilon=args.epsilon,
                )
                if len(traj["logps"]) == 0:
                    continue
                trajs.append(traj)

        for traj in trajs:
            if len(traj["logps"]) == 0:
                continue

            logps_sum = torch.stack(traj["logps"]).sum()

            if args.gfn_reward_source == "autodock_proxy":
                assert autodock_reward_oracle is not None
                r, log_r_float, meta = autodock_reward_oracle.evaluate(traj["terminal"])
                if not (math.isfinite(r) and math.isfinite(log_r_float)):
                    continue
                log_R = torch.tensor(log_r_float, dtype=torch.float32, device=device)
                R = torch.tensor(r, dtype=torch.float32, device=device)
            else:
                log_R, R = reward_from_triple(
                    traj["terminal"], X_all, triple_model, device,
                    log_target=log_target,
                    model_is_logprobs=args.model_is_logprobs,
                )
                r = float(R.item())
                if id_map is not None:
                    B1_tmp, B2_tmp, B3_tmp = traj["terminal"]
                    meta = {
                        "B1_id": "|".join(str(int(id_map[i])) for i in B1_tmp),
                        "B2_id": "|".join(str(int(id_map[i])) for i in B2_tmp),
                        "B3_id": "|".join(str(int(id_map[i])) for i in B3_tmp),
                    }
                else:
                    meta = {}

            beta_logR = args.beta * log_R
            batch_logps_sum.append(logps_sum)
            batch_beta_logR.append(beta_logR)
            r = float(R.item())
            yv = float((-log_R).item())
            rewards_seen.append(r)
            terminals_in_batch += 1
            batch_rewards_this_step.append(r)
            yhats_seen.append(yv)
            B1, B2, B3 = traj["terminal"]
            rec = {"reward": r, "yhat": yv, "B1": B1, "B2": B2, "B3": B3, **meta}
            topm_records.append(rec)
            if args.gfn_reward_source == "autodock_proxy":
                encountered_rows.append({
                    "rank": len(encountered_rows) + 1,
                    "reward": r,
                    "log_reward": float(log_R.item()),
                    "autodock_proxy_value": r,
                    "proxy_reward": np.nan,
                    "proxy_yhat": np.nan,
                    "B1_id": meta.get("B1_id", ""),
                    "B2_id": meta.get("B2_id", ""),
                    "B3_id": meta.get("B3_id", ""),
                    "autodock_proxy_values": meta.get("autodock_proxy_values", ""),
                    "reward_source": "autodock_proxy",
                })
            if len(topm_records) > args.topm * 5:
                topm_records = sorted(
                    topm_records, key=lambda t: float(t["reward"]), reverse=True
                )[:args.topm]

        if not batch_logps_sum:
            continue

        batch_logps_sum = torch.stack(batch_logps_sum)
        batch_beta_logR = torch.stack(batch_beta_logR)

        tb = tb_loss(logZ, batch_logps_sum, batch_beta_logR)
        loss = tb

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        tb_v = float(tb.item())
        total_v = float(loss.item())
        log1p_v = math.log1p(tb_v)
        ema_loss = log1p_v if ema_loss is None else 0.98 * ema_loss + 0.02 * log1p_v

        if recorder is not None:
            recorder.update(
                step=step,
                log1p_tb_loss=log1p_v,
                logZ=logZ.item(),
                batch_rewards=batch_rewards_this_step,
                terminals_in_batch=terminals_in_batch,
                total_loss=total_v,
            )

        if step % args.log_interval == 0:
            if batch_rewards_this_step:
                br = np.asarray(batch_rewards_this_step, dtype=np.float64)
                br_mean, br_std = float(br.mean()), float(br.std())
                if recorder is None or recorder._ema_reward is None:
                    ema_str = "nan"
                else:
                    ema_str = f"{float(recorder._ema_reward):.3g}"
                br_msg = f" | batchR={br_mean:.3g}±{br_std:.3g} [EMA={ema_str}]"
            else:
                br_msg = " | batchR=nan nT=0"
            top_str = "nan"
            if topm_records:
                top_str = f"{np.log(max(max(float(t['reward']) for t in topm_records), 1e-30)):.3g}"
            progress_write(
                f"[Step {step}/{args.steps}] log(1+TB loss)={log1p_v:.6f} | "
                f"EMA={ema_loss:.6f} | logZ={logZ.item():.3f}"
                + f" | top={top_str}" + br_msg
            )

        # Save checkpoint at end
        if args.save_model and step == args.steps:
            ck = {
                "policy_state": policy.state_dict(),
                "logZ": float(logZ.item()),
                "args": vars(args),
                "state_repr": "hierarchical",
                "augmented_dim": int(augmented_dim),
                "N_total": int(N_total),
                "meta": {
                    "state_repr": "hierarchical",
                    "augmented_dim": int(augmented_dim),
                    "N_total": int(N_total),
                    "hidden_dim": int(args.hidden_dim),
                    "n_layers": int(args.n_layers),
                    "max_clusters": int(args.max_clusters),
                    "T": T,
                    "pool_sizes": list(pool_sizes),
                },
            }
            policypath = os.path.join(args.outdir, "gfn_policy.pt")
            torch.save(ck, policypath)
            print(f"Saved gfn policy at {policypath}")

    print("Training complete.")

    # ---- End-of-training analytics ----
    if len(yhats_seen) > 0:
        hist_path = os.path.join(args.outdir, "yhat_hist.png")
        vals = np.array(yhats_seen, dtype=np.float64)
        plt.figure(figsize=(10, 5))
        plt.hist(vals, bins=50)
        plt.title("Histogram of yhat (= -log R)")
        plt.xlabel("yhat")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=180)
        plt.close()
        print(f"[Summary] Saved yhat histogram to {hist_path}")

    # Top-k
    if topm_records:
        topk_sorted = sorted(
            topm_records, key=lambda t: float(t["reward"]), reverse=True
        )[:args.topm]

        def map_to_ids(xs):
            return "|".join(str(int(id_map[i])) for i in xs)

        rows = []
        for i, rec in enumerate(topk_sorted, 1):
            r = float(rec["reward"])
            yv = float(rec["yhat"])
            B1 = rec["B1"]; B2 = rec["B2"]; B3 = rec["B3"]
            row = {"rank": i, "reward": r, "yhat": yv}
            if id_map is not None:
                row.update({
                    "B1_id": map_to_ids(B1),
                    "B2_id": map_to_ids(B2),
                    "B3_id": map_to_ids(B3),
                })
            rows.append(row)

        topk_df = pd.DataFrame(rows)
        topk_csv = os.path.join(args.outdir, "topm_rewards.csv")
        topk_df.to_csv(topk_csv, index=False)
        print(f"[Top-{args.topm}] Saved to {topk_csv}")

    if args.gfn_reward_source == "autodock_proxy" and encountered_rows:
        encountered_df = pd.DataFrame(encountered_rows)
        if args.encountered_out:
            os.makedirs(os.path.dirname(args.encountered_out), exist_ok=True)
            encountered_df.to_csv(args.encountered_out, index=False)
        if args.append_encountered_dataset:
            append_df = encountered_df[["B1_id", "B2_id", "B3_id", "log_reward"]].rename(
                columns={"log_reward": "y"})
            header_needed = not os.path.exists(args.append_encountered_dataset)
            os.makedirs(os.path.dirname(args.append_encountered_dataset), exist_ok=True)
            append_df.to_csv(args.append_encountered_dataset, mode="a",
                             header=header_needed, index=False)


if __name__ == "__main__":
    main()