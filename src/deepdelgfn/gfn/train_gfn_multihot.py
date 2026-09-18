#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-hot (flat) GFlowNet for DEL library design, following the state
representation of Koziarski et al. (2024) "Towards DNA-Encoded Library
Generation with GFlowNets".

State representation:
    A binary vector x = x1 | x2 | x3 ∈ {0,1}^{N1+N2+N3}, where each entry
    indicates whether the corresponding building block is selected for the
    library. The library is the Cartesian product of selected blocks across
    cycles.

Action space:
    Flat index i ∈ {0, ..., N-1} denoting "flip bit i from 0 to 1".
    The trajectory terminates when each cycle has reached its target size
    (s1, s2, s3).

Policy:
    An MLP that maps the binary state vector to logits over all N actions.
    Invalid actions (already selected, or pool full) are masked to -inf
    before sampling.

Reward:
    Same as the deepsets GFN: the frozen DeepDEL TripleDeepSet model
    (or the autodock_proxy oracle), ensuring a fair comparison.

Training:
    Trajectory Balance (TB) loss:
        L_TB = (logZ + Σ_t log π(a_t|s_{t-1}) - β * log R(x))^2

No action subsampling is used; the full valid action set is evaluated at
each step. For large pools, use --max-pool{1,2,3} to limit the number of
building blocks considered per cycle.
"""

import os
import argparse
import random
import time
import math
import csv
import json
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from contextlib import nullcontext

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
    reward_from_triple,
    AutodockProxyRewardOracle,
    MetricsRecorder,
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
# Multi-hot Environment
# ============================================================================

class MultiHotEnv:
    """Binary-vector DEL environment.

    State: x ∈ {0,1}^{N} where N = N1+N2+N3 (truncated pool sizes).
    Position layout:
        [0, N1)           -> pool 1 (cycle 1)
        [N1, N1+N2)       -> pool 2 (cycle 2)
        [N1+N2, N1+N2+N3) -> pool 3 (cycle 3)

    Actions: flat index i ∈ {0, ..., N-1} that flips x[i] from 0 to 1.
    Terminal: count1==s1 and count2==s2 and count3==s3.
    """

    def __init__(
        self,
        sizes: Tuple[int, int, int],
        pool_sizes: Tuple[int, int, int],
        device: torch.device,
        allowed_per_cycle: Optional[Tuple[List[int], List[int], List[int]]] = None,
    ):
        self.s1, self.s2, self.s3 = sizes
        self.device = device

        # Pool sizes (after truncation)
        self.n1, self.n2, self.n3 = pool_sizes
        self.N = self.n1 + self.n2 + self.n3

        # Offsets for each cycle's block within the flat vector
        self.off1 = 0
        self.off2 = self.n1
        self.off3 = self.n1 + self.n2

        # Allowed BB row indices per cycle (maps positions -> original BB indices)
        if allowed_per_cycle is None:
            self.allowed1 = list(range(self.n1))
            self.allowed2 = list(range(self.n2))
            self.allowed3 = list(range(self.n3))
        else:
            self.allowed1 = list(allowed_per_cycle[0])
            self.allowed2 = list(allowed_per_cycle[1])
            self.allowed3 = list(allowed_per_cycle[2])

        if self.s1 > len(self.allowed1):
            raise ValueError(f"size1={self.s1} exceeds pool-1 size={len(self.allowed1)}")
        if self.s2 > len(self.allowed2):
            raise ValueError(f"size2={self.s2} exceeds pool-2 size={len(self.allowed2)}")
        if self.s3 > len(self.allowed3):
            raise ValueError(f"size3={self.s3} exceeds pool-3 size={len(self.allowed3)}")

        self.reset()

    def reset(self):
        self.x = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0

    def is_terminal(self) -> bool:
        return self.count1 == self.s1 and self.count2 == self.s2 and self.count3 == self.s3

    def state_vector(self) -> torch.Tensor:
        """Return the binary state vector [N]."""
        return self.x

    def valid_action_mask(self) -> torch.Tensor:
        """Return a boolean mask [N] of valid actions (can flip 0->1)."""
        mask = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        # Pool 1
        if self.count1 < self.s1:
            s1_slice = slice(self.off1, self.off1 + self.n1)
            mask[s1_slice] = (self.x[s1_slice] == 0)
        # Pool 2
        if self.count2 < self.s2:
            s2_slice = slice(self.off2, self.off2 + self.n2)
            mask[s2_slice] = (self.x[s2_slice] == 0)
        # Pool 3
        if self.count3 < self.s3:
            s3_slice = slice(self.off3, self.off3 + self.n3)
            mask[s3_slice] = (self.x[s3_slice] == 0)
        return mask

    def step(self, action_idx: int):
        """Flip bit action_idx from 0 to 1."""
        if self.x[action_idx] != 0:
            raise ValueError(f"Invalid action: bit {action_idx} is already 1.")
        self.x[action_idx] = 1.0
        if action_idx < self.off2:
            self.count1 += 1
        elif action_idx < self.off3:
            self.count2 += 1
        else:
            self.count3 += 1

    def terminal_sets(self) -> Tuple[List[int], List[int], List[int]]:
        """Map positions back to original BB row indices for reward computation."""
        B1 = []
        B2 = []
        B3 = []
        for i in range(self.n1):
            if self.x[self.off1 + i] == 1:
                B1.append(self.allowed1[i])
        for i in range(self.n2):
            if self.x[self.off2 + i] == 1:
                B2.append(self.allowed2[i])
        for i in range(self.n3):
            if self.x[self.off3 + i] == 1:
                B3.append(self.allowed3[i])
        B1.sort(); B2.sort(); B3.sort()
        return B1, B2, B3


# ============================================================================
# Multi-hot Policy (MLP)
# ============================================================================

class MultiHotPolicy(nn.Module):
    """MLP policy: binary state vector -> logits over all actions.

    Architecture: n_layers of [Linear(N, hidden) -> ReLU] then Linear(hidden, N).
    The first layer maps from N (state dim) to hidden_dim; intermediate layers
    are hidden->hidden; the final layer maps hidden->N (logits).
    """

    def __init__(
        self,
        n_state: int,
        hidden_dim: int = 512,
        n_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_state = n_state
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers: List[nn.Module] = []
        in_dim = n_state
        for i in range(n_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        # Final linear head: hidden -> N (logits over actions)
        layers.append(nn.Linear(in_dim, n_state))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [B, N] binary float tensor
        returns: [B, N] logits over all actions
        """
        return self.net(state)


# ============================================================================
# Rollout functions
# ============================================================================

def rollout_trajectory_multihot(
    env: MultiHotEnv,
    policy: MultiHotPolicy,
    rng: np.random.Generator,
    device: torch.device,
    epsilon: float = 0.05,
) -> Dict:
    """Roll out a single trajectory using the full valid action set.

    Returns:
        actions: list[int] (flat action indices)
        logps: list[torch.Tensor] (per-step log forward probabilities)
        terminal: (B1, B2, B3) lists of original BB indices
    """
    env.reset()
    actions, logps = [], []

    while not env.is_terminal():
        s = env.state_vector().unsqueeze(0)  # [1, N]
        mask = env.valid_action_mask().unsqueeze(0)  # [1, N]

        # Policy forward in FP32
        logits = policy(s.float()).squeeze(0)  # [N]
        # Mask invalid actions
        logits = logits.masked_fill(~mask, float("-inf"))

        # Sanitize: if any NaN/Inf in valid positions, fall back to uniform
        valid_logits = logits[mask]
        if not torch.isfinite(valid_logits).all():
            print("[NaNGuard] non-finite logits in multihot rollout; falling back to uniform")
            # Uniform over valid actions
            log_probs = mask.float() / mask.float().sum()
            idx = torch.multinomial(log_probs, 1).item()
            logp = torch.log(log_probs[idx])
        else:
            # ε-greedy
            n_valid = int(mask.sum().item())
            if torch.rand((), device=device).item() < epsilon:
                # Random valid action
                valid_indices = torch.where(mask)[0]
                idx = valid_indices[torch.randint(n_valid, (1,), device=device)].item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                idx = dist.sample().item()

            # log π(a|s) = logits[a] - logsumexp(logits[valid])
            logp = logits[idx] - torch.logsumexp(logits[mask], dim=-1)

        actions.append(idx)
        logps.append(logp)
        env.step(idx)

    B1, B2, B3 = env.terminal_sets()
    return {
        "actions": actions,
        "logps": logps,
        "terminal": (B1, B2, B3),
    }


def rollout_trajectories_batched_multihot(
    *,
    sizes: Tuple[int, int, int],
    pool_sizes: Tuple[int, int, int],
    policy: MultiHotPolicy,
    rng: np.random.Generator,
    batch_trajectories: int,
    device: torch.device,
    epsilon: float = 0.05,
    allowed_per_cycle: Optional[Tuple[List[int], List[int], List[int]]] = None,
) -> List[Dict]:
    """Roll out a batch of trajectories with batched policy evaluation.

    At each step, all active environments' states are stacked into [B', N],
    the policy produces [B', N] logits in one forward pass, and actions are
    sampled per-environment after masking.
    """
    envs = [
        MultiHotEnv(sizes, pool_sizes, device, allowed_per_cycle=allowed_per_cycle)
        for _ in range(batch_trajectories)
    ]
    for e in envs:
        e.reset()

    done = [False] * batch_trajectories
    actions_taken: List[List[int]] = [[] for _ in range(batch_trajectories)]
    logps_taken: List[List[torch.Tensor]] = [[] for _ in range(batch_trajectories)]

    T = sum(sizes)
    N = pool_sizes[0] + pool_sizes[1] + pool_sizes[2]

    for _t in range(T):
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break

        # Stack states and masks for active envs
        states = torch.stack([envs[gi].state_vector() for gi in active], dim=0)  # [B', N]
        masks = torch.stack([envs[gi].valid_action_mask() for gi in active], dim=0)  # [B', N]

        # Policy forward
        logits = policy(states.float())  # [B', N]
        # Mask invalid actions
        logits = logits.masked_fill(~masks, float("-inf"))

        # Sample per active trajectory
        for bi, gi in enumerate(active):
            env = envs[gi]
            mask = masks[bi]
            traj_logits = logits[bi]

            valid_indices = torch.where(mask)[0]
            n_valid = len(valid_indices)
            if n_valid == 0:
                done[gi] = True
                continue

            valid_logits = traj_logits[mask]
            if not torch.isfinite(valid_logits).all():
                print("[NaNGuard] non-finite logits in batched multihot; uniform fallback")
                # Uniform
                log_probs = mask.float() / n_valid
                idx = torch.multinomial(log_probs, 1).item()
                logp = torch.log(log_probs[idx])
            else:
                if torch.rand((), device=device).item() < epsilon:
                    idx = valid_indices[torch.randint(n_valid, (1,), device=device)].item()
                else:
                    dist = torch.distributions.Categorical(logits=traj_logits)
                    idx = dist.sample().item()
                logp = traj_logits[idx] - torch.logsumexp(traj_logits[mask], dim=-1)

            actions_taken[gi].append(idx)
            logps_taken[gi].append(logp)
            env.step(idx)
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
        "Multi-hot (flat) GFlowNet for DEL library design with Trajectory Balance"
    )
    ap.add_argument("--bbs", default="data/bbs.csv", help="bbs.csv with 'SMILES' (optional 'Name','pool')")
    ap.add_argument("--size1", type=int, default=6, help="Target |B1|")
    ap.add_argument("--size2", type=int, default=6, help="Target |B2|")
    ap.add_argument("--size3", type=int, default=6, help="Target |B3|")
    ap.add_argument("--deepdel", default="models/deepdel.pt", help="Path to DeepDel model (for reward)")

    # Pool limiting
    ap.add_argument("--bb-pool-size", type=int, default=None, help="Limit all three pools to the same first N BBs (CSV order). If None, use all.")

    # Features (for reward model input)
    ap.add_argument("--bb-fp-bits", type=int, default=2048, help="ECFP bits for DeepDEL model input.")
    ap.add_argument("--bb-fp-radius", type=int, default=2)

    # Policy architecture
    ap.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension of the MLP policy.")
    ap.add_argument("--n-layers", type=int, default=5, help="Number of hidden layers in the MLP policy.")
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in the MLP policy.")

    # Tempered target
    ap.add_argument("--beta", type=float, default=50.0, help="Reward exponent: target p(x) ∝ R(x)^beta.")
    ap.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)

    # Forbidden BBs
    ap.add_argument("--forbidden-bb1", default=None, help="Pipe-delimited BB IDs that pool 1 must never select.")
    ap.add_argument("--forbidden-bb2", default=None, help="Pipe-delimited BB IDs that pool 2 must never select.")
    ap.add_argument("--forbidden-bb3", default=None, help="Pipe-delimited BB IDs that pool 3 must never select.")

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
    ap.add_argument("--topm", type=int, default=100, help="Report and save top-m terminal states by reward.")
    ap.add_argument("--topm-distance", type=str, default="tanimoto", choices=["tanimoto", "cosine", "euclidean"])

    # Reward source
    ap.add_argument(
        "--gfn-reward-source",
        choices=["deepdel", "autodock_proxy"],
        default="deepdel",
        help="Terminal reward source for TB.",
    )
    ap.add_argument("--autodock-model", type=str, default=None)
    ap.add_argument("--autodock-proxy-reward", type=str, default="threshold", choices=REWARD_MODES)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--autodock-proxy-batch-size", type=int, default=4096)
    ap.add_argument("--autodock-proxy-device", type=str, default=None)
    ap.add_argument("--max-weight", type=float, default=None)
    ap.add_argument("--weight-source", choices=["smiles", "bb_sum"], default="bb_sum")
    ap.add_argument("--reaction-mode", type=str, default="amide_sulfonamide", choices=["amide_sulfonamide", "amide_amide", "amide_amide_legacy"])
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
    ap.add_argument("--batched-rollouts", action="store_true", default=True, help="Batch policy evaluation (always on for multihot).")
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
                torch.set_num_threads(max(1, min(32, int(os.environ["SLURM_CPUS_PER_TASK"]) // 4)))
            except Exception:
                pass

    # Enable TF32 on Ampere+
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        try:
            n_gpu = torch.cuda.device_count()
            cur = torch.cuda.current_device()
            print(f"[CUDA] visible devices = {n_gpu} (using cuda:{cur} = {torch.cuda.get_device_name(cur)})")
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
    N = len(smiles)
    id_map = df_bbs["ID"].to_numpy() if "ID" in df_bbs.columns else None

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

    pool_arr = df_bbs["pool"].to_numpy() if "pool" in df_bbs.columns else np.zeros(N, dtype=int)
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
        print(f"[Pools] No pool column; sampling from full BB universe (N={N}).")

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
            allowed_lists = [list(range(N)), list(range(N)), list(range(N))]
        for cycle in (1, 2, 3):
            forb = forbidden_idx_per_cycle[cycle]
            if not forb:
                continue
            before = len(allowed_lists[cycle - 1])
            allowed_lists[cycle - 1] = [idx for idx in allowed_lists[cycle - 1] if idx not in forb]
            removed = before - len(allowed_lists[cycle - 1])
            print(f"[Forbidden] Pool {cycle}: removed {removed} BB(s); remaining={len(allowed_lists[cycle - 1])}.")
        for cycle in (1, 2, 3):
            remaining = len(allowed_lists[cycle - 1])
            need = sizes[cycle - 1]
            if remaining < need:
                raise ValueError(
                    f"After forbidden BBs, pool {cycle} only has {remaining} candidate(s) but size{cycle}={need}."
                )

    # ---- Apply pool size limit (--bb-pool-size) ----
    if args.bb_pool_size is not None:
        bb_pool_size = int(args.bb_pool_size)
        if N < bb_pool_size:
            raise ValueError(
                f"bbs.csv has {N} rows but --bb-pool-size={bb_pool_size} "
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
            allowed_lists[0],
            allowed_lists[1],
            allowed_lists[2],
        )
        pool_sizes = (len(allowed_lists[0]), len(allowed_lists[1]), len(allowed_lists[2]))
    else:
        allowed_per_cycle = None
        pool_sizes = (N, N, N)

    N_total = sum(pool_sizes)
    print(f"[MultiHot] State dimension N = {N_total} (pool sizes: {pool_sizes})")

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
    dropout = ckpt_args.get("dropout", 0.1)
    shared_phi = ckpt_args.get("shared_phi", False)
    pooling = ckpt_args.get("pooling", "mean")
    output_head = ckpt_args.get("output_head", "linear")
    lib_size = ckpt_args.get("lib_size", None)
    log_target = bool(ckpt_args.get("log_target", False))

    triple_model = TripleDeepSet(
        d_in=d_in, d_hidden=d_h, d_rho=d_rho, dropout=dropout,
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
    X_all = torch.from_numpy(X_in).to(device)  # [N, d_in]

    # ---- Policy ----
    policy = MultiHotPolicy(
        n_state=N_total,
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
            if pck_state_repr is not None and str(pck_state_repr) != "multihot":
                raise ValueError(
                    f"Checkpoint state_repr mismatch: checkpoint={pck_state_repr!r}, expected='multihot'."
                )
            pck_n_state = pck.get("n_state", pck.get("meta", {}).get("n_state"))
            if pck_n_state is not None and int(pck_n_state) != int(N_total):
                raise ValueError(
                    f"Checkpoint n_state mismatch: checkpoint={int(pck_n_state)}, current={int(N_total)}."
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
    print(f"[Info] Episode length T = {T}, N_state = {N_total}")
    if device.type == "cuda":
        print(f"[CUDA] AMP enabled: {use_amp} | TF32: matmul={torch.backends.cuda.matmul.allow_tf32}")

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

        t0 = time.time()
        trajs = rollout_trajectories_batched_multihot(
            sizes=sizes,
            pool_sizes=pool_sizes,
            policy=policy,
            rng=rng,
            batch_trajectories=args.batch_trajectories,
            device=device,
            epsilon=args.epsilon,
            allowed_per_cycle=allowed_per_cycle,
        )
        t_rollout = time.time() - t0

        t1 = time.time()
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

            r = float(R.item()); yv = float((-log_R).item())
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
                topm_records = sorted(topm_records, key=lambda t: float(t["reward"]), reverse=True)[:args.topm]
        t_reward = time.time() - t1

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

        # EMA for display
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
                f"[Step {step}/{args.steps}] log(1+TB loss)={log1p_v:.6f} | EMA={ema_loss:.6f} | logZ={logZ.item():.3f}"
                + f" | top={top_str}" + br_msg
            )

        # Save checkpoint at end
        if args.save_model and step == args.steps:
            ck = {
                "policy_state": policy.state_dict(),
                "logZ": float(logZ.item()),
                "args": vars(args),
                "state_repr": "multihot",
                "n_state": int(N_total),
                "meta": {
                    "state_repr": "multihot",
                    "n_state": int(N_total),
                    "hidden_dim": int(args.hidden_dim),
                    "n_layers": int(args.n_layers),
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
    else:
        print("[Summary] No yhat recorded; histogram skipped.")

    # Top-k (by reward)
    if topm_records:
        topk_sorted = sorted(topm_records, key=lambda t: float(t["reward"]), reverse=True)[:args.topm]

        def map_to_ids(xs):
            return "|".join(str(int(id_map[i])) for i in xs)

        rows = []
        for i, rec in enumerate(topk_sorted, 1):
            r = float(rec["reward"]); yv = float(rec["yhat"])
            B1 = rec["B1"]; B2 = rec["B2"]; B3 = rec["B3"]
            if id_map is not None:
                row = {
                    "rank": i,
                    "reward": r,
                    "yhat": yv,
                    "B1_id": map_to_ids(B1),
                    "B2_id": map_to_ids(B2),
                    "B3_id": map_to_ids(B3),
                }
                if args.gfn_reward_source == "autodock_proxy":
                    row.update({
                        "autodock_proxy_value": r,
                        "proxy_reward": np.nan,
                        "proxy_yhat": np.nan,
                        "autodock_proxy_values": rec.get("autodock_proxy_values", ""),
                        "reward_source": "autodock_proxy",
                    })
            rows.append(row)

        topk_df = pd.DataFrame(rows)
        topk_csv = os.path.join(args.outdir, "topm_rewards.csv")
        topk_df.to_csv(topk_csv, index=False)
        print(f"[Top-{args.topm}] Saved to {topk_csv}")

        if args.gfn_reward_source == "autodock_proxy":
            actual_csv = os.path.join(args.outdir, "topm_actual_scores.csv")
            topk_df.to_csv(actual_csv, index=False)
            print(f"[Top-{args.topm}] Saved autodock_proxy rewards to {actual_csv}")
    else:
        print("[Top-m] No terminal records to report.")

    if args.gfn_reward_source == "autodock_proxy" and encountered_rows:
        encountered_df = pd.DataFrame(encountered_rows)
        if args.encountered_out:
            os.makedirs(os.path.dirname(args.encountered_out), exist_ok=True)
            encountered_df.to_csv(args.encountered_out, index=False)
            print(f"[Encountered] Saved {len(encountered_df)} row(s) to {args.encountered_out}")
        if args.append_encountered_dataset:
            append_df = encountered_df[["B1_id", "B2_id", "B3_id", "log_reward"]].rename(columns={"log_reward": "y"})
            header_needed = not os.path.exists(args.append_encountered_dataset)
            os.makedirs(os.path.dirname(args.append_encountered_dataset), exist_ok=True)
            append_df.to_csv(args.append_encountered_dataset, mode="a", header=header_needed, index=False)
            print(f"[Append] Added {len(append_df)} row(s) to {args.append_encountered_dataset}")

    # Plots from recorder
    if recorder is not None and len(recorder.step_history) > 0:
        hist = recorder.step_history
        steps_arr = np.array([h["step"] for h in hist], dtype=np.int64)
        log1p_tb = np.array([h["log1p_tb_loss"] for h in hist], dtype=np.float64)
        log1p_tb_ema = np.array([h["log1p_tb_loss_ema"] for h in hist], dtype=np.float64)
        br_mean = np.array([h["batch_reward_mean"] for h in hist], dtype=np.float64)
        br_ema = np.array([h["batch_reward_ema"] for h in hist], dtype=np.float64)

        def _savefig(base):
            png = os.path.join(args.outdir, f"{base}.png")
            plt.tight_layout()
            plt.savefig(png, dpi=180)
            if not args.no_svg:
                svg = os.path.join(args.outdir, f"{base}.svg")
                plt.savefig(svg)
            plt.close()
            print(f"[Summary] Saved {base}.png")

        if np.isfinite(br_mean).any():
            plt.figure(figsize=(9, 4.5))
            plt.plot(steps_arr, br_mean, label="batch avg reward", alpha=0.5)
            if np.isfinite(br_ema).any():
                plt.plot(steps_arr, br_ema, label=f"EMA reward (α={0.98})", linewidth=2)
            plt.xlabel("step"); plt.ylabel("mean reward")
            plt.title(
                f"Batch reward per step, {args.size1}×{args.size2}×{args.size3} libraries (multihot),\n"
                f"hidden={args.hidden_dim}, layers={args.n_layers}, N_state={N_total},\n"
                f"β={args.beta}, batch={args.batch_trajectories}, lr={args.lr}"
            )
            plt.legend(loc="best")
            _savefig("batch_avg_reward")

        plt.figure(figsize=(9, 4.5))
        plt.plot(steps_arr, log1p_tb, label="log(1+TB loss)")
        plt.xlabel("step"); plt.ylabel("log(1+TB loss)"); plt.title("log(1+TB loss) per step")
        plt.legend(loc="best")
        _savefig("loss_log1p_tb")

        plt.figure(figsize=(9, 4.5))
        plt.plot(steps_arr, log1p_tb_ema, label="log(1+TB loss) (EMA)")
        plt.xlabel("step"); plt.ylabel("log(1+TB loss) (EMA)"); plt.title(f"log(1+TB loss) EMA (alpha={0.98})")
        plt.legend(loc="best")
        _savefig("loss_log1p_tb_ema")

    # Top-K stats + diversity
    if topm_records:
        topk_sorted = sorted(topm_records, key=lambda t: float(t["reward"]), reverse=True)[:args.topm]
        reps = [build_triple_representation(rec["B1"], rec["B2"], rec["B3"], bb_ecfp_table) for rec in topk_sorted]
        rewards_topk = np.array([float(rec["reward"]) for rec in topk_sorted], dtype=np.float64)
        r_mean, r_std = float(rewards_topk.mean()), float(rewards_topk.std())
        print(f"[Top-K Stats] reward mean={r_mean:.6g} std={r_std:.6g}")


if __name__ == "__main__":
    main()