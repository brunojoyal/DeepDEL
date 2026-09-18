#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RxnFlow-style GFlowNet for constructing DEL triples (B1, B2, B3) of fixed shape (s1, s2, s3),
trained with Trajectory Balance, using action subsampling à la RxnFlow Sec. 3.3.
Reward oracle: R(x) = exp(- TripleDeepSet(x)), with the TripleDeepSet loaded frozen.

Features:
- State emb: concat of SUM(Phi(bb)) for each partial set (B1, B2, B3) -> R^{3*d_h}.
- Phi can be either the legacy frozen DeepDEL embedding table, a trainable per-BB
  table initialized from DeepDEL, or a trainable Phi network initialized from DeepDEL
  (--phi-train-mode).
- Action emb: 3*D block. By default D = bb_fp_bits and blocks contain raw BB ECFPs;
  with --action-repr=phi, D = d_h and blocks contain the cycle-specific GFN Phi
  embeddings: Phi1(bb) ⊕ 0 ⊕ 0, 0 ⊕ Phi2(bb) ⊕ 0, or 0 ⊕ 0 ⊕ Phi3(bb).
- Joint scorer g_theta(s, a) parameterizes log edge flow logF; forward policy over a subsampled
  partial action set A* uses importance weights w(a) and masked log-softmax:
    log π(a|s;A*) = log w(a) + g_theta(s,a) - logsumexp_{a'∈A*}(log w(a') + g_theta(s,a')).

Trajectory Balance loss (tempered target):
    L_TB = ( logZ + Σ_t log π(a_t|s_{t-1};A*_t) - β * log R(x) )^2
where β = --beta (reward exponent).

"""

# flags
#  --joint-dim 512 --state-dim 1024 --action-dim 512 --grad-clip 0 --subsample-ratio-start 0 --subsample-ratio-end 0 --lr 1e-4 --logz-lr 1 --steps 200 --beta 100 --batch-trajectories 4 --k -1
#  --triple-ckpt outputs/threshold-9.0_proxy.pt --policy-ckpt outputs/gfn_policy.pt


import os, argparse, random, time
from typing import List, Tuple, Dict, Optional, Set


# from comet_ml import start
# from comet_ml.integration.pytorch import log_model
import numpy as np
import pandas as pd
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from contextlib import nullcontext


def _pad_and_stack_action_emb(
    per_traj_actions: List[List[Tuple[int, int]]],
    *,
    A1: torch.Tensor,
    A2: torch.Tensor,
    A3: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a padded batch of action embeddings from per-trajectory candidate lists.

    Returns:
      A_emb: [B, Kmax, 3*D]
      valid_mask: [B, Kmax] boolean

    We preserve the exact candidate sets produced by subsample-ratio (variable K per trajectory),
    but pad to a common Kmax for batched GPU evaluation.
    """
    B = len(per_traj_actions)
    Kmax = max((len(a) for a in per_traj_actions), default=0)
    if Kmax == 0:
        return (
            torch.empty((B, 0, A1.shape[1]), device=device, dtype=A1.dtype),
            torch.empty((B, 0), device=device, dtype=torch.bool),
        )

    D = A1.shape[1]
    out = torch.zeros((B, Kmax, D), device=device, dtype=A1.dtype)
    valid = torch.zeros((B, Kmax), device=device, dtype=torch.bool)
    for b, acts in enumerate(per_traj_actions):
        if not acts:
            continue
        idxs = torch.as_tensor([i for (_, i) in acts], device=device, dtype=torch.long)
        cycles = [c for (c, _) in acts]

        # Group indices by cycle so we can index_select from the cached per-cycle tables.
        pos1 = [j for j, c in enumerate(cycles) if c == 1]
        pos2 = [j for j, c in enumerate(cycles) if c == 2]
        pos3 = [j for j, c in enumerate(cycles) if c == 3]

        if pos1:
            out[b, torch.as_tensor(pos1, device=device), :] = A1.index_select(0, idxs[pos1])
        if pos2:
            out[b, torch.as_tensor(pos2, device=device), :] = A2.index_select(0, idxs[pos2])
        if pos3:
            out[b, torch.as_tensor(pos3, device=device), :] = A3.index_select(0, idxs[pos3])

        valid[b, : len(acts)] = True

    return out, valid

# ---------- No-deps progress utilities ----------
def _should_log_progress(i: int, total: int | None) -> bool:
    """Heuristic for when to emit progress lines."""
    if i <= 0:
        return False
    if total is None or total <= 0:
        return (i % 1000) == 0
    # aim for ~50 updates max
    every = max(1, total // 50)
    return (i % every) == 0 or i == total


def progress_iter(iterable, *, total: int | None = None, desc: str = "", enabled: bool = True):
    """Lightweight iterator wrapper that prints occasional progress updates."""
    if not enabled:
        yield from iterable
        return
    for i, x in enumerate(iterable, start=1):
        if _should_log_progress(i, total):
            if total is None:
                print(f"{desc}: {i}")
            else:
                print(f"{desc}: {i}/{total}")
        yield x


def progress_write(msg: str):
    print(msg)

# ---------- Utilities (ECFPs) ----------

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# === [ADD IMPORTS for metrics/logging/plots] ===
import json
import math
from dataclasses import dataclass, field

from deepdelgfn.autodock_proxy.model import load_autodock_proxy
from deepdelgfn.rewards import REWARD_MODES, reward as compute_library_reward
from deepdelgfn.deepdel.fourier import fourier_dim, fourier_encode_torch
from deepdelgfn.models.deepsets import Phi, TripleDeepSet
from deepdelgfn.utils.weights import smiles_weight_array, bb_weight_lookup, bb_sum_weights
import deepdelgfn.mols.dels as tri_mod


# --- Metrics Recorder ---
@dataclass
class MetricsRecorder:
    outdir: str
    ema_alpha: float = 0.98
    step_history: list = field(default_factory=list)
    _ema_loss: float = None
    _ema_reward: float = None
    _ema_phase_loss: float = None
    _ema_total_loss: float = None
    _csv_path: str = None
    _jsonl_path: str = None

    def __post_init__(self):
        os.makedirs(self.outdir, exist_ok=True)
        self._csv_path = os.path.join(self.outdir, "metrics.csv")
        self._jsonl_path = os.path.join(self.outdir, "metrics.jsonl")
        self._csv_written = False

    def update(self, step: int, log1p_tb_loss: float, logZ: float,
               batch_rewards: list, terminals_in_batch: int,
               phase_loss: Optional[float] = None,
               total_loss: Optional[float] = None,
               phase_target: Optional[float] = None,
               phase_subsample_circ_var_mean: Optional[float] = None,
               phase_subsample_target_abs_err_mean: Optional[float] = None,
               phase_subsample_circ_mean_mean: Optional[float] = None,
               phase_subsample_count_mean: Optional[float] = None,
               phase_subsample_circ_var_by_t: Optional[list] = None,
               phase_subsample_target_abs_err_by_t: Optional[list] = None):
        # EMA for display/logging
        if self._ema_loss is None:
            self._ema_loss = log1p_tb_loss
        else:
            self._ema_loss = self.ema_alpha * self._ema_loss + (1 - self.ema_alpha) * log1p_tb_loss

        empty_batch = int(len(batch_rewards) == 0)
        if batch_rewards:
            arr = np.asarray(batch_rewards, dtype=np.float64)
            rmean = float(np.nanmean(arr))
            rstd  = float(np.nanstd(arr))
            rmin  = float(np.nanmin(arr))
            rmax  = float(np.nanmax(arr))
        else:
            rmean = rstd = rmin = rmax = float("nan")
        # EMA for batch reward
        if not math.isnan(rmean):
            if self._ema_reward is None:
                self._ema_reward = rmean
            else:
                self._ema_reward = self.ema_alpha * self._ema_reward + (1 - self.ema_alpha) * rmean
        else:
            # keep previous EMA if no valid reward
            pass

        # EMA for phase loss (only when finite)
        if phase_loss is not None and math.isfinite(phase_loss):
            if self._ema_phase_loss is None:
                self._ema_phase_loss = phase_loss
            else:
                self._ema_phase_loss = self.ema_alpha * self._ema_phase_loss + (1 - self.ema_alpha) * phase_loss

        # EMA for total loss (only when finite)
        if total_loss is not None and math.isfinite(total_loss):
            if self._ema_total_loss is None:
                self._ema_total_loss = total_loss
            else:
                self._ema_total_loss = self.ema_alpha * self._ema_total_loss + (1 - self.ema_alpha) * total_loss

        row = {
            "step": step,
            "log1p_tb_loss": float(log1p_tb_loss),
            "log1p_tb_loss_ema": float(self._ema_loss),
            "logZ": float(logZ),
            "phase_loss": float(phase_loss) if phase_loss is not None else float("nan"),
            "phase_loss_ema": float(self._ema_phase_loss) if self._ema_phase_loss is not None else float("nan"),
            "total_loss": float(total_loss) if total_loss is not None else float("nan"),
            "total_loss_ema": float(self._ema_total_loss) if self._ema_total_loss is not None else float("nan"),
            "phase_target": float(phase_target) if phase_target is not None else float("nan"),
            "phase_subsample_circ_var_mean": float(phase_subsample_circ_var_mean) if phase_subsample_circ_var_mean is not None else float("nan"),
            "phase_subsample_target_abs_err_mean": float(phase_subsample_target_abs_err_mean) if phase_subsample_target_abs_err_mean is not None else float("nan"),
            "phase_subsample_circ_mean_mean": float(phase_subsample_circ_mean_mean) if phase_subsample_circ_mean_mean is not None else float("nan"),
            "phase_subsample_count_mean": float(phase_subsample_count_mean) if phase_subsample_count_mean is not None else float("nan"),
            "phase_subsample_circ_var_by_t": json.dumps(phase_subsample_circ_var_by_t) if phase_subsample_circ_var_by_t is not None else "[]",
            "phase_subsample_target_abs_err_by_t": json.dumps(phase_subsample_target_abs_err_by_t) if phase_subsample_target_abs_err_by_t is not None else "[]",
            "batch_reward_mean": rmean,
            "batch_reward_std": rstd,
            "batch_reward_min": rmin,
            "batch_reward_max": rmax,
            "batch_reward_ema": float(self._ema_reward) if self._ema_reward is not None else float("nan"),  # <--- ADD
            "terminals_in_batch": int(terminals_in_batch),
            "empty_batch": empty_batch,
        }
        self.step_history.append(row)

        mode = "w" if not self._csv_written else "a"
        write_header = not self._csv_written
        with open(self._csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._csv_written = True
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")




# --- Distance utilities for Top-K diversity ---
def generalized_tanimoto_distance(a: np.ndarray, b: np.ndarray) -> float:
    # assumes nonnegative vectors
    ab = float(np.dot(a, b))
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    denom = aa + bb - ab
    if denom <= 0:
        return 0.0
    sim = ab / denom
    sim = max(0.0, min(1.0, sim))
    return 1.0 - sim

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0: return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def build_triple_representation(B1, B2, B3, bb_ecfp_table: torch.Tensor) -> np.ndarray:
    # mean ECFP(D) per cycle (D = bb_ecfp_table.shape[1] = bb_fp_bits),
    # then concat -> [3*D].
    with torch.no_grad():
        def mean_vec(idxs):
            if len(idxs) == 0:
                return torch.zeros(bb_ecfp_table.shape[1], device=bb_ecfp_table.device, dtype=bb_ecfp_table.dtype)
            t = torch.as_tensor(idxs, device=bb_ecfp_table.device, dtype=torch.long)
            return bb_ecfp_table.index_select(0, t).float().mean(0)
        m1 = mean_vec(B1); m2 = mean_vec(B2); m3 = mean_vec(B3)
        v = torch.cat([m1, m2, m3], dim=0).cpu().numpy().astype(np.float64, copy=False)
    return v

def mean_pairwise_distance(vecs: list, metric: str = "tanimoto") -> float:
    if len(vecs) < 2: return 0.0
    if metric == "cosine":
        dist = cosine_distance
    elif metric == "euclidean":
        dist = euclidean_distance
    else:
        dist = generalized_tanimoto_distance
    n = len(vecs)
    s = 0.0; c = 0
    for i in range(n):
        for j in range(i+1, n):
            s += dist(vecs[i], vecs[j]); c += 1
    return s / max(1, c)


def set_seed(seed: Optional[int]):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def smiles_to_morgan_bits(smiles: str, n_bits: int = 2048, radius: int = 2, dtype=np.float32,
                          append_molecular_weight: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=dtype)
    DataStructs.ConvertToNumpyArray(fp, arr)
    if append_molecular_weight:
        from rdkit.Chem import Descriptors
        mw = float(Descriptors.MolWt(mol))
        arr = np.append(arr, np.array(mw, dtype=dtype))
    return arr

def load_bbs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "SMILES" not in df.columns:
        raise ValueError("bbs.csv must include a 'SMILES' column.")
    if "Name" not in df.columns:
        df["Name"] = [f"BB_%d" % i for i in range(len(df))]
    if "pool" not in df.columns:
        df["pool"] = 0
    df["pool"] = df["pool"].fillna(0).astype(int)
    return df

def load_or_create_ecfp_cache(
    smiles: List[str],
    *,
    n_bits: int,
    radius: int,
    cache_path: str,
    progress_desc: str,
    progress_enabled: bool = False,
    append_molecular_weight: bool = False,
) -> np.ndarray:
    expected_dim = n_bits + (1 if append_molecular_weight else 0)
    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path)
            X = z["X"]
            if X.shape == (len(smiles), expected_dim):
                print(f"[Cache] Loaded ECFP from {cache_path}")
                return X
            print(f"[Cache] Shape mismatch in {cache_path} (expected ({len(smiles)},{expected_dim}), got {tuple(X.shape)}); recomputing.")
        except Exception as e:
            print(f"[Cache] Failed to load {cache_path}: {e}. Recomputing.")

    feats = []
    for smi in progress_iter(smiles, total=len(smiles), desc=progress_desc, enabled=progress_enabled):
        arr = smiles_to_morgan_bits(smi, n_bits=n_bits, radius=radius, dtype=np.float32,
                                    append_molecular_weight=append_molecular_weight)
        if arr is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        feats.append(arr)
    X = np.stack(feats, 0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, X=X)
    print(f"[Cache] Saved ECFP to {cache_path}")
    return X

 # ---------- TripleDeepSet (frozen oracle) ----------

class GFlowNetPhi(nn.Module):
    """GFN-side DeepSets embedding Φ initialized from the DeepDEL proxy φ.

    This module separates the reward/proxy network from the policy network. In
    ``frozen_table`` mode it exactly preserves the previous implementation:
    DeepDEL φ is evaluated once and the resulting tensors are used as fixed
    features. In ``trainable_table`` mode those tensors initialize free
    per-building-block embeddings. In ``trainable_network`` mode the actual
    Φ_i neural networks are initialized from DeepDEL φ_i and trained jointly
    with the GFlowNet objective, which is the paper-faithful P2P transfer path.

    When ``init_model`` is None in ``trainable_network`` mode (--phi-random-init),
    the Φ networks are left at PyTorch's default Kaiming uniform initialization
    and trained from scratch.
    """

    VALID_MODES = {"frozen_table", "trainable_table", "trainable_network"}

    def __init__(
        self,
        *,
        mode: str,
        d_in: int,
        d_hidden: int,
        dropout: float,
        shared_phi: bool,
        init_tables: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        init_model: Optional[TripleDeepSet] = None,
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown phi train mode {mode!r}; expected one of {sorted(self.VALID_MODES)}")
        self.mode = mode
        self.shared_phi = bool(shared_phi)
        self.d_hidden = int(d_hidden)

        table1, table2, table3 = [t.detach().float() for t in init_tables]
        if table1.shape != table2.shape or table1.shape != table3.shape:
            raise ValueError(
                f"All initial Phi tables must have the same shape; got "
                f"{tuple(table1.shape)}, {tuple(table2.shape)}, {tuple(table3.shape)}"
            )

        if self.mode == "frozen_table":
            self.register_buffer("phi1_table", table1.clone())
            self.register_buffer("phi2_table", table2.clone())
            self.register_buffer("phi3_table", table3.clone())
        elif self.mode == "trainable_table":
            self.phi1_embedding = nn.Embedding.from_pretrained(table1.clone(), freeze=False)
            if self.shared_phi:
                self.phi2_embedding = self.phi1_embedding
                self.phi3_embedding = self.phi1_embedding
            else:
                self.phi2_embedding = nn.Embedding.from_pretrained(table2.clone(), freeze=False)
                self.phi3_embedding = nn.Embedding.from_pretrained(table3.clone(), freeze=False)
        else:  # trainable_network
            self.phi = Phi(d_in, d_hidden, dropout)
            if init_model is not None:
                self.phi.load_state_dict(init_model.phi.state_dict())
            if self.shared_phi:
                self.phi2 = self.phi
                self.phi3 = self.phi
            else:
                self.phi2 = Phi(d_in, d_hidden, dropout)
                self.phi3 = Phi(d_in, d_hidden, dropout)
                if init_model is not None:
                    self.phi2.load_state_dict(init_model.phi2.state_dict())
                    self.phi3.load_state_dict(init_model.phi3.state_dict())

        if self.mode == "frozen_table":
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    def tables(self, X_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cycle-specific [N, d_h] Φ tables for the current optimizer step.

        For trainable modes these tensors remain connected to the trainable
        parameters, so gradients from TB flow back into Φ. For frozen_table they
        are non-differentiable buffers.
        """
        if self.mode == "frozen_table":
            return self.phi1_table, self.phi2_table, self.phi3_table
        if self.mode == "trainable_table":
            return (
                self.phi1_embedding.weight,
                self.phi2_embedding.weight,
                self.phi3_embedding.weight,
            )
        return self.phi(X_all), self.phi2(X_all), self.phi3(X_all)


# ---------- RxnFlow-style joint scorer (log edge flow) ----------
class JointEdgePolicy(nn.Module):
    """
    g_theta(s, a): state-action joint scorer -> scalar logF (log edge flow).
    Allows separate widths for state and action towers plus joint head.

    interaction:
      - "none"     : original concat [h_s, h_a]
      - "hadamard" : concat [h_s, h_a, (W_s h_s) ⊙ (W_a h_a)]
      - "film"     : h_a <- γ(h_s) ⊙ h_a + β(h_s); concat [h_s, h_a_mod]
      - "bilinear" : low-rank bilinear: z_int = Σ_r (U_r h_s) ⊙ (V_r h_a); concat [h_s, h_a, z_int]

    Threshold conditioning: when ``fourier_n_freqs > 0``, the policy accepts
    a threshold tensor and concatenates its Fourier encoding to the state
    embedding before the state tower.
    """
    def __init__(
        self,
        d_h: int,
        d_state: int,
        d_action: int,
        d_joint: int,
        action_in_dim: int,
        *,
        interaction: str = "none",
        bilinear_rank: int = 8,
        bilinear_out: int = 128,
        bilinear_in: int = 128,
        use_layernorm: bool = False,
        phase_regularization: bool = False,
        fourier_n_freqs: int = 0,
        fourier_freq_scale: float = 1.5,
        fourier_linear: bool = False,
        fourier_append_raw: bool = False,
        fourier_threshold_center: Optional[float] = None,
        fourier_threshold_scale: Optional[float] = None,
    ):
        super().__init__()
        self.interaction = interaction.lower()
        self.d_state = d_state
        self.d_action = d_action
        self.d_int = bilinear_out
        self.phase_regularization = bool(phase_regularization)

        # Fourier threshold conditioning.
        self.fourier_n_freqs = int(fourier_n_freqs)
        self.fourier_freq_scale = float(fourier_freq_scale)
        self.fourier_linear = bool(fourier_linear)
        self.fourier_append_raw = bool(fourier_append_raw)
        self.fourier_threshold_center = fourier_threshold_center
        self.fourier_threshold_scale = fourier_threshold_scale
        self.fourier_dim = fourier_dim(self.fourier_n_freqs, self.fourier_append_raw) if self.fourier_n_freqs > 0 else 0

        # Light towers (your current version)
        self.state_proj = nn.Sequential(
            nn.Linear(3*d_h + self.fourier_dim, d_state),
            nn.ReLU(),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_in_dim, d_action),
            nn.ReLU(),
        )

        # Interaction params (created only if needed)
        if self.interaction == "hadamard":
            # d_star = min(d_state, d_action)
            self.align_s = nn.Linear(d_state, bilinear_in, bias=False)
            self.align_a = nn.Linear(d_action, bilinear_in, bias=False)
            in_dim = d_state + d_action + bilinear_out

        elif self.interaction == "film":
            self.film_gamma = nn.Linear(d_state, d_action, bias=True)
            self.film_beta  = nn.Linear(d_state, d_action, bias=True)
            in_dim = d_state + d_action  # concat [h_s, h_a_mod]

        elif self.interaction == "bilinear":
            # U: d_state -> (rank * d_int), V: d_action -> (rank * d_int)
            self.U = nn.Linear(d_state, bilinear_rank * bilinear_out, bias=False)
            self.V = nn.Linear(d_action, bilinear_rank * bilinear_out, bias=False)
            self.bilinear_rank = bilinear_rank
            in_dim = d_state + d_action + bilinear_out

        elif self.interaction == "none":
            in_dim = d_state + d_action

        else:
            raise ValueError(f"Unknown interaction '{interaction}'")

        self.pre_head_norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.joint_head = nn.Sequential(
            nn.Linear(in_dim, d_joint),
            nn.ReLU(),
            nn.Linear(d_joint, d_joint),
            nn.ReLU(),
            nn.Linear(d_joint, 1)  # scalar logF
        )
        if self.phase_regularization:
            self.phase_head = nn.Sequential(
                nn.Linear(in_dim, d_joint),
                nn.ReLU(),
                nn.Linear(d_joint, d_joint),
                nn.ReLU(),
                nn.Linear(d_joint, 1),  # scalar phase φ(a|s)
            )
            self._init_phase_biases()
        else:
            self.phase_head = None

    def _init_phase_biases(self):
        """Initialize phase-head biases away from zero to avoid cos-gradient plateaus."""
        if self.phase_head is None:
            return
        for module in self.phase_head.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.uniform_(module.bias, -math.pi, math.pi)

    def forward(
        self,
        state_emb: torch.Tensor,
        action_emb: torch.Tensor,
        *,
        return_phase: bool = False,
        threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        state_emb: [B, 3*d_h]
        action_emb: [B, K, 3*D]  (D = selected action base dim; K = #candidate actions per state)
        threshold: [B] optional threshold conditioning values
        returns: logF logits [B, K], or (logF, phase) if return_phase=True
        """
        B, K, _ = action_emb.shape

        if self.fourier_n_freqs > 0 and threshold is not None:
            f = fourier_encode_torch(
                threshold,
                n_freqs=self.fourier_n_freqs,
                freq_scale=self.fourier_freq_scale,
                linear=self.fourier_linear,
                append_raw=self.fourier_append_raw,
                threshold_center=self.fourier_threshold_center,
                threshold_scale=self.fourier_threshold_scale,
            )
            state_emb = torch.cat([state_emb, f], dim=-1)

        hs = self.state_proj(state_emb)                 # [B, d_state]
        ha = self.action_proj(action_emb.view(B*K, -1)) # [B*K, d_action]
        ha = ha.view(B, K, -1)                          # [B, K, d_action]
        hs_expand = hs.unsqueeze(1).expand(-1, K, -1)   # [B, K, d_state]

        if self.interaction == "none":
            joint = torch.cat([hs_expand, ha], dim=-1)

        elif self.interaction == "hadamard":
            # align to same width then feature-wise product
            hs_al = self.align_s(hs_expand)           
            ha_al = self.align_a(ha)                    
            prod  = hs_al * ha_al                     
            joint = torch.cat([hs_expand, ha, prod], dim=-1)

        elif self.interaction == "film":
            # feature-wise affine modulation of action by state
            gamma = self.film_gamma(hs).unsqueeze(1)    # [B, 1, d_action]
            beta  = self.film_beta(hs).unsqueeze(1)     # [B, 1, d_action]
            ha_mod = gamma * ha + beta                  # [B, K, d_action]
            joint = torch.cat([hs_expand, ha_mod], dim=-1)

        elif self.interaction == "bilinear":
            # low-rank bilinear features
            Uh = self.U(hs).view(B, 1, self.bilinear_rank, self.d_int).expand(-1, K, -1, -1)
            Vh = self.V(ha).view(B, K, self.bilinear_rank, self.d_int)
            z_int = (Uh * Vh).sum(dim=2)                # [B, K, d_int]
            joint = torch.cat([hs_expand, ha, z_int], dim=-1)

        else:
            raise RuntimeError("unreachable")

        joint = self.pre_head_norm(joint)
        logF = self.joint_head(joint).squeeze(-1)       # [B, K]
        if return_phase:
            if self.phase_head is None:
                raise RuntimeError("return_phase=True requires phase_regularization=True in JointEdgePolicy")
            phase = self.phase_head(joint).squeeze(-1)  # [B, K]
            return logF, phase
        return logF



class TripleSetEnv:
    """
    Holds 3 sets of BB indices and running sums of phi embeddings.
    Supports subsampling of candidate actions per cycle.

    Pool-aware: if `allowed_per_cycle` is provided, each cycle's action universe
    is restricted to the given list of BB row indices. This lets the GFN
    enforce DEL chemistry constraints (e.g., for AmpC: cycle 1/2 = amino-acid
    pool, cycle 3 = sulfonyl-chloride pool) so the policy can never propose a
    chemistry-invalid combination. When `allowed_per_cycle` is None, all three
    cycles draw from the full 0..N-1 universe (legacy behaviour).
    """
    def __init__(self, phi1_table: torch.Tensor, phi2_table: torch.Tensor, phi3_table: torch.Tensor,
                 bb_ecfp_table: torch.Tensor,
                 sizes: Tuple[int,int,int], device: torch.device,
                 allowed_per_cycle: Optional[Tuple[List[int], List[int], List[int]]] = None):
        self.phi1_table = phi1_table  # [N, d_h]
        self.phi2_table = phi2_table
        self.phi3_table = phi3_table
        self.bb_ecfp_table = bb_ecfp_table  # [N, bb_fp_bits]
        self.N = phi1_table.shape[0]
        self.d_phi = phi1_table.shape[1]
        self.s1, self.s2, self.s3 = sizes
        self.device = device

        # Per-cycle allowed BB row indices. We deep-copy at construction time so
        # `reset()` can rebuild fresh remaining-pool lists without re-scanning.
        if allowed_per_cycle is None:
            full = list(range(self.N))
            self._allowed1, self._allowed2, self._allowed3 = full, full, full
        else:
            a1, a2, a3 = allowed_per_cycle
            self._allowed1 = list(a1)
            self._allowed2 = list(a2)
            self._allowed3 = list(a3)
            # Validate that the requested subset sizes can actually be drawn
            # from each cycle's pool. Without this check, training would silently
            # fail to terminate trajectories.
            if self.s1 > len(self._allowed1):
                raise ValueError(f"size1={self.s1} exceeds pool-1 size={len(self._allowed1)}")
            if self.s2 > len(self._allowed2):
                raise ValueError(f"size2={self.s2} exceeds pool-2 size={len(self._allowed2)}")
            if self.s3 > len(self._allowed3):
                raise ValueError(f"size3={self.s3} exceeds pool-3 size={len(self._allowed3)}")

        # Swap-delete pools for fast sampling of remaining BBs without O(N) Python scans.
        # For each cycle, we maintain:
        #   rem{c}: list of remaining indices (size shrinks)
        #   pos{c}: dict idx->position in rem{c}
        self.reset()

    def reset(self):
        self.B1, self.B2, self.B3 = set(), set(), set()
        self.H1 = torch.zeros(self.d_phi, device=self.device)
        self.H2 = torch.zeros(self.d_phi, device=self.device)
        self.H3 = torch.zeros(self.d_phi, device=self.device)

        # Remaining pools (Python lists) + position maps.
        # This makes membership O(1) and sampling O(k), instead of O(N) scans.
        # Each cycle starts from its allowed pool (full 0..N-1 if no pool column).
        self._rem1 = list(self._allowed1)
        self._rem2 = list(self._allowed2)
        self._rem3 = list(self._allowed3)
        self._pos1 = {idx: p for p, idx in enumerate(self._rem1)}
        self._pos2 = {idx: p for p, idx in enumerate(self._rem2)}
        self._pos3 = {idx: p for p, idx in enumerate(self._rem3)}

    def is_terminal(self) -> bool:
        return (len(self.B1)==self.s1) and (len(self.B2)==self.s2) and (len(self.B3)==self.s3)

    def state_embedding(self) -> torch.Tensor:
        return torch.cat([self.H1, self.H2, self.H3], dim=0)  # [3*d_h]

    def remaining_per_cycle(self) -> Tuple[List[int], List[int], List[int]]:
        # Kept for compatibility/debugging, but now O(1) (no full scans).
        rem1 = self._rem1 if len(self.B1) < self.s1 else []
        rem2 = self._rem2 if len(self.B2) < self.s2 else []
        rem3 = self._rem3 if len(self.B3) < self.s3 else []
        return rem1, rem2, rem3

    def _sample_from_remaining(self, rem: List[int], k: int, rng: np.random.Generator) -> List[int]:
        if k <= 0 or len(rem) == 0:
            return []
        if k >= len(rem):
            # Return all remaining if k exceeds (rare once sizes small).
            return list(rem)
        # Uniform without replacement.
        # Using numpy choice on the list is fine; the key is we no longer build the list via O(N) scans.
        return rng.choice(rem, size=k, replace=False).tolist()

    def subsample_actions(self, ratio: float, min_per_cycle: int, rng: np.random.Generator):
        rem1, rem2, rem3 = self.remaining_per_cycle()
        acts: List[Tuple[int,int]] = []
        logw = []

        def pick(rem: List[int], cycle: int):
            if len(rem) == 0: return
            k = max(min_per_cycle, int(np.ceil(len(rem) * ratio)))
            k = min(k, len(rem))
            subset = self._sample_from_remaining(rem, k, rng)
            w = np.log(len(rem)) - np.log(len(subset))
            for idx in subset:
                acts.append((cycle, idx))
                logw.append(w)

        pick(rem1, 1); pick(rem2, 2); pick(rem3, 3)
        if not acts:
            return [], torch.empty(0, device=self.device)
        return acts, torch.tensor(logw, dtype=torch.float32, device=self.device)

    def action_embedding_block(self, actions: List[Tuple[int,int]]) -> torch.Tensor:
        D = self.bb_ecfp_table.shape[1]  # = bb_fp_bits
        if not actions:
            return torch.empty(0, 3*D, device=self.device, dtype=self.bb_ecfp_table.dtype)
        idxs = torch.tensor([i for (_, i) in actions], device=self.device, dtype=torch.long)
        cycles = torch.tensor([c for (c, _) in actions], device=self.device, dtype=torch.long)  # in {1,2,3}
        V = self.bb_ecfp_table.index_select(0, idxs)  # [K, D]
        K = V.shape[0]
        A = torch.zeros(K, 3*D, device=self.device, dtype=V.dtype)
        mask1 = (cycles == 1); mask2 = (cycles == 2); mask3 = (cycles == 3)
        if mask1.any(): A[mask1, 0:D]     = V[mask1]
        if mask2.any(): A[mask2, D:2*D]   = V[mask2]
        if mask3.any(): A[mask3, 2*D:3*D] = V[mask3]
        return A  # [K, 3*D]

    def step(self, action: Tuple[int, int]):
        c, idx = action
        if c == 1:
            if idx in self.B1 or len(self.B1) >= self.s1:
                raise ValueError("Invalid action.")
            self.B1.add(idx)
            self.H1 = self.H1 + self.phi1_table[idx]
            # swap-delete remove from remaining pool
            p = self._pos1.pop(idx)
            last = self._rem1[-1]
            self._rem1[p] = last
            self._pos1[last] = p
            self._rem1.pop()
        elif c == 2:
            if idx in self.B2 or len(self.B2) >= self.s2:
                raise ValueError("Invalid action.")
            self.B2.add(idx)
            self.H2 = self.H2 + self.phi2_table[idx]
            p = self._pos2.pop(idx)
            last = self._rem2[-1]
            self._rem2[p] = last
            self._pos2[last] = p
            self._rem2.pop()
        else:
            if idx in self.B3 or len(self.B3) >= self.s3:
                raise ValueError("Invalid action.")
            self.B3.add(idx)
            self.H3 = self.H3 + self.phi3_table[idx]
            p = self._pos3.pop(idx)
            last = self._rem3[-1]
            self._rem3[p] = last
            self._pos3[last] = p
            self._rem3.pop()


class CachedActionEmbeddings:
    """Cycle-specific action embedding tables to avoid per-step allocation/copy.

    For each cycle c in {1,2,3}, we store a [N, 3*D] table where only the
    corresponding 1/3 block is filled and the other two blocks are zero-padded.

    In legacy mode the same raw BB ECFP table is used for all cycles. In phi
    mode, cycle-specific pretrained DeepDEL tables are used, yielding actions
    like (phi1(bb), 0, 0), (0, phi2(bb), 0), and (0, 0, phi3(bb)).
    """

    def __init__(
        self,
        table1: torch.Tensor,
        table2: Optional[torch.Tensor] = None,
        table3: Optional[torch.Tensor] = None,
    ):
        if table1.ndim != 2:
            raise ValueError("table1 must be 2D [N, D]")
        table2 = table1 if table2 is None else table2
        table3 = table1 if table3 is None else table3
        if table2.ndim != 2 or table3.ndim != 2:
            raise ValueError("table2/table3 must be 2D [N, D]")
        if table1.shape != table2.shape or table1.shape != table3.shape:
            raise ValueError(
                f"All action tables must have the same shape; got "
                f"{tuple(table1.shape)}, {tuple(table2.shape)}, {tuple(table3.shape)}"
            )
        if table1.device != table2.device or table1.device != table3.device:
            raise ValueError("All action tables must be on the same device")
        if table1.dtype != table2.dtype or table1.dtype != table3.dtype:
            table2 = table2.to(dtype=table1.dtype)
            table3 = table3.to(dtype=table1.dtype)

        N, D = table1.shape
        if D <= 0:
            raise ValueError(f"Expected action table shape [N, D>0], got [N,{D}]")

        device = table1.device
        dtype = table1.dtype

        # Allocate once (on GPU) and fill blocks.
        A1 = torch.zeros((N, 3 * D), device=device, dtype=dtype)
        A2 = torch.zeros((N, 3 * D), device=device, dtype=dtype)
        A3 = torch.zeros((N, 3 * D), device=device, dtype=dtype)
        A1[:, 0:D] = table1
        A2[:, D : 2 * D] = table2
        A3[:, 2 * D : 3 * D] = table3

        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.base_dim = D

    def embed_actions(self, actions: List[Tuple[int, int]]) -> torch.Tensor:
        """Return [K, 3*D] for a list of (cycle,idx) actions."""
        if not actions:
            return torch.empty((0, self.A1.shape[1]), device=self.A1.device, dtype=self.A1.dtype)
        idxs = torch.as_tensor([i for (_, i) in actions], device=self.A1.device, dtype=torch.long)
        cycles = [c for (c, _) in actions]
        pos1 = [j for j, c in enumerate(cycles) if c == 1]
        pos2 = [j for j, c in enumerate(cycles) if c == 2]
        pos3 = [j for j, c in enumerate(cycles) if c == 3]
        out = torch.zeros((len(actions), self.A1.shape[1]), device=self.A1.device, dtype=self.A1.dtype)
        if pos1:
            out[torch.as_tensor(pos1, device=self.A1.device)] = self.A1.index_select(0, idxs[pos1])
        if pos2:
            out[torch.as_tensor(pos2, device=self.A1.device)] = self.A2.index_select(0, idxs[pos2])
        if pos3:
            out[torch.as_tensor(pos3, device=self.A1.device)] = self.A3.index_select(0, idxs[pos3])
        return out



# ---------- Rollouts + TB loss with subsampling ----------

def _phase_subsample_circular_stats(
    phase_values: torch.Tensor,
    *,
    target: Optional[float],
) -> Dict[str, float]:
    """Circular diagnostics for all phases in one subsampled action set A*(s).

    Phases are interpreted modulo 2π. The circular variance is ``1 - R``, where
    ``R`` is the mean resultant length. The target error is the mean shortest
    angular distance to ``target`` in radians.
    """
    with torch.no_grad():
        vals = phase_values.detach().float().reshape(-1)
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            return {
                "circ_var": float("nan"),
                "circ_mean": float("nan"),
                "target_abs_err": float("nan"),
                "count": 0.0,
            }

        two_pi = 2.0 * math.pi
        wrapped = torch.remainder(vals, two_pi)
        c = torch.cos(wrapped).mean()
        s = torch.sin(wrapped).mean()
        resultant = torch.sqrt(c * c + s * s).clamp(0.0, 1.0)
        circ_var = 1.0 - resultant
        circ_mean = torch.remainder(torch.atan2(s, c), two_pi)

        if target is None:
            target_abs_err = torch.tensor(float("nan"), device=wrapped.device)
        else:
            target_t = torch.tensor(float(target), device=wrapped.device, dtype=wrapped.dtype)
            # Shortest signed angular distance in [-π, π), then absolute value.
            delta = torch.remainder(wrapped - target_t + math.pi, two_pi) - math.pi
            target_abs_err = delta.abs().mean()

        return {
            "circ_var": float(circ_var.item()),
            "circ_mean": float(circ_mean.item()),
            "target_abs_err": float(target_abs_err.item()),
            "count": float(vals.numel()),
        }


def _summarize_phase_subsample_stats(
    stats: List[Dict[str, float]],
    *,
    T: int,
) -> Dict[str, object]:
    """Aggregate per-A*(s) circular phase diagnostics by trajectory position."""
    if not stats:
        return {
            "circ_var_mean": float("nan"),
            "target_abs_err_mean": float("nan"),
            "circ_mean_mean": float("nan"),
            "count_mean": float("nan"),
            "circ_var_by_t": [float("nan")] * int(T),
            "target_abs_err_by_t": [float("nan")] * int(T),
        }

    def finite_mean(values: List[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    circ_var_by_t: List[float] = []
    target_abs_err_by_t: List[float] = []
    for t in range(int(T)):
        at_t = [x for x in stats if int(x.get("t", -1)) == t]
        circ_var_by_t.append(finite_mean([float(x.get("circ_var", float("nan"))) for x in at_t]))
        target_abs_err_by_t.append(finite_mean([float(x.get("target_abs_err", float("nan"))) for x in at_t]))

    return {
        "circ_var_mean": finite_mean([float(x.get("circ_var", float("nan"))) for x in stats]),
        "target_abs_err_mean": finite_mean([float(x.get("target_abs_err", float("nan"))) for x in stats]),
        "circ_mean_mean": finite_mean([float(x.get("circ_mean", float("nan"))) for x in stats]),
        "count_mean": finite_mean([float(x.get("count", float("nan"))) for x in stats]),
        "circ_var_by_t": circ_var_by_t,
        "target_abs_err_by_t": target_abs_err_by_t,
    }

def rollout_trajectory(env: TripleSetEnv, policy: JointEdgePolicy,
                       rng: np.random.Generator,
                       subsample_ratio: float, min_per_cycle: int,
                       action_cache: Optional[CachedActionEmbeddings] = None,
                       use_amp: bool = False,
                       epsilon: float = 0.05,
                       phase_regularization: bool = False,
                       phase_target: Optional[float] = None,
                       threshold: Optional[float] = None) -> Dict:
    """
    Roll out one trajectory using subsampled partial action sets A*(s).
    Sampling is done with an unweighted softmax over the subsample (logits = logF).
    For TB, we estimate the full forward log-prob via:
        log π_F(a|s) ≈ logF[a] - logsumexp_{b∈A*}( log w[b] + logF[b] )
    Returns:
      - actions: list[(cycle, idx)]
      - logps:  list[tensor scalar]  (sum used in TB)
      - phases: list[tensor scalar] selected φ(a|s), only populated when enabled
      - phase_subsample_stats: circular stats over every subsampled candidate set A*(s)
      - terminal: (B1,B2,B3) lists of indices
      - threshold: the threshold conditioning value used for this trajectory
    """
    env.reset()
    actions, logps, phases = [], [], []
    phase_subsample_stats: List[Dict[str, float]] = []

    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    # Even if AMP is enabled globally, we run the POLICY forward in FP32.
    # This avoids rare FP16 overflows in Linear/ReLU stacks that can produce inf logits.
    policy_autocast_ctx = (lambda: torch.cuda.amp.autocast(enabled=False)) if use_amp else nullcontext

    # Precompute threshold tensor for policy conditioning.
    threshold_tensor = None
    if threshold is not None:
        threshold_tensor = torch.tensor([threshold], dtype=torch.float32, device=env.device)

    while not env.is_terminal():
        # Subsample partial action space A*(s)
        Astar, logw = env.subsample_actions(subsample_ratio, min_per_cycle, rng)
        if len(Astar) == 0:
            break

        s_emb = env.state_embedding().unsqueeze(0)             # [1, 3*d_h]
        if action_cache is None:
            A_emb = env.action_embedding_block(Astar).unsqueeze(0)  # [1, K, 6144]
        else:
            A_emb = action_cache.embed_actions(Astar).unsqueeze(0)

        # Defensive: catch device mismatches early with a clearer error.
        # (Most common culprit is forgetting `policy.to(device)`.)
        if s_emb.device != A_emb.device or s_emb.device != next(policy.parameters()).device:
            raise RuntimeError(
                "Device mismatch in rollout_trajectory: "
                f"state_emb={s_emb.device}, action_emb={A_emb.device}, policy={next(policy.parameters()).device}"
            )

        if not torch.isfinite(s_emb).all():
            print("[NaNGuard] state_emb has non-finite values");  # optionally dump step/id
        if not torch.isfinite(A_emb).all():
            print("[NaNGuard] action_emb has non-finite values")
        # Policy forward in FP32 for numerical stability.
        with policy_autocast_ctx():
            if phase_regularization:
                logF, phase = policy(s_emb.float(), A_emb.float(), return_phase=True, threshold=threshold_tensor)
                logF = logF.squeeze(0)      # [K], parameterizes log edge flow
                phase = phase.squeeze(0)    # [K], parameterizes action-state phase
            else:
                logF = policy(s_emb.float(), A_emb.float(), threshold=threshold_tensor).squeeze(0)  # [K], parameterizes log edge flow
                phase = None

        # 1) Sample action from unweighted logits over the subsample (behavior policy)
        logits_sample = logF.float()  # [K]

        # sanitize logits: if any NaN/Inf, fall back to uniform
        if not torch.isfinite(logits_sample).all():
            print("Infinite logits")
            idx = torch.randint(len(Astar), (1,), device=logits_sample.device).item()
        else:
            # ε-greedy on device to avoid Python RNG & sync quirks
            if torch.rand((), device=logits_sample.device).item() < (1-anneal_ratio(epsilon, 1))*epsilon:
                idx = torch.randint(len(Astar), (1,), device=logits_sample.device).item()
            else:
                dist = torch.distributions.Categorical(logits=logits_sample)
                idx = dist.sample().item()


        # 2) TB uses an estimate of the FULL forward probability (divide-only HT correction)
        #    log π_F(a|s) ≈ logF[a] - logsumexp(logw + logF) over the subsample
        log_den = torch.logsumexp((logw + logF).float(), dim=-1)
        logp_tb = logF.float()[idx] - log_den

        a = Astar[idx]
        actions.append(a)
        logps.append(logp_tb)
        if phase_regularization:
            phase_subsample_stats.append({
                "t": float(len(actions) - 1),
                **_phase_subsample_circular_stats(phase.float(), target=phase_target),
            })
            phases.append(phase.float()[idx])
        env.step(a)

    B1 = sorted(list(env.B1)); B2 = sorted(list(env.B2)); B3 = sorted(list(env.B3))
    return {
        "actions": actions,
        "logps": logps,
        "phases": phases,
        "phase_subsample_stats": phase_subsample_stats,
        "terminal": (B1, B2, B3),
        "threshold": threshold,
    }


def rollout_trajectories_batched(
    *,
    phi1_table: torch.Tensor,
    phi2_table: torch.Tensor,
    phi3_table: torch.Tensor,
    sizes: Tuple[int, int, int],
    policy: JointEdgePolicy,
    action_cache: CachedActionEmbeddings,
    rng: np.random.Generator,
    subsample_ratio: float,
    min_per_cycle: int,
    batch_trajectories: int,
    device: torch.device,
    use_amp: bool = False,
    epsilon: float = 0.05,
    allowed_per_cycle: Optional[Tuple[List[int], List[int], List[int]]] = None,
    phase_regularization: bool = False,
    phase_target: Optional[float] = None,
    thresholds: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Roll out a batch of trajectories while preserving subsample-ratio semantics.

    Key idea: at each environment step, we compute the candidate sets A*(s)
    independently per trajectory (variable K), then pad to Kmax and evaluate
    `policy(state_emb, action_emb)` in a single batched call.

    `allowed_per_cycle`, when provided, restricts each cycle's action space
    (see TripleSetEnv for details).

    `thresholds`, when provided, is an array of per-trajectory threshold values
    used for threshold conditioning.
    """
    # The first D columns of A1 contain the selected per-BB action representation
    # for cycle 1 (raw ECFP in legacy mode, phi1 in phi mode). TripleSetEnv only
    # needs this placeholder for the legacy non-cache embedding path, which this
    # batched rollout implementation does not use.
    action_base_dim = action_cache.A1.shape[1] // 3
    envs = [
        TripleSetEnv(
            phi1_table, phi2_table, phi3_table,
            action_cache.A1[:, :action_base_dim], sizes, device,
            allowed_per_cycle=allowed_per_cycle,
        )
        for _ in range(batch_trajectories)
    ]
    # NOTE: TripleSetEnv expects an action table only for the legacy embedding
    # path; we pass a slice to satisfy shape but will NOT use env.action_embedding_block.
    for e in envs:
        e.reset()

    done = [False] * batch_trajectories
    actions_taken: List[List[Tuple[int, int]]] = [[] for _ in range(batch_trajectories)]
    logps_taken: List[List[torch.Tensor]] = [[] for _ in range(batch_trajectories)]
    phases_taken: List[List[torch.Tensor]] = [[] for _ in range(batch_trajectories)]
    phase_subsample_stats_taken: List[List[Dict[str, float]]] = [[] for _ in range(batch_trajectories)]

    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    policy_autocast_ctx = (lambda: torch.cuda.amp.autocast(enabled=False)) if use_amp else nullcontext

    # Upper bound on steps; some trajectories may terminate early (but with fixed sizes should match T).
    T = sum(sizes)
    for _t in range(T):
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break

        # Build per-trajectory candidate sets and padded embeddings
        per_actions: List[List[Tuple[int, int]]] = []
        per_logw: List[torch.Tensor] = []
        state_embs = []
        active_to_global = []

        for gi in active:
            env = envs[gi]
            Astar, logw = env.subsample_actions(subsample_ratio, min_per_cycle, rng)
            if len(Astar) == 0:
                done[gi] = True
                continue
            per_actions.append(Astar)
            per_logw.append(logw)
            state_embs.append(env.state_embedding())
            active_to_global.append(gi)

        if not per_actions:
            continue

        state_emb = torch.stack(state_embs, dim=0)  # [B', 3*d_h]
        A_emb, valid_mask = _pad_and_stack_action_emb(
            per_actions, A1=action_cache.A1, A2=action_cache.A2, A3=action_cache.A3, device=device
        )  # [B', Kmax, 6144]

        # Policy forward in FP32 for numerical stability.
        # Gather thresholds for active trajectories.
        threshold_batch = None
        if thresholds is not None:
            active_thresholds = [thresholds[gi] for gi in active_to_global]
            threshold_batch = torch.tensor(active_thresholds, dtype=torch.float32, device=device)

        with policy_autocast_ctx():
            if phase_regularization:
                logF, phase = policy(state_emb.float(), A_emb.float(), return_phase=True, threshold=threshold_batch)  # [B', Kmax]
            else:
                logF = policy(state_emb.float(), A_emb.float(), threshold=threshold_batch)  # [B', Kmax]
                phase = None

        # Mask padded positions so sampling never selects them.
        logF = logF.masked_fill(~valid_mask, -1e9)

        # Sample an action per active trajectory.
        for bi, gi in enumerate(active_to_global):
            Astar = per_actions[bi]
            logw = per_logw[bi]
            K = len(Astar)
            if K == 0:
                done[gi] = True
                continue

            logits_sample = logF[bi, :K].float()
            if not torch.isfinite(logits_sample).all():
                idx = torch.randint(K, (1,), device=device).item()
            else:
                if torch.rand((), device=device).item() < (1 - anneal_ratio(epsilon, 1)) * epsilon:
                    idx = torch.randint(K, (1,), device=device).item()
                else:
                    idx = torch.distributions.Categorical(logits=logits_sample).sample().item()

            # TB correction: log π_F(a|s) ≈ logF[a] - logsumexp(logw + logF)
            # logw is length K on device.
            log_den = torch.logsumexp((logw + logF[bi, :K]).float(), dim=-1)
            logp_tb = logF[bi, idx].float() - log_den

            a = Astar[idx]
            actions_taken[gi].append(a)
            logps_taken[gi].append(logp_tb)
            if phase_regularization:
                phase_subsample_stats_taken[gi].append({
                    "t": float(len(actions_taken[gi]) - 1),
                    **_phase_subsample_circular_stats(phase[bi, :K].float(), target=phase_target),
                })
                phases_taken[gi].append(phase[bi, idx].float())
            envs[gi].step(a)
            done[gi] = envs[gi].is_terminal()

    out = []
    for gi in range(batch_trajectories):
        env = envs[gi]
        B1 = sorted(list(env.B1)); B2 = sorted(list(env.B2)); B3 = sorted(list(env.B3))
        out.append({
            "actions": actions_taken[gi],
            "logps": logps_taken[gi],
            "phases": phases_taken[gi],
            "phase_subsample_stats": phase_subsample_stats_taken[gi],
            "terminal": (B1, B2, B3),
            "threshold": float(thresholds[gi]) if thresholds is not None else None,
        })
    return out


class ConditionalLogZ(nn.Module):
    """Threshold-conditioned log-partition function for Trajectory Balance.

    When the reward R(x; τ) depends on a threshold τ, the partition function
    Z(τ) = Σ_x R(x; τ) also depends on τ. This module learns logZ(τ) as a
    function of the Fourier-encoded threshold.

    When fourier_n_freqs=0, this reduces to a learnable scalar (backward
    compatible with non-conditioned training).
    """

    def __init__(
        self,
        fourier_n_freqs: int = 0,
        fourier_freq_scale: float = 1.5,
        fourier_linear: bool = False,
        fourier_append_raw: bool = False,
        fourier_threshold_center: Optional[float] = None,
        fourier_threshold_scale: Optional[float] = None,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.fourier_n_freqs = int(fourier_n_freqs)
        self.fourier_freq_scale = float(fourier_freq_scale)
        self.fourier_linear = bool(fourier_linear)
        self.fourier_append_raw = bool(fourier_append_raw)
        self.fourier_threshold_center = fourier_threshold_center
        self.fourier_threshold_scale = fourier_threshold_scale

        if self.fourier_n_freqs > 0:
            fourier_encoding_dim = fourier_dim(self.fourier_n_freqs, self.fourier_append_raw)
            self.net = nn.Sequential(
                nn.Linear(fourier_encoding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            # Initialize the final layer to output ~0 so logZ starts near 0.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            # Non-conditioned: learnable scalar (backward compatible).
            self.logZ_scalar = nn.Parameter(torch.tensor(0.0))

    def forward(self, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute logZ for the given threshold(s).

        Args:
            threshold: [B] tensor of threshold values, or None for non-conditioned.

        Returns:
            [B] tensor of logZ values (or scalar if threshold is None and
            fourier_n_freqs=0).
        """
        if self.fourier_n_freqs > 0:
            if threshold is None:
                raise ValueError("ConditionalLogZ requires threshold when fourier_n_freqs > 0")
            f = fourier_encode_torch(
                threshold,
                n_freqs=self.fourier_n_freqs,
                freq_scale=self.fourier_freq_scale,
                linear=self.fourier_linear,
                append_raw=self.fourier_append_raw,
                threshold_center=self.fourier_threshold_center,
                threshold_scale=self.fourier_threshold_scale,
            )
            return self.net(f).squeeze(-1)  # [B]
        else:
            # Non-conditioned: return scalar expanded to batch size if needed.
            if threshold is not None:
                return self.logZ_scalar.expand(threshold.shape[0])
            return self.logZ_scalar


def _ckpt_fourier_args(pck_args: Dict[str, object], pck_meta: Dict[str, object]) -> Dict[str, object]:
    """Extract the GFN-policy Fourier conditioning args saved in a checkpoint."""
    def _v(meta_key: str, args_key: str, default):
        if pck_meta.get(meta_key) is not None:
            return pck_meta[meta_key]
        if pck_args.get(args_key) is not None:
            return pck_args[args_key]
        return default

    return {
        "n_freqs": int(_v("gfn_fourier_n_freqs", "fourier_n_freqs", 0)),
        "freq_scale": float(_v("gfn_fourier_freq_scale", "fourier_freq_scale", 1.5)),
        "linear": bool(_v("gfn_fourier_linear", "fourier_linear", False)),
        "append_raw": bool(_v("gfn_fourier_append_raw", "fourier_append_raw", False)),
        "threshold_center": pck_args.get("fourier_threshold_center"),
        "threshold_scale": pck_args.get("fourier_threshold_scale"),
    }


def _fold_threshold_into_state_proj(
    policy_state: Dict[str, torch.Tensor],
    *,
    d_h: int,
    threshold: float,
    n_freqs: int,
    freq_scale: float,
    linear: bool,
    append_raw: bool,
    threshold_center: Optional[float],
    threshold_scale: Optional[float],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Fold threshold conditioning into the policy's first layer (specialization).

    The conditioned ``JointEdgePolicy`` has ``state_proj.0`` (a ``Linear`` whose
    input is ``3*d_h + fourier_dim``) immediately followed by ReLU. Because that
    layer is linear, conditioning at a *fixed* threshold τ is exactly equivalent
    to a non-conditioned layer with

        W'  = W[:, :3*d_h]
        b'  = b + W[:, 3*d_h:] @ fourier(τ)

    so the resulting (folded) weights reproduce the conditioned policy's logF
    at τ. Returns ``(new_policy_state, fourier_dim)``.
    """
    w_key, b_key = "state_proj.0.weight", "state_proj.0.bias"
    if w_key not in policy_state or b_key not in policy_state:
        raise ValueError("--strip-conditioning: checkpoint policy has no state_proj keys; cannot fold.")
    weight = policy_state[w_key]
    bias = policy_state[b_key]
    d_out, in_dim = int(weight.shape[0]), int(weight.shape[1])
    if in_dim <= 3 * d_h:
        raise ValueError(
            "--strip-conditioning: checkpoint policy has no threshold conditioning to strip "
            f"(state_proj input dim {in_dim} <= 3*d_h {3 * d_h})."
        )
    fdim = in_dim - 3 * d_h
    t = torch.tensor([float(threshold)], dtype=weight.dtype)
    f = fourier_encode_torch(
        t,
        n_freqs=n_freqs,
        freq_scale=freq_scale,
        linear=linear,
        append_raw=append_raw,
        threshold_center=threshold_center,
        threshold_scale=threshold_scale,
    )  # [1, fdim]
    if int(f.shape[1]) != fdim:
        raise ValueError(
            f"--strip-conditioning: checkpoint policy embeds {fdim} conditioning dims "
            f"but the checkpoint Fourier args encode {int(f.shape[1])}. Check --fourier-* args."
        )
    w_core = weight[:, : 3 * d_h].contiguous()
    w_t = weight[:, 3 * d_h :]  # [d_out, fdim]
    b_fold = bias + (w_t @ f.squeeze(0).unsqueeze(1)).squeeze(1)  # [d_out]
    new_state = dict(policy_state)
    new_state[w_key] = w_core
    new_state[b_key] = b_fold.contiguous()
    return new_state, int(fdim)


def _verify_folded_policy(
    folded_policy: nn.Module,
    policy_state: Dict[str, torch.Tensor],
    pck_fourier: Dict[str, object],
    *,
    d_h: int,
    d_state: int,
    d_action: int,
    d_joint: int,
    action_in_dim: int,
    threshold: float,
    device: torch.device,
) -> None:
    """Assert the folded non-conditioned policy matches the conditioned policy at τ.

    Runs a few random state/action batches through both policies and checks that
    logF agrees at the specialized threshold. This is the Experiment 2 smoke test
    that the threshold fold is exact.
    """
    cond = JointEdgePolicy(
        d_h=d_h,
        d_state=d_state,
        d_action=d_action,
        d_joint=d_joint,
        action_in_dim=action_in_dim,
        interaction="none",
        fourier_n_freqs=pck_fourier["n_freqs"],
        fourier_freq_scale=pck_fourier["freq_scale"],
        fourier_linear=pck_fourier["linear"],
        fourier_append_raw=pck_fourier["append_raw"],
        fourier_threshold_center=pck_fourier["threshold_center"],
        fourier_threshold_scale=pck_fourier["threshold_scale"],
    )
    cond.load_state_dict(policy_state)
    cond.to(device).eval()
    folded_policy.to(device).eval()
    torch.manual_seed(0)
    s = torch.randn(4, 3 * d_h, device=device)
    a = torch.randn(4, 7, action_in_dim, device=device)
    t = torch.full((4,), float(threshold), device=device)
    with torch.no_grad():
        out_cond = cond(s, a, threshold=t)
        out_folded = folded_policy(s, a)  # non-conditioned
    err = float((out_cond - out_folded).abs().max().item())
    print(f"[Strip] Fold check: max |ΔlogF| over 4 random states x 7 actions = {err:.3e}")
    if not math.isfinite(err) or err > 1e-3:
        raise RuntimeError(f"--strip-conditioning fold check failed: max |ΔlogF| = {err:.3e}")


def tb_loss(logZ: torch.Tensor, logps_sum: torch.Tensor, beta_logR: torch.Tensor) -> torch.Tensor:
    """
    L_TB = (logZ + sum_t log π(a_t|s_{t-1};A*_t) - beta * log R(x))^2

    logZ can be a scalar or a [B] tensor of per-trajectory logZ values.
    """
    resid = logZ + logps_sum - beta_logR
    return (resid ** 2).mean()


def phase_loss(phase_sums: torch.Tensor) -> torch.Tensor:
    """Phase synchronization loss: mean(1 - cos(sum_t φ(a_t|s_t)))."""
    return (1.0 - torch.cos(phase_sums.float())).mean()

# ---------- Reward oracle wrapper (frozen TripleDeepSet) ----------

# ---------- Helper parsing ----------

def _parse_pipe_separated_ints(value: Optional[str]) -> List[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(part) for part in text.split("|") if part.strip()]


@torch.no_grad()
def reward_from_triple(triple_sets: Tuple[List[int],List[int],List[int]],
                       X_all: torch.Tensor,
                       triple_model: TripleDeepSet,
                       device: torch.device,
                       *,
                       log_target: bool = False,
                       model_is_logprobs: bool = False,
                       threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (log_R, R) for a single terminal triple.

    Three checkpoint conventions are supported:

    * ``log_target=True`` (recommended, NaN-safe): the regression head was
      trained to output ``log R(x)`` directly. We return that as ``log_R`` and
      also expose ``R = exp(log_R)`` for diagnostics. **No log() of a model
      output is ever computed**, so the TB loss cannot become NaN due to a
      non-positive reward prediction.
    * ``model_is_logprobs=True`` (legacy): the head outputs ``-log R``, i.e.
      the docking-score-like quantity, so ``log_R = -yhat`` and
      ``R = exp(log_R)``.
    * Otherwise (legacy): the head outputs ``R`` directly. We compute
      ``log_R = log(R + 1e-40)``. This is the path that previously caused TB
      loss NaNs whenever ``R`` collapsed to zero or went negative.

    ``threshold`` is the threshold conditioning value for threshold-conditioned
    models. If None, the model is called without threshold conditioning.
    """
    B1, B2, B3 = triple_sets
    s1, s2, s3 = len(B1), len(B2), len(B3)
    m1 = torch.ones(s1, device=device); m2 = torch.ones(s2, device=device); m3 = torch.ones(s3, device=device)

    X1 = X_all[torch.tensor(B1, dtype=torch.long, device=device)]
    X2 = X_all[torch.tensor(B2, dtype=torch.long, device=device)]
    X3 = X_all[torch.tensor(B3, dtype=torch.long, device=device)]
    X1 = X1.unsqueeze(0); m1 = m1.unsqueeze(0)
    X2 = X2.unsqueeze(0); m2 = m2.unsqueeze(0)
    X3 = X3.unsqueeze(0); m3 = m3.unsqueeze(0)

    if threshold is not None:
        threshold_tensor = torch.tensor([threshold], dtype=torch.float32, device=device)
        out = triple_model(((X1, m1), (X2, m2), (X3, m3)), threshold_tensor)  # [1]
    else:
        out = triple_model(((X1, m1), (X2, m2), (X3, m3)))  # [1]

    if log_target:
        # Model is trained on log R (recommended). No log() of a regression
        # output -- this is what eliminates the TB-loss NaN failure mode.
        log_R = out
        R = torch.exp(log_R)
    elif model_is_logprobs:
        # Model outputs yhat = -log R.
        log_R = -out
        R = torch.exp(log_R)
    else:
        # Model outputs R directly. Guard log() with a tiny floor;
        # this is the failure mode that the --log-target path is meant to fix.
        R = out
        log_R = torch.log(R + 1e-40)
    return log_R, R


class AutodockProxyRewardOracle:
    """Terminal-library reward oracle backed by the autodock proxy.

    The GFN still uses DeepDEL φ tables as state representations. This oracle is
    only responsible for computing the terminal TB target in
    ``--gfn-reward-source autodock_proxy`` mode by enumerating products for a
    proposed library, scoring those products with the autodock proxy, and
    reducing molecule scores to a library reward.
    """

    def __init__(
        self,
        *,
        bbs_df: pd.DataFrame,
        id_map: Optional[np.ndarray],
        autodock_model: str,
        reward_mode: str,
        threshold: Optional[float],
        alpha: float,
        k: int,
        batch_size: int,
        device: str,
        reaction_mode: str,
        max_weight: Optional[float] = None,
        weight_source: str = "bb_sum",
    ):
        if id_map is None:
            raise ValueError("autodock_proxy reward source requires the BB CSV to contain an ID column.")
        if not os.path.exists(autodock_model):
            raise FileNotFoundError(f"Autodock proxy artifact not found: {autodock_model}")
        self.id_map = id_map
        self.reward_mode = str(reward_mode)
        self.threshold = threshold
        self.alpha = float(alpha)
        self.k = int(k)
        self.batch_size = int(batch_size)
        self.max_weight = None if max_weight is None else float(max_weight)
        self.weight_source = str(weight_source)
        if self.weight_source not in {"smiles", "bb_sum"}:
            raise ValueError("weight_source must be 'smiles' or 'bb_sum'")
        self.bb_weights = None
        if self.max_weight is not None and self.weight_source == "bb_sum":
            self.bb_weights = bb_weight_lookup({int(row.ID): str(row.SMILES) for row in bbs_df.itertuples(index=False)})

        d1, d2, d3 = tri_mod.PoolIO.split_by_pool(bbs_df)
        self.builder = tri_mod.TrimerBuilder(d1, d2, d3, reaction_mode=reaction_mode)
        self.proxy = load_autodock_proxy(autodock_model, device=device)
        self.pred_cache: Dict[str, float] = {}
        self.reward_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], Tuple[float, float]] = {}
        print(
            f"[Reward] autodock_proxy source enabled: artifact={autodock_model} "
            f"reward={self.reward_mode} threshold={self.threshold} alpha={self.alpha} "
            f"batch_size={self.batch_size} reaction_mode={reaction_mode} "
            f"max_weight={self.max_weight} weight_source={self.weight_source}"
        )

    def _ids_for_indices(self, xs: List[int]) -> List[int]:
        return [int(self.id_map[int(i)]) for i in xs]

    @staticmethod
    def _join_ids(ids: List[int]) -> str:
        return "|".join(str(int(x)) for x in ids)

    def _enumerate_product_records(self, b1_ids: List[int], b2_ids: List[int], b3_ids: List[int]) -> List[Tuple[str, Tuple[int, int, int]]]:
        records: List[Tuple[str, Tuple[int, int, int]]] = []
        for i in b1_ids:
            for j in b2_ids:
                for k in b3_ids:
                    try:
                        rec = self.builder.build_trimer_by_ids(int(i), int(j), int(k))
                    except Exception:
                        rec = None
                    smi = getattr(rec, "smi", None) if rec is not None else None
                    if smi:
                        records.append((str(smi), (int(i), int(j), int(k))))
        # Deduplicate by SMILES while preserving deterministic order/provenance.
        dedup: Dict[str, Tuple[int, int, int]] = {}
        for smi, ids in records:
            dedup.setdefault(smi, ids)
        return [(smi, ids) for smi, ids in dedup.items()]

    def _score_smiles(self, smiles: List[str]) -> List[float]:
        missing = [s for s in smiles if s not in self.pred_cache]
        if missing:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                y = self.proxy.predict_smiles(batch, batch_size=self.batch_size).ravel()
                m = min(len(batch), len(y))
                for smi, val in zip(batch[:m], y[:m]):
                    self.pred_cache[smi] = float(val)
        vals = []
        for smi in smiles:
            v = self.pred_cache.get(smi)
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
        return vals

    def _values_and_weights(self, product_records: List[Tuple[str, Tuple[int, int, int]]]) -> Tuple[List[float], Optional[np.ndarray]]:
        smiles = [smi for smi, _ in product_records]
        missing = [s for s in smiles if s not in self.pred_cache]
        if missing:
            self._score_smiles(missing)
        vals: List[float] = []
        weight_values: List[float] = []
        for smi, bb_ids in product_records:
            v = self.pred_cache.get(smi)
            if v is None or not math.isfinite(float(v)):
                continue
            vals.append(float(v))
            if self.max_weight is not None:
                if self.weight_source == "smiles":
                    weight_values.append(float(smiles_weight_array([smi])[0]))
                elif self.weight_source == "bb_sum" and self.bb_weights is not None:
                    weight_values.append(float(bb_sum_weights([bb_ids], bb_weights=self.bb_weights)[0]))
        weights = np.asarray(weight_values, dtype=float) if self.max_weight is not None else None
        return vals, weights

    def evaluate(self, triple_sets: Tuple[List[int], List[int], List[int]],
                 threshold: Optional[float] = None) -> Tuple[float, float, Dict[str, object]]:
        """Evaluate the reward for a terminal triple.

        ``threshold`` overrides the oracle's default threshold for this call,
        enabling per-trajectory threshold conditioning.
        """
        B1, B2, B3 = triple_sets
        # Use per-call threshold if provided, otherwise fall back to the oracle's default.
        effective_threshold = threshold if threshold is not None else self.threshold
        key = (tuple(B1), tuple(B2), tuple(B3), effective_threshold)
        b1_ids = self._ids_for_indices(B1)
        b2_ids = self._ids_for_indices(B2)
        b3_ids = self._ids_for_indices(B3)

        if key in self.reward_cache:
            R, log_R = self.reward_cache[key]
        else:
            product_records = self._enumerate_product_records(b1_ids, b2_ids, b3_ids)
            vals, weights = self._values_and_weights(product_records)
            if vals:
                R = float(
                    compute_library_reward(
                        np.asarray(vals, dtype=np.float64),
                        mode=self.reward_mode,
                        k=self.k,
                        threshold=effective_threshold if effective_threshold is not None else 0,
                        alpha=self.alpha,
                        weights=weights,
                        max_weight=self.max_weight,
                    )
                )
            else:
                R = float("nan")
            log_R = float(np.log(max(R, 1e-30))) if math.isfinite(R) else float("nan")
            self.reward_cache[key] = (R, log_R)

        meta = {
            "B1_id": self._join_ids(b1_ids),
            "B2_id": self._join_ids(b2_ids),
            "B3_id": self._join_ids(b3_ids),
        }
        return R, log_R, meta




def anneal_ratio(step, total):
    return (math.log(step+1)/math.log(1+total))**10


def _pareto_front_indices(neg_t: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated points for pairs (neg_t, reward).

    Point i dominates point j iff neg_t[i] <= neg_t[j] AND reward[i] >= reward[j],
    with at least one strict inequality.
    """
    n = len(neg_t)
    if n == 0:
        return np.array([], dtype=int)
    # Sort by neg_t ascending, then by reward descending for efficient sweep.
    order = np.lexsort((-rewards, neg_t))
    front_indices = []
    best_reward = -np.inf
    for idx in order:
        r = rewards[idx]
        # A point is non-dominated if its reward is strictly greater than all
        # points seen so far (which have neg_t <= current neg_t).
        # Points with equal neg_t: only the one(s) with max reward survive.
        if r > best_reward:
            front_indices.append(idx)
            best_reward = r
        elif r == best_reward:
            # Equal reward and equal or higher neg_t: dominated by earlier point
            # with same reward but lower/equal neg_t. Skip.
            pass
    return np.array(front_indices, dtype=int)


def pareto_topm_selection(records: List[Dict], m: int) -> List[Dict]:
    """Select m points by iteratively extracting the Pareto front.

    Points are pairs (-t, r(x)) where t is the threshold used when sampling x
    and r(x) is the reward. We repeatedly extract the non-dominated set
    (Pareto front) from the remaining points until at least m are selected.
    If the final front would exceed m, we take only what's needed, ordered
    by reward descending (ties broken by threshold descending) for determinism.

    Records must have "reward" and "threshold" keys.
    """
    if len(records) <= m:
        # Sort by reward descending for consistent output.
        return sorted(records, key=lambda t: (float(t["reward"]), float(t.get("threshold", 0.0))), reverse=True)

    remaining = list(records)
    selected: List[Dict] = []

    while len(selected) < m and remaining:
        neg_t = np.array([-float(rec.get("threshold", 0.0)) for rec in remaining])
        rewards = np.array([float(rec["reward"]) for rec in remaining])
        front_idx = _pareto_front_indices(neg_t, rewards)
        if len(front_idx) == 0:
            # Fallback: should not happen, but take top by reward.
            front_idx = np.argsort(-rewards)[: max(1, m - len(selected))]
        front_records = [remaining[i] for i in front_idx]
        # Sort front by reward descending, then threshold descending for determinism.
        front_records.sort(key=lambda t: (float(t["reward"]), float(t.get("threshold", 0.0))), reverse=True)
        needed = m - len(selected)
        selected.extend(front_records[:needed])
        # Remove front from remaining.
        front_set = set(front_idx.tolist())
        remaining = [rec for i, rec in enumerate(remaining) if i not in front_set]

    return selected[:m]

# ---------- Main training ----------

def main():


    # experiment = start(
    #     api_key="kFA8DNbuam0cOHYL6Y0w4h5Qk",
    #     project_name="del-gfn-2",
    #     workspace="bruno-joyal",

    # )
    ap = argparse.ArgumentParser("RxnFlow-style GFlowNet for DEL triples with TB and action subsampling")
    ap.add_argument("--bbs", default="data/bbs.csv", help="bbs.csv with 'SMILES' (optional 'Name','pool')")
    ap.add_argument("--size1", type=int, default=6, help="Target |B1|")
    ap.add_argument("--size2", type=int, default=6, help="Target |B2|")
    ap.add_argument("--size3", type=int, default=6, help="Target |B3|")
    ap.add_argument("--deepdel", default="models/deepdel.pt", help="Path to DeepDel model (for reward)")
    ap.add_argument(
        "--bb-pool-size",
        type=int,
        default=None,
        help="Limit all three pools to the same first N building blocks (by CSV order). If None, use all.",
    )


    # Features
    ap.add_argument("--bb-fp-bits", type=int, default=2048,
                    help="Raw action ECFP width D (per-cycle) used when --action-repr=ecfp. Action embedding layout is 3*D.")
    ap.add_argument("--bb-fp-radius", type=int, default=2)
    ap.add_argument(
        "--action-repr",
        choices=["ecfp", "phi"],
        default="ecfp",
        help=(
            "Action representation for the GFlowNet policy. 'ecfp' uses raw BB fingerprint blocks "
            "(legacy); 'phi' uses cycle-specific GFN Phi embeddings "
            "(Phi1(bb),0,0), (0,Phi2(bb),0), (0,0,Phi3(bb))."
        ),
    )
    ap.add_argument(
        "--phi-train-mode",
        choices=["frozen_table", "trainable_table", "trainable_network"],
        default="frozen_table",
        help=(
            "How to use the DeepDEL phi representation inside the GFlowNet. "
            "'frozen_table' preserves legacy behavior: precompute DeepDEL phi(bb) once and freeze it. "
            "'trainable_table' initializes one trainable per-BB embedding table from DeepDEL phi(bb). "
            "'trainable_network' initializes trainable GFN Phi networks from DeepDEL phi and recomputes "
            "Phi(X_all) with gradients each optimization step; this is the paper-faithful P2P mode."
        ),
    )
    ap.add_argument(
        "--phi-lr",
        type=float,
        default=None,
        help="Learning rate for trainable GFN Phi parameters. Defaults to --lr.",
    )
    ap.add_argument(
        "--phi-weight-decay",
        type=float,
        default=None,
        help="Weight decay for trainable GFN Phi parameters. Defaults to --weight-decay.",
    )
    ap.add_argument(
        "--phi-random-init",
        action="store_true",
        help="When set, trainable_network Φ is randomly initialized instead of loading DeepDEL weights.",
    )
    ap.add_argument(
        "--phi-hidden-dim",
        type=int,
        default=None,
        help="Override the hidden dimension for GFN Φ networks. If None, uses the DeepDEL checkpoint's hidden_dim.",
    )

    # Policy model sizes
    ap.add_argument("--joint-dim", type=int, default=512, help="Hidden dim shared by state/action towers and joint head")
    ap.add_argument("--state-dim", type=int, default=1024, help="Override hidden dim for state tower (default: joint-dim)")
    ap.add_argument("--action-dim", type=int, default=512, help="Override hidden dim for action tower (default: joint-dim)")

    # Tempered target: beta = reward exponent (NO behavior temperature)
    ap.add_argument("--beta", type=float, default=50.0, help="Reward exponent: target p(x) ∝ R(x)^beta (beta=1/tau).")
    # Backward-compat alias for older scripts (maps to beta)
    ap.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)

    # SUBSAMPLE parameters (RxnFlow-style)
    ap.add_argument(
        "--forbidden-bb1",
        default=None,
        help="Pipe-delimited list of combined-BB IDs that pool 1 (B1) must never select (e.g., '340|1215').",
    )
    ap.add_argument(
        "--forbidden-bb2",
        default=None,
        help="Pipe-delimited list of combined-BB IDs that pool 2 (B2) must never select.",
    )
    ap.add_argument(
        "--forbidden-bb3",
        default=None,
        help="Pipe-delimited list of combined-BB IDs that pool 3 (B3) must never select.",
    )
    ap.add_argument("--subsample-ratio", type=float, default=0.001,
                    help="Uniform per-cycle subsampling ratio for remaining BBs (0<r<=1).")
    ap.add_argument("--subsample-ratio-start", type=float, default=None,
                    help="Uniform per-cycle subsampling ratio for remaining BBs (0<r<=1).")
    ap.add_argument("--subsample-ratio-end", type=float, default=None,
                    help="Uniform per-cycle subsampling ratio for remaining BBs (0<r<=1).")    
    ap.add_argument("--min-per-cycle", type=int, default=8,
                    help="Minimum #candidates per cycle kept in the subsample (if cycle not full).")

    # Training
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-trajectories", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--logz-lr", type=float, default=1)
    ap.add_argument("--invert-reward", type=bool, default=False)

    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--phase-regularization",
        action="store_true",
        help="Enable phase-augmented GFlowNet regularization with learned action-state phases.",
    )
    ap.add_argument(
        "--phase-lambda",
        type=float,
        default=0.0,
        help="Nonnegative multiplier λ for phase loss mean(1-cos(sum_t φ(a_t|s_t))).",
    )

    # Logging / outputs
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--sample-n", type=int, default=256)

    # Pure-inference rollout after training
    ap.add_argument("--inference-steps", type=int, default=0,
                    help="Number of additional trajectories to sample in pure inference mode after training (0=disabled).")
    ap.add_argument("--inference-subsample-ratio", type=float, default=None,
                    help="Fixed subsample ratio for post-training inference rollouts. Defaults to final training ratio.")

    # Top-m reporting
    ap.add_argument("--topm", type=int, default=100, help="Report and save top-m terminal states by reward at end.")

    # Reward source. In both modes DeepDEL is loaded for φ state embeddings;
    # this switch only controls the terminal reward used in TB.
    ap.add_argument(
        "--gfn-reward-source",
        choices=["deepdel", "autodock_proxy"],
        default="deepdel",
        help="Terminal reward source for TB. 'deepdel' preserves legacy behavior; 'autodock_proxy' enumerates/scales libraries with the autodock proxy.",
    )
    ap.add_argument("--autodock-model", type=str, default=None, help="Autodock proxy artifact used when --gfn-reward-source=autodock_proxy.")
    ap.add_argument("--autodock-proxy-reward", type=str, default="threshold", choices=REWARD_MODES, help="Library reward reducer for autodock_proxy mode.")
    ap.add_argument("--threshold", type=float, default=None, help="Threshold for autodock_proxy threshold reward mode.")
    ap.add_argument("--alpha", type=float, default=1.0, help="Smoothing alpha for autodock_proxy threshold reward mode.")
    ap.add_argument("--k", type=int, default=10, help="k for autodock_proxy topk_mean reward mode.")
    ap.add_argument("--autodock-proxy-batch-size", type=int, default=4096, help="Batch size for autodock proxy molecule scoring.")
    ap.add_argument("--autodock-proxy-device", type=str, default=None, help="Device for autodock proxy scoring; defaults to --device.")
    ap.add_argument("--max-weight", type=float, default=None, help="Maximum molecular weight allowed to contribute to autodock_proxy library rewards.")
    ap.add_argument("--weight-source", choices=["smiles", "bb_sum"], default="bb_sum", help="How to compute molecular weights for --max-weight in autodock_proxy reward mode.")
    ap.add_argument("--reaction-mode", type=str, default="amide_sulfonamide", choices=["amide_sulfonamide", "amide_amide", "amide_amide_legacy"], help="DEL reaction mode used to enumerate autodock_proxy reward libraries.")
    ap.add_argument("--append-encountered-dataset", type=str, default=None, help="If set in autodock_proxy mode, append every finite encountered library as B1_id,B2_id,B3_id,y=log_reward.")
    ap.add_argument("--encountered-out", type=str, default=None, help="CSV path for all encountered autodock_proxy-reward terminal libraries.")

    # CPU / cache
    ap.add_argument("--cpu-threads", type=int, default=None, help="Limit torch CPU threads to avoid oversubscription.")
    ap.add_argument(
        "--ecfp-cache-dir",
        type=str,
        default=None,
        help="Directory to cache ECFP tables (avoids recomputing each run).",
    )

    # Resume
    ap.add_argument("--policy-ckpt", type=str, default=None, help="Path to a saved policy checkpoint to resume (loads policy_state and logZ)")

    # --- Logging / analytics controls ---
    ap.add_argument("--log-metrics", type=int, default=10, help="Write per-step metrics.csv/jsonl and plots.")
    ap.add_argument("--eval-every", type=int, default=0, help="If >0, compute Top-K stats every N steps (costly).")
    ap.add_argument("--topm-distance", type=str, default="tanimoto", choices=["tanimoto","cosine","euclidean"],
                    help="Distance metric for Top-M diversity.")
    ap.add_argument("--plot-logy-loss", type=int, default=0, help="Deprecated/no effect: loss plots already show log(1+TB loss).")
    ap.add_argument("--no-svg", type=int, default=1, help="If 1, do not save SVG versions of plots.")
    ap.add_argument("--progress", type=int, default=1, help="Print lightweight progress updates.")
    ap.add_argument("--progress-nested", type=int, default=0, help="Print lightweight progress for inner rollouts (slower).")
    ap.add_argument("--batched-rollouts", action="store_true", help="Batch policy evaluation across trajectories (pads variable-K candidate sets).")
    ap.add_argument("--time-breakdown", type=int, default=0, help="If 1, print a simple per-step time breakdown.")
    ap.add_argument("--model_is_logprobs", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument(
        "--append-molecular-weight",
        action="store_true",
        help=(
            "Append the molecular weight of each building block as an extra "
            "scalar feature to the ECFP fingerprint, increasing the input "
            "dimension from bb_fp_bits to bb_fp_bits + 1."
        ),
    )

    # Threshold conditioning
    ap.add_argument("--threshold-min", type=float, default=None,
                    help="Lower bound of the uniform threshold sampling interval for GFN training (default: -12.0).")
    ap.add_argument("--threshold-max", type=float, default=None,
                    help="Upper bound of the uniform threshold sampling interval for GFN training (default: -6.0).")
    ap.add_argument("--fourier-n-freqs", type=int, default=0,
                    help="Number of Fourier frequencies for policy threshold conditioning (0=disabled, or read from DeepDEL checkpoint).")
    ap.add_argument("--fourier-freq-scale", type=float, default=1.5,
                    help="Geometric frequency scale for Fourier encoding.")
    ap.add_argument("--fourier-linear", action="store_true",
                    help="Use linear frequencies (i+1) instead of geometric (freq_scale^i).")
    ap.add_argument("--fourier-append-raw", action="store_true",
                    help="Append the raw threshold value to the Fourier encoding.")
    ap.add_argument("--fourier-threshold-center", type=float, default=None)
    ap.add_argument("--fourier-threshold-scale", type=float, default=None)
    ap.add_argument("--topm-threshold", type=float, default=None,
                    help="Fixed threshold for top-m selection during inference. When set, top-m is selected only from inference steps at this threshold.")
    ap.add_argument("--strip-conditioning", action="store_true",
                    help="Resume from a threshold-conditioned policy checkpoint and fold the fixed --threshold into a "
                         "non-conditioned policy. Requires --policy-ckpt (a conditioned pretraining checkpoint) and "
                         "--threshold; the current run must NOT pass --fourier-n-freqs > 0. Used for fixed-threshold "
                         "specialization after threshold-conditioned pretraining.")

    args = ap.parse_args()


    hyper_params = {
        "learning_rate": args.lr,
        "logz_learning_rate": args.logz_lr,
        "steps": args.steps,
        "batch_size": args.batch_trajectories,
    }
    # Handle backward-compat temperature alias
    if args.temperature is not None and args.beta == 1.0:
        args.beta = float(args.temperature)
        print(f"[Compat] Using --temperature={args.temperature} as beta.")
    if args.phase_lambda < 0:
        raise ValueError("--phase-lambda must be nonnegative.")
    phase_regularization_enabled = bool(args.phase_regularization and args.phase_lambda > 0)
    if args.phase_lambda > 0 and not args.phase_regularization:
        print("[Phase] --phase-lambda > 0 but --phase-regularization is not set; phase loss is disabled.")

    # Resolve threshold sampling interval (backward-compatible with --threshold).
    if args.threshold is not None:
        if args.threshold_min is not None or args.threshold_max is not None:
            raise ValueError(
                "--threshold cannot be combined with --threshold-min/--threshold-max."
            )
        args.threshold_min = args.threshold
        args.threshold_max = args.threshold
    else:
        if args.threshold_min is None:
            args.threshold_min = -12.0
        if args.threshold_max is None:
            args.threshold_max = -6.0
    args.threshold_min = float(args.threshold_min)
    args.threshold_max = float(args.threshold_max)
    if args.threshold_min > args.threshold_max:
        raise ValueError(
            f"--threshold-min ({args.threshold_min}) cannot exceed --threshold-max ({args.threshold_max})."
        )
    threshold_conditioning_enabled = args.fourier_n_freqs > 0
    print(
        f"[Threshold] interval=[{args.threshold_min}, {args.threshold_max}] "
        f"fourier_n_freqs={args.fourier_n_freqs} conditioning_enabled={threshold_conditioning_enabled}"
    )

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

    # Enable TF32 (fast FP32) on NVIDIA Ampere+ and set preferred matmul precision
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # train_gfn is a single-process / single-GPU script: every model & tensor
        # below is placed on a single `device`. If SLURM gave us more than one
        # CUDA device, the others will sit idle for the whole run -- that is a
        # configuration bug worth flagging loudly so it stops happening silently.
        try:
            n_gpu = torch.cuda.device_count()
            cur = torch.cuda.current_device()
            print(
                f"[CUDA] visible devices = {n_gpu} "
                f"(using cuda:{cur} = {torch.cuda.get_device_name(cur)})"
            )
            if n_gpu > 1:
                print(
                    "[CUDA] WARNING: train_gfn currently uses a single GPU. "
                    f"The other {n_gpu - 1} CUDA device(s) visible to this process will "
                    "sit idle. Either request a single-GPU SLURM allocation "
                    "(e.g. --gpus-per-node=a100:1), launch independent runs per "
                    "GPU via CUDA_VISIBLE_DEVICES, or extend this script with "
                    "DDP to make use of them."
                )
        except Exception as e:  # pragma: no cover - defensive
            print(f"[CUDA] WARN: could not query CUDA devices: {e}")

    use_amp = (device.type == "cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    # Proper AMP requires a GradScaler for stability.
    # (We still run the policy forward in FP32 inside rollouts.)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    recorder = MetricsRecorder(args.outdir, ema_alpha=0.98) if args.log_metrics else None
    # ---- Load BBs and compute raw ECFPs / model input features ----
    df_bbs = load_bbs(args.bbs)
    smiles = df_bbs["SMILES"].tolist()
    N = len(smiles)
    id_map = df_bbs["ID"].to_numpy() if "ID" in df_bbs.columns else None

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

    # Pool-aware action universe: if the bbs.csv has a `pool` column with values
    # in {1,2,3}, restrict each cycle's policy to only emit indices from the
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
                f"Pool column present in {args.bbs}, but at least one of pools {1,2,3} is empty: "
                f"|pool1|={len(idx1)}, |pool2|={len(idx2)}, |pool3|={len(idx3)}."
            )
        allowed_lists = [idx1, idx2, idx3]
        print(
            f"[Pools] Pool-aware action sampling enabled: "
            f"|pool1|={len(idx1)}, |pool2|={len(idx2)}, |pool3|={len(idx3)}"
        )
    else:
        print(f"[Pools] No pool column found in {args.bbs}; sampling from full BB universe (N={N}).")

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
            raise ValueError("Forbidden BB flags require the combined BB CSV to include an ID column.")
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
            print(
                f"[Forbidden] Pool {cycle}: excluding {len(found)} BB(s) from candidate set."
            )
        if missing:
            print(
                f"[Forbidden] Pool {cycle}: {len(missing)} ID(s) not found in {args.bbs}: {sorted(set(missing))}"
            )

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
            print(
                f"[Forbidden] Pool {cycle}: removed {removed} BB(s); remaining={len(allowed_lists[cycle - 1])}."
            )
        for cycle in (1, 2, 3):
            remaining = len(allowed_lists[cycle - 1])
            need = sizes[cycle - 1]
            if remaining < need:
                raise ValueError(
                    f"After applying forbidden BBs, pool {cycle} only has {remaining} candidate(s) but size{cycle}={need}."
                )

    # Apply pool size limit (--bb-pool-size). When no pool column exists,
    # all three cycles share the same first N building blocks.
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
    else:
        allowed_per_cycle = None
    cache_dir = args.ecfp_cache_dir or os.path.join(args.outdir, "cache")
    bb_fp_bits = int(args.bb_fp_bits)
    mw_suffix = "_mw" if args.append_molecular_weight else ""
    bb_ecfp = load_or_create_ecfp_cache(
        smiles,
        n_bits=bb_fp_bits,
        radius=args.bb_fp_radius,
        cache_path=os.path.join(cache_dir, f"ecfp{bb_fp_bits}_r{args.bb_fp_radius}{mw_suffix}.npz"),
        progress_desc=f"ECFP({bb_fp_bits}) for actions",
        progress_enabled=False,
        append_molecular_weight=args.append_molecular_weight,
    )
    # Raw ECFP table is used directly in legacy action mode and retained for
    # existing end-of-run diversity diagnostics in phi action mode.
    bb_ecfp_table = torch.from_numpy(bb_ecfp).to(device)  # [N, bb_fp_bits]

    # ---- Load frozen TripleDeepSet checkpoint ----
    ckpt = torch.load(args.deepdel, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    # Read actual input dimension from checkpoint, falling back to bb_fp_bits
    # (which may not account for MW augmentation in older checkpoints).
    d_in = int(ckpt.get("d_in", ckpt_args.get("bb_fp_bits", 2048)))
    d_h   = ckpt_args.get("hidden_dim", 256)
    d_rho = ckpt_args.get("rho_dim", 256)
    dropout = ckpt_args.get("dropout", 0.1)
    shared_phi = ckpt_args.get("shared_phi", False)
    pooling = ckpt_args.get("pooling", "mean")
    output_head = ckpt_args.get("output_head", "linear")
    lib_size = ckpt_args.get("lib_size", None)

    # Read Fourier threshold conditioning args from checkpoint (if present).
    # The DeepDEL model MUST use the checkpoint's Fourier args to load the
    # state_dict correctly. The GFN policy can use its own Fourier args from CLI.
    ckpt_fourier_n_freqs = int(ckpt_args.get("fourier_n_freqs", 0))
    ckpt_fourier_freq_scale = float(ckpt_args.get("fourier_freq_scale", 1.5))
    ckpt_fourier_linear = bool(ckpt_args.get("fourier_linear", False))
    ckpt_fourier_append_raw = bool(ckpt_args.get("fourier_append_raw", False))
    ckpt_fourier_threshold_center = ckpt_args.get("fourier_threshold_center")
    ckpt_fourier_threshold_scale = ckpt_args.get("fourier_threshold_scale")
    ckpt_fourier_condition = str(ckpt_args.get("fourier_condition", "rho"))

    # Store GFN policy Fourier args from CLI (may differ from checkpoint).
    gfn_fourier_n_freqs = args.fourier_n_freqs
    gfn_fourier_freq_scale = args.fourier_freq_scale
    gfn_fourier_linear = args.fourier_linear
    gfn_fourier_append_raw = args.fourier_append_raw
    gfn_fourier_threshold_center = args.fourier_threshold_center
    gfn_fourier_threshold_scale = args.fourier_threshold_scale

    # If CLI Fourier args are not set, use checkpoint's args for the policy too.
    if gfn_fourier_n_freqs == 0 and ckpt_fourier_n_freqs > 0:
        gfn_fourier_n_freqs = ckpt_fourier_n_freqs
        gfn_fourier_freq_scale = ckpt_fourier_freq_scale
        gfn_fourier_linear = ckpt_fourier_linear
        gfn_fourier_append_raw = ckpt_fourier_append_raw
        gfn_fourier_threshold_center = ckpt_fourier_threshold_center
        gfn_fourier_threshold_scale = ckpt_fourier_threshold_scale
        print(
            f"[Fourier] Using checkpoint Fourier args for GFN policy: n_freqs={gfn_fourier_n_freqs}, "
            f"freq_scale={gfn_fourier_freq_scale}, linear={gfn_fourier_linear}, "
            f"append_raw={gfn_fourier_append_raw}"
        )
    elif gfn_fourier_n_freqs > 0:
        print(
            f"[Fourier] Using CLI Fourier args for GFN policy: n_freqs={gfn_fourier_n_freqs}, "
            f"freq_scale={gfn_fourier_freq_scale}, linear={gfn_fourier_linear}, "
            f"append_raw={gfn_fourier_append_raw}"
        )

    log_target = bool(ckpt_args.get("log_target", False))

    # DeepDEL model uses checkpoint's Fourier args (required for state_dict loading).
    triple_model = TripleDeepSet(d_in=d_in, d_hidden=d_h, d_rho=d_rho, dropout=dropout,
                                 shared_phi=shared_phi, pooling=pooling,
                                 output_head=output_head, reward_bound_k=lib_size,
                                 fourier_n_freqs=ckpt_fourier_n_freqs,
                                 fourier_freq_scale=ckpt_fourier_freq_scale,
                                 fourier_linear=ckpt_fourier_linear,
                                 fourier_append_raw=ckpt_fourier_append_raw,
                                 fourier_condition=ckpt_fourier_condition,
                                 fourier_threshold_center=ckpt_fourier_threshold_center,
                                 fourier_threshold_scale=ckpt_fourier_threshold_scale)
    triple_model.load_state_dict(ckpt["model_state"])
    print(f"[Info] Loaded DeepDEL TripleDeepSet pooling={pooling} output_head={output_head} fourier_n_freqs={ckpt_fourier_n_freqs}")
    triple_model.to(device).eval()
    for p in triple_model.parameters(): p.requires_grad_(False)


    if d_in != bb_fp_bits:
        print(
            f"[Info] TripleDeepSet d_in={d_in} differs from action ECFP width "
            f"bb_fp_bits={bb_fp_bits}; that's fine (the action encoder is "
            f"independent of the φ encoder)."
        )

    # Build model-input fingerprints for state φ sums
    print(f"Building model-input fingerprints for state φ (d_in={d_in}, radius={args.bb_fp_radius}) ...")
    # When MW augmentation is enabled, the φ input dimension already includes the
    # MW feature (d_in = bb_fp_bits + 1), so we must also append MW to these caches.
    X_in = load_or_create_ecfp_cache(
        smiles,
        n_bits=int(d_in) - (1 if args.append_molecular_weight else 0),
        radius=args.bb_fp_radius,
        cache_path=os.path.join(cache_dir, f"ecfp{int(d_in)}_r{args.bb_fp_radius}{mw_suffix}.npz"),
        progress_desc=f"ECFP({d_in}) for state φ",
        progress_enabled=False,
        append_molecular_weight=args.append_molecular_weight,
    )
    X_all = torch.from_numpy(X_in).to(device)  # [N, d_in]

    # ---- GFN Phi hidden dimension ----
    # --phi-hidden-dim overrides the DeepDEL checkpoint's d_h for GFN Φ.
    # This allows the GFN Phi networks to use a different (smaller) hidden
    # dimension than the frozen reward model, controlling total trainable
    # parameters without touching the reward oracle.
    phi_hidden_dim = int(args.phi_hidden_dim) if args.phi_hidden_dim is not None else int(d_h)
    if args.phi_hidden_dim is not None:
        print(f"[Phi] Overriding hidden_dim: checkpoint={int(d_h)} → phi_hidden_dim={phi_hidden_dim}")
    else:
        print(f"[Phi] Using checkpoint hidden_dim={phi_hidden_dim} for GFN Phi.")

    # Precompute DeepDEL φ tables once for initialization. These tensors are
    # frozen snapshots; trainable GFN Φ modes below use them only as initial
    # values, not as static policy features.
    with torch.no_grad():
        init_phi1_table = triple_model.phi(X_all)   # [N, d_h]
        init_phi2_table = triple_model.phi2(X_all)  # [N, d_h]
        init_phi3_table = triple_model.phi3(X_all)  # [N, d_h]

    gfn_phi = GFlowNetPhi(
        mode=args.phi_train_mode,
        d_in=int(d_in),
        d_hidden=phi_hidden_dim,
        dropout=float(dropout),
        shared_phi=bool(shared_phi),
        init_tables=(init_phi1_table, init_phi2_table, init_phi3_table),
        init_model=None if args.phi_random_init else triple_model,
    ).to(device)
    if args.phi_random_init and args.phi_train_mode == "trainable_network":
        print("[Phi] Random init — Φ networks trained from scratch (not initialized from DeepDEL).")
    n_phi_params = sum(p.numel() for p in gfn_phi.parameters() if p.requires_grad)
    print(
        f"[Phi] train_mode={args.phi_train_mode} shared_phi={bool(shared_phi)} "
        f"phi_hidden_dim={phi_hidden_dim} trainable_params={n_phi_params:,}"
    )

    # Determine the action representation dimensionality. The actual action
    # cache is rebuilt each optimizer step in phi action mode so trainable Φ
    # tables remain attached to the TB computation graph.
    if args.action_repr == "ecfp":
        action_base_dim = bb_fp_bits
    elif args.action_repr == "phi":
        action_base_dim = phi_hidden_dim
    else:  # argparse choices should make this unreachable
        raise ValueError(f"Unknown --action-repr={args.action_repr!r}")
    action_in_dim = 3 * int(action_base_dim)
    print(
        f"[Policy] action_repr={args.action_repr} action_base_dim={action_base_dim} "
        f"action_in_dim={action_in_dim}"
    )

    # ---- Policy + TB parameters ----
    d_state = args.state_dim if args.state_dim is not None else args.joint_dim
    d_action = args.action_dim if args.action_dim is not None else args.joint_dim
    print(f"[Policy] d_state={d_state}, d_action={d_action}, d_joint={args.joint_dim}")

    policy = JointEdgePolicy(
        d_h=phi_hidden_dim,
        d_state=d_state,
        d_action=d_action,
        d_joint=args.joint_dim,
        action_in_dim=action_in_dim,
        interaction="none",     # "none" | "hadamard" | "film" | "bilinear"
        bilinear_rank=8,
        bilinear_out=32,
        bilinear_in=256,
        phase_regularization=phase_regularization_enabled,
        fourier_n_freqs=gfn_fourier_n_freqs,
        fourier_freq_scale=gfn_fourier_freq_scale,
        fourier_linear=gfn_fourier_linear,
        fourier_append_raw=gfn_fourier_append_raw,
        fourier_threshold_center=gfn_fourier_threshold_center,
        fourier_threshold_scale=gfn_fourier_threshold_scale,
    )
    # IMPORTANT: move policy to the selected device.
    # Without this, state/action embeddings created on CUDA will be fed into a CPU policy,
    # causing: "Expected all tensors to be on the same device ... mat1 on cuda:0 ... on cpu".
    policy = policy.to(device)

    # Threshold-conditioned logZ: when Fourier conditioning is enabled, logZ
    # is a function of the threshold. Otherwise, it's a learnable scalar.
    logZ_net = ConditionalLogZ(
        fourier_n_freqs=gfn_fourier_n_freqs,
        fourier_freq_scale=gfn_fourier_freq_scale,
        fourier_linear=gfn_fourier_linear,
        fourier_append_raw=gfn_fourier_append_raw,
        fourier_threshold_center=gfn_fourier_threshold_center,
        fourier_threshold_scale=gfn_fourier_threshold_scale,
        hidden_dim=128,
    ).to(device)
    print(f"[LogZ] ConditionalLogZ fourier_n_freqs={gfn_fourier_n_freqs}")

    # ---- Threshold-conditioning strip (fixed-threshold specialization) ----
    # When active, we resume from a conditioned pretraining checkpoint and fold
    # the learned threshold conditioning at `--threshold` into a non-conditioned
    # policy (no Fourier args). The resulting network is architecturally
    # identical to the `regular` variant and differs only in initialization.
    if args.strip_conditioning:
        if args.policy_ckpt is None or not os.path.exists(args.policy_ckpt):
            raise ValueError(
                "--strip-conditioning requires --policy-ckpt pointing to an existing "
                "threshold-conditioned pretraining checkpoint."
            )
        if gfn_fourier_n_freqs > 0:
            raise ValueError(
                "--strip-conditioning requires a non-conditioned policy; do not pass "
                "--fourier-n-freqs > 0 (nor a threshold-conditioned DeepDEL)."
            )
        if args.threshold is None:
            raise ValueError("--strip-conditioning requires --threshold to fix the specialized threshold.")
        print(
            f"[Strip] Will fold threshold conditioning at τ={args.threshold} into a non-conditioned policy "
            "initialized from the pretraining checkpoint."
        )

    # ---- Optional resume from policy checkpoint ----
    resume_checkpoint = None
    if args.policy_ckpt is not None:
        if not os.path.exists(args.policy_ckpt):
            # In the active-learning inner loop the same --policy-ckpt path is
            # used as the warmup output and the main-training input. During the
            # warmup command it normally does not exist yet, so start fresh and
            # let --save-model create it at the end.
            print(f"[Resume] No existing policy checkpoint at {args.policy_ckpt}; starting from scratch.")
        else:
            try:
                pck = torch.load(args.policy_ckpt, map_location="cpu")
                resume_checkpoint = pck
                pck_args = pck.get("args", {}) or {}
                pck_meta = pck.get("meta", {}) or {}
                pck_fourier = _ckpt_fourier_args(pck_args, pck_meta)
                pck_action_repr = pck.get("action_repr", pck_args.get("action_repr"))
                pck_action_in_dim = pck.get("action_in_dim", pck_meta.get("action_in_dim"))
                pck_phi_train_mode = pck.get(
                    "phi_train_mode",
                    pck_args.get("phi_train_mode", pck_meta.get("phi_train_mode")),
                )
                if pck_action_repr is not None and str(pck_action_repr) != str(args.action_repr):
                    raise ValueError(
                        f"Policy checkpoint action_repr mismatch: checkpoint={pck_action_repr!r}, "
                        f"current={args.action_repr!r}. Start a fresh policy checkpoint or use the same mode."
                    )
                if pck_phi_train_mode is not None and str(pck_phi_train_mode) != str(args.phi_train_mode):
                    raise ValueError(
                        f"Policy checkpoint phi_train_mode mismatch: checkpoint={pck_phi_train_mode!r}, "
                        f"current={args.phi_train_mode!r}. Start a fresh policy checkpoint or use the same mode."
                    )
                if pck_action_in_dim is not None and int(pck_action_in_dim) != int(action_in_dim):
                    raise ValueError(
                        f"Policy checkpoint action_in_dim mismatch: checkpoint={int(pck_action_in_dim)}, "
                        f"current={int(action_in_dim)}. This usually means action_repr, bb_fp_bits, "
                        "or the DeepDEL hidden_dim changed."
                    )
                state = pck.get("policy_state", None)
                if state is None:
                    if args.strip_conditioning:
                        raise ValueError(
                            "--strip-conditioning: the pretrain checkpoint has no 'policy_state'; "
                            "refusing to start a specialized run from an empty checkpoint."
                        )
                    print(f"[Resume] WARNING: No 'policy_state' in {args.policy_ckpt}; skipping policy load.")
                else:
                    if args.strip_conditioning:
                        if pck_fourier["n_freqs"] <= 0:
                            raise ValueError(
                                "--strip-conditioning: the pretrain checkpoint is NOT threshold-conditioned "
                                f"(fourier_n_freqs={pck_fourier['n_freqs']})."
                            )
                        state, folded_fdim = _fold_threshold_into_state_proj(
                            state,
                            d_h=phi_hidden_dim,
                            threshold=args.threshold,
                            **pck_fourier,
                        )
                        print(
                            f"[Strip] Folded {folded_fdim} threshold-conditioning dims into state_proj "
                            f"at τ={args.threshold}."
                        )
                    missing, unexpected = policy.load_state_dict(state, strict=False)
                    if missing or unexpected:
                        print(f"[Resume] Loaded with non-strict match. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                    if args.strip_conditioning:
                        _verify_folded_policy(
                            policy,
                            pck.get("policy_state", {}),
                            pck_fourier,
                            d_h=phi_hidden_dim,
                            d_state=d_state,
                            d_action=d_action,
                            d_joint=args.joint_dim,
                            action_in_dim=action_in_dim,
                            threshold=args.threshold,
                            device=device,
                        )
                phi_state = pck.get("gfn_phi_state", None)
                if phi_state is not None:
                    missing_phi, unexpected_phi = gfn_phi.load_state_dict(phi_state, strict=False)
                    if missing_phi or unexpected_phi:
                        print(
                            f"[Resume] Loaded GFN Phi with non-strict match. "
                            f"Missing keys: {len(missing_phi)}, Unexpected keys: {len(unexpected_phi)}"
                        )
                elif args.phi_train_mode != "frozen_table":
                    print(
                        f"[Resume] WARNING: No 'gfn_phi_state' in {args.policy_ckpt}; "
                        "GFN Phi remains initialized from the current DeepDEL checkpoint."
                    )
                # Make sure we end up on the target device even after loading.
                policy = policy.to(device)
                gfn_phi = gfn_phi.to(device)
                # Restore the partition function.
                # New checkpoints save `logZ_net_state` (a ConditionalLogZ). On the
                # strip path we evaluate the conditioned logZ at the specialized
                # threshold and seed the non-conditioned scalar with that value.
                logz_state = pck.get("logZ_net_state", None)
                if logz_state is not None:
                    if args.strip_conditioning:
                        tmp_logz = ConditionalLogZ(
                            fourier_n_freqs=pck_fourier["n_freqs"],
                            fourier_freq_scale=pck_fourier["freq_scale"],
                            fourier_linear=pck_fourier["linear"],
                            fourier_append_raw=pck_fourier["append_raw"],
                            fourier_threshold_center=pck_fourier["threshold_center"],
                            fourier_threshold_scale=pck_fourier["threshold_scale"],
                            hidden_dim=128,
                        )
                        tmp_logz.load_state_dict(logz_state)
                        with torch.no_grad():
                            specialized_logz = float(tmp_logz(torch.tensor([args.threshold])).item())
                        logZ_net.logZ_scalar.data.fill_(specialized_logz)
                        logZ_net = logZ_net.to(device)
                        print(
                            f"[Strip] Specialized logZ(τ={args.threshold}) = {specialized_logz:.4f} "
                            "set as the scalar logZ initialization."
                        )
                    elif int(getattr(logZ_net, "fourier_n_freqs", 0)) == int(pck_fourier["n_freqs"]):
                        missing_lz, unexpected_lz = logZ_net.load_state_dict(logz_state, strict=False)
                        if missing_lz or unexpected_lz:
                            print(
                                f"[Resume] logZ_net loaded with non-strict match. "
                                f"Missing keys: {len(missing_lz)}, Unexpected keys: {len(unexpected_lz)}"
                            )
                    else:
                        print(
                            "[Resume] WARNING: logZ_net architecture mismatch with checkpoint; "
                            "logZ restarts from its initialization."
                        )
                elif "logZ" in pck:
                    # Legacy scalar checkpoints.
                    if getattr(logZ_net, "logZ_scalar", None) is not None:
                        with torch.no_grad():
                            logZ_net.logZ_scalar.data.fill_(float(pck["logZ"]))
                        logZ_net = logZ_net.to(device)
                        print(f"[Resume] logZ set to {float(pck['logZ']):.4f}")
                print(f"[Resume] Loaded checkpoint from {args.policy_ckpt}")
            except Exception as e:
                raise RuntimeError(f"[Resume] ERROR loading {args.policy_ckpt}: {e}") from e

    def _assert_same_device(*tensors: torch.Tensor, name: str = ""):
        """Debug helper: ensure all tensors are on the same device."""
        devs = [t.device for t in tensors if isinstance(t, torch.Tensor)]
        if not devs:
            return
        if any(d != devs[0] for d in devs[1:]):
            raise RuntimeError(f"Device mismatch{name and ' in ' + name}: {devs}")
    
    opt_param_groups = [
        {"params": policy.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": logZ_net.parameters(), "lr": args.logz_lr},
    ]
    phi_trainable_params = [p for p in gfn_phi.parameters() if p.requires_grad]
    if phi_trainable_params:
        opt_param_groups.append(
            {
                "params": phi_trainable_params,
                "lr": float(args.phi_lr if args.phi_lr is not None else args.lr),
                "weight_decay": float(
                    args.phi_weight_decay if args.phi_weight_decay is not None else args.weight_decay
                ),
            }
        )
    opt = torch.optim.Adam(opt_param_groups)
    if resume_checkpoint is not None and "optimizer_state" in resume_checkpoint:
        if args.strip_conditioning:
            # The strip fold changes state_proj/logZ parameter shapes, so Adam's
            # saved moments cannot be transferred. We restart the optimizer on
            # the transferred weights.
            print(
                "[Resume] Optimizer state not transferred (--strip-conditioning changes "
                "parameter shapes); starting a fresh optimizer on the specialized weights."
            )
        else:
            try:
                opt.load_state_dict(resume_checkpoint["optimizer_state"])
                print(f"[Resume] Optimizer state loaded from {args.policy_ckpt}")
            except Exception as exc:
                raise RuntimeError(f"[Resume] ERROR loading optimizer state from {args.policy_ckpt}: {exc}") from exc
    n_policy_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_logz_params = sum(p.numel() for p in logZ_net.parameters() if p.requires_grad)
    n_total_params = n_policy_params + n_logz_params + n_phi_params
    print(
        f"[Params] policy={n_policy_params:,} logZ={n_logz_params:,} "
        f"phi={n_phi_params:,} total_trainable={n_total_params:,}"
    )
    trainable_modules_for_clipping = [policy, logZ_net]
    if phi_trainable_params:
        trainable_modules_for_clipping.append(gfn_phi)
    static_ecfp_action_cache = CachedActionEmbeddings(bb_ecfp_table) if args.action_repr == "ecfp" else None

    # ---- Training loop ----
    T = args.size1 + args.size2 + args.size3
    phase_target = (2.0 * math.pi / float(T)) if T > 0 else float("nan")
    print(f"[Info] Episode length T = {T}, N_BB = {N}, d_h = {d_h}")
    print(f"[Info] Subsampling ratio per cycle = {args.subsample_ratio}, min_per_cycle = {args.min_per_cycle}")
    print(
        f"[Phase] enabled={phase_regularization_enabled} lambda={args.phase_lambda:g} "
        f"target_per_step=2π/T={phase_target:.6g} rad"
    )
    if device.type == "cuda":
        print(f"[CUDA] AMP enabled: {use_amp} | TF32: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}")

    ema_loss = None
    rng = np.random.default_rng(args.seed if args.seed is not None else None)

    # Tracking rewards & top-k
    rewards_seen: List[float] = []
    yhats_seen: List[float] = []
    topm_records: List[Dict[str, object]] = []
    encountered_rows: List[Dict[str, object]] = []

    # Outer training loop: we print periodically based on --log-interval.
    for step in range(1, args.steps + 1):
        batch_logps_sum = []
        batch_beta_logR = []
        batch_phase_sums = []
        batch_phase_subsample_stats: List[Dict[str, float]] = []
        batch_rewards_this_step: List[float] = []   # [ADD] collect rewards for this batch
        terminals_in_batch = 0                      # [ADD]

        # Fresh env per trajectory
        if args.subsample_ratio_start is not None:
            ss_ratio = args.subsample_ratio_start + anneal_ratio(step, args.steps) * (args.subsample_ratio_end - args.subsample_ratio_start)
        else:
            ss_ratio = args.subsample_ratio

        # Build the current GFN Φ tables for this optimization step. In
        # trainable modes these tensors are part of the autograd graph, so the
        # TB loss below updates Φ instead of merely training P on static φ(bb)
        # features. Raw-ECFP action mode still uses Φ for state embeddings.
        phi1_table, phi2_table, phi3_table = gfn_phi.tables(X_all)
        if args.action_repr == "ecfp":
            action_env_table = bb_ecfp_table
            action_cache = static_ecfp_action_cache
        elif args.action_repr == "phi":
            action_env_table = phi1_table
            action_cache = CachedActionEmbeddings(phi1_table, phi2_table, phi3_table)
        else:
            raise ValueError(f"Unknown --action-repr={args.action_repr!r}")

        # Sample thresholds for this batch of trajectories.
        batch_thresholds = rng.uniform(
            args.threshold_min, args.threshold_max, size=args.batch_trajectories
        )

        t0 = time.time()
        if args.batched_rollouts:
            trajs = rollout_trajectories_batched(
                phi1_table=phi1_table,
                phi2_table=phi2_table,
                phi3_table=phi3_table,
                sizes=sizes,
                policy=policy,
                action_cache=action_cache,
                rng=rng,
                subsample_ratio=ss_ratio,
                min_per_cycle=args.min_per_cycle,
                batch_trajectories=args.batch_trajectories,
                device=device,
                use_amp=use_amp,
                epsilon=args.epsilon,
                allowed_per_cycle=allowed_per_cycle,
                phase_regularization=phase_regularization_enabled,
                phase_target=phase_target,
                thresholds=batch_thresholds,
            )
        else:
            inner_iter = progress_iter(
                range(args.batch_trajectories),
                total=args.batch_trajectories,
                desc=f"Rollouts@{step}",
                enabled=bool(args.progress and args.progress_nested),
            )
            trajs = []
            for traj_idx in inner_iter:
                env = TripleSetEnv(
                    phi1_table, phi2_table, phi3_table, action_env_table, sizes, device,
                    allowed_per_cycle=allowed_per_cycle,
                )
                traj = rollout_trajectory(
                    env,
                    policy,
                    rng,
                    subsample_ratio=ss_ratio,
                    min_per_cycle=args.min_per_cycle,
                    action_cache=action_cache,
                    use_amp=use_amp,
                    epsilon=args.epsilon,
                    phase_regularization=phase_regularization_enabled,
                    phase_target=phase_target,
                    threshold=float(batch_thresholds[traj_idx]),
                )
                # Ensure terminal trajectory for TB
                if len(traj["logps"]) == 0:
                    continue
                if not env.is_terminal():
                    continue
                trajs.append(traj)
        t_rollout = time.time() - t0

        t1 = time.time()
        for traj in trajs:
            if len(traj["logps"]) == 0:
                continue

            # In batched mode, we expect terminal; in non-batched, we filtered above.
            logps_sum = torch.stack(traj["logps"]).sum()  # scalar tensor
            if phase_regularization_enabled:
                phases = traj.get("phases", [])
                if len(phases) != len(traj["logps"]):
                    continue
                phase_sum = torch.stack(phases).sum()
            traj_threshold = traj.get("threshold", None)
            if args.gfn_reward_source == "autodock_proxy":
                assert autodock_reward_oracle is not None
                r, log_r_float, meta = autodock_reward_oracle.evaluate(traj["terminal"], threshold=traj_threshold)
                if not (math.isfinite(r) and math.isfinite(log_r_float)):
                    continue
                log_R = torch.tensor(log_r_float, dtype=torch.float32, device=device)
                R = torch.tensor(r, dtype=torch.float32, device=device)
            else:
                log_R, R = reward_from_triple(
                    traj["terminal"], X_all, triple_model, device,
                    log_target=log_target,
                    model_is_logprobs=args.model_is_logprobs,
                    threshold=traj_threshold,
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

            # TB with tempered target: beta * log R.
            # IMPORTANT: when `log_target=True`, log_R is the model output
            # directly (no log() of a regression output is ever taken), which
            # is what protects this loop from the NaN failure mode.
            beta_logR = args.beta * log_R

            batch_logps_sum.append(logps_sum)
            batch_beta_logR.append(beta_logR)
            if phase_regularization_enabled:
                batch_phase_sums.append(phase_sum)
                batch_phase_subsample_stats.extend(traj.get("phase_subsample_stats", []))

            # Track reward/log_R and update top-k. We keep `yv = -log_R` as a
            # docking-score-like quantity for backward-compatible diagnostics.
            r = float(R.item()); yv = float((-log_R).item())

            rewards_seen.append(r)
            terminals_in_batch += 1
            batch_rewards_this_step.append(r)
            yhats_seen.append(yv)
            B1, B2, B3 = traj["terminal"]
            rec = {"reward": r, "yhat": yv, "B1": B1, "B2": B2, "B3": B3, "threshold": traj_threshold, **meta}
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
                    "threshold": traj_threshold if traj_threshold is not None else 0.0,
                    "reward_source": "autodock_proxy",
                })
            if len(topm_records) > args.topm * 5:
                topm_records = sorted(topm_records, key=lambda t: float(t["reward"]), reverse=True)[:args.topm]
        t_reward = time.time() - t1

        if not batch_logps_sum:
            continue

        batch_logps_sum = torch.stack(batch_logps_sum)   # [B]
        batch_beta_logR = torch.stack(batch_beta_logR)   # [B]

        # Compute per-trajectory logZ from the threshold-conditioned logZ network.
        batch_thresholds_tensor = torch.tensor(batch_thresholds, dtype=torch.float32, device=device)
        logZ_batch = logZ_net(batch_thresholds_tensor)  # [B]

        tb = tb_loss(logZ_batch, batch_logps_sum, batch_beta_logR)
        phase_component = None
        phase_diag = _summarize_phase_subsample_stats([], T=T)
        if phase_regularization_enabled:
            if len(batch_phase_sums) != batch_logps_sum.shape[0]:
                continue
            batch_phase_sums_t = torch.stack(batch_phase_sums)  # [B]
            phase_component = phase_loss(batch_phase_sums_t)
            phase_diag = _summarize_phase_subsample_stats(batch_phase_subsample_stats, T=T)
            loss = tb + float(args.phase_lambda) * phase_component
        else:
            loss = tb

        if use_amp:
            # AMP-safe update
            scaler.scale(loss).backward()
            # Unscale before clipping so clip threshold is in true grad units.
            scaler.unscale_(opt)
            if args.grad_clip and args.grad_clip > 0:
                clip_params = [p for m in trainable_modules_for_clipping for p in m.parameters()]
                torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                clip_params = [p for m in trainable_modules_for_clipping for p in m.parameters()]
                torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if args.time_breakdown:
            t_total = t_rollout + t_reward
            # NOTE: backward/opt time not broken out separately; rollout dominates in old mode.
            print(f"[Time] step={step} rollout={t_rollout:.4f}s reward={t_reward:.4f}s total~={t_total:.4f}s | terminals={terminals_in_batch}")

        # EMA for display
        tb_v = float(tb.item())
        phase_v = float(phase_component.item()) if phase_component is not None else float("nan")
        total_v = float(loss.item())
        log1p_v = math.log1p(tb_v)
        ema_loss = log1p_v if ema_loss is None else 0.98*ema_loss + 0.02*log1p_v

        # [ADD] Record metrics for this step
        if recorder is not None:
            # Report mean logZ across the batch for logging.
            mean_logZ = float(logZ_batch.mean().item())
            recorder.update(
                step=step,
                log1p_tb_loss=log1p_v,
                logZ=mean_logZ,
                batch_rewards=batch_rewards_this_step,
                terminals_in_batch=terminals_in_batch,
                phase_loss=phase_v if phase_regularization_enabled else None,
                total_loss=total_v,
                phase_target=phase_target if phase_regularization_enabled else None,
                phase_subsample_circ_var_mean=phase_diag["circ_var_mean"] if phase_regularization_enabled else None,
                phase_subsample_target_abs_err_mean=phase_diag["target_abs_err_mean"] if phase_regularization_enabled else None,
                phase_subsample_circ_mean_mean=phase_diag["circ_mean_mean"] if phase_regularization_enabled else None,
                phase_subsample_count_mean=phase_diag["count_mean"] if phase_regularization_enabled else None,
                phase_subsample_circ_var_by_t=phase_diag["circ_var_by_t"] if phase_regularization_enabled else None,
                phase_subsample_target_abs_err_by_t=phase_diag["target_abs_err_by_t"] if phase_regularization_enabled else None,
            )
            # experiment.log_metrics({"log1p_tb_loss": log1p_v, "batch_rewards": np.mean(batch_rewards_this_step)})


        if step % args.log_interval == 0:
            if batch_rewards_this_step:
                br = np.asarray(batch_rewards_this_step, dtype=np.float64)
                br_mean, br_std = float(br.mean()), float(br.std())
                br_min, br_max = float(br.min()), float(br.max())
                if recorder is None or recorder._ema_reward is None:
                    ema_str = "nan"
                else:
                    ema_str = f"{float(recorder._ema_reward):.3g}"
                br_msg = f" | batchR={br_mean:.3g}±{br_std:.3g} [EMA={ema_str}]"
            else:
                br_msg = " | batchR=nan nT=0"
            if phase_regularization_enabled:
                var_by_t = phase_diag["circ_var_by_t"]
                err_by_t = phase_diag["target_abs_err_by_t"]
                phase_msg = (
                    f" | phase_loss={phase_v:.6f} | phase_target={phase_target:.6g}"
                    f" | cand_phase_var={float(phase_diag['circ_var_mean']):.6g}"
                    f" | cand_phase_err={float(phase_diag['target_abs_err_mean']):.6g}"
                    f" | cand_phase_K={float(phase_diag['count_mean']):.3g}"
                    f" | var_by_t={np.array2string(np.asarray(var_by_t, dtype=np.float64), precision=3, separator=',')}"
                    f" | err_by_t={np.array2string(np.asarray(err_by_t, dtype=np.float64), precision=3, separator=',')}"
                    f" | total={total_v:.6g}"
                )
            else:
                phase_msg = ""
            progress_write(
                f"[Step {step}/{args.steps}] log(1+TB loss)={log1p_v:.6f} | EMA={ema_loss:.6f} | logZ={mean_logZ:.3f}"
                + phase_msg
                + f" | top = {np.log(max(max(float(t['reward']) for t in topm_records), 1e-30)):.3g}" + br_msg + f" | subsampling = {ss_ratio:.3g}"
            )

        # Periodically save checkpoint
        if args.save_model and (step == args.steps):
            ck = {
                
                "policy_state": policy.state_dict(),
                "gfn_phi_state": gfn_phi.state_dict(),
                "logZ_net_state": logZ_net.state_dict(),
                "optimizer_state": opt.state_dict(),
                "args": vars(args),
                "action_repr": args.action_repr,
                "phi_train_mode": args.phi_train_mode,
                "action_base_dim": int(action_base_dim),
                "action_in_dim": int(action_in_dim),
                "meta": {
                    "d_h": int(d_h),
                    "phi_hidden_dim": int(phi_hidden_dim),
                    "T": T,
                    "action_repr": args.action_repr,
                    "phi_train_mode": args.phi_train_mode,
                    "action_base_dim": int(action_base_dim),
                    "action_in_dim": int(action_in_dim),
                    "phase_regularization": phase_regularization_enabled,
                    "phase_lambda": float(args.phase_lambda),
                    "gfn_fourier_n_freqs": gfn_fourier_n_freqs,
                    "gfn_fourier_freq_scale": gfn_fourier_freq_scale,
                    "gfn_fourier_linear": gfn_fourier_linear,
                    "gfn_fourier_append_raw": gfn_fourier_append_raw,
                    "gfn_fourier_threshold_center": gfn_fourier_threshold_center,
                    "gfn_fourier_threshold_scale": gfn_fourier_threshold_scale,
                    "strip_conditioning": bool(args.strip_conditioning),
                    "threshold_min": float(args.threshold_min),
                    "threshold_max": float(args.threshold_max),
                }
            }
            policypath = os.path.join(args.outdir, "gfn_policy.pt")
            torch.save(ck, policypath)
            print("Saved gfn policy at "+policypath)
    print("Training complete.")

    # ================================================================
    # Post-training pure-inference rollouts
    # ================================================================
    if args.inference_steps > 0:
        inf_subsample_ratio = args.inference_subsample_ratio if args.inference_subsample_ratio is not None else ss_ratio
        # Determine the threshold to use for inference.
        # If --topm-threshold is set, use it for all inference trajectories and
        # clear topm_records so top-m is selected only from inference.
        # Otherwise, use None (no threshold conditioning, or sampled thresholds).
        inf_threshold = args.topm_threshold
        if inf_threshold is not None:
            print(
                f"\n[Inference] Sampling {args.inference_steps} additional trajectories "
                f"with subsample_ratio={inf_subsample_ratio:.4g} at fixed threshold={inf_threshold} ..."
            )
            # Clear training top-m records so top-m is selected only from inference.
            topm_records.clear()
        else:
            print(
                f"\n[Inference] Sampling {args.inference_steps} additional trajectories "
                f"with subsample_ratio={inf_subsample_ratio:.4g} ..."
            )
        policy.eval()
        # Rebuild phi tables and action cache in eval mode (no grad).
        with torch.no_grad():
            inf_phi1_table, inf_phi2_table, inf_phi3_table = gfn_phi.tables(X_all)
        if args.action_repr == "ecfp":
            inf_action_env_table = bb_ecfp_table
            inf_action_cache = static_ecfp_action_cache
        elif args.action_repr == "phi":
            inf_action_env_table = inf_phi1_table
            inf_action_cache = CachedActionEmbeddings(inf_phi1_table, inf_phi2_table, inf_phi3_table)
        else:
            raise ValueError(f"Unknown --action-repr={args.action_repr!r}")

        inf_remaining = args.inference_steps
        while inf_remaining > 0:
            chunk = min(inf_remaining, args.batch_trajectories)
            # Prepare thresholds for this chunk.
            if inf_threshold is not None:
                inf_thresholds = np.full(chunk, inf_threshold, dtype=np.float64)
            else:
                inf_thresholds = None
            with torch.no_grad():
                if args.batched_rollouts:
                    inf_trajs = rollout_trajectories_batched(
                        phi1_table=inf_phi1_table,
                        phi2_table=inf_phi2_table,
                        phi3_table=inf_phi3_table,
                        sizes=sizes,
                        policy=policy,
                        action_cache=inf_action_cache,
                        rng=rng,
                        subsample_ratio=inf_subsample_ratio,
                        min_per_cycle=args.min_per_cycle,
                        batch_trajectories=chunk,
                        device=device,
                        use_amp=use_amp,
                        epsilon=0.0,  # pure inference — no epsilon exploration
                        allowed_per_cycle=allowed_per_cycle,
                        phase_regularization=False,
                        thresholds=inf_thresholds,
                    )
                else:
                    inf_trajs = []
                    for traj_idx in range(chunk):
                        env = TripleSetEnv(
                            inf_phi1_table, inf_phi2_table, inf_phi3_table,
                            inf_action_env_table, sizes, device,
                            allowed_per_cycle=allowed_per_cycle,
                        )
                        traj_threshold = float(inf_thresholds[traj_idx]) if inf_thresholds is not None else None
                        traj = rollout_trajectory(
                            env,
                            policy,
                            rng,
                            subsample_ratio=inf_subsample_ratio,
                            min_per_cycle=args.min_per_cycle,
                            action_cache=inf_action_cache,
                            use_amp=use_amp,
                            epsilon=0.0,  # pure inference
                            phase_regularization=False,
                            threshold=traj_threshold,
                        )
                        if len(traj["logps"]) == 0 or not env.is_terminal():
                            continue
                        inf_trajs.append(traj)

            # Compute rewards and accumulate into the same tracking lists.
            for traj in inf_trajs:
                if len(traj["logps"]) == 0:
                    continue
                traj_threshold = traj.get("threshold", None)
                if args.gfn_reward_source == "autodock_proxy":
                    assert autodock_reward_oracle is not None
                    r, log_r_float, meta = autodock_reward_oracle.evaluate(traj["terminal"], threshold=traj_threshold)
                    if not (math.isfinite(r) and math.isfinite(log_r_float)):
                        continue
                    log_R = torch.tensor(log_r_float, dtype=torch.float32, device=device)
                    R = torch.tensor(r, dtype=torch.float32, device=device)
                else:
                    log_R, R = reward_from_triple(
                        traj["terminal"], X_all, triple_model, device,
                        log_target=log_target,
                        model_is_logprobs=args.model_is_logprobs,
                        threshold=traj_threshold,
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

                r = float(R.item())
                yv = float((-log_R).item())
                rewards_seen.append(r)
                yhats_seen.append(yv)
                B1, B2, B3 = traj["terminal"]
                rec = {"reward": r, "yhat": yv, "B1": B1, "B2": B2, "B3": B3, "threshold": traj_threshold, **meta}
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
                        "threshold": traj_threshold if traj_threshold is not None else 0.0,
                        "reward_source": "autodock_proxy",
                    })
                if len(topm_records) > args.topm * 5:
                    topm_records = sorted(topm_records, key=lambda t: float(t["reward"]), reverse=True)[:args.topm]

            inf_remaining -= chunk
            if inf_remaining > 0:
                print(f"[Inference] {args.inference_steps - inf_remaining}/{args.inference_steps} trajectories completed...")

        print(
            f"[Inference] Done — {args.inference_steps} inference trajectories sampled. "
            f"Top-m records now total {len(topm_records)}."
        )

    # ---- End-of-training analytics: histogram + top-k ----
    if len(yhats_seen) > 0:
        hist_path = os.path.join(args.outdir, "yhat_hist.png")
        vals = np.array(yhats_seen, dtype=np.float64)
        pd.DataFrame({"yhat": vals}).to_csv(
            os.path.splitext(hist_path)[0] + ".csv", index=False
        )
        plt.figure(figsize=(10,5))
        plt.hist(vals, bins=50)
        plt.title(
            "Histogram of yhat (= -log R)\n"
            f"lib={args.size1}x{args.size2}x{args.size3} | beta={args.beta} | "
            f"hidden(joint/state/action)={args.joint_dim}/{args.state_dim}/{args.action_dim} | "
            f"phase_lambda={args.phase_lambda} | Fourier n_freqs={args.fourier_n_freqs}"
        )
        plt.xlabel("yhat")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=180)
        plt.close()
        print(f"[Summary] Saved yhat histogram to {hist_path}")
    else:
        print("[Summary] No yhat recorded; histogram skipped.")

    # Top-m selection: use Pareto selection when threshold conditioning is enabled,
    # otherwise fall back to top-m by reward.
    if topm_records:
        if threshold_conditioning_enabled:
            topm_sorted = pareto_topm_selection(topm_records, args.topm)
            print(f"[Selection] Pareto front selection enabled (threshold conditioning): selected {len(topm_sorted)} records")
        else:
            topm_sorted = sorted(topm_records, key=lambda t: float(t["reward"]), reverse=True)[:args.topm]

        def join_ints(xs): 
            return "|".join(map(str, xs))

        def map_to_ids(xs):
            # map 0-based indices -> external IDs; assumes id_map is a 1D array
            return "|".join(str(int(id_map[i])) for i in xs)

        rows = []
        for i, rec in enumerate(topm_sorted, 1):
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
                    "threshold": rec.get("threshold"),
                }
                if args.gfn_reward_source == "autodock_proxy":
                    row.update({
                        "autodock_proxy_value": r,
                        "proxy_reward": np.nan,
                        "proxy_yhat": np.nan,
                        "reward_source": "autodock_proxy",
                    })
            rows.append(row)

        topm_df = pd.DataFrame(rows)
        topm_csv = os.path.join(args.outdir, "topm_rewards.csv")
        topm_df.to_csv(topm_csv, index=False)
        print(f"[Top-{args.topm}] Saved to {topm_csv}")

        if args.gfn_reward_source == "autodock_proxy":
            # Compatibility with active_learning_stage.py docking selection,
            # which reads topm_actual_scores.csv. The reward column is now
            # autodock_proxy_value in this mode.
            actual_csv = os.path.join(args.outdir, "topm_actual_scores.csv")
            topm_df.to_csv(actual_csv, index=False)
            print(f"[Top-{args.topm}] Saved autodock_proxy rewards to {actual_csv}")
    else:
        print("[Top-m] No terminal records to report.")

    if args.gfn_reward_source == "autodock_proxy" and encountered_rows:
        encountered_df = pd.DataFrame(encountered_rows)
        if args.encountered_out:
            os.makedirs(os.path.dirname(args.encountered_out), exist_ok=True)
            encountered_df.to_csv(args.encountered_out, index=False)
            print(f"[Encountered] Saved {len(encountered_df)} autodock_proxy-reward row(s) to {args.encountered_out}")
        if args.append_encountered_dataset:
            append_df = encountered_df[["B1_id", "B2_id", "B3_id", "threshold", "log_reward"]].rename(columns={"log_reward": "y"})
            header_needed = not os.path.exists(args.append_encountered_dataset)
            os.makedirs(os.path.dirname(args.append_encountered_dataset), exist_ok=True)
            append_df.to_csv(args.append_encountered_dataset, mode="a", header=header_needed, index=False)
            print(f"[Append] Added {len(append_df)} encountered autodock_proxy reward row(s) to {args.append_encountered_dataset}")
    # === End-of-training analytics ===

    # 2) Plots from recorder
    if recorder is not None and len(recorder.step_history) > 0:
        hist = recorder.step_history
        steps = np.array([h["step"] for h in hist], dtype=np.int64)
        log1p_tb = np.array([h["log1p_tb_loss"] for h in hist], dtype=np.float64)
        log1p_tb_ema = np.array([h["log1p_tb_loss_ema"] for h in hist], dtype=np.float64)
        br_mean = np.array([h["batch_reward_mean"] for h in hist], dtype=np.float64)

        def _savefig(base):
            png = os.path.join(args.outdir, f"{base}.png")
            plt.tight_layout()
            plt.savefig(png, dpi=180)
            if not args.no_svg:
                svg = os.path.join(args.outdir, f"{base}.svg")
                plt.savefig(svg)
            plt.close()
            # Keep the complete recorder table beside every plot.  This is
            # intentionally a sidecar rather than the live metrics.csv so
            # each image remains independently reproducible.
            pd.DataFrame(hist).to_csv(
                os.path.join(args.outdir, f"{base}.csv"), index=False
            )
            print(f"[Summary] Saved {base}.png")

        plot_context = (
            f"lib={args.size1}x{args.size2}x{args.size3} | beta={args.beta} | "
            f"hidden(joint/state/action)={args.joint_dim}/{args.state_dim}/{args.action_dim} | "
            f"phi_hidden={phi_hidden_dim} | phase_lambda={args.phase_lambda} | "
            f"Fourier n_freqs={args.fourier_n_freqs} | action_repr={args.action_repr}"
        )

        br_ema = np.array([h["batch_reward_ema"] for h in hist], dtype=np.float64)

        # Pre-compute phase/total loss arrays (raw + EMA) when phase regularization is active
        phase_loss_vals = None
        phase_loss_ema_vals = None
        total_loss_vals = None
        total_loss_ema_vals = None
        if phase_regularization_enabled:
            phase_loss_vals = np.array([h["phase_loss"] for h in hist], dtype=np.float64)
            phase_loss_ema_vals = np.array([h["phase_loss_ema"] for h in hist], dtype=np.float64)
            total_loss_vals = np.array([h["total_loss"] for h in hist], dtype=np.float64)
            total_loss_ema_vals = np.array([h["total_loss_ema"] for h in hist], dtype=np.float64)

        # Batch average reward + EMA
        if np.isfinite(br_mean).any():
            plt.figure(figsize=(9,4.5))
            plt.plot(steps, br_mean, label="batch avg reward", alpha=0.5)
            if np.isfinite(br_ema).any():
                plt.plot(steps, br_ema, label=f"EMA reward (α={0.98})", linewidth=2)
            plt.xlabel("step"); plt.ylabel("mean reward")
            plt.title(f"Batch reward per step\n{plot_context}")
            plt.legend(loc="best")
            _savefig("batch_avg_reward")


        # log(1+TB loss)
        plt.figure(figsize=(9,4.5))
        plt.plot(steps, log1p_tb, label="log(1+TB loss)")
        plt.xlabel("step"); plt.ylabel("log(1+TB loss)"); plt.title(f"log(1+TB loss) per step\n{plot_context}")
        plt.legend(loc="best")
        _savefig("loss_log1p_tb")

        # log(1+TB loss) EMA, with phase loss EMA overlaid on a secondary axis when active
        fig, ax1 = plt.subplots(figsize=(9,4.5))
        ax1.plot(steps, log1p_tb_ema, label="log(1+TB loss) (EMA)", color="C0")
        ax1.set_xlabel("step")
        ax1.set_ylabel("log(1+TB loss) (EMA)")
        overlay_phase = (
            phase_regularization_enabled
            and phase_loss_ema_vals is not None
            and np.isfinite(phase_loss_ema_vals).any()
        )
        if overlay_phase:
            ax2 = ax1.twinx()
            ax2.plot(steps, phase_loss_ema_vals, label="phase loss (EMA)", color="C1", linewidth=2)
            ax2.set_ylabel("phase loss (EMA)")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
            ax1.set_title(f"log(1+TB loss) EMA with phase loss overlay\n{plot_context}")
        else:
            ax1.legend(loc="best")
            ax1.set_title(f"log(1+TB loss) EMA\n{plot_context}")
        _savefig("loss_log1p_tb_ema")

        # Phase loss and total loss (only when phase regularization is enabled)
        if phase_regularization_enabled:
            # Phase loss (EMA prominent, raw faint)
            if np.isfinite(phase_loss_vals).any():
                plt.figure(figsize=(9,4.5))
                plt.plot(steps, phase_loss_vals, label="phase loss", color="C1", alpha=0.4)
                if np.isfinite(phase_loss_ema_vals).any():
                    plt.plot(steps, phase_loss_ema_vals, label=f"phase loss (EMA α={0.98})", color="C1", linewidth=2)
                plt.xlabel("step"); plt.ylabel("phase loss"); plt.title(f"Phase regularization loss per step\n{plot_context}")
                plt.legend(loc="best")
                _savefig("loss_phase")

            # Total loss (TB + λ·phase) (EMA prominent, raw faint)
            if np.isfinite(total_loss_vals).any():
                plt.figure(figsize=(9,4.5))
                plt.plot(steps, total_loss_vals, label="total loss (TB + λ·phase)", color="C2", alpha=0.4)
                if np.isfinite(total_loss_ema_vals).any():
                    plt.plot(steps, total_loss_ema_vals, label=f"total loss (EMA α={0.98})", color="C2", linewidth=2)
                plt.xlabel("step"); plt.ylabel("total loss"); plt.title(f"Total loss per step\n{plot_context}")
                plt.legend(loc="best")
                _savefig("loss_total")

    # 3) Top-m stats + diversity
    if topm_records:
        topm_sorted = sorted(topm_records, key=lambda t: float(t["reward"]), reverse=True)[:args.topm]
        # Compute diversity on Top-m using selected metric
        reps = [build_triple_representation(rec["B1"], rec["B2"], rec["B3"], bb_ecfp_table) for rec in topm_sorted]
        # div_mean = mean_pairwise_distance(reps, metric=args.topm_distance)
        rewards_topm = np.array([float(rec["reward"]) for rec in topm_sorted], dtype=np.float64)
        r_mean, r_std = float(rewards_topm.mean()), float(rewards_topm.std())
        print(f"[Top-m Stats] reward mean={r_mean:.6g} std={r_std:.6g}")


    # log_model(experiment, model=policy, model_name="Joint Edge Policy")

if __name__ == "__main__":
    main()
