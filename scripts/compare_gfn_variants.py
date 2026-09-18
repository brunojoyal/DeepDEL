#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified evaluation and comparison of three GFN DEL-design variants:

  1. DeepDel GFN  (state-action flow model, train_gfn.py)
  2. Flat Multihot GFN  (DEL-GFlowNet, train_gfn_multihot.py)
  3. Hierarchical Multihot GFN  (H-DEL-GFlowNet, train_gfn_hierarchical.py)

Instead of sampling from each model, this script reads the pre-saved top-M
candidates from each variant's topm_rewards.csv (produced during training)
and compares them directly.

Produces:
  - Comparison table (reward, diversity, top-100, top-1) — Table 2 in paper
  - Chemical property distribution plots — Figure 5 in paper (drawn from a
    random subsample of molecules taken from the top-K libraries)
  - Training dynamics overlay plots (TB loss, reward EMA)

Usage:
  python scripts/compare_gfn_variants.py \
      --bbs data/bbs_JP.csv \
      --deepsets-topm outputs/deepsets_gfn/topm_rewards.csv \
      --multihot-topm outputs/multihot_gfn/topm_rewards.csv \
      --hierarchical-topm outputs/hierarchical_gfn/topm_rewards.csv
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Reuse shared infrastructure
from deepdelgfn.gfn.train_gfn import set_seed, load_bbs
import deepdelgfn.mols.dels as tri_mod


# ============================================================================
# Load pre-saved top-M candidates from a topm_rewards.csv
# ============================================================================

def load_topm_from_csv(
    csv_path: str,
    id_to_idx: Dict[int, int],
) -> List[Dict]:
    """Load top-M terminal states from a variant's topm_rewards.csv.

    The CSV columns are expected to be:
      rank, reward, yhat, B1_id, B2_id, B3_id

    where B*_id are pipe-separated BB ID strings (e.g. "42|17|8").

    Returns a list of dicts with keys: reward, log_reward, terminal.
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] topm CSV not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        r = float(row["reward"])
        # Parse pipe-separated BB IDs back to 0-based index lists
        B1_ids = [int(x) for x in str(row["B1_id"]).split("|")]
        B2_ids = [int(x) for x in str(row["B2_id"]).split("|")]
        B3_ids = [int(x) for x in str(row["B3_id"]).split("|")]
        B1 = [id_to_idx[bid] for bid in B1_ids]
        B2 = [id_to_idx[bid] for bid in B2_ids]
        B3 = [id_to_idx[bid] for bid in B3_ids]
        results.append({
            "reward": r,
            "log_reward": math.log(max(r, 1e-30)),
            "terminal": (B1, B2, B3),
        })
    return results


# ============================================================================
# Chemical property computation
# ============================================================================

def compute_library_properties(
    B1: List[int],
    B2: List[int],
    B3: List[int],
    smiles_all: List[str],
    allowed_per_cycle: Tuple[List[int], List[int], List[int]],
    reaction_mode: str,
) -> Dict[str, float]:
    """Compute average chemical properties of molecules in a library.

    Enumerates the Cartesian product of selected BBs in each cycle, builds
    trimers, and computes molecular descriptors. Returns the mean of each
    descriptor over all valid products.
    """
    bbs_df = pd.DataFrame({
        "ID": list(range(len(smiles_all))),
        "SMILES": smiles_all,
        "pool": [1] * len(smiles_all),
    })

    d1_raw, d2_raw, d3_raw = tri_mod.PoolIO.split_by_pool(bbs_df)
    builder = tri_mod.TrimerBuilder(d1_raw, d2_raw, d3_raw, reaction_mode=reaction_mode)

    # Restrict to allowed pools only
    allowed1 = set(allowed_per_cycle[0])
    allowed2 = set(allowed_per_cycle[1])
    allowed3 = set(allowed_per_cycle[2])

    b1_ids = [i for i in B1 if i in allowed1]
    b2_ids = [i for i in B2 if i in allowed2]
    b3_ids = [i for i in B3 if i in allowed3]

    if not b1_ids or not b2_ids or not b3_ids:
        return {
            "MolWt": float("nan"), "cLogP": float("nan"),
            "HBA": float("nan"), "HBD": float("nan"),
            "PSA": float("nan"), "NumAtoms": float("nan"),
            "RotBonds": float("nan"), "sp3Atoms": float("nan"),
        }

    mw_vals, logp_vals, hba_vals, hbd_vals = [], [], [], []
    psa_vals, numatoms_vals, rotb_vals, sp3_vals = [], [], [], []

    for i in b1_ids:
        for j in b2_ids:
            for k in b3_ids:
                try:
                    rec = builder.build_trimer_by_ids(int(i), int(j), int(k))
                except Exception:
                    continue
                smi = getattr(rec, "smi", None) if rec is not None else None
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mw_vals.append(Descriptors.MolWt(mol))
                logp_vals.append(Descriptors.MolLogP(mol))
                hba_vals.append(rdMolDescriptors.CalcNumHBA(mol))
                hbd_vals.append(rdMolDescriptors.CalcNumHBD(mol))
                psa_vals.append(Descriptors.TPSA(mol))
                numatoms_vals.append(mol.GetNumHeavyAtoms())
                rotb_vals.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
                sp3_vals.append(Descriptors.FractionCSP3(mol))

    def safe_mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "MolWt": safe_mean(mw_vals),
        "cLogP": safe_mean(logp_vals),
        "HBA": safe_mean(hba_vals),
        "HBD": safe_mean(hbd_vals),
        "PSA": safe_mean(psa_vals),
        "NumAtoms": safe_mean(numatoms_vals),
        "RotBonds": safe_mean(rotb_vals),
        "sp3Atoms": safe_mean(sp3_vals),
    }


def sample_molecule_properties(
    libraries: List[Dict],
    smiles_all: List[str],
    allowed_per_cycle: Tuple[List[int], List[int], List[int]],
    reaction_mode: str,
    n_molecules: int,
    rng: np.random.Generator,
    max_attempts_factor: int = 50,
) -> Dict[str, List[float]]:
    """Sample individual molecules uniformly at random from a set of libraries.

    Each library is the Cartesian product B1 x B2 x B3 of its selected building
    blocks (restricted to the allowed pools). Molecules are drawn by picking a
    library with probability proportional to its size and then picking one BB
    uniformly from each cycle, so every (library, molecule) pair is equally
    likely. Duplicate BB triples (the same molecule present in several
    libraries) are counted only once.

    Returns per-molecule descriptor values as a dict property -> list of values.
    """
    properties = ["MolWt", "cLogP", "HBA", "HBD", "PSA", "NumAtoms", "RotBonds", "sp3Atoms"]
    prop_vals: Dict[str, List[float]] = {p: [] for p in properties}

    allowed1 = set(allowed_per_cycle[0])
    allowed2 = set(allowed_per_cycle[1])
    allowed3 = set(allowed_per_cycle[2])

    lib_ids: List[Tuple[List[int], List[int], List[int]]] = []
    weights: List[float] = []
    for rec in libraries:
        B1, B2, B3 = rec["terminal"]
        b1 = [i for i in B1 if i in allowed1]
        b2 = [j for j in B2 if j in allowed2]
        b3 = [k for k in B3 if k in allowed3]
        if not (b1 and b2 and b3):
            continue
        lib_ids.append((b1, b2, b3))
        weights.append(float(len(b1) * len(b2) * len(b3)))

    if not lib_ids:
        print("    [Warning] No valid libraries to sample molecules from.")
        return prop_vals

    probs = np.asarray(weights, dtype=np.float64)
    probs /= probs.sum()
    probs[-1] = 1.0 - probs[:-1].sum()  # avoid float round-off in rng.choice

    seen: Set[Tuple[int, int, int]] = set()
    max_attempts = max_attempts_factor * n_molecules
    attempts = 0
    while len(seen) < n_molecules and attempts < max_attempts:
        attempts += 1
        li = int(rng.choice(len(lib_ids), p=probs))
        b1, b2, b3 = lib_ids[li]
        i = int(b1[rng.integers(len(b1))])
        j = int(b2[rng.integers(len(b2))])
        k = int(b3[rng.integers(len(b3))])
        triple = (i, j, k)
        if triple in seen:
            continue
        seen.add(triple)
        try:
            trimer = tri_mod.TrimerBuilder.build_trimer_from_smiles(
                smiles_all[i], smiles_all[j], smiles_all[k],
                reaction_mode=reaction_mode,
            )
        except Exception:
            continue
        smi = getattr(trimer, "smi", None) if trimer is not None else None
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        vals = (
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            Descriptors.TPSA(mol),
            float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            Descriptors.FractionCSP3(mol),
        )
        for p, v in zip(properties, vals):
            prop_vals[p].append(v)

    print(f"    Sampled {len(prop_vals['MolWt'])} unique molecules "
          f"from {len(lib_ids)} libraries ({attempts} draw attempts)")
    return prop_vals


# ============================================================================
# Metrics computation
# ============================================================================

def binary_vector_from_triple(
    B1: List[int],
    B2: List[int],
    B3: List[int],
    allowed_per_cycle: Tuple[List[int], List[int], List[int]],
) -> np.ndarray:
    """Convert terminal triple to a pooled binary vector.

    The vector has the same layout as the multihot/hierarchical state:
    [pool1 | pool2 | pool3], with 1 at positions whose allowed indices are in B*.
    """
    a1, a2, a3 = allowed_per_cycle
    vec = np.zeros(len(a1) + len(a2) + len(a3), dtype=np.float64)

    # Build mapping: global_idx -> position
    map1 = {idx: pos for pos, idx in enumerate(a1)}
    map2 = {idx: pos + len(a1) for pos, idx in enumerate(a2)}
    map3 = {idx: pos + len(a1) + len(a2) for pos, idx in enumerate(a3)}

    for idx in B1:
        if idx in map1:
            vec[map1[idx]] = 1.0
    for idx in B2:
        if idx in map2:
            vec[map2[idx]] = 1.0
    for idx in B3:
        if idx in map3:
            vec[map3[idx]] = 1.0
    return vec


def compute_diversity(
    results: List[Dict],
    allowed_per_cycle: Tuple[List[int], List[int], List[int]],
) -> float:
    """Compute mean pairwise Hamming distance between binary vectors (paper Eq. 2)."""
    if len(results) < 2:
        return 0.0

    vectors = []
    for rec in results:
        B1, B2, B3 = rec["terminal"]
        vec = binary_vector_from_triple(B1, B2, B3, allowed_per_cycle)
        vectors.append(vec)

    n = len(vectors)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += float(np.sum(vectors[i] != vectors[j]))
            count += 1
    return total_dist / max(1, count)


def compute_statistics(
    results: List[Dict],
    top_k: int = 100,
) -> Dict:
    """Compute summary statistics from a list of results."""
    rewards = np.array([r["reward"] for r in results])
    sorted_idx = np.argsort(rewards)[::-1]

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    top_k_rewards = rewards[sorted_idx[:top_k]]
    top_k_mean = float(np.mean(top_k_rewards)) if len(top_k_rewards) > 0 else float("nan")

    top_1_reward = float(rewards[sorted_idx[0]]) if len(sorted_idx) > 0 else float("nan")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "top_k_mean": top_k_mean,
        "top_1_reward": top_1_reward,
        "sorted_rewards": rewards[sorted_idx].tolist(),
    }


# ============================================================================
# Training dynamics
# ============================================================================

def load_training_dynamics(metrics_csv: str) -> Dict:
    """Load metrics.csv from a training run and extract dynamics."""
    if not os.path.exists(metrics_csv):
        return {"steps": [], "tb_loss_ema": [], "reward_ema": [], "cummax_reward": []}

    df = pd.read_csv(metrics_csv)
    df = df.drop_duplicates(subset=['step'], keep='last')
    result = {
        "steps": df["step"].tolist(),
        "tb_loss_ema": df["log1p_tb_loss_ema"].tolist() if "log1p_tb_loss_ema" in df.columns else [],
        "reward_ema": df["batch_reward_ema"].tolist() if "batch_reward_ema" in df.columns else [],
        "logZ": df["logZ"].tolist() if "logZ" in df.columns else [],
    }

    # Compute running cumulative maximum of batch_reward_max — the highest
    # reward discovered so far at each training step.
    if "batch_reward_max" in df.columns:
        br_max = df["batch_reward_max"].values.astype(np.float64)
        cummax = np.maximum.accumulate(br_max, axis=0)
        result["cummax_reward"] = cummax.tolist()
    else:
        result["cummax_reward"] = []

    return result


# ============================================================================
# Plotting
# ============================================================================

def plot_property_distributions(
    top_k_results: Dict[str, List[Dict]],
    smiles_all: List[str],
    allowed_per_cycle: Tuple[List[int], List[int], List[int]],
    reaction_mode: str,
    out_path: str,
    top_k: int = 100,
    n_molecules: Optional[int] = None,
    seed: int = 42,
):
    """Plot chemical property distributions (paper Figure 5).

    If ``n_molecules`` (> 0) is given, the distributions are computed over
    individual molecules: that many molecules are sampled uniformly at random
    from the union of the top-``top_k`` libraries of each variant. Otherwise
    (legacy behaviour), every top-``top_k`` library contributes its average
    property value.
    """
    properties = ["MolWt", "cLogP", "HBA", "HBD", "PSA", "NumAtoms", "RotBonds", "sp3Atoms"]
    prop_labels = [
        "Total Molweight", "cLogP", "H-Acceptors", "H-Donors",
        "Polar Surface Area", "Non-H Atoms", "Rotatable Bonds", "sp3-Atoms",
    ]

    use_molecules = n_molecules is not None and n_molecules > 0
    rng = np.random.default_rng(seed)

    # Compute properties for top-K of each method
    method_props: Dict[str, Dict[str, List[float]]] = {}
    for method_name, results in top_k_results.items():
        sorted_by_reward = sorted(results, key=lambda r: r["reward"], reverse=True)[:top_k]
        if use_molecules:
            print(f"  [{method_name}] Sampling {n_molecules} random molecules "
                  f"from top-{len(sorted_by_reward)} libraries...")
            prop_vals = sample_molecule_properties(
                sorted_by_reward, smiles_all, allowed_per_cycle, reaction_mode,
                n_molecules=n_molecules, rng=rng,
            )
        else:
            prop_vals: Dict[str, List[float]] = {p: [] for p in properties}
            for rec in sorted_by_reward:
                B1, B2, B3 = rec["terminal"]
                props = compute_library_properties(
                    B1, B2, B3, smiles_all, allowed_per_cycle, reaction_mode,
                )
                for p in properties:
                    v = props.get(p, float("nan"))
                    if math.isfinite(v):
                        prop_vals[p].append(v)
        method_props[method_name] = prop_vals

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    colors = {
        "DeepDel GFN": "C0",
        "Flat Multihot": "C1",
        "Hierarchical Multihot": "C2",
    }

    for ax_idx, (prop, label) in enumerate(zip(properties, prop_labels)):
        ax = axes[ax_idx]
        for method_name in method_props:
            vals = method_props[method_name].get(prop, [])
            if vals:
                ax.hist(vals, bins=30, alpha=0.5, density=True,
                        label=method_name, color=colors.get(method_name))
        ax.set_title(f"Property = {label}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if ax_idx == 0:
            ax.legend(fontsize=7)

    if use_molecules:
        title = (f"Distributions of chemical properties of {n_molecules} randomly sampled "
                 f"molecules from top-{top_k} libraries")
    else:
        title = (f"Distributions of average chemical library properties of "
                 f"top-{top_k} generated libraries")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[Plot] Saved property distributions to {out_path}")


def plot_training_dynamics(
    dynamics: Dict[str, Dict],
    out_path: str,
    plot_log_rewards: bool = False,
):
    """Overlay training curves for all variants."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        "DeepDel GFN": "C0",
        "Flat Multihot": "C1",
        "Hierarchical Multihot": "C2",
    }
    markers = {
        "DeepDel GFN": "o",
        "Flat Multihot": "s",
        "Hierarchical Multihot": "^",
    }

    for method_name, dyn in dynamics.items():
        steps = dyn.get("steps", [])
        if not steps:
            continue
        color = colors.get(method_name, "gray")
        marker = markers.get(method_name, ".")

        # Subsample for clarity
        if len(steps) > 500:
            idx = np.linspace(0, len(steps) - 1, 500, dtype=int)
        else:
            idx = np.arange(len(steps))

        if dyn.get("tb_loss_ema"):
            axes[0].plot(
                np.array(steps)[idx], np.array(dyn["tb_loss_ema"])[idx],
                color=color, label=method_name, linewidth=1, alpha=0.8,
            )
        if dyn.get("reward_ema"):
            vals = np.array(dyn["reward_ema"])
            if plot_log_rewards:
                vals = np.log(np.maximum(vals, 1e-30))
            axes[1].plot(
                np.array(steps)[idx], vals[idx],
                color=color, label=method_name, linewidth=1, alpha=0.8,
            )
        if dyn.get("logZ"):
            axes[2].plot(
                np.array(steps)[idx], np.array(dyn["logZ"])[idx],
                color=color, label=method_name, linewidth=1, alpha=0.8,
            )

    axes[0].set_title("TB Loss EMA")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("log(1+TB loss) EMA")
    axes[0].legend(fontsize=7)

    axes[1].set_title("Reward EMA")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Log Batch Reward EMA" if plot_log_rewards else "Batch Reward EMA")
    axes[1].legend(fontsize=7)

    axes[2].set_title("log Z")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("logZ")
    axes[2].legend(fontsize=7)

    plt.suptitle("Training Dynamics Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[Plot] Saved training dynamics to {out_path}")


def plot_highest_reward_so_far(
    dynamics: Dict[str, Dict],
    out_path: str,
    plot_log_rewards: bool = False,
):
    """Plot the highest discovered reward so far (cumulative max) for each variant."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "DeepDel GFN": "C0",
        "Flat Multihot": "C1",
        "Hierarchical Multihot": "C2",
    }

    for method_name, dyn in dynamics.items():
        steps = dyn.get("steps", [])
        cummax = dyn.get("cummax_reward", [])
        if not steps or not cummax:
            continue
        color = colors.get(method_name, "gray")

        # Subsample for clarity
        if len(steps) > 500:
            idx = np.linspace(0, len(steps) - 1, 500, dtype=int)
        else:
            idx = np.arange(len(steps))

        ax.plot(
            np.array(steps)[idx],
            np.log(np.maximum(np.array(cummax)[idx], 1e-30)) if plot_log_rewards else np.array(cummax)[idx],
            color=color,
            label=method_name,
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_title("Highest Discovered Log-Reward So Far" if plot_log_rewards else "Highest Discovered Reward So Far", fontsize=13)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cumulative Max Log-Reward" if plot_log_rewards else "Cumulative Max Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[Plot] Saved highest reward so far to {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        "Compare three GFN DEL-design variants using pre-saved top-M candidates"
    )
    ap.add_argument("--bbs", default="data/bbs_JP.csv")
    ap.add_argument("--deepsets-topm", required=True,
                    help="Path to deepsets GFN topm_rewards.csv")
    ap.add_argument("--multihot-topm", required=True,
                    help="Path to multihot GFN topm_rewards.csv")
    ap.add_argument("--hierarchical-topm", required=True,
                    help="Path to hierarchical GFN topm_rewards.csv")

    # Library sizing (must match training)
    ap.add_argument("--bb-pool-size", type=int, default=1000,
                    help="Number of BBs shared across all three pools (must match training).")

    # Statistics
    ap.add_argument("--top-k", type=int, default=100,
                    help="Number of top samples for top-K reward stats and for the "
                         "property-distribution library pool")
    ap.add_argument("--plot-n-molecules", type=int, default=None,
                    help="If set (> 0), property distribution plots are drawn from this "
                         "many randomly sampled individual molecules (drawn from the "
                         "top-k libraries of each variant) instead of per-library "
                         "averages of the whole top-k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reaction-mode", type=str, default="amide_amide_legacy")
    ap.add_argument("--plot-log-rewards", action="store_true",
                    help="Plot log-rewards on training dynamics / highest-reward plots")

    # Training dynamics
    ap.add_argument("--deepsets-metrics", type=str, default=None,
                    help="Path to metrics.csv from deepsets training")
    ap.add_argument("--multihot-metrics", type=str, default=None,
                    help="Path to metrics.csv from multihot training")
    ap.add_argument("--hierarchical-metrics", type=str, default=None,
                    help="Path to metrics.csv from hierarchical training")

    # Output
    ap.add_argument("--outdir", default="outputs/comparison")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load BBs ----
    df_bbs = load_bbs(args.bbs)
    smiles = df_bbs["SMILES"].tolist()
    N_full = len(smiles)
    id_map = df_bbs["ID"].to_numpy() if "ID" in df_bbs.columns else None

    if id_map is None:
        print("[ERROR] BBs CSV must contain an 'ID' column for topm_rewards.csv lookups.", file=sys.stderr)
        sys.exit(1)

    # Build reverse map: BB ID -> 0-based index
    id_to_idx: Dict[int, int] = {int(bb_id): int(i) for i, bb_id in enumerate(id_map.tolist())}

    # ---- Pool-aware action universe ----
    pool_arr = df_bbs["pool"].to_numpy() if "pool" in df_bbs.columns else np.zeros(N_full, dtype=int)

    bb_pool_size = args.bb_pool_size
    if np.any(pool_arr != 0):
        idx1 = np.where(pool_arr == 1)[0].tolist()
        idx2 = np.where(pool_arr == 2)[0].tolist()
        idx3 = np.where(pool_arr == 3)[0].tolist()
        if not (idx1 and idx2 and idx3):
            raise ValueError(
                f"Pool column present in {args.bbs}, but at least one pool is empty."
            )
        idx1 = idx1[:bb_pool_size]
        idx2 = idx2[:bb_pool_size]
        idx3 = idx3[:bb_pool_size]
    else:
        # No pool column: all three pools share the same first N BBs.
        if N_full < bb_pool_size:
            raise ValueError(
                f"bbs.csv has {N_full} rows but --bb-pool-size={bb_pool_size} "
                f"requires at least {bb_pool_size}."
            )
        idx1 = list(range(0, bb_pool_size))
        idx2 = list(range(0, bb_pool_size))
        idx3 = list(range(0, bb_pool_size))
        print(f"[Pools] No pool column — all three pools share the same first {bb_pool_size} BBs: "
              f"pool1=0:{bb_pool_size}, pool2=0:{bb_pool_size}, pool3=0:{bb_pool_size}")

    allowed_per_cycle: Tuple[List[int], List[int], List[int]] = (idx1, idx2, idx3)
    print(f"[Pools] pool1={len(idx1)}, pool2={len(idx2)}, pool3={len(idx3)}")

    # ---- Load top-M candidates from each variant ----
    print(f"\n{'='*60}")
    print(f"Loading top-M candidates from pre-saved topm_rewards.csv files...")
    print(f"{'='*60}\n")

    all_results = {
        "DeepDel GFN": load_topm_from_csv(args.deepsets_topm, id_to_idx),
        "Flat Multihot": load_topm_from_csv(args.multihot_topm, id_to_idx),
        "Hierarchical Multihot": load_topm_from_csv(args.hierarchical_topm, id_to_idx),
    }

    for name, results in all_results.items():
        print(f"  {name}: loaded {len(results)} top-M candidates")

    # ---- Compute statistics ----
    print(f"\n{'='*60}")
    print(f"Computing statistics...")
    print(f"{'='*60}\n")

    stats = {}
    diversity = {}
    for name, results in all_results.items():
        if not results:
            print(f"  {name}: no data (skipping)")
            continue
        stats[name] = compute_statistics(results, top_k=args.top_k)
        diversity[name] = compute_diversity(results, allowed_per_cycle)
        print(f"  {name}: {len(results)} candidates")
        print(f"    Mean reward: {stats[name]['mean_reward']:.4f}")
        print(f"    Top-{args.top_k} reward: {stats[name]['top_k_mean']:.4f}")
        print(f"    Top-1 reward: {stats[name]['top_1_reward']:.4f}")
        print(f"    Diversity: {diversity[name]:.4f}")

    # ---- Save comparison table ----
    print(f"\n{'='*60}")
    print(f"Comparison Table (paper Table 2 format)")
    print(f"{'='*60}\n")

    table_path = os.path.join(args.outdir, "comparison_table.csv")
    rows = []
    for name in ["DeepDel GFN", "Flat Multihot", "Hierarchical Multihot"]:
        if name not in stats:
            continue
        s = stats[name]
        d = diversity[name]
        rows.append({
            "Method": name,
            "Mean Reward": f"{s['mean_reward']:.4f}",
            "Std Reward": f"{s['std_reward']:.4f}",
            "Diversity": f"{d:.4f}",
            f"Top-{args.top_k} Reward": f"{s['top_k_mean']:.4f}",
            "Top-1 Reward": f"{s['top_1_reward']:.4f}",
        })
        # Print to console
        print(f"  {name:25s}  Mean R={s['mean_reward']:.4f}  "
              f"Div={d:.4f}  Top-{args.top_k}={s['top_k_mean']:.4f}  "
              f"Top-1={s['top_1_reward']:.4f}")

    table_df = pd.DataFrame(rows)
    table_df.to_csv(table_path, index=False)
    print(f"\nSaved comparison table to {table_path}")

    # Also save full results
    for name, results in all_results.items():
        if not results:
            continue
        res_path = os.path.join(args.outdir, f"{name.replace(' ', '_').lower()}_samples.csv")
        res_df = pd.DataFrame([{
            "reward": r["reward"],
            "log_reward": r["log_reward"],
            "B1": "|".join(str(i) for i in r["terminal"][0]),
            "B2": "|".join(str(i) for i in r["terminal"][1]),
            "B3": "|".join(str(i) for i in r["terminal"][2]),
        } for r in results])
        res_df.to_csv(res_path, index=False)
        print(f"Saved {len(res_df)} candidates to {res_path}")

    # ---- Property distribution plots ----
    print(f"\n{'='*60}")
    print(f"Generating property distribution plots...")
    print(f"{'='*60}\n")

    # Filter to methods that have data
    plot_results = {k: v for k, v in all_results.items() if v}
    if plot_results:
        plot_property_distributions(
            top_k_results=plot_results,
            smiles_all=smiles,
            allowed_per_cycle=allowed_per_cycle,
            reaction_mode=args.reaction_mode,
            out_path=os.path.join(args.outdir, "property_distributions.png"),
            top_k=args.top_k,
            n_molecules=args.plot_n_molecules,
            seed=args.seed,
        )

    # ---- Training dynamics plots ----
    dynamics: Dict[str, Dict] = {}
    if args.deepsets_metrics:
        dynamics["DeepDel GFN"] = load_training_dynamics(args.deepsets_metrics)
    if args.multihot_metrics:
        dynamics["Flat Multihot"] = load_training_dynamics(args.multihot_metrics)
    if args.hierarchical_metrics:
        dynamics["Hierarchical Multihot"] = load_training_dynamics(args.hierarchical_metrics)

    if dynamics:
        print(f"\n[Training Dynamics] Loaded metrics for {len(dynamics)} variant(s)")
        plot_training_dynamics(
            dynamics=dynamics,
            out_path=os.path.join(args.outdir, "training_dynamics.png"),
            plot_log_rewards=args.plot_log_rewards,
        )
        plot_highest_reward_so_far(
            dynamics=dynamics,
            out_path=os.path.join(args.outdir, "highest_reward_so_far.png"),
            plot_log_rewards=args.plot_log_rewards,
        )

    print(f"\n[Compare] All outputs saved to {args.outdir}")
    print("[Compare] Done.")


if __name__ == "__main__":
    main()