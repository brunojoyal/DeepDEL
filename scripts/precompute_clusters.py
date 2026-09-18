#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precompute building-block clusters for the hierarchical DEL-GFlowNet (H-DEL-GFN).

Clusters are computed on the *truncated* pools only — the first N1, N2, N3
building blocks assigned to pools 1, 2, 3 in bbs.csv.  This matches the
paper's setting: N1=90, N2=89, N3=197.

Method (per Koziarski et al. 2024, Section 3.2):
  - 2048-bit ECFP4 fingerprints (radius=2)
  - Agglomerative Clustering with Jaccard distance (average linkage)
  - 10 clusters for cycles 1 & 2, 20 clusters for cycle 3

Output: a JSON file mapping global BB row-indices to cluster labels within
each cycle, e.g.
    {
      "1": {"0": 0, "1": 2, ...},   // cycle 1: global_idx -> cluster_id
      "2": {"193": 5, ...},          // cycle 2
      "3": {"458": 17, ...}          // cycle 3
    }
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
from sklearn.cluster import AgglomerativeClustering

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# ECFP utilities
# ---------------------------------------------------------------------------

def smiles_to_ecfp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute ECFP fingerprint as a dense numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def jaccard_distance_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard distance between two binary ECFP vectors."""
    inter = float(np.dot(a, b))
    union = float(np.sum(a) + np.sum(b) - inter)
    if union == 0.0:
        return 0.0
    return 1.0 - inter / union


def compute_ecfps(smiles_list: List[str], n_bits: int, radius: int) -> np.ndarray:
    """Compute ECFPs for a list of SMILES, returning [N, n_bits] float64 array."""
    fps = []
    for smi in smiles_list:
        fps.append(smiles_to_ecfp(smi, n_bits=n_bits, radius=radius))
    return np.stack(fps, axis=0)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_pool(
    ecfps: np.ndarray,
    global_indices: List[int],
    n_clusters: int,
    cycle_label: str,
    random_state: int = 42,
) -> Dict[int, int]:
    """Run AgglomerativeClustering on the given ECFPs and return
    {global_idx: cluster_id} mapping.
    """
    print(f"  Cycle {cycle_label}: clustering {len(global_indices)} BBs into {n_clusters} clusters ...")
    if len(global_indices) <= n_clusters:
        print(f"  WARNING: pool size ({len(global_indices)}) <= n_clusters ({n_clusters}); "
              f"each BB gets its own cluster.")
        # Fall back to a smaller number of clusters
        n_clusters = max(1, len(global_indices))
        print(f"  Using n_clusters={n_clusters} instead.")

    # Precompute the Jaccard distance matrix
    N = ecfps.shape[0]
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = jaccard_distance_binary(ecfps[i], ecfps[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)

    # Map back to global indices
    mapping: Dict[int, int] = {}
    for local_idx, label in enumerate(labels):
        global_idx = global_indices[local_idx]
        mapping[str(global_idx)] = int(label)

    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}")
    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Precompute BB clusters for hierarchical DEL-GFlowNet."
    )
    ap.add_argument("--bbs", default="data/bbs.csv",
                    help="Path to bbs.csv with columns SMILES, pool (1|2|3), ID.")
    ap.add_argument("--bb-pool-size", type=int, default=1000,
                    help="Number of BBs shared across all three pools (default 1000).")
    ap.add_argument("--n-clusters", type=int, default=20,
                    help="Number of clusters per cycle (default 20).")
    ap.add_argument("--fp-bits", type=int, default=2048,
                    help="ECFP fingerprint length (default 2048).")
    ap.add_argument("--fp-radius", type=int, default=2,
                    help="ECFP radius (default 2).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for clustering reproducibility.")
    ap.add_argument("--out", type=str, default="data/clusters.json",
                    help="Output JSON path.")
    ap.add_argument("--ecfp-cache-dir", type=str, default=None,
                    help="Directory to cache ECFP arrays (optional).")

    args = ap.parse_args()

    # Validate
    if not os.path.exists(args.bbs):
        raise FileNotFoundError(f"bbs.csv not found: {args.bbs}")

    df = pd.read_csv(args.bbs)
    if "SMILES" not in df.columns:
        raise ValueError(f"bbs.csv missing required column: SMILES")

    smiles_all = df["SMILES"].tolist()
    N = len(smiles_all)

    bb_pool_size = args.bb_pool_size
    n_clusters = args.n_clusters

    # Partition pools: if CSV has a 'pool' column, use it; otherwise
    # all three cycles share the same first N BBs.
    if "pool" in df.columns:
        pool_arr = df["pool"].to_numpy(dtype=int)
        pool1_indices = np.where(pool_arr == 1)[0].tolist()
        pool2_indices = np.where(pool_arr == 2)[0].tolist()
        pool3_indices = np.where(pool_arr == 3)[0].tolist()
        if not (pool1_indices and pool2_indices and pool3_indices):
            raise ValueError(
                f"At least one pool is empty: "
                f"|pool1|={len(pool1_indices)}, |pool2|={len(pool2_indices)}, |pool3|={len(pool3_indices)}"
            )
        # Truncate pools to bb_pool_size
        pool1_indices = pool1_indices[:bb_pool_size]
        pool2_indices = pool2_indices[:bb_pool_size]
        pool3_indices = pool3_indices[:bb_pool_size]
    else:
        # No pool column: all three cycles share the same first N BBs.
        if N < bb_pool_size:
            raise ValueError(
                f"bbs.csv has {N} rows but --bb-pool-size={bb_pool_size} "
                f"requires at least {bb_pool_size}."
            )
        pool1_indices = list(range(0, bb_pool_size))
        pool2_indices = list(range(0, bb_pool_size))
        pool3_indices = list(range(0, bb_pool_size))
        print(f"[Pools] No pool column — all three pools share the same first {bb_pool_size} BBs: "
              f"pool1=0:{bb_pool_size}, pool2=0:{bb_pool_size}, pool3=0:{bb_pool_size}")

    print(f"[Pools] pool1={len(pool1_indices)}, pool2={len(pool2_indices)}, pool3={len(pool3_indices)} "
          f"(bb_pool_size={bb_pool_size})")

    # Compute ECFPs once for the shared pool (all three cycles use the same BBs)
    print("[ECFP] Computing fingerprints for shared pool ...")
    ecfp_shared = compute_ecfps(
        [smiles_all[i] for i in pool1_indices],
        n_bits=args.fp_bits, radius=args.fp_radius,
    )

    # Optional caching
    if args.ecfp_cache_dir:
        os.makedirs(args.ecfp_cache_dir, exist_ok=True)
        np.save(os.path.join(args.ecfp_cache_dir, "ecfp_shared.npy"), ecfp_shared)
        print(f"[Cache] Saved ECFPs to {args.ecfp_cache_dir}")

    # Cluster once and assign the same mapping to all three cycles
    print("[Clustering] Running Agglomerative Clustering (Jaccard, average linkage) ...")
    clusters: Dict[str, Dict[str, int]] = {}

    shared_mapping = cluster_pool(
        ecfp_shared, pool1_indices, n_clusters, "shared", random_state=args.seed,
    )
    clusters["1"] = shared_mapping
    clusters["2"] = shared_mapping
    clusters["3"] = shared_mapping

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"[Done] Saved cluster assignments to {args.out}")

    # Summary
    for cycle in ("1", "2", "3"):
        n_unique = len(set(clusters[cycle].values()))
        print(f"  Cycle {cycle}: {len(clusters[cycle])} BBs → {n_unique} clusters")


if __name__ == "__main__":
    main()