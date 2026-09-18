#!/usr/bin/env python3
"""Select Experiment 1 leaders and prepare their molecules for physical docking.

Selection uses a three-fidelity pipeline:

1. **Fidelity 2 (DeepDEL)**: GFN terminal reward, used for the top-100 shortlist.
   The ``rank`` column in the proxy CSV reflects descending Fidelity 2 order.
2. **Fidelity 1 (docking proxy)**: ``autodock_proxy_value`` — library reward
   from the learned autodock proxy model. Used to rank candidates *within*
   the Fidelity 2 top-100, then to select six diverse leaders via K-means
   clustering over Morgan-fingerprint library representations.
3. **Fidelity 0 (physical docking)**: not computed here; this script only
   selects the leaders that will later be physically docked.

The selection is three-stage: Fidelity 2 top-100 shortlist → Fidelity 1 ranking
→ K-means clustering (one leader per cluster, best Fidelity 1 scorer per
cluster).  Leaders are assigned ``proxy_leader_rank`` 1–6 ordered by
descending Fidelity 1 score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["B1_id", "B2_id", "B3_id"]

# ---------------------------------------------------------------------------
# Fingerprint helpers (reuse the same Morgan-fingerprint + cycle-mean-concat
# representation already used by aggregate_experiment1.top_k_diversity).
# ---------------------------------------------------------------------------

def _import_rdkit():
    """Import RDKit from the tree rather than third-party vendored copies."""
    SRC = str(Path(__file__).resolve().parents[1] / "src")
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise SystemExit(
            "RDKit is required for fingerprint-based leader clustering.\n"
            "Load the rdkit module or activate the pytorch_env environment."
        ) from exc
    return Chem, DataStructs, AllChem
def _smiles_to_fingerprint(smiles: str, *, radius: int, n_bits: int) -> np.ndarray | None:
    """Return a dense bit-vector for a single SMILES or None on failure."""
    Chem, _, AllChem = _import_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    arr = np.zeros((int(n_bits),), dtype=np.float64)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _library_fingerprint(b1_str: str, b2_str: str, b3_str: str,
                         bb_fps: dict, *, fp_bits: int) -> np.ndarray:
    """Concatenate cycle-level mean Morgan fingerprints → [3 * fp_bits] vector."""
    parts = []
    for id_str in (str(b1_str), str(b2_str), str(b3_str)):
        ids = [int(x) for x in id_str.split("|")]
        fps = np.stack([bb_fps[i] for i in ids], axis=0)  # [size, fp_bits]
        parts.append(fps.mean(axis=0))                      # [fp_bits]
    return np.concatenate(parts)                            # [3 * fp_bits]


def _build_bb_fingerprint_cache(bbs_csv: str | Path, *, radius: int, n_bits: int) -> dict[int, np.ndarray]:
    """Return ``{bb_id: Morgan fingerprint array}`` for every BB in the CSV."""
    bbs = pd.read_csv(bbs_csv)
    smiles_col = "SMILES" if "SMILES" in bbs.columns else "smiles"
    if "ID" not in bbs.columns or smiles_col not in bbs.columns:
        raise SystemExit(f"BBS CSV must contain ID and SMILES columns: {bbs_csv}")
    cache: dict[int, np.ndarray] = {}
    for bb_id, smi in bbs[["ID", smiles_col]].itertuples(index=False):
        fp = _smiles_to_fingerprint(str(smi), radius=radius, n_bits=n_bits)
        if fp is not None:
            cache[int(bb_id)] = fp
    if not cache:
        raise SystemExit("no valid fingerprints could be computed from BBS CSV")
    return cache


# ---------------------------------------------------------------------------
# Leader selection
# ---------------------------------------------------------------------------

def select_leaders(
    proxy_csv: str | Path,
    bbs_csv: str | Path,
    *,
    top_k: int = 100,
    n_leaders: int = 6,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select *n_leaders* diverse leaders from the Fidelity 2 top-k shortlist.

    Parameters
    ----------
    proxy_csv:
        Output of ``eval_autodock_proxy_topm_scores``.  Must contain
        ``B1_id, B2_id, B3_id, rank, autodock_proxy_value, topm_smiles``.
    bbs_csv:
        Building-block CSV (``ID, SMILES``).
    top_k:
        Take the first *top_k* rows by ascending ``rank`` (Fidelity 2 order).
    n_leaders:
        Number of K-means clusters → leaders.
    fp_radius / fp_bits:
        Morgan fingerprint parameters for the library representation.
    random_state:
        Seed for KMeans reproducibility.

    Returns
    -------
    pd.DataFrame with columns ``proxy_leader_rank, B1_id, B2_id, B3_id, …``.
    """

    df = pd.read_csv(proxy_csv)

    # --- validation ---------------------------------------------------------
    missing = set(KEY) - set(df.columns)
    if missing:
        raise ValueError(f"proxy CSV is missing columns: {sorted(missing)}")
    if "rank" not in df.columns:
        raise ValueError("proxy CSV must retain the DeepDEL (Fidelity 2) rank in 'rank'")
    if "autodock_proxy_value" not in df.columns:
        raise ValueError("proxy CSV must contain autodock_proxy_value (Fidelity 1)")

    # --- Fidelity 2 top-k shortlist (repeated library identities kept) ------
    df = df.sort_values("rank", kind="mergesort").head(top_k).copy()
    if len(df) < top_k:
        raise ValueError(f"expected {top_k} DeepDEL (Fidelity 2) candidates, found {len(df)}")
    df["autodock_proxy_value"] = pd.to_numeric(df["autodock_proxy_value"], errors="coerce")
    df = df.dropna(subset=["autodock_proxy_value"])
    if len(df) < n_leaders:
        raise ValueError(f"fewer than {n_leaders} Fidelity 1 proxy-scored candidates")

    # --- fingerprint representation -----------------------------------------
    bb_fps = _build_bb_fingerprint_cache(bbs_csv, radius=fp_radius, n_bits=fp_bits)
    vectors = np.stack([
        _library_fingerprint(
            str(row.B1_id), str(row.B2_id), str(row.B3_id),
            bb_fps, fp_bits=fp_bits,
        )
        for row in df.itertuples(index=False)
    ], axis=0)  # [N, 3 * fp_bits]

    if vectors.shape[0] < n_leaders:
        raise ValueError(
            f"only {vectors.shape[0]} library fingerprints; "
            f"need at least {n_leaders} unique candidates for K-means"
        )

    # --- K-means clustering -------------------------------------------------
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise SystemExit(
            "scikit-learn is required for K-means leader clustering. "
            "Load scipy-stack/2025a or install scikit-learn."
        )

    kmeans = KMeans(n_clusters=n_leaders, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)  # [N]

    # --- pick best Fidelity 1 scorer per cluster ----------------------------
    df["_cluster"] = cluster_labels
    # Within each cluster, sort descending by Fidelity 1, then ascending by
    # Fidelity 2 rank as tie-breaker.
    leaders = (
        df.sort_values(
            ["_cluster", "autodock_proxy_value", "rank"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("_cluster", sort=False)
        .first()
        .reset_index(drop=True)
    )

    # --- assign proxy_leader_rank 1..n_leaders by descending Fidelity 1 -----
    leaders = leaders.sort_values(
        "autodock_proxy_value", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    leaders.insert(0, "proxy_leader_rank", range(1, len(leaders) + 1))

    # keep only relevant columns (drop internal _cluster)
    return leaders[[*KEY, "rank", "autodock_proxy_value", "topm_smiles", "proxy_leader_rank"]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select diverse Experiment 1 leaders via K-means clustering "
                    "over library Morgan-fingerprint representations."
    )
    ap.add_argument("--proxy-csv", required=True,
                    help="Proxy-eval CSV (contains Fidelity 2 rank + Fidelity 1 autodock_proxy_value).")
    ap.add_argument("--bbs-csv", required=True,
                    help="Building-block CSV (ID, SMILES).")
    ap.add_argument("--leaders-csv", required=True,
                    help="Output path for the selected leaders CSV.")
    ap.add_argument("--unscored-csv", required=True,
                    help="Output path for enumerated leader product SMILES.")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Number of Fidelity 2 candidates to consider.")
    ap.add_argument("--n-leaders", type=int, default=6,
                    help="Number of leaders (K-means clusters) to select.")
    ap.add_argument("--fp-radius", type=int, default=2,
                    help="Morgan fingerprint radius.")
    ap.add_argument("--fp-bits", type=int, default=2048,
                    help="Morgan fingerprint bit count.")
    ap.add_argument("--random-state", type=int, default=42,
                    help="Random seed for KMeans reproducibility.")
    args = ap.parse_args()

    leaders = select_leaders(
        args.proxy_csv, args.bbs_csv,
        top_k=args.top_k, n_leaders=args.n_leaders,
        fp_radius=args.fp_radius, fp_bits=args.fp_bits,
        random_state=args.random_state,
    )
    Path(args.leaders_csv).parent.mkdir(parents=True, exist_ok=True)
    leaders.to_csv(args.leaders_csv, index=False)

    rows = []
    for _, rec in leaders.iterrows():
        smiles = str(rec.get("topm_smiles", ""))
        for product_index, smi in enumerate(smiles.split(";")):
            if smi.strip():
                rows.append({
                    "proxy_leader_rank": int(rec.proxy_leader_rank),
                    "B1_id": rec.B1_id,
                    "B2_id": rec.B2_id,
                    "B3_id": rec.B3_id,
                    "product_index": product_index,
                    "smiles": smi.strip(),
                })
    if not rows:
        raise ValueError(
            "proxy output has no topm_smiles; rerun proxy evaluation with --save-topm-smiles"
        )
    pd.DataFrame(rows).to_csv(args.unscored_csv, index=False)


if __name__ == "__main__":
    main()