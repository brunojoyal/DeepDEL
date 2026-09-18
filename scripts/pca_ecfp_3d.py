#!/usr/bin/env python
"""PCA on ECFP (Morgan) fingerprints and 3D scatter plot.

Default input matches this repo layout:
  data/bbs.csv with a `SMILES` column.

Dependencies (user installs):
  - rdkit
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - (optional) plotly

Example:
  python scripts/pca_ecfp_3d.py \
    --csv data/bbs.csv --smiles-col SMILES \
    --radius 2 --n-bits 2048 \
    --max-points 5000 \
    --out-prefix outputs/bbs_ecfp_pca
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _ecfp_bitvector(smiles: str, radius: int, n_bits: int, use_chirality: bool) -> DataStructs.cDataStructs.ExplicitBitVect | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        useChirality=use_chirality,
    )


def _bitvect_to_numpy(fp: DataStructs.cDataStructs.ExplicitBitVect) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute ECFP, PCA to 3D, and plot.")

    p.add_argument("--csv", default="data/bbs.csv", help="Input CSV file (default: data/bbs.csv)")
    p.add_argument("--smiles-col", default="SMILES", help="Name of the SMILES column (default: SMILES)")
    p.add_argument("--id-col", default="ID", help="Optional ID column to carry through (default: ID)")
    p.add_argument("--name-col", default="Name", help="Optional name column to carry through (default: Name)")

    p.add_argument("--radius", type=int, default=2, help="Morgan radius (ECFP4 => radius=2) (default: 2)")
    p.add_argument("--n-bits", type=int, default=2048, help="Number of bits in fingerprint (default: 2048)")
    p.add_argument(
        "--use-chirality",
        action="store_true",
        help="Include chirality in ECFP",
    )

    p.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optionally subsample to this many points (0 = no subsampling)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for subsampling (default: 0)",
    )

    p.add_argument(
        "--kmeans-k",
        type=int,
        default=0,
        help="If >0, run KMeans clustering on ECFP vectors before PCA and color points by cluster.",
    )

    p.add_argument(
        "--out-prefix",
        default="outputs/bbs_ecfp_pca",
        help="Output prefix (writes <prefix>.csv, <prefix>.png, optionally <prefix>.html)",
    )
    p.add_argument(
        "--plot-backend",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help="Plot backend (default: matplotlib)",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=6.0,
        help="Marker size for scatter plot (default: 6.0)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Marker alpha for scatter plot (default: 0.6)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise SystemExit(f"SMILES column '{args.smiles_col}' not found. Available: {list(df.columns)}")

    keep_cols: list[str] = [c for c in [args.id_col, args.name_col, args.smiles_col] if c in df.columns]
    df = df[keep_cols].copy()
    df = df.dropna(subset=[args.smiles_col])

    if args.max_points and len(df) > args.max_points:
        df = df.sample(n=args.max_points, random_state=args.seed).reset_index(drop=True)

    fps: list[DataStructs.cDataStructs.ExplicitBitVect] = []
    valid_rows: list[int] = []
    for i, smi in enumerate(df[args.smiles_col].astype(str).tolist()):
        fp = _ecfp_bitvector(smi, radius=args.radius, n_bits=args.n_bits, use_chirality=args.use_chirality)
        if fp is None:
            continue
        fps.append(fp)
        valid_rows.append(i)

    if not fps:
        raise SystemExit("No valid RDKit molecules parsed from SMILES.")

    df = df.iloc[valid_rows].reset_index(drop=True)
    X = np.vstack([_bitvect_to_numpy(fp) for fp in fps]).astype(np.float32)

    clusters: np.ndarray | None = None
    if args.kmeans_k and args.kmeans_k > 0:
        if args.kmeans_k > len(X):
            raise SystemExit(f"--kmeans-k ({args.kmeans_k}) cannot exceed number of valid molecules ({len(X)}).")
        km = KMeans(n_clusters=args.kmeans_k, random_state=args.seed, n_init="auto")
        clusters = km.fit_predict(X).astype(int)
        counts = np.bincount(clusters, minlength=args.kmeans_k)
        print("KMeans cluster sizes:")
        for c, n in enumerate(counts.tolist()):
            print(f"  cluster {c}: {n}")
        print(f"count std {np.std(counts)}, mean {np.mean(counts)}")

    pca = PCA(n_components=3, random_state=args.seed)
    Z = pca.fit_transform(X)

    df_out = df.copy()
    df_out["PC1"] = Z[:, 0]
    df_out["PC2"] = Z[:, 1]
    df_out["PC3"] = Z[:, 2]
    if clusters is not None:
        df_out["cluster"] = clusters

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = str(out_prefix) + ".csv"
    df_out.to_csv(out_csv, index=False)

    explained = pca.explained_variance_ratio_
    title = (
        f"PCA on ECFP (radius={args.radius}, nBits={args.n_bits}, N={len(df_out)})\n"
        f"Explained var: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}, PC3={explained[2]:.3f}"
    )
    if clusters is not None:
        title += f"\nKMeans clusters: k={args.kmeans_k}"

    if args.plot_backend == "matplotlib":
        import matplotlib

        matplotlib.use("Agg")  # safe for headless
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        if clusters is None:
            ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=args.point_size, alpha=args.alpha)
        else:
            # Discrete colormap; will repeat if k > 20
            cmap = plt.get_cmap("tab20")
            for c in range(args.kmeans_k):
                mask = clusters == c
                if not np.any(mask):
                    continue
                ax.scatter(
                    Z[mask, 0],
                    Z[mask, 1],
                    Z[mask, 2],
                    s=args.point_size,
                    alpha=args.alpha,
                    color=cmap(c % 20),
                    label=f"cluster {c}",
                )
            ax.legend(loc="best", fontsize=7, markerscale=1.2, frameon=True)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(title)
        fig.tight_layout()

        out_png = str(out_prefix) + ".png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_png}")

    else:
        import plotly.express as px

        plot_df = df_out.copy()
        if clusters is not None:
            # Make categorical for discrete colors
            plot_df["cluster"] = plot_df["cluster"].astype(str)

        fig = px.scatter_3d(
            plot_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color=("cluster" if clusters is not None else None),
            hover_data=keep_cols,
            title=title,
            opacity=args.alpha,
        )
        out_html = str(out_prefix) + ".html"
        fig.write_html(out_html)
        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
