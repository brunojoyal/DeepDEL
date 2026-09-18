#!/usr/bin/env python3
"""Cluster scored DEL library CSVs using DeepDEL phi representations.

Input rows are expected to describe complete libraries/candidates with columns::

    path,file,reward,bb1_ids,bb2_ids,bb3_ids

For each row, this script reconstructs the same DeepDEL phi representation used
by the GFN state embedding: pooled phi1(bb1_ids), pooled phi2(bb2_ids), and
pooled phi3(bb3_ids), concatenated into one vector. It then runs k-means on
those row vectors, copies source files into per-cluster directories, and writes a
self-contained output CSV sorted by cluster then reward.

Example::

    python scripts/cluster_libraries_by_phi.py \
      --input-csv outputs/library_rewards_sorted.csv \
      --deepdel-checkpoint models/deepdel_last.pt \
      --bbs-csv bbs_combined.csv \
      --out-dir outputs/library_reward_clusters \
      --out-csv outputs/library_reward_clusters/clustered_library_rewards.csv \
      --combined-libraries-csv outputs/library_reward_clusters/combined_clustered_libraries.csv \
      --n-clusters 20
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REQUIRED_INPUT_COLUMNS = ("path", "file", "reward", "bb1_ids", "bb2_ids", "bb3_ids")
REQUIRED_LIBRARY_COLUMNS = ("bb1_id", "bb2_id", "bb3_id")
SMILES_JOINER = "|"


def _require_dependencies():
    """Import heavy scientific dependencies with a clearer failure message."""

    try:
        import numpy as np
        import pandas as pd
        import torch
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise SystemExit(
            "Missing required Python dependency while running cluster_libraries_by_phi.py. "
            "This script needs numpy, pandas, torch, rdkit, and scikit-learn. "
            "Run it in the same environment used for DeepDEL/GFN jobs.\n"
            f"Original import error: {exc}"
        ) from exc

    RDLogger.DisableLog("rdApp.*")
    return np, pd, torch, Chem, DataStructs, AllChem, KMeans


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K-means cluster library reward rows using DeepDEL phi representations."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV with columns path,file,reward,bb1_ids,bb2_ids,bb3_ids.",
    )
    p.add_argument(
        "--deepdel-checkpoint",
        type=Path,
        required=False,
        default="models/deepdel.pt",
        help="DeepDEL checkpoint produced by deepdelgfn.deepdel.train_offline.",
    )
    p.add_argument(
        "--bbs-csv",
        type=Path,
        default=Path("bbs_combined.csv"),
        help="Building-block CSV with ID and SMILES columns. Default: bbs_combined.csv.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=False,
        default="outputs",
        help="Output directory. Source files are copied under cluster_XXX subdirectories.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output clustered CSV. Default: <out-dir>/clustered_library_rewards.csv.",
    )
    p.add_argument(
        "--combined-libraries-csv",
        type=Path,
        default=None,
        help=(
            "Output molecule-level CSV containing all clustered source libraries with "
            "library and cluster columns added. Default: <out-dir>/combined_clustered_libraries.csv."
        ),
    )
    p.add_argument("--n-clusters", type=int, required=True, help="Number of k-means clusters.")
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for DeepDEL phi table construction. Default: auto.",
    )
    p.add_argument("--batch-size", type=int, default=4096, help="BB batch size for phi table construction.")
    p.add_argument("--random-state", type=int, default=0, help="KMeans random seed. Default: 0.")
    p.add_argument(
        "--reward-ascending",
        action="store_true",
        help="Sort rewards ascending within each cluster. Default is descending.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory / overwriting output CSV and copied files.",
    )
    p.add_argument(
        "--missing-source",
        choices=("error", "warn", "skip"),
        default="error",
        help="How to handle a source path that does not exist when copying. Default: error.",
    )
    p.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Use physical copies or symlinks in cluster directories. Default: copy.",
    )
    return p.parse_args()


def resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return path


def validate_output_dir(out_dir: Path, out_csv: Path, combined_libraries_csv: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"Output directory is non-empty: {out_dir}. Pass --overwrite to allow writing into it."
        )
    if out_csv.exists() and not overwrite:
        raise SystemExit(f"Output CSV already exists: {out_csv}. Pass --overwrite to replace it.")
    if combined_libraries_csv.exists() and not overwrite:
        raise SystemExit(
            f"Combined libraries CSV already exists: {combined_libraries_csv}. Pass --overwrite to replace it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def parse_id_list(value: object) -> list[int]:
    """Parse pipe-separated BB IDs into ints, accepting blank/NaN as empty."""

    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    ids: list[int] = []
    for token in text.split("|"):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Could not parse BB ID {token!r} from value {value!r}") from exc
    return ids


def smiles_to_morgan_bits(smiles: str, *, n_bits: int, radius: int, np, Chem, DataStructs, AllChem):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    arr = np.zeros((int(n_bits),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_bbs_and_features(bbs_csv: Path, n_bits: int, radius: int, np, pd, Chem, DataStructs, AllChem):
    bbs = pd.read_csv(bbs_csv)
    missing = {"ID", "SMILES"} - set(bbs.columns)
    if missing:
        raise SystemExit(f"{bbs_csv} is missing required column(s): {sorted(missing)}")

    ids: list[int] = []
    smiles: list[str] = []
    features = []
    invalid: list[tuple[int, str]] = []

    for _, row in bbs.iterrows():
        bb_id = int(row["ID"])
        smi = str(row["SMILES"])
        arr = smiles_to_morgan_bits(
            smi,
            n_bits=n_bits,
            radius=radius,
            np=np,
            Chem=Chem,
            DataStructs=DataStructs,
            AllChem=AllChem,
        )
        if arr is None:
            invalid.append((bb_id, smi))
            continue
        ids.append(bb_id)
        smiles.append(smi)
        features.append(arr)

    if not features:
        raise SystemExit(f"No valid BB SMILES found in {bbs_csv}")
    if invalid:
        print(f"[WARN] Skipped {len(invalid)} invalid BB SMILES while building phi tables.", file=sys.stderr)

    id_to_idx = {bb_id: i for i, bb_id in enumerate(ids)}
    id_to_smiles = {bb_id: smi for bb_id, smi in zip(ids, smiles)}
    return bbs, id_to_idx, id_to_smiles, np.stack(features, axis=0).astype(np.float32, copy=False)


def _get_ckpt_arg(ckpt_args: dict, name: str, default):
    return ckpt_args.get(name, default)


def load_deepdel_model(checkpoint: Path, d_in: int, device, torch):
    from deepdelgfn.deepdel.train_offline import TripleDeepSet

    ckpt = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise SystemExit(f"{checkpoint} does not look like a DeepDEL train_offline checkpoint.")
    ckpt_args = ckpt.get("args", {}) or {}

    hidden_dim = int(_get_ckpt_arg(ckpt_args, "hidden_dim", 256))
    rho_dim = int(_get_ckpt_arg(ckpt_args, "rho_dim", 256))
    dropout = float(_get_ckpt_arg(ckpt_args, "dropout", 0.1))
    shared_phi = bool(_get_ckpt_arg(ckpt_args, "shared_phi", True))
    pooling = str(_get_ckpt_arg(ckpt_args, "pooling", "mean")).lower()
    output_head = str(_get_ckpt_arg(ckpt_args, "output_head", "linear")).lower()
    lib_size = _get_ckpt_arg(ckpt_args, "lib_size", None)

    model = TripleDeepSet(
        d_in=int(d_in),
        d_hidden=hidden_dim,
        d_rho=rho_dim,
        dropout=dropout,
        shared_phi=shared_phi,
        pooling=pooling,
        output_head=output_head,
        reward_bound_k=lib_size,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt_args, pooling


def build_phi_tables(X_all, model, batch_size: int, device, np, torch):
    xs = torch.from_numpy(X_all).float()
    phi1_parts = []
    phi2_parts = []
    phi3_parts = []
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, xs.shape[0], batch_size):
            xb = xs[start : start + batch_size].to(device)
            phi1_parts.append(model.phi(xb).detach().cpu().numpy())
            phi2_parts.append(model.phi2(xb).detach().cpu().numpy())
            phi3_parts.append(model.phi3(xb).detach().cpu().numpy())
    return (
        np.concatenate(phi1_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(phi2_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(phi3_parts, axis=0).astype(np.float32, copy=False),
    )


def pool_phi(phi_table, indices: Sequence[int], pooling: str, np):
    if not indices:
        return np.zeros((phi_table.shape[1],), dtype=np.float32)
    pooled = phi_table[np.asarray(indices, dtype=np.int64)].sum(axis=0)
    if pooling == "sum":
        return pooled.astype(np.float32, copy=False)
    return (pooled / float(len(indices))).astype(np.float32, copy=False)


def ids_to_indices(ids: Sequence[int], id_to_idx: dict[int, int], *, row_number: int, column: str) -> list[int]:
    missing = [bb_id for bb_id in ids if bb_id not in id_to_idx]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        suffix = "..." if len(missing) > 10 else ""
        raise KeyError(f"Row {row_number}: {column} references BB ID(s) not found in bbs CSV: {preview}{suffix}")
    return [id_to_idx[bb_id] for bb_id in ids]


def add_smiles_column(ids_per_row: Iterable[Sequence[int]], id_to_smiles: dict[int, str]) -> list[str]:
    values: list[str] = []
    for ids in ids_per_row:
        values.append(SMILES_JOINER.join(id_to_smiles.get(int(bb_id), "") for bb_id in ids))
    return values


def make_row_representations(df, id_to_idx: dict[int, int], phi1, phi2, phi3, pooling: str, np):
    reps = []
    bb1_lists: list[list[int]] = []
    bb2_lists: list[list[int]] = []
    bb3_lists: list[list[int]] = []

    for row_i, row in df.iterrows():
        bb1_ids = parse_id_list(row["bb1_ids"])
        bb2_ids = parse_id_list(row["bb2_ids"])
        bb3_ids = parse_id_list(row["bb3_ids"])
        bb1_lists.append(bb1_ids)
        bb2_lists.append(bb2_ids)
        bb3_lists.append(bb3_ids)

        i1 = ids_to_indices(bb1_ids, id_to_idx, row_number=int(row_i), column="bb1_ids")
        i2 = ids_to_indices(bb2_ids, id_to_idx, row_number=int(row_i), column="bb2_ids")
        i3 = ids_to_indices(bb3_ids, id_to_idx, row_number=int(row_i), column="bb3_ids")
        rep = np.concatenate(
            [
                pool_phi(phi1, i1, pooling, np),
                pool_phi(phi2, i2, pooling, np),
                pool_phi(phi3, i3, pooling, np),
            ],
            axis=0,
        )
        reps.append(rep)

    if not reps:
        raise SystemExit("Input CSV has no rows to cluster.")
    return np.stack(reps, axis=0).astype(np.float32, copy=False), bb1_lists, bb2_lists, bb3_lists


def run_kmeans(X, n_clusters: int, random_state: int, KMeans, np):
    if n_clusters < 1:
        raise SystemExit("--n-clusters must be >= 1")
    if n_clusters > X.shape[0]:
        raise SystemExit(f"--n-clusters ({n_clusters}) cannot exceed number of rows ({X.shape[0]}).")

    try:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        clusters = km.fit_predict(X).astype(int)
    except TypeError:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = km.fit_predict(X).astype(int)

    centers = km.cluster_centers_.astype(np.float32, copy=False)
    distances = np.linalg.norm(X - centers[clusters], axis=1).astype(np.float32, copy=False)
    return clusters, distances


def unique_destination(cluster_dir: Path, preferred_name: str, row_index: int) -> Path:
    preferred = Path(preferred_name).name or f"row_{row_index}.csv"
    dest = cluster_dir / preferred
    if not dest.exists():
        return dest
    stem = Path(preferred).stem
    suffix = Path(preferred).suffix
    return cluster_dir / f"{stem}__row{row_index}{suffix}"


def copy_cluster_files(df, out_dir: Path, *, copy_mode: str, missing_source: str, overwrite: bool) -> list[str]:
    copied_paths: list[str] = []
    for row_i, row in df.iterrows():
        cluster = int(row["cluster"])
        cluster_dir = out_dir / f"cluster_{cluster:03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        src = resolve_existing_path(Path(str(row["path"])))
        dest = unique_destination(cluster_dir, str(row.get("file", src.name)), int(row_i))

        if not src.exists():
            msg = f"Source file does not exist for row {row_i}: {src}"
            if missing_source == "error":
                raise SystemExit(msg)
            if missing_source == "warn":
                print(f"[WARN] {msg}", file=sys.stderr)
            copied_paths.append("")
            continue

        if dest.exists():
            if not overwrite:
                raise SystemExit(f"Destination already exists: {dest}. Pass --overwrite to replace it.")
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()

        if copy_mode == "symlink":
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)
        copied_paths.append(str(dest))
    return copied_paths


def write_combined_libraries_csv(df, out_csv: Path, *, pd, missing_source: str) -> int:
    """Write one molecule-level CSV combining every clustered source library.

    The output preserves each source library's molecule columns (including
    bb1_id/bb2_id/bb3_id) and prepends the library filename and assigned cluster.
    """

    parts = []
    for row_i, row in df.iterrows():
        src = resolve_existing_path(Path(str(row["path"])))
        if not src.exists():
            msg = f"Source file does not exist for row {row_i}: {src}"
            if missing_source == "error":
                raise SystemExit(msg)
            if missing_source == "warn":
                print(f"[WARN] {msg}", file=sys.stderr)
            continue

        library_df = pd.read_csv(src)
        missing_cols = set(REQUIRED_LIBRARY_COLUMNS) - set(library_df.columns)
        if missing_cols:
            raise SystemExit(f"{src} missing required molecule column(s): {sorted(missing_cols)}")

        library_name = str(row.get("file", "")).strip() or src.name
        cluster = int(row["cluster"])

        library_out = library_df.copy()
        rename_existing = {}
        if "library" in library_out.columns:
            rename_existing["library"] = "source_library"
        if "cluster" in library_out.columns:
            rename_existing["cluster"] = "source_cluster"
        if rename_existing:
            library_out = library_out.rename(columns=rename_existing)

        library_out.insert(0, "cluster", cluster)
        library_out.insert(0, "library", library_name)
        parts.append(library_out)

    if parts:
        combined = pd.concat(parts, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["library", "cluster", *REQUIRED_LIBRARY_COLUMNS])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return int(len(combined))


def main() -> None:
    args = parse_args()
    np, pd, torch, Chem, DataStructs, AllChem, KMeans = _require_dependencies()

    input_csv = resolve_existing_path(args.input_csv)
    checkpoint = resolve_existing_path(args.deepdel_checkpoint)
    bbs_csv = resolve_existing_path(args.bbs_csv)
    out_dir = args.out_dir
    out_csv = args.out_csv or (out_dir / "clustered_library_rewards.csv")
    combined_libraries_csv = args.combined_libraries_csv or (out_dir / "combined_clustered_libraries.csv")

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    if not checkpoint.exists():
        raise SystemExit(f"DeepDEL checkpoint not found: {checkpoint}")
    if not bbs_csv.exists():
        raise SystemExit(f"BB CSV not found: {bbs_csv}")
    validate_output_dir(out_dir, out_csv, combined_libraries_csv, args.overwrite)

    df = pd.read_csv(input_csv)
    missing_cols = set(REQUIRED_INPUT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise SystemExit(f"{input_csv} missing required column(s): {sorted(missing_cols)}")
    if df.empty:
        raise SystemExit(f"{input_csv} has no rows")

    # Load checkpoint once to get the BB ECFP dimensions before feature construction.
    ckpt_meta = torch.load(checkpoint, map_location="cpu")
    ckpt_args = ckpt_meta.get("args", {}) or {}
    n_bits = int(_get_ckpt_arg(ckpt_args, "bb_fp_bits", 2048))
    radius = int(_get_ckpt_arg(ckpt_args, "bb_fp_radius", 2))

    print(f"[Info] Loading BBs from {bbs_csv} with Morgan radius={radius}, n_bits={n_bits}", flush=True)
    _, id_to_idx, id_to_smiles, X_all = load_bbs_and_features(
        bbs_csv, n_bits, radius, np, pd, Chem, DataStructs, AllChem
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Loading DeepDEL checkpoint {checkpoint} on {device}", flush=True)
    model, ckpt_args, pooling = load_deepdel_model(checkpoint, X_all.shape[1], device, torch)
    print(f"[Info] Building phi tables for {X_all.shape[0]} BBs; pooling={pooling}", flush=True)
    phi1, phi2, phi3 = build_phi_tables(X_all, model, args.batch_size, device, np, torch)

    print(f"[Info] Building row phi representations for {len(df)} rows", flush=True)
    X_rows, bb1_lists, bb2_lists, bb3_lists = make_row_representations(
        df, id_to_idx, phi1, phi2, phi3, pooling, np
    )

    print(f"[Info] Running KMeans with k={args.n_clusters}", flush=True)
    clusters, distances = run_kmeans(X_rows, args.n_clusters, args.random_state, KMeans, np)

    out_df = df.copy()
    out_df["cluster"] = clusters
    out_df["cluster_distance"] = distances
    out_df["bb1_smiles"] = add_smiles_column(bb1_lists, id_to_smiles)
    out_df["bb2_smiles"] = add_smiles_column(bb2_lists, id_to_smiles)
    out_df["bb3_smiles"] = add_smiles_column(bb3_lists, id_to_smiles)

    out_df["reward"] = pd.to_numeric(out_df["reward"], errors="coerce")
    out_df = out_df.sort_values(
        by=["cluster", "reward"],
        ascending=[True, bool(args.reward_ascending)],
        kind="mergesort",
    ).reset_index(drop=True)

    print(f"[Info] Copying source files into {out_dir}", flush=True)
    copied_paths = copy_cluster_files(
        out_df,
        out_dir,
        copy_mode=args.copy_mode,
        missing_source=args.missing_source,
        overwrite=args.overwrite,
    )
    out_df["clustered_path"] = copied_paths

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"[Info] Writing combined molecule-level libraries CSV to {combined_libraries_csv}", flush=True)
    n_combined_molecules = write_combined_libraries_csv(
        out_df,
        combined_libraries_csv,
        pd=pd,
        missing_source=args.missing_source,
    )

    counts = np.bincount(clusters, minlength=args.n_clusters)
    print("[Info] KMeans cluster sizes:", flush=True)
    for cluster, count in enumerate(counts.tolist()):
        print(f"  cluster {cluster:03d}: {count}", flush=True)
    print(f"Wrote clustered CSV: {out_csv}", flush=True)
    print(f"Wrote combined libraries CSV ({n_combined_molecules} molecules): {combined_libraries_csv}", flush=True)
    print(f"Wrote cluster directories under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()