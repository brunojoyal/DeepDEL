#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump

RDLogger.DisableLog("rdApp.*")  # quiet RDKit warnings


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Convert a SMILES string to a dense ECFP (Morgan fingerprint) bit
    vector.  Returns ``None`` if the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def collect_csvs(paths):
    """Recursively collect CSV files from a list of file and directory
    paths.  Deduplicates and returns sorted absolute paths.
    """
    csvs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            sys.exit(f"[ERROR] Path not found: {p}")
        if p.is_file():
            if p.suffix.lower() == ".csv":
                csvs.append(str(p.resolve()))
            else:
                sys.exit(f"[ERROR] Not a CSV file: {p}")
        elif p.is_dir():
            found = [str(x.resolve()) for x in p.rglob("*.csv")]
            if not found:
                print(f"[WARN] No CSVs found under directory: {p}")
            csvs.extend(found)
        else:
            sys.exit(f"[ERROR] Unsupported path type: {p}")

    # Deduplicate and sort for stability
    csvs = sorted(set(csvs))
    return csvs


def ensure_parent_dir(path_str):
    """Create parent directories for a file path if they don't exist."""
    p = Path(path_str)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def main():
    """CLI entry point: reads and merges CSVs, featurizes SMILES to ECFPs,
    trains a ``RandomForestRegressor``, evaluates on a held-out test set,
    and saves the model as a ``.joblib`` artifact.
    """
    ap = argparse.ArgumentParser(description="Train ECFP-based model on docking scores.")
    ap.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        help="One or more CSV files and/or directories containing CSVs (directories are searched recursively).",
    )
    ap.add_argument("--smiles_col", default="smiles", help="Column with SMILES (default: smiles)")
    ap.add_argument("--target_col", default="docking_score", help="Column with target (default: docking_score)")
    ap.add_argument("--radius", type=int, default=2, help="ECFP radius (default: 2)")
    ap.add_argument("--n_bits", type=int, default=2048, help="ECFP length (default: 2048)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--n_estimators", type=int, default=300, help="RF trees (default: 300)")
    ap.add_argument("--model_out", default="models/autodock_model.joblib", help="Output model file")
    ap.add_argument("--preds_out", default="outputs/autodock_proxy/predictions.csv", help="Output predictions file")
    ap.add_argument(
        "--sep",
        default=",",
        help="CSV delimiter (default: ','). Set to '\\t' for TSV, etc.",
    )
    ap.add_argument(
        "--clip_positive_targets",
        action="store_true",
        help="Clip positive target values to 0.0 before training.",
    )
    args = ap.parse_args()

    csv_files = collect_csvs(args.inputs)
    if not csv_files:
        sys.exit("[ERROR] No CSVs found from provided inputs.")


    # Load and MERGE CSVs
    dfs = []
    for path in csv_files:
        print(f"[INFO] Reading {path}")
        try:
            df_part = pd.read_csv(path, sep=args.sep)
        except Exception as e:
            sys.exit(f"[ERROR] Failed to read {path}: {e}")
        dfs.append(df_part)

    if not dfs:
        sys.exit("[ERROR] No CSVs loaded.")

    df = pd.concat(dfs, ignore_index=True)

    # Basic sanity checks
    if args.smiles_col not in df.columns or args.target_col not in df.columns:
        sys.exit(f"[ERROR] CSVs must contain '{args.smiles_col}' and '{args.target_col}'.")

    # Drop rows with missing values in required cols
    before = len(df)
    df = df.dropna(subset=[args.smiles_col, args.target_col])
    if len(df) < before:
        print(f"[INFO] Dropped {before - len(df)} rows with missing SMILES/target.")

    smiles_list = df[args.smiles_col].astype(str).tolist()
    targets = df[args.target_col].astype(float).values
    if args.clip_positive_targets:
        n_clipped = int(np.count_nonzero(targets > 0.0))
        targets = np.minimum(targets, 0.0)
        print(f"[INFO] Clipped {n_clipped} positive targets to 0.0.")

    X_list, y_list, valid_smiles = [], [], []
    print(f"[INFO] Generating ECFPs for {len(smiles_list)} molecules from {len(csv_files)} CSV(s)...")
    for i, (sm, t) in enumerate(zip(smiles_list, targets), 1):
        fp = smiles_to_ecfp(sm, radius=args.radius, n_bits=args.n_bits)
        if fp is not None:
            X_list.append(fp)
            y_list.append(t)
            valid_smiles.append(sm)
        

    if not X_list:
        sys.exit("[ERROR] No valid molecules found.")

    X = np.vstack(X_list)
    y = np.array(y_list)
    smiles = np.array(valid_smiles)



    # Split (after merging)
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, y, smiles, test_size=args.test_size, random_state=args.seed
    )

    # Train RF
    print("[INFO] Training RandomForest...")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Ensure output directories exist
    ensure_parent_dir(args.model_out)
    ensure_parent_dir(args.preds_out)

    # Save model
    dump(
        {
            "model": model,
            "radius": args.radius,
            "n_bits": args.n_bits,
            "smiles_col": args.smiles_col,
            "target_col": args.target_col,
        },
        args.model_out
    )
    print(f"[INFO] Model saved to {args.model_out}")

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULT] RMSE: {rmse:.4f}")
    print(f"[RESULT] MAE : {mae:.4f}")
    print(f"[RESULT] R^2 : {r2:.4f}")
    if len(y_test) >= 2:
        pearson = np.corrcoef(y_test, y_pred)[0, 1]
        print(f"[RESULT] Pearson r: {pearson:.4f}")

    # Save predictions
    pd.DataFrame({
        "smiles": smiles_test,
        "y_true": y_test,
        "y_pred": y_pred,
    }).to_csv(args.preds_out, index=False)
    print(f"[INFO] Predictions saved to {args.preds_out}")


if __name__ == "__main__":
    main()
