#!/usr/bin/env python3
"""
grid_mols.py — Draw selected molecules from a CSV into a single PNG grid.

Requirements:
  - rdkit (e.g., conda install -c conda-forge rdkit)
  - pandas (e.g., pip install pandas)

Examples:
  # show SMILES under each molecule (default)
  python grid_mols.py --csv molecules.csv --ids 101,102,103 --out grid.png

  # hide labels entirely
  python grid_mols.py --csv molecules.csv --ids-file ids.txt --no-labels --out grid.png

  # specify grid layout
  python grid_mols.py --csv molecules.csv --ids 10,11,12,13 --rows 2 --cols 2 --out grid.png
"""
import argparse
import math
import sys
from typing import List, Tuple, Optional

import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def parse_args():
    p = argparse.ArgumentParser(description="Draw selected molecules from a CSV into a PNG grid.")
    p.add_argument("--csv", required=True, help="Path to CSV containing molecules.")
    p.add_argument("--smiles-col", default="SMILES", help="SMILES column name (default: SMILES).")
    p.add_argument("--id-col", default="ID", help="ID column name (default: ID).")

    ids = p.add_mutually_exclusive_group(required=True)
    ids.add_argument("--ids", help="Comma-separated list of IDs to draw.")
    ids.add_argument("--ids-file", help="Text file with one ID per line.")

    p.add_argument("--rows", type=int, help="Number of rows in the grid.")
    p.add_argument("--cols", type=int, help="Number of columns in the grid.")
    p.add_argument("--cell", nargs=2, type=int, metavar=("W", "H"),
                   default=[300, 300], help="Sub-image (cell) size in pixels, default 300 300.")
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--no-labels", action="store_true",
                   help="If set, do not render labels under molecules. (Default: show SMILES)")
    p.add_argument("--no-sanitize", action="store_true", help="Disable RDKit sanitization (not recommended).")
    p.add_argument("--kekulize", action="store_true", help="Attempt kekulization before drawing.")
    p.add_argument("--strict-grid", action="store_true",
                   help="If set and N > rows*cols, truncate to fit instead of auto-expanding rows.")
    return p.parse_args()


def read_id_list(args) -> List[str]:
    if args.ids:
        return [tok.strip() for tok in args.ids.split(",") if tok.strip()]
    with open(args.ids_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def choose_grid(n: int, rows: Optional[int], cols: Optional[int]) -> Tuple[int, int]:
    if rows and cols:
        return rows, cols
    if cols and not rows:
        rows = math.ceil(n / cols)
        return rows, cols
    if rows and not cols:
        cols = math.ceil(n / rows)
        return rows, cols
    cols = math.ceil(math.sqrt(n))  # near-square
    rows = math.ceil(n / cols)
    return rows, cols


def prepare_mol(smiles: str, sanitize: bool, kekulize: bool) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=not sanitize is False)
        if mol is None and not sanitize:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass
        return mol
    except Exception:
        return None


def main():
    args = parse_args()
    raw_ids = read_id_list(args)
    if len(raw_ids) == 0:
        print("No IDs provided.", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns or args.id_col not in df.columns:
        print(f"CSV must contain columns '{args.smiles_col}' and '{args.id_col}'.", file=sys.stderr)
        sys.exit(2)

    # Robust ID type handling (int vs string)
    id_series = df[args.id_col]
    if is_integer_dtype(id_series):
        try:
            ids = [int(x) for x in raw_ids]
        except ValueError:
            print("Some provided IDs are not integers, but the CSV ID column is integer-typed.",
                  file=sys.stderr)
            sys.exit(2)
        df_indexed = df.set_index(args.id_col, drop=False)
    else:
        df[args.id_col] = df[args.id_col].astype(str).str.strip()
        ids = [str(x).strip() for x in raw_ids]
        df_indexed = df.set_index(args.id_col, drop=False)

    smiles_list, keep_ids = [], []
    missing = []
    for _id in ids:
        if _id not in df_indexed.index:
            missing.append(str(_id))
            continue
        smiles_list.append(str(df_indexed.loc[_id, args.smiles_col]))
        keep_ids.append(_id)

    if len(keep_ids) == 0:
        print("None of the requested IDs were found in the CSV.", file=sys.stderr)
        if len(missing):
            print("Missing IDs: " + ", ".join(missing), file=sys.stderr)
        sys.exit(1)

    # Build molecules and legends (labels=SMILES unless --no-labels)
    mols, legends = [], []
    for smi in smiles_list:
        mol = prepare_mol(smi, sanitize=not args.no_sanitize, kekulize=args.kekulize)
        if mol is None:
            mols.append(None)
            legends.append(f"{smi} (parse error)" if not args.no_labels else "")
        else:
            mols.append(mol)
            legends.append("" if args.no_labels else smi)

    failed_count = sum(1 for m in mols if m is None)
    # Remove failed ones (and corresponding legends)
    keep = [(m, l) for m, l in zip(mols, legends) if m is not None]
    if not keep:
        print("All requested molecules failed to parse/draw.", file=sys.stderr)
        sys.exit(1)
    mols, legends = zip(*keep)
    mols, legends = list(mols), list(legends)

    n = len(mols)
    rows, cols = choose_grid(n, args.rows, args.cols)

    capacity = rows * cols
    if n > capacity:
        if args.strict_grid:
            mols = mols[:capacity]
            legends = legends[:capacity]
            n = capacity
        else:
            rows = math.ceil(n / cols)

    w, h = args.cell
    # If labels are hidden, pass None so RDKit doesn't reserve legend space
    legends_arg = None if args.no_labels else legends

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=(w, h),
        legends=legends_arg,
        useSVG=False,
        returnPNG=False,
    )
    img.save(args.out)

    print(f"Saved {n} molecule(s) to {args.out} in a {rows}x{cols} grid (cell {w}x{h}px).")
    if missing:
        print(f"IDs not found in CSV ({len(missing)}): {', '.join(missing)}", file=sys.stderr)
    if failed_count:
        print(f"Failed to parse/draw {failed_count} molecule(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
