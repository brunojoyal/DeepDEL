#!/usr/bin/env python3
"""
histogram_from_csv.py

Create a histogram from a numeric column in one or more CSVs.

Usage examples:
  # Basic: save to histogram.png (single CSV)
  python histogram_from_csv.py --csv data.csv --column value

  # Merge all CSVs in a folder and plot
  python histogram_from_csv.py --folder path/to/csvs/ --column value

  # Control bins and output path
  python histogram_from_csv.py --csv data.csv --column value --bins 40 --out value_hist.png

  # Log scale on y-axis, density instead of counts
  python histogram_from_csv.py --csv data.csv --column value --logy --density

  # Clip extreme outliers to the 1st–99th percentiles before plotting
  python histogram_from_csv.py --csv data.csv --column value --clip-low 0.01 --clip-high 0.99

  # Explicit numeric range
  python histogram_from_csv.py --csv data.csv --column value --range -10 10

  # Hard numeric cutoffs (discard values below/above a threshold)
  python histogram_from_csv.py --csv data.csv --column value --min-cutoff 0
  python histogram_from_csv.py --csv data.csv --column value --max-cutoff 100
  python histogram_from_csv.py --csv data.csv --column value --min-cutoff 0 --max-cutoff 100

  # Compare two datasets on the same image. Add --csv2/--folder2 to overlay a
  # second dataset. Both distributions are normalized to area 1 (density) and
  # share the same bins, so counts do not need to match.
  python histogram_from_csv.py --folder path/to/csvs/ --csv2 other.csv --column value \
      --label1 library --label2 control --out comparison.png

  # Both sides can be a single CSV or a folder of CSVs; use --column2 if the
  # second dataset uses a different column name, and --alpha to tune overlap.
  python histogram_from_csv.py --csv data.csv --folder2 path/to/csvs/ --column score \
      --column2 reward --alpha 0.4 --out overlay.png
"""

import argparse
import sys
import math
import pathlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend so this works on servers/CLI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a histogram from a CSV column.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", default=None, help="Path to a single CSV file.")
    source.add_argument(
        "--folder",
        default=None,
        help="Path to a folder of CSV files; all *.csv files will be read and merged.",
    )
    source2 = p.add_mutually_exclusive_group(required=False)
    source2.add_argument(
        "--csv2",
        default=None,
        help="Optional path to a second CSV file to overlay on the same plot.",
    )
    source2.add_argument(
        "--folder2",
        default=None,
        help="Optional path to a folder of CSV files to overlay on the same plot.",
    )

    p.add_argument("--column", required=True, help="Column name to plot.")
    p.add_argument(
        "--column2",
        default=None,
        help="Column name for the second dataset (defaults to --column).",
    )
    p.add_argument("--out", default="histogram.png", help="Output image path (e.g., histogram.png).")
    p.add_argument(
        "--bins",
        type=str,
        default="auto",
        help='Number of bins or a numpy binning strategy, e.g. an integer "40" or one of: "auto", "fd", "sturges", "doane", "sqrt", "scott", "rice".',
    )
    p.add_argument("--density", action="store_true", help="Normalize to a density (area=1) instead of counts.")
    p.add_argument("--logx", action="store_true", help="Log scale on x-axis.")
    p.add_argument("--logy", action="store_true", help="Log scale on y-axis.")
    p.add_argument("--title", default=None, help="Optional plot title.")
    p.add_argument("--xlabel", default=None, help="Optional x-axis label (defaults to column name).")
    p.add_argument(
        "--label1",
        default=None,
        help="Legend label for the first dataset (defaults to its source path).",
    )
    p.add_argument(
        "--label2",
        default=None,
        help="Legend label for the second dataset (defaults to its source path).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency of the overlaid histograms in [0, 1] (default: 0.5).",
    )
    p.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not draw a legend when overlaying two datasets.",
    )
    p.add_argument(
        "--encoding",
        default=None,
        help="Optional file encoding for the CSV (e.g., utf-8, latin1). If omitted, pandas will guess."
    )
    p.add_argument(
        "--clip-low",
        type=float,
        default=None,
        help="Lower quantile in [0,1] for clipping (e.g., 0.01 keeps values >= 1st percentile)."
    )
    p.add_argument(
        "--clip-high",
        type=float,
        default=None,
        help="Upper quantile in [0,1] for clipping (e.g., 0.99 keeps values <= 99th percentile)."
    )
    p.add_argument(
        "--range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Explicit numeric range for the histogram (values outside are ignored)."
    )
    p.add_argument(
        "--min-cutoff",
        type=float,
        default=None,
        help="Discard values below this threshold (keep values >= MIN_CUTOFF)."
    )
    p.add_argument(
        "--max-cutoff",
        type=float,
        default=None,
        help="Discard values above this threshold (keep values <= MAX_CUTOFF)."
    )
    return p.parse_args()


def coerce_bins(bins_arg: str):
    """
    Convert --bins argument to either an int or a string recognized by numpy/matplotlib.
    """
    try:
        # If user passed an integer as string, convert it.
        as_int = int(bins_arg)
        if as_int <= 0:
            raise ValueError
        return as_int
    except ValueError:
        # Leave as a strategy string ("auto", "fd", etc.)
        return bins_arg


def maybe_clip(values: np.ndarray, qlow: Optional[float], qhigh: Optional[float]) -> np.ndarray:
    if qlow is None and qhigh is None:
        return values
    v = values.copy()
    if qlow is not None:
        if not (0.0 <= qlow <= 1.0):
            raise ValueError("--clip-low must be in [0,1].")
    if qhigh is not None:
        if not (0.0 <= qhigh <= 1.0):
            raise ValueError("--clip-high must be in [0,1].")
    if qlow is not None or qhigh is not None:
        lo = np.quantile(v, qlow) if qlow is not None else -np.inf
        hi = np.quantile(v, qhigh) if qhigh is not None else np.inf
        v = v[(v >= lo) & (v <= hi)]
    return v


def apply_cutoffs(values: np.ndarray, min_cutoff: Optional[float], max_cutoff: Optional[float]) -> np.ndarray:
    """Discard values below min_cutoff and/or above max_cutoff."""
    if min_cutoff is None and max_cutoff is None:
        return values
    v = values.copy()
    if min_cutoff is not None:
        v = v[v >= min_cutoff]
    if max_cutoff is not None:
        v = v[v <= max_cutoff]
    return v


def _load_source(csv_path: Optional[str], folder_path: Optional[str], encoding: Optional[str]) -> pd.DataFrame:
    """Load and combine CSV(s) from a single file or a folder into one DataFrame."""
    if csv_path is not None:
        # Single CSV path
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as e:
            print(f"Error reading CSV: {e}", file=sys.stderr)
            sys.exit(2)

    # Folder of CSVs
    folder = pathlib.Path(folder_path)
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.", file=sys.stderr)
        sys.exit(2)

    csv_files = sorted(folder.rglob("*.csv"))
    if not csv_files:
        print(f"Error: no *.csv files found in '{folder_path}'.", file=sys.stderr)
        sys.exit(2)

    dfs = []
    for fp in csv_files:
        try:
            dfs.append(pd.read_csv(fp, encoding=encoding))
        except Exception as e:
            print(f"Error reading '{fp}': {e}", file=sys.stderr)
            sys.exit(2)

    print(f"Loaded {len(csv_files)} CSV(s) from '{folder_path}'.")
    return pd.concat(dfs, ignore_index=True)


def _extract_values(df: pd.DataFrame, column: str) -> np.ndarray:
    """Coerce a column to numeric and return it as a numpy array (NaN -> 0)."""
    col = pd.to_numeric(df[column], errors="coerce").astype(float)
    return np.nan_to_num(col.to_numpy(), nan=0.0)


def _source_label(csv_path: Optional[str], folder_path: Optional[str], column: str) -> str:
    """Build a default legend label from the source path and column name."""
    src = csv_path if csv_path is not None else folder_path
    name = pathlib.Path(src).name if src is not None else "dataset"
    if name in ("", "."):
        name = "dataset"
    return f"{name}:{column}"

def _prepare_values(vals: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Apply quantile clipping and hard cutoffs to a values array."""
    if vals.size == 0:
        print("No data found in the selected column.", file=sys.stderr)
        sys.exit(4)

    try:
        vals = maybe_clip(vals, args.clip_low, args.clip_high)
    except ValueError as e:
        print(f"Invalid clipping arguments: {e}", file=sys.stderr)
        sys.exit(5)

    if vals.size == 0:
        print("All values removed by clipping; nothing to plot.", file=sys.stderr)
        sys.exit(6)

    if args.min_cutoff is not None or args.max_cutoff is not None:
        vals = apply_cutoffs(vals, args.min_cutoff, args.max_cutoff)
        if vals.size == 0:
            print("All values removed by cutoffs; nothing to plot.", file=sys.stderr)
            sys.exit(10)

    return vals



def main():
    args = parse_args()

    compare = args.csv2 is not None or args.folder2 is not None
    column1 = args.column
    column2 = args.column2 if args.column2 else args.column

    # Validate cutoff ordering once (the same cutoffs apply to both datasets).
    if (
        args.min_cutoff is not None
        and args.max_cutoff is not None
        and args.min_cutoff > args.max_cutoff
    ):
        print("--min-cutoff must be less than or equal to --max-cutoff.", file=sys.stderr)
        sys.exit(9)

    # Load and prepare the first dataset.
    df1 = _load_source(args.csv, args.folder, args.encoding)
    if column1 not in df1.columns:
        print(f"Column '{column1}' not found. Available columns: {list(df1.columns)}", file=sys.stderr)
        sys.exit(3)
    vals1 = _prepare_values(_extract_values(df1, column1), args)
    print(f"[dataset 1] Mean: {np.mean(vals1)}, Std: {np.std(vals1)}, "
          f"Min: {np.min(vals1)}, Max: {np.max(vals1)}")

    # Load and prepare the second dataset (only in compare mode).
    vals2 = None
    if compare:
        df2 = _load_source(args.csv2, args.folder2, args.encoding)
        if column2 not in df2.columns:
            print(f"Column '{column2}' not found. Available columns: {list(df2.columns)}", file=sys.stderr)
            sys.exit(3)
        vals2 = _prepare_values(_extract_values(df2, column2), args)
        print(f"[dataset 2] Mean: {np.mean(vals2)}, Std: {np.std(vals2)}, "
              f"Min: {np.min(vals2)}, Max: {np.max(vals2)}")

    # Histogram configuration.
    bins = coerce_bins(args.bins)
    hist_range: Optional[Tuple[float, float]] = None
    if args.range is not None:
        lo, hi = args.range
        if not (lo < hi):
            print("--range MIN must be less than MAX.", file=sys.stderr)
            sys.exit(7)
        hist_range = (lo, hi)

    xlabel = args.xlabel if args.xlabel else column1

    # Overlay mode: both distributions are normalized to area 1 and share bins,
    # so datasets with different numbers of elements are directly comparable.
    density = True if compare else args.density
    if compare and not args.density:
        print("Overlay mode: normalizing both histograms to density (area=1) for comparison.")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    if compare:
        combined = np.concatenate([vals1, vals2])
        bin_edges = np.histogram_bin_edges(combined, bins=bins, range=hist_range)
        label1 = args.label1 if args.label1 else _source_label(args.csv, args.folder, column1)
        label2 = args.label2 if args.label2 else _source_label(args.csv2, args.folder2, column2)
        ax.hist(
            vals1,
            bins=bin_edges,
            range=hist_range,
            density=True,
            alpha=args.alpha,
            label=label1,
            color="#1f77b4",
        )
        ax.hist(
            vals2,
            bins=bin_edges,
            range=hist_range,
            density=True,
            alpha=args.alpha,
            label=label2,
            color="#ff7f0e",
        )
        if not args.no_legend:
            ax.legend()
        ax.set_ylabel("Density")
    else:
        ax.hist(vals1, bins=bins, range=hist_range, density=density)
        ax.set_ylabel("Density" if density else "Count")

    ax.set_xlabel(xlabel)
    if args.title:
        ax.set_title(args.title)

    if args.logx:
        # Avoid log(<=0)
        if compare:
            positive = (np.concatenate([vals1, vals2]) > 0).sum()
            total = vals1.size + vals2.size
        else:
            positive = (vals1 > 0).sum()
            total = vals1.size
        if positive < total:
            print("Warning: some values <= 0 cannot be shown on a log-x axis.", file=sys.stderr)
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    try:
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving figure: {e}", file=sys.stderr)
        sys.exit(8)

    print(f"Saved histogram to: {args.out}")


if __name__ == "__main__":
    main()
