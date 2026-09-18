#!/usr/bin/env python3
"""
quantile_of_value.py

Compute two quantiles for a value within a CSV column:
1. The empirical quantile (percentile rank in [0,1]) based on the observed dataset.
2. A theoretical quantile assuming the dataset is normally distributed.

By default the empirical quantile uses the "mean" definition (average of strict and
weak ranks), matching SciPy's percentileofscore(..., kind="mean") / 100.

Usage:
  # Single CSV
  python quantile_of_value.py --csv path/to/file.csv --column col_name --value 3.14

  # Merge all CSVs in a folder
  python quantile_of_value.py --folder path/to/csvs/ --column col_name --value 3.14

  # Optional: choose empirical ranking method among: mean (default), strict, weak
  python quantile_of_value.py --csv data.csv --column x --value 10 --method strict
"""

import argparse
import math
import pathlib
import sys
from typing import Literal

import pandas as pd
import numpy as np


Method = Literal["mean", "strict", "weak"]


def percentile_rank(values: np.ndarray, x: float, method: Method = "mean") -> float:
    """
    Return empirical quantile of x in values as a float in [0, 1].

    methods:
      - "strict":   P(V <  x)
      - "weak":     P(V <= x)
      - "mean":     average of strict and weak (handles ties more neutrally)

    NaNs in `values` are ignored.
    """
    vals = values[~np.isnan(values)]
    n = vals.size
    if n == 0:
        raise ValueError("Selected column has no numeric data after dropping NaNs.")

    # Handle +/- inf and NaN x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("Provided value is NaN; cannot compute quantile.")

    # Vectorized counts
    if method == "strict":
        count = np.count_nonzero(vals < x)
        return count / n
    elif method == "weak":
        count = np.count_nonzero(vals <= x)
        return count / n
    elif method == "mean":
        count_lt = np.count_nonzero(vals < x)
        count_le = np.count_nonzero(vals <= x)
        return 0.5 * (count_lt + count_le) / n
    else:
        raise ValueError(f"Unknown method: {method}")


def normal_cdf_quantile(values: np.ndarray, x: float) -> float:
    """
    Return the theoretical quantile of x in [0, 1] assuming values follow a normal
    distribution parameterized by the sample mean and population standard deviation.

    NaNs in `values` are ignored.
    """
    vals = values[~np.isnan(values)]
    n = vals.size
    if n == 0:
        raise ValueError("Selected column has no numeric data after dropping NaNs.")

    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=0))

    if sigma == 0.0:
        return 0.5 if x == mu else (0.0 if x < mu else 1.0)

    z = (x - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """Load and combine CSV(s) into a single DataFrame."""
    if args.csv is not None:
        try:
            return pd.read_csv(args.csv, encoding=args.encoding)
        except Exception as e:
            print(f"Error reading CSV: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        folder = pathlib.Path(args.folder)
        if not folder.is_dir():
            print(f"Error: '{args.folder}' is not a directory.", file=sys.stderr)
            sys.exit(2)

        csv_files = sorted(folder.rglob("*.csv"))
        if not csv_files:
            print(f"Error: no *.csv files found in '{args.folder}'.", file=sys.stderr)
            sys.exit(2)

        dfs = []
        for fp in csv_files:
            try:
                dfs.append(pd.read_csv(fp, encoding=args.encoding))
            except Exception as e:
                print(f"Error reading '{fp}': {e}", file=sys.stderr)
                sys.exit(2)

        print(f"Loaded {len(csv_files)} CSV(s) from '{args.folder}'.")
        return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Quantile of a value within a CSV column.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", default=None, help="Path to a single CSV file.")
    source.add_argument(
        "--folder",
        default=None,
        help="Path to a folder of CSV files; all *.csv files will be read and merged.",
    )
    parser.add_argument("--column", required=True, help="Column name to analyze.")
    parser.add_argument("--value", required=True, type=float, help="Value whose quantile to compute.")
    parser.add_argument(
        "--method",
        choices=["mean", "strict", "weak"],
        default="mean",
        help='Ranking method: "mean" (default), "strict" (<), or "weak" (<=).'
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional file encoding (e.g., utf-8, latin1). If omitted, pandas will guess."
    )
    args = parser.parse_args()

    df = _load_dataframe(args)

    if args.column not in df.columns:
        print(f"Column '{args.column}' not found. Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(3)

    col = df[args.column]

    # Coerce to numeric (so numbers stored as strings work); drop non-convertible values as NaN
    col_num = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)

    try:
        empirical_q = percentile_rank(col_num, args.value, method=args.method)  # in [0,1]
        normal_q = normal_cdf_quantile(col_num, args.value)  # in [0,1]
    except Exception as e:
        print(f"Error computing quantile: {e}", file=sys.stderr)
        sys.exit(4)

    # Print both empirical and normal-assumption quantiles as fractions and percentages.
    print(f"empirical_quantile: {empirical_q:.6f}")
    print(f"empirical_percentage: {empirical_q * 100:.4f}%")
    print(f"normal_quantile: {normal_q:.6f}")
    print(f"normal_percentage: {normal_q * 100:.4f}%")


if __name__ == "__main__":
    main()
