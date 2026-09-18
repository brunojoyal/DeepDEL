#!/usr/bin/env python3
"""
csv_hist.py — Plot a histogram for a column in a CSV.

Usage examples:
  python csv_hist.py --file data.csv --column Age
  python csv_hist.py -f data.csv -c "Sale Price" --bins 40 --out hist.png
  python csv_hist.py -f data.csv -c score --delimiter ';' --title "Score distribution"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot a histogram for a CSV column.")
    p.add_argument("-f", "--file", required=True, help="Path to the CSV file.")
    p.add_argument("-c", "--column", required=True, help="Column name to plot.")
    p.add_argument("-d", "--delimiter", default=None,
                   help="CSV delimiter (auto-detected by pandas if omitted).")
    p.add_argument("-b", "--bins", type=int, default=30,
                   help="Number of histogram bins (default: 30).")
    p.add_argument("--dropna", action="store_true",
                   help="Drop NaN values (after numeric conversion).")
    p.add_argument("--out", default=None,
                   help="Save the plot to this path instead of showing it (e.g., hist.png).")
    p.add_argument("--title", default=None, help="Custom title for the plot.")
    p.add_argument("--width", type=int, default=8, help="Figure width in inches (default: 8).")
    p.add_argument("--height", type=int, default=5, help="Figure height in inches (default: 5).")
    p.add_argument("--exclude-zeros", action="store_true",
               help="Exclude exact zeros before plotting.")
    return p.parse_args()


def main():

    
    args = parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        sys.exit(f"Error: file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, sep=args.delimiter)
    except Exception as e:
        sys.exit(f"Error reading CSV: {e}")

    if args.column not in df.columns:
        cols = ", ".join(map(str, df.columns.tolist()))
        sys.exit(f"Error: column '{args.column}' not found.\nAvailable columns: {cols}")

    # Coerce the selected column to numeric; non-numeric values become NaN
    s = pd.to_numeric(df[args.column], errors="coerce")

    if args.dropna:
        s = s.dropna()

    zeros_mask = (s == 0)
    num_zeros = int(zeros_mask.sum())
    s_nz = s[~zeros_mask]

    if args.exclude_zeros:
        s_plot = s_nz
    else:
        s_plot = s


    if s.empty:
        sys.exit("Error: no numeric data to plot after conversion/dropna.")

    plt.figure(figsize=(args.width, args.height))
    data = s_plot
    plt.hist(data, bins=args.bins)
    xlabel = args.column
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(args.title or f"Histogram of {args.column}")
    plt.tight_layout()

    if args.out:
        try:
            plt.savefig(args.out, dpi=150)
            print(f"Saved histogram to {args.out}")
        except Exception as e:
            sys.exit(f"Error saving figure: {e}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
