#!/usr/bin/env python3

"""Clean a CSV by dropping rows where a given column is NaN/empty.

Usage:
  python clean_csv_drop_nan.py -i input.csv -c COLUMN -o output.csv

Notes:
  - Treats as missing: true NaN values (pandas), empty strings, and common
    sentinel strings such as 'nan', 'na', 'null', 'none' (case-insensitive).
  - Preserves the original row order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Drop rows from a CSV where the specified column is NaN/empty and write a cleaned CSV."
        )
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input CSV path",
    )
    p.add_argument(
        "-c",
        "--column",
        required=True,
        help="Column name to check for missing values",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. If omitted, writes '<input_stem>_clean.csv' next to the input file."
        ),
    )
    p.add_argument(
        "--keep-original-na",
        action="store_true",
        help=(
            "Do not treat sentinel strings like 'nan', 'na', 'null', 'none' as missing; only drop true NaNs."
        ),
    )
    return p


def clean_csv_drop_nan(
    input_path: Path,
    column: str,
    output_path: Optional[Path] = None,
    keep_original_na: bool = False,
) -> tuple[int, int, Path]:
    """Return (kept_rows, dropped_rows, output_path)."""

    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            "pandas is required for this script. Install it with: pip install pandas"
        ) from e

    input_path = input_path.expanduser().resolve()
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_clean.csv")
    else:
        output_path = output_path.expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if column not in df.columns:
        cols = ", ".join(map(str, df.columns))
        raise SystemExit(f"Column '{column}' not found. Available columns: {cols}")

    missing = df[column].isna()
    if not keep_original_na:
        # Also treat empty/whitespace and common strings as missing.
        as_str = df[column].astype(str)
        normalized = as_str.str.strip().str.lower()
        sentinels = {"", "nan", "na", "n/a", "null", "none"}
        missing = missing | normalized.isin(sentinels)

    before = len(df)
    cleaned = df.loc[~missing].copy()
    after = len(cleaned)
    dropped = before - after

    cleaned.to_csv(output_path, index=False)
    return after, dropped, output_path


def main() -> None:
    args = _build_parser().parse_args()
    kept, dropped, out = clean_csv_drop_nan(
        input_path=args.input,
        column=args.column,
        output_path=args.output,
        keep_original_na=args.keep_original_na,
    )
    print(f"Wrote: {out}")
    print(f"Kept rows: {kept}")
    print(f"Dropped rows: {dropped}")


if __name__ == "__main__":
    main()

