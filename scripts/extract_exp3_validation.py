#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract a stratified validation set from a DeepDEL dataset CSV.

The DeepDEL dataset has columns ``B1_id,B2_id,B3_id,threshold,y``. The target
``y`` is a continuous regression value, so stratification is performed on
``y`` binned into equal-frequency quantiles (matching the convention used by
``deepdelgfn.deepdel.train_offline.build_csv_split_stratified``, i.e.
``pd.qcut(y, q=n_bins, duplicates="drop")``).

Samples are drawn without replacement, proportionally to each bin's size, so
the validation subset preserves the source ``y`` distribution.

Self-contained: uses only the Python standard library (``csv`` + ``random``)
so it runs without pandas/numpy. The source CSV is never modified.
"""

import argparse
import csv
import os
import random


def allocate_samples(target_total, bin_sizes):
    """Allocate ``target_total`` samples across bins proportionally.

    Uses the largest-remainder method so the total is exactly ``target_total``,
    with each bin getting at least one sample (when it has at least one row)
    and never more than its size.
    """
    total = sum(bin_sizes)
    if target_total > total:
        raise ValueError(
            f"requested {target_total} samples but only {total} rows available"
        )

    base = []
    for size in bin_sizes:
        base.append((target_total * size) // total)

    # Distribute the remainder to bins with the largest fractional part.
    remainder = target_total - sum(base)
    frac = [
        (target_total * size / total) - (target_total * size // total)
        for size in bin_sizes
    ]
    order = sorted(range(len(bin_sizes)), key=lambda j: (-frac[j], j))
    for j in order[:remainder]:
        base[j] += 1

    # Clamp to each bin's size and make sure non-empty bins keep >= 1 sample.
    for j in range(len(bin_sizes)):
        base[j] = min(base[j], bin_sizes[j])
    for j in range(len(bin_sizes)):
        if bin_sizes[j] > 0 and base[j] == 0:
            # Steal one sample from the bin with the largest surplus.
            donor = max(
                range(len(bin_sizes)),
                key=lambda k: base[k] if base[k] > 1 else -1,
            )
            if base[donor] > 1:
                base[donor] -= 1
                base[j] += 1

    if sum(base) != target_total:
        raise RuntimeError(
            f"internal allocation error: got {sum(base)}, expected {target_total}"
        )
    return base


def main():
    ap = argparse.ArgumentParser(
        description="Extract a stratified validation set from a DeepDEL CSV."
    )
    ap.add_argument(
        "--dataset",
        default="outputs/deepdel_dataset.csv",
        help="Source CSV with B1_id,B2_id,B3_id,threshold,y",
    )
    ap.add_argument(
        "--output",
        default="datasets/exp3_validation.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of rows to extract",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of y quantile bins for stratification",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = ap.parse_args()

    # Read rows once; preserve the original column order and header.
    header = None
    rows = []
    ys = []
    with open(args.dataset, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            rows.append(row)
            ys.append(float(row[header.index("y")]))

    n = len(rows)
    if n == 0:
        raise ValueError("source dataset is empty")
    if args.num_samples > n:
        raise ValueError(
            f"cannot extract {args.num_samples} samples from {n} rows"
        )

    # Order rows by y, then chunk into equal-frequency quantile bins (the
    # stdlib equivalent of pd.qcut(y, q=n_bins, duplicates="drop")).
    order = sorted(range(n), key=lambda i: ys[i])
    bins = min(args.bins, n)
    bin_sizes = [n // bins] * bins
    for j in range(n % bins):
        bin_sizes[j] += 1

    # Allocate the requested samples across bins and sample each bin.
    allocations = allocate_samples(args.num_samples, bin_sizes)
    rng = random.Random(args.seed)

    selected = []
    start = 0
    for size, k in zip(bin_sizes, allocations):
        chunk = order[start : start + size]
        start += size
        selected.extend(rng.sample(chunk, k))

    selected.sort()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in selected:
            writer.writerow(rows[i])

    print(f"[extract] source rows: {n:,}")
    print(f"[extract] bins: {bins}, per-bin allocation: {allocations}")
    print(f"[extract] extracted {len(selected):,} rows -> {args.output}")
    print(f"[extract] seed: {args.seed}")


if __name__ == "__main__":
    main()
