#!/usr/bin/env python3
"""Predict expected AmpC pKi>=6 hits for one or more scored-library CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepdelgfn.ampc_hitrate import (  # noqa: E402
    DEFAULT_PKI,
    DEFAULT_SCORE_FLOOR,
    ampc_expected_hits,
    ampc_hit_proportion,
    clean_ampc_docking_scores,
)


def _read_scores(path: Path, score_col: str) -> list[float]:
    scores: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if score_col not in reader.fieldnames:
            raise ValueError(
                f"CSV {path} is missing score column {score_col!r}; "
                f"columns={reader.fieldnames}"
            )
        for row in reader:
            try:
                scores.append(float(row.get(score_col, "")))
            except (TypeError, ValueError):
                continue
    return scores


def predict_one(path: Path, *, score_col: str, pki: float, score_floor: float) -> dict[str, object]:
    raw_scores = _read_scores(path, score_col)
    scores = clean_ampc_docking_scores(raw_scores, score_floor=score_floor)
    prop = ampc_hit_proportion(scores, pki=pki, score_floor=score_floor)
    expected = ampc_expected_hits(scores, pki=pki, score_floor=score_floor)
    return {
        "library": str(path),
        "n_molecules": int(scores.size),
        f"prop_pki_ge_{pki:g}": prop,
        f"expected_hits_pki_ge_{pki:g}": expected,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("libraries", nargs="+", help="Scored-library CSV file(s).")
    ap.add_argument(
        "--score-col",
        default="docking_score",
        help="Docking score column name (default: docking_score).",
    )
    ap.add_argument(
        "--pki",
        type=float,
        default=DEFAULT_PKI,
        help="pKi threshold (default: 6.0 = 1 µM).",
    )
    ap.add_argument(
        "--clamp-min",
        type=float,
        default=DEFAULT_SCORE_FLOOR,
        help="Replace scores below this by this value (default: -120).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output CSV path. If omitted, print CSV to stdout.",
    )
    args = ap.parse_args()

    rows = [
        predict_one(Path(p), score_col=args.score_col, pki=args.pki, score_floor=args.clamp_min)
        for p in args.libraries
    ]
    fieldnames = list(rows[0].keys()) if rows else []

    if args.out:
        out = Path(args.out)
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} row(s) to {out}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()