#!/usr/bin/env python3
"""collect_autodock_proxy_nn_grid_results.py

Collect results from a Slurm hyperparameter grid launched by
`submit_autodock_proxy_nn_grid.py`.

This script scans each run directory for a Slurm output file and extracts
the printed metrics:
  [RESULT] RMSE: ...
  [RESULT] MAE : ...
  [RESULT] R^2 : ...

It writes a leaderboard CSV sorted by (RMSE asc, MAE asc).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional


RMSE_RE = re.compile(r"^\[RESULT\]\s+RMSE:\s+([0-9eE+\-\.]+)\s*$")
MAE_RE = re.compile(r"^\[RESULT\]\s+MAE\s*:\s+([0-9eE+\-\.]+)\s*$")
R2_RE = re.compile(r"^\[RESULT\]\s+R\^2\s*:\s+([0-9eE+\-\.]+)\s*$")


def parse_metrics_from_text(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        m = RMSE_RE.match(line)
        if m:
            out["rmse"] = float(m.group(1))
        m = MAE_RE.match(line)
        if m:
            out["mae"] = float(m.group(1))
        m = R2_RE.match(line)
        if m:
            out["r2"] = float(m.group(1))
    return out


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def find_slurm_out(run_dir: Path) -> Optional[Path]:
    # Best effort: common patterns.
    candidates = list(run_dir.glob("slurm-*.out")) + list(run_dir.glob("*.out"))
    if not candidates:
        return None
    # Choose the largest (most complete)
    candidates.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return candidates[0]


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_metrics_from_predictions_csv(preds_csv: Path) -> Dict[str, float]:
    """Compute RMSE/MAE/R^2 from a predictions CSV with columns y_true,y_pred.

    This avoids relying on Slurm stdout locations.
    """
    if not preds_csv.exists():
        return {}
    try:
        with preds_csv.open("r", newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                return {}
            if "y_true" not in r.fieldnames or "y_pred" not in r.fieldnames:
                return {}
            y_true = []
            y_pred = []
            for row in r:
                yt = _safe_float(row.get("y_true", ""))
                yp = _safe_float(row.get("y_pred", ""))
                # drop NaNs
                if yt != yt or yp != yp:
                    continue
                y_true.append(yt)
                y_pred.append(yp)
    except Exception:
        return {}

    if len(y_true) == 0:
        return {}

    # Compute metrics (no numpy dependency)
    n = float(len(y_true))
    se = 0.0
    ae = 0.0
    ybar = sum(y_true) / n
    ss_tot = 0.0
    ss_res = 0.0
    for yt, yp in zip(y_true, y_pred):
        diff = yp - yt
        se += diff * diff
        ae += abs(diff)
        ss_res += diff * diff
        dy = yt - ybar
        ss_tot += dy * dy

    mse = se / n
    rmse = mse ** 0.5
    mae = ae / n
    r2 = float("nan")
    if ss_tot > 0:
        r2 = 1.0 - (ss_res / ss_tot)

    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def main():
    ap = argparse.ArgumentParser("Collect NN proxy grid search results")
    ap.add_argument("--run_root", help="Path like outputs/autodock_proxy_nn_grid/<run_id>")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <run_root>/leaderboard.csv)")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    runs_dir = run_root / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")

    out_path = Path(args.out) if args.out else (run_root / "leaderboard.csv")
    rows = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        slurm_out = find_slurm_out(run_dir)
        preds_csv = run_dir / "predictions.csv"

        # Prefer computing from predictions.csv (robust). Fall back to slurm stdout parsing.
        metrics = compute_metrics_from_predictions_csv(preds_csv)
        if not metrics and slurm_out:
            metrics = parse_metrics_from_text(read_text(slurm_out))

        row = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "slurm_out": str(slurm_out) if slurm_out else "",
            "preds_csv": str(preds_csv) if preds_csv.exists() else "",
            "rmse": metrics.get("rmse", float("nan")),
            "mae": metrics.get("mae", float("nan")),
            "r2": metrics.get("r2", float("nan")),
        }
        rows.append(row)

    # Sort: best = lowest RMSE, then lowest MAE; NaNs last.
    def key(r):
        rmse = r["rmse"]
        mae = r["mae"]
        rmse_key = (1, 0.0) if rmse != rmse else (0, rmse)
        mae_key = (1, 0.0) if mae != mae else (0, mae)
        return rmse_key + mae_key

    rows.sort(key=key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_name"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] wrote {out_path}")
    if rows:
        print("Top 5:")
        for r in rows[:5]:
            print(f"  {r['run_name']} | rmse={r['rmse']:.4g} mae={r['mae']:.4g} r2={r['r2']:.4g}")


if __name__ == "__main__":
    main()
