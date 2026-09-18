#!/usr/bin/env python3
"""collect_deepdel_grid_results.py

Collect results from a Slurm hyperparameter grid launched by
`submit_deepdel_grid.py`.

This mirrors `collect_autodock_proxy_nn_grid_results.py`: it scans each run
directory for a Slurm/stdout file and extracts printed metrics. It does *not*
load DeepDEL checkpoints.

DeepDEL training currently prints lines such as:

  Validation MSE = 0.123456
  ✓ Saved BEST to ... (MSE=0.123456)
  ✓ Saved LAST to ... (MSE=0.234567)
  [Early stopping] ...

The leaderboard is sorted by (best_mse asc, last_mse asc), with NaNs last.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


FLOAT = r"([0-9eE+\-.]+|nan|NaN|inf|Inf|INF)"
VAL_MSE_RE = re.compile(rf"^Validation\s+MSE\s*=\s*{FLOAT}\s*$")
SAVED_BEST_RE = re.compile(rf".*Saved\s+BEST\s+to\s+.*\(MSE={FLOAT}\)\s*$")
SAVED_LAST_RE = re.compile(rf".*Saved\s+LAST\s+to\s+.*\(MSE={FLOAT}\)\s*$")
EARLY_STOP_RE = re.compile(r"^\[Early stopping\]")


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def find_slurm_out(run_dir: Path) -> Optional[Path]:
    # Best effort: common patterns. Choose the largest file as likely most complete.
    candidates = list(run_dir.glob("slurm-*.out")) + list(run_dir.glob("*.out"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return candidates[0]


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_metrics_from_text(text: str) -> Dict[str, float]:
    val_mses: List[float] = []
    best_mse = float("nan")
    last_mse = float("nan")
    early_stopped = False

    for line in text.splitlines():
        m = VAL_MSE_RE.match(line)
        if m:
            val_mses.append(_safe_float(m.group(1)))
            continue

        m = SAVED_BEST_RE.match(line)
        if m:
            best_mse = _safe_float(m.group(1))
            continue

        m = SAVED_LAST_RE.match(line)
        if m:
            last_mse = _safe_float(m.group(1))
            continue

        if EARLY_STOP_RE.match(line):
            early_stopped = True

    finite_vals = [v for v in val_mses if math.isfinite(v)]
    min_val_mse = min(finite_vals) if finite_vals else float("nan")
    final_val_mse = val_mses[-1] if val_mses else float("nan")

    if not math.isfinite(best_mse):
        best_mse = min_val_mse
    if not math.isfinite(last_mse):
        last_mse = final_val_mse

    return {
        "best_mse": float(best_mse),
        "last_mse": float(last_mse),
        "min_val_mse": float(min_val_mse),
        "final_val_mse": float(final_val_mse),
        "n_val_points": float(len(val_mses)),
        "early_stopped": float(1 if early_stopped else 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser("Collect DeepDEL grid search results")
    ap.add_argument("--run_root", help="Path like outputs/deepdel_grid/<run_id>")
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
        metrics = parse_metrics_from_text(read_text(slurm_out)) if slurm_out else {}

        row = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "slurm_out": str(slurm_out) if slurm_out else "",
            "best_mse": metrics.get("best_mse", float("nan")),
            "last_mse": metrics.get("last_mse", float("nan")),
            "min_val_mse": metrics.get("min_val_mse", float("nan")),
            "final_val_mse": metrics.get("final_val_mse", float("nan")),
            "n_val_points": int(metrics.get("n_val_points", 0)),
            "early_stopped": int(metrics.get("early_stopped", 0)),
        }
        rows.append(row)

    # Sort: best = lowest best_mse, then lowest last_mse; NaNs last.
    def key(r):
        best = r["best_mse"]
        last = r["last_mse"]
        best_key = (1, 0.0) if best != best else (0, best)
        last_key = (1, 0.0) if last != last else (0, last)
        return best_key + last_key

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
            print(
                f"  {r['run_name']} | best_mse={r['best_mse']:.6g} "
                f"last_mse={r['last_mse']:.6g} n_val={r['n_val_points']}"
            )


if __name__ == "__main__":
    main()
