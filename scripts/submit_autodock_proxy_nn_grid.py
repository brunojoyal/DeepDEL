#!/usr/bin/env python3
"""submit_autodock_proxy_nn_grid.py

Submit a grid hyperparameter search for the PyTorch ECFP proxy using Slurm.

This script submits many jobs that all run the same job script:
  job_autodock_proxy_nn.sh

Hyperparameters are passed via Slurm environment exports (sbatch --export).

Outputs
-------
Creates a directory like:
  outputs/autodock_proxy_nn_grid/<run_id>/

Inside it:
  - manifest.csv : one row per submitted job with hyperparams + job_id
  - runs/<run_name>/ : per-run outputs
      - model.pt
      - predictions.csv
      - slurm-%j.out (written by Slurm if your cluster is configured so)

Notes
-----
- This script does not require numpy/pandas/torch; it only shells out to sbatch.
- You can change the default grid below or pass a JSON grid via --grid-json.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _product(grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def _fmt(v: Any) -> str:
    # pretty + stable string for run names
    if isinstance(v, float):
        # avoid characters not great in filenames
        return f"{v:.3g}".replace("+", "").replace("-", "m").replace(".", "p")
    return str(v)


def run_name_from_params(p: Dict[str, Any]) -> str:
    parts = [
        f"nb{p['N_BITS']}",
        f"r{p['RADIUS']}",
        f"hd{p['HIDDEN_DIM']}",
        f"nl{p['N_LAYERS']}",
        f"do{_fmt(p['DROPOUT'])}",
        f"lr{_fmt(p['LR'])}",
        f"wd{_fmt(p['WEIGHT_DECAY'])}",
        f"bs{p['BATCH_SIZE']}",
        f"ep{p['EPOCHS']}",
        f"sd{p['SEED']}",
    ]
    if int(p.get("DEDUP_SMILES", 0)) == 1:
        parts.append("dedup")
    if int(p.get("STANDARDIZE_Y", 0)) == 1:
        parts.append("stdy")
    return "_".join(parts)


def sbatch_submit(job_script: str, export_env: Dict[str, str], extra_sbatch_args: List[str]) -> str:
    export_str = "ALL," + ",".join([f"{k}={v}" for k, v in export_env.items()])
    cmd = ["sbatch", "--parsable", "--export", export_str, *extra_sbatch_args, job_script]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        quoted_cmd = " ".join(shlex.quote(c) for c in e.cmd)
        msg = [
            f"sbatch submission failed with exit code {e.returncode}",
            f"Command: {quoted_cmd}",
        ]
        if e.stdout:
            msg.append(f"stdout:\n{e.stdout.rstrip()}")
        if e.stderr:
            msg.append(f"stderr:\n{e.stderr.rstrip()}")
        raise RuntimeError("\n".join(msg)) from e
    # sbatch --parsable returns jobid[;cluster]
    return res.stdout.strip().split(";")[0]


def main():
    ap = argparse.ArgumentParser("Submit Slurm grid search for train_autodock_proxy_nn.py")
    ap.add_argument("--job-script", default="jobs/job_autodock_proxy_nn.sh", help="Slurm job script to run")
    ap.add_argument(
        "--inputs",
        default="data/scored_libraries",
        help="Value for INPUTS env var passed to job script (string, can contain spaces).",
    )
    ap.add_argument(
        "--out-root",
        default="outputs/autodock_proxy_nn_grid",
        help="Root directory to store run manifests and per-run outputs.",
    )
    ap.add_argument("--run-id", default=None, help="Override run id (default: timestamp)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch commands that would be submitted, but do not submit.",
    )
    ap.add_argument(
        "--grid-json",
        default=None,
        help=(
            "Optional JSON file specifying a hyperparameter grid. Keys must match env vars in job script, "
            "values are lists. If omitted, uses the built-in default grid."
        ),
    )
    ap.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Extra sbatch args (repeatable), e.g. --sbatch-arg=--time=2:00:00",
    )
    args = ap.parse_args()

    if not Path(args.job_script).is_file():
        raise FileNotFoundError(f"Slurm job script not found: {args.job_script}")

    run_id = args.run_id or _now_id()
    out_root = Path(args.out_root) / run_id
    runs_dir = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Default grid (tweak freely)

    # grid: Dict[str, Sequence[Any]] = {
    #     "N_BITS": [2**12,2**13,2**14,2**15],
    #     "HIDDEN_DIM": [1024],
    #     "N_LAYERS": [2],
    #     "DROPOUT": [0.1],
    #     "LR": [1e-5],
    #     "WEIGHT_DECAY": [1e-2],
    #     "BATCH_SIZE": [1024],
    #     "EPOCHS": [10],
    #     "PATIENCE": [5],
    #     "VAL_SIZE": [0.1],
    #     "TEST_SIZE": [0.1],
    #     "SEED": [42],
    #     "STANDARDIZE_Y": [1],
    #     "DEDUP_SMILES": [1],
    #     "RADIUS": [2,3],
    # }
    grid: Dict[str, Sequence[Any]] = {
        "N_BITS": [2**14],
        "HIDDEN_DIM": [2048, 4096],
        "N_LAYERS": [2,3],
        "DROPOUT": [0.5, 0.6,0.7],
        "LR": [5e-6],
        "WEIGHT_DECAY": [1e-3],
        "BATCH_SIZE": [1024],
        "EPOCHS": [1000],
        "PATIENCE": [10],
        "VAL_SIZE": [0.1],
        "TEST_SIZE": [0.1],
        "SEED": [42],
        "STANDARDIZE_Y": [1],
        "DEDUP_SMILES": [1],
        "RADIUS": [3,4],
    }

    if args.grid_json:
        with open(args.grid_json, "r") as f:
            user_grid = json.load(f)
        if not isinstance(user_grid, dict):
            raise ValueError("--grid-json must contain a JSON object mapping key -> list")
        grid = user_grid

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = None

        for p in _product(grid):
            # Ensure all values are strings for env
            p_env = {k: str(v) for k, v in p.items()}

            run_name = run_name_from_params({k: (int(v) if str(v).isdigit() else v) for k, v in p_env.items()})
            run_dir = runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            export_env = dict(p_env)
            export_env["INPUTS"] = args.inputs
            export_env["MODEL_OUT"] = str(run_dir / "model.pt")
            export_env["PREDS_OUT"] = str(run_dir / "predictions.csv")

            if args.dry_run:
                job_id = "DRYRUN"
                cmd_preview = [
                    "sbatch",
                    "--parsable",
                    "--export",
                    "ALL," + ",".join([f"{k}={v}" for k, v in export_env.items()]),
                    *args.sbatch_arg,
                    args.job_script,
                ]
                print(" ".join(shlex.quote(c) for c in cmd_preview))
            else:
                job_id = sbatch_submit(args.job_script, export_env, args.sbatch_arg)
                print(f"[submit] {run_name} -> {job_id}")

            row = {"run_id": run_id, "run_name": run_name, "job_id": job_id, **export_env}
            if writer is None:
                cols = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
            writer.writerow(row)

    print(f"[done] manifest: {manifest_path}")
    print(f"[done] runs dir:  {runs_dir}")


if __name__ == "__main__":
    main()
