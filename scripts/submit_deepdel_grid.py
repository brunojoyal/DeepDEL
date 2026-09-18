#!/usr/bin/env python3
"""submit_deepdel_grid.py

Submit a Slurm hyperparameter grid for offline DeepDEL training.

This is the DeepDEL analogue of scripts/submit_autodock_proxy_nn_grid.py.  Each
submitted job runs jobs/job_deepdel_offline.sh and trains
`deepdelgfn.deepdel.train_offline` using a user-supplied DeepDEL dataset CSV.

Required inputs
---------------
DeepDEL training requires two CSV files:

  * --bbs: building block CSV with at least ID and SMILES columns
  * --dataset: DeepDEL dataset CSV with B1_id,B2_id,B3_id,y columns

Outputs
-------
Creates a directory like:

  outputs/deepdel_grid/<run_id>/

Inside it:

  * manifest.csv: one row per submitted job with hyperparams + job_id
  * runs/<run_name>/slurm-<jobid>.out

By default, grid jobs do not save DeepDEL checkpoints. Use --save-checkpoints
to write best_deepdel.pt and last_deepdel.pt per run.

Hyperparameters are passed via Slurm environment exports (sbatch --export).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
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
    """Pretty + stable string for run names."""
    if isinstance(v, float):
        return f"{v:.3g}".replace("+", "").replace("-", "m").replace(".", "p")
    return str(v).replace("/", "-").replace(" ", "")


def _as_int_if_int_string(v: str) -> Any:
    try:
        if str(v).strip().isdigit() or (str(v).startswith("-") and str(v)[1:].isdigit()):
            return int(v)
    except Exception:
        pass
    return v


def _truthy(p: Dict[str, Any], key: str) -> bool:
    try:
        return int(p.get(key, 0)) == 1
    except Exception:
        return str(p.get(key, "")).lower() in {"true", "yes", "on"}


def run_name_from_params(p: Dict[str, Any]) -> str:
    parts = [
        f"fp{p['BB_FP_BITS']}",
        f"r{p['BB_FP_RADIUS']}",
        f"hd{p['HIDDEN_DIM']}",
        f"rho{p['RHO_DIM']}",
        f"do{_fmt(p['DROPOUT'])}",
        f"lr{_fmt(p['LR'])}",
        f"wd{_fmt(p['WEIGHT_DECAY'])}",
        f"bs{p['BATCH_SIZE']}",
        f"ep{p['EPOCHS']}",
        f"pat{p['PATIENCE']}",
        f"vf{_fmt(p['VAL_FRAC'])}",
        f"pool{p.get('POOLING', 'mean')}",
        f"sd{p['SEED']}",
    ]
    if _truthy(p, "SHARED_PHI"):
        parts.append("sharedphi")
    if _truthy(p, "LOG_TARGET"):
        parts.append("logy")
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


def _has_sbatch_option(args: Sequence[str], option: str) -> bool:
    """Return True if args already contains an sbatch option like --output."""
    prefix = option + "="
    return any(a == option or a.startswith(prefix) for a in args)


def main() -> None:
    ap = argparse.ArgumentParser("Submit Slurm grid search for DeepDEL offline training")
    ap.add_argument("--job-script", default="jobs/job_deepdel_offline.sh", help="Slurm job script to run")
    ap.add_argument("--bbs", required=True, help="bbs.csv with ID and SMILES columns")
    ap.add_argument("--dataset", required=True, help="DeepDEL dataset CSV with B1_id,B2_id,B3_id,y columns")
    ap.add_argument(
        "--out-root",
        default="outputs/deepdel_grid",
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
            "Optional JSON file specifying a hyperparameter grid. Keys must match env vars in "
            "jobs/job_deepdel_offline.sh, values are lists. If omitted, uses the built-in grid."
        ),
    )
    ap.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Extra sbatch args (repeatable), e.g. --sbatch-arg=--time=8:00:00",
    )
    ap.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save best_deepdel.pt and last_deepdel.pt in each run directory. Default: no checkpoints.",
    )
    ap.add_argument(
        "--save-best-only",
        action="store_true",
        help="With --save-checkpoints, save only best_deepdel.pt and disable last_deepdel.pt.",
    )
    args = ap.parse_args()

    if args.save_best_only and not args.save_checkpoints:
        raise ValueError("--save-best-only requires --save-checkpoints")

    if not Path(args.job_script).is_file():
        raise FileNotFoundError(f"Slurm job script not found: {args.job_script}")
    if not Path(args.bbs).is_file():
        raise FileNotFoundError(f"bbs CSV not found: {args.bbs}")
    if not Path(args.dataset).is_file():
        raise FileNotFoundError(f"DeepDEL dataset CSV not found: {args.dataset}")

    run_id = args.run_id or _now_id()
    out_root = Path(args.out_root) / run_id
    runs_dir = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Default grid (tweak freely or replace with --grid-json).
    grid: Dict[str, Sequence[Any]] = {
        "BB_FP_BITS": [1024,2048],
        "BB_FP_RADIUS": [2],
        "HIDDEN_DIM": [1024,2048],
        "RHO_DIM": [1024,2048],
        "DROPOUT": [0],
        "LR": [1e-4,1e-5],
        "WEIGHT_DECAY": [1e-6,1e-7],
        "BATCH_SIZE": [1024],
        "EPOCHS": [50],
        "PATIENCE": [8],
        "VAL_FRAC": [0.2],
        "VAL_BINS": [10],
        "VALIDATION_TYPE": ["random"], #stratified or random
        "SEED": [0],
        "SHARED_PHI": [0],
        "POOLING": ["sum"],
        "LOG_TARGET": [1],
        "NUM_WORKERS": [-1],
        "PREFETCH_FACTOR": [4],
        "PERSISTENT_WORKERS": [1],
        "PIN_MEMORY": [1],
        "AMP": [1],
        "TF32": [1],
        "MAX_WORKERS": [32],
    }

    if args.grid_json:
        with open(args.grid_json, "r") as f:
            user_grid = json.load(f)
        if not isinstance(user_grid, dict):
            raise ValueError("--grid-json must contain a JSON object mapping key -> list")
        for key, values in user_grid.items():
            if not isinstance(values, list):
                raise ValueError(f"Grid value for {key!r} must be a list")
        grid = user_grid

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = None

        for p in _product(grid):
            # Ensure all values are strings for env exports.
            p_env = {k: str(v) for k, v in p.items()}

            run_name = run_name_from_params({k: _as_int_if_int_string(v) for k, v in p_env.items()})
            run_dir = runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            slurm_out = run_dir / "slurm-%j.out"

            export_env = dict(p_env)
            export_env["RUN_ID"] = run_id
            export_env["BBS"] = args.bbs
            export_env["DATASET"] = args.dataset
            if args.save_checkpoints:
                export_env["SAVE_BEST"] = str(run_dir / "best_deepdel.pt")
                export_env["SAVE_LAST"] = "" if args.save_best_only else str(run_dir / "last_deepdel.pt")
            else:
                export_env["SAVE_BEST"] = ""
                export_env["SAVE_LAST"] = ""

            sbatch_args = list(args.sbatch_arg)
            if not _has_sbatch_option(sbatch_args, "--output"):
                sbatch_args = ["--output", str(slurm_out), *sbatch_args]

            if args.dry_run:
                job_id = "DRYRUN"
                cmd_preview = [
                    "sbatch",
                    "--parsable",
                    "--export",
                    "ALL," + ",".join([f"{k}={v}" for k, v in export_env.items()]),
                    *sbatch_args,
                    args.job_script,
                ]
                print(" ".join(shlex.quote(c) for c in cmd_preview))
            else:
                job_id = sbatch_submit(args.job_script, export_env, sbatch_args)
                print(f"[submit] {run_name} -> {job_id}")

            row = {
                "run_id": run_id,
                "run_name": run_name,
                "job_id": job_id,
                "slurm_out": str(slurm_out),
                **export_env,
            }
            if writer is None:
                cols = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
            writer.writerow(row)

    print(f"[done] manifest: {manifest_path}")
    print(f"[done] runs dir:  {runs_dir}")


if __name__ == "__main__":
    main()
