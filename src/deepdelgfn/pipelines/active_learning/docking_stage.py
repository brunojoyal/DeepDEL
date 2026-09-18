#!/usr/bin/env python3
"""Docking-stage functions for the active learning pipeline.

This module was extracted from active_learning_stage.py.  It covers the
``dock_prepare``, ``dock_batch``, ``dock_finalize``, ``dock_launch``, and
``dock`` (legacy) stages, together with DOCK3 command builders, manifest
I/O, Slurm submission helpers, and per-candidate output aggregation.
"""

import csv
import fcntl
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from deepdelgfn.rewards import reward

from .config import (
    BBS_CSV,
    BBS_FLAGS,
    CFG,
    CFG_PATH,
    DOCK3_DOCKFILES_DIR,
    DOCK3_INDOCK_TEMPLATE,
    DOCK3_LIGBUILD_TIMEOUT,
    DOCK3_TIMEOUT,
    DOCK_CPU,
    DOCK_JOBS,
    DOCKING_OUT_DIR,
    PROJECT_ROOT,
    RUN_ID,
    RUN_ROOT,
    _autodock_kind,
    _autodock_model_path,
    _cfg_get,
    _concat_csvs_with_matching_headers,
    _csv_data_row_count,
    _gfn_reward_source,
    _outer_deepdel_paths,
    _plot_title_metadata,
    _outer_threshold,
    _reaction_mode,
    _reward_weight_flags,
    _fixed_threshold,
    _vina_target,
    run_cmd,
)
from .docking_candidates import (
    _canonical_smiles,
    _load_previous_docking_scores,
    gather_docking_candidates_by_inner_loop,
)
from .inner import _inner_dir, _inner_docking_deepdel_path


# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

_DOCK_SCORE_COL_CANDIDATES = ("docking_score", "trimer_score", "score")
_AL_CANDIDATE_COL = "__al_candidate_index"
_AL_DOCKING_CSV_COL = "__al_docking_csv"
_AL_SCORE_SOURCE_COL = "score_source"


def _docking_score_column(df: pd.DataFrame) -> str:
    for c in _DOCK_SCORE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(
        f"Docking CSV is missing a recognised score column "
        f"(expected one of {_DOCK_SCORE_COL_CANDIDATES})."
    )


def compute_threshold_reward_from_docking_csv(
    path: Path, threshold: float, alpha: float = 1.0
) -> float:
    df = pd.read_csv(path)
    col = _docking_score_column(df)
    vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    return float(reward(vals, mode="threshold", threshold=threshold, alpha=alpha))


# ---------------------------------------------------------------------------
# CSV utility (specific to docking)
# ---------------------------------------------------------------------------


def _write_csv_remaining_after_prefix(input_csv: Path, out_csv: Path, done_rows: int) -> int:
    """Write input rows after the first ``done_rows`` data rows to ``out_csv``."""
    written = 0
    with input_csv.open("r", newline="", encoding="utf-8") as fin, out_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator="\n")
        header = next(reader)
        writer.writerow(header)
        for i, row in enumerate(reader):
            if i >= int(done_rows):
                writer.writerow(row)
                written += 1
    return written


# ---------------------------------------------------------------------------
# Docking manifest I/O
# ---------------------------------------------------------------------------


def _docking_pool_dir(outer_loop: int, inner_loop: Optional[int] = None) -> Path:
    outer_dir, _, _ = _outer_deepdel_paths(outer_loop)
    if inner_loop is not None:
        return outer_dir / "inners" / f"inner_{int(inner_loop)}" / "docking_pool"
    return outer_dir / "docking_pool"


def _dock_manifest_path(outer_loop: int, inner_loop: Optional[int] = None) -> Path:
    return _docking_pool_dir(outer_loop, inner_loop) / "dock_manifest.json"


def _read_dock_manifest(outer_loop: int, inner_loop: Optional[int] = None) -> dict:
    manifest_path = _dock_manifest_path(outer_loop, inner_loop)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Docking manifest not found: {manifest_path}")
    with manifest_path.open("r") as f:
        return json.load(f)


def _write_dock_manifest(
    outer_loop: int, manifest: dict, inner_loop: Optional[int] = None
) -> Path:
    manifest_path = _dock_manifest_path(outer_loop, inner_loop)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp.replace(manifest_path)
    print(f"[dock-split] Wrote docking manifest: {manifest_path}")
    return manifest_path


def _split_csv_into_shards(input_csv: Path, shard_dir: Path, *, batch_rows: int) -> list[dict]:
    """Split ``input_csv`` into shard CSVs with at most ``batch_rows`` data rows."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    for stale in shard_dir.glob("shard_*.csv"):
        stale.unlink()
    for stale in shard_dir.glob("shard_*.docked.csv"):
        stale.unlink()

    if batch_rows <= 0:
        raise ValueError("docking.batch_rows must be > 0")
    shards: list[dict] = []
    with input_csv.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            return shards
        writer = None
        fout = None
        rows_in_shard = 0
        shard_index = -1
        try:
            for row in reader:
                if writer is None or rows_in_shard >= batch_rows:
                    if fout is not None:
                        fout.close()
                    shard_index += 1
                    shard_csv = shard_dir / f"shard_{shard_index:06d}.csv"
                    docked_csv = shard_dir / f"shard_{shard_index:06d}.docked.csv"
                    fout = shard_csv.open("w", newline="", encoding="utf-8")
                    writer = csv.writer(fout, lineterminator="\n")
                    writer.writerow(header)
                    rows_in_shard = 0
                    shards.append(
                        {
                            "index": shard_index,
                            "input": str(shard_csv),
                            "output": str(docked_csv),
                            "rows": 0,
                        }
                    )
                writer.writerow(row)
                rows_in_shard += 1
                shards[-1]["rows"] = rows_in_shard
        finally:
            if fout is not None:
                fout.close()
    return shards


# ---------------------------------------------------------------------------
# DOCK3 pool command
# ---------------------------------------------------------------------------


def _dock3_pool_cmd(unscored_csv: Path, out_csv: Path, work_dir: Path) -> str:
    """Build the subprocess command for either DOCK3 or Vina docking.

    When ``docking.backend`` is ``"vina"`` this emits a Vina command with
    the receptor, box, and engine settings configured in the pipeline config
    (falling back to the sEH/4JNC defaults hard-coded in
    ``dock_library_parallel``).
    """
    backend = str(_cfg_get("docking.backend", "dock3")).lower()
    local_omltk = _cfg_get(
        "docking.local_omltk",
        "/project/rrg-mailhoto/share/dockingpackages/omlab_toolkit/src",
    )
    pythonpath_parts = [str(PROJECT_ROOT / "src")]
    if local_omltk:
        pythonpath_parts.append(str(local_omltk))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env_prefix = "env PYTHONPATH=" + shlex.quote(":".join(pythonpath_parts)) + " "

    cmd = (
        env_prefix
        + "python -u -m deepdelgfn.mols.dock_library_parallel"
        + f" --unscored-csv {shlex.quote(str(unscored_csv))}"
        + f" --out-csv {shlex.quote(str(out_csv))}"
        + f" --n-proc {DOCK_JOBS}"
        + f" --backend {shlex.quote(backend)}"
        + f" --work-dir {shlex.quote(str(work_dir))}"
    )

    if backend == "vina":
        target = _vina_target()
        if target:
            cmd += f" --target {shlex.quote(target)}"
        receptor = _cfg_get("docking.receptor", None)
        if receptor:
            cmd += f" --receptor {shlex.quote(str(receptor))}"
        box = _cfg_get("docking.box", None)
        if box is not None:
            cmd += f" --box {shlex.quote(str(box))}"
        engine = _cfg_get("docking.engine", None)
        if engine:
            cmd += f" --engine {shlex.quote(str(engine))}"
        exhaustiveness = _cfg_get("docking.exhaustiveness", None)
        if exhaustiveness is not None:
            cmd += f" --exhaustiveness {int(exhaustiveness)}"
        cpu = _cfg_get("docking.cpu", None)
        if cpu is not None:
            cmd += f" --cpu {int(cpu)}"
        seed = _cfg_get("docking.seed", None)
        if seed is not None:
            cmd += f" --seed {int(seed)}"
    else:
        cmd += (
            f" --indock {shlex.quote(DOCK3_INDOCK_TEMPLATE)}"
            + f" --dockfiles {shlex.quote(DOCK3_DOCKFILES_DIR)}"
            + f" --dock64 {shlex.quote(str(_cfg_get('docking.dock64', '/project/rrg-mailhoto/share/dock64')))}"
            + f" --dock3-timeout {DOCK3_TIMEOUT}"
            + f" --ligbuild-timeout {DOCK3_LIGBUILD_TIMEOUT}"
        )

    return cmd


# ---------------------------------------------------------------------------
# Candidate metadata appending (file-locked)
# ---------------------------------------------------------------------------


def _append_docked_candidate_row(
    docked_candidates_csv: Path,
    *,
    outer_loop: int,
    row,
    cand_idx: int,
    docking_out_csv: Path,
    thr_reward: float,
) -> None:
    inner_loop = int(row["inner_loop"])
    out_row = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "run_id": RUN_ID,
        "outer_loop": int(outer_loop),
        "inner_loop": inner_loop,
        "B1_id": row["B1_id"],
        "B2_id": row["B2_id"],
        "B3_id": row["B3_id"],
        "threshold": float(_fixed_threshold()),
        "threshold_reward": float(thr_reward),
        "docking_csv": str(docking_out_csv),
    }
    for meta_col in (
        "selection_mode",
        "leader_rank",
        "leader_similarity_threshold",
        "autodock_proxy_value",
    ):
        if meta_col in row:
            out_row[meta_col] = row[meta_col]
    docked_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    lock_path = docked_candidates_csv.with_suffix(docked_candidates_csv.suffix + ".lock")
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            pd.DataFrame([out_row]).to_csv(
                docked_candidates_csv,
                mode="a",
                header=not docked_candidates_csv.exists(),
                index=False,
            )
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Docking metrics plot
# ---------------------------------------------------------------------------


def _write_docking_metrics_plot(docked_candidates_csv: Path, outer_loop: int) -> None:
    try:
        if docked_candidates_csv.exists():
            ddf = pd.read_csv(docked_candidates_csv)
            if len(ddf) > 0:
                ddf = ddf.loc[
                    pd.to_numeric(ddf.get("outer_loop"), errors="coerce") == int(outer_loop)
                ].copy()
                if len(ddf) > 0:
                    ddf["inner_loop"] = pd.to_numeric(ddf["inner_loop"], errors="coerce")
                    ddf["threshold_reward"] = pd.to_numeric(
                        ddf["threshold_reward"], errors="coerce"
                    )
                    agg = (
                        ddf.groupby("inner_loop", sort=True)["threshold_reward"]
                        .agg(avg_threshold_reward="mean", max_threshold_reward="max")
                        .reset_index()
                    )
                    agg["global_step"] = (
                        int(outer_loop) * int(CFG["num_inner_loops"]) + agg["inner_loop"].astype(int)
                    )
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    agg = agg.sort_values(by="global_step")
                    plt.figure(figsize=(6, 4))
                    plt.plot(
                        agg["global_step"],
                        agg["avg_threshold_reward"],
                        marker="o",
                        label="avg threshold_reward",
                    )
                    plt.plot(
                        agg["global_step"],
                        agg["max_threshold_reward"],
                        marker="x",
                        label="max threshold_reward",
                    )
                    plt.xlabel("inner_loop step")
                    plt.ylabel("threshold_reward")
                    plt.title(_plot_title_metadata(prefix="Docking threshold_reward over inner loops | "))
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    out_png = RUN_ROOT / "inner_loop_docking_metrics.png"
                    agg.to_csv(out_png.with_suffix(".csv"), index=False)
                    plt.tight_layout()
                    plt.savefig(out_png, dpi=200)
                    plt.close()
                    print(f"[plot] Saved inner-loop docking metrics plot to {out_png}")
    except Exception as e:
        print(f"[WARN] Failed to generate plot: {e}")


# ---------------------------------------------------------------------------
# library scoring command (generate-only)
# ---------------------------------------------------------------------------


def _score_library_generate_only_cmd(blocks: str, out_csv: Path, out_dir: Path) -> str:
    return (
        "python -u -m deepdelgfn.mols.score_library"
        + f" --bbs-combined {shlex.quote(BBS_CSV)}"
        + f" --reaction-mode {shlex.quote(_reaction_mode())}"
        + " --generate-only"
        + f" --out-csv {shlex.quote(str(out_csv))}"
        + f" --out-dir {shlex.quote(str(out_dir))}"
        + f" --blocks {shlex.quote(blocks)}"
    )


# ---------------------------------------------------------------------------
# dock_prepare (split stage)
# ---------------------------------------------------------------------------


def dock_prepare_outer_loop(outer_loop: int, inner_loop: Optional[int] = None) -> None:
    """Prepare active-learning docking shards without running DOCK3."""
    outer_dir, _, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    if inner_loop is not None:
        snapshot = _inner_docking_deepdel_path(outer_loop, int(inner_loop))
        deepdel_model_for_selection = snapshot if snapshot.exists() else deepdel_model_last
        if not snapshot.exists():
            print(
                f"[dock-split][WARN] Per-inner DeepDEL snapshot not found at {snapshot}; "
                f"falling back to {deepdel_model_last}"
            )
    else:
        deepdel_model_for_selection = deepdel_model_last
    top_df = gather_docking_candidates_by_inner_loop(
        outer_dir,
        deepdel_model_path=deepdel_model_for_selection,
        only_inner_loop=inner_loop,
    )
    pool_dir = _docking_pool_dir(outer_loop, inner_loop)
    generated_dir = pool_dir / "generated_libraries"
    shard_dir = pool_dir / "dock_shards"
    work_dir_name = str(_cfg_get("docking.backend", "dock3")).lower() + "_work"
    work_dir = pool_dir / work_dir_name
    for p in (pool_dir, generated_dir, shard_dir, work_dir):
        p.mkdir(parents=True, exist_ok=True)

    docking_batch_csv = (
        (pool_dir / "top_candidates_for_docking.csv")
        if inner_loop is not None
        else (outer_dir / "top_candidates_for_docking.csv")
    )
    docked_candidates_csv = RUN_ROOT / "docked_candidates.csv"
    docking_csv_dir = DOCKING_OUT_DIR
    docking_csv_dir.mkdir(parents=True, exist_ok=True)

    batch_rows = int(_cfg_get("docking.batch_rows", 5000))
    manifest = {
        "run_id": RUN_ID,
        "outer_loop": int(outer_loop),
        "inner_loop": None if inner_loop is None else int(inner_loop),
        "batch_rows": batch_rows,
        "pool_dir": str(pool_dir),
        "generated_dir": str(generated_dir),
        "shard_dir": str(shard_dir),
        "work_dir": str(work_dir),
        "candidate_meta_csv": str(pool_dir / "candidate_meta.csv"),
        "pooled_all_csv": str(pool_dir / "pooled_all.csv"),
        "pooled_to_dock_csv": str(pool_dir / "pooled_to_dock.csv"),
        "pooled_docked_csv": str(pool_dir / "pooled_docked_scores.csv"),
        "docked_candidates_csv": str(docked_candidates_csv),
        "n_candidates": 0,
        "n_pooled_rows": 0,
        "n_to_dock": 0,
        "n_batches": 0,
        "failed_candidate_indices": [],
        "shards": [],
    }

    if len(top_df) == 0:
        print("[dock-split] No docking candidates found; writing empty manifest.")
        _write_dock_manifest(outer_loop, manifest, inner_loop)
        return

    top_df = top_df.reset_index(drop=True).copy()
    top_df["inner_loop"] = pd.to_numeric(top_df["inner_loop"], errors="coerce").astype(int)
    batch_cols = [
        c
        for c in [
            "B1_id",
            "B2_id",
            "B3_id",
            "inner_loop",
            "selection_mode",
            "leader_rank",
            "leader_similarity_threshold",
            "autodock_proxy_value",
        ]
        if c in top_df.columns
    ]
    top_df[batch_cols].to_csv(docking_batch_csv, index=False)

    candidate_rows = []
    pooled_frames = []
    failed_candidate_indices: set[int] = set()
    for cand_idx, row in top_df.iterrows():
        b1, b2, b3, inner_loop_val = (
            row["B1_id"],
            row["B2_id"],
            row["B3_id"],
            int(row["inner_loop"]),
        )
        docking_out_csv = (
            docking_csv_dir / f"{RUN_ID}_outer{outer_loop}_inner{inner_loop_val}_{cand_idx}.csv"
        )
        candidate_row = row.to_dict()
        candidate_row[_AL_CANDIDATE_COL] = int(cand_idx)
        candidate_row[_AL_DOCKING_CSV_COL] = str(docking_out_csv)
        candidate_rows.append(candidate_row)
        if docking_out_csv.exists():
            docking_out_csv.unlink()

        unscored_csv = generated_dir / f"candidate_{cand_idx}_unscored.csv"
        if unscored_csv.exists():
            unscored_csv.unlink()
        blocks = str(b1) + "," + str(b2) + "," + str(b3)
        if (
            run_cmd(_score_library_generate_only_cmd(blocks, unscored_csv, generated_dir))
            != 0
        ):
            failed_candidate_indices.add(int(cand_idx))
            print(
                f"[dock-split] Failed generating unscored library for candidate {cand_idx}; "
                "marking reward as NaN."
            )
            continue
        try:
            lib_df = pd.read_csv(unscored_csv)
        except Exception as e:
            failed_candidate_indices.add(int(cand_idx))
            print(f"[dock-split] Failed reading generated library {unscored_csv}: {e}")
            continue
        lib_df[_AL_CANDIDATE_COL] = int(cand_idx)
        lib_df[_AL_DOCKING_CSV_COL] = str(docking_out_csv)
        pooled_frames.append(lib_df)

    candidate_meta_csv = Path(manifest["candidate_meta_csv"])
    pd.DataFrame(candidate_rows).to_csv(candidate_meta_csv, index=False)
    manifest["n_candidates"] = len(candidate_rows)
    manifest["failed_candidate_indices"] = sorted(failed_candidate_indices)

    if not pooled_frames:
        print(
            "[dock-split] No pooled molecules were generated; finalizer will record "
            "NaN rewards where needed."
        )
        _write_dock_manifest(outer_loop, manifest, inner_loop)
        return

    pooled_df = pd.concat(pooled_frames, ignore_index=True)
    pooled_df["docking_score"] = np.nan
    backend_label = str(_cfg_get("docking.backend", "dock3")).lower()
    pooled_df[_AL_SCORE_SOURCE_COL] = backend_label
    pooled_df["__canonical_smiles"] = (
        pooled_df["smiles"].map(_canonical_smiles) if "smiles" in pooled_df.columns else ""
    )
    empty_smiles_mask = ~pooled_df["__canonical_smiles"].astype(str).str.strip().astype(bool)
    if empty_smiles_mask.any():
        pooled_df.loc[empty_smiles_mask, "docking_score"] = 0.0
        pooled_df.loc[empty_smiles_mask, _AL_SCORE_SOURCE_COL] = "empty_smiles"
        print(
            f"[dock-split] Assigned 0.0 to {int(empty_smiles_mask.sum())} generated "
            "row(s) with empty/invalid SMILES."
        )

    previous_scores = _load_previous_docking_scores()
    if previous_scores:
        hit_mask = pooled_df["__canonical_smiles"].map(
            lambda key: bool(key) and key in previous_scores
        )
        if hit_mask.any():
            pooled_df.loc[hit_mask, "docking_score"] = (
                pooled_df.loc[hit_mask, "__canonical_smiles"].map(previous_scores).astype(float)
            )
            pooled_df.loc[hit_mask, _AL_SCORE_SOURCE_COL] = "reused"
        print(
            f"[dock-split] Reused {int(hit_mask.sum())}/{len(pooled_df)} molecule score(s) "
            "from prior autodock inputs."
        )

    pooled_all_csv = Path(manifest["pooled_all_csv"])
    pooled_df.to_csv(pooled_all_csv, index=False)
    manifest["n_pooled_rows"] = int(len(pooled_df))

    miss_mask = pooled_df["docking_score"].isna()
    n_to_dock = int(miss_mask.sum())
    manifest["n_to_dock"] = n_to_dock
    if n_to_dock > 0:
        miss_df = pooled_df.loc[miss_mask].copy().reset_index(drop=True)
        pooled_to_dock_csv = Path(manifest["pooled_to_dock_csv"])
        miss_df.drop(
            columns=["docking_score", _AL_SCORE_SOURCE_COL, "__canonical_smiles"],
            errors="ignore",
        ).to_csv(pooled_to_dock_csv, index=False)
        shards = _split_csv_into_shards(pooled_to_dock_csv, shard_dir, batch_rows=batch_rows)
        manifest["shards"] = shards
        manifest["n_batches"] = len(shards)
        print(
            f"[dock-split] Prepared {n_to_dock} molecule(s) for {backend_label} as {len(shards)} "
            f"shard job(s) of up to {batch_rows} rows."
        )
    else:
        print(
            "[dock-split] All pooled molecules were found in prior autodock inputs; "
            "no new docking calls needed."
        )

    _write_dock_manifest(outer_loop, manifest, inner_loop)


# ---------------------------------------------------------------------------
# dock_batch (array task)
# ---------------------------------------------------------------------------


def dock_batch_outer_loop(outer_loop: int, inner_loop: Optional[int] = None) -> None:
    """Run one prepared docking shard. Intended for Slurm array tasks."""
    idx_raw = os.environ.get("DOCK_BATCH_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx_raw in (None, ""):
        raise RuntimeError(
            "DOCK_BATCH_INDEX or SLURM_ARRAY_TASK_ID is required for STAGE=dock_batch"
        )
    batch_idx = int(idx_raw)
    manifest = _read_dock_manifest(outer_loop, inner_loop)
    shards = manifest.get("shards", []) or []
    if batch_idx < 0 or batch_idx >= len(shards):
        print(
            f"[dock-batch] Batch index {batch_idx} is outside manifest shard range "
            f"0..{len(shards) - 1}; exiting."
        )
        return

    shard = shards[batch_idx]
    shard_csv = Path(shard["input"])
    docked_csv = Path(shard["output"])
    expected_rows = int(shard["rows"])
    work_dir_default = _docking_pool_dir(outer_loop, inner_loop) / (str(_cfg_get("docking.backend", "dock3")).lower() + "_work")
    work_dir = (
        Path(
            str(
                manifest.get(
                    "work_dir",
                    work_dir_default,
                )
            )
        )
        / f"shard_{batch_idx:06d}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = _csv_data_row_count(docked_csv)
    if existing_rows == expected_rows:
        print(
            f"[dock-batch] Reusing complete shard output {docked_csv} "
            f"({existing_rows}/{expected_rows} rows)."
        )
        return
    if existing_rows > expected_rows:
        raise RuntimeError(
            f"Shard output has too many rows: expected {expected_rows}, "
            f"got {existing_rows} at {docked_csv}"
        )
    if existing_rows > 0:
        remaining_csv = docked_csv.with_suffix(".remaining_input.csv")
        remaining_docked_csv = docked_csv.with_suffix(".remaining_docked.csv")
        remaining_count = _write_csv_remaining_after_prefix(
            shard_csv, remaining_csv, existing_rows
        )
        if remaining_docked_csv.exists():
            remaining_docked_csv.unlink()
        print(
            f"[dock-batch] Resuming shard {batch_idx}: {existing_rows}/{expected_rows} rows "
            f"already written; docking {remaining_count} remaining."
        )
        if run_cmd(_dock3_pool_cmd(remaining_csv, remaining_docked_csv, work_dir)) != 0:
            raise RuntimeError(f"Docking shard resume failed for batch {batch_idx}")
        done_rows = _csv_data_row_count(remaining_docked_csv)
        if done_rows != remaining_count:
            raise RuntimeError(
                f"Shard resume row count mismatch for batch {batch_idx}: "
                f"expected {remaining_count}, got {done_rows}"
            )
        _concat_csvs_with_matching_headers([docked_csv, remaining_docked_csv], docked_csv)
        print(f"[dock-batch] Combined resumed shard output: {docked_csv}")
        return

    if docked_csv.exists():
        docked_csv.unlink()
    print(
        f"[dock-batch] Docking shard {batch_idx}/{len(shards) - 1}: "
        f"{expected_rows} molecule(s) with {DOCK_JOBS} worker(s)."
    )
    if run_cmd(_dock3_pool_cmd(shard_csv, docked_csv, work_dir)) != 0:
        raise RuntimeError(f"Docking shard failed for batch {batch_idx}")
    done_rows = _csv_data_row_count(docked_csv)
    if done_rows != expected_rows:
        raise RuntimeError(
            f"Shard row count mismatch for batch {batch_idx}: "
            f"expected {expected_rows}, got {done_rows}"
        )


# ---------------------------------------------------------------------------
# dock_finalize
# ---------------------------------------------------------------------------


def dock_finalize_outer_loop(outer_loop: int, inner_loop: Optional[int] = None) -> None:
    """Combine split docking shards and write per-candidate active-learning outputs."""
    manifest = _read_dock_manifest(outer_loop, inner_loop)
    candidate_meta_csv = Path(manifest["candidate_meta_csv"])
    docked_candidates_csv = Path(manifest["docked_candidates_csv"])
    failed_candidate_indices = {
        int(x) for x in manifest.get("failed_candidate_indices", [])
    }
    if not candidate_meta_csv.exists():
        print("[dock-finalize] No candidate metadata found; nothing to finalize.")
        return
    candidate_meta = pd.read_csv(candidate_meta_csv)

    pooled_all_csv = Path(manifest["pooled_all_csv"])
    if pooled_all_csv.exists() and _csv_data_row_count(pooled_all_csv) > 0:
        pooled_df = pd.read_csv(pooled_all_csv)
        miss_mask = pooled_df["docking_score"].isna()
        expected_to_dock = int(miss_mask.sum())
        shards = manifest.get("shards", []) or []
        pooled_docked_csv = Path(manifest["pooled_docked_csv"])
        if expected_to_dock > 0:
            shard_outputs = []
            for shard in shards:
                output = Path(shard["output"])
                expected_rows = int(shard["rows"])
                got_rows = _csv_data_row_count(output)
                if got_rows != expected_rows:
                    raise RuntimeError(
                        f"Incomplete shard output {output}: "
                        f"expected {expected_rows}, got {got_rows}"
                    )
                shard_outputs.append(output)
            combined_rows = _concat_csvs_with_matching_headers(shard_outputs, pooled_docked_csv)
            if combined_rows != expected_to_dock:
                raise RuntimeError(
                    f"Combined pooled docking row count mismatch: "
                    f"expected {expected_to_dock}, got {combined_rows}"
                )
            docked_df = pd.read_csv(pooled_docked_csv)
            if "docking_score" not in docked_df.columns:
                raise RuntimeError(
                    f"Pooled docking output missing docking_score column: {pooled_docked_csv}"
                )
            backend_label = str(_cfg_get("docking.backend", "dock3")).lower()
            pooled_df.loc[miss_mask, "docking_score"] = pd.to_numeric(
                docked_df["docking_score"], errors="coerce"
            ).to_numpy(dtype=float)
            pooled_df.loc[miss_mask, _AL_SCORE_SOURCE_COL] = backend_label
            pooled_df.to_csv(pooled_all_csv, index=False)
            print(
                f"[dock-finalize] Combined {len(shard_outputs)} shard output(s) into "
                f"{pooled_docked_csv}"
            )
        else:
            print(
                "[dock-finalize] No new docking scores were required; "
                "using reused/empty scores only."
            )

        final_drop_cols = [
            _AL_CANDIDATE_COL,
            _AL_DOCKING_CSV_COL,
            _AL_SCORE_SOURCE_COL,
            "__canonical_smiles",
        ]
        for _, row in candidate_meta.iterrows():
            cand_idx = int(row[_AL_CANDIDATE_COL])
            docking_out_csv = Path(str(row[_AL_DOCKING_CSV_COL]))
            if cand_idx in failed_candidate_indices:
                thr_reward = float("nan")
                _append_docked_candidate_row(
                    docked_candidates_csv,
                    outer_loop=outer_loop,
                    row=row,
                    cand_idx=cand_idx,
                    docking_out_csv=docking_out_csv,
                    thr_reward=thr_reward,
                )
                continue
            cand_df = pooled_df.loc[
                pd.to_numeric(pooled_df[_AL_CANDIDATE_COL], errors="coerce") == cand_idx
            ].copy()
            if cand_df.empty:
                thr_reward = float("nan")
            else:
                cand_out = cand_df.drop(columns=final_drop_cols, errors="ignore")
                cand_out.to_csv(docking_out_csv, index=False)
                try:
                    thr_reward = compute_threshold_reward_from_docking_csv(
                        docking_out_csv,
                            threshold=float(_fixed_threshold()),
                        alpha=float(CFG.get("threshold_alpha", 1.0)),
                    )
                except Exception:
                    thr_reward = float("nan")
            _append_docked_candidate_row(
                docked_candidates_csv,
                outer_loop=outer_loop,
                row=row,
                cand_idx=cand_idx,
                docking_out_csv=docking_out_csv,
                thr_reward=thr_reward,
            )
    else:
        for _, row in candidate_meta.iterrows():
            cand_idx = int(row[_AL_CANDIDATE_COL])
            _append_docked_candidate_row(
                docked_candidates_csv,
                outer_loop=outer_loop,
                row=row,
                cand_idx=cand_idx,
                docking_out_csv=Path(str(row[_AL_DOCKING_CSV_COL])),
                thr_reward=float("nan"),
            )

    _write_docking_metrics_plot(docked_candidates_csv, outer_loop)


# ---------------------------------------------------------------------------
# Slurm submission helpers
# ---------------------------------------------------------------------------


def _slurm_flag_parts_from_cfg(defaults: dict) -> list[str]:
    """Translate a small CFG['slurm'][stage] dict into sbatch CLI flags."""
    parts: list[str] = []
    if defaults.get("job-name"):
        parts.extend(["--job-name", str(defaults["job-name"])])
    if defaults.get("account"):
        parts.extend(["--account", str(defaults["account"])])
    if defaults.get("time"):
        parts.extend(["--time", str(defaults["time"])])
    if defaults.get("ntasks_per_node"):
        parts.extend(["--ntasks-per-node", str(int(defaults["ntasks_per_node"]))])
    if defaults.get("cpus"):
        parts.extend(["--cpus-per-task", str(int(defaults["cpus"]))])
    if defaults.get("mem") is not None and str(defaults.get("mem")) != "":
        parts.extend(["--mem", str(defaults["mem"])])
    if defaults.get("gpus_per_node"):
        parts.append(f"--gpus-per-node={defaults['gpus_per_node']}")
    return parts


def _submit_sbatch(cmd_parts: list[str], *, wait: bool = False) -> str:
    printable = " ".join(shlex.quote(str(x)) for x in cmd_parts)
    print(f"[dock-launch] Submitting: {printable}")
    try:
        result = subprocess.run(cmd_parts, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        if stderr:
            print(f"[dock-launch][sbatch stderr] {stderr}")
        raise
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stderr:
        print(f"[dock-launch][sbatch stderr] {stderr}")
    job_id = stdout.splitlines()[-1].split(";")[0] if stdout else ""
    if wait:
        print(f"[dock-launch] Waited for submitted job; sbatch return code={result.returncode}")
    else:
        print(f"[dock-launch] Submitted job_id={job_id}")
    return job_id


# ---------------------------------------------------------------------------
# dock_launch
# ---------------------------------------------------------------------------


def dock_launch_outer_loop(outer_loop: int, inner_loop: Optional[int] = None) -> None:
    """Submit dynamic docking shard array and finalizer based on the manifest.

    This stage intentionally runs after ``dock_prepare`` because only then do we
    know how many molecules still require DOCK3 and therefore how many 5k-row
    shard jobs are needed.  It waits for the finalizer by default so the Slurm
    dependency from one outer loop to the next can target this launcher job.
    """
    manifest = _read_dock_manifest(outer_loop, inner_loop)
    n_batches = int(manifest.get("n_batches", 0))
    if n_batches <= 0:
        print(
            "[dock-launch] Manifest has no docking shards; finalizing directly in launcher job."
        )
        dock_finalize_outer_loop(outer_loop, inner_loop)
        return

    slurm_cfg = CFG.get("slurm", {}) or {}
    dock_defaults = dict(slurm_cfg.get("dock", {}) or {})
    finalize_defaults = dict(slurm_cfg.get("dock_finalize", {}) or {})
    if not finalize_defaults:
        finalize_defaults = {
            "job-name": "deepdel-dock-fin",
            "time": "1:00:00",
            "ntasks_per_node": 1,
            "cpus": 1,
            "mem": "8G",
            "gpus_per_node": None,
            "account": dock_defaults.get("account"),
        }

    job_script = str(PROJECT_ROOT / "jobs" / "active_learning_job.sh")
    slurm_out_dir = RUN_ROOT / "slurm"
    slurm_out_dir.mkdir(parents=True, exist_ok=True)

    batch_export = (
        f"ALL,DEL_GFN_RUN_ID={RUN_ID},RUN_ID={RUN_ID},OUTER_LOOP={outer_loop},"
        f"STAGE=dock_batch,DEL_GFN_CONFIG={CFG_PATH}"
    )
    if inner_loop is not None:
        batch_export += f",INNER_LOOP={int(inner_loop)}"
    batch_cmd = [
        "sbatch",
        "--parsable",
        f"--array=0-{n_batches - 1}",
        "--export",
        batch_export,
        "--output",
        str(
            slurm_out_dir
            / (
                f"dock_outer{outer_loop}_inner{int(inner_loop)}_%A_%a.out"
                if inner_loop is not None
                else f"dock_outer{outer_loop}_%A_%a.out"
            )
        ),
        *_slurm_flag_parts_from_cfg(dock_defaults),
        job_script,
    ]
    array_job_id = _submit_sbatch(batch_cmd)
    manifest["array_job_id"] = array_job_id

    finalize_export = (
        f"ALL,DEL_GFN_RUN_ID={RUN_ID},RUN_ID={RUN_ID},OUTER_LOOP={outer_loop},"
        f"STAGE=dock_finalize,DEL_GFN_CONFIG={CFG_PATH}"
    )
    if inner_loop is not None:
        finalize_export += f",INNER_LOOP={int(inner_loop)}"
    finalize_cmd = [
        "sbatch",
        "--parsable",
        "--wait"
        if bool(_cfg_get("docking.launch_wait_for_finalize", True))
        else "--parsable",
        "--export",
        finalize_export,
        "--dependency",
        f"afterok:{array_job_id}",
        "--output",
        str(
            slurm_out_dir
            / (
                f"dock_finalize_outer{outer_loop}_inner{int(inner_loop)}_%j.out"
                if inner_loop is not None
                else f"dock_finalize_outer{outer_loop}_%j.out"
            )
        ),
        *_slurm_flag_parts_from_cfg(finalize_defaults),
        job_script,
    ]
    # Avoid passing --parsable twice when not waiting.
    if finalize_cmd[2] == "--parsable":
        del finalize_cmd[2]
    finalizer_job_id = _submit_sbatch(
        finalize_cmd, wait=bool(_cfg_get("docking.launch_wait_for_finalize", True))
    )
    manifest["finalizer_job_id"] = finalizer_job_id
    _write_dock_manifest(outer_loop, manifest, inner_loop)


# ---------------------------------------------------------------------------
# dock_outer_loop (legacy monolithic dock)
# ---------------------------------------------------------------------------


def dock_outer_loop(outer_loop: int, inner_loop: Optional[int] = None):
    outer_dir, _, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    if inner_loop is not None:
        snapshot = _inner_docking_deepdel_path(outer_loop, int(inner_loop))
        deepdel_model_for_selection = snapshot if snapshot.exists() else deepdel_model_last
        if not snapshot.exists():
            print(
                f"[dock][WARN] Per-inner DeepDEL snapshot not found at {snapshot}; "
                f"falling back to {deepdel_model_last}"
            )
    else:
        deepdel_model_for_selection = deepdel_model_last
    top_df = gather_docking_candidates_by_inner_loop(
        outer_dir,
        deepdel_model_path=deepdel_model_for_selection,
        only_inner_loop=inner_loop,
    )
    if len(top_df) == 0:
        return
    pool_dir = (
        _docking_pool_dir(outer_loop, inner_loop)
        if inner_loop is not None
        else outer_dir / "docking_pool"
    )
    docking_batch_csv = (
        (pool_dir / "top_candidates_for_docking.csv")
        if inner_loop is not None
        else (outer_dir / "top_candidates_for_docking.csv")
    )
    docking_batch_csv.parent.mkdir(parents=True, exist_ok=True)
    top_df["inner_loop"] = pd.to_numeric(top_df["inner_loop"], errors="coerce").astype(int)
    batch_cols = [
        c
        for c in [
            "B1_id",
            "B2_id",
            "B3_id",
            "inner_loop",
            "selection_mode",
            "leader_rank",
            "leader_similarity_threshold",
            "autodock_proxy_value",
        ]
        if c in top_df.columns
    ]
    top_df[batch_cols].to_csv(docking_batch_csv, index=False)

    docked_candidates_csv = RUN_ROOT / "docked_candidates.csv"
    docking_csv_dir = DOCKING_OUT_DIR
    docking_csv_dir.mkdir(parents=True, exist_ok=True)

    backend = _cfg_get("docking.backend", "dock3")
    use_pool = (
        bool(_cfg_get("docking.pool_active_learning_libraries", True)) and backend == "dock3"
    )
    if use_pool:
        generated_dir = pool_dir / "generated_libraries"
        work_dir_name = str(backend).lower() + "_work"
        work_dir = pool_dir / work_dir_name
        pool_dir.mkdir(parents=True, exist_ok=True)
        generated_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        candidate_meta = {}
        pooled_frames = []
        failed_candidate_indices = set()
        for cand_idx, row in top_df.reset_index(drop=True).iterrows():
            b1, b2, b3, inner_loop_val = (
                row["B1_id"],
                row["B2_id"],
                row["B3_id"],
                int(row["inner_loop"]),
            )
            docking_out_csv = (
                docking_csv_dir
                / f"{RUN_ID}_outer{outer_loop}_inner{inner_loop_val}_{cand_idx}.csv"
            )
            if docking_out_csv.exists():
                docking_out_csv.unlink()
            candidate_meta[int(cand_idx)] = {"row": row, "docking_out_csv": docking_out_csv}

            unscored_csv = generated_dir / f"candidate_{cand_idx}_unscored.csv"
            if unscored_csv.exists():
                unscored_csv.unlink()
            blocks = str(b1) + "," + str(b2) + "," + str(b3)
            if (
                run_cmd(
                    _score_library_generate_only_cmd(blocks, unscored_csv, generated_dir)
                )
                != 0
            ):
                failed_candidate_indices.add(int(cand_idx))
                print(
                    f"[dock-pool] Failed generating unscored library for candidate {cand_idx}; "
                    "marking reward as NaN."
                )
                continue
            try:
                lib_df = pd.read_csv(unscored_csv)
            except Exception as e:
                failed_candidate_indices.add(int(cand_idx))
                print(f"[dock-pool] Failed reading generated library {unscored_csv}: {e}")
                continue
            lib_df[_AL_CANDIDATE_COL] = int(cand_idx)
            lib_df[_AL_DOCKING_CSV_COL] = str(docking_out_csv)
            pooled_frames.append(lib_df)

        if pooled_frames:
            pooled_df = pd.concat(pooled_frames, ignore_index=True)
            pooled_df["docking_score"] = np.nan
            pooled_df[_AL_SCORE_SOURCE_COL] = backend
            pooled_df["__canonical_smiles"] = (
                pooled_df["smiles"].map(_canonical_smiles)
                if "smiles" in pooled_df.columns
                else ""
            )
            empty_smiles_mask = ~pooled_df["__canonical_smiles"].astype(str).str.strip().astype(
                bool
            )
            if empty_smiles_mask.any():
                pooled_df.loc[empty_smiles_mask, "docking_score"] = 0.0
                pooled_df.loc[empty_smiles_mask, _AL_SCORE_SOURCE_COL] = "empty_smiles"
                print(
                    f"[dock-pool] Assigned 0.0 to {int(empty_smiles_mask.sum())} generated "
                    "row(s) with empty/invalid SMILES."
                )

            previous_scores = _load_previous_docking_scores()
            if previous_scores:
                hit_mask = pooled_df["__canonical_smiles"].map(
                    lambda key: bool(key) and key in previous_scores
                )
                if hit_mask.any():
                    pooled_df.loc[hit_mask, "docking_score"] = (
                        pooled_df.loc[hit_mask, "__canonical_smiles"]
                        .map(previous_scores)
                        .astype(float)
                    )
                    pooled_df.loc[hit_mask, _AL_SCORE_SOURCE_COL] = "reused"
                print(
                    f"[dock-pool] Reused {int(hit_mask.sum())}/{len(pooled_df)} molecule "
                    "score(s) from prior autodock inputs."
                )

            miss_mask = pooled_df["docking_score"].isna()
            pooled_to_dock_csv = pool_dir / "pooled_to_dock.csv"
            pooled_docked_csv = pool_dir / "pooled_docked_scores.csv"
            if miss_mask.any():
                miss_df = pooled_df.loc[miss_mask].copy().reset_index(drop=True)
                miss_df.drop(
                    columns=["docking_score", _AL_SCORE_SOURCE_COL, "__canonical_smiles"],
                    errors="ignore",
                ).to_csv(pooled_to_dock_csv, index=False)
                expected_docked_rows = int(miss_mask.sum())
                existing_docked_rows = _csv_data_row_count(pooled_docked_csv)

                if existing_docked_rows == expected_docked_rows:
                    print(
                        f"[dock-pool] Reusing complete pooled DOCK3 output with "
                        f"{existing_docked_rows}/{expected_docked_rows} row(s): "
                        f"{pooled_docked_csv}"
                    )
                elif existing_docked_rows > expected_docked_rows:
                    raise RuntimeError(
                        f"Existing pooled docking output has too many rows: "
                        f"expected {expected_docked_rows}, got {existing_docked_rows} "
                        f"at {pooled_docked_csv}. Move it aside before resuming."
                    )
                elif existing_docked_rows > 0:
                    remaining_csv = pool_dir / "pooled_to_dock.remaining.csv"
                    remaining_docked_csv = pool_dir / "pooled_docked_scores.remaining.csv"
                    remaining_count = _write_csv_remaining_after_prefix(
                        pooled_to_dock_csv,
                        remaining_csv,
                        existing_docked_rows,
                    )
                    if remaining_docked_csv.exists():
                        remaining_docked_csv.unlink()
                    backup_csv = pool_dir / "pooled_docked_scores.partial_before_resume.csv"
                    if not backup_csv.exists():
                        shutil.copy2(pooled_docked_csv, backup_csv)
                        print(f"[dock-pool] Backed up partial pooled output to {backup_csv}")
                    print(
                        f"[dock-pool] Resuming pooled DOCK3 output: "
                        f"{existing_docked_rows}/{expected_docked_rows} row(s) already written; "
                        f"docking {remaining_count} remaining molecule(s) with {DOCK_JOBS} worker(s)."
                    )
                    if (
                        run_cmd(
                            _dock3_pool_cmd(
                                remaining_csv, remaining_docked_csv, work_dir
                            )
                        )
                        != 0
                    ):
                        raise RuntimeError(
                            "Pooled DOCK3 active-learning docking resume failed."
                        )
                    remaining_done = _csv_data_row_count(remaining_docked_csv)
                    if remaining_done != remaining_count:
                        raise RuntimeError(
                            f"Pooled docking resume row count mismatch: "
                            f"expected {remaining_count}, got {remaining_done} "
                            f"at {remaining_docked_csv}"
                        )
                    combined_rows = _concat_csvs_with_matching_headers(
                        [pooled_docked_csv, remaining_docked_csv],
                        pooled_docked_csv,
                    )
                    print(
                        f"[dock-pool] Combined resumed pooled output: "
                        f"{combined_rows} row(s) -> {pooled_docked_csv}"
                    )
                else:
                    print(
                        f"[dock-pool] Docking {len(miss_df)} pooled molecule(s) across all "
                        f"selected libraries with {DOCK_JOBS} worker(s)."
                    )
                    if (
                        run_cmd(
                            _dock3_pool_cmd(
                                pooled_to_dock_csv, pooled_docked_csv, work_dir
                            )
                        )
                        != 0
                    ):
                        raise RuntimeError("Pooled DOCK3 active-learning docking failed.")
                docked_df = pd.read_csv(pooled_docked_csv)
                if "docking_score" not in docked_df.columns:
                    raise RuntimeError(
                        f"Pooled docking output missing docking_score column: "
                        f"{pooled_docked_csv}"
                    )
                if len(docked_df) != int(miss_mask.sum()):
                    raise RuntimeError(
                        f"Pooled docking output row count mismatch: "
                        f"expected {int(miss_mask.sum())}, got {len(docked_df)}"
                    )
                pooled_df.loc[miss_mask, "docking_score"] = pd.to_numeric(
                    docked_df["docking_score"], errors="coerce"
                ).to_numpy(dtype=float)
                pooled_df.loc[miss_mask, _AL_SCORE_SOURCE_COL] = backend
            else:
                print(
                    "[dock-pool] All pooled molecules were found in prior autodock inputs; "
                    "no new DOCK3 calls needed."
                )

            final_drop_cols = [
                _AL_CANDIDATE_COL,
                _AL_DOCKING_CSV_COL,
                _AL_SCORE_SOURCE_COL,
                "__canonical_smiles",
            ]
            for cand_idx, meta in candidate_meta.items():
                row = meta["row"]
                docking_out_csv = meta["docking_out_csv"]
                if cand_idx in failed_candidate_indices:
                    thr_reward = float("nan")
                    _append_docked_candidate_row(
                        docked_candidates_csv,
                        outer_loop=outer_loop,
                        row=row,
                        cand_idx=cand_idx,
                        docking_out_csv=docking_out_csv,
                        thr_reward=thr_reward,
                    )
                    continue
                cand_df = pooled_df.loc[
                    pooled_df[_AL_CANDIDATE_COL] == int(cand_idx)
                ].copy()
                if cand_df.empty:
                    thr_reward = float("nan")
                else:
                    cand_out = cand_df.drop(columns=final_drop_cols, errors="ignore")
                    cand_out.to_csv(docking_out_csv, index=False)
                    try:
                        thr_reward = compute_threshold_reward_from_docking_csv(
                            docking_out_csv,
                            threshold=float(_fixed_threshold()),
                            alpha=float(CFG.get("threshold_alpha", 1.0)),
                        )
                    except Exception:
                        thr_reward = float("nan")
                _append_docked_candidate_row(
                    docked_candidates_csv,
                    outer_loop=outer_loop,
                    row=row,
                    cand_idx=cand_idx,
                    docking_out_csv=docking_out_csv,
                    thr_reward=thr_reward,
                )
        else:
            for cand_idx, meta in candidate_meta.items():
                _append_docked_candidate_row(
                    docked_candidates_csv,
                    outer_loop=outer_loop,
                    row=meta["row"],
                    cand_idx=cand_idx,
                    docking_out_csv=meta["docking_out_csv"],
                    thr_reward=float("nan"),
                )

        _write_docking_metrics_plot(docked_candidates_csv, outer_loop)
        return

    if bool(_cfg_get("docking.pool_active_learning_libraries", True)) and backend != "dock3":
        print(
            "[dock-pool] Pooled active-learning docking currently supports dock3 only; "
            f"falling back to per-library backend={backend}."
        )

    for cand_idx, row in top_df.reset_index(drop=True).iterrows():
        b1, b2, b3, inner_loop_val = (
            row["B1_id"],
            row["B2_id"],
            row["B3_id"],
            int(row["inner_loop"]),
        )
        docking_out_csv = (
            docking_csv_dir
            / f"{RUN_ID}_outer{outer_loop}_inner{inner_loop_val}_{cand_idx}.csv"
        )
        if docking_out_csv.exists():
            docking_out_csv.unlink()
        # The candidate IDs (b1/b2/b3) come from topm_actual_scores.csv, which is
        # produced against the *combined* BBs CSV (globally unique IDs across all
        # three pools). Pass --bbs-combined so score_library uses the matching ID
        # space; it will internally split the CSV by `pool` for slot-specific
        # chemistry while sharing a single global id->SMILES map for lookups.
        cmd = (
            "python -u -m deepdelgfn.mols.score_library"
            + f" --bbs-combined {shlex.quote(BBS_CSV)}"
            + f" --reaction-mode {shlex.quote(_reaction_mode())}"
            + f" --docking-backend {backend}"
            # DOCK3 backend requires the INDOCK template + dockfiles directory.
            + (
                f" --indock-template {shlex.quote(DOCK3_INDOCK_TEMPLATE)}"
                f" --dockfiles {shlex.quote(DOCK3_DOCKFILES_DIR)}"
                f" --dock3-timeout {DOCK3_TIMEOUT}"
                f" --ligbuild-timeout {DOCK3_LIGBUILD_TIMEOUT}"
                if backend == "dock3"
                else ""
            )
            + f" --out-csv {shlex.quote(str(docking_out_csv))}"
            + f" --out-dir {shlex.quote(str(docking_csv_dir))}"
            + f" --blocks {shlex.quote(str(b1) + ',' + str(b2) + ',' + str(b3))}"
            + f" --cpu {DOCK_CPU} --jobs {DOCK_JOBS}"
        )
        if backend == "vina":
            target = _vina_target()
            if target:
                cmd += f" --target {shlex.quote(target)}"
        if run_cmd(cmd) == 0:
            try:
                thr_reward = compute_threshold_reward_from_docking_csv(
                    docking_out_csv,
                            threshold=float(_fixed_threshold()),
                    alpha=float(CFG.get("threshold_alpha", 1.0)),
                )
            except Exception:
                thr_reward = float("nan")
        else:
            thr_reward = float("nan")
        _append_docked_candidate_row(
            docked_candidates_csv,
            outer_loop=outer_loop,
            row=row,
            cand_idx=cand_idx,
            docking_out_csv=docking_out_csv,
            thr_reward=thr_reward,
        )

    _write_docking_metrics_plot(docked_candidates_csv, outer_loop)