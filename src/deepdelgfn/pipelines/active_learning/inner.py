#!/usr/bin/env python3
"""Inner-loop stage functions for the active learning pipeline.

This module was extracted from active_learning_stage.py.  It covers the
``inner_train``, ``eval_topm_batch``, ``eval_topm_finalize``, and
``deepdel_update`` stages, together with GFN command builders, evaluation
statistics persistence and plotting.
"""

import csv
import json
import os
import shlex
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    BBS_CSV,
    BBS_FLAGS,
    CFG,
    CPUS,
    MODEL_ROOT,
    PROJECT_ROOT,
    RUN_ID,
    RUN_ROOT,
    _autodock_model_path,
    _cfg_get,
    _concat_csvs_with_matching_headers,
    _csv_data_row_count,
    _deepdel_eval_dataset_flag,
    _fourier_flags,
    _gfn_reward_source,
    _outer_deepdel_paths,
    _outer_threshold,
    _plot_title_metadata,
    _reaction_mode,
    _reward_weight_flags,
    _threshold_spec,
    _threshold_interval,
    _threshold_weight_flags,
    _topm_threshold,
    run_cmd,
)

from .val_mse_tracker import record_deepdel_val_mse


# ---------------------------------------------------------------------------
# Path / context helpers
# ---------------------------------------------------------------------------


def _inner_dir(outer_loop: int, inner_loop: int) -> Path:
    outer_dir, _, _ = _outer_deepdel_paths(outer_loop)
    inner_dir = outer_dir / "inners" / f"inner_{int(inner_loop)}"
    inner_dir.mkdir(parents=True, exist_ok=True)
    return inner_dir


# ---------------------------------------------------------------------------
# GFN command builders
# ---------------------------------------------------------------------------


def _forbidden_flags() -> str:
    forbidden_cfg = _cfg_get("gfn.forbidden", {}) or {}
    flags = []
    for cycle in (1, 2, 3):
        key = f"forbidden_bb{cycle}"
        raw = forbidden_cfg.get(key)
        if not raw:
            continue
        text = str(raw).strip()
        if text:
            flags.append(f" --{key.replace('_', '-')} {shlex.quote(text)}")
    return "".join(flags)


def _gfn_arch_and_phase_flags() -> str:
    phase_enabled = bool(_cfg_get("gfn.phase_regularization.enabled", False))
    phase_lambda = float(_cfg_get("gfn.phase_regularization.lambda", 0.0))
    if phase_lambda < 0:
        raise ValueError("CFG['gfn']['phase_regularization']['lambda'] must be nonnegative")
    phi_train_mode = str(_cfg_get("gfn.model.phi_train_mode", "frozen_table"))
    if phi_train_mode not in {"frozen_table", "trainable_table", "trainable_network"}:
        raise ValueError(
            "CFG['gfn']['model']['phi_train_mode'] must be one of "
            "'frozen_table', 'trainable_table', or 'trainable_network'"
        )
    phi_lr = _cfg_get("gfn.train.phi_lr", None)
    phi_weight_decay = _cfg_get("gfn.train.phi_weight_decay", None)
    return (
        f" --joint-dim {int(_cfg_get('gfn.model.joint_dim', 512))}"
        f" --state-dim {int(_cfg_get('gfn.model.state_dim', 1024))}"
        f" --action-dim {int(_cfg_get('gfn.model.action_dim', 512))}"
        f" --action-repr {shlex.quote(str(_cfg_get('gfn.model.action_repr', 'ecfp')))}"
        f" --phi-train-mode {shlex.quote(phi_train_mode)}"
        f" --bb-fp-bits {int(_cfg_get('gfn.model.bb_fp_bits', 2048))}"
        f" --bb-fp-radius {int(_cfg_get('gfn.model.bb_fp_radius', 2))}"
        + (f" --phi-lr {float(phi_lr)}" if phi_lr is not None else "")
        + (f" --phi-weight-decay {float(phi_weight_decay)}" if phi_weight_decay is not None else "")
        + (" --phase-regularization" if phase_enabled else "")
        + f" --phase-lambda {phase_lambda}"
    )


def _autodock_proxy_reward_flags(deepdel_dataset: Path, inner_dir: Path, *, outer_loop: int) -> str:
    """Flags that make train_gfn use the autodock proxy as the TB reward source."""
    reward_cfg = CFG.get("autodock_proxy_reward", {}) or {}
    threshold_min, threshold_max = _threshold_interval()
    flags = (
        " --gfn-reward-source autodock_proxy"
        f" --autodock-model {shlex.quote(str(_autodock_model_path()))}"
        f" --autodock-proxy-reward {shlex.quote(str(reward_cfg.get('reward', _cfg_get('eval_topm.reward', 'threshold'))))}"
        f" --threshold-min {threshold_min} --threshold-max {threshold_max}"
        f" --alpha {float(CFG.get('threshold_alpha', 1.0))}"
        f" --autodock-proxy-batch-size {int(reward_cfg.get('batch_pred_size', 4096))}"
        f" --reaction-mode {shlex.quote(_reaction_mode())}"
        + _reward_weight_flags()
    )
    proxy_device = reward_cfg.get("device")
    if proxy_device:
        flags += f" --autodock-proxy-device {shlex.quote(str(proxy_device))}"
    if bool(reward_cfg.get("append_encountered_to_deepdel", True)):
        flags += f" --append-encountered-dataset {shlex.quote(str(deepdel_dataset))}"
    if bool(reward_cfg.get("save_encountered_csv", True)):
        flags += f" --encountered-out {shlex.quote(str(inner_dir / 'encountered_rewards.csv'))}"
    return flags


def _build_inner_train_cmd(outer_loop: int, inner_loop: int) -> str:
    from .config import _scheduled_gfn_train_steps

    _, deepdel_dataset, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    inner_dir = _inner_dir(outer_loop, inner_loop)
    policy_ckpt = inner_dir / "gfn_policy.pt"
    if not deepdel_model_last.exists():
        raise FileNotFoundError(f"DeepDEL checkpoint not found: {deepdel_model_last}")
    steps = _scheduled_gfn_train_steps(inner_loop)
    cmd = (
        "python -u -m deepdelgfn.gfn.train_gfn"
        f" --size1 {CFG['lib_size']} --topm {CFG['topm']} --size2 {CFG['lib_size']} --size3 {CFG['lib_size']} "
        f"--logz-lr {float(_cfg_get('gfn.train.logz_lr', 1.0))} "
        f"--lr {float(_cfg_get('gfn.train.lr', 5e-4))} "
        + ("--batched-rollouts " if bool(_cfg_get("gfn.train.batched_rollouts", True)) else "")
        + f"--steps {steps} "
        + f"--batch-trajectories {int(CFG['gfn_train_batch'])} "
        + ("--save-model " if bool(_cfg_get("gfn.train.save_model", True)) else "")
        + f"--beta {float(CFG.get('beta', 500.0))} "
        + f"--subsample-ratio-start {CFG.get('train_gfn_subsample_ratio_start', 0.00)} "
        + f"--subsample-ratio-end {CFG.get('train_gfn_subsample_ratio_end', 0.02)} "
        + BBS_FLAGS.lstrip()
        + _forbidden_flags()
        + _gfn_arch_and_phase_flags()
        + f" --epsilon {float(_cfg_get('gfn.train.epsilon', 0.05))}"
        + f" --weight-decay {float(_cfg_get('gfn.train.weight_decay', 1e-5))}"
        + f" --grad-clip {float(_cfg_get('gfn.train.grad_clip', 1.0))}"
        + f" --min-per-cycle {int(_cfg_get('gfn.train.min_per_cycle', 8))}"
        + (
            " --append-molecular-weight"
            if bool(_cfg_get("gfn.model.append_molecular_weight", False))
            else ""
        )
        + _fourier_flags("gfn.model")
    )

    # Threshold flags: always communicate an interval to train_gfn.
    # In fixed-threshold mode, the interval collapses to [v, v].
    _mode, threshold_min, threshold_max = _threshold_spec()
    cmd += f" --threshold-min {threshold_min} --threshold-max {threshold_max}"
    if _gfn_reward_source() == "autodock_proxy":
        cmd += _autodock_proxy_reward_flags(deepdel_dataset, inner_dir, outer_loop=outer_loop)
    # Post-training pure-inference rollouts
    inf_steps = _cfg_get("gfn.inference_steps", 0)
    if int(inf_steps) > 0:
        cmd += f" --inference-steps {int(inf_steps)}"
        inf_subsample = _cfg_get("gfn.inference_subsample_ratio", None)
        if inf_subsample is not None:
            cmd += f" --inference-subsample-ratio {float(inf_subsample)}"
        # Add topm-threshold for fixed-threshold top-m selection during inference.
        # Only add when Fourier conditioning is enabled (otherwise threshold is ignored).
        if int(_cfg_get("gfn.model.fourier_n_freqs", 0)) > 0:
            cmd += f" --topm-threshold {_topm_threshold()}"
    return (
        cmd
        + f" --outdir {shlex.quote(str(inner_dir))}"
        + f" --policy-ckpt {shlex.quote(str(policy_ckpt))}"
        + f" --deepdel {shlex.quote(str(deepdel_model_last))}"
    )


# ---------------------------------------------------------------------------
# Inner training stage
# ---------------------------------------------------------------------------


def _write_random_topm_csv(outer_loop: int, inner_loop: int) -> Path:
    """Write a random top-m ``topm_rewards.csv`` in ``train_gfn``'s output schema.

    Random sampling replaces only the GFN sampling step of the active-learning
    inner loop: ``lib_size`` distinct building blocks are drawn uniformly (without
    replacement) from each of the three pools for each of ``topm`` libraries. The
    resulting CSV mirrors ``train_gfn``'s ``topm_rewards.csv`` column layout so the
    unchanged ``eval_topm`` / ``deepdel_update`` stages consume it directly.

    The DeepDEL ``reward`` / ``yhat`` columns are left NaN: random libraries are not
    scored by the (frozen) DeepDEL model at sampling time, only by the docking
    proxy afterwards.
    """
    from deepdelgfn.mols import dels as tri_mod

    lib_size = int(CFG.get("lib_size", 0) or 0)
    topm = int(CFG.get("topm", 0) or 0)
    if lib_size <= 0:
        raise ValueError("CFG['lib_size'] must be > 0 for random sampling")
    if topm <= 0:
        raise ValueError("CFG['topm'] must be > 0 for random sampling")

    df_full = tri_mod.PoolIO.load_pool(BBS_CSV)
    df1, df2, df3 = tri_mod.PoolIO.split_by_pool(df_full)
    pool_ids = [
        df1["ID"].astype(int).tolist(),
        df2["ID"].astype(int).tolist(),
        df3["ID"].astype(int).tolist(),
    ]
    for slot, ids in enumerate(pool_ids, start=1):
        if len(ids) < lib_size:
            raise ValueError(
                f"Random sampling needs at least lib_size={lib_size} building "
                f"blocks in pool {slot}, but {BBS_CSV} has {len(ids)}."
            )

    seed = CFG.get("sampling_seed")
    if seed is None:
        rng = np.random.default_rng()
    else:
        num_inner = int(CFG.get("num_inner_loops", 1))
        turn = int(outer_loop) * num_inner + int(inner_loop)
        rng = np.random.default_rng(int(seed) + turn)

    threshold = _outer_threshold(outer_loop)
    rows = []
    for rank in range(1, topm + 1):
        b1 = rng.choice(pool_ids[0], size=lib_size, replace=False)
        b2 = rng.choice(pool_ids[1], size=lib_size, replace=False)
        b3 = rng.choice(pool_ids[2], size=lib_size, replace=False)
        rows.append(
            {
                "rank": rank,
                "reward": np.nan,
                "yhat": np.nan,
                "B1_id": "|".join(str(int(x)) for x in b1),
                "B2_id": "|".join(str(int(x)) for x in b2),
                "B3_id": "|".join(str(int(x)) for x in b3),
                "threshold": threshold,
            }
        )

    inner_dir = _inner_dir(outer_loop, inner_loop)
    topm_csv = inner_dir / "topm_rewards.csv"
    pd.DataFrame(rows).to_csv(topm_csv, index=False)
    print(
        f"[random-sampler] Wrote {topm} random libraries "
        f"(lib_size={lib_size}) to {topm_csv}"
    )
    return topm_csv


def inner_train_outer_loop(outer_loop: int, inner_loop: int) -> None:
    strategy = str(CFG.get("sampling_strategy", "gfn")).strip().lower()
    if strategy == "random":
        print(
            f"\n==================== INNER TRAIN (random) {outer_loop}.{inner_loop} START ===================="
        )
        _write_random_topm_csv(outer_loop, inner_loop)
        print(
            f"==================== INNER TRAIN (random) {outer_loop}.{inner_loop} END ======================\n"
        )
        return
    if strategy != "gfn":
        raise ValueError(
            f"CFG['sampling_strategy'] must be 'gfn' or 'random'; got {strategy!r}"
        )
    print(f"\n==================== INNER TRAIN {outer_loop}.{inner_loop} START ====================")
    if run_cmd(_build_inner_train_cmd(outer_loop, inner_loop)) != 0:
        raise RuntimeError(f"Failure during GFN training for inner loop {outer_loop}.{inner_loop}.")
    print(f"==================== INNER TRAIN {outer_loop}.{inner_loop} END ======================\n")


# ---------------------------------------------------------------------------
# Eval-topm shard helpers
# ---------------------------------------------------------------------------


def _eval_topm_num_shards() -> int:
    n = _cfg_get("eval_topm.num_shards", None)
    if n is None:
        # Derive the shard count from the molecule upper bound. Each top-m
        # library enumerates lib_size^3 products, so the total proxy-scoring
        # work is topm * lib_size^3. batch_rows is the maximum number of
        # molecules per shard (mirrors docking.batch_rows).
        topm = int(CFG.get("topm", 10000))
        lib_size = int(CFG.get("lib_size", 20))
        batch_rows = int(_cfg_get("eval_topm.batch_rows", 2000000))
        if batch_rows <= 0:
            raise ValueError("CFG['eval_topm']['batch_rows'] must be > 0")
        n_molecules = topm * (lib_size ** 3)
        n = (n_molecules + batch_rows - 1) // batch_rows
    n = int(n)
    if n <= 0:
        raise ValueError("CFG['eval_topm']['num_shards'] must be > 0")
    return n


def _eval_topm_shard_dir(outer_loop: int, inner_loop: int) -> Path:
    return _inner_dir(outer_loop, inner_loop) / "eval_topm_shards"


def _eval_topm_actual_shard_path(outer_loop: int, inner_loop: int, shard_index: int) -> Path:
    return _eval_topm_shard_dir(outer_loop, inner_loop) / f"shard_{int(shard_index):06d}.actual_scores.csv"


def _eval_topm_dataset_shard_path(outer_loop: int, inner_loop: int, shard_index: int) -> Path:
    return _eval_topm_shard_dir(outer_loop, inner_loop) / f"shard_{int(shard_index):06d}.dataset_rows.csv"


def _build_eval_topm_cmd(
    outer_loop: int,
    inner_loop: int,
    *,
    out_csv: Path,
    dataset_out: Path,
    shard_index: int,
    num_shards: int,
) -> str:
    inner_dir = _inner_dir(outer_loop, inner_loop)
    topm_csv = inner_dir / "topm_rewards.csv"
    threshold = _outer_threshold(outer_loop)
    return (
        "python -u -m deepdelgfn.mols.eval_autodock_proxy_topm_scores"
        + BBS_FLAGS
        + f" --autodock-model {shlex.quote(str(_autodock_model_path()))}"
        + f" --topm-csv {shlex.quote(str(topm_csv))}"
        + f" --out {shlex.quote(str(out_csv))}"
        + f" --dataset-out {shlex.quote(str(dataset_out))}"
        + f" --reward {shlex.quote(str(_cfg_get('eval_topm.reward', 'threshold')))}"
        + f" --batch-pred-size {int(_cfg_get('eval_topm.batch_pred_size', 4096))}"
        + f" --num-workers {int(_cfg_get('eval_topm.num_workers', -1))}"
        + f" --device {shlex.quote(str(_cfg_get('eval_topm.device', 'auto')))}"
        + f" --torch-num-threads {int(_cfg_get('eval_topm.torch_num_threads', 1))}"
        + f" --torch-interop-threads {int(_cfg_get('eval_topm.torch_interop_threads', 1))}"
        + f" --score-log-seconds {float(_cfg_get('eval_topm.score_log_seconds', 30.0) or 0.0)}"
        + f" --threshold {threshold} --alpha {float(CFG.get('threshold_alpha', 1.0))}"
        + f" --reaction-mode {shlex.quote(_reaction_mode())}"
        + _threshold_weight_flags()
        + (" --log-target" if bool(CFG.get("log_reward_target", True)) else "")
        + f" --shard-index {int(shard_index)} --num-shards {int(num_shards)}"
    )


def eval_topm_batch_outer_loop(outer_loop: int, inner_loop: int) -> None:
    if _gfn_reward_source() != "deepdel":
        print(
            "[eval-topm-batch] Skipping: GFN reward source is autodock_proxy; "
            "train_gfn already wrote topm_actual_scores.csv."
        )
        return
    idx_raw = os.environ.get("EVAL_TOPM_BATCH_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx_raw in (None, ""):
        raise RuntimeError(
            "EVAL_TOPM_BATCH_INDEX or SLURM_ARRAY_TASK_ID is required for STAGE=eval_topm_batch"
        )
    shard_index = int(idx_raw)
    n_shards = _eval_topm_num_shards()
    shard_dir = _eval_topm_shard_dir(outer_loop, inner_loop)
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_csv = _eval_topm_actual_shard_path(outer_loop, inner_loop, shard_index)
    dataset_out = _eval_topm_dataset_shard_path(outer_loop, inner_loop, shard_index)
    for p in (out_csv, dataset_out):
        if p.exists():
            p.unlink()
    if (
        run_cmd(
            _build_eval_topm_cmd(
                outer_loop,
                inner_loop,
                out_csv=out_csv,
                dataset_out=dataset_out,
                shard_index=shard_index,
                num_shards=n_shards,
            )
        )
        != 0
    ):
        raise RuntimeError(
            f"Failure during top-m eval shard {shard_index} for inner loop "
            f"{outer_loop}.{inner_loop}."
        )


# ---------------------------------------------------------------------------
# Eval-topm statistics columns & helpers
# ---------------------------------------------------------------------------

_EVAL_TOPM_STATS_COLUMNS = [
    "timestamp",
    "run_id",
    "outer_loop",
    "inner_loop",
    "global_step",
    "topm_csv",
    "pred_col",
    "actual_col",
    "n_total",
    "n_finite_pairs",
    "mse",
    "pearson_r",
    "spearman_r",
    "mean_appended_y",
    "max_appended_y",
    "n_appended_y",
    "mean_deepdel_model_value",
    "max_deepdel_model_value",
    "n_deepdel_model_value",
]


def _finite_numeric_summary(values) -> tuple[float, float, int]:
    """Return mean/max/count for finite numeric values, or (nan, nan, 0)."""
    arr = pd.to_numeric(values, errors="coerce")
    finite = arr.notna() & np.isfinite(arr.to_numpy(dtype=float))
    n_finite = int(finite.sum())
    if n_finite == 0:
        return float("nan"), float("nan"), 0
    finite_values = arr.loc[finite].astype(float)
    return float(finite_values.mean()), float(finite_values.max()), n_finite


def _compute_appended_y_stats(dataset_rows: Path) -> tuple[float, float, int]:
    """Compute target-y summary stats for rows appended to the DeepDEL dataset."""
    try:
        df = pd.read_csv(dataset_rows)
    except Exception as e:
        print(f"[eval-topm-finalize] WARNING: Could not read appended dataset rows {dataset_rows}: {e}")
        return float("nan"), float("nan"), 0
    if "y" not in df.columns:
        print(
            f"[eval-topm-finalize] WARNING: Cannot compute appended-y stats; "
            f"missing column 'y' in {dataset_rows}"
        )
        return float("nan"), float("nan"), 0
    return _finite_numeric_summary(df["y"])


def _compute_eval_topm_stats(
    topm_out: Path,
    *,
    outer_loop: int,
    inner_loop: int,
    mean_appended_y: float = float("nan"),
    max_appended_y: float = float("nan"),
    n_appended_y: int = 0,
) -> Optional[dict]:
    """Compute proxy-vs-evaluated reward statistics for finalized top-m rows."""
    pred_col = "proxy_reward"
    actual_col = "autodock_proxy_value"
    try:
        df = pd.read_csv(topm_out)
    except Exception as e:
        print(f"[eval-topm-finalize] WARNING: Could not read {topm_out} for evaluation stats: {e}")
        return
    missing = [c for c in (pred_col, actual_col) if c not in df.columns]
    if missing:
        print(
            "[eval-topm-finalize] WARNING: Skipping evaluation stats; "
            f"missing column(s): {', '.join(missing)}"
        )
        return
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    actual = pd.to_numeric(df[actual_col], errors="coerce")
    mean_deepdel_model_value, max_deepdel_model_value, n_deepdel_model_value = _finite_numeric_summary(pred)
    valid = (
        pred.notna()
        & actual.notna()
        & np.isfinite(pred.to_numpy(dtype=float))
        & np.isfinite(actual.to_numpy(dtype=float))
    )
    n_total = int(len(df))
    n_valid = int(valid.sum())
    if n_valid == 0:
        print(
            "[eval-topm-finalize] WARNING: Skipping evaluation stats; "
            f"no finite paired values for {pred_col} vs {actual_col} in {topm_out}"
        )
        return None
    pred_valid = pred.loc[valid].astype(float)
    actual_valid = actual.loc[valid].astype(float)
    mse = float(np.mean(np.square(pred_valid.to_numpy() - actual_valid.to_numpy())))
    pearson = float(pred_valid.corr(actual_valid, method="pearson")) if n_valid >= 2 else float("nan")
    spearman = (
        float(pred_valid.rank(method="average").corr(actual_valid.rank(method="average"), method="pearson"))
        if n_valid >= 2
        else float("nan")
    )
    stats = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "run_id": RUN_ID,
        "outer_loop": int(outer_loop),
        "inner_loop": int(inner_loop),
        "global_step": int(outer_loop) * int(CFG["num_inner_loops"]) + int(inner_loop),
        "topm_csv": str(topm_out),
        "pred_col": pred_col,
        "actual_col": actual_col,
        "n_total": n_total,
        "n_finite_pairs": n_valid,
        "mse": mse,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mean_appended_y": float(mean_appended_y),
        "max_appended_y": float(max_appended_y),
        "n_appended_y": int(n_appended_y),
        "mean_deepdel_model_value": mean_deepdel_model_value,
        "max_deepdel_model_value": max_deepdel_model_value,
        "n_deepdel_model_value": n_deepdel_model_value,
    }
    print(
        "[eval-topm-finalize] Evaluation stats "
        f"({pred_col} vs {actual_col}, finite pairs={n_valid}/{n_total}): "
        f"mse={mse:.8g}, pearson_r={pearson:.8g}, spearman_r={spearman:.8g}"
    )
    print(
        "[eval-topm-finalize] Appended/model summaries: "
        f"mean_appended_y={float(mean_appended_y):.8g}, "
        f"max_appended_y={float(max_appended_y):.8g} (n={int(n_appended_y)}), "
        f"mean_deepdel_model_value={mean_deepdel_model_value:.8g}, "
        f"max_deepdel_model_value={max_deepdel_model_value:.8g} (n={n_deepdel_model_value})"
    )
    return stats


def _update_eval_topm_stats_csv(stats: dict) -> Path:
    """Upsert one eval-topm stats row into the run-level stats CSV."""
    out_csv = RUN_ROOT / "eval_topm_stats.csv"
    row_df = pd.DataFrame([stats], columns=_EVAL_TOPM_STATS_COLUMNS)
    if out_csv.exists() and _csv_data_row_count(out_csv) > 0:
        try:
            old_df = pd.read_csv(out_csv)
        except Exception as e:
            print(
                f"[eval-topm-finalize] WARNING: Could not read existing stats CSV {out_csv}; "
                f"rewriting it: {e}"
            )
            old_df = pd.DataFrame(columns=_EVAL_TOPM_STATS_COLUMNS)
        for col in _EVAL_TOPM_STATS_COLUMNS:
            if col not in old_df.columns:
                old_df[col] = np.nan
        old_df = old_df[_EVAL_TOPM_STATS_COLUMNS]
        same_loop = pd.to_numeric(old_df["outer_loop"], errors="coerce").eq(
            int(stats["outer_loop"])
        ) & pd.to_numeric(old_df["inner_loop"], errors="coerce").eq(int(stats["inner_loop"]))
        out_df = pd.concat([old_df.loc[~same_loop], row_df], ignore_index=True)
    else:
        out_df = row_df
    out_df["global_step"] = pd.to_numeric(out_df["global_step"], errors="coerce")
    out_df = out_df.sort_values(by=["global_step", "outer_loop", "inner_loop"]).reset_index(
        drop=True
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    out_df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(out_csv)
    print(f"[eval-topm-finalize] Updated eval-topm stats CSV: {out_csv}")
    return out_csv


def _write_eval_topm_stats_plot(stats_csv: Path) -> None:
    """Plot the evolution of eval-topm statistics over active-learning steps."""
    try:
        if not stats_csv.exists() or _csv_data_row_count(stats_csv) <= 0:
            return
        df = pd.read_csv(stats_csv)
        if len(df) == 0:
            return
        for col in ("global_step", "mse", "pearson_r", "spearman_r"):
            if col not in df.columns:
                print(
                    f"[eval-topm-finalize] WARNING: Cannot plot eval-topm stats; "
                    f"missing column: {col}"
                )
                return
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in (
            "mean_appended_y",
            "max_appended_y",
            "mean_deepdel_model_value",
            "max_deepdel_model_value",
        ):
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["global_step"]).sort_values(by="global_step")
        if len(df) == 0:
            return
        df["log_mse"] = np.where(df["mse"] > 0, np.log(df["mse"]), np.nan)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(8, 10.5), sharex=True)
        plot_specs = [
            ("log_mse", "log(MSE)", "log MSE"),
            ("pearson_r", "Pearson r", "Pearson r"),
            ("spearman_r", "Spearman r", "Spearman rank r"),
        ]
        for ax, (col, ylabel, title) in zip(axes[:3], plot_specs):
            valid_data = df[["global_step", col]].dropna()
            if len(valid_data) > 0:
                ax.plot(valid_data["global_step"], valid_data[col], marker="o", label=col)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        avg_ax = axes[3]
        appended_valid = df[["global_step", "mean_appended_y"]].dropna()
        appended_max_valid = df[["global_step", "max_appended_y"]].dropna()
        model_valid = df[["global_step", "mean_deepdel_model_value"]].dropna()
        model_max_valid = df[["global_step", "max_deepdel_model_value"]].dropna()
        if len(appended_valid) > 0:
            avg_ax.plot(
                appended_valid["global_step"],
                appended_valid["mean_appended_y"],
                marker="o",
                color="tab:blue",
                label="avg appended y",
            )
        if len(appended_max_valid) > 0:
            avg_ax.plot(
                appended_max_valid["global_step"],
                appended_max_valid["max_appended_y"],
                marker="^",
                linestyle="--",
                color="tab:blue",
                label="max appended y",
            )
        avg_ax.set_ylabel("appended y", color="tab:blue")
        avg_ax.tick_params(axis="y", labelcolor="tab:blue")
        avg_ax.set_title("Appended target y and DeepDEL model value")
        avg_ax.grid(True, alpha=0.3)

        model_ax = avg_ax.twinx()
        if len(model_valid) > 0:
            model_ax.plot(
                model_valid["global_step"],
                model_valid["mean_deepdel_model_value"],
                marker="s",
                color="tab:orange",
                label="avg DeepDEL model value",
            )
        if len(model_max_valid) > 0:
            model_ax.plot(
                model_max_valid["global_step"],
                model_max_valid["max_deepdel_model_value"],
                marker="D",
                linestyle="--",
                color="tab:orange",
                label="max DeepDEL model value",
            )
        model_ax.set_ylabel("DeepDEL model value", color="tab:orange")
        model_ax.tick_params(axis="y", labelcolor="tab:orange")

        handles, labels = avg_ax.get_legend_handles_labels()
        handles2, labels2 = model_ax.get_legend_handles_labels()
        if handles or handles2:
            avg_ax.legend(handles + handles2, labels + labels2, loc="best")
        axes[-1].set_xlabel("inner_loop step")
        fig.suptitle("Top-m evaluation statistics over active learning")
        fig.suptitle(_plot_title_metadata(prefix="Top-m evaluation statistics over active learning | "))
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_png = RUN_ROOT / "eval_topm_stats_evolution.png"
        df.to_csv(out_png.with_suffix(".csv"), index=False)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[plot] Saved eval-topm stats evolution plot to {out_png}")
    except Exception as e:
        print(f"[WARN] Failed to generate eval-topm stats plot: {e}")


def _record_eval_topm_stats(
    topm_out: Path,
    *,
    outer_loop: int,
    inner_loop: int,
    mean_appended_y: float = float("nan"),
    max_appended_y: float = float("nan"),
    n_appended_y: int = 0,
) -> None:
    stats = _compute_eval_topm_stats(
        topm_out,
        outer_loop=outer_loop,
        inner_loop=inner_loop,
        mean_appended_y=mean_appended_y,
        max_appended_y=max_appended_y,
        n_appended_y=n_appended_y,
    )
    if stats is None:
        return
    stats_csv = _update_eval_topm_stats_csv(stats)
    _write_eval_topm_stats_plot(stats_csv)


# ---------------------------------------------------------------------------
# Eval-topm finalize stage
# ---------------------------------------------------------------------------


def eval_topm_finalize_outer_loop(outer_loop: int, inner_loop: int) -> None:
    if _gfn_reward_source() != "deepdel":
        print("[eval-topm-finalize] Skipping: GFN reward source is autodock_proxy.")
        return
    _outer_dir, deepdel_dataset, _ = _outer_deepdel_paths(outer_loop)
    inner_dir = _inner_dir(outer_loop, inner_loop)
    topm_out = inner_dir / "topm_actual_scores.csv"
    n_shards = _eval_topm_num_shards()
    actuals = [_eval_topm_actual_shard_path(outer_loop, inner_loop, i) for i in range(n_shards)]
    datasets = [_eval_topm_dataset_shard_path(outer_loop, inner_loop, i) for i in range(n_shards)]
    for p in actuals + datasets:
        if not p.exists():
            raise FileNotFoundError(f"Expected eval_topm shard output is missing: {p}")
    _concat_csvs_with_matching_headers(actuals, topm_out)
    tmp_dataset_rows = _eval_topm_shard_dir(outer_loop, inner_loop) / "dataset_rows_all.csv"
    _concat_csvs_with_matching_headers(datasets, tmp_dataset_rows)
    appended = _csv_data_row_count(tmp_dataset_rows)
    mean_appended_y, max_appended_y, n_appended_y = _compute_appended_y_stats(tmp_dataset_rows)
    with tmp_dataset_rows.open("r", newline="", encoding="utf-8") as fin, deepdel_dataset.open(
        "a", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        try:
            next(reader)
        except StopIteration:
            return
        writer = csv.writer(fout, lineterminator="\n")
        for row in reader:
            writer.writerow(row)
    print(
        f"[eval-topm-finalize] Wrote {topm_out} and appended {appended} row(s) to "
        f"{deepdel_dataset} (finite y mean={mean_appended_y:.8g}, "
        f"max={max_appended_y:.8g}, n={n_appended_y})"
    )
    _record_eval_topm_stats(
        topm_out,
        outer_loop=outer_loop,
        inner_loop=inner_loop,
        mean_appended_y=mean_appended_y,
        max_appended_y=max_appended_y,
        n_appended_y=n_appended_y,
    )


# ---------------------------------------------------------------------------
# DeepDEL update helpers
# ---------------------------------------------------------------------------

_DEEPDEL_UPDATE_STATS_COLUMNS = [
    "timestamp",
    "run_id",
    "outer_loop",
    "inner_loop",
    "global_step",
    "last_val_mse",
    "best_val_mse",
    "epochs_completed",
    "train_mse_json",
    "val_mse_json",
]


def _update_deepdel_update_stats_csv(stats: dict) -> Path:
    """Upsert one deepdel-update stats row into the run-level stats CSV."""
    out_csv = RUN_ROOT / "deepdel_update_stats.csv"
    row_df = pd.DataFrame([stats], columns=_DEEPDEL_UPDATE_STATS_COLUMNS)
    if out_csv.exists() and _csv_data_row_count(out_csv) > 0:
        try:
            old_df = pd.read_csv(out_csv)
        except Exception as e:
            print(
                f"[deepdel-update] WARNING: Could not read existing stats CSV {out_csv}; "
                f"rewriting it: {e}"
            )
            old_df = pd.DataFrame(columns=_DEEPDEL_UPDATE_STATS_COLUMNS)
        for col in _DEEPDEL_UPDATE_STATS_COLUMNS:
            if col not in old_df.columns:
                old_df[col] = np.nan
        old_df = old_df[_DEEPDEL_UPDATE_STATS_COLUMNS]
        same_loop = pd.to_numeric(old_df["outer_loop"], errors="coerce").eq(
            int(stats["outer_loop"])
        ) & pd.to_numeric(old_df["inner_loop"], errors="coerce").eq(int(stats["inner_loop"]))
        out_df = pd.concat([old_df.loc[~same_loop], row_df], ignore_index=True)
    else:
        out_df = row_df
    out_df["global_step"] = pd.to_numeric(out_df["global_step"], errors="coerce")
    out_df = out_df.sort_values(by=["global_step", "outer_loop", "inner_loop"]).reset_index(
        drop=True
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    out_df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(out_csv)
    print(f"[deepdel-update] Updated deepdel-update stats CSV: {out_csv}")
    return out_csv


def _write_deepdel_update_stats_plot(stats_csv: Path) -> None:
    """Plot the evolution of deepdel-update training metrics over active-learning steps."""
    try:
        if not stats_csv.exists() or _csv_data_row_count(stats_csv) <= 0:
            return
        df = pd.read_csv(stats_csv)
        if len(df) == 0:
            return
        for col in ("global_step", "last_val_mse", "best_val_mse", "epochs_completed"):
            if col not in df.columns:
                print(
                    "[deepdel-update] WARNING: Cannot plot deepdel-update stats; "
                    f"missing column: {col}"
                )
                return
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["global_step"]).sort_values(by="global_step")
        if len(df) == 0:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

        # ---- Top panel: last_val_mse and best_val_mse vs global_step ----
        ax_top = axes[0]
        valid_last = df[["global_step", "last_val_mse"]].dropna()
        valid_best = df[["global_step", "best_val_mse"]].dropna()
        if len(valid_last) > 0:
            ax_top.plot(
                valid_last["global_step"],
                valid_last["last_val_mse"],
                marker="o",
                label="last val MSE",
                color="tab:blue",
            )
        if len(valid_best) > 0:
            ax_top.plot(
                valid_best["global_step"],
                valid_best["best_val_mse"],
                marker="s",
                label="best val MSE",
                color="tab:orange",
            )
        ax_top.set_ylabel("MSE")
        ax_top.set_title("DeepDEL update summary (last & best val MSE)")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend()

        # ---- Bottom panel: per-epoch train/val MSE lines per cycle ----
        ax_bot = axes[1]
        global_steps = df["global_step"].to_numpy(dtype=float)
        if len(global_steps) > 0:
            norm = Normalize(vmin=global_steps.min(), vmax=global_steps.max())
            cmap = plt.get_cmap("viridis")
            sm = ScalarMappable(norm=norm, cmap=cmap)

            n_cycles_plotted = 0
            for _, row_data in df.iterrows():
                gs = float(row_data["global_step"])
                try:
                    train_mse = json.loads(row_data.get("train_mse_json", "[]"))
                    val_mse = json.loads(row_data.get("val_mse_json", "[]"))
                except Exception:
                    continue
                if not train_mse and not val_mse:
                    continue
                color = cmap(norm(gs))
                epochs = list(range(1, len(train_mse) + 1))
                if len(train_mse) > 0:
                    ax_bot.plot(
                        epochs,
                        train_mse,
                        color=color,
                        linewidth=1.0,
                        alpha=0.7,
                        label="train" if n_cycles_plotted == 0 else "",
                    )
                val_epochs = list(range(1, len(val_mse) + 1))
                if len(val_mse) > 0:
                    ax_bot.plot(
                        val_epochs,
                        val_mse,
                        color=color,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.7,
                        label="val" if n_cycles_plotted == 0 else "",
                    )
                n_cycles_plotted += 1

            cbar = fig.colorbar(sm, ax=ax_bot, label="global_step")
            cbar  # suppress unused-variable warning (used for side effect)
        ax_bot.set_xlabel("epoch")
        ax_bot.set_ylabel("MSE")
        ax_bot.set_title("Per-epoch train/val MSE (darker = later cycles)")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend()

        fig.suptitle("DeepDEL update training metrics over active learning")
        fig.suptitle(_plot_title_metadata(prefix="DeepDEL update training metrics over active learning | "))
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_png = RUN_ROOT / "deepdel_update_stats_evolution.png"
        df.to_csv(out_png.with_suffix(".csv"), index=False)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[plot] Saved deepdel-update stats evolution plot to {out_png}")
    except Exception as e:
        print(f"[WARN] Failed to generate deepdel-update stats plot: {e}")


def _record_deepdel_update_stats(
    stats_json_path: Path, *, outer_loop: int, inner_loop: int
) -> None:
    """Read the stats JSON from train_offline and record into the run-level CSV and plot."""
    if not stats_json_path.exists():
        print(f"[deepdel-update] WARNING: Stats JSON not found at {stats_json_path}; skipping recording.")
        return
    try:
        with stats_json_path.open("r") as f:
            ts = json.load(f)
    except Exception as e:
        print(f"[deepdel-update] WARNING: Could not read stats JSON {stats_json_path}: {e}")
        return
    stats = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "run_id": RUN_ID,
        "outer_loop": int(outer_loop),
        "inner_loop": int(inner_loop),
        "global_step": int(outer_loop) * int(CFG["num_inner_loops"]) + int(inner_loop),
        "last_val_mse": float(ts.get("last_val_mse", float("nan"))),
        "best_val_mse": float(ts.get("best_val_mse", float("nan"))),
        "epochs_completed": int(ts.get("epochs_completed", 0)),
        "train_mse_json": json.dumps(ts.get("train_mse", [])),
        "val_mse_json": json.dumps(ts.get("val_mse", [])),
    }
    print(
        "[deepdel-update] Training stats: "
        f"last_val_mse={stats['last_val_mse']:.8g}, "
        f"best_val_mse={stats['best_val_mse']:.8g}, "
        f"epochs_completed={stats['epochs_completed']}"
    )
    stats_csv = _update_deepdel_update_stats_csv(stats)
    _write_deepdel_update_stats_plot(stats_csv)


# ---------------------------------------------------------------------------
# DeepDEL update command builder & stage
# ---------------------------------------------------------------------------


def _inner_docking_deepdel_path(outer_loop: int, inner_loop: int) -> Path:
    return _inner_dir(outer_loop, inner_loop) / "models" / "deepdel_for_docking.pt"


def _build_deepdel_update_cmd(outer_loop: int, inner_loop: int) -> str:
    _outer_dir, deepdel_dataset, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    inner_dir = _inner_dir(outer_loop, inner_loop)
    stats_json = inner_dir / "deepdel_update_stats.json"
    pooling = str(_cfg_get("deepdel_offline.model.pooling", "mean")).lower()
    output_head = str(_cfg_get("deepdel_offline.model.output_head", "linear")).lower()
    lib_size = int(CFG.get("lib_size", 0) or 0)
    mode = str(_cfg_get("deepdel_offline.after_gfn.mode", "finetune")).lower()
    common = (
        "python -u -m deepdelgfn.deepdel.train_offline"
        + BBS_FLAGS
        + f" --bb-fp-bits {int(_cfg_get('deepdel_offline.model.bb_fp_bits', 2048))}"
        + f" --bb-fp-radius {int(_cfg_get('deepdel_offline.model.bb_fp_radius', 2))}"
        + f" --hidden-dim {int(_cfg_get('deepdel_offline.model.hidden_dim', 512))}"
        + f" --rho-dim {int(_cfg_get('deepdel_offline.model.rho_dim', 512))}"
        + f" --dropout {float(_cfg_get('deepdel_offline.model.dropout', 0.1))}"
        + f" --pooling {shlex.quote(pooling)}"
        + (" --shared-phi" if bool(_cfg_get("deepdel_offline.model.shared_phi", True)) else "")
        + f" --output-head {shlex.quote(output_head)}"
        + (f" --lib-size {lib_size}" if output_head != "linear" and lib_size > 0 else "")
        + (
            " --append-molecular-weight"
            if bool(_cfg_get("deepdel_offline.model.append_molecular_weight", False))
            else ""
        )
        + _fourier_flags("deepdel_offline.model")
        + (" --log-target" if bool(CFG.get("log_reward_target", True)) else "")
        + f" --dataset {shlex.quote(str(deepdel_dataset))} --save-last {shlex.quote(str(deepdel_model_last))}"
        + _deepdel_eval_dataset_flag()
        + f" --stats-json {shlex.quote(str(stats_json))}"
    )
    if mode == "retrain":
        cfg_prefix = "deepdel_offline.after_gfn"
        return (
            common
            + f" --lr {float(_cfg_get(cfg_prefix + '.lr', _cfg_get('deepdel_offline.train.lr', 5e-5)))}"
            + f" --weight-decay {float(_cfg_get(cfg_prefix + '.weight_decay', _cfg_get('deepdel_offline.train.weight_decay', 1e-5)))}"
            + f" --loss {shlex.quote(str(_cfg_get(cfg_prefix + '.loss', _cfg_get('deepdel_offline.train.loss', 'mse'))))}"
            + f" --epochs {int(_cfg_get(cfg_prefix + '.epochs', _cfg_get('deepdel_offline.train.epochs', 20)))}"
            + f" --patience {int(_cfg_get(cfg_prefix + '.patience', _cfg_get('deepdel_offline.train.patience', 0)))}"
            + f" --validation-type {shlex.quote(str(_cfg_get(cfg_prefix + '.validation_type', _cfg_get('deepdel_offline.train.validation_type', 'random'))))}"
            + f" --val-frac {float(_cfg_get(cfg_prefix + '.val_frac', _cfg_get('deepdel_offline.train.val_frac', 0.2)))}"
            + f" --val-bins {int(_cfg_get(cfg_prefix + '.val_bins', _cfg_get('deepdel_offline.train.val_bins', 10)))}"
            + f" --batch-size {int(_cfg_get(cfg_prefix + '.batch_size', _cfg_get('deepdel_offline.train.batch_size', 128)))}"
            + f" --num-workers {int(_cfg_get(cfg_prefix + '.num_workers', _cfg_get('deepdel_offline.train.num_workers', -1)))}"
            + f" --prefetch-factor {int(_cfg_get(cfg_prefix + '.prefetch_factor', _cfg_get('deepdel_offline.train.prefetch_factor', 4)))}"
            + (
                " --persistent-workers"
                if bool(
                    _cfg_get(
                        cfg_prefix + ".persistent_workers",
                        _cfg_get("deepdel_offline.train.persistent_workers", True),
                    )
                )
                else ""
            )
            + (
                " --amp"
                if bool(_cfg_get(cfg_prefix + ".amp", _cfg_get("deepdel_offline.train.amp", True)))
                else ""
            )
            + (
                " --tf32"
                if bool(_cfg_get(cfg_prefix + ".tf32", _cfg_get("deepdel_offline.train.tf32", True)))
                else ""
            )
            + f" --max-workers {int(_cfg_get(cfg_prefix + '.max_workers', _cfg_get('deepdel_offline.train.max_workers', 32)))}"
        )
    if mode == "finetune":
        cfg_prefix = "deepdel_offline.finetune"
        return (
            common
            + f" --resume {shlex.quote(str(deepdel_model_last))}"
            + f" --lr {float(_cfg_get(cfg_prefix + '.lr', 1e-5))}"
            + f" --weight-decay {float(_cfg_get(cfg_prefix + '.weight_decay', 1e-5))}"
            + f" --loss {shlex.quote(str(_cfg_get(cfg_prefix + '.loss', _cfg_get('deepdel_offline.train.loss', 'mse'))))}"
            + f" --epochs {int(_cfg_get(cfg_prefix + '.epochs', 1))}"
            + f" --patience {int(_cfg_get(cfg_prefix + '.patience', 0))}"
            + f" --validation-type {shlex.quote(str(_cfg_get(cfg_prefix + '.validation_type', 'random')))}"
            + f" --val-frac {float(_cfg_get(cfg_prefix + '.val_frac', _cfg_get('deepdel_offline.train.val_frac', 0.2)))}"
            + f" --val-bins {int(_cfg_get(cfg_prefix + '.val_bins', _cfg_get('deepdel_offline.train.val_bins', 10)))}"
            + (
                " --resume-optimizer"
                if bool(_cfg_get(cfg_prefix + ".resume_optimizer", True))
                else ""
            )
            + f" --batch-size {int(_cfg_get(cfg_prefix + '.batch_size', 128))}"
            + f" --num-workers {int(_cfg_get(cfg_prefix + '.num_workers', -1))}"
            + f" --prefetch-factor {int(_cfg_get(cfg_prefix + '.prefetch_factor', 4))}"
            + (
                " --persistent-workers"
                if bool(_cfg_get(cfg_prefix + ".persistent_workers", True))
                else ""
            )
            + (" --amp" if bool(_cfg_get(cfg_prefix + ".amp", True)) else "")
            + (" --tf32" if bool(_cfg_get(cfg_prefix + ".tf32", True)) else "")
        )
    raise ValueError("CFG['deepdel_offline']['after_gfn']['mode'] must be 'retrain' or 'finetune'")


def deepdel_update_outer_loop(outer_loop: int, inner_loop: int) -> None:
    print(f"\n==================== DEEPDEL UPDATE {outer_loop}.{inner_loop} START ====================")
    if run_cmd(_build_deepdel_update_cmd(outer_loop, inner_loop)) != 0:
        raise RuntimeError(
            f"Failure during DeepDEL update after inner loop {outer_loop}.{inner_loop}."
        )
    _, _, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    snapshot = _inner_docking_deepdel_path(outer_loop, inner_loop)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(deepdel_model_last, snapshot)
    print(
        f"[deepdel-update] Snapshotted DeepDEL checkpoint for docking leader selection: "
        f"{deepdel_model_last} -> {snapshot}"
    )
    inner_dir = _inner_dir(outer_loop, inner_loop)
    stats_json = inner_dir / "deepdel_update_stats.json"
    _record_deepdel_update_stats(stats_json, outer_loop=outer_loop, inner_loop=inner_loop)
    step = int(outer_loop) * int(CFG["num_inner_loops"]) + int(inner_loop) + 1
    record_deepdel_val_mse(
        stats_json,
        step=step,
        phase="update",
        outer_loop=outer_loop,
        inner_loop=inner_loop,
    )
    print(
        f"==================== DEEPDEL UPDATE {outer_loop}.{inner_loop} END ======================\n"
    )