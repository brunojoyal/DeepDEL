#!/usr/bin/env python3
"""Prepare-stage functions for the active learning pipeline.

This module was extracted from active_learning_stage.py.  It covers the
``prepare_proxy``, ``prepare_dataset``, ``prepare_dataset_batch``,
``prepare_dataset_finalize``, and ``prepare_deepdel`` stages.
"""

import os
import shlex
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    BBS_CSV,
    BBS_FLAGS,
    CFG,
    CPUS,
    DOCKING_OUT_DIR,
    FALLBACK_AUTODOCK_MODEL_JOBLIB,
    FALLBACK_AUTODOCK_MODEL_PT,
    FALLBACK_DEEPDEL_MODEL_PT,
    MODEL_ROOT,
    PROJECT_ROOT,
    RUN_ROOT,
    _fixed_threshold,
    _autodock_kind,
    _autodock_model_path,
    _cfg_get,
    _concat_csvs_with_matching_headers,
    _csv_data_row_count,
    _deepdel_dataset_device_flag,
    _deepdel_dataset_seed_flag,
    _deepdel_eval_dataset_flag,
    _deepdel_model_arch_flags,
    _deepdel_validation_csv,
    _fourier_flags,
    _outer_deepdel_paths,
    _pretrained_deepdel_source,
    _reaction_mode,
    _reward_weight_flags,
    _threshold_interval,
    run_cmd,
    torch_cuda_available,
)

from .val_mse_tracker import record_deepdel_val_mse


# ---------------------------------------------------------------------------
# Autodock proxy training command
# ---------------------------------------------------------------------------


def _build_train_autodock_cmd() -> str:
    kind = _autodock_kind()
    inputs = " ".join(
        shlex.quote(str(p))
        for p in _cfg_get(
            "paths.autodock_proxy_inputs",
            ["data/scored_libraries"],
        )
    )
    smiles_col = _cfg_get("autodock_proxy.smiles_col", "smiles")
    target_col = _cfg_get("autodock_proxy.target_col", "docking_score")
    common_cols = (
        f" --smiles_col {shlex.quote(str(smiles_col))}"
        f" --target_col {shlex.quote(str(target_col))}"
    )
    target_transform_flags = (
        " --clip_positive_targets"
        if bool(_cfg_get("autodock_proxy.clip_positive_targets", False))
        else ""
    )
    if kind == "nn":
        nn_cfg = _cfg_get("autodock_proxy.nn", {}) or {}
        # Optional ECFP cache (skips RDKit featurization if a valid cache exists).
        ecfp_cache = nn_cfg.get("ecfp_cache")
        cache_flags = ""
        if ecfp_cache:
            cache_flags += f" --ecfp_cache {shlex.quote(str(ecfp_cache))}"
            if bool(nn_cfg.get("overwrite_ecfp_cache", False)):
                cache_flags += " --overwrite_ecfp_cache"
        return (
            "python -u -m deepdelgfn.autodock_proxy.train_nn"
            f" --model_out {_autodock_model_path()}"
            f" --inputs {inputs}"
            + common_cols
            + f" --n_bits {int(nn_cfg.get('n_bits', 4096))}"
            + f" --radius {int(nn_cfg.get('radius', 2))}"
            + f" --hidden_dim {int(nn_cfg.get('hidden_dim', 512))}"
            + f" --n_layers {int(nn_cfg.get('n_layers', 3))}"
            + f" --dropout {float(nn_cfg.get('dropout', 0.1))}"
            + f" --lr {float(nn_cfg.get('lr', 3e-4))}"
            + f" --weight_decay {float(nn_cfg.get('weight_decay', 1e-5))}"
            + f" --batch_size {int(nn_cfg.get('batch_size', 1024))}"
            + f" --epochs {int(nn_cfg.get('epochs', 50))}"
            + f" --patience {int(nn_cfg.get('patience', 8))}"
            + f" --test_size {float(nn_cfg.get('test_size', 0.2))}"
            + f" --val_size {float(nn_cfg.get('val_size', 0.1))}"
            + (" --standardize_y" if bool(nn_cfg.get("standardize_y", True)) else "")
            + (" --dedup_smiles" if bool(nn_cfg.get("dedup_smiles", True)) else "")
            + target_transform_flags
            + cache_flags
        )
    return (
        "python -u -m deepdelgfn.autodock_proxy.train_rf"
        f" --model_out {_autodock_model_path()}"
        f" --inputs {inputs}"
        + common_cols
        + f" --n_estimators {int(_cfg_get('autodock_proxy.n_estimators', 100))}"
        + target_transform_flags
    )


# ---------------------------------------------------------------------------
# Dataset generation helpers
# ---------------------------------------------------------------------------


def _prepare_paths_and_threshold(outer_loop: int) -> tuple[Path, Path, float]:
    """Return prepare paths and a backward-compatible fixed threshold.

    The threshold is not needed to construct these paths. Resolve it through
    the interval-aware helper so interval-only configurations do not fail
    while entering a path-only stage such as dataset sharding.
    """
    outer_dir = RUN_ROOT / f"outer_{outer_loop}"
    deepdel_dir = outer_dir / "deepdel"
    deepdel_dir.mkdir(parents=True, exist_ok=True)
    deepdel_dataset = deepdel_dir / "deepdel_dataset.csv"
    threshold = float(_fixed_threshold())
    return deepdel_dir, deepdel_dataset, threshold


def _prepare_paths_and_threshold_interval(outer_loop: int) -> tuple[Path, Path, float, float]:
    """Like ``_prepare_paths_and_threshold`` but returns the (min, max) interval."""
    outer_dir = RUN_ROOT / f"outer_{outer_loop}"
    deepdel_dir = outer_dir / "deepdel"
    deepdel_dir.mkdir(parents=True, exist_ok=True)
    deepdel_dataset = deepdel_dir / "deepdel_dataset.csv"
    threshold_min, threshold_max = _threshold_interval()
    return deepdel_dir, deepdel_dataset, threshold_min, threshold_max


def _prepare_dataset_shard_dir(outer_loop: int) -> Path:
    # Use the interval-aware helper so interval-only configs never require
    # CFG['threshold_value'].
    deepdel_dir, _, _, _ = _prepare_paths_and_threshold_interval(outer_loop)
    return deepdel_dir / "dataset_shards"


def _prepare_dataset_shard_path(outer_loop: int, shard_index: int) -> Path:
    return _prepare_dataset_shard_dir(outer_loop) / f"shard_{int(shard_index):06d}.csv"


def _prepare_dataset_shard_triplets(shard_index: int) -> int:
    total = int(CFG["deepdel_dataset_initial_size"])
    n_shards = int(_cfg_get("deepdel_dataset.num_shards", 1))
    if total < 0:
        raise ValueError("CFG['deepdel_dataset_initial_size'] must be >= 0")
    if n_shards <= 0:
        raise ValueError("CFG['deepdel_dataset']['num_shards'] must be > 0")
    if shard_index < 0 or shard_index >= n_shards:
        raise RuntimeError(f"Dataset shard index {shard_index} is outside range 0..{n_shards - 1}")
    base = total // n_shards
    rem = total % n_shards
    return int(base + (1 if shard_index < rem else 0))


def _build_generate_deepdel_dataset_cmd(
    *,
    out_csv: Path,
    num_triplets: int,
    threshold: Optional[float] = None,
    threshold_min: Optional[float] = None,
    threshold_max: Optional[float] = None,
    shard_index: Optional[int] = None,
) -> str:
    write_batch_size = int(
        _cfg_get("deepdel_dataset.write_batch_size", min(50, int(num_triplets)))
        or min(50, int(num_triplets))
    )
    default_batch_pred_size = 512 if _deepdel_dataset_device_flag().strip() == "--device cpu" else 8192
    batch_pred_size = int(
        _cfg_get("deepdel_dataset.batch_pred_size", default_batch_pred_size) or default_batch_pred_size
    )
    torch_num_threads = int(_cfg_get("deepdel_dataset.torch_num_threads", 1) or 0)
    torch_interop_threads = int(_cfg_get("deepdel_dataset.torch_interop_threads", 1) or 0)
    score_log_seconds = float(_cfg_get("deepdel_dataset.score_log_seconds", 30.0) or 0.0)
    bad_builds_csv = Path(out_csv).with_suffix(".bad_builds.csv")
    return (
        "python -u -m deepdelgfn.deepdel.generate_dataset"
        + BBS_FLAGS
        + f" --autodock-model {_autodock_model_path()}"
        + _deepdel_dataset_device_flag()
        + f" --n-threads {int(_cfg_get('deepdel_dataset.n_threads', CPUS) or CPUS)}"
        + f" --chunksize {int(_cfg_get('deepdel_dataset.chunksize', 1) or 1)}"
        + f" --write-batch-size {write_batch_size}"
        + f" --batch-pred-size {batch_pred_size}"
        + f" --torch-num-threads {torch_num_threads}"
        + f" --torch-interop-threads {torch_interop_threads}"
        + f" --score-log-seconds {score_log_seconds}"
        + f" --slow-triplet-log-seconds {float(_cfg_get('deepdel_dataset.slow_triplet_log_seconds', 30.0) or 0.0)}"
        + f" --reward {_cfg_get('deepdel_dataset.reward', 'threshold')}"
        + f" --min-size1 {CFG['lib_size']} --min-size2 {CFG['lib_size']} --min-size3 {CFG['lib_size']}"
        + f" --max-size1 {CFG['lib_size']} --max-size2 {CFG['lib_size']} --max-size3 {CFG['lib_size']}"
        + f" --num-triplets {int(num_triplets)}"
        + f" --out {shlex.quote(str(out_csv))}"
        + f" --bad-builds-csv {shlex.quote(str(bad_builds_csv))}"
        + (
            f" --threshold-min {threshold_min} --threshold-max {threshold_max}"
            if threshold_min is not None and threshold_max is not None
            else f" --threshold {threshold}"
        )
        + f" --alpha {float(CFG.get('threshold_alpha', 1.0))}"
        + f" --reaction-mode {shlex.quote(_reaction_mode())}"
        + _reward_weight_flags()
        + (" --log-target" if bool(CFG.get("log_reward_target", True)) else "")
        + _deepdel_dataset_seed_flag(shard_index=shard_index)
    )


# ---------------------------------------------------------------------------
# Proxy stage
# ---------------------------------------------------------------------------


def prepare_proxy_outer_loop(outer_loop: int) -> None:
    """Train or stage the autodock proxy artifact for one outer loop."""
    if not bool(CFG["inner_loop_only"]):
        if outer_loop == 0 and not bool(CFG["train_autodock_model_on_first_pass"]):
            dst = _autodock_model_path()
            kind = _autodock_kind()
            configured = _cfg_get("autodock_proxy.pretrained_model")
            fallback = (
                Path(str(configured)).expanduser()
                if configured
                else (FALLBACK_AUTODOCK_MODEL_PT if kind == "nn" else FALLBACK_AUTODOCK_MODEL_JOBLIB)
            )
            if not fallback.is_absolute():
                fallback = PROJECT_ROOT / fallback
            if not fallback.exists():
                raise FileNotFoundError(f"Fallback artifact not found at {fallback}")
            shutil.copy2(fallback, dst)
            print(f"[prepare] Using pretrained autodock model: {fallback} -> {dst}")
        else:
            if run_cmd(_build_train_autodock_cmd()) != 0:
                raise RuntimeError("Failed training autodock model.")


# ---------------------------------------------------------------------------
# Dataset stage
# ---------------------------------------------------------------------------


def prepare_dataset_outer_loop(outer_loop: int) -> None:
    """Generate or stage the initial DeepDEL dataset for one outer loop."""
    _, deepdel_dataset, threshold_min, threshold_max = _prepare_paths_and_threshold_interval(outer_loop)

    if not bool(CFG["inner_loop_only"]):
        if outer_loop == 0 and bool(CFG["generate_deepdel_dataset_on_first_pass"]) or outer_loop != 0:
            if deepdel_dataset.exists():
                deepdel_dataset.unlink()
            cmd = _build_generate_deepdel_dataset_cmd(
                out_csv=deepdel_dataset,
                num_triplets=int(CFG["deepdel_dataset_initial_size"]),
                threshold_min=threshold_min,
                threshold_max=threshold_max,
            )
            if run_cmd(cmd) != 0:
                raise RuntimeError("Failed generating DeepDel dataset.")
        elif outer_loop == 0:
            initial_dataset = _cfg_get("deepdel_dataset.initial_dataset")
            if not initial_dataset:
                raise RuntimeError(
                    "generate_deepdel_dataset_on_first_pass=False requires "
                    "CFG['deepdel_dataset']['initial_dataset'] or "
                    "scripts/active_learning_submit.py --initial-deepdel-dataset."
                )
            src = Path(str(initial_dataset)).expanduser()
            if not src.is_absolute():
                src = PROJECT_ROOT / src
            if not src.is_file():
                raise FileNotFoundError(f"Initial DeepDEL dataset not found: {src}")
            if deepdel_dataset.exists():
                deepdel_dataset.unlink()
            shutil.copy2(src, deepdel_dataset)
            print(f"[prepare] Using initial DeepDEL dataset: {src} -> {deepdel_dataset}")


def prepare_dataset_batch_outer_loop(outer_loop: int) -> None:
    """Generate one DeepDEL dataset shard. Intended for Slurm array tasks."""
    idx_raw = os.environ.get("PREPARE_DATASET_BATCH_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx_raw in (None, ""):
        raise RuntimeError(
            "PREPARE_DATASET_BATCH_INDEX or SLURM_ARRAY_TASK_ID is required for "
            "STAGE=prepare_dataset_batch"
        )
    shard_index = int(idx_raw)
    _, _, threshold_min, threshold_max = _prepare_paths_and_threshold_interval(outer_loop)
    shard_dir = _prepare_dataset_shard_dir(outer_loop)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_csv = _prepare_dataset_shard_path(outer_loop, shard_index)
    num_triplets = _prepare_dataset_shard_triplets(shard_index)

    if shard_csv.exists():
        shard_csv.unlink()
    print(
        f"[prepare-dataset-batch] outer={outer_loop} shard={shard_index} "
        f"num_triplets={num_triplets} out={shard_csv}"
    )
    cmd = _build_generate_deepdel_dataset_cmd(
        out_csv=shard_csv,
        num_triplets=num_triplets,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        shard_index=shard_index,
    )
    if run_cmd(cmd) != 0:
        raise RuntimeError(f"Failed generating DeepDEL dataset shard {shard_index}.")
    got_rows = _csv_data_row_count(shard_csv)
    if got_rows != num_triplets:
        raise RuntimeError(
            f"Dataset shard row count mismatch for shard {shard_index}: "
            f"expected {num_triplets}, got {got_rows} at {shard_csv}"
        )


def prepare_dataset_finalize_outer_loop(outer_loop: int) -> None:
    """Concatenate split DeepDEL dataset shards into the canonical dataset CSV."""
    _, deepdel_dataset, _ = _prepare_paths_and_threshold(outer_loop)
    n_shards = int(_cfg_get("deepdel_dataset.num_shards", 1))
    expected_total = int(CFG["deepdel_dataset_initial_size"])
    if n_shards <= 0:
        raise ValueError("CFG['deepdel_dataset']['num_shards'] must be > 0")

    shard_paths = [_prepare_dataset_shard_path(outer_loop, i) for i in range(n_shards)]
    expected_rows = [_prepare_dataset_shard_triplets(i) for i in range(n_shards)]
    for shard_path, expected in zip(shard_paths, expected_rows):
        if not shard_path.exists():
            raise FileNotFoundError(f"Expected dataset shard is missing: {shard_path}")
        got = _csv_data_row_count(shard_path)
        if got != expected:
            raise RuntimeError(
                f"Dataset shard row count mismatch: expected {expected}, got {got} at {shard_path}"
            )

    if deepdel_dataset.exists():
        deepdel_dataset.unlink()
    combined_rows = _concat_csvs_with_matching_headers(shard_paths, deepdel_dataset)
    if combined_rows != expected_total:
        raise RuntimeError(
            f"Combined DeepDEL dataset row count mismatch: expected {expected_total}, "
            f"got {combined_rows} at {deepdel_dataset}"
        )
    print(
        f"[prepare-dataset-finalize] Combined {n_shards} shard(s) "
        f"into {deepdel_dataset} ({combined_rows} rows)."
    )


# ---------------------------------------------------------------------------
# DeepDEL model stage
# ---------------------------------------------------------------------------


def dataset_row_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    try:
        df = pd.read_csv(path)
        if df.empty:
            return 0, 0
        total = int(len(df))
        if "y" not in df.columns:
            return total, 0
        valid = int(pd.to_numeric(df["y"], errors="coerce").notna().sum())
        return total, valid
    except Exception:
        return 0, 0


def stage_initial_deepdel_model(outer_loop: int) -> Path:
    """Copy a pretrained DeepDEL checkpoint into the run-local model path."""
    if int(outer_loop) != 0:
        raise ValueError("Only outer_loop=0 can stage the initial pretrained DeepDEL model.")
    _, _, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    src = _pretrained_deepdel_source()
    if deepdel_model_last.exists():
        deepdel_model_last.unlink()
    shutil.copy2(src, deepdel_model_last)
    print(f"[prepare] Using pretrained DeepDEL model: {src} -> {deepdel_model_last}")
    return deepdel_model_last


def _build_initial_deepdel_train_cmd(deepdel_dataset: Path, deepdel_model_last: Path) -> str:
    pooling = str(_cfg_get("deepdel_offline.model.pooling", "mean")).lower()
    output_head = str(_cfg_get("deepdel_offline.model.output_head", "linear")).lower()
    lib_size = int(CFG.get("lib_size", 0) or 0)
    return (
        "python -u -m deepdelgfn.deepdel.train_offline"
        + BBS_FLAGS
        + f" --bb-fp-bits {int(_cfg_get('deepdel_offline.model.bb_fp_bits', 2048))}"
        + f" --bb-fp-radius {int(_cfg_get('deepdel_offline.model.bb_fp_radius', 2))}"
        + f" --lr {float(_cfg_get('deepdel_offline.train.lr', 5e-5))}"
        + f" --weight-decay {float(_cfg_get('deepdel_offline.train.weight_decay', 1e-5))}"
        + f" --loss {shlex.quote(str(_cfg_get('deepdel_offline.train.loss', 'mse')))}"
        + f" --epochs {int(_cfg_get('deepdel_offline.train.epochs', 20))}"
        + f" --patience {int(_cfg_get('deepdel_offline.train.patience', 0))}"
        + f" --validation-type {shlex.quote(str(_cfg_get('deepdel_offline.train.validation_type', 'random')))}"
        + f" --val-frac {float(_cfg_get('deepdel_offline.train.val_frac', 0.2))}"
        + f" --val-bins {int(_cfg_get('deepdel_offline.train.val_bins', 10))}"
        + f" --batch-size {int(_cfg_get('deepdel_offline.train.batch_size', 128))}"
        + f" --num-workers {int(_cfg_get('deepdel_offline.train.num_workers', 8))}"
        + f" --prefetch-factor {int(_cfg_get('deepdel_offline.train.prefetch_factor', 1))}"
        + (" --persistent-workers" if bool(_cfg_get("deepdel_offline.train.persistent_workers", False)) else "")
        + (" --amp" if bool(_cfg_get("deepdel_offline.train.amp", True)) else "")
        + (" --tf32" if bool(_cfg_get("deepdel_offline.train.tf32", True)) else "")
        + f" --max-workers {int(_cfg_get('deepdel_offline.train.max_workers', 32))}"
        # DeepDEL TripleDeepSet architecture (must match for any later finetune resume).
        + f" --hidden-dim {int(_cfg_get('deepdel_offline.model.hidden_dim', 512))}"
        + f" --rho-dim {int(_cfg_get('deepdel_offline.model.rho_dim', 512))}"
        + f" --dropout {float(_cfg_get('deepdel_offline.model.dropout', 0.1))}"
        + f" --pooling {shlex.quote(pooling)}"
        + (" --shared-phi" if bool(_cfg_get("deepdel_offline.model.shared_phi", True)) else "")
        + f" --output-head {shlex.quote(output_head)}"
        + (f" --lib-size {lib_size}" if output_head != "linear" and lib_size > 0 else "")
        + _fourier_flags("deepdel_offline.model")
        + (
            " --append-molecular-weight"
            if bool(_cfg_get("deepdel_offline.model.append_molecular_weight", False))
            else ""
        )
        + (" --log-target" if bool(CFG.get("log_reward_target", True)) else "")
        + f" --dataset {shlex.quote(str(deepdel_dataset))}"
        + f" --save-last {shlex.quote(str(deepdel_model_last))}"
        + _deepdel_eval_dataset_flag()
        + f" --stats-json {shlex.quote(str(deepdel_model_last.parent / 'initial_deepdel_stats.json'))}"
    )


def train_initial_deepdel(outer_loop: int) -> Path:
    """Train the initial DeepDEL checkpoint for an outer loop during prepare."""
    _, deepdel_dataset, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    total_rows, valid_rows = dataset_row_counts(deepdel_dataset)
    print(f"[DeepDel dataset] path={deepdel_dataset} rows={total_rows} valid_y={valid_rows}")
    if valid_rows <= 0:
        raise RuntimeError(f"DeepDEL dataset has no valid y rows: {deepdel_dataset}")
    if run_cmd(_build_initial_deepdel_train_cmd(deepdel_dataset, deepdel_model_last)) != 0:
        raise RuntimeError("Failure when training initial DeepDel checkpoint during prepare.")
    return deepdel_model_last


def _build_eval_only_deepdel_cmd(
    deepdel_dataset: Path, deepdel_model_last: Path, stats_json: Path
) -> str:
    """Build an eval-only command to score a pretrained checkpoint on the external set."""
    eval_csv = _deepdel_validation_csv()
    loss = _cfg_get("deepdel_offline.train.loss", "mse")
    return (
        "python -u -m deepdelgfn.deepdel.train_offline"
        + BBS_FLAGS
        + _deepdel_model_arch_flags()
        + f" --loss {shlex.quote(str(loss))}"
        + f" --dataset {shlex.quote(str(deepdel_dataset))}"
        + f" --resume {shlex.quote(str(deepdel_model_last))}"
        + f" --eval-dataset {shlex.quote(str(eval_csv))}"
        + " --eval-only"
        + f" --stats-json {shlex.quote(str(stats_json))}"
    )


def _eval_initial_deepdel_if_configured(outer_loop: int, stats_json: Path) -> None:
    """Score a staged pretrained DeepDEL checkpoint on the external validation set."""
    if _deepdel_validation_csv() is None:
        print("[dd-init] No external validation CSV configured; skipping step-0 external eval.")
        return
    _, deepdel_dataset, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    if run_cmd(_build_eval_only_deepdel_cmd(deepdel_dataset, deepdel_model_last, stats_json)) != 0:
        raise RuntimeError(
            "Failure when evaluating the initial DeepDEL checkpoint on the "
            "external validation set."
        )


def prepare_deepdel_outer_loop(outer_loop: int) -> None:
    """Train the initial DeepDEL checkpoint for one outer loop."""
    _, _, deepdel_model_last = _outer_deepdel_paths(outer_loop)
    stats_json = deepdel_model_last.parent / "initial_deepdel_stats.json"
    if outer_loop == 0 and not bool(CFG.get("train_deepdel_model_on_first_pass", True)):
        stage_initial_deepdel_model(outer_loop)
        _eval_initial_deepdel_if_configured(outer_loop, stats_json)
    else:
        train_initial_deepdel(outer_loop)
    if outer_loop == 0:
        record_deepdel_val_mse(
            stats_json, step=0, phase="dd-init", outer_loop=outer_loop, inner_loop=None
        )


def prepare_outer_loop(outer_loop: int):
    """Backward-compatible monolithic prepare stage.

    New submissions use prepare_proxy -> prepare_dataset -> prepare_deepdel so
    Slurm does not have to allocate many CPUs and a GPU to the same job.
    """
    prepare_proxy_outer_loop(outer_loop)
    prepare_dataset_outer_loop(outer_loop)
    prepare_deepdel_outer_loop(outer_loop)