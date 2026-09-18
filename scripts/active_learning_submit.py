#!/usr/bin/env python3
"""
DEL-GFN-2 active learning launcher.

This script submits an active-learning run on Slurm from a user-provided JSON
configuration file. The submitter applies any CLI overrides, writes the resolved
configuration to RUN_ROOT/config.json, and each per-stage worker consumes that
run-local JSON via `src/deepdelgfn/pipelines/active_learning_stage.py`.

Active learning loop (one outer loop = prepare → branched inner/dock DAG):

    prepare → inner_train_i → eval_topm_i → deepdel_update_i ─┬─→ inner_train_{i+1}
                                                              └─→ dock_prepare_i
                                                                  → dock_launch_i
                                                                  → dock_batch_i[]
                                                                  → dock_finalize_i

Docking is submitted after every inner-loop DeepDEL update, but the next GFN
training step depends only on that update job, not on docking.  The next outer
loop's prepare stage still waits for all per-inner docking launchers from the
previous outer loop so proxy retraining can keep the previous scored-library
semantics.

Model dependencies / dimension chain (important when tuning architecture):

    ┌──────────────┐ φ(.) (hidden_dim)        ┌──────────────────────────┐
    │   DeepDEL    │ ─── precomputed ───────► │ GFN env state embedding  │
    │ TripleDeepSet│   φ tables [N, d_h]      │   3·d_h  (= 3·hidden_dim)│
    └──────────────┘                          └──────────────────────────┘
                                                       │
                                                       ▼
                                         ┌────────────────────────────┐
                                         │ JointEdgePolicy            │
                                         │  state_proj: 3·d_h → state │
                                         │              dim = state_dim
                                         │  action tower: 3·D    →    │
                                         │              action_dim    │
                                         │  (D = bb_fp_bits for ECFP, │
                                         │   D = d_h for φ actions)   │
                                         │  joint head: …→ joint_dim  │
                                         └────────────────────────────┘

"""
import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


# ------------------------ Legacy/default template ------------------------
# Runtime settings are loaded from --config. This in-code template is retained
# only as a Python-side reference for exporting/updating the default JSON config
# and should not be edited as the active run configuration.
CFG = {
    "lib_size": 20,
    # Global DEL chemistry mode used by every product-enumeration path:
    #   "amide_sulfonamide" -> AmpC-style B3 sulfonyl chloride coupling
    #   "amide_amide"       -> sEH-style B3 amino-acid amide coupling
    "reaction_mode": "amide_sulfonamide",
    "num_inner_loops": 12,
    "num_outer_loops": 1,
    # Optional linear schedule for main GFN training steps over inner loops.
    # For inner_loop i in [0, num_inner_loops-1], active_learning_stage.py uses:
    #   start + i/(num_inner_loops-1) * (end-start)
    # Keeping start == end preserves the legacy constant-step behavior.
    "gfn_train_steps_start": 50000,
    "gfn_train_steps_end": 50000,
    "gfn_train_batch": 1,
    "topm": 10000,
    # Terminal reward source used by GFN Trajectory Balance.
    #   "deepdel"        ->  DeepDEL regression head supplies log R / R.
    #   "autodock_proxy" ->  DeepDEL still supplies φ state embeddings, but terminal
    #                        library rewards are computed by enumerating products and
    #                        scoring them with the autodock proxy; every encountered
    #                        finite log reward is appended to the DeepDEL dataset.
    # Reward names accepted by DeepDEL/autodock-proxy paths include:
    #   "threshold", "topk_mean", "mean", "ampc_pki6_hits", "ampc_pki6_proportion".
    # For AmpC, "ampc_pki6_hits" predicts X*N, the expected number of products
    # with pKi>=6 (Ki<=1 µM), after clamping docking scores below -120 to -120.
    "gfn_reward_source": "deepdel",
    "autodock_proxy_reward": {
        "reward": "threshold",
        "batch_pred_size": 4096,
        "device": "cuda",
        "append_encountered_to_deepdel": True,
        "save_encountered_csv": True,
    },
    # Docking candidate selection from each inner loop's top-m list.
    #   mode="top"     -> dock the top n_to_dock_per_inner rows by autodock_proxy_value.
    #   mode="leaders" -> score-prioritized diverse leaders in DeepDEL φ-space
    #                     using cosine similarity and a quick binary search over
    #                     the leader threshold to obtain n_leaders_to_dock picks.
    "docking_selection": {
        "mode": "top",
        "n_to_dock_per_inner": 15,
        "n_leaders_to_dock": 15,
        "leader_similarity": "cosine",
        "leader_binary_search_iters": 24,
    },
    # Legacy fallback for old configs / scripts. Prefer CFG["docking_selection"].
    "topn_to_dock_per_inner": 5,
    "inner_loop_only": False,
    # First prepare-stage proxy handling:
    #   True  -> train a fresh autodock proxy before generating the initial DeepDEL dataset.
    #   False -> copy a pretrained proxy into RUN_ROOT/models and immediately generate the
    #            DeepDEL dataset from it. Configure the source artifact below at
    #            CFG["autodock_proxy"]["pretrained_model"].
    "train_autodock_model_on_first_pass": True,
    "generate_deepdel_dataset_on_first_pass": True,
    # First prepare-stage DeepDEL handling:
    #   True  -> train a fresh initial DeepDEL checkpoint from the first-pass dataset.
    #   False -> copy a pretrained DeepDEL checkpoint into RUN_ROOT/outer_0/models
    #            and begin active learning with the first GFlowNet training pass.
    #            Configure the source artifact at CFG["deepdel_offline"]["pretrained_model"].
    "train_deepdel_model_on_first_pass": True,
    "sleep_between_cycles": 0,
    "docking_out_dir": "data/scored_libraries/active_learning",
    "docking": {
        "backend": "dock3",
        # Active-learning DOCK3 can be much longer than the training stages.  When
        # enabled, the submitter schedules a short dock-prepare job followed by a
        # dock-launcher job.  The launcher reads the generated docking manifest and
        # submits a Slurm array with one 3-hour/64-CPU task per shard.
        "split_active_learning_jobs": True,
        "batch_rows": 5000,
        # The launcher itself is a tiny 1-CPU bookkeeping job.  It submits the
        # shard array and waits for the finalizer so outer-loop dependencies can
        # target one Slurm job.  Its walltime should cover queue wait + shard run.
        "launch_time": "24:00:00",
        # Python package path for omltk.docking used by pooled DOCK3 docking.
        "local_omltk": "/project/rrg-mailhoto/share/dockingpackages/omlab_toolkit/src",
        # Outer subprocess timeout used by Dock3Scorer for ligbuild and dock64.
        "dock3_timeout": 900,
        # Inner DB2/protomer timeout written into ligbuild custom_parms.json.
        "ligbuild_timeout": 600,
    },
    "threshold_value": -80,
    # Smoothing parameter for the 'threshold' reward:
    #   reward = sum( 1 / (1 + exp((value - threshold) / threshold_alpha)) )
    # Larger alpha -> smoother (softer) sigmoid; alpha=1.0 reproduces the legacy reward.
    "threshold_alpha": 10.0,
    # Optional molecular-weight cutoff for threshold rewards. If None, all
    # products contribute. If set, products with molecular weight > cutoff
    # contribute score 0. `threshold_weight_source` controls how weights are
    # computed: "bb_sum" approximates product weight by summing BB weights;
    # "smiles" computes exact product weight from the generated product SMILES.
    "threshold_max_weight": 500,
    "threshold_weight_source": "bb_sum",
    # When True, every script that produces or consumes the DeepDEL target
    # operates on log R(x) instead of R(x):
    #   - generate_dataset.py     : writes y = log R
    #   - train_offline.py        : persists --log-target into ckpt args
    #   - eval_autodock_proxy_topm_scores.py: appends y = log(autodock_proxy_value)
    #   - train_gfn.py            : interprets the regression output as log R
    #                               directly (TB never has to take log() of a
    #                               possibly non-positive R prediction).
    # This is the recommended setting -- it's the only configuration that
    # eliminates the TB-loss-NaN failure mode triggered when DeepDEL's R
    # prediction collapses to <= 0.
    "log_reward_target": True,
    "deepdel_dataset_initial_size": 200000,



    # ---------------- Pipeline command parameters (exposed) ----------------
    # These are the previously hard-coded CLI args in active_learning_cpu.py / active_learning_gpu.py.
    # They are persisted into RUN_ROOT/config.json and consumed by the stage scripts.

    "paths": {
        # Common inputs.
        # AmpC uses two amino-acid pools (B1, B2) and one sulfonyl-chloride pool (B3).
        # Each consumer script (train_gfn / generate_dataset / train_offline /
        # eval_autodock_proxy_topm_scores) accepts --bbs1 / --bbs2 / --bbs3 to mirror this.
        "bbs1_csv": "data/bbs_aminoacids_purged.csv",
        "bbs2_csv": "data/bbs_aminoacids_purged.csv",
        "bbs3_csv": "data/bbs_sulfonylchlorides_purged.csv",
        # Autodock proxy training inputs (space-separated paths passed after --inputs).
        # Train scripts recursively scan the listed directories for *.csv files.
        "autodock_proxy_inputs": [
            "data/scored_libraries",
        ],
    },

    "autodock_proxy": {
        # kind: "rf" (train_autodock_proxy.py -> .joblib) or "nn" (train_autodock_proxy_nn.py -> .pt)
        "kind": "nn",
        # Optional pretrained proxy artifact used when
        # CFG["train_autodock_model_on_first_pass"] is False.
        # If None, the stage worker falls back to models/autodock_model.pt for kind="nn"
        # or models/autodock_model.joblib for kind="rf".
        # Relative paths are resolved from the project root by active_learning_stage.py.
        "pretrained_model": "models/autodock_model.pt",
        # Clip positive docking targets to zero for proxy training. The NN
        # ECFP cache retains raw targets and remains reusable.
        "clip_positive_targets": False,
        "n_estimators": 500,
        # NN hyperparams (used when kind == "nn")
        "nn": {
            "n_bits": 16384, # for testing: 1024
            "radius": 2,
            "hidden_dim": 1024,
            "n_layers": 3,
            "dropout": 0.5,
            "lr": 5e-6,
            "weight_decay": 2e-3,
            "batch_size": 1024,
            "epochs": 60,
            "patience": 5,
            "test_size": 0.01,
            "val_size": 0.1,
            "standardize_y": True,
            "dedup_smiles": True,
            # ECFP cache: when set, train_nn.py loads precomputed fingerprints
            # from this .npz file instead of recomputing them every outer loop.
            # The cache is auto-invalidated if the source CSVs (paths/sizes/mtimes)
            # or featurization config (radius, n_bits, columns, dedup/filter
            # flags) change. Set to None to disable caching.
            "ecfp_cache": "data/ecfp_cache/ecfp_r2_n16384.npz",
            "overwrite_ecfp_cache": False,
        },
    },

    "deepdel_dataset": {
        # Optional existing DeepDEL dataset CSV used for outer loop 0 when
        # generate_deepdel_dataset_on_first_pass=False. Relative paths are
        # resolved from the project root by this submitter, then
        # copied into RUN_ROOT/outer_0/deepdel/deepdel_dataset.csv so the rest
        # of the pipeline can use the standard run-local path.
        "initial_dataset": "models/deepdel_dataset.csv",
        "reward": "threshold",
        # n_threads is set dynamically from SLURM_CPUS_PER_TASK in the CPU script unless overridden.
        # If set to null, the CPU script uses SLURM_CPUS_PER_TASK.
        # Keep this below the Slurm CPU request for CPU-only NN proxy scoring:
        # product/fingerprint workers run in a multiprocessing pool while the
        # main process does PyTorch inference.  Leaving room for the main
        # process avoids oversubscription and priority-wasting idle cores.
        "n_threads": 8,
        "batch_pred_size": 256,
        "write_batch_size": 10,
        "torch_num_threads": 1,
        "torch_interop_threads": 1,
        "score_log_seconds": 30.0,
        "slow_triplet_log_seconds": 30.0,
        # Device used by generate_dataset.py for autodock-proxy inference.
        # "cpu" avoids mixing a GPU with many CPUs in the default prepare_dataset
        # job. Set to "cuda" and request --prepare-dataset-gpus-per-node to use
        # GPU batched inference for this stage; "auto" uses CUDA when visible.
        "device": "auto",
        # Split initial DeepDEL dataset generation into a Slurm array of shorter
        # independent shard jobs, then concatenate the shard CSVs before initial
        # DeepDEL training. Disable to use the legacy single prepare_dataset job.
        "split_jobs": True,
        "num_shards": 50,
        # Base RNG seed for split shards. None preserves the legacy unseeded
        # generate_dataset.py behavior; an integer gives reproducible, distinct
        # per-shard streams via seed + shard_index.
        "seed": None,
    },

    "gfn": {
        # GFN-specific knobs. The BB CSVs come from CFG["paths"]["bbs{1,2,3}_csv"]
        # above and are forwarded by active_learning_stage.py as --bbs1/--bbs2/--bbs3.

        # ---- Policy architecture ----
        # MUST match between LogZ warmup and main training (the latter resumes
        # the policy from --policy-ckpt of the former), so it lives outside
        # `logz`/`train`. The φ output dim `d_h` consumed by the policy is
        # inherited automatically from the DeepDEL checkpoint args
        # (= CFG["deepdel_offline"]["model"]["hidden_dim"]); there is no GFN
        # flag to set it.
        "model": {
            "joint_dim": 4096,     # --joint-dim (joint head width)
            "state_dim": 4096,    # --state-dim (state tower projection: 3*d_h -> state_dim)
            "action_dim": 2048,    # --action-dim (action tower hidden width)
            "action_repr": "phi", # --action-repr: "ecfp" -> (ecfp,0,0), "phi" -> (phi1,0,0)/(0,phi2,0)/(0,0,phi3)
            "phi_train_mode": "trainable_network",  # --phi-train-mode: "frozen_table", "trainable_table", or "trainable_network"
            "bb_fp_bits": 2048,   # --bb-fp-bits (raw action ECFP width D when action_repr="ecfp")
            "bb_fp_radius": 2,    # --bb-fp-radius (Morgan radius for raw ECFPs and DeepDEL phi inputs)
        },

        # Optional per-cycle forbidden BB lists (pipe-delimited strings of combined IDs).
        # When set, train_gfn.py removes these IDs from the corresponding pool action space.
        "forbidden": {
            # Example:
            # "forbidden_bb1": "487|660|3180",
            # "forbidden_bb2": "3953|6447|6960|4149",
            # "forbidden_bb3": "10369|10533|13505" ,
        },

        # Main GFN training phase
        "train": {
            "logz_lr": 1,
            "lr": 3e-5,
            "batched_rollouts": True,
            "save_model": True,
            # Optimizer / exploration knobs (defaults from train_gfn.py)
            "epsilon": 0.05,
            "weight_decay": 0,
            "grad_clip": 1,
            "min_per_cycle": 8,
        },

        # Optional phase-augmented GFlowNet regularization:
        #   L = L_TB + lambda * mean(1 - cos(sum_t phi(a_t|s_t))).
        # Defaults preserve legacy TB-only training.
        "phase_regularization": {
            "enabled": False,
            "lambda": 2500.0**2/8.0,
        },
    },

    "deepdel_offline": {
        # Optional pretrained DeepDEL artifact used when
        # CFG["train_deepdel_model_on_first_pass"] is False. Relative paths are
        # resolved from the project root by the submitter/stage worker.
        "pretrained_model": "models/deepdel.pt",

        # DeepDEL TripleDeepSet architecture / featurization. These settings are
        # shared by the initial offline train and any post-GFN retrain/finetune.
        # NOTE: this bb_fp_bits is DeepDEL's BB input fingerprint width. It is
        # distinct from CFG["gfn"]["model"]["bb_fp_bits"], which only controls
        # the GFN policy's raw action ECFP width when action_repr="ecfp".
        # Values below mirror the best DeepDEL offline grid run in
        # outputs/deepdel_grid/20260513_111120/leaderboard.csv.
        "model": {
            "bb_fp_bits": 4096,   # DeepDEL input ECFP width (--bb-fp-bits)
            "bb_fp_radius": 2,    # DeepDEL input ECFP radius (--bb-fp-radius)
            "hidden_dim": 6144,   # phi hidden width  (--hidden-dim)
            "rho_dim": 6144,      # rho hidden width  (--rho-dim)
            "dropout": 0.0,       # --dropout
            "shared_phi": False,  # --shared-phi flag (single phi shared across the 3 BBs)
            # --pooling: "mean" preserves legacy DeepDEL behavior; "sum" matches
            # canonical DeepSets aggregation rho(sum_b phi(b)). This is stored
            # in the checkpoint and must match for finetuning/resume.
            "pooling": "sum",
            # Final activation applied to the raw rho logit. "linear" preserves
            # legacy behavior; "sigmoid_scaled" = log(1+k^3)*sigmoid(z) (bounded
            # in [0, log(1+k^3)]); "softplus_scaled" = log(1+k^3)*softplus(z)
            # (non-negative, unbounded above). Non-linear heads use CFG["lib_size"].
            "output_head": "linear",
        },


        # train_deepdel_offline.py parameters
        "train": {
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "epochs": 30,
            "patience": 4,
            "validation_type": "random",
            "val_frac": 0.2,
            "val_bins": 10,
            "batch_size": 256, #1024,#256,
            "num_workers": -1,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "amp": True,
            "tf32": True,
            "max_workers": 32,
        },
        # What to do after each GFN run has appended/evaluated new rows in the
        # DeepDEL dataset. mode="retrain" trains a fresh model on the entire
        # augmented dataset after every inner loop; mode="finetune" keeps the
        # legacy checkpoint-resume behavior using the "finetune" block below.
        "after_gfn": {
            "mode": "retrain",
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "epochs": 30,
            "patience": 4,
            "validation_type": "random",
            "val_frac": 0.2,
            "val_bins": 10,
            "batch_size": 256,#1024,
            "num_workers": -1,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "amp": True,
            "tf32": True,
            "max_workers": 32,
        },
        "finetune": {
            "lr": 1e-5,
            "weight_decay": 1e-5,
            "epochs": 3,
            "patience": 0,
            "resume_optimizer": True,
            "batch_size": 1024,
            "num_workers": -1,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "amp": True,
            "tf32": True,
        },
    },

    "eval_topm": {
        "reward": "threshold",
        "batch_pred_size": 4096,
        "batch_rows": 2000000,
        "num_workers": -1,
        "device": "auto",
        "split_jobs": True,
        "num_shards": 40,
        "torch_num_threads": 1,
        "torch_interop_threads": 1,
        "score_log_seconds": 30.0,
    },

    # GFlowNet training hyperparams (passed to train_deepdel_gfn.py)
    # Note: train_deepdel_gfn.py expects --beta, --subsample-ratio-start, --subsample-ratio-end.
    "beta": 2000.0,
    "train_gfn_subsample_ratio_start": 0.02,
    "train_gfn_subsample_ratio_end": 0.02,

    # Default Slurm resource requests for each stage. These are persisted to
    # RUN_ROOT/config.json and translated to sbatch flags in `_build_sbatch_cmd`.
    # Per-stage CLI overrides exposed by `_build_parser` take precedence
    # (e.g. `--prepare-mem 0`, `--train-gpus-per-node a100:1`).
    #
    # Slurm resource requests, per stage. Recognised keys:
    #   - "job-name"        → --job-name
    #   - "time"            → --time
    #   - "ntasks_per_node" → --ntasks-per-node      (int, optional; defaults to 1)
    #   - "cpus"            → --cpus-per-task
    #   - "mem"             → --mem                  (e.g. "120G", "32G", or "0" for all node RAM)
    #   - "gpus_per_node"   → --gpus-per-node        (e.g. "a100:4", "a100_3g.20gb:1"; None = no GPU)
    #   - "account"         → --account
    "slurm": {
        "prepare_proxy": {
            "job-name": "dd-proxy", "time": "3:00:00",
            "ntasks_per_node": 1, "cpus": 1, "mem": "120G",
            "gpus_per_node": "a100_3g.20gb:1", "account": "def-yvesbrun",
        },
        "prepare_dataset": {
            "job-name": "dd-dataset", "time": "12:00:00",
            "ntasks_per_node": 1, "cpus": 8, "mem": "8G",
            "gpus_per_node": "a100_1g.5gb:1", "account": "def-yvesbrun",
        },
        "prepare_deepdel": {
            "job-name": "dd-init", "time": "3:00:00",
            "ntasks_per_node": 1, "cpus": 8, "mem": "32G",
            "gpus_per_node": "a100_3g.20gb:1", "account": "def-yvesbrun",
        },
        # Legacy monolithic prepare profiles retained for backward-compatible
        # manual STAGE=prepare submissions and as fallbacks for older configs.
        "prepare_cpu": {
            "job-name": "dd-prepare", "time": "3:00:00",
            "ntasks_per_node": 1, "cpus": 64, "mem": "120G",
            "gpus_per_node": None, "account": "def-yvesbrun",
        },
        # Used when autodock_proxy.kind == "nn" (proxy NN training + dataset gen
        # both want a GPU). Currently configured for a full Narval A100 node.
        "prepare_gpu": {
            "job-name": "deepdel", "time": "3:00:00",
            "ntasks_per_node": 1, "cpus": 48, "mem": "256G",
            "gpus_per_node": "a100:1", "account": "def-yvesbrun",
        },
        "train": {
            # NOTE: each inner stage runs `deepdelgfn.gfn.train_gfn` and then
            # finetunes DeepDEL. Both are single-process,
            # single-GPU scripts -- they place every model/tensor on `cuda:0`
            # and have no DataParallel/DDP. Allocating multiple GPUs here just
            # wastes the extra ones. If you ever DDP-ify `train_gfn`, bump
            # this back to e.g. "a100:4" and switch the launcher to torchrun.
            "job-name": "deepdel-gfn", "time": "12:00:00",
            "ntasks_per_node": 1, "cpus": 1, "mem": "32G",
            "gpus_per_node": "a100:1", "account": "def-yvesbrun",
        },
        "eval_topm": {
            "job-name": "dd-eval", "time": "3:00:00",
            "ntasks_per_node": 1, "cpus": 16, "mem": "16G",
            "gpus_per_node": "a100_1g.5gb:1", "account": "def-yvesbrun",
        },
        "eval_topm_finalize": {
            "job-name": "dd-eval-fin", "time": "1:00:00",
            "ntasks_per_node": 1, "cpus": 1, "mem": "8G",
            "gpus_per_node": None, "account": "def-yvesbrun",
        },
        "deepdel_update": {
            "job-name": "dd-update", "time": "3:00:00",
            "ntasks_per_node": 1, "cpus": 8, "mem": "64G",
            "gpus_per_node": "a100:1", "account": "def-yvesbrun",
        },

        "dock": {
            "job-name": "deepdel-dock", "time": "3:00:00",
            "ntasks_per_node": 1, "cpus": 64, "mem": "120G",
            "gpus_per_node": None, "account": "def-mailhoto",
        },
        "dock_finalize": {
            "job-name": "deepdel-dock-fin", "time": "1:00:00",
            "ntasks_per_node": 1, "cpus": 1, "mem": "8G",
            "gpus_per_node": None, "account": "def-mailhoto",
        },
    },
}


def _run_id() -> str:
    env_id = os.environ.get("DEL_GFN_RUN_ID")
    if env_id:
        return env_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _scratch_root() -> Path:
    return Path(os.environ.get("SCRATCH") or os.getcwd())


def _run_root(run_id: str) -> Path:
    return _scratch_root() / "del-gfn-2" / "outputs" / "runs" / run_id


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config_json(path: str) -> dict:
    config_path = Path(str(path)).expanduser()
    if not config_path.is_absolute():
        config_path = _project_root() / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Active learning config JSON not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Active learning config must be a JSON object: {config_path}")
    print(f"[config] loaded {config_path}")
    return cfg


def _resolve_existing_file(path: str, *, description: str) -> Path:
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = _project_root() / p
    if not p.is_file():
        raise FileNotFoundError(f"{description} not found: {p}")
    return p


def _autodock_kind() -> str:
    return str(CFG.get("autodock_proxy", {}).get("kind", "rf")).lower()


def _active_learning_target() -> Optional[str]:
    """Return the configured target using canonical receptor spelling."""
    target_names = {"clpp": "ClpP", "mpro": "Mpro", "seh": "sEH", "tblr1": "TBLR1"}
    configured = CFG.get("target", (CFG.get("docking", {}) or {}).get("target"))
    if configured is None:
        return None
    key = str(configured).strip().lower()
    if key not in target_names:
        raise ValueError(f"Unknown active-learning target {configured!r}")
    return target_names[key]


def _run_autodock_model_path(run_root: Path) -> Path:
    suffix = ".pt" if _autodock_kind() == "nn" else ".joblib"
    return run_root / "models" / f"autodock_model{suffix}"


def _run_deepdel_model_path(run_root: Path, *, outer_loop: int = 0) -> Path:
    return run_root / f"outer_{int(outer_loop)}" / "models" / "deepdel.pt"


def _pretrained_autodock_source() -> Path:
    proxy_cfg = CFG.get("autodock_proxy", {}) or {}
    configured = proxy_cfg.get("pretrained_model")
    if configured:
        return _resolve_existing_file(configured, description="Pretrained autodock proxy artifact")
    fallback = _project_root() / "models" / ("autodock_model.pt" if _autodock_kind() == "nn" else "autodock_model.joblib")
    if not fallback.is_file():
        raise FileNotFoundError(f"Fallback autodock proxy artifact not found: {fallback}")
    return fallback


def _pretrained_deepdel_source() -> Path:
    deepdel_cfg = CFG.get("deepdel_offline", {}) or {}
    configured = deepdel_cfg.get("pretrained_model")
    if configured:
        return _resolve_existing_file(configured, description="Pretrained DeepDEL artifact")
    fallback = _project_root() / "models" / "deepdel.pt"
    if not fallback.is_file():
        raise FileNotFoundError(f"Fallback DeepDEL artifact not found: {fallback}")
    return fallback


def _stage_initial_deepdel_dataset(run_root: Path, *, dry_run: bool = False) -> Path:
    """Stage an existing DeepDEL dataset into the run-local outer-0 path."""
    initial_dataset = (CFG.get("deepdel_dataset", {}) or {}).get("initial_dataset")
    if not initial_dataset:
        raise RuntimeError(
            "generate_deepdel_dataset_on_first_pass=False requires "
            "CFG['deepdel_dataset']['initial_dataset'] "
            "or --initial-deepdel-dataset."
        )
    dataset_src = _resolve_existing_file(initial_dataset, description="Initial DeepDEL dataset")
    dataset_dst = run_root / "outer_0" / "deepdel" / "deepdel_dataset.csv"
    if dry_run:
        print(f"[dry-run] Would stage initial DeepDEL dataset: {dataset_src} -> {dataset_dst}")
    else:
        dataset_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_src, dataset_dst)
        print(f"[run] Staged initial DeepDEL dataset: {dataset_src} -> {dataset_dst}")
    return dataset_dst


def _stage_deepdel_model_artifact(run_root: Path, *, dry_run: bool = False) -> Path:
    """Stage a pretrained DeepDEL checkpoint into the run-local outer-0 model path.

    This lets the split prepare DAG skip the first prepare_deepdel Slurm job when
    CFG['train_deepdel_model_on_first_pass'] is False. The first inner GFlowNet
    training pass consumes RUN_ROOT/outer_0/models/deepdel.pt as usual.
    """
    model_src = _pretrained_deepdel_source()
    model_dst = _run_deepdel_model_path(run_root, outer_loop=0)
    if dry_run:
        print(f"[dry-run] Would stage DeepDEL artifact: {model_src} -> {model_dst}")
    else:
        model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_src, model_dst)
        print(f"[run] Staged DeepDEL artifact: {model_src} -> {model_dst}")
    return model_dst


def _stage_autodock_proxy_artifact(run_root: Path, *, dry_run: bool = False) -> Path:
    """Stage a pretrained autodock proxy into the run-local model path.

    This lets the split prepare DAG skip the prepare_proxy Slurm job when
    CFG['train_autodock_model_on_first_pass'] is False: the submitter copies
    the already-trained artifact before submitting prepare_dataset.
    """
    model_src = _pretrained_autodock_source()
    model_dst = _run_autodock_model_path(run_root)
    if dry_run:
        print(f"[dry-run] Would stage autodock proxy artifact: {model_src} -> {model_dst}")
    else:
        model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_src, model_dst)
        print(f"[run] Staged autodock proxy artifact: {model_src} -> {model_dst}")
    return model_dst


def _build_combined_bbs_csv(
    *, bbs1_csv: str, bbs2_csv: str, bbs3_csv: str, out_path: Path
) -> Path:
    """Combine the three pool CSVs into one with globally unique IDs and a `pool` column.

    Each input must have a SMILES column (any of SMILES/Smiles/smiles). The output is
    written with columns `ID`, `SMILES`, `Name?`, `pool` where IDs are renumbered
    starting at 1 across all three pools so they remain globally unique. The `pool`
    column tags origin (1, 2, or 3) so consumers can split back into per-cycle pools.
    """
    def _load(path: str, pool_idx: int) -> tuple[list[dict[str, object]], bool]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"BB CSV (pool {pool_idx}) not found: {path}")

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            smiles_col = next((c for c in ("SMILES", "Smiles", "smiles") if c in columns), None)
            if smiles_col is None:
                raise ValueError(f"BB CSV {path} has no SMILES column.")
            name_col = next((c for c in ("Name", "name") if c in columns), None)

            rows: list[dict[str, object]] = []
            for row in reader:
                out_row: dict[str, object] = {
                    "SMILES": row.get(smiles_col, ""),
                    "pool": int(pool_idx),
                }
                if name_col is not None:
                    out_row["Name"] = row.get(name_col, "")
                rows.append(out_row)
        return rows, name_col is not None

    loaded = [_load(bbs1_csv, 1), _load(bbs2_csv, 2), _load(bbs3_csv, 3)]
    parts = [rows for rows, _has_name in loaded]
    include_name = any(has_name for _rows, has_name in loaded)
    combined = [row for rows in parts for row in rows]
    # Renumber IDs globally to keep them unique even if pool-2/3 share row indices with pool-1.
    for idx, row in enumerate(combined, start=1):
        row["ID"] = idx
        if include_name:
            row.setdefault("Name", "")

    fieldnames = ["ID", "SMILES"]
    if include_name:
        fieldnames.append("Name")
    fieldnames.append("pool")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    print(
        f"[run] Wrote combined BBs CSV: {out_path} "
        f"(pool sizes: {len(parts[0])}, {len(parts[1])}, {len(parts[2])}; total {len(combined)})"
    )
    return out_path


def run_cmd(cmd: str) -> int:
    print(f"\n→ Running: {cmd}")
    return subprocess.run(shlex.split(cmd), check=False).returncode


def _build_sbatch_cmd(
    job_script: str,
    *,
    job_name: Optional[str],
    run_id: str,
    config_path: Path,
    outer_loop: int,
    stage: str,
    inner_loop: Optional[int] = None,
    time_limit: Optional[str] = None,
    ntasks_per_node: Optional[int] = None,
    cpus_per_task: Optional[int] = None,
    mem: Optional[str] = None,
    gpus_per_node: Optional[str] = None,
    account: Optional[str] = None,
    dependency: Optional[str] = None,
    array: Optional[str] = None,
) -> list[str]:
    """Assemble the sbatch invocation for a stage. Slurm CLI flags override
    any matching `#SBATCH` directives in the job script.

    Note on memory: Slurm accepts `--mem=0` to mean "all memory on the node",
    so passing `mem="0"` is intentional for full-node jobs.
    """
    export_vars = (
        f"ALL,DEL_GFN_RUN_ID={run_id},RUN_ID={run_id},"
        f"OUTER_LOOP={outer_loop},STAGE={stage},DEL_GFN_CONFIG={config_path}"
    )
    if inner_loop is not None:
        export_vars += f",INNER_LOOP={int(inner_loop)}"
    cmd_parts: list[str] = ["sbatch", "--parsable", "--export", export_vars]
    if job_name:
        cmd_parts.extend(["--job-name", job_name])
    if account:
        cmd_parts.extend(["--account", account])
    if time_limit:
        cmd_parts.extend(["--time", time_limit])
    if ntasks_per_node:
        cmd_parts.extend(["--ntasks-per-node", str(int(ntasks_per_node))])
    if cpus_per_task:
        cmd_parts.extend(["--cpus-per-task", str(int(cpus_per_task))])
    # `mem` may legitimately be the string "0" (= all node RAM); the previous
    # `if mem:` form already keeps that, but be explicit for clarity.
    if mem is not None and str(mem) != "":
        cmd_parts.extend(["--mem", str(mem)])
    if gpus_per_node:
        # Narval accepts e.g. `--gpus-per-node=a100:4` or `--gpus-per-node=a100_3g.20gb:1`.
        cmd_parts.append(f"--gpus-per-node={gpus_per_node}")
    if dependency:
        cmd_parts.extend(["--dependency", f"afterok:{dependency}"])
    if array:
        cmd_parts.append(f"--array={array}")
    cmd_parts.append(job_script)
    return cmd_parts


def submit_job(
    job_script: str,
    *,
    run_id: str,
    job_name: Optional[str],
    config_path: Path,
    outer_loop: int,
    stage: str,
    inner_loop: Optional[int] = None,
    time_limit: Optional[str] = None,
    ntasks_per_node: Optional[int] = None,
    cpus_per_task: Optional[int] = None,
    mem: Optional[str] = None,
    gpus_per_node: Optional[str] = None,
    account: Optional[str] = None,
    dependency: Optional[str] = None,
    array: Optional[str] = None,
) -> str:
    cmd_parts = _build_sbatch_cmd(
        job_script,
        job_name=job_name,
        run_id=run_id, config_path=config_path, outer_loop=outer_loop, stage=stage,
        inner_loop=inner_loop,
        time_limit=time_limit, ntasks_per_node=ntasks_per_node,
        cpus_per_task=cpus_per_task, mem=mem,
        gpus_per_node=gpus_per_node, account=account, dependency=dependency,
        array=array,
    )

    inner_label = f", inner={inner_loop}" if inner_loop is not None else ""
    print(f"\n[submit] {stage} (outer={outer_loop}{inner_label}) -> {' '.join(cmd_parts)}")
    try:
        result = subprocess.run(cmd_parts, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Surface sbatch's own stdout/stderr so allocation/partition/account errors
        # don't get swallowed by `capture_output=True`.
        print(f"[submit][ERROR] sbatch exited with code {e.returncode}")
        if e.stdout:
            print("---- sbatch stdout ----")
            print(e.stdout.rstrip())
        if e.stderr:
            print("---- sbatch stderr ----")
            print(e.stderr.rstrip())
        raise
    job_id = result.stdout.strip().split(";")[0]
    print(f"[submit] job_id={job_id}")
    return job_id


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Submit the split CPU/GPU active learning pipeline to Slurm from a JSON config. "
            "This script writes the resolved run config JSON and then submits a branched active-learning DAG per outer loop: "
            "prepare_proxy -> prepare_dataset -> prepare_deepdel -> inner_i/deepdel_update_i, "
            "with a per-inner docking branch that does not block inner_{i+1}."
        )
    )
    p.add_argument(
        "--config",
        "-c",
        required=True,
        help=(
            "Path to the active-learning JSON config. The loaded config is the "
            "base run configuration; supported CLI flags then override it before "
            "RUN_ROOT/config.json is written. Example: configs/active_learning_default.json"
        ),
    )
    # Per-stage resource overrides (defaults come from CFG["slurm"], applied in main()).
    # Each stage exposes the full set of Slurm knobs we use:
    #   --<stage>-time / --<stage>-ntasks-per-node / --<stage>-cpus /
    #   --<stage>-mem / --<stage>-gpus-per-node / --<stage>-account.
    # `--<stage>-mem 0` is supported (means "all memory on the node" to Slurm).
    stage_specs = (
        ("prepare", "prepare"),  # legacy monolithic prepare override group
        ("prepare_proxy", "prepare-proxy"),
        ("prepare_dataset", "prepare-dataset"),
        ("prepare_deepdel", "prepare-deepdel"),
        ("train", "train"),
        ("eval_topm", "eval-topm"),
        ("eval_topm_finalize", "eval-topm-finalize"),
        ("deepdel_update", "deepdel-update"),
        ("dock", "dock"),
    )
    for stage, cli_prefix in stage_specs:
        g = p.add_argument_group(f"{stage} stage Slurm overrides")
        g.add_argument(f"--{cli_prefix}-time", default=None,
                       help=f"Requested Slurm time for {stage} stage, e.g. 12:00:00.")
        g.add_argument(f"--{cli_prefix}-ntasks-per-node", type=int, default=None,
                       help=f"--ntasks-per-node for {stage} stage.")
        g.add_argument(f"--{cli_prefix}-cpus", type=int, default=None,
                       help=f"--cpus-per-task for {stage} stage.")
        g.add_argument(f"--{cli_prefix}-mem", default=None,
                       help=f"--mem for {stage} stage (e.g. 120G; '0' = all node RAM).")
        g.add_argument(f"--{cli_prefix}-gpus-per-node", default=None,
                       help=f"--gpus-per-node spec for {stage} stage (e.g. a100:4).")
        g.add_argument(f"--{cli_prefix}-account", default=None,
                       help=f"--account for {stage} stage (RAP allocation).")
        g.add_argument(f"--{cli_prefix}-job-name", default=None,
                       help=f"--job-name for {stage} stage.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch commands that would be submitted, but do not submit jobs.",
    )

    # Training hyperparams
    p.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Reward exponent beta used by train_deepdel_gfn.py (target p(x) ∝ R(x)^beta).",
    )
    p.add_argument(
        "--train-gfn-subsample-ratio-start",
        type=float,
        default=None,
        help="Starting subsample ratio for GFN training (annealed over steps).",
    )
    p.add_argument(
        "--train-gfn-subsample-ratio-end",
        type=float,
        default=None,
        help="Final subsample ratio for GFN training (annealed over steps).",
    )
    p.add_argument(
        "--gfn-train-steps-start",
        type=int,
        default=None,
        help=(
            "Starting number of main GFN training steps for inner loop 0. "
            "Used with --gfn-train-steps-end to linearly schedule steps over num_inner_loops."
        ),
    )
    p.add_argument(
        "--gfn-train-steps-end",
        type=int,
        default=None,
        help=(
            "Final number of main GFN training steps for the last inner loop. "
            "Used with --gfn-train-steps-start to linearly schedule steps over num_inner_loops."
        ),
    )
    p.add_argument(
        "--gfn-phase-regularization",
        action="store_true",
        help="Enable phase-regularized GFlowNet training (adds --phase-regularization to train_gfn.py).",
    )
    p.add_argument(
        "--gfn-phase-lambda",
        type=float,
        default=None,
        help="Nonnegative lambda multiplier for GFlowNet phase regularization.",
    )
    p.add_argument(
        "--phi-train-mode",
        choices=["frozen_table", "trainable_table", "trainable_network"],
        default=None,
        help=(
            "How the GFN policy interacts with the pretrained DeepDEL φ embeddings. "
            "'frozen_table' uses fixed precomputed φ tables; 'trainable_table' learns "
            "a per-BB offset on top of frozen φ; 'trainable_network' backpropagates "
            "through the φ network itself. Overrides CFG['gfn']['model']['phi_train_mode']."
        ),
    )
    p.add_argument(
        "--threshold-alpha",
        type=float,
        default=None,
        help=(
            "Smoothing parameter alpha for the 'threshold' reward: "
            "reward = sum(1/(1+exp((value-threshold)/alpha))). "
            "Larger alpha -> softer sigmoid; alpha=1.0 reproduces the legacy reward."
        ),
    )
    p.add_argument(
        "--threshold-max-weight",
        type=float,
        default=None,
        help=(
            "Optional molecular-weight cutoff for threshold rewards. Products with "
            "weight > cutoff contribute 0. Omit to preserve legacy behavior."
        ),
    )
    p.add_argument(
        "--threshold-weight-source",
        choices=["smiles", "bb_sum"],
        default=None,
        help=(
            "How to compute molecular weights for --threshold-max-weight: "
            "'smiles' uses exact product weights from product SMILES; "
            "'bb_sum' uses approximate summed building-block weights."
        ),
    )
    p.add_argument(
        "--gfn-reward-source",
        choices=["deepdel", "autodock_proxy"],
        default=None,
        help="Terminal reward source for GFN TB. Overrides CFG['gfn_reward_source'].",
    )
    for cycle in (1, 2, 3):
        p.add_argument(
            f"--forbidden-bb{cycle}",
            default=None,
            help=(
                "Pipe-delimited list of combined BB IDs that pool "
                f"{cycle} must never select during GFlowNet training."
            ),
        )
    p.add_argument(
        "--deepdel-train-weight-decay",
        type=float,
        default=None,
        help="Adam weight decay for initial DeepDEL offline training.",
    )
    p.add_argument(
        "--deepdel-train-patience",
        type=int,
        default=None,
        help="Early stopping patience on validation MSE for initial DeepDEL offline training (<=0 disables).",
    )
    p.add_argument(
        "--deepdel-finetune-weight-decay",
        type=float,
        default=None,
        help="Adam weight decay for per-inner-loop DeepDEL finetuning.",
    )
    p.add_argument(
        "--deepdel-finetune-patience",
        type=int,
        default=None,
        help="Early stopping patience on validation MSE for per-inner-loop DeepDEL finetuning (<=0 disables).",
    )

    # ECFP cache for the autodock NN proxy (train_nn.py).
    p.add_argument(
        "--ecfp-cache",
        default=None,
        help=(
            "Path to the cached ECFP .npz file used by the autodock NN proxy. "
            "Overrides CFG['autodock_proxy']['nn']['ecfp_cache']. "
            "Pass an empty string ('') to disable the cache for this run."
        ),
    )
    p.add_argument(
        "--overwrite-ecfp-cache",
        action="store_true",
        help=(
            "If set, force the autodock proxy to recompute ECFPs and overwrite "
            "the existing cache file at --ecfp-cache (or the configured path)."
        ),
    )
    p.add_argument(
        "--autodock-nn-test-size",
        type=float,
        default=None,
        help=(
            "Held-out test fraction for the autodock NN proxy. "
            "Overrides CFG['autodock_proxy']['nn']['test_size']."
        ),
    )
    p.add_argument(
        "--skip-first-autodock-training",
        action="store_true",
        help=(
            "Set CFG['train_autodock_model_on_first_pass']=False for this run. "
            "The prepare stage will copy a pretrained proxy artifact and go directly "
            "to generating the initial DeepDEL dataset."
        ),
    )
    p.add_argument(
        "--pretrained-autodock-model",
        default=None,
        help=(
            "Path to the pretrained autodock proxy artifact to use when first-pass "
            "proxy training is skipped. Overrides CFG['autodock_proxy']['pretrained_model']. "
            "If omitted, the stage worker falls back to models/autodock_model.pt for NN "
            "or models/autodock_model.joblib for RF."
        ),
    )
    p.add_argument(
        "--initial-deepdel-dataset",
        default=None,
        help=(
            "Path to an existing DeepDEL dataset CSV to use when "
            "CFG['generate_deepdel_dataset_on_first_pass'] is False. "
            "Overrides CFG['deepdel_dataset']['initial_dataset']; relative paths "
            "are resolved from the project root by the submitter."
        ),
    )
    deepdel_first_pass = p.add_mutually_exclusive_group()
    deepdel_first_pass.add_argument(
        "--train-first-deepdel-model",
        action="store_true",
        help=(
            "Set CFG['train_deepdel_model_on_first_pass']=True for this run. "
            "The first prepare_deepdel stage will train a fresh initial DeepDEL checkpoint."
        ),
    )
    deepdel_first_pass.add_argument(
        "--skip-first-deepdel-training",
        action="store_true",
        help=(
            "Set CFG['train_deepdel_model_on_first_pass']=False for this run. "
            "The submitter will copy a pretrained DeepDEL checkpoint and start with "
            "the first GFlowNet training pass."
        ),
    )
    p.add_argument(
        "--pretrained-deepdel-model",
        default=None,
        help=(
            "Path to the pretrained DeepDEL checkpoint to use when first-pass "
            "DeepDEL training is skipped. Overrides CFG['deepdel_offline']['pretrained_model']; "
            "defaults to models/deepdel.pt."
        ),
    )
    p.add_argument(
        "--deepdel-dataset-device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help=(
            "Device used by prepare_dataset / generate_dataset.py for autodock-proxy inference. "
            "Default CFG value is 'cpu' to avoid requesting many CPUs and a GPU together. "
            "Use 'cuda' together with --prepare-dataset-gpus-per-node to enable GPU inference."
        ),
    )
    p.add_argument(
        "--prepare-dataset-shards",
        type=int,
        default=None,
        help=(
            "Split prepare_dataset into this many Slurm array tasks. "
            "Overrides CFG['deepdel_dataset']['num_shards'] and enables split_jobs."
        ),
    )
    p.add_argument(
        "--no-split-prepare-dataset",
        action="store_true",
        help="Disable split prepare_dataset array jobs and use the legacy single prepare_dataset job.",
    )
    p.add_argument(
        "--deepdel-dataset-seed",
        type=int,
        default=None,
        help=(
            "Base RNG seed for DeepDEL dataset generation. In split mode each shard uses seed + shard_index. "
            "Overrides CFG['deepdel_dataset']['seed']."
        ),
    )
    p.add_argument(
        "--deepdel-dataset-n-threads",
        type=int,
        default=None,
        help="CPU worker processes per prepare_dataset shard. Overrides CFG['deepdel_dataset']['n_threads'].",
    )
    p.add_argument(
        "--deepdel-dataset-batch-pred-size",
        type=int,
        default=None,
        help="Autodock-proxy inference batch size for prepare_dataset. Overrides CFG['deepdel_dataset']['batch_pred_size'].",
    )
    p.add_argument(
        "--deepdel-dataset-write-batch-size",
        type=int,
        default=None,
        help="Rows between output CSV flushes for prepare_dataset. Overrides CFG['deepdel_dataset']['write_batch_size'].",
    )
    p.add_argument(
        "--deepdel-dataset-torch-num-threads",
        type=int,
        default=None,
        help="PyTorch intra-op threads for CPU prepare_dataset proxy inference. Overrides CFG['deepdel_dataset']['torch_num_threads'].",
    )
    p.add_argument(
        "--deepdel-dataset-torch-interop-threads",
        type=int,
        default=None,
        help="PyTorch inter-op threads for CPU prepare_dataset proxy inference. Overrides CFG['deepdel_dataset']['torch_interop_threads'].",
    )
    p.add_argument(
        "--deepdel-dataset-score-log-seconds",
        type=float,
        default=None,
        help="Heartbeat interval while scoring fingerprints in prepare_dataset. Overrides CFG['deepdel_dataset']['score_log_seconds'].",
    )
    p.add_argument(
        "--eval-topm-shards",
        type=int,
        default=None,
        help="Split eval_topm into this many Slurm array tasks. Overrides CFG['eval_topm']['num_shards'].",
    )
    p.add_argument(
        "--eval-topm-batch-rows",
        type=int,
        default=None,
        help="Max molecules per eval_topm Slurm array task (default CFG['eval_topm']['batch_rows']=2000000). The shard count is derived as ceil(topm * lib_size^3 / batch_rows).",
    )
    p.add_argument(
        "--eval-topm-device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Device used by eval_autodock_proxy_topm_scores.py for proxy inference.",
    )
    p.add_argument(
        "--eval-topm-batch-pred-size",
        type=int,
        default=None,
        help="Autodock-proxy inference batch size for eval_topm shards.",
    )
    p.add_argument(
        "--eval-topm-num-workers",
        type=int,
        default=None,
        help="CPU worker processes per eval_topm shard.",
    )
    p.add_argument(
        "--eval-topm-score-log-seconds",
        type=float,
        default=None,
        help="Heartbeat interval while scoring fingerprints in eval_topm.",
    )
    p.add_argument(
        "--dock3-timeout",
        type=int,
        default=None,
        help=(
            "Outer per-molecule DOCK3 subprocess timeout in seconds, used for "
            "ligbuild and dock64 calls. Overrides CFG['docking']['dock3_timeout']."
        ),
    )
    p.add_argument(
        "--ligbuild-timeout",
        type=int,
        default=None,
        help=(
            "Inner ligbuild DB2/protomer timeout in seconds written to "
            "custom_parms.json. Overrides CFG['docking']['ligbuild_timeout']."
        ),
    )
    p.add_argument(
        "--dock-batch-rows",
        type=int,
        default=None,
        help="Number of molecules per split active-learning docking array task (default CFG['docking']['batch_rows']=5000).",
    )
    p.add_argument(
        "--no-split-dock-jobs",
        action="store_true",
        help="Disable split active-learning docking and use the legacy single STAGE=dock job.",
    )
    return p



def main():
    global CFG
    args = _build_parser().parse_args()
    CFG = _load_config_json(args.config)
    run_id = _run_id()
    run_root = _run_root(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "models").mkdir(parents=True, exist_ok=True)

    # Apply CLI overrides into CFG before persisting config.json.
    # This keeps the config JSON the single source of truth for CPU/GPU stage scripts.
    if args.beta is not None:
        CFG["beta"] = float(args.beta)
    if args.train_gfn_subsample_ratio_start is not None:
        CFG["train_gfn_subsample_ratio_start"] = float(args.train_gfn_subsample_ratio_start)
    if args.train_gfn_subsample_ratio_end is not None:
        CFG["train_gfn_subsample_ratio_end"] = float(args.train_gfn_subsample_ratio_end)
    if args.gfn_train_steps_start is not None:
        if args.gfn_train_steps_start <= 0:
            raise ValueError("--gfn-train-steps-start must be > 0")
        CFG["gfn_train_steps_start"] = int(args.gfn_train_steps_start)
    if args.gfn_train_steps_end is not None:
        if args.gfn_train_steps_end <= 0:
            raise ValueError("--gfn-train-steps-end must be > 0")
        CFG["gfn_train_steps_end"] = int(args.gfn_train_steps_end)
    if args.phi_train_mode is not None:
        CFG.setdefault("gfn", {}).setdefault("model", {})["phi_train_mode"] = str(args.phi_train_mode)
    if args.gfn_phase_regularization:
        CFG.setdefault("gfn", {}).setdefault("phase_regularization", {})["enabled"] = True
    if args.gfn_phase_lambda is not None:
        if args.gfn_phase_lambda < 0:
            raise ValueError("--gfn-phase-lambda must be nonnegative")
        phase_cfg = CFG.setdefault("gfn", {}).setdefault("phase_regularization", {})
        phase_cfg["lambda"] = float(args.gfn_phase_lambda)
        if args.gfn_phase_lambda > 0:
            phase_cfg["enabled"] = True
    if args.threshold_alpha is not None:
        CFG["threshold_alpha"] = float(args.threshold_alpha)
    if args.threshold_max_weight is not None:
        CFG["threshold_max_weight"] = float(args.threshold_max_weight)
    if args.threshold_weight_source is not None:
        CFG["threshold_weight_source"] = str(args.threshold_weight_source)
    if args.gfn_reward_source is not None:
        CFG["gfn_reward_source"] = args.gfn_reward_source
    if any(getattr(args, f"forbidden_bb{c}") is not None for c in (1, 2, 3)):
        forbidden_cfg = CFG.setdefault("gfn", {}).setdefault("forbidden", {})
        for cycle in (1, 2, 3):
            val = getattr(args, f"forbidden_bb{cycle}")
            if val is None:
                continue
            text = str(val).strip()
            if text:
                forbidden_cfg[f"forbidden_bb{cycle}"] = text
            elif f"forbidden_bb{cycle}" in forbidden_cfg:
                del forbidden_cfg[f"forbidden_bb{cycle}"]
    if args.deepdel_train_weight_decay is not None:
        CFG.setdefault("deepdel_offline", {}).setdefault("train", {})["weight_decay"] = float(args.deepdel_train_weight_decay)
    if args.deepdel_train_patience is not None:
        CFG.setdefault("deepdel_offline", {}).setdefault("train", {})["patience"] = int(args.deepdel_train_patience)
    if args.deepdel_finetune_weight_decay is not None:
        CFG.setdefault("deepdel_offline", {}).setdefault("finetune", {})["weight_decay"] = float(args.deepdel_finetune_weight_decay)
    if args.deepdel_finetune_patience is not None:
        CFG.setdefault("deepdel_offline", {}).setdefault("finetune", {})["patience"] = int(args.deepdel_finetune_patience)

    # Apply autodock-proxy ECFP cache overrides into CFG (consumed by
    # active_learning_stage.py::_build_train_autodock_cmd via the persisted
    # config JSON).
    nn_cfg = CFG.setdefault("autodock_proxy", {}).setdefault("nn", {})
    if args.ecfp_cache is not None:
        # Empty string explicitly disables the cache for this run.
        nn_cfg["ecfp_cache"] = args.ecfp_cache if args.ecfp_cache != "" else None
    if args.overwrite_ecfp_cache:
        nn_cfg["overwrite_ecfp_cache"] = True
    if args.autodock_nn_test_size is not None:
        nn_cfg["test_size"] = float(args.autodock_nn_test_size)
    if args.skip_first_autodock_training:
        CFG["train_autodock_model_on_first_pass"] = False
    if args.pretrained_autodock_model is not None:
        CFG.setdefault("autodock_proxy", {})["pretrained_model"] = args.pretrained_autodock_model
    if args.initial_deepdel_dataset is not None:
        CFG.setdefault("deepdel_dataset", {})["initial_dataset"] = args.initial_deepdel_dataset
    if args.train_first_deepdel_model:
        CFG["train_deepdel_model_on_first_pass"] = True
    elif args.skip_first_deepdel_training:
        CFG["train_deepdel_model_on_first_pass"] = False
    if args.pretrained_deepdel_model is not None:
        CFG.setdefault("deepdel_offline", {})["pretrained_model"] = args.pretrained_deepdel_model
    if args.deepdel_dataset_device is not None:
        CFG.setdefault("deepdel_dataset", {})["device"] = str(args.deepdel_dataset_device)
    elif getattr(args, "prepare_dataset_gpus_per_node", None):
        # If the user explicitly allocates a GPU to prepare_dataset, make the
        # stage use it unless they also explicitly selected --deepdel-dataset-device.
        CFG.setdefault("deepdel_dataset", {})["device"] = "cuda"
    if args.prepare_dataset_shards is not None:
        if args.prepare_dataset_shards <= 0:
            raise ValueError("--prepare-dataset-shards must be > 0")
        CFG.setdefault("deepdel_dataset", {})["num_shards"] = int(args.prepare_dataset_shards)
        CFG.setdefault("deepdel_dataset", {})["split_jobs"] = True
    if args.no_split_prepare_dataset:
        CFG.setdefault("deepdel_dataset", {})["split_jobs"] = False
    if args.deepdel_dataset_seed is not None:
        CFG.setdefault("deepdel_dataset", {})["seed"] = int(args.deepdel_dataset_seed)
    dataset_override_checks = (
        ("deepdel_dataset_n_threads", "n_threads", int, 1),
        ("deepdel_dataset_batch_pred_size", "batch_pred_size", int, 1),
        ("deepdel_dataset_write_batch_size", "write_batch_size", int, 1),
        ("deepdel_dataset_torch_num_threads", "torch_num_threads", int, 0),
        ("deepdel_dataset_torch_interop_threads", "torch_interop_threads", int, 0),
        ("deepdel_dataset_score_log_seconds", "score_log_seconds", float, 0),
    )
    for arg_name, cfg_key, caster, minimum in dataset_override_checks:
        value = getattr(args, arg_name)
        if value is None:
            continue
        cast_value = caster(value)
        if cast_value < minimum:
            flag = "--" + arg_name.replace("_", "-")
            raise ValueError(f"{flag} must be >= {minimum}")
        CFG.setdefault("deepdel_dataset", {})[cfg_key] = cast_value
    if args.eval_topm_shards is not None:
        if args.eval_topm_shards <= 0:
            raise ValueError("--eval-topm-shards must be > 0")
        CFG.setdefault("eval_topm", {})["num_shards"] = int(args.eval_topm_shards)
        CFG.setdefault("eval_topm", {})["split_jobs"] = True
    if args.eval_topm_batch_rows is not None:
        if args.eval_topm_batch_rows <= 0:
            raise ValueError("--eval-topm-batch-rows must be > 0")
        CFG.setdefault("eval_topm", {})["batch_rows"] = int(args.eval_topm_batch_rows)
    if args.eval_topm_device is not None:
        CFG.setdefault("eval_topm", {})["device"] = str(args.eval_topm_device)
    eval_override_checks = (
        ("eval_topm_batch_pred_size", "batch_pred_size", int, 1),
        ("eval_topm_num_workers", "num_workers", int, -1),
        ("eval_topm_score_log_seconds", "score_log_seconds", float, 0),
    )
    for arg_name, cfg_key, caster, minimum in eval_override_checks:
        value = getattr(args, arg_name)
        if value is None:
            continue
        cast_value = caster(value)
        if cast_value < minimum:
            flag = "--" + arg_name.replace("_", "-")
            raise ValueError(f"{flag} must be >= {minimum}")
        CFG.setdefault("eval_topm", {})[cfg_key] = cast_value
    # Derive eval_topm shard count from the molecule upper bound unless the user
    # explicitly set it via --eval-topm-shards. Each top-m library enumerates
    # lib_size^3 products, so the total proxy-scoring work is topm * lib_size^3.
    # batch_rows is the maximum number of molecules per shard (mirrors
    # docking.batch_rows).
    if args.eval_topm_shards is None:
        eval_topm_cfg = CFG.setdefault("eval_topm", {})
        topm = int(CFG.get("topm", 10000))
        lib_size = int(CFG.get("lib_size", 20))
        batch_rows = int(eval_topm_cfg.get("batch_rows", 2000000))
        if batch_rows <= 0:
            raise ValueError("CFG['eval_topm']['batch_rows'] must be > 0")
        n_molecules = topm * (lib_size ** 3)
        eval_topm_cfg["num_shards"] = max(1, (n_molecules + batch_rows - 1) // batch_rows)
    if args.dock3_timeout is not None:
        if args.dock3_timeout <= 0:
            raise ValueError("--dock3-timeout must be > 0 seconds")
        CFG.setdefault("docking", {})["dock3_timeout"] = int(args.dock3_timeout)
    if args.ligbuild_timeout is not None:
        if args.ligbuild_timeout <= 0:
            raise ValueError("--ligbuild-timeout must be > 0 seconds")
        CFG.setdefault("docking", {})["ligbuild_timeout"] = int(args.ligbuild_timeout)
    if args.dock_batch_rows is not None:
        if args.dock_batch_rows <= 0:
            raise ValueError("--dock-batch-rows must be > 0")
        CFG.setdefault("docking", {})["batch_rows"] = int(args.dock_batch_rows)
    if args.no_split_dock_jobs:
        CFG.setdefault("docking", {})["split_active_learning_jobs"] = False


    # Build the combined BBs CSV used by all consumer scripts. The originals
    # remain untouched; the combined file has globally unique IDs + a `pool` column
    # so chemistry-aware code paths (TrimerBuilder) can recover per-cycle pools.
    paths_cfg = CFG.setdefault("paths", {})
    bbs1 = paths_cfg.get("bbs1_csv", "data/bbs_aminoacids_purged.csv")
    bbs2 = paths_cfg.get("bbs2_csv", "data/bbs_aminoacids_purged.csv")
    bbs3 = paths_cfg.get("bbs3_csv", "data/bbs_sulfonylchlorides_purged.csv")
    combined_path = run_root / "bbs_combined.csv"
    _build_combined_bbs_csv(bbs1_csv=bbs1, bbs2_csv=bbs2, bbs3_csv=bbs3, out_path=combined_path)
    paths_cfg["bbs_csv"] = str(combined_path)

    config_path = run_root / "config.json"
    with open(config_path, "w") as f:
        json.dump(CFG, f, indent=2, sort_keys=True)

    print(f"[run] RUN_ID={run_id}")
    print(f"[run] RUN_ROOT={run_root}")
    print(f"[run] TARGET={_active_learning_target()}")

    job_script = "jobs/active_learning_job.sh"

    all_job_ids = []
    prev_dock_job_id: Optional[str] = None

    for outer_loop in range(int(CFG["num_outer_loops"])):
        if bool(CFG["inner_loop_only"]):
            raise ValueError("inner_loop_only=True is not supported in the split active learning pipeline.")

        def _submit(**kwargs) -> str:
            if args.dry_run:
                cmd_parts = _build_sbatch_cmd(
                    job_script,
                    run_id=run_id,
                    config_path=config_path,
                    outer_loop=outer_loop,
                    **kwargs,
                )
                inner_label = f", inner={kwargs.get('inner_loop')}" if kwargs.get("inner_loop") is not None else ""
                print(
                    f"\n[dry-run submit] {kwargs['stage']} (outer={outer_loop}{inner_label}) -> "
                    + " ".join(cmd_parts)
                )
                # Return a fake job id so the dependency chain can be printed.
                suffix = ""
                if kwargs.get("inner_loop") is not None:
                    suffix = f"_{kwargs['inner_loop']}"
                return f"DRYRUN_{kwargs['stage']}_{outer_loop}{suffix}"
            return submit_job(job_script, run_id=run_id, config_path=config_path, outer_loop=outer_loop, **kwargs)

        slurm_defaults = CFG.get("slurm", {})

        # New prepare DAG uses separate jobs so we do not request many CPUs and
        # a GPU in the same allocation. Legacy monolithic `prepare` defaults are
        # retained as fallbacks for old configs/manual STAGE=prepare submissions.
        legacy_prepare_defaults = slurm_defaults.get("prepare_gpu") or slurm_defaults.get("prepare_cpu", {})
        prepare_proxy_defaults = slurm_defaults.get("prepare_proxy") or legacy_prepare_defaults
        prepare_dataset_defaults = slurm_defaults.get("prepare_dataset") or slurm_defaults.get("prepare_cpu", {})
        prepare_deepdel_defaults = slurm_defaults.get("prepare_deepdel") or legacy_prepare_defaults
        train_defaults = slurm_defaults.get("train", {})
        eval_topm_defaults = slurm_defaults.get("eval_topm") or train_defaults
        eval_topm_finalize_defaults = slurm_defaults.get("eval_topm_finalize") or {
            "job-name": "dd-eval-fin", "time": "1:00:00", "ntasks_per_node": 1,
            "cpus": 1, "mem": "8G", "gpus_per_node": None,
            "account": eval_topm_defaults.get("account"),
        }
        deepdel_update_defaults = slurm_defaults.get("deepdel_update") or slurm_defaults.get("prepare_deepdel") or train_defaults
        dock_defaults = slurm_defaults.get("dock", {})

        def _resolve_stage(stage: str, defaults: dict) -> dict:
            """Merge CFG["slurm"][stage] with --<stage>-* CLI overrides.

            Returns the kwargs dict consumed by `_submit(stage=...)`. Any field
            not provided by CLI falls back to the CFG dict.
            """
            cli = lambda field: getattr(args, f"{stage}_{field}", None)
            time_limit = cli("time") or defaults.get("time") or "12:00:00"
            ntasks = cli("ntasks_per_node")
            if ntasks is None:
                ntasks = defaults.get("ntasks_per_node")
            cpus = cli("cpus")
            if cpus is None:
                cpus = defaults.get("cpus")
            mem = cli("mem")
            if mem is None:
                mem = defaults.get("mem")
            gpus = cli("gpus_per_node")
            if gpus is None:
                gpus = defaults.get("gpus_per_node")
            account = cli("account") or defaults.get("account")
            job_name = cli("job_name") or defaults.get("job-name")
            return {
                "job_name": job_name,
                "time_limit": time_limit,
                "ntasks_per_node": ntasks,
                "cpus_per_task": cpus,
                "mem": mem,
                "gpus_per_node": gpus,
                "account": account,
            }

        prepare_proxy_kwargs = _resolve_stage("prepare_proxy", prepare_proxy_defaults)
        prepare_dataset_kwargs = _resolve_stage("prepare_dataset", prepare_dataset_defaults)
        prepare_deepdel_kwargs = _resolve_stage("prepare_deepdel", prepare_deepdel_defaults)
        train_kwargs = _resolve_stage("train", train_defaults)
        eval_topm_kwargs = _resolve_stage("eval_topm", eval_topm_defaults)
        eval_topm_finalize_kwargs = _resolve_stage("eval_topm_finalize", eval_topm_finalize_defaults)
        deepdel_update_kwargs = _resolve_stage("deepdel_update", deepdel_update_defaults)
        dock_kwargs = _resolve_stage("dock", dock_defaults)

        skip_prepare_proxy = (
            outer_loop == 0
            and not bool(CFG.get("train_autodock_model_on_first_pass", True))
        )
        prepare_stage_job_ids: list[str] = []
        if skip_prepare_proxy:
            _stage_autodock_proxy_artifact(run_root, dry_run=args.dry_run)
            prepare_dataset_dependency = prev_dock_job_id
        else:
            base_name = prepare_proxy_kwargs.get("job_name") or "dd-proxy"
            prepare_proxy_kwargs["job_name"] = f"{base_name}-o{outer_loop}"
            prepare_proxy_job_id = _submit(
                stage="prepare_proxy",
                **prepare_proxy_kwargs,
                # Chain outer loops sequentially: outer N's first prepare job waits
                # for outer N-1's dock to finish successfully. For outer_loop == 0
                # this is None, so no --dependency flag is added.
                dependency=prev_dock_job_id,
            )
            prepare_stage_job_ids.append(prepare_proxy_job_id)
            prepare_dataset_dependency = prepare_proxy_job_id

        dataset_cfg = CFG.get("deepdel_dataset", {}) or {}
        skip_prepare_dataset = (
            outer_loop == 0
            and not bool(CFG.get("generate_deepdel_dataset_on_first_pass", True))
        )
        split_prepare_dataset = bool(dataset_cfg.get("split_jobs", False))

        if skip_prepare_dataset:
            _stage_initial_deepdel_dataset(run_root, dry_run=args.dry_run)
            prepare_dataset_job_id = prepare_dataset_dependency
        else:
            base_name = prepare_dataset_kwargs.get("job_name") or "dd-dataset"

        if not skip_prepare_dataset and split_prepare_dataset:
            num_dataset_shards = int(dataset_cfg.get("num_shards", 1))
            if num_dataset_shards <= 0:
                raise ValueError("CFG['deepdel_dataset']['num_shards'] must be > 0 when split_jobs=True")

            batch_kwargs = dict(prepare_dataset_kwargs)
            batch_kwargs["job_name"] = f"{base_name}-batch-o{outer_loop}"
            prepare_dataset_batch_job_id = _submit(
                stage="prepare_dataset_batch",
                **batch_kwargs,
                dependency=prepare_dataset_dependency,
                array=f"0-{num_dataset_shards - 1}",
            )
            prepare_stage_job_ids.append(prepare_dataset_batch_job_id)

            finalize_kwargs = dict(prepare_dataset_kwargs)
            finalize_kwargs["job_name"] = f"{base_name}-fin-o{outer_loop}"
            finalize_kwargs["time_limit"] = "1:00:00"
            finalize_kwargs["cpus_per_task"] = 1
            finalize_kwargs["mem"] = "8G"
            finalize_kwargs["gpus_per_node"] = None
            prepare_dataset_job_id = _submit(
                stage="prepare_dataset_finalize",
                **finalize_kwargs,
                dependency=prepare_dataset_batch_job_id,
            )
            prepare_stage_job_ids.append(prepare_dataset_job_id)
        elif not skip_prepare_dataset:
            prepare_dataset_kwargs["job_name"] = f"{base_name}-o{outer_loop}"
            prepare_dataset_job_id = _submit(
                stage="prepare_dataset",
                **prepare_dataset_kwargs,
                dependency=prepare_dataset_dependency,
            )
            prepare_stage_job_ids.append(prepare_dataset_job_id)

        skip_prepare_deepdel = (
            outer_loop == 0
            and not bool(CFG.get("train_deepdel_model_on_first_pass", True))
        )
        if skip_prepare_deepdel:
            _stage_deepdel_model_artifact(run_root, dry_run=args.dry_run)
            prev_inner_or_prepare_job_id = prepare_dataset_job_id
        else:
            base_name = prepare_deepdel_kwargs.get("job_name") or "dd-init"
            prepare_deepdel_kwargs["job_name"] = f"{base_name}-o{outer_loop}"
            prepare_deepdel_job_id = _submit(
                stage="prepare_deepdel",
                **prepare_deepdel_kwargs,
                dependency=prepare_dataset_job_id,
            )
            prepare_stage_job_ids.append(prepare_deepdel_job_id)
            prev_inner_or_prepare_job_id = prepare_deepdel_job_id
        inner_job_ids: list[str] = []
        dock_stage_job_ids: list[str] = []
        dock_branch_terminal_job_ids: list[str] = []
        num_inner_loops = int(CFG["num_inner_loops"])
        eval_cfg = CFG.get("eval_topm", {}) or {}
        eval_num_shards = int(eval_cfg.get("num_shards", 1))
        if eval_num_shards <= 0:
            raise ValueError("CFG['eval_topm']['num_shards'] must be > 0")

        def _submit_docking_branch(*, inner_loop: int, dependency: str) -> str:
            """Submit a per-inner docking branch and return its terminal job id.

            The returned id is intentionally *not* used for the next inner-loop
            dependency.  It is only collected so the next outer loop can wait
            for all docking branches from this outer loop, preserving existing
            proxy-retraining semantics across outer loops.
            """

            if bool(CFG.get("docking", {}).get("split_active_learning_jobs", True)):
                dock_prepare_kwargs = dict(dock_kwargs)
                dock_prepare_kwargs["job_name"] = f"{dock_prepare_kwargs.get('job_name') or 'deepdel-dock'}-prep-o{outer_loop}-i{inner_loop}"
                dock_prepare_job_id = _submit(
                    stage="dock_prepare",
                    inner_loop=inner_loop,
                    **dock_prepare_kwargs,
                    dependency=dependency,
                )
                dock_stage_job_ids.append(dock_prepare_job_id)

                dock_launch_kwargs = dict(dock_kwargs)
                dock_launch_kwargs["job_name"] = f"{dock_launch_kwargs.get('job_name') or 'deepdel-dock'}-launch-o{outer_loop}-i{inner_loop}"
                # The launcher submits the dynamic shard array/finalizer after
                # dock_prepare knows how many shards are needed.  By default it
                # waits for the finalizer, making this launcher the terminal
                # per-inner docking dependency for the next outer loop.
                dock_launch_kwargs["time_limit"] = str(CFG.get("docking", {}).get("launch_time", "24:00:00"))
                dock_launch_kwargs["cpus_per_task"] = 1
                dock_launch_kwargs["mem"] = "8G"
                dock_job_id = _submit(
                    stage="dock_launch",
                    inner_loop=inner_loop,
                    **dock_launch_kwargs,
                    dependency=dock_prepare_job_id,
                )
                dock_stage_job_ids.append(dock_job_id)
                return dock_job_id

            dock_job_id = _submit(
                stage="dock",
                inner_loop=inner_loop,
                **dict(dock_kwargs),
                dependency=dependency,
            )
            dock_stage_job_ids.append(dock_job_id)
            return dock_job_id

        for inner_loop in range(num_inner_loops):
            inner_kwargs = dict(train_kwargs)
            inner_kwargs["job_name"] = f"{inner_kwargs.get('job_name') or 'deepdel-gfn'}-o{outer_loop}-i{inner_loop}"
            train_job_id = _submit(
                stage="inner_train",
                inner_loop=inner_loop,
                **inner_kwargs,
                dependency=prev_inner_or_prepare_job_id,
            )
            inner_job_ids.append(train_job_id)

            if str(CFG.get("gfn_reward_source", "deepdel")).lower() == "deepdel":
                eval_kwargs = dict(eval_topm_kwargs)
                eval_kwargs["job_name"] = f"{eval_kwargs.get('job_name') or 'dd-eval'}-o{outer_loop}-i{inner_loop}"
                eval_job_id = _submit(
                    stage="eval_topm_batch",
                    inner_loop=inner_loop,
                    **eval_kwargs,
                    dependency=train_job_id,
                    array=f"0-{eval_num_shards - 1}",
                )
                inner_job_ids.append(eval_job_id)

                eval_fin_kwargs = dict(eval_topm_finalize_kwargs)
                eval_fin_kwargs["job_name"] = f"{eval_fin_kwargs.get('job_name') or 'dd-eval-fin'}-o{outer_loop}-i{inner_loop}"
                eval_finalize_job_id = _submit(
                    stage="eval_topm_finalize",
                    inner_loop=inner_loop,
                    **eval_fin_kwargs,
                    dependency=eval_job_id,
                )
                inner_job_ids.append(eval_finalize_job_id)
                update_dependency = eval_finalize_job_id
            else:
                update_dependency = train_job_id

            update_kwargs = dict(deepdel_update_kwargs)
            update_kwargs["job_name"] = f"{update_kwargs.get('job_name') or 'dd-update'}-o{outer_loop}-i{inner_loop}"
            update_job_id = _submit(
                stage="deepdel_update",
                inner_loop=inner_loop,
                **update_kwargs,
                dependency=update_dependency,
            )
            inner_job_ids.append(update_job_id)

            dock_branch_terminal_job_ids.append(
                _submit_docking_branch(inner_loop=inner_loop, dependency=update_job_id)
            )

            # Advance the GFN/DeepDEL chain from the update job only.  The
            # docking branch above proceeds independently and can overlap with
            # the next inner-loop GFlowNet training step.
            prev_inner_or_prepare_job_id = update_job_id

        all_job_ids.extend(prepare_stage_job_ids)
        all_job_ids.extend(inner_job_ids)
        all_job_ids.extend(dock_stage_job_ids)
        prev_dock_job_id = ":".join(dock_branch_terminal_job_ids) if dock_branch_terminal_job_ids else prev_inner_or_prepare_job_id

    print("\n[summary] submitted jobs:")
    for jid in all_job_ids:
        print(f"  - {jid}")
    print("[summary] monitor with: squeue -u $USER")


if __name__ == "__main__":
    main()
