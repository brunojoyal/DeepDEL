#!/usr/bin/env python3
"""Module-level configuration, constants, and utility helpers for active learning.

This module was extracted from active_learning_stage.py.  It sets up the
global ``CFG`` dict, resolves all HPC / scratch / model paths, and exposes
small helpers used by every other stage module (``_cfg_get``, ``run_cmd``,
etc.).
"""

import csv
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from deepdelgfn.rewards import reward

# ---------------- Narval/HPC paths & thread defaults ----------------
SCRATCH = os.environ.get("SCRATCH") or os.getcwd()
RUN_ID = os.environ.get("DEL_GFN_RUN_ID") or os.environ.get("RUN_ID")
if not RUN_ID:
    RUN_ID = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_job{os.environ.get('SLURM_JOB_ID', 'local')}"
    )

RUN_ROOT = Path(SCRATCH) / "del-gfn-2" / "outputs" / "runs" / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)

CFG_PATH = os.environ.get("DEL_GFN_CONFIG")
if not CFG_PATH:
    raise RuntimeError("DEL_GFN_CONFIG is required (path to active learning config JSON)")
with open(CFG_PATH, "r") as f:
    CFG = json.load(f)

MODEL_ROOT = RUN_ROOT / "models"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# After the move into src/deepdelgfn/pipelines/, this file lives 4 levels
# below the repo root: <repo>/src/deepdelgfn/pipelines/active_learning/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[4]
FALLBACK_AUTODOCK_MODEL_JOBLIB = PROJECT_ROOT / "models" / "autodock_model.joblib"
FALLBACK_AUTODOCK_MODEL_PT = PROJECT_ROOT / "models" / "autodock_model.pt"
FALLBACK_DEEPDEL_MODEL_PT = PROJECT_ROOT / "models" / "deepdel.pt"

CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
DOCK_JOBS = int(os.environ.get("DEL_GFN_DOCK_JOBS", str(CPUS)))
DOCK_CPU = int(os.environ.get("DEL_GFN_DOCK_CPU", "1"))
DOCKING_OUT_DIR = Path(CFG["docking_out_dir"])

# Short config names map to the canonical receptor/box names used by Vina.
_TARGET_NAMES = {
    "clpp": "ClpP",
    "mpro": "Mpro",
    "seh": "sEH",
    "tblr1": "TBLR1",
}


def _vina_target() -> Optional[str]:
    """Return the configured Vina target in canonical scorer spelling."""
    configured = CFG.get("target", _cfg_get("docking.target"))
    if configured is None:
        return None
    key = str(configured).strip().lower()
    if key not in _TARGET_NAMES:
        raise ValueError(
            f"Unknown active-learning target {configured!r}; "
            f"valid targets are: {', '.join(_TARGET_NAMES)}"
        )
    return _TARGET_NAMES[key]


def active_learning_target() -> Optional[str]:
    """Return the configured active-learning target for human-readable logging."""
    return _vina_target()

# ---------------- Building-block CSV ----------------
# AmpC uses two amino-acid pools (B1, B2) and one sulfonyl-chloride pool (B3).
# To avoid a deep refactor of train_gfn / train_offline (which assume a single
# flat BB universe with globally unique IDs), the submitter merges the three
# input CSVs into a single combined CSV (RUN_ROOT/bbs_combined.csv) with a
# `pool` column tagging origin (1/2/3) and globally renumbered IDs. Consumer
# scripts that build a TrimerBuilder split this combined frame by `pool`.
def _bbs_csv() -> str:
    paths = CFG.get("paths", {}) or {}
    p = paths.get("bbs_csv") or paths.get("bbs_combined_csv")
    if not p:
        raise RuntimeError(
            "CFG['paths']['bbs_csv'] is not set. "
            "Re-run scripts/active_learning_submit.py to generate a combined BBs CSV."
        )
    return str(p)


BBS_CSV = _bbs_csv()
BBS_FLAGS = f" --bbs {shlex.quote(BBS_CSV)}"

# Original per-pool CSVs (kept available for tools like score_library.py that
# operate on real chemistry and need separate pools, not a combined frame).
def _bbs_pool_paths() -> tuple[str, str, str]:
    paths = CFG.get("paths", {}) or {}
    p1 = paths.get("bbs1_csv") or "data/bbs_aminoacids_purged.csv"
    p2 = paths.get("bbs2_csv") or "data/bbs_aminoacids_purged.csv"
    p3 = paths.get("bbs3_csv") or "data/bbs_sulfonylchlorides_purged.csv"
    return str(p1), str(p2), str(p3)


BBS1_CSV, BBS2_CSV, BBS3_CSV = _bbs_pool_paths()


def _cfg_get(path: str, default=None):
    cur = CFG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _plot_title_metadata(*, prefix: str = "") -> str:
    """Return a compact, reproducibility-oriented title suffix for AL plots."""
    def v(path, default="?"):
        value = _cfg_get(path, default)
        return str(value).replace("\n", " ")

    return (
        f"{prefix}run={RUN_ID} | lib={v('lib_size')} beta={v('beta')} "
        f"GFN hidden(joint/state/action)={v('gfn.model.joint_dim')}/"
        f"{v('gfn.model.state_dim')}/{v('gfn.model.action_dim')} "
        f"DeepDEL hidden/rho={v('deepdel_offline.model.hidden_dim')}/{v('deepdel_offline.model.rho_dim')} "
        f"lambda={v('gfn.phase_regularization.lambda', 0)} "
        f"Fourier(GFN/DeepDEL)={v('gfn.model.fourier_n_freqs', 0)}/"
        f"{v('deepdel_offline.model.fourier_n_freqs', 0)}"
    )


def _reaction_mode() -> str:
    """Return the active DEL chemistry mode.

    `reaction_mode` is intentionally top-level because it controls every
    chemistry-enumeration path in active learning.  The nested
    `docking.reaction_mode` key is retained only as a backwards-compatible
    fallback for older run/config JSONs.
    """
    return str(_cfg_get("reaction_mode", _cfg_get("docking.reaction_mode", "amide_sulfonamide")))


# DOCK3 backend assets. `score_library.py --docking-backend dock3` requires
# both `--indock-template` (the INDOCK file) and `--dockfiles` (the directory
# of pre-computed grids/spheres). The repo ships these under ampc_dockfiles/.
# We do *not* need to pre-patch the `DOCK 3.7 parameter` header: Dock3Scorer
# rewrites it to `DOCK 3.8 parameter` in-memory before invoking dock64.
DOCK3_INDOCK_TEMPLATE = str(_cfg_get("docking.indock_template", PROJECT_ROOT / "ampc_dockfiles" / "INDOCK"))
DOCK3_DOCKFILES_DIR = str(_cfg_get("docking.dockfiles", PROJECT_ROOT / "ampc_dockfiles"))
DOCK3_TIMEOUT = int(_cfg_get("docking.dock3_timeout", 600))
DOCK3_LIGBUILD_TIMEOUT = int(_cfg_get("docking.ligbuild_timeout", 300))


def _autodock_kind() -> str:
    return str(_cfg_get("autodock_proxy.kind", "rf")).lower()


def _autodock_model_path() -> Path:
    if _autodock_kind() == "nn":
        return MODEL_ROOT / "autodock_model.pt"
    return MODEL_ROOT / "autodock_model.joblib"


def _pretrained_deepdel_source() -> Path:
    configured = _cfg_get("deepdel_offline.pretrained_model")
    src = Path(str(configured)).expanduser() if configured else FALLBACK_DEEPDEL_MODEL_PT
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not src.is_file():
        raise FileNotFoundError(f"Pretrained DeepDEL artifact not found: {src}")
    return src


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rem:04.1f}s"
    hours, rem = divmod(minutes, 60)
    return f"{int(hours)}h{int(rem)}m"


def _scheduled_gfn_train_steps(inner_loop: int) -> int:
    """Main GFN training steps for an inner loop.

    New configs can provide CFG["gfn_train_steps_start"] and
    CFG["gfn_train_steps_end"] to linearly schedule steps across the inner
    active-learning loop. Old configs with only CFG["gfn_train_steps"] remain
    valid and use a constant schedule.
    """
    legacy_steps = int(CFG.get("gfn_train_steps", 5000))
    start = int(CFG.get("gfn_train_steps_start", legacy_steps))
    end = int(CFG.get("gfn_train_steps_end", start))
    num_inner = int(CFG.get("num_inner_loops", 1))
    if num_inner <= 1:
        steps = end
    else:
        frac = float(inner_loop) / float(num_inner - 1)
        steps = int(round(start + frac * (end - start)))
    if steps <= 0:
        raise ValueError(
            "Scheduled GFN train steps must be > 0; got "
            f"{steps} for inner_loop={inner_loop} "
            f"(start={start}, end={end}, num_inner_loops={num_inner})"
        )
    return steps


def run_cmd(cmd: str) -> int:
    start = time.perf_counter()
    print(f"\n[{datetime.now().isoformat(sep=' ', timespec='seconds')}] → Running:\n{cmd}")
    try:
        result = subprocess.run(shlex.split(cmd), check=True)
        elapsed = time.perf_counter() - start
        print(
            f"[OK] Command finished with exit code {result.returncode} "
            f"(duration={_format_duration(elapsed)})"
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start
        print(
            f"[ERROR] Command failed with exit code {e.returncode} "
            f"(duration={_format_duration(elapsed)}): {cmd}"
        )
        return e.returncode


def torch_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _reward_weight_flags() -> str:
    """Molecular-weight cutoff flags shared by threshold/AmpC reward modes.

    New configs can use reward_max_weight/reward_weight_source; legacy
    threshold_max_weight/threshold_weight_source remain supported.
    """
    max_weight = CFG.get("reward_max_weight", CFG.get("threshold_max_weight"))
    if max_weight is None:
        return ""
    weight_source = str(CFG.get("reward_weight_source", CFG.get("threshold_weight_source", "bb_sum")))
    if weight_source not in {"smiles", "bb_sum"}:
        raise ValueError("Reward weight source must be 'smiles' or 'bb_sum'")
    return f" --max-weight {float(max_weight)} --weight-source {shlex.quote(weight_source)}"


def _threshold_weight_flags() -> str:
    return _reward_weight_flags()


def _gfn_reward_source() -> str:
    src = str(CFG.get("gfn_reward_source", "deepdel")).lower()
    if src not in {"deepdel", "autodock_proxy"}:
        raise ValueError("CFG['gfn_reward_source'] must be 'deepdel' or 'autodock_proxy'")
    return src


def _outer_threshold(outer_loop: int) -> float:
    # Backward compatible helper: prefer explicit fixed threshold, otherwise
    # fall back to the configured interval's lower bound.
    return float(_fixed_threshold())


def _fixed_threshold() -> float:
    """Return the fixed threshold value.

    Either-or semantics:
      - If CFG contains "threshold_value", use it (fixed-threshold mode).
      - Else, if CFG contains (threshold_min, threshold_max), use threshold_min
        as the representative fixed threshold (e.g. for plotting/metrics).

    This function should be used anywhere a *single* threshold is required.
    """
    if "threshold_value" in CFG and CFG.get("threshold_value") is not None:
        return float(CFG["threshold_value"])
    threshold_min, _ = _threshold_interval()
    return float(threshold_min)


def _threshold_spec() -> tuple[str, float, float]:
    """Validate and resolve threshold configuration.

    Returns:
        (mode, threshold_min, threshold_max)

    Modes:
      - "fixed": threshold_value is set; interval collapses to [v, v]
      - "interval": threshold_min/threshold_max are set; use that interval

    Raises:
      ValueError on mixed/partial configuration.
    """
    has_value = "threshold_value" in CFG and CFG.get("threshold_value") is not None
    has_min = "threshold_min" in CFG and CFG.get("threshold_min") is not None
    has_max = "threshold_max" in CFG and CFG.get("threshold_max") is not None

    if has_value and (has_min or has_max):
        raise ValueError(
            "Invalid threshold configuration: provide either 'threshold_value' "
            "(fixed mode) OR both 'threshold_min' and 'threshold_max' (interval mode), not both."
        )
    if (has_min and not has_max) or (has_max and not has_min):
        raise ValueError(
            "Invalid threshold configuration: 'threshold_min' and 'threshold_max' "
            "must be provided together (interval mode)."
        )

    if has_value:
        v = float(CFG["threshold_value"])
        return "fixed", v, v
    if has_min and has_max:
        tmin = float(CFG["threshold_min"])
        tmax = float(CFG["threshold_max"])
        if tmin > tmax:
            raise ValueError(
                f"threshold_min ({tmin}) cannot exceed threshold_max ({tmax})."
            )
        return "interval", tmin, tmax

    # Neither provided: keep historical default interval.
    return "interval", -12.0, -6.0


def _threshold_interval() -> tuple[float, float]:
    """Return the (min, max) threshold sampling interval.

    Falls back to ``threshold_value`` for both bounds when the interval keys
    are not present, preserving backward compatibility with fixed-threshold
    configs.
    """
    # Maintain backward compatibility while enforcing either/or semantics.
    _mode, threshold_min, threshold_max = _threshold_spec()
    return float(threshold_min), float(threshold_max)


def _topm_threshold() -> float:
    """Return the fixed threshold to use for top-m selection during inference.

    Falls back to ``threshold_min``, then ``threshold_value`` if not set.
    """
    if "topm_threshold" in CFG:
        return float(CFG["topm_threshold"])
    threshold_min, _ = _threshold_interval()
    return threshold_min


def _fourier_flags(section: str = "deepdel_offline.model") -> str:
    """Build Fourier threshold conditioning CLI flags from config.

    ``section`` is the config path prefix where Fourier args are stored
    (e.g. ``deepdel_offline.model`` for DeepDEL training, ``gfn`` for GFN
    policy conditioning).
    """
    n_freqs = int(_cfg_get(f"{section}.fourier_n_freqs", 0))
    if n_freqs <= 0:
        return ""
    freq_scale = float(_cfg_get(f"{section}.fourier_freq_scale", 1.5))
    linear = bool(_cfg_get(f"{section}.fourier_linear", False))
    append_raw = bool(_cfg_get(f"{section}.fourier_append_raw", False))
    condition = str(_cfg_get(f"{section}.fourier_condition", "rho"))
    threshold_min, threshold_max = _threshold_interval()
    threshold_center = 0.5 * (threshold_min + threshold_max)
    threshold_scale = float(CFG.get("threshold_alpha", 1.0))
    if threshold_scale <= 0.0:
        raise ValueError("threshold_alpha must be > 0 when Fourier conditioning is enabled")
    flags = (
        f" --fourier-n-freqs {n_freqs} --fourier-freq-scale {freq_scale}"
        f" --fourier-threshold-center {threshold_center}"
        f" --fourier-threshold-scale {threshold_scale}"
    )
    if linear:
        flags += " --fourier-linear"
    if append_raw:
        flags += " --fourier-append-raw"
    if section.startswith("deepdel_offline"):
        flags += f" --fourier-condition {condition}"
    return flags


def _deepdel_dataset_device_flag() -> str:
    """Return the device flag for DeepDEL dataset generation.

    CFG["deepdel_dataset"]["device"] may be:
      - "auto" (default): use CUDA only when an NN proxy and CUDA are visible.
      - "cuda": force GPU inference for the NN proxy; submit prepare_dataset with a GPU.
      - "cpu": force CPU inference; useful for high-CPU/no-GPU Slurm jobs.
    """
    requested = str(_cfg_get("deepdel_dataset.device", "auto")).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("CFG['deepdel_dataset']['device'] must be one of: auto, cpu, cuda")
    if requested == "cuda":
        return " --device cuda"
    if requested == "cpu":
        return " --device cpu"
    return " --device cuda" if _autodock_kind() == "nn" and torch_cuda_available() else " --device cpu"


def _deepdel_dataset_seed_flag(*, shard_index: Optional[int] = None) -> str:
    seed = _cfg_get("deepdel_dataset.seed")
    if seed is None:
        return ""
    seed_int = int(seed)
    if shard_index is not None:
        seed_int += int(shard_index)
    return f" --seed {seed_int}"


# ---------------------------------------------------------------------------
# External DeepDEL validation-set helpers
# ---------------------------------------------------------------------------


def _deepdel_validation_csv() -> Optional[Path]:
    """Return the configured external DeepDEL validation CSV, or ``None``.

    Read from ``CFG['deepdel_offline']['validation_csv']`` (with a fallback to
    ``CFG['paths']['deepdel_validation_csv']``). Relative paths resolve against
    ``PROJECT_ROOT``. Returning ``None`` disables external validation tracking.
    """
    configured = _cfg_get("deepdel_offline.validation_csv") or _cfg_get(
        "paths.deepdel_validation_csv"
    )
    if not configured:
        return None
    p = Path(str(configured)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _deepdel_eval_dataset_flag() -> str:
    """Return `` --eval-dataset <path>`` when an external validation CSV is set."""
    path = _deepdel_validation_csv()
    if path is None:
        return ""
    return f" --eval-dataset {shlex.quote(str(path))}"


def _deepdel_model_arch_flags() -> str:
    """Return the shared DeepDEL model-architecture CLI flags.

    Kept identical across dd-init training, DeepDEL updates (finetune/retrain),
    and ``--eval-only`` so the reconstructed model always matches any resumed
    checkpoint.
    """
    pooling = str(_cfg_get("deepdel_offline.model.pooling", "mean")).lower()
    output_head = str(_cfg_get("deepdel_offline.model.output_head", "linear")).lower()
    lib_size = int(CFG.get("lib_size", 0) or 0)
    return (
        f" --bb-fp-bits {int(_cfg_get('deepdel_offline.model.bb_fp_bits', 2048))}"
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
    )


# ---------------------------------------------------------------------------
# Shared path helpers used across stage modules
# ---------------------------------------------------------------------------


def _outer_deepdel_paths(outer_loop: int) -> tuple[Path, Path, Path]:
    outer_dir = RUN_ROOT / f"outer_{outer_loop}"
    deepdel_dataset = outer_dir / "deepdel" / "deepdel_dataset.csv"
    outer_model_dir = outer_dir / "models"
    outer_model_dir.mkdir(parents=True, exist_ok=True)
    deepdel_model_last = outer_model_dir / "deepdel.pt"
    return outer_dir, deepdel_dataset, deepdel_model_last


def _csv_data_row_count(path: Path) -> int:
    """Return the number of data rows in a CSV, excluding the header."""
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _concat_csvs_with_matching_headers(inputs: list[Path], out_csv: Path) -> int:
    """Concatenate CSVs, writing one shared header and validating all headers match."""
    header = None
    written = 0
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, lineterminator="\n")
        for path in inputs:
            with path.open("r", newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                this_header = next(reader)
                if header is None:
                    header = this_header
                    writer.writerow(header)
                elif this_header != header:
                    raise RuntimeError(f"CSV header mismatch while concatenating {path}")
                for row in reader:
                    writer.writerow(row)
                    written += 1
    tmp_csv.replace(out_csv)
    return written