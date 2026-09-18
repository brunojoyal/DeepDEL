#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate an ID-only DeepDel dataset: rows of B1_id|B2_id|B3_id|y

Architecture (fast path):
  - CPU worker pool enumerates trimer products and computes ECFP fingerprints
    as *sparse on-bit index* arrays (no torch, no CUDA).
  - The main process loads the autodock proxy *once* (e.g. on CUDA), buffers
    sparse fingerprints across triplets, runs a few large batched GPU
    inferences, scatters predictions back to their triplets, computes the
    reward, and streams rows to CSV.

This avoids the previous design where N CPU workers each loaded a copy of the
torch model on the GPU and only ever scored ~27 SMILES at a time per triplet,
which both blew GPU memory and left the GPU mostly idle.
"""

import argparse
import csv
import math
import multiprocessing
import os
import random
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from deepdelgfn.autodock_proxy.model import load_autodock_proxy
from deepdelgfn.mols.proxy_scoring import (
    SparseFPProxyScorer,
    configure_cpu_thread_env,
    smi_to_sparse_fp,
)

# Your modules
import deepdelgfn.mols.dels as tri_mod
from deepdelgfn.rewards import REWARD_MODES, reward
from deepdelgfn.utils.weights import smiles_weight_array, bb_weight_lookup, bb_sum_weights


# --------------------------- Worker globals ---------------------------
# Heavy state created once per worker. Workers do *not* touch CUDA or any
# torch model -- they only enumerate SMILES and compute Morgan fingerprints.
_worker_df: Optional[pd.DataFrame] = None
_worker_builder: Optional[tri_mod.TrimerBuilder] = None
_worker_n_bits: int = 2048
_worker_radius: int = 2
_worker_lib_sample_cap: int = 0
_worker_fp_dtype = np.uint16
_worker_seed_base: int = 0
_worker_slow_triplet_log_seconds: float = 0.0


def init_worker(
    bbs_path: str,
    n_bits: int,
    radius: int,
    lib_sample_cap: int,
    fp_dtype_name: str,
    seed_base: int,
    slow_triplet_log_seconds: float,
    reaction_mode: str,
) -> None:
    """Initialize per-worker state (RDKit builder, fingerprint parameters)
    for multiprocessing workers.  Workers do not touch CUDA or torch.
    """
    global _worker_df, _worker_builder
    global _worker_n_bits, _worker_radius, _worker_lib_sample_cap
    global _worker_fp_dtype, _worker_seed_base, _worker_slow_triplet_log_seconds

    _worker_df = load_bbs(bbs_path)
    df1, df2, df3 = split_bbs_for_generation(_worker_df)
    _worker_builder = tri_mod.TrimerBuilder(df1, df2, df3, reaction_mode=str(reaction_mode))
    _worker_n_bits = int(n_bits)
    _worker_radius = int(radius)
    _worker_lib_sample_cap = int(lib_sample_cap)
    _worker_fp_dtype = np.dtype(fp_dtype_name)
    _worker_seed_base = int(seed_base)
    _worker_slow_triplet_log_seconds = float(slow_triplet_log_seconds)


# --------------------------- Utils ---------------------------

def set_seed(seed: Optional[int]):
    """Set Python, numpy, and random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _serialize_id_list(ids: List[int]) -> str:
    """Join a list of integer IDs with ``|`` as separator
    (B1_id/B2_id/B3_id format).
    """
    return "|".join(str(int(x)) for x in ids)


def _log_target_value(y_val: float, *, reward_mode: str) -> float:
    """Convert a positive reward value to the ``log R(x)`` convention
    used for DeepDEL training.

    For ``ampc_pki6_hits`` the raw reward is the expected number of hits,
    so zero expected hits is represented as ``R = 0`` and the stored
    target is ``log(R)`` clipped to ``log(1e-30)``. A negative value here
    means the reward invariant was broken upstream; raising is preferable
    to silently writing the ``log(1e-30) ~= -69`` sentinel that destabilizes
    training.
    """

    y = float(y_val)
    if reward_mode == "ampc_pki6_hits" and y < 0.0:
        raise ValueError(
            "ampc_pki6_hits reward must be non-negative before log-target "
            f"conversion; got {y}."
        )
    return float(np.log(max(y, 1e-30)))


def load_bbs(csv_path: str) -> pd.DataFrame:
    """Load building blocks CSV and ensure it has ``SMILES``, ``ID``,
    ``Name``, and ``pool`` columns (defaulting pool to 0).
    """
    df = pd.read_csv(csv_path)
    if "SMILES" not in df.columns:
        raise ValueError("bbs.csv must include a 'SMILES' column.")
    if "ID" not in df.columns:
        raise ValueError("bbs.csv must include an 'ID' column (ID-only dataset).")
    if "Name" not in df.columns:
        df["Name"] = [f"BB_{i}" for i in range(len(df))]
    if "pool" not in df.columns:
        df["pool"] = 0
    return df


def split_bbs_for_generation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a combined BB DataFrame into per-cycle (pool) DataFrames
    based on the ``pool`` column.  For legacy single-pool inputs, returns
    the same frame three times.
    """
    if "pool" not in df.columns:
        d = df.reset_index(drop=True)
        return d, d.copy(), d.copy()
    pools = pd.to_numeric(df["pool"], errors="coerce").fillna(0).astype(int)
    if not np.any(pools.to_numpy() != 0):
        d = df.reset_index(drop=True)
        return d, d.copy(), d.copy()
    return tri_mod.PoolIO.split_by_pool(df)


def _smi_to_sparse_fp(smi: str, *, radius: int, n_bits: int, fp_dtype: np.dtype) -> Optional[np.ndarray]:
    """Return the on-bit indices of a Morgan fingerprint as a sparse numpy
    array, or ``None`` for invalid SMILES.
    """
    return smi_to_sparse_fp(smi, radius=radius, n_bits=n_bits, fp_dtype=fp_dtype)


# -------------------- Dataset sampler core -------------------

class TripleSampler:
    """Samples three subsets of local per-pool BB indices, each with
    configurable minimum/maximum sizes, for constructing one dataset
    triplet.
    """

    def __init__(
        self,
        pool_sizes: Tuple[int, int, int],
        min_sizes: Tuple[int, int, int],
        max_sizes: Tuple[int, int, int],
        seed: Optional[int] = None,
    ):
        n1, n2, n3 = (int(x) for x in pool_sizes)
        if n1 <= 0 or n2 <= 0 or n3 <= 0:
            raise ValueError(f"All BB pools must be non-empty; got sizes {(n1, n2, n3)}")
        self.min1, self.min2, self.min3 = min_sizes
        self.max1, self.max2, self.max3 = max_sizes
        for a, b in [(self.min1, self.max1), (self.min2, self.max2), (self.min3, self.max3)]:
            assert 1 <= a <= b
        self.idx1 = np.arange(n1)
        self.idx2 = np.arange(n2)
        self.idx3 = np.arange(n3)
        self.rng = np.random.default_rng(seed)

    def _pick_subset(self, pool_idx: np.ndarray, smin: int, smax: int) -> np.ndarray:
        k = int(self.rng.integers(smin, smax + 1))
        if k > len(pool_idx):
            raise ValueError("Requested subset larger than available indices.")
        return self.rng.choice(pool_idx, size=k, replace=False)

    def sample_triple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        I = self._pick_subset(self.idx1, self.min1, self.max1)
        J = self._pick_subset(self.idx2, self.min2, self.max2)
        K = self._pick_subset(self.idx3, self.min3, self.max3)
        return I, J, K


# -------------------- Worker function (CPU-only) --------------------

def process_triple_worker(
    args_tuple: Tuple,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, Tuple[int, int, int], np.ndarray]], List[Dict[str, object]]]:
    """Multiprocessing worker that enumerates trimer products for one
    sampled (I, J, K) triplet, deduplicates by SMILES, computes sparse
    Morgan fingerprints, and returns the records.  Logs slow triplets and
    build failures.
    """
    global _worker_builder, _worker_n_bits, _worker_radius
    global _worker_lib_sample_cap, _worker_fp_dtype, _worker_seed_base, _worker_slow_triplet_log_seconds

    triplet_idx, I, J, K = args_tuple
    t0 = time.perf_counter()

    builder = _worker_builder
    if builder is None:
        raise RuntimeError("Worker not initialized")

    n_bits = _worker_n_bits
    radius = _worker_radius
    lib_sample_cap = _worker_lib_sample_cap
    fp_dtype = _worker_fp_dtype
    rng = np.random.default_rng(_worker_seed_base + int(triplet_idx))

    def _bb_info(slot: int, local_idx: int) -> Tuple[object, object, object]:
        df_slot = {1: builder.df1, 2: builder.df2, 3: builder.df3}[slot]
        row = df_slot.loc[int(local_idx)]
        return row.get("ID", int(local_idx)), row.get("Name", f"BB_{local_idx}"), row.get("SMILES", "")

    # Enumerate trimer SMILES with BB-index provenance.
    records: List[Tuple[str, Tuple[int, int, int]]] = []
    build_failures: List[Dict[str, object]] = []
    for i in I:
        for j in J:
            for k in K:
                ii, jj, kk = int(i), int(j), int(k)
                try:
                    rec = builder.build_trimer(ii, jj, kk)
                except Exception as e:
                    bb1_id, bb1_name, bb1_smi = _bb_info(1, ii)
                    bb2_id, bb2_name, bb2_smi = _bb_info(2, jj)
                    bb3_id, bb3_name, bb3_smi = _bb_info(3, kk)
                    build_failures.append(
                        {
                            "triplet_idx": int(triplet_idx),
                            "bb1_local_idx": ii,
                            "bb2_local_idx": jj,
                            "bb3_local_idx": kk,
                            "bb1_id": bb1_id,
                            "bb2_id": bb2_id,
                            "bb3_id": bb3_id,
                            "bb1_name": bb1_name,
                            "bb2_name": bb2_name,
                            "bb3_name": bb3_name,
                            "bb1_smiles": bb1_smi,
                            "bb2_smiles": bb2_smi,
                            "bb3_smiles": bb3_smi,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        }
                    )
                    continue
                if rec is None:
                    continue
                smi = getattr(rec, "smi", None)
                if smi:
                    records.append((smi, (ii, jj, kk)))

    if lib_sample_cap and len(records) > lib_sample_cap:
        sel = rng.choice(np.arange(len(records)), size=lib_sample_cap, replace=False)
        records = [records[int(idx)] for idx in sel]

    # Dedup within triplet (matches old behaviour).
    dedup: Dict[str, Tuple[int, int, int]] = {}
    for smi, ids in records:
        dedup.setdefault(smi, ids)

    # Featurize.
    out: List[Tuple[str, Tuple[int, int, int], np.ndarray]] = []
    for smi, ids in dedup.items():
        bits = _smi_to_sparse_fp(smi, radius=radius, n_bits=n_bits, fp_dtype=fp_dtype)
        if bits is None:
            continue
        out.append((smi, ids, bits))

    elapsed = time.perf_counter() - t0
    if _worker_slow_triplet_log_seconds > 0 and elapsed >= _worker_slow_triplet_log_seconds:
        print(
            "[worker][slow-triplet] "
            f"triplet_idx={int(triplet_idx)} elapsed={elapsed:.1f}s "
            f"sizes=({len(I)},{len(J)},{len(K)}) "
            f"raw_records={len(records)} dedup={len(dedup)} fps={len(out)} "
            f"build_failures={len(build_failures)} "
            f"I={','.join(str(int(x)) for x in I.tolist())} "
            f"J={','.join(str(int(x)) for x in J.tolist())} "
            f"K={','.join(str(int(x)) for x in K.tolist())}",
            flush=True,
        )

    return (int(triplet_idx), I, J, K, out, build_failures)




# --------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser("Generate DeepDel dataset (ID-only)")
    ap.add_argument("--bbs", required=True, help="bbs.csv (must contain SMILES and ID; optional pool)")
    ap.add_argument(
        "--autodock-model",
        type=str,
        default="models/autodock_model.joblib",
        help="Autodock proxy artifact (.joblib RF or .pt NN).",
    )

    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device for proxy inference (loaded ONCE in the main process). "
            "Use 'cuda' on a GPU node for the NN proxy. Default: cpu."
        ),
    )
    # subset sizes
    ap.add_argument("--min-size1", type=int, default=6)
    ap.add_argument("--min-size2", type=int, default=6)
    ap.add_argument("--min-size3", type=int, default=6)
    ap.add_argument("--max-size1", type=int, default=6)
    ap.add_argument("--max-size2", type=int, default=6)
    ap.add_argument("--max-size3", type=int, default=6)

    # how many triples to generate
    ap.add_argument("--num-triplets", type=int, default=10000)

    # chemistry
    ap.add_argument(
        "--reaction-mode",
        type=str,
        default=tri_mod.REACTION_MODE_AMIDE_SULFONAMIDE,
        choices=list(tri_mod.VALID_REACTION_MODES),
        help="DEL reaction mode used to enumerate products.",
    )

    # reward
    ap.add_argument("--reward", required=True, choices=REWARD_MODES)
    ap.add_argument("--k", type=int, default=None, help="k for topk_mean")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed threshold for 'threshold' reward (backward-compat alias for --threshold-min/--threshold-max).")
    ap.add_argument("--threshold-min", type=float, default=None,
                    help="Lower bound of the uniform threshold sampling interval (default: -12.0).")
    ap.add_argument("--threshold-max", type=float, default=None,
                    help="Upper bound of the uniform threshold sampling interval (default: -6.0).")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="smoothing parameter alpha for 'threshold' reward.")
    ap.add_argument("--max-weight", type=float, default=None,
                    help="Maximum molecular weight (Daltons) allowed to contribute to threshold reward.")
    ap.add_argument("--weight-source", choices=["smiles", "bb_sum"], default="bb_sum",
                    help="How to compute molecular weights when applying --max-weight (default: bb_sum).")
    ap.add_argument(
        "--log-target",
        action="store_true",
        help=(
            "Store y = log R(x) instead of y = R(x). This is the recommended "
            "convention when the dataset will be used to train a regression "
            "head whose output is consumed by a GFlowNet (Trajectory Balance "
            "needs log R, not R). It also eliminates the TB-loss NaN failure "
            "mode that occurs when the model's R prediction collapses to <= 0."
        ),
    )


    # enumeration
    ap.add_argument("--lib-sample-cap", type=int, default=10000,
                    help="Cap # products per triple (0=disable)")
    ap.add_argument("--batch-pred-size", type=int, default=8192,
                    help="GPU/CPU forward-pass batch size in main process.")
    ap.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help=(
            "Torch/BLAS intra-op threads for main-process proxy inference. "
            "0 preserves PyTorch/environment defaults. For CPU-only split shards, "
            "1 is usually safest because RDKit work is already multiprocessing-parallel."
        ),
    )
    ap.add_argument(
        "--torch-interop-threads",
        type=int,
        default=0,
        help="Torch inter-op threads for main-process proxy inference (0=default).",
    )
    ap.add_argument(
        "--score-log-seconds",
        type=float,
        default=0.0,
        help="Emit heartbeat logs during long proxy scoring flushes every N seconds (0=disable).",
    )

    # output
    ap.add_argument("--out", type=str, default=os.path.join("outputs", "deepdel", "deepdel_dataset.csv"))
    ap.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    ap.add_argument(
        "--bad-builds-csv",
        default=None,
        help=(
            "Optional CSV path where trimer build failures are logged with BB provenance. "
            "Failures are skipped instead of aborting dataset generation."
        ),
    )

    # reproducibility
    ap.add_argument("--seed", type=int, default=None)

    # parallelism
    ap.add_argument("--n-threads", type=int, default=-1,
                    help="Number of CPU worker processes (-1 = all CPUs)")
    ap.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Chunksize for multiprocessing.Pool.imap_unordered.",
    )

    ap.add_argument(
        "--write-batch-size",
        type=int,
        default=1000,
        help="How many rows to buffer before flushing to CSV.",
    )

    ap.add_argument(
        "--slow-triplet-log-seconds",
        type=float,
        default=0.0,
        help="Log worker triplet details when one triplet takes at least this many seconds (0=disable).",
    )

    args = ap.parse_args()
    configure_cpu_thread_env(args.torch_num_threads, args.torch_interop_threads)
    set_seed(args.seed)

    # Resolve threshold sampling interval (backward-compatible).
    if args.threshold is not None:
        if args.threshold_min is not None or args.threshold_max is not None:
            raise ValueError(
                "--threshold cannot be combined with --threshold-min/--threshold-max."
            )
        args.threshold_min = args.threshold
        args.threshold_max = args.threshold
    else:
        if args.threshold_min is None:
            args.threshold_min = -12.0
        if args.threshold_max is None:
            args.threshold_max = -6.0
    args.threshold_min = float(args.threshold_min)
    args.threshold_max = float(args.threshold_max)
    if args.threshold_min > args.threshold_max:
        raise ValueError(
            f"--threshold-min ({args.threshold_min}) cannot exceed --threshold-max ({args.threshold_max})."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # --- Load the proxy ONCE in the main process ---
    if not os.path.exists(args.autodock_model):
        raise FileNotFoundError(f"Artifact not found: {args.autodock_model}")

    print(f"[main] Loading proxy on device='{args.device}' from {args.autodock_model}")
    proxy = load_autodock_proxy(args.autodock_model, device=args.device)
    n_bits = int(proxy.n_bits)
    radius = int(proxy.radius)
    print(f"[main] Proxy n_bits={n_bits}, radius={radius}")
    print(f"[main] reaction_mode={args.reaction_mode}")

    fp_dtype = np.uint16 if n_bits <= 65535 else np.uint32

    # --- BBs (also load in main, just for sampling and ID maps) ---
    df = load_bbs(args.bbs)
    df1, df2, df3 = split_bbs_for_generation(df)
    id_map1 = df1["ID"].to_numpy()
    id_map2 = df2["ID"].to_numpy()
    id_map3 = df3["ID"].to_numpy()
    print(f"[main] BB pool sizes: |B1|={len(df1)}, |B2|={len(df2)}, |B3|={len(df3)}")

    # --- Sampler ---
    sampler = TripleSampler(
        (len(df1), len(df2), len(df3)),
        (args.min_size1, args.min_size2, args.min_size3),
        (args.max_size1, args.max_size2, args.max_size3),
        seed=args.seed,
    )

    # --- Worker pool ---
    # Use spawn so that any CUDA state in the main process does not pollute
    # forked workers. Workers themselves do not touch torch/CUDA.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    print(f"[main] mp start method: {multiprocessing.get_start_method()}")

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if args.n_threads > 0:
        n_jobs = args.n_threads
    elif slurm_cpus is not None and slurm_cpus.isdigit():
        n_jobs = int(slurm_cpus)
    else:
        n_jobs = cpu_count()

    print(f"[main] Pre-sampling {args.num_triplets} triplets...")
    triples = [sampler.sample_triple() for _ in range(args.num_triplets)]
    worker_args = [(i, triples[i][0], triples[i][1], triples[i][2]) for i in range(args.num_triplets)]

    # Sample one threshold per triplet from the configured interval. The
    # threshold is used to compute the reward for that triplet and is also
    # written to the CSV as a conditioning input column.
    threshold_rng = np.random.default_rng(args.seed)
    thresholds = threshold_rng.uniform(
        args.threshold_min, args.threshold_max, size=args.num_triplets
    )
    threshold_by_triplet = {i: float(thresholds[i]) for i in range(args.num_triplets)}
    print(
        f"[main] Threshold interval=[{args.threshold_min}, {args.threshold_max}] "
        f"(sampled per triplet)"
    )

    print(f"[main] Processing {args.num_triplets} triplets with {n_jobs} CPU workers; "
          f"GPU/CPU batch size for proxy = {args.batch_pred_size}")
    print(
        f"[main] multiprocessing chunksize={int(args.chunksize)}; "
        f"write_batch_size={int(args.write_batch_size)}; "
        f"slow_triplet_log_seconds={float(args.slow_triplet_log_seconds)}; "
        f"score_log_seconds={float(args.score_log_seconds)}; "
        f"torch_num_threads={int(args.torch_num_threads)}; "
        f"torch_interop_threads={int(args.torch_interop_threads)}"
    )

    # --- GPU scorer (main process) ---
    scorer = SparseFPProxyScorer(
        proxy,
        batch_size=int(args.batch_pred_size),
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        score_log_seconds=float(args.score_log_seconds),
    )

    # --- Streaming pipeline ---
    new_file = (not os.path.exists(args.out)) or (not args.append)
    mode = "a" if args.append and os.path.exists(args.out) else "w"

    # Buffers for batched scoring across triplets.
    fp_buffer: List[np.ndarray] = []
    fp_buffer_owner: List[int] = []  # triplet_idx for each fp
    pending: Dict[int, dict] = {}  # triplet_idx -> {ids, expected, vals}

    # Wait until either we have a full GPU batch *or* the producer is exhausted.
    # We use a slightly larger buffer than gpu_batch so there's always a full
    # batch ready when we flush; the scorer chunks internally if needed.
    flush_threshold = max(int(args.batch_pred_size), 4096)

    seed_base = args.seed if args.seed is not None else 0

    rows_written = 0
    bad_build_rows = 0

    bb_weights_cache = None
    if args.max_weight is not None and args.weight_source == "bb_sum":
        bb_weights_cache = bb_weight_lookup({int(row.ID): str(row.SMILES) for row in df.itertuples(index=False)})

    def _finalize_triplet(tidx: int, wr) -> None:
        """Compute reward and write CSV row for a fully-scored triplet."""
        nonlocal rows_written
        info = pending.pop(tidx)
        vals = np.asarray(info["vals"], dtype=np.float32)
        weights = None
        if args.max_weight is not None:
            if args.weight_source == "smiles":
                weights = smiles_weight_array(info.get("smiles", []))
            elif args.weight_source == "bb_sum":
                combos = info.get("combo_ids")
                if combos:
                    weights = bb_sum_weights(combos, bb_weights=bb_weights_cache or {})

        if vals.size == 0:
            y_val = float("nan")
        else:
            threshold = threshold_by_triplet[tidx]
            y_val = reward(
                vals,
                mode=args.reward,
                k=args.k,
                threshold=threshold,
                alpha=args.alpha,
                weights=weights,
                max_weight=args.max_weight,
            )
            if args.log_target and y_val is not None and np.isfinite(y_val):
                # Take log R(x) so the regression head learns log-rewards
                # directly. The threshold reward includes a +1 baseline and is
                # >= 1 by construction, so log R is finite even when all
                # products are filtered out by a molecular-weight cutoff.
                y_val = _log_target_value(y_val, reward_mode=args.reward)

        I_ids = [int(id_map1[int(x)]) for x in info["I"].tolist()]
        J_ids = [int(id_map2[int(x)]) for x in info["J"].tolist()]
        K_ids = [int(id_map3[int(x)]) for x in info["K"].tolist()]
        wr.writerow([
            _serialize_id_list(I_ids),
            _serialize_id_list(J_ids),
            _serialize_id_list(K_ids),
            float(threshold_by_triplet[tidx]),
            float(y_val) if (y_val is not None and np.isfinite(y_val)) else "",
        ])
        rows_written += 1

    def _flush(wr, *, force: bool = False) -> None:
        """Score the current fp buffer in big batches, scatter results, finalize triplets."""
        if not fp_buffer:
            return
        if (not force) and len(fp_buffer) < flush_threshold:
            return

        n_fps = len(fp_buffer)
        n_owners = len(set(fp_buffer_owner))
        rows_before = rows_written
        print(
            f"[main] Flushing {n_fps} fingerprints from {n_owners} pending triplets "
            f"(force={force}, rows_written={rows_written})",
            flush=True,
        )
        score_t0 = time.perf_counter()
        preds = scorer.score(fp_buffer)
        score_elapsed = time.perf_counter() - score_t0
        print(
            f"[main] Scored {n_fps} fingerprints in {score_elapsed:.1f}s "
            f"({n_fps / max(score_elapsed, 1e-9):.1f} fps/s)",
            flush=True,
        )
        # Scatter back.
        for owner, p in zip(fp_buffer_owner, preds):
            info = pending.get(owner)
            if info is None:
                continue
            if math.isfinite(float(p)):
                info["vals"].append(float(p))
            info["remaining"] -= 1
            if info["remaining"] == 0:
                _finalize_triplet(owner, wr)
        print(
            f"[main] Flush finalized {rows_written - rows_before} triplets; "
            f"rows_written={rows_written}/{args.num_triplets}; pending={len(pending)}",
            flush=True,
        )
        fp_buffer.clear()
        fp_buffer_owner.clear()

    bad_build_fieldnames = [
        "triplet_idx",
        "bb1_local_idx", "bb2_local_idx", "bb3_local_idx",
        "bb1_id", "bb2_id", "bb3_id",
        "bb1_name", "bb2_name", "bb3_name",
        "bb1_smiles", "bb2_smiles", "bb3_smiles",
        "error_type", "error_message",
    ]

    bad_f = None
    bad_wr = None
    if args.bad_builds_csv:
        bad_path = os.path.abspath(args.bad_builds_csv)
        os.makedirs(os.path.dirname(bad_path), exist_ok=True)
        bad_exists = os.path.exists(bad_path) and os.path.getsize(bad_path) > 0
        bad_f = open(bad_path, "a" if bad_exists else "w", newline="")
        bad_wr = csv.DictWriter(bad_f, fieldnames=bad_build_fieldnames, extrasaction="ignore")
        if not bad_exists:
            bad_wr.writeheader()

    try:
        with open(args.out, mode, newline="") as f:
            wr = csv.writer(f)
            if new_file:
                wr.writerow(["B1_id", "B2_id", "B3_id", "threshold", "y"])

            with Pool(
                processes=n_jobs,
                initializer=init_worker,
                initargs=(args.bbs, n_bits, radius, int(args.lib_sample_cap),
                          str(np.dtype(fp_dtype).name), int(seed_base), float(args.slow_triplet_log_seconds),
                          str(args.reaction_mode)),
            ) as pool:
                it = pool.imap_unordered(process_triple_worker, worker_args, chunksize=int(args.chunksize))
                last_progress = 0
                for triplet_idx, I, J, K, records, build_failures in it:
                    if build_failures:
                        bad_build_rows += len(build_failures)
                        if bad_wr is not None:
                            bad_wr.writerows(build_failures)
                            if bad_f is not None:
                                bad_f.flush()
                        else:
                            first = build_failures[0]
                            print(
                                "[warn] skipped "
                                f"{len(build_failures)} trimer build failure(s) in triplet_idx={triplet_idx}; "
                                f"first=({first.get('bb1_id')},{first.get('bb2_id')},{first.get('bb3_id')}): "
                                f"{first.get('error_type')}: {first.get('error_message')}",
                                flush=True,
                            )

                    # Register triplet.
                    if not records:
                        # No valid products: write NaN row immediately.
                        pending[triplet_idx] = {"I": I, "J": J, "K": K, "vals": [], "remaining": 0, "smiles": []}
                        _finalize_triplet(triplet_idx, wr)
                    else:
                        pending[triplet_idx] = {
                            "I": I, "J": J, "K": K,
                            "vals": [],
                            "remaining": len(records),
                            "smiles": [smi for smi, _, _ in records],
                            "combo_ids": [
                                (int(id_map1[i]), int(id_map2[j]), int(id_map3[k]))
                                for _, (i, j, k), _ in records
                            ] if args.max_weight is not None and args.weight_source == "bb_sum" else None,
                        }
                        fp_buffer.extend([bits for _, _, bits in records])
                        fp_buffer_owner.extend([triplet_idx] * len(records))

                    _flush(wr, force=False)

                    if rows_written - last_progress >= int(args.write_batch_size):
                        f.flush()
                        print(f"[main] {rows_written}/{args.num_triplets} triplets finalized")
                        last_progress = rows_written

                # Drain remaining fingerprints.
                _flush(wr, force=True)

                # Any triplets still pending shouldn't exist if all fps were scored,
                # but defensively finalize them with whatever we have.
                for tidx in list(pending.keys()):
                    _finalize_triplet(tidx, wr)

            f.flush()
    finally:
        if bad_f is not None:
            bad_f.close()

    print(f"[Done] Wrote {rows_written} rows to {args.out}")
    if bad_build_rows:
        msg = f"[Done] Skipped {bad_build_rows} trimer build failure(s)"
        if args.bad_builds_csv:
            msg += f"; details written to {args.bad_builds_csv}"
        print(msg)


if __name__ == "__main__":
    main()
