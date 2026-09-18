#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate autodock-proxy rewards for top-m GFN terminal libraries.

This evaluator consumes fixed top-m rows rather than sampling random triples,
but it uses the same sparse-fingerprint / parent-process proxy-scoring backend
as ``deepdel.generate_dataset`` via ``deepdelgfn.mols.proxy_scoring``.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import deepdelgfn.mols.dels as tri_mod
from deepdelgfn.autodock_proxy.model import load_autodock_proxy
from deepdelgfn.mols.proxy_scoring import SparseFPProxyScorer, configure_cpu_thread_env, smi_to_sparse_fp
from deepdelgfn.rewards import AMPC_HITRATE_REWARD_MODES, REWARD_MODES, reward
from deepdelgfn.utils.weights import bb_sum_weights, bb_weight_lookup, smiles_weight_array


_ENUM_BUILDER = None
_ENUM_ID2IDX = None
_ENUM_USE_BY_IDS = None
_ENUM_RADIUS = 2
_ENUM_N_BITS = 2048
_ENUM_FP_DTYPE = np.uint16


def _resolve_workers(n: int) -> int:
    if int(n) < 0:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None and str(slurm_cpus).isdigit():
            return max(0, int(slurm_cpus))
        return max(0, os.cpu_count() or 0)
    return max(0, int(n))


def _timing(label: str, t0: float) -> float:
    now = time.time()
    print(f"[Timing] {label}: {now - t0:.2f}s", flush=True)
    return now


def _parse_id_list(s) -> List[int]:
    text = str(s)
    if not text or text.lower() == "nan":
        return []
    return [int(x) for x in text.split("|") if x != ""]


def _log_target_value(y_val: float, *, reward_mode: str) -> float:
    y = float(y_val)
    if reward_mode == "ampc_pki6_hits" and y < 0.0:
        raise ValueError(f"ampc_pki6_hits reward must be non-negative before log-target conversion; got {y}.")
    return float(np.log(max(y, 1e-30)))


def _row_slice_bounds(n_rows: int, shard_index: Optional[int], num_shards: Optional[int]) -> tuple[int, int]:
    if shard_index is None and num_shards is None:
        return 0, int(n_rows)
    if shard_index is None or num_shards is None:
        raise ValueError("--shard-index and --num-shards must be provided together")
    n = int(n_rows); s = int(shard_index); k = int(num_shards)
    if k <= 0:
        raise ValueError("--num-shards must be > 0")
    if s < 0 or s >= k:
        raise ValueError(f"--shard-index must be in 0..{k - 1}; got {s}")
    base = n // k; rem = n % k
    start = s * base + min(s, rem)
    stop = start + base + (1 if s < rem else 0)
    return int(start), int(stop)


class TopMAggregator:
    def __init__(self, k: int, lower_is_better: bool):
        import heapq
        self.k = int(k); self.lower = bool(lower_is_better); self.heap = []; self.hq = heapq
        self.sign = -1.0 if self.lower else 1.0

    def add(self, value: float, smi: Optional[str] = None) -> None:
        if not math.isfinite(float(value)):
            return
        item = (self.sign * float(value), float(value), smi)
        if len(self.heap) < self.k:
            self.hq.heappush(self.heap, item)
        elif item[0] > self.heap[0][0]:
            self.hq.heapreplace(self.heap, item)

    def values_and_smiles_sorted(self) -> Tuple[List[float], List[Optional[str]]]:
        items = [(v, s) for (_key, v, s) in self.heap]
        items.sort(reverse=not self.lower)
        return [v for v, _s in items], [s for _v, s in items]


def _init_enum_worker(bbs_path: str, radius: int, n_bits: int, fp_dtype_name: str, reaction_mode: str) -> None:
    global _ENUM_BUILDER, _ENUM_ID2IDX, _ENUM_USE_BY_IDS, _ENUM_RADIUS, _ENUM_N_BITS, _ENUM_FP_DTYPE
    df_full = tri_mod.PoolIO.load_pool(bbs_path)
    df1, df2, df3 = tri_mod.PoolIO.split_by_pool(df_full)
    _ENUM_BUILDER = tri_mod.TrimerBuilder(df1, df2, df3, reaction_mode=str(reaction_mode))
    _ENUM_ID2IDX = {int(bb_id): int(idx) for idx, bb_id in enumerate(df_full["ID"].tolist())}
    _ENUM_USE_BY_IDS = hasattr(_ENUM_BUILDER, "build_trimer_by_ids")
    _ENUM_RADIUS = int(radius); _ENUM_N_BITS = int(n_bits); _ENUM_FP_DTYPE = np.dtype(fp_dtype_name)


def _enumerate_and_featurize_row(args_tuple: tuple) -> dict:
    rec, proxy_reward_col, proxy_yhat_col, rank_col, threshold_col = args_tuple
    builder = _ENUM_BUILDER
    if builder is None:
        raise RuntimeError("Enumeration worker was not initialized.")
    id2idx = None if _ENUM_USE_BY_IDS else _ENUM_ID2IDX
    b1_raw, b2_raw, b3_raw = rec["B1_id"], rec["B2_id"], rec["B3_id"]
    b1_ids, b2_ids, b3_ids = _parse_id_list(b1_raw), _parse_id_list(b2_raw), _parse_id_list(b3_raw)
    rank_val = rec.get(rank_col) if rank_col else None
    rank_val = int(rank_val) if rank_val is not None and not pd.isna(rank_val) else None
    product_records: List[Tuple[str, Tuple[int, int, int], np.ndarray]] = []
    for i in b1_ids:
        for j in b2_ids:
            for k in b3_ids:
                try:
                    if _ENUM_USE_BY_IDS:
                        built = builder.build_trimer_by_ids(int(i), int(j), int(k))
                    else:
                        if id2idx is None or int(i) not in id2idx or int(j) not in id2idx or int(k) not in id2idx:
                            continue
                        built = builder.build_trimer(int(id2idx[int(i)]), int(id2idx[int(j)]), int(id2idx[int(k)]))
                except Exception:
                    built = None
                smi = getattr(built, "smi", None) if built is not None else None
                if not smi:
                    continue
                bits = smi_to_sparse_fp(str(smi), radius=_ENUM_RADIUS, n_bits=_ENUM_N_BITS, fp_dtype=_ENUM_FP_DTYPE)
                if bits is not None:
                    product_records.append((str(smi), (int(i), int(j), int(k)), bits))
    row_threshold = float(rec.get(threshold_col)) if threshold_col and rec.get(threshold_col) is not None and not pd.isna(rec.get(threshold_col)) else None
    return {
        "rank": rank_val,
        "proxy_reward": float(rec.get(proxy_reward_col)) if proxy_reward_col else np.nan,
        "proxy_yhat": float(rec.get(proxy_yhat_col)) if proxy_yhat_col else np.nan,
        "B1_id": b1_raw, "B2_id": b2_raw, "B3_id": b3_raw,
        "product_records": product_records,
        "threshold": row_threshold,
    }


def _write_dataset_rows(path: str, rows: list[dict], *, append: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=["B1_id", "B2_id", "B3_id", "threshold", "y"])
    if append:
        df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
    else:
        df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate autodock-proxy library reward (ID-based)")
    ap.add_argument("--bbs", required=True)
    ap.add_argument("--topm-csv", required=True)
    ap.add_argument("--autodock-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dataset-out", default=None)
    ap.add_argument("--reward", required=True, choices=REWARD_MODES)
    ap.add_argument("--k", default=None, type=int)
    ap.add_argument("--threshold", default=None, type=float)
    ap.add_argument("--alpha", default=1.0, type=float)
    ap.add_argument(
        "--reaction-mode",
        default=tri_mod.REACTION_MODE_AMIDE_SULFONAMIDE,
        choices=list(tri_mod.VALID_REACTION_MODES),
        help="DEL reaction mode used to enumerate products.",
    )
    ap.add_argument("--max-weight", default=None, type=float)
    ap.add_argument("--weight-source", default="bb_sum", choices=["smiles", "bb_sum"])
    ap.add_argument("--n-top", type=int, default=10)
    ag = ap.add_mutually_exclusive_group()
    ag.add_argument("--lower-is-better", action="store_true", default=True)
    ag.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--save-topm-smiles", action="store_true")
    ap.add_argument("--batch-pred-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-num-threads", type=int, default=1)
    ap.add_argument("--torch-interop-threads", type=int, default=1)
    ap.add_argument("--score-log-seconds", type=float, default=0.0)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--log-target", action="store_true")
    args = ap.parse_args()

    if args.dataset and args.dataset_out:
        raise ValueError("Use only one of --dataset or --dataset-out")
    configure_cpu_thread_env(args.torch_num_threads, args.torch_interop_threads)
    total_t0 = time.time(); t_phase = total_t0
    args.num_workers = _resolve_workers(args.num_workers)
    lower_is_better = not args.higher_is_better
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    full_topm_df = pd.read_csv(args.topm_csv)
    required_cols = {"B1_id", "B2_id", "B3_id"}
    if not required_cols.issubset(full_topm_df.columns):
        missing = ", ".join(sorted(required_cols - set(full_topm_df.columns)))
        raise ValueError(f"topm CSV must include B1_id,B2_id,B3_id (missing: {missing})")
    start, stop = _row_slice_bounds(len(full_topm_df), args.shard_index, args.num_shards)
    topm_df = full_topm_df.iloc[start:stop].copy().reset_index(drop=True)
    print(f"[Shard] rows={start}:{stop} of {len(full_topm_df)}", flush=True)
    t_phase = _timing(f"loaded top-m CSV slice ({len(topm_df):,} rows)", t_phase)

    if not os.path.exists(args.autodock_model):
        raise FileNotFoundError(f"Autodock proxy artifact not found: {args.autodock_model}")
    proxy_device = None if str(args.device).lower() == "auto" else str(args.device)
    proxy = load_autodock_proxy(args.autodock_model, device=proxy_device)
    n_bits = int(proxy.n_bits); radius = int(proxy.radius)
    fp_dtype = np.uint16 if n_bits <= 65535 else np.uint32
    print(f"[Artifact] proxy={args.autodock_model} n_bits={n_bits} radius={radius} device={getattr(proxy, 'device', 'cpu')}", flush=True)
    print(f"[Config] reaction_mode={args.reaction_mode}", flush=True)
    t_phase = _timing("loaded proxy", t_phase)

    df_full = tri_mod.PoolIO.load_pool(args.bbs)
    bb_weights = None
    if args.max_weight is not None and args.weight_source == "bb_sum":
        bb_weights = bb_weight_lookup({int(row.ID): str(row.SMILES) for row in df_full.itertuples(index=False)})

    proxy_reward_col = "reward" if "reward" in topm_df.columns else None
    proxy_yhat_col = "yhat" if "yhat" in topm_df.columns else None
    rank_col = "rank" if "rank" in topm_df.columns else None
    # Prefer a fixed CLI --threshold over any per-row CSV column so Experiment 1
    # (and other fixed-threshold evals) score every library at the same T.
    # Fall back to the CSV column only when --threshold is omitted.
    csv_has_threshold = "threshold" in topm_df.columns
    if args.threshold is not None:
        threshold_col = None
        fixed_threshold = float(args.threshold)
        print(
            f"[Config] Using fixed --threshold={fixed_threshold} for all rows"
            + (" (ignoring topm CSV 'threshold' column)" if csv_has_threshold else ""),
            flush=True,
        )
    elif csv_has_threshold:
        threshold_col = "threshold"
        fixed_threshold = None
        print("[Config] Using per-row threshold from topm CSV column 'threshold'", flush=True)
    else:
        threshold_col = None
        fixed_threshold = 0.0
        print("[Config] No --threshold and no CSV threshold column; defaulting to 0.0", flush=True)
    enum_args = [(rec, proxy_reward_col, proxy_yhat_col, rank_col, threshold_col) for rec in topm_df.to_dict("records")]

    if args.num_workers > 1 and len(enum_args) > 1:
        chunksize = max(1, len(enum_args) // (args.num_workers * 4))
        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_enum_worker, initargs=(args.bbs, radius, n_bits, str(np.dtype(fp_dtype).name), str(args.reaction_mode))) as ex:
            row_infos = list(ex.map(_enumerate_and_featurize_row, enum_args, chunksize=chunksize))
    else:
        _init_enum_worker(args.bbs, radius, n_bits, str(np.dtype(fp_dtype).name), str(args.reaction_mode))
        row_infos = [_enumerate_and_featurize_row(x) for x in enum_args]
    t_phase = _timing("enumerated products and sparse ECFPs", t_phase)

    unique_bits: Dict[str, np.ndarray] = {}
    for info in row_infos:
        for smi, _bb_ids, bits in info["product_records"]:
            unique_bits.setdefault(smi, bits)
    unique_smiles = list(unique_bits.keys())
    scorer = SparseFPProxyScorer(proxy, batch_size=int(args.batch_pred_size), torch_num_threads=int(args.torch_num_threads), torch_interop_threads=int(args.torch_interop_threads), score_log_seconds=float(args.score_log_seconds))
    preds = scorer.score([unique_bits[s] for s in unique_smiles]) if unique_smiles else np.array([], dtype=np.float32)
    pred_cache = {s: float(v) for s, v in zip(unique_smiles, preds)}
    t_phase = _timing(f"predicted proxy values ({len(pred_cache):,} cached)", t_phase)

    rows_out = []; append_rows = []
    for info in row_infos:
        product_records = info["product_records"]
        product_smiles = [s for s, _bb_ids, _bits in product_records]
        autodock_proxy_value = np.nan; autodock_proxy_vals: List[float] = []; smi_top: List[Optional[str]] = []
        if product_smiles:
            # topk_mean uses an explicit top-k mean of molecule docking scores.
            # All other modes (threshold/mean/ampc_*) go through rewards.reward so
            # autodock_proxy_value is the library-level reward DeepDEL approximates.
            # Do not route threshold/mean through this branch merely because --k is set.
            if args.reward == "topk_mean":
                topm = TopMAggregator(k=args.n_top if args.k is None else int(args.k), lower_is_better=lower_is_better)
                for smi in product_smiles:
                    v = pred_cache.get(smi)
                    if v is not None and math.isfinite(float(v)):
                        topm.add(float(v), smi)
                autodock_proxy_vals, smi_top = topm.values_and_smiles_sorted()
                if autodock_proxy_vals:
                    autodock_proxy_value = float(np.mean(autodock_proxy_vals))
            else:

                vals = []; aligned_weights = []
                for smi, bb_ids, _bits in product_records:
                    v = pred_cache.get(smi)
                    if v is None or not math.isfinite(float(v)):
                        continue
                    vals.append(float(v))
                    if args.max_weight is not None:
                        if args.weight_source == "smiles":
                            aligned_weights.append(float(smiles_weight_array([smi])[0]))
                        elif args.weight_source == "bb_sum" and bb_weights is not None:
                            aligned_weights.append(float(bb_sum_weights([bb_ids], bb_weights=bb_weights)[0]))
                weights_slice = np.asarray(aligned_weights, dtype=float) if aligned_weights else None
                row_threshold = info.get("threshold")
                if fixed_threshold is not None:
                    effective_threshold = float(fixed_threshold)
                elif row_threshold is not None:
                    effective_threshold = float(row_threshold)
                else:
                    effective_threshold = 0.0
                autodock_proxy_value = float(reward(np.asarray(vals, dtype=float), mode=args.reward, k=args.k, threshold=effective_threshold, alpha=args.alpha, weights=weights_slice, max_weight=args.max_weight))
                autodock_proxy_vals = vals
        row_threshold = info.get("threshold")
        if fixed_threshold is not None:
            effective_threshold = float(fixed_threshold)
        elif row_threshold is not None:
            effective_threshold = float(row_threshold)
        else:
            effective_threshold = 0.0
        rec = {
            "rank": info["rank"], "proxy_reward": info["proxy_reward"], "proxy_yhat": info["proxy_yhat"],
            "B1_id": info["B1_id"], "B2_id": info["B2_id"], "B3_id": info["B3_id"],
            "autodock_proxy_value": autodock_proxy_value,
            "threshold": effective_threshold,
        }
        if args.save_topm_smiles:
            # Experiment 1 uses this column to physically dock selected proxy
            # leaders, so persist every valid product, not only the proxy top-N.
            rec["topm_smiles"] = ";".join(dict.fromkeys(product_smiles))
        rows_out.append(rec)
        if args.dataset is not None or args.dataset_out is not None:
            y_to_append = autodock_proxy_value
            if args.log_target and y_to_append is not None and np.isfinite(y_to_append):
                y_to_append = _log_target_value(y_to_append, reward_mode=args.reward)
            append_rows.append({
                "B1_id": info["B1_id"],
                "B2_id": info["B2_id"],
                "B3_id": info["B3_id"],
                "threshold": effective_threshold,
                "y": y_to_append,
            })


    out_df = pd.DataFrame(rows_out)
    if "autodock_proxy_value" in out_df.columns and out_df["autodock_proxy_value"].notna().any():
        out_df = out_df.sort_values(by=["autodock_proxy_value"], ascending=False).reset_index(drop=True)
    out_df.to_csv(args.out, index=False)
    print(f"[Done] Wrote results to {args.out}", flush=True)
    if append_rows:
        if args.dataset_out:
            _write_dataset_rows(args.dataset_out, append_rows, append=False)
            print(f"[DatasetOut] Wrote {len(append_rows)} row(s) to {args.dataset_out}", flush=True)
        elif args.dataset:
            _write_dataset_rows(args.dataset, append_rows, append=True)
            print(f"[Append] Added {len(append_rows)} row(s) to {args.dataset}", flush=True)
    _timing("wrote outputs", t_phase)
    print(f"[Timing] total eval_autodock_proxy_topm_scores: {time.time() - total_t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()