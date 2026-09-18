#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metropolis-Hastings sampler over DEL libraries targeting the DeepDEL (F2) reward.

Experiment 1-MCMC baseline. Replaces the GFlowNet training/selection step with
an MCMC sampler over fixed-shape DEL libraries (B1, B2, B3), targeting
``p(L) ∝ R(L)^β`` where ``R`` is the frozen DeepDEL library reward.  The rest
of the Experiment 1 pipeline (F1 proxy evaluation, aggregation) is unchanged
and consumes the exact same ``topm_rewards.csv`` schema produced by the
GFlowNet trainers.

Sampling moves are symmetric (Hastings ratio = 1):

* ``swap1`` — replace one BB in one cycle with an out-of-set BB;
* ``swap2`` — replace two BBs in one cycle with two out-of-set BBs;
* ``cycle`` — resample an entire cycle from its pool;
* ``mix``   — 70% swap1 / 20% swap2 / 10% cycle.

Optional parallel tempering over a beta ladder (``--temper-ladder``) improves
mixing in the highly peaked target ``R^β`` (``β = --beta``); only the cold
replica (first ladder entry) is used for the reported samples.

Outputs (all in ``--outdir``):

* ``topm_rewards.csv`` — same schema as the GFlowNet trainers
  (``rank,reward,yhat,B1_id,B2_id,B3_id,threshold``);
* ``mcmc_trace.csv`` — per-step chain record with a ``phase`` column
  (``burnin``/``sampling``);
* ``mcmc_summary.json`` — acceptance rates, unique-library counts, ESS;
* ``run_config.csv`` — resolved CLI configuration;
* ``wall_clock_seconds.txt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd  # noqa: F401  (kept for parity with trainer imports)
import torch

# Reuse the shared loading/feature helpers from the DeepSets GFN trainer so the
# reward oracle and pools are bit-identical to training.
from deepdelgfn.gfn.train_gfn import (
    _parse_pipe_separated_ints,
    load_bbs,
    load_or_create_ecfp_cache,
    set_seed,
)
from deepdelgfn.models.deepsets import TripleDeepSet

# Library identity: (tuple(sorted B1), tuple(sorted B2), tuple(sorted B3)).
LibKey = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


def library_key(B1: Sequence[int], B2: Sequence[int], B3: Sequence[int]) -> LibKey:
    """Canonical immutable identity for a library (sorted per cycle)."""
    return (tuple(sorted(B1)), tuple(sorted(B2)), tuple(sorted(B3)))


def load_reward_model(deepdel_path: str, device: torch.device):
    """Load the frozen DeepDEL reward oracle exactly as ``train_gfn`` does.

    Returns ``(model, meta)`` where ``meta`` holds the resolved checkpoint
    configuration needed for feature construction and reward interpretation.
    """
    ckpt = torch.load(deepdel_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    d_in = int(ckpt.get("d_in", ckpt_args.get("bb_fp_bits", 2048)))
    d_h = ckpt_args.get("hidden_dim", 256)
    d_rho = ckpt_args.get("rho_dim", 256)
    dropout = ckpt_args.get("dropout", 0.1)
    shared_phi = ckpt_args.get("shared_phi", False)
    pooling = ckpt_args.get("pooling", "mean")
    output_head = ckpt_args.get("output_head", "linear")
    lib_size = ckpt_args.get("lib_size", None)
    append_molecular_weight = bool(ckpt_args.get("append_molecular_weight", False))
    ckpt_fourier_n_freqs = int(ckpt_args.get("fourier_n_freqs", 0))
    ckpt_fourier_freq_scale = float(ckpt_args.get("fourier_freq_scale", 1.5))
    ckpt_fourier_linear = bool(ckpt_args.get("fourier_linear", False))
    ckpt_fourier_append_raw = bool(ckpt_args.get("fourier_append_raw", False))
    ckpt_fourier_threshold_center = ckpt_args.get("fourier_threshold_center")
    ckpt_fourier_threshold_scale = ckpt_args.get("fourier_threshold_scale")
    ckpt_fourier_condition = str(ckpt_args.get("fourier_condition", "rho"))
    log_target = bool(ckpt_args.get("log_target", False))

    model = TripleDeepSet(
        d_in=d_in,
        d_hidden=d_h,
        d_rho=d_rho,
        dropout=dropout,
        shared_phi=shared_phi,
        pooling=pooling,
        output_head=output_head,
        reward_bound_k=lib_size,
        fourier_n_freqs=ckpt_fourier_n_freqs,
        fourier_freq_scale=ckpt_fourier_freq_scale,
        fourier_linear=ckpt_fourier_linear,
        fourier_append_raw=ckpt_fourier_append_raw,
        fourier_condition=ckpt_fourier_condition,
        fourier_threshold_center=ckpt_fourier_threshold_center,
        fourier_threshold_scale=ckpt_fourier_threshold_scale,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {
        "d_in": int(d_in),
        "d_h": int(d_h),
        "d_rho": int(d_rho),
        "dropout": float(dropout),
        "shared_phi": bool(shared_phi),
        "pooling": str(pooling),
        "output_head": str(output_head),
        "lib_size": None if lib_size is None else int(lib_size),
        "append_molecular_weight": append_molecular_weight,
        "fourier_n_freqs": ckpt_fourier_n_freqs,
        "fourier_freq_scale": ckpt_fourier_freq_scale,
        "fourier_linear": ckpt_fourier_linear,
        "fourier_append_raw": ckpt_fourier_append_raw,
        "fourier_condition": ckpt_fourier_condition,
        "fourier_threshold_center": ckpt_fourier_threshold_center,
        "fourier_threshold_scale": ckpt_fourier_threshold_scale,
        "log_target": log_target,
    }
    return model, meta


@torch.no_grad()
def batched_log_rewards(
    triples: Sequence[Tuple[List[int], List[int], List[int]]],
    X_all: torch.Tensor,
    model: TripleDeepSet,
    *,
    log_target: bool,
    model_is_logprobs: bool,
    threshold: Optional[float],
) -> np.ndarray:
    """Compute ``log R`` for a batch of fixed-shape libraries in one forward.

    Mirrors ``train_gfn.reward_from_triple`` semantics for the three checkpoint
    conventions (``log_target``, legacy ``model_is_logprobs``, and raw reward)
    but evaluates all libraries in a single batched model call.
    """
    B = len(triples)
    if B == 0:
        return np.zeros(0, dtype=np.float64)
    device = X_all.device
    B1s = torch.tensor([t[0] for t in triples], dtype=torch.long, device=device)
    B2s = torch.tensor([t[1] for t in triples], dtype=torch.long, device=device)
    B3s = torch.tensor([t[2] for t in triples], dtype=torch.long, device=device)
    X1 = X_all[B1s]  # [B, s1, d]
    X2 = X_all[B2s]  # [B, s2, d]
    X3 = X_all[B3s]  # [B, s3, d]
    m1 = torch.ones_like(B1s, dtype=torch.float32)
    m2 = torch.ones_like(B2s, dtype=torch.float32)
    m3 = torch.ones_like(B3s, dtype=torch.float32)
    if threshold is not None:
        t = torch.full((B,), float(threshold), dtype=torch.float32, device=device)
        out = model(((X1, m1), (X2, m2), (X3, m3)), t)
    else:
        out = model(((X1, m1), (X2, m2), (X3, m3)))
    out = out.reshape(-1)
    if log_target:
        logR = out
    elif model_is_logprobs:
        logR = -out
    else:
        logR = torch.log(out + 1e-40)
    return logR.cpu().numpy().astype(np.float64)


def estimate_ess(x: np.ndarray, max_lag: int = 2000) -> float:
    """Effective sample size via the initial-positive-sequence autocorrelation
    estimator.  Diagnostic only; not used for inference."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float(n)
    xc = x - x.mean()
    var = float(np.dot(xc, xc))
    if var <= 0.0:
        return float(n)
    acf_sum = 0.0
    for lag in range(1, min(max_lag, n)):
        c = float(np.dot(xc[:-lag], xc[lag:]) / var)
        if c <= 0.0:
            break
        acf_sum += c
    rho = 1.0 + 2.0 * acf_sum
    return float(n / max(rho, 1.0))


class MCMCSampler:
    """Parallel Metropolis-Hastings sampler over DEL libraries.

    All proposals are symmetric, so the Hastings ratio is 1 and the acceptance
    probability is ``min(1, exp(β · ΔlogR))``.  Proposals for all chains are
    evaluated in one batched ``TripleDeepSet`` forward per iteration.
    """

    def __init__(
        self,
        *,
        sizes: Tuple[int, int, int],
        allowed: Tuple[List[int], List[int], List[int]],
        X_all: torch.Tensor,
        model: TripleDeepSet,
        log_target: bool,
        model_is_logprobs: bool,
        threshold: Optional[float],
        beta: float,
        proposal_mode: str,
        rng: np.random.Generator,
        reward_cache_size: int,
    ):
        self.sizes = tuple(int(s) for s in sizes)
        self.allowed = [list(a) for a in allowed]
        for c in range(3):
            if len(self.allowed[c]) < self.sizes[c]:
                raise ValueError(
                    f"allowed pool {c + 1} has {len(self.allowed[c])} BBs but size{c + 1}={self.sizes[c]}"
                )
        self.X_all = X_all
        self.model = model
        self.log_target = log_target
        self.model_is_logprobs = model_is_logprobs
        self.threshold = threshold
        self.beta = float(beta)
        self.proposal_mode = proposal_mode
        self.rng = rng
        self.reward_cache_size = int(reward_cache_size)
        self._cache: Dict[LibKey, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    # ------------------------------ proposals ------------------------------
    def random_state(self) -> LibKey:
        return tuple(
            tuple(
                sorted(self.rng.choice(self.allowed[c], size=self.sizes[c], replace=False).tolist())
            )
            for c in range(3)
        )

    def _propose_swap(self, state: LibKey, k: int) -> LibKey:
        cycles = [list(state[0]), list(state[1]), list(state[2])]
        c = int(self.rng.integers(0, 3))
        pool = self.allowed[c]
        in_set = set(cycles[c])
        out_candidates = [i for i in pool if i not in in_set]
        if not out_candidates:
            return state
        kk = min(k, len(cycles[c]), len(out_candidates))
        remove = self.rng.choice(cycles[c], size=kk, replace=False)
        add = self.rng.choice(out_candidates, size=kk, replace=False)
        new_set = sorted((in_set - set(remove.tolist())) | set(add.tolist()))
        cycles[c] = new_set
        return (tuple(cycles[0]), tuple(cycles[1]), tuple(cycles[2]))

    def _propose_cycle(self, state: LibKey) -> LibKey:
        cycles = [list(state[0]), list(state[1]), list(state[2])]
        c = int(self.rng.integers(0, 3))
        new_set = sorted(self.rng.choice(self.allowed[c], size=self.sizes[c], replace=False).tolist())
        cycles[c] = new_set
        return (tuple(cycles[0]), tuple(cycles[1]), tuple(cycles[2]))

    def propose(self, state: LibKey) -> LibKey:
        mode = self.proposal_mode
        if mode == "mix":
            r = self.rng.random()
            if r < 0.7:
                return self._propose_swap(state, k=1)
            if r < 0.9:
                return self._propose_swap(state, k=2)
            return self._propose_cycle(state)
        if mode == "swap2":
            return self._propose_swap(state, k=2)
        if mode == "cycle":
            return self._propose_cycle(state)
        return self._propose_swap(state, k=1)

    # ------------------------------ rewards ------------------------------
    def _logR_batch(self, keys: Sequence[LibKey]) -> np.ndarray:
        out = np.empty(len(keys), dtype=np.float64)
        missing: List[int] = []
        for i, key in enumerate(keys):
            v = self._cache.get(key)
            if v is None:
                missing.append(i)
            else:
                self._cache_hits += 1
                out[i] = v
        if missing:
            self._cache_misses += len(missing)
            triples = [[list(part) for part in keys[i]] for i in missing]
            vals = batched_log_rewards(
                triples,
                self.X_all,
                self.model,
                log_target=self.log_target,
                model_is_logprobs=self.model_is_logprobs,
                threshold=self.threshold,
            )
            for i, v in zip(missing, vals):
                vv = float(v)
                out[i] = vv
                if len(self._cache) < self.reward_cache_size:
                    self._cache[keys[i]] = vv
        return out
# ------------------------------ run ------------------------------
    def run(
        self,
        *,
        n_chains: int,
        steps: int,
        burn_in: int,
        thin: int,
        temper_ladder: Optional[List[float]],
        swap_replicas_every: int,
        topm: int,
        keep_repeats: bool,
        id_map: Optional[np.ndarray],
        trace_path: str,
        log_interval: int,
    ) -> Dict[str, object]:
        if temper_ladder is not None:
            if len(temper_ladder) != n_chains:
                raise ValueError("--temper-ladder length must equal --n-chains")
            betas = np.asarray(temper_ladder, dtype=np.float64)
        else:
            betas = np.full(n_chains, self.beta, dtype=np.float64)

        def id_str(xs: Sequence[int]) -> str:
            if id_map is None:
                return "|".join(str(int(i)) for i in xs)
            return "|".join(str(int(id_map[i])) for i in xs)

        states = [self.random_state() for _ in range(n_chains)]
        keys = [library_key(*st) for st in states]
        logR = self._logR_batch(keys)

        accepted_total = np.zeros(n_chains, dtype=np.int64)
        accepted_sampling = np.zeros(n_chains, dtype=np.int64)
        n_total = np.zeros(n_chains, dtype=np.int64)
        n_sampling = np.zeros(n_chains, dtype=np.int64)
        post_burnin_logR: List[List[float]] = [[] for _ in range(n_chains)]

        # Top-m tracking.  In dedupe mode (default) keep the best reward per
        # unique library; in keep-repeats mode keep a bounded list of the best
        # TOPM*5 samples (mirrors the GFN trainer's pruning).
        best_by_key: Dict[LibKey, Dict[str, object]] = {}
        top_records: List[Dict[str, object]] = []

        t0 = time.time()
        with open(trace_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["phase", "chain", "step", "accepted", "log_reward", "reward",
                 "B1_id", "B2_id", "B3_id", "threshold"]
            )
            for step in range(burn_in + steps):
                proposals = [self.propose(st) for st in states]
                prop_keys = [library_key(*p) for p in proposals]
                prop_logR = self._logR_batch(prop_keys)

                log_alpha = betas * (prop_logR - logR)
                u = self.rng.random(n_chains)
                acc = np.log(u) < log_alpha

                for c in range(n_chains):
                    n_total[c] += 1
                    sampling = step >= burn_in
                    if sampling:
                        n_sampling[c] += 1
                    if acc[c]:
                        accepted_total[c] += 1
                        if sampling:
                            accepted_sampling[c] += 1
                        states[c] = proposals[c]
                        keys[c] = prop_keys[c]
                        logR[c] = prop_logR[c]
                    if sampling and (step - burn_in) % thin == 0:
                        post_burnin_logR[c].append(float(logR[c]))

                    reward = math.exp(float(logR[c]))
                    writer.writerow(
                        [
                            "sampling" if sampling else "burnin",
                            c,
                            step,
                            int(acc[c]),
                            float(logR[c]),
                            reward,
                            id_str(states[c][0]),
                            id_str(states[c][1]),
                            id_str(states[c][2]),
                            "" if self.threshold is None else self.threshold,
                        ]
                    )
                    rec = {
                        "reward": reward,
                        "yhat": float(-logR[c]),
                        "B1_id": id_str(states[c][0]),
                        "B2_id": id_str(states[c][1]),
                        "B3_id": id_str(states[c][2]),
                        "threshold": self.threshold,
                    }
                    if keep_repeats:
                        top_records.append(rec)
                        if len(top_records) > topm * 5:
                            top_records = sorted(
                                top_records, key=lambda r: float(r["reward"]), reverse=True
                            )[: topm * 5]
                    else:
                        prev = best_by_key.get(keys[c])
                        if prev is None or reward > float(prev["reward"]):
                            best_by_key[keys[c]] = rec

                # Replica exchange between adjacent replicas.
                if (
                    temper_ladder is not None
                    and swap_replicas_every > 0
                    and (step + 1) % swap_replicas_every == 0
                ):
                    for c in range(n_chains - 1):
                        log_alpha_x = (betas[c] - betas[c + 1]) * (logR[c + 1] - logR[c])
                        if self.rng.random() < math.exp(min(0.0, log_alpha_x)):
                            states[c], states[c + 1] = states[c + 1], states[c]
                            keys[c], keys[c + 1] = keys[c + 1], keys[c]
                            logR[c], logR[c + 1] = logR[c + 1], logR[c]

                if log_interval > 0 and (step + 1) % log_interval == 0:
                    ar = float(accepted_total.sum()) / max(1, int(n_total.sum()))
                    n_uniq = len(best_by_key) if not keep_repeats else len(top_records)
                    print(
                        f"[MCMC] step {step + 1}/{burn_in + steps} "
                        f"acceptance={ar:.3f} unique={n_uniq} "
                        f"elapsed={time.time() - t0:.1f}s",
                        flush=True,
                    )

        wall = time.time() - t0
        return self._finalize(
            best_by_key=best_by_key,
            top_records=top_records,
            topm=topm,
            keep_repeats=keep_repeats,
            accepted_total=accepted_total,
            accepted_sampling=accepted_sampling,
            n_total=n_total,
            n_sampling=n_sampling,
            post_burnin_logR=post_burnin_logR,
            betas=betas,
            steps=steps,
            burn_in=burn_in,
            thin=thin,
            temper_ladder=temper_ladder,
            wall=wall,
        )

    def _finalize(
        self,
        *,
        best_by_key: Dict[LibKey, Dict[str, object]],
        top_records: List[Dict[str, object]],
        topm: int,
        keep_repeats: bool,
        accepted_total: np.ndarray,
        accepted_sampling: np.ndarray,
        n_total: np.ndarray,
        n_sampling: np.ndarray,
        post_burnin_logR: List[List[float]],
        betas: np.ndarray,
        steps: int,
        burn_in: int,
        thin: int,
        temper_ladder: Optional[List[float]],
        wall: float,
    ) -> Dict[str, object]:
        # ---- Assemble the top-m shortlist (same schema as the GFN trainers). ----
        if keep_repeats:
            topm_sorted = sorted(
                top_records, key=lambda r: float(r["reward"]), reverse=True
            )[:topm]
        else:
            topm_sorted = sorted(
                best_by_key.values(), key=lambda r: float(r["reward"]), reverse=True
            )[:topm]
        if len(topm_sorted) < topm:
            print(
                f"[MCMC] WARNING: only {len(topm_sorted)} unique libraries visited; "
                f"requested topm={topm}. Increase --mcmc-steps/--n-chains or lower --topm.",
                flush=True,
            )

        rows = []
        for i, rec in enumerate(topm_sorted, 1):
            rows.append(
                {
                    "rank": i,
                    "reward": float(rec["reward"]),
                    "yhat": float(rec["yhat"]),
                    "B1_id": rec["B1_id"],
                    "B2_id": rec["B2_id"],
                    "B3_id": rec["B3_id"],
                    "threshold": rec["threshold"],
                }
            )

        # ---- Diagnostics. ----
        ar_total = accepted_total / np.maximum(n_total, 1)
        ar_sampling = accepted_sampling / np.maximum(n_sampling, 1)
        ess_per_chain = [
            estimate_ess(np.asarray(post_burnin_logR[c], dtype=np.float64))
            for c in range(len(post_burnin_logR))
        ]
        if keep_repeats:
            unique_keys = set(
                library_key(
                    tuple(int(x) for x in row["B1_id"].split("|")),
                    tuple(int(x) for x in row["B2_id"].split("|")),
                    tuple(int(x) for x in row["B3_id"].split("|")),
                )
                for row in top_records
            )
        else:
            unique_keys = set(best_by_key.keys())
        sampling_unique = unique_keys

        summary = {
            "n_chains": int(len(post_burnin_logR)),
            "steps": int(steps),
            "burn_in": int(burn_in),
            "thin": int(thin),
            "proposal_mode": self.proposal_mode,
            "temper_ladder": None if temper_ladder is None else [float(b) for b in temper_ladder],
            "beta_cold": float(betas[0]),
            "acceptance_rate_mean": float(ar_total.mean()),
            "acceptance_rate_per_chain": [float(v) for v in ar_total],
            "acceptance_rate_sampling_mean": float(ar_sampling.mean()),
            "acceptance_rate_sampling_per_chain": [float(v) for v in ar_sampling],
            "unique_libraries_all": int(len(unique_keys)),
            "unique_libraries_sampling": int(len(sampling_unique)),
            "ess_per_chain": [float(v) for v in ess_per_chain],
            "ess_mean": float(np.mean(ess_per_chain)),
            "reward_cache_hits": int(self._cache_hits),
            "reward_cache_misses": int(self._cache_misses),
            "topm_available": int(len(topm_sorted)),
            "topm_reward_top1": float(rows[0]["reward"]) if rows else float("nan"),
            "topm_reward_mean": float(np.mean([r["reward"] for r in rows])) if rows else float("nan"),
            "wall_clock_seconds": float(wall),
        }
        return {"topm_rows": rows, "summary": summary}


def build_allowed_lists(
    df_bbs: pd.DataFrame,
    *,
    sizes: Tuple[int, int, int],
    bb_pool_size: Optional[int],
    forbidden_bb1: Optional[str],
    forbidden_bb2: Optional[str],
    forbidden_bb3: Optional[str],
) -> Tuple[List[int], List[int], List[int]]:
    """Replicate ``train_gfn``'s pool/forbidden/bb-pool-size resolution."""
    N = len(df_bbs)
    pool_arr = df_bbs["pool"].to_numpy() if "pool" in df_bbs.columns else np.zeros(N, dtype=int)
    allowed_lists: Optional[List[List[int]]] = None
    if np.any(pool_arr != 0):
        idx1 = np.where(pool_arr == 1)[0].tolist()
        idx2 = np.where(pool_arr == 2)[0].tolist()
        idx3 = np.where(pool_arr == 3)[0].tolist()
        if not (idx1 and idx2 and idx3):
            raise ValueError(
                f"Pool column present, but at least one of pools 1,2,3 is empty: "
                f"|pool1|={len(idx1)}, |pool2|={len(idx2)}, |pool3|={len(idx3)}."
            )
        allowed_lists = [idx1, idx2, idx3]
        print(
            f"[Pools] Pool-aware action sampling enabled: "
            f"|pool1|={len(idx1)}, |pool2|={len(idx2)}, |pool3|={len(idx3)}"
        )
    else:
        print(f"[Pools] No pool column found; sampling from full BB universe (N={N}).")

    id_map = df_bbs["ID"].to_numpy() if "ID" in df_bbs.columns else None
    id_to_idx = (
        {int(bb_id): int(i) for i, bb_id in enumerate(id_map.tolist())}
        if id_map is not None
        else None
    )

    forbidden_idx_per_cycle: Dict[int, set] = {1: set(), 2: set(), 3: set()}
    forbidden_ids_raw = {
        1: _parse_pipe_separated_ints(forbidden_bb1),
        2: _parse_pipe_separated_ints(forbidden_bb2),
        3: _parse_pipe_separated_ints(forbidden_bb3),
    }
    for cycle, raw_ids in forbidden_ids_raw.items():
        if not raw_ids:
            continue
        if id_to_idx is None:
            raise ValueError("Forbidden BB flags require the combined BB CSV to include an ID column.")
        found = 0
        for bb_id in raw_ids:
            idx = id_to_idx.get(int(bb_id))
            if idx is None:
                continue
            forbidden_idx_per_cycle[cycle].add(int(idx))
            found += 1
        print(f"[Forbidden] Pool {cycle}: excluding {found} BB(s) from candidate set.")

    if any(forbidden_idx_per_cycle[c] for c in (1, 2, 3)):
        if allowed_lists is None:
            allowed_lists = [list(range(N)), list(range(N)), list(range(N))]
        for cycle in (1, 2, 3):
            forb = forbidden_idx_per_cycle[cycle]
            if not forb:
                continue
            before = len(allowed_lists[cycle - 1])
            allowed_lists[cycle - 1] = [idx for idx in allowed_lists[cycle - 1] if idx not in forb]
            print(
                f"[Forbidden] Pool {cycle}: removed {before - len(allowed_lists[cycle - 1])} BB(s); "
                f"remaining={len(allowed_lists[cycle - 1])}."
            )
            if len(allowed_lists[cycle - 1]) < sizes[cycle - 1]:
                raise ValueError(
                    f"After applying forbidden BBs, pool {cycle} only has "
                    f"{len(allowed_lists[cycle - 1])} candidate(s) but size{cycle}={sizes[cycle - 1]}."
                )

    if bb_pool_size is not None:
        bb_pool_size = int(bb_pool_size)
        if N < bb_pool_size:
            raise ValueError(
                f"bbs.csv has {N} rows but --bb-pool-size={bb_pool_size} requires at least {bb_pool_size}."
            )
        if allowed_lists is None:
            allowed_lists = [
                list(range(0, bb_pool_size)),
                list(range(0, bb_pool_size)),
                list(range(0, bb_pool_size)),
            ]
            print(
                f"[Pools] No pool column — all three pools share the same first {bb_pool_size} BBs."
            )
        else:
            for cycle in (1, 2, 3):
                before = len(allowed_lists[cycle - 1])
                allowed_lists[cycle - 1] = allowed_lists[cycle - 1][:bb_pool_size]
                after = len(allowed_lists[cycle - 1])
                if after < sizes[cycle - 1]:
                    raise ValueError(
                        f"--bb-pool-size={bb_pool_size} leaves only {after} candidate(s) in pool "
                        f"{cycle} but size{cycle}={sizes[cycle - 1]}."
                    )
                print(f"[Pools] Truncated pool {cycle}: {before} -> {after} (bb_pool_size={bb_pool_size})")

    return allowed_lists[0], allowed_lists[1], allowed_lists[2]
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Metropolis-Hastings baseline sampler for Experiment 1 "
        "(targets the DeepDEL F2 library reward; F1 proxy is post-hoc only)."
    )
    ap.add_argument("--bbs", default="data/bbs_JP.csv", help="bbs.csv with 'SMILES' (optional 'ID','pool').")
    ap.add_argument("--deepdel", default="models/deepdel.pt", help="Frozen DeepDEL reward checkpoint.")
    ap.add_argument("--size1", type=int, default=20, help="Target |B1|.")
    ap.add_argument("--size2", type=int, default=20, help="Target |B2|.")
    ap.add_argument("--size3", type=int, default=20, help="Target |B3|.")
    ap.add_argument("--bb-pool-size", type=int, default=None,
                    help="Limit all three pools to the same first N BBs (by CSV order).")
    ap.add_argument("--bb-fp-bits", type=int, default=2048)
    ap.add_argument("--bb-fp-radius", type=int, default=2)
    ap.add_argument("--forbidden-bb1", default=None, help="Pipe-delimited BB IDs excluded from pool 1.")
    ap.add_argument("--forbidden-bb2", default=None, help="Pipe-delimited BB IDs excluded from pool 2.")
    ap.add_argument("--forbidden-bb3", default=None, help="Pipe-delimited BB IDs excluded from pool 3.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed threshold used for the DeepDEL reward (recorded in outputs; "
                         "also passed to threshold-conditioned models).")
    ap.add_argument("--beta", type=float, default=200.0,
                    help="Reward exponent: target p(L) ∝ R(L)^beta (beta = 1/tau).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cpu-threads", type=int, default=None)
    ap.add_argument("--ecfp-cache-dir", type=str, default=None)
    ap.add_argument("--model_is_logprobs", action="store_true",
                    help="Legacy: interpret model output as -log R when the checkpoint is not log_target.")
    ap.add_argument("--topm", type=int, default=100, help="Number of ranked candidates saved.")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--n-chains", type=int, default=8, help="Number of parallel MCMC chains.")
    ap.add_argument("--mcmc-steps", type=int, default=100000, help="Post-burn-in MH iterations per chain.")
    ap.add_argument("--burn-in", type=int, default=10000, help="Burn-in iterations per chain.")
    ap.add_argument("--thin", type=int, default=1, help="Keep every Nth post-burn-in sample for diagnostics.")
    ap.add_argument("--proposal-mode", choices=["swap1", "swap2", "cycle", "mix"], default="swap1",
                    help="Proposal family (all symmetric).")
    ap.add_argument("--temper-ladder", type=str, default=None,
                    help="Comma-separated beta ladder for parallel tempering; length must equal --n-chains.")
    ap.add_argument("--swap-replicas-every", type=int, default=100,
                    help="Attempt replica exchange every N steps (parallel tempering only).")
    ap.add_argument("--reward-cache-size", type=int, default=1000000,
                    help="Max cached log-R entries per run (reset when full).")
    ap.add_argument("--keep-repeats", action="store_true",
                    help="Retain repeated library identities in the top-m shortlist (GFN parity).")
    ap.add_argument("--log-interval", type=int, default=10000, help="Progress print frequency.")
    return ap
def main() -> None:
    args = build_parser().parse_args()
    if args.topm < 1:
        raise ValueError("--topm must be >= 1")
    if args.n_chains < 1:
        raise ValueError("--n-chains must be >= 1")
    if args.mcmc_steps < 1 or args.burn_in < 0:
        raise ValueError("--mcmc-steps must be >= 1 and --burn-in >= 0")
    if args.thin < 1:
        raise ValueError("--thin must be >= 1")
    if args.reward_cache_size < 1:
        raise ValueError("--reward-cache-size must be >= 1")

    temper_ladder: Optional[List[float]] = None
    if args.temper_ladder:
        temper_ladder = [float(x) for x in args.temper_ladder.split(",") if x.strip()]

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cpu":
        if args.cpu_threads is not None:
            torch.set_num_threads(max(1, int(args.cpu_threads)))
        elif "SLURM_CPUS_PER_TASK" in os.environ:
            try:
                torch.set_num_threads(max(1, min(32, int(os.environ["SLURM_CPUS_PER_TASK"]))))
            except Exception:
                pass
        else:
            # On shared login nodes torch would otherwise spawn one thread per
            # core (e.g. 192), thrashing the machine.  Cap at a sane default.
            torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

    # ---- Load BBs, reward model, and model-input fingerprints. ----
    df_bbs = load_bbs(args.bbs)
    smiles = df_bbs["SMILES"].tolist()
    N = len(smiles)
    id_map = df_bbs["ID"].to_numpy() if "ID" in df_bbs.columns else None

    model, meta = load_reward_model(args.deepdel, device)
    log_target = meta["log_target"]
    append_mw = meta["append_molecular_weight"]
    print(
        f"[Model] DeepDEL d_in={meta['d_in']} hidden={meta['d_h']} rho={meta['d_rho']} "
        f"pooling={meta['pooling']} output_head={meta['output_head']} "
        f"fourier_n_freqs={meta['fourier_n_freqs']} log_target={log_target} "
        f"append_mw={append_mw}"
    )

    cache_dir = args.ecfp_cache_dir or os.path.join(args.outdir, "cache")
    mw_suffix = "_mw" if append_mw else ""
    X_in = load_or_create_ecfp_cache(
        smiles,
        n_bits=int(meta["d_in"]) - (1 if append_mw else 0),
        radius=args.bb_fp_radius,
        cache_path=os.path.join(
            cache_dir, f"ecfp{int(meta['d_in'])}_r{args.bb_fp_radius}{mw_suffix}.npz"
        ),
        progress_desc=f"ECFP({meta['d_in']}) for reward oracle",
        progress_enabled=False,
        append_molecular_weight=append_mw,
    )
    X_all = torch.from_numpy(X_in).to(device)  # [N, d_in]

    sizes = (args.size1, args.size2, args.size3)
    if any(s <= 0 for s in sizes):
        raise ValueError("All sizes must be positive.")
    allowed = build_allowed_lists(
        df_bbs,
        sizes=sizes,
        bb_pool_size=args.bb_pool_size,
        forbidden_bb1=args.forbidden_bb1,
        forbidden_bb2=args.forbidden_bb2,
        forbidden_bb3=args.forbidden_bb3,
    )

    rng = np.random.default_rng(args.seed)
    sampler = MCMCSampler(
        sizes=sizes,
        allowed=allowed,
        X_all=X_all,
        model=model,
        log_target=log_target,
        model_is_logprobs=args.model_is_logprobs,
        threshold=args.threshold,
        beta=args.beta,
        proposal_mode=args.proposal_mode,
        rng=rng,
        reward_cache_size=args.reward_cache_size,
    )

    trace_path = os.path.join(args.outdir, "mcmc_trace.csv")
    result = sampler.run(
        n_chains=args.n_chains,
        steps=args.mcmc_steps,
        burn_in=args.burn_in,
        thin=args.thin,
        temper_ladder=temper_ladder,
        swap_replicas_every=args.swap_replicas_every,
        topm=args.topm,
        keep_repeats=args.keep_repeats,
        id_map=id_map,
        trace_path=trace_path,
        log_interval=args.log_interval,
    )
    topm_rows = result["topm_rows"]
    summary = result["summary"]

    # ---- Write outputs. ----
    topm_df = pd.DataFrame(topm_rows)
    topm_csv = os.path.join(args.outdir, "topm_rewards.csv")
    topm_df.to_csv(topm_csv, index=False)
    print(f"[Top-{args.topm}] Saved {len(topm_rows)} ranked candidates to {topm_csv}")

    summary_path = os.path.join(args.outdir, "mcmc_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"[MCMC] Wrote summary to {summary_path}")

    with open(os.path.join(args.outdir, "wall_clock_seconds.txt"), "w") as fh:
        fh.write(f"{summary['wall_clock_seconds']:.6f}\n")

    config_rows = []
    for key, value in sorted(vars(args).items()):
        config_rows.append({"key": key, "value": value})
    for key, value in sorted(meta.items()):
        config_rows.append({"key": f"checkpoint.{key}", "value": value})
    config_rows.append({"key": "n_bbs", "value": N})
    config_rows.append({"key": "pool_sizes", "value": "|".join(str(len(a)) for a in allowed)})
    pd.DataFrame(config_rows).to_csv(os.path.join(args.outdir, "run_config.csv"), index=False)

    print(
        f"[MCMC] Done — acceptance={summary['acceptance_rate_mean']:.3f} "
        f"unique={summary['unique_libraries_all']} "
        f"ess_mean={summary['ess_mean']:.0f} "
        f"wall={summary['wall_clock_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()