"""Shared AutoDock-proxy product scoring utilities.

This module contains the reusable fast path originally implemented inside
``deepdel.generate_dataset``: represent Morgan fingerprints as sparse on-bit
index arrays, stage them into reusable dense batches, and run proxy inference
from one parent process.  It is intentionally independent of how libraries are
sampled/enumerated so both random DeepDEL dataset generation and fixed top-m
evaluation can use it.
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional

import numpy as np


def configure_cpu_thread_env(torch_num_threads: Optional[int], torch_interop_threads: Optional[int]) -> None:
    """Constrain CPU math thread pools before torch is imported."""

    requested = int(torch_num_threads or 0)
    if requested > 0:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(requested)

    interop = int(torch_interop_threads or 0)
    if interop > 0:
        os.environ["TORCH_NUM_INTEROP_THREADS"] = str(interop)


def configure_torch_runtime(torch_module, torch_num_threads: Optional[int], torch_interop_threads: Optional[int]) -> None:
    """Apply torch CPU thread limits after torch has been imported."""

    requested = int(torch_num_threads or 0)
    interop = int(torch_interop_threads or 0)
    if requested > 0:
        torch_module.set_num_threads(requested)
    if interop > 0:
        try:
            torch_module.set_num_interop_threads(interop)
        except RuntimeError as exc:
            print(f"[proxy-scoring][warn] Could not set torch inter-op threads to {interop}: {exc}", flush=True)


def smi_to_sparse_fp(smi: str, *, radius: int, n_bits: int, fp_dtype: np.dtype) -> Optional[np.ndarray]:
    """Return Morgan fingerprint on-bit indices, or ``None`` for invalid SMILES."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    return np.asarray(list(fp.GetOnBits()), dtype=fp_dtype)


class SparseFPProxyScorer:
    """Score sparse ECFP rows with either the NN or RF AutoDock proxy.

    For NN proxies, this mirrors the dataset-generation fast path: a reusable
    dense torch tensor is filled from sparse indices, transferred to the proxy
    device, inferred, then zeroed only at touched cells.  For RF proxies, sparse
    rows are densified in numpy batches and passed to ``proxy.predict_ecfp``.
    """

    def __init__(
        self,
        proxy,
        *,
        batch_size: int,
        torch_num_threads: Optional[int] = None,
        torch_interop_threads: Optional[int] = None,
        score_log_seconds: float = 0.0,
    ):
        self.proxy = proxy
        self.n_bits = int(proxy.n_bits)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.score_log_seconds = float(score_log_seconds)
        self._is_nn = hasattr(proxy, "device") and hasattr(proxy, "model")

        self._torch = None
        self._device = None
        self._use_cuda = False
        self._staging_cpu = None
        if self._is_nn:
            import torch

            configure_torch_runtime(torch, torch_num_threads, torch_interop_threads)
            self._torch = torch
            self._device = proxy.device
            self._use_cuda = torch.device(self._device).type == "cuda"
            print(
                "[proxy-scoring] Torch runtime: "
                f"device={self._device}, cuda={self._use_cuda}, "
                f"num_threads={torch.get_num_threads()}, "
                f"num_interop_threads={torch.get_num_interop_threads()}, "
                f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
                f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}, "
                f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}",
                flush=True,
            )
            self._staging_cpu = torch.zeros(
                (self.batch_size, self.n_bits),
                dtype=torch.float32,
                pin_memory=self._use_cuda,
            )

    def score(self, sparse_rows: List[np.ndarray]) -> np.ndarray:
        """Score sparse fingerprint rows in order."""

        if self._is_nn:
            return self._score_nn(sparse_rows)
        return self._score_dense_numpy(sparse_rows)

    def _score_dense_numpy(self, sparse_rows: List[np.ndarray]) -> np.ndarray:
        n = len(sparse_rows)
        if n == 0:
            return np.array([], dtype=np.float32)
        out = np.empty(n, dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            X = np.zeros((end - start, self.n_bits), dtype=np.float32)
            for r, idx in enumerate(sparse_rows[start:end]):
                if idx.size:
                    X[r, np.asarray(idx, dtype=np.int64)] = 1.0
            y = self.proxy.predict_ecfp(X)
            out[start:end] = np.asarray(y, dtype=np.float32).reshape(-1)[: end - start]
        return out

    def _score_nn(self, sparse_rows: List[np.ndarray]) -> np.ndarray:
        torch = self._torch
        if torch is None or self._staging_cpu is None:
            raise RuntimeError("NN scorer was not initialized correctly")
        n = len(sparse_rows)
        if n == 0:
            return np.array([], dtype=np.float32)

        out = np.empty(n, dtype=np.float32)
        proxy = self.proxy
        model = proxy.model
        model.eval()
        device = self._device
        use_cuda = self._use_cuda
        n_batches = int(math.ceil(n / self.batch_size))
        log_threshold = float(self.score_log_seconds)
        score_start = time.perf_counter()
        next_log = score_start + log_threshold if log_threshold > 0 else float("inf")

        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, n, self.batch_size), start=1):
                end = min(start + self.batch_size, n)
                B = end - start
                batch_t0 = time.perf_counter()
                if log_threshold > 0 and (batch_idx == 1 or batch_t0 >= next_log):
                    print(
                        f"[proxy-scoring] batch {batch_idx}/{n_batches} start "
                        f"rows={start}:{end} elapsed={batch_t0 - score_start:.1f}s",
                        flush=True,
                    )
                    next_log = batch_t0 + log_threshold

                view = self._staging_cpu[:B]
                touched_cols: List[np.ndarray] = []
                touched_rows: List[int] = []
                for r, idx in enumerate(sparse_rows[start:end]):
                    if idx.size == 0:
                        continue
                    cols = torch.from_numpy(np.asarray(idx, dtype=np.int64))
                    view[r].index_fill_(0, cols, 1.0)
                    touched_cols.append(idx)
                    touched_rows.append(r)

                xb = view.to(device, non_blocking=use_cuda)
                yb = model(xb).detach().to("cpu").numpy().astype(np.float32, copy=False).reshape(-1)
                if proxy.y_norm_enabled:
                    yb = yb * float(proxy.y_std) + float(proxy.y_mean)
                out[start:end] = yb

                for r, idx in zip(touched_rows, touched_cols):
                    if idx.size == 0:
                        continue
                    cols = torch.from_numpy(np.asarray(idx, dtype=np.int64))
                    view[r].index_fill_(0, cols, 0.0)

        return out
