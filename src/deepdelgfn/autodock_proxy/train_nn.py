#!/usr/bin/env python3
"""train_autodock_proxy_nn.py

Train an ECFP (Morgan fingerprint) -> neural network regression model to predict
AutoDock/Vina docking scores.

This is a PyTorch-based alternative to `train_autodock_proxy.py` (which uses a
RandomForestRegressor). The input handling (CSV recursion, columns, etc.) is
kept intentionally similar so existing workflows can swap scripts easily.

Checkpoint format (.pt):
  {
    "model_state_dict": ...,
    "model_config": {...},
    "fp_config": {"radius": int, "n_bits": int},
    "data_config": {"smiles_col": str, "target_col": str},
    "y_normalization": {"enabled": bool, "mean": float, "std": float},
    "args": {...}
  }
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem, DataStructs

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# PyTorch is expected to be provided by your environment modules.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


RDLogger.DisableLog("rdApp.*")  # quiet RDKit warnings


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Convert a SMILES string to a dense ECFP (Morgan fingerprint) bit
    vector.  Returns ``None`` if the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def collect_csvs(paths: Iterable[str]) -> List[str]:
    """Recursively collect CSV files from a list of file and directory
    paths.  Deduplicates and returns sorted absolute paths.
    """
    csvs: List[str] = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            sys.exit(f"[ERROR] Path not found: {pth}")
        if pth.is_file():
            if pth.suffix.lower() == ".csv":
                csvs.append(str(pth.resolve()))
            else:
                sys.exit(f"[ERROR] Not a CSV file: {pth}")
        elif pth.is_dir():
            found = [str(x.resolve()) for x in pth.rglob("*.csv")]
            if not found:
                print(f"[WARN] No CSVs found under directory: {pth}")
            csvs.extend(found)
        else:
            sys.exit(f"[ERROR] Unsupported path type: {pth}")

    return sorted(set(csvs))


def ensure_parent_dir(path_str: str) -> None:
    """Create parent directories for a file path if they don't exist."""
    p = Path(path_str)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


# ---------------------- ECFP cache helpers (v2: incremental) ----------------------
#
# The cache is a single .npz holding the row-concatenation of per-source CSV
# feature blocks, plus a JSON sidecar that records, for each source CSV, its
# (path, size, mtime_ns) AND its slice into the concatenated arrays
# (`start`, `count`). On the next run, sources whose (size, mtime_ns) still
# match are kept verbatim (we slice their feature blocks out of the cache),
# and only new/changed sources are featurized.
#
# Featurization-defining options (`radius`, `n_bits`, `smiles_col`,
# `target_col`, `sep`) DO gate cache reuse. Filters that act on the assembled
# rows after loading (`dedup_smiles`, `ignore_zero_labels`) do NOT — they are
# applied as post-cache filters, so toggling them never invalidates the cache.

CACHE_VERSION = 2


def _meta_path(cache_path: str) -> str:
    """Return the JSON metadata sidecar path for a given ``.npz`` ECFP
    cache file.
    """
    return str(cache_path) + ".meta.json"


def _stat_source(path: str) -> Dict[str, Any]:
    """Return a dict with ``path``, ``size``, and ``mtime_ns`` for a CSV
    source, used for cache validation.
    """
    try:
        st = os.stat(path)
        return {"path": str(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except OSError:
        # Unstat-able files won't match anything saved -> forces recompute for that source.
        return {"path": str(path), "size": None, "mtime_ns": None}


def _build_global_meta(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the featurization-defining metadata dict (version,
    smiles_col, target_col, radius, n_bits, sep) that gates cache reuse.
    """
    return {
        "version": CACHE_VERSION,
        "smiles_col": str(args.smiles_col),
        "target_col": str(args.target_col),
        "radius": int(args.radius),
        "n_bits": int(args.n_bits),
        "sep": str(args.sep),
    }


def _global_meta_matches(saved: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str]:
    """Compare saved cache metadata against expected values.

    Returns ``(True, "")`` on match, or ``(False, reason)`` on mismatch.
    """
    keys = ["version", "smiles_col", "target_col", "radius", "n_bits", "sep"]
    for k in keys:
        if saved.get(k) != expected.get(k):
            return False, f"meta mismatch on '{k}': saved={saved.get(k)!r} expected={expected.get(k)!r}"
    return True, ""


def _featurize_one_csv(
    path: str,
    *,
    smiles_col: str,
    target_col: str,
    radius: int,
    n_bits: int,
    sep: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a single CSV and featurize its rows into ``(X_uint8, y_float32,
    smiles_object)``.  Drops missing/non-numeric values and unparseable SMILES.

    No dedup or zero-label filtering is done here — those are post-cache
    filters applied to the assembled rows.  CSVs missing the required SMILES
    or target column are skipped by returning an empty block.
    """
    print(f"[INFO]   featurizing {path}")
    df = pd.read_csv(path, sep=sep)
    if smiles_col not in df.columns or target_col not in df.columns:
        missing = [col for col in (smiles_col, target_col) if col not in df.columns]
        print(
            f"[WARN]   skipping {path}: missing required column(s) {missing}; "
            f"expected '{smiles_col}' and '{target_col}'."
        )
        return (
            np.zeros((0, n_bits), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
            np.array([], dtype=object),
        )
    df = df.dropna(subset=[smiles_col, target_col])
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    smiles_list = df[smiles_col].astype(str).tolist()
    targets = df[target_col].astype(float).values

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    s_list: List[str] = []
    for sm, t in zip(smiles_list, targets):
        fp = smiles_to_ecfp(sm, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        X_list.append(fp)
        y_list.append(float(t))
        s_list.append(sm)

    if not X_list:
        return (
            np.zeros((0, n_bits), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
            np.array([], dtype=object),
        )

    X = np.vstack(X_list).astype(np.uint8, copy=False)
    y = np.asarray(y_list, dtype=np.float32)
    s = np.asarray(s_list, dtype=object)
    print(f"[INFO]     -> {X.shape[0]} rows from {path}")
    return X, y, s


def _load_or_update_ecfp_cache(
    cache_path: Optional[str],
    csv_files: List[str],
    args: argparse.Namespace,
    *,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or incrementally update an ECFP feature cache (``.npz``).

    For sources whose (path, size, mtime_ns) match the cached copy, the
    previous feature block is reused; only new/changed CSVs are featurized.
    This enables incremental retraining with minimal RDKit work.

    The resulting ``X`` is uint8 (bit matrix); the caller is responsible for
    casting to float32 at training time and for applying any post-cache
    filters (dedup, zero-label filter, etc.).
    """
    expected_global = _build_global_meta(args)
    smiles_col = str(args.smiles_col)
    target_col = str(args.target_col)
    radius = int(args.radius)
    n_bits = int(args.n_bits)
    sep = str(args.sep)

    # Load existing cache (if any & valid).
    saved_meta: Optional[Dict[str, Any]] = None
    saved_X: Optional[np.ndarray] = None
    saved_y: Optional[np.ndarray] = None
    saved_smiles: Optional[np.ndarray] = None

    if cache_path and not overwrite:
        cache_p = Path(cache_path)
        meta_p = Path(_meta_path(cache_path))
        if cache_p.exists() and meta_p.exists():
            try:
                with open(meta_p, "r") as f:
                    saved_meta = json.load(f)
            except Exception as e:
                print(f"[WARN] Could not read cache meta {meta_p}: {e}")
                saved_meta = None

            if saved_meta is not None:
                ok, reason = _global_meta_matches(saved_meta, expected_global)
                if not ok:
                    print(f"[INFO] ECFP cache at {cache_p} has incompatible config ({reason}); rebuilding from scratch.")
                    saved_meta = None
                else:
                    try:
                        data = np.load(cache_p, allow_pickle=True)
                        saved_X = data["X"]
                        saved_y = data["y"]
                        saved_smiles = data["smiles"]
                        if saved_X.shape[0] != saved_y.shape[0] or saved_X.shape[0] != saved_smiles.shape[0]:
                            raise ValueError(
                                f"cache arrays have inconsistent row counts: "
                                f"X={saved_X.shape}, y={saved_y.shape}, smiles={saved_smiles.shape}"
                            )
                    except Exception as e:
                        print(f"[WARN] Failed to load ECFP cache {cache_p}: {e}; rebuilding from scratch.")
                        saved_meta = None
                        saved_X = None
                        saved_y = None
                        saved_smiles = None

    saved_sources = (saved_meta or {}).get("sources", []) or []
    saved_by_path: Dict[str, Dict[str, Any]] = {s["path"]: s for s in saved_sources}

    # Fast path: if the cache exactly covers the current source list in the
    # same order and every source stat still matches, return the cached arrays
    # directly.  This avoids vstack/concatenate plus cache rewrite, which can
    # otherwise transiently double the full ECFP matrix in RAM on common
    # "nothing changed, retrain proxy" runs.
    if saved_X is not None and saved_y is not None and saved_smiles is not None:
        exact_cache_hit = len(saved_sources) == len(csv_files)
        total_count = 0
        if exact_cache_hit:
            for expected_start, (path, src) in enumerate(zip(csv_files, saved_sources)):
                cur = _stat_source(path)
                start = src.get("start")
                count = src.get("count")
                if (
                    src.get("path") != str(path)
                    or src.get("size") != cur.get("size")
                    or src.get("mtime_ns") != cur.get("mtime_ns")
                    or not isinstance(start, int)
                    or not isinstance(count, int)
                    or int(start) != total_count
                    or int(count) < 0
                ):
                    exact_cache_hit = False
                    break
                total_count += int(count)
        if exact_cache_hit and total_count == int(saved_X.shape[0]):
            print(
                f"[INFO] ECFP cache: exact hit for {len(csv_files)} source(s); "
                f"using cached arrays directly ({saved_X.shape[0]} rows) without rewriting."
            )
            return (
                saved_X.astype(np.uint8, copy=False),
                saved_y.astype(np.float32, copy=False),
                saved_smiles,
            )

    # Decide for each current CSV: keep slice from saved cache, or recompute.
    out_X_blocks: List[np.ndarray] = []
    out_y_blocks: List[np.ndarray] = []
    out_smiles_blocks: List[np.ndarray] = []
    new_sources: List[Dict[str, Any]] = []

    n_kept = 0
    n_kept_rows = 0
    n_new = 0
    n_new_rows = 0

    for path in csv_files:
        cur = _stat_source(path)
        prev = saved_by_path.get(str(path))
        can_keep = (
            saved_X is not None
            and prev is not None
            and prev.get("size") == cur.get("size")
            and prev.get("mtime_ns") == cur.get("mtime_ns")
            and isinstance(prev.get("start"), int)
            and isinstance(prev.get("count"), int)
        )
        if can_keep:
            start = int(prev["start"])
            count = int(prev["count"])
            end = start + count
            X_blk = saved_X[start:end]
            y_blk = saved_y[start:end]
            s_blk = saved_smiles[start:end]
            # Sanity check: dimensions still consistent with current n_bits.
            if X_blk.ndim != 2 or X_blk.shape[1] != n_bits:
                # Shouldn't happen because n_bits is in the global meta, but be safe.
                print(f"[WARN] Cached slice for {path} has wrong width; recomputing.")
                can_keep = False

        if can_keep:
            out_X_blocks.append(X_blk)
            out_y_blocks.append(y_blk)
            out_smiles_blocks.append(s_blk)
            n_kept += 1
            n_kept_rows += int(X_blk.shape[0])
            new_sources.append({
                "path": cur["path"],
                "size": cur["size"],
                "mtime_ns": cur["mtime_ns"],
                # `start` / `count` filled in after concatenation.
            })
        else:
            X_blk, y_blk, s_blk = _featurize_one_csv(
                path,
                smiles_col=smiles_col,
                target_col=target_col,
                radius=radius,
                n_bits=n_bits,
                sep=sep,
            )
            out_X_blocks.append(X_blk)
            out_y_blocks.append(y_blk)
            out_smiles_blocks.append(s_blk)
            n_new += 1
            n_new_rows += int(X_blk.shape[0])
            new_sources.append({
                "path": cur["path"],
                "size": cur["size"],
                "mtime_ns": cur["mtime_ns"],
            })

    # Concatenate (handle empty case).
    if out_X_blocks:
        X_all = np.vstack(out_X_blocks).astype(np.uint8, copy=False) if any(b.size for b in out_X_blocks) \
            else np.zeros((0, n_bits), dtype=np.uint8)
        y_all = np.concatenate(out_y_blocks).astype(np.float32, copy=False) if any(b.size for b in out_y_blocks) \
            else np.zeros((0,), dtype=np.float32)
        smiles_all = np.concatenate(out_smiles_blocks) if any(b.size for b in out_smiles_blocks) \
            else np.array([], dtype=object)
    else:
        X_all = np.zeros((0, n_bits), dtype=np.uint8)
        y_all = np.zeros((0,), dtype=np.float32)
        smiles_all = np.array([], dtype=object)

    # Fill in slice offsets for the new meta.
    cursor = 0
    for src, blk in zip(new_sources, out_X_blocks):
        src["start"] = int(cursor)
        src["count"] = int(blk.shape[0])
        cursor += int(blk.shape[0])

    n_dropped = max(0, len(saved_sources) - n_kept)
    print(
        f"[INFO] ECFP cache: kept={n_kept} sources ({n_kept_rows} rows), "
        f"new={n_new} sources ({n_new_rows} rows), dropped={n_dropped} stale sources; "
        f"total rows after assembly: {X_all.shape[0]}"
    )

    # Persist updated cache. We always rewrite if a cache_path was given so
    # newly featurized CSVs are saved even when most sources were kept.
    if cache_path:
        meta_to_save = dict(expected_global)
        meta_to_save["sources"] = new_sources
        _save_ecfp_cache(cache_path, X_all, y_all, smiles_all, meta_to_save)

    return X_all, y_all, smiles_all


def _save_ecfp_cache(
    cache_path: str,
    X: np.ndarray,
    y: np.ndarray,
    smiles: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """Atomically save the ECFP feature matrix and metadata sidecar to
    disk using temporary files and ``os.replace``.
    """
    ensure_parent_dir(cache_path)
    cache_p = Path(cache_path)
    meta_p = Path(_meta_path(cache_path))
    tmp_suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_cache_p = cache_p.with_name(cache_p.name + tmp_suffix)
    tmp_meta_p = meta_p.with_name(meta_p.name + tmp_suffix)
    X_to_save = X.astype(np.uint8, copy=False)
    y_to_save = y.astype(np.float32, copy=False)
    try:
        # Write to unique temp files in the same directory, then atomically
        # replace the public paths. This prevents readers from observing a
        # partially-written .npz (which manifests as "File is not a zip file").
        with open(tmp_cache_p, "wb") as f:
            np.savez_compressed(f, X=X_to_save, y=y_to_save, smiles=np.asarray(smiles, dtype=object))
            f.flush()
            os.fsync(f.fileno())
        with open(tmp_meta_p, "w") as f:
            json.dump(meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_cache_p, cache_p)
        os.replace(tmp_meta_p, meta_p)
    except Exception:
        # Best-effort cleanup. Ignore unlink errors so the original exception
        # remains visible to the caller.
        try:
            tmp_cache_p.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            tmp_meta_p.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    print(f"[INFO] Saved ECFP cache to {cache_path} (and meta to {_meta_path(cache_path)}).")


class NumpyDataset(Dataset):
    """Simple PyTorch dataset wrapping ``(X, y)`` numpy arrays.

    Kept for backwards compatibility; the training loop below uses an
    on-device index-based mini-batcher by default.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        return x, y


def _to_uint8_tensor(X: np.ndarray, *, pin_memory: bool = False) -> torch.Tensor:
    """Wrap a uint8 ndarray as a contiguous CPU tensor (zero-copy when
    possible).  Optionally pins memory for faster GPU transfers.
    """
    t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.uint8))
    if pin_memory and torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except Exception as e:  # pragma: no cover - defensive
            print(f"[WARN] Could not pin X tensor ({e}); falling back to pageable memory.")
    return t


def _to_float32_tensor(y: np.ndarray, *, pin_memory: bool = False) -> torch.Tensor:
    """Wrap a float32 ndarray as a contiguous CPU tensor.  Optionally pins
    memory for faster GPU transfers.
    """
    t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    if pin_memory and torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except Exception as e:  # pragma: no cover - defensive
            print(f"[WARN] Could not pin y tensor ({e}); falling back to pageable memory.")
    return t


def _iter_indexed_batches(
    X_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    row_idx_cpu: torch.Tensor,
    batch_size: int,
    *,
    device: torch.device,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    standardize_y: bool = False,
):
    """Yield ``(xb_float32_on_device, yb_float32_on_device)`` mini-batches
    by indexing into a single uint8 feature matrix and float32 target vector.

    Supports on-the-fly y-standardization to avoid materializing normalized
    copies.
    """
    n = int(row_idx_cpu.numel())
    if n == 0:
        return
    if shuffle:
        order = torch.randperm(n, generator=generator)
    else:
        order = torch.arange(n)

    use_cuda = device.type == "cuda"
    for start in range(0, n, batch_size):
        split_rows = row_idx_cpu.index_select(0, order[start : start + batch_size])
        xb_u8 = X_cpu.index_select(0, split_rows)
        yb = y_cpu.index_select(0, split_rows)
        if standardize_y:
            yb = (yb - float(y_mean)) / float(y_std)
        if use_cuda:
            # non_blocking only helps with pinned host memory. The default path
            # intentionally avoids full-dataset pinning to keep RAM bounded.
            xb_u8 = xb_u8.to(device, non_blocking=bool(X_cpu.is_pinned()))
            yb = yb.to(device, non_blocking=bool(y_cpu.is_pinned()))
        xb = xb_u8.to(dtype=torch.float32)
        yield xb, yb


class ECFPMLP(nn.Module):
    def __init__(
        self,
        n_bits: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        layers: List[nn.Module] = []
        in_dim = n_bits
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[np.ndarray] = []
    yhs: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        ys.append(yb.detach().cpu().numpy())
        yhs.append(yhat.detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(yhs, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Train ECFP-based NN on docking scores (PyTorch).")
    ap.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        help="One or more CSV files and/or directories containing CSVs (directories are searched recursively).",
    )
    ap.add_argument("--smiles_col", default="smiles", help="Column with SMILES (default: smiles)")
    ap.add_argument(
        "--target_col", default="docking_score", help="Column with target (default: docking_score)"
    )
    ap.add_argument("--radius", type=int, default=2, help="ECFP radius (default: 2)")
    ap.add_argument("--n_bits", type=int, default=4096, help="ECFP length (default: 4096)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default: ','). Set to \\t for TSV, etc.")
    ap.add_argument(
        "--ignore_zero_labels",
        action="store_true",
        help="Ignore rows whose target label is exactly 0.0 before creating train/validation/test splits.",
    )
    ap.add_argument(
        "--clip_positive_targets",
        action="store_true",
        help="Clip positive target values to 0.0 before training (cache remains unchanged).",
    )

    # Data deduplication
    ap.add_argument(
        "--dedup_smiles",
        action="store_true",
        help=(
            "Deduplicate rows by SMILES BEFORE featurization/training, keeping the first occurrence. "
            "This can drastically reduce RDKit featurization time when many duplicates exist."
        ),
    )

    # NN / training hyperparameters
    ap.add_argument("--hidden_dim", type=int, default=512, help="Hidden width (default: 512)")
    ap.add_argument("--n_layers", type=int, default=2, help="Number of hidden layers (default: 2)")
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout probability (default: 0.1)")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 1e-3)")
    ap.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay (default: 1e-5)")
    ap.add_argument("--batch_size", type=int, default=512, help="Batch size (default: 512)")
    ap.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50)")
    ap.add_argument("--patience", type=int, default=8, help="Early stopping patience on val loss (default: 8)")
    ap.add_argument("--val_size", type=float, default=0.1, help="Validation fraction from train set (default: 0.1)")
    ap.add_argument(
        "--standardize_y",
        action="store_true",
        help="Standardize targets (train mean/std) for NN stability; predictions are unstandardized.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cpu' or 'cuda'). Default: auto-detect.",
    )

    ap.add_argument(
        "--model_out",
        default="models/autodock_model_nn.pt",
        help="Output model checkpoint (.pt)",
    )
    ap.add_argument(
        "--preds_out",
        default="outputs/autodock_proxy/predictions_nn.csv",
        help="Output predictions file",
    )

    # ECFP cache
    ap.add_argument(
        "--ecfp_cache",
        default=None,
        help=(
            "Optional path to a cached ECFP feature matrix (.npz). If provided and the file "
            "exists with matching metadata (sources, radius, n_bits, columns, dedup/filter "
            "flags), it will be loaded instead of recomputing fingerprints. Otherwise, "
            "fingerprints are computed and saved to this path for next time."
        ),
    )
    ap.add_argument(
        "--overwrite_ecfp_cache",
        action="store_true",
        help="If set, recompute ECFPs and overwrite an existing cache file at --ecfp_cache.",
    )
    ap.add_argument(
        "--ecfp_cache_only",
        action="store_true",
        help="Compute (or load) the ECFP cache then exit without training. Requires --ecfp_cache.",
    )

    args = ap.parse_args()

    # Repro
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # A100/H100 perf: enable TF32 + cuDNN benchmark for our (fixed-shape) MLP.
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            n_gpu = torch.cuda.device_count()
            cur = torch.cuda.current_device()
            print(f"[INFO] CUDA devices visible: {n_gpu} (using cuda:{cur} = {torch.cuda.get_device_name(cur)})")
            if n_gpu > 1:
                print(
                    "[INFO] NOTE: train_nn currently uses a single GPU. "
                    "The other CUDA devices visible to this process will sit idle. "
                    "Use a single-GPU SLURM allocation, or extend this script with DDP, to avoid wasting them."
                )
        except Exception as e:  # pragma: no cover - defensive
            print(f"[WARN] Could not query CUDA devices: {e}")


    csv_files = collect_csvs(args.inputs)
    if not csv_files:
        sys.exit("[ERROR] No CSVs found from provided inputs.")

    if args.ecfp_cache_only and not args.ecfp_cache:
        sys.exit("[ERROR] --ecfp_cache_only requires --ecfp_cache to be set.")

    # ---------------- ECFP cache: incremental load/update ----------------
    # Featurize per-CSV, reusing slices from the existing cache for any source
    # whose (path, size, mtime_ns) is unchanged. Only new/changed CSVs incur
    # RDKit work; the cache is then rewritten with the assembled rows.
    print(f"[INFO] Assembling ECFPs over {len(csv_files)} CSV(s) (cache={args.ecfp_cache!r}, overwrite={bool(args.overwrite_ecfp_cache)})...")
    X_uint8, y, smiles = _load_or_update_ecfp_cache(
        args.ecfp_cache,
        csv_files,
        args,
        overwrite=bool(args.overwrite_ecfp_cache),
    )

    if args.ecfp_cache_only:
        print("[INFO] --ecfp_cache_only set; cache is ready, exiting before training.")
        return

    if X_uint8.shape[0] == 0:
        sys.exit("[ERROR] No valid molecules found across input CSVs.")

    # ---------------- Post-cache filters (do NOT invalidate the cache) ----------------
    # Keep filters as row indices instead of slicing X_uint8.  Slicing the ECFP
    # matrix for zero-label filtering / deduplication can duplicate tens or
    # hundreds of GB.  y/smiles indexing below is small relative to X, and X is
    # gathered only batch-by-batch during training.
    active_idx = np.arange(int(y.shape[0]), dtype=np.int64)

    if args.clip_positive_targets:
        n_clipped = int(np.count_nonzero(y[active_idx] > 0.0))
        y = np.minimum(y, 0.0)
        print(f"[INFO] Post-cache transform: clipped {n_clipped} positive targets to 0.0.")

    if args.ignore_zero_labels:
        before = int(active_idx.shape[0])
        active_idx = active_idx[y[active_idx] < -5.0]
        removed = before - int(active_idx.shape[0])
        print(f"[INFO] Post-cache filter: dropped {removed} rows with target >= -5.0.")

    if args.dedup_smiles:
        before = int(active_idx.shape[0])
        # First-occurrence dedup within the active rows, preserving input CSV
        # order, without slicing X_uint8.
        _, first_pos = np.unique(smiles[active_idx], return_index=True)
        active_idx = active_idx[np.sort(first_pos)]
        after = int(active_idx.shape[0])
        print(f"[INFO] Post-cache filter: deduplicated SMILES {before} -> {after} (kept first occurrence)")

    if active_idx.shape[0] == 0:
        sys.exit("[ERROR] No rows remain after post-cache filters.")

    # Keep the bit matrix as uint8: ~4x less host memory and ~4x less PCIe
    # traffic vs float32. We cast to float32 on the GPU inside the batch loop.
    y = y.astype(np.float32, copy=False)

    # Split row indices only. train_test_split copies the index vector, not the
    # huge ECFP matrix.
    train_idx, test_idx = train_test_split(
        active_idx, test_size=args.test_size, random_state=args.seed
    )

    # Split validation indices from train indices for early stopping.
    if args.val_size and args.val_size > 0:
        train_idx, val_idx = train_test_split(
            train_idx, test_size=args.val_size, random_state=args.seed
        )
    else:
        val_idx = np.array([], dtype=np.int64)

    # Optional y standardization.  Compute train statistics from an indexed y
    # view; normalization itself is applied per mini-batch to avoid y copies.
    y_train = y[train_idx]
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-12)
    del y_train

    # Materialize one CPU tensor view over the full uint8 matrix and one y
    # tensor.  By default these are not full-dataset pinned copies; batch-level
    # gathers keep peak memory close to one X matrix plus small index vectors.
    print(
        f"[INFO] Materializing CPU tensors: "
        f"X_all={X_uint8.shape} uint8 (~{X_uint8.nbytes / 1e9:.2f} GB), "
        f"rows train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, "
        f"standardize_y={bool(args.standardize_y)}"
    )
    X_all_t = _to_uint8_tensor(X_uint8, pin_memory=False)
    y_all_t = _to_float32_tensor(y, pin_memory=False)
    train_idx_t = torch.from_numpy(np.ascontiguousarray(train_idx, dtype=np.int64))
    val_idx_t = torch.from_numpy(np.ascontiguousarray(val_idx, dtype=np.int64))
    test_idx_t = torch.from_numpy(np.ascontiguousarray(test_idx, dtype=np.int64))

    model = ECFPMLP(
        n_bits=int(args.n_bits),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_trainable:,}")

    # opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    opt = torch.optim.RMSprop(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.MSELoss()

    # Deterministic shuffle generator for reproducibility (CPU side).
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(int(args.seed))

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    print("[INFO] Training MLP...")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for xb, yb in _iter_indexed_batches(
            X_all_t, y_all_t, train_idx_t, int(args.batch_size),
            device=device, shuffle=True, generator=shuffle_gen,
            y_mean=y_mean, y_std=y_std, standardize_y=bool(args.standardize_y),
        ):
            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            bs = int(yb.shape[0])
            loss_sum += float(loss.detach().item()) * bs
            n_seen += bs

        train_loss = (loss_sum / n_seen) if n_seen > 0 else float("nan")

        # Validation
        if int(val_idx_t.numel()) > 0:
            with torch.no_grad():
                model.eval()
                vloss_sum = 0.0
                v_seen = 0
                for xb, yb in _iter_indexed_batches(
                    X_all_t, y_all_t, val_idx_t, int(args.batch_size),
                    device=device, shuffle=False,
                    y_mean=y_mean, y_std=y_std, standardize_y=bool(args.standardize_y),
                ):
                    yhat = model(xb)
                    bs = int(yb.shape[0])
                    vloss_sum += float(loss_fn(yhat, yb).detach().item()) * bs
                    v_seen += bs
                val_loss = (vloss_sum / v_seen) if v_seen > 0 else float("nan")
            print(f"[INFO] Epoch {epoch:03d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")


            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if args.patience and bad_epochs >= int(args.patience):
                    print(f"[INFO] Early stopping triggered (patience={args.patience}).")
                    break
        else:
            print(f"[INFO] Epoch {epoch:03d} | train MSE: {train_loss:.6f}")

    # Restore best val model (if we had a val split)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Ensure output directories exist
    ensure_parent_dir(args.model_out)
    ensure_parent_dir(args.preds_out)

    # Save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "arch": "ECFPMLP",
            "n_bits": int(args.n_bits),
            "hidden_dim": int(args.hidden_dim),
            "n_layers": int(args.n_layers),
            "dropout": float(args.dropout),
        },
        "fp_config": {"radius": int(args.radius), "n_bits": int(args.n_bits)},
        "data_config": {"smiles_col": args.smiles_col, "target_col": args.target_col},
        "y_normalization": {
            "enabled": bool(args.standardize_y),
            "mean": float(y_mean),
            "std": float(y_std),
        },
        "args": vars(args),
    }
    torch.save(ckpt, args.model_out)
    print(f"[INFO] Model saved to {args.model_out}")

    # Evaluate on test set (same on-device batching strategy as training).
    model.eval()
    y_true_chunks: List[np.ndarray] = []
    y_pred_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in _iter_indexed_batches(
            X_all_t, y_all_t, test_idx_t, int(args.batch_size),
            device=device, shuffle=False,
            y_mean=y_mean, y_std=y_std, standardize_y=bool(args.standardize_y),
        ):
            yhat = model(xb)
            y_true_chunks.append(yb.detach().cpu().numpy())
            y_pred_chunks.append(yhat.detach().cpu().numpy())
    y_true_n = np.concatenate(y_true_chunks, axis=0) if y_true_chunks else np.array([], dtype=np.float32)
    y_pred_n = np.concatenate(y_pred_chunks, axis=0) if y_pred_chunks else np.array([], dtype=np.float32)
    if args.standardize_y:

        y_true = y_true_n * y_std + y_mean
        y_pred = y_pred_n * y_std + y_mean
    else:
        y_true, y_pred = y_true_n, y_pred_n

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"[RESULT] RMSE: {rmse:.4f}")
    print(f"[RESULT] MAE : {mae:.4f}")
    print(f"[RESULT] R^2 : {r2:.4f}")
    if len(y_true) >= 2:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        print(f"[RESULT] Pearson r: {pearson:.4f}")

    # Save predictions
    pd.DataFrame({"smiles": smiles[test_idx], "y_true": y_true, "y_pred": y_pred}).to_csv(args.preds_out, index=False)
    print(f"[INFO] Predictions saved to {args.preds_out}")


if __name__ == "__main__":
    main()
