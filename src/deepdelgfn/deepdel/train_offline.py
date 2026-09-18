#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure offline training for DeepDel using a pre-generated ID-only dataset CSV.

- Expects dataset CSV with columns: B1_id, B2_id, B3_id, y
- Uses bbs.csv (with SMILES + ID [+ optional pool]) to compute BB fingerprints for inputs
- Splits CSV into train/val (random or stratified)
- Trains for N epochs; evaluates on the CSV-held-out split
- Optional checkpoint resume (model and optimizer)

No online sampling. No metas. No logging of new rows.
"""

import argparse, os, csv, random, time, math
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from deepdelgfn.deepdel.fourier import fourier_dim, fourier_encode_torch
from deepdelgfn.models.deepsets import Phi, TripleDeepSet
from deepdelgfn.models.checkpoint import DeepSetConfig

import matplotlib.pyplot as plt

# Optional Kendall's tau
try:
    from scipy.stats import kendalltau
except Exception:
    kendalltau = None

# Optional memory monitoring
try:
    import psutil
except Exception:
    psutil = None

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ----------------------- Utils / Featurization -----------------------

def set_seed(seed: Optional[int]):
    """Set Python, numpy, and PyTorch random seeds for reproducibility."""
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def plot_dataset_sample(df: pd.DataFrame, output_path: str, sample_size: int = 1000, seed: Optional[int] = None, title_suffix: str = ""):
    """Save a scatter plot of a random dataset sample as ``(-threshold, y)``."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if "threshold" not in df.columns:
        raise ValueError("Dataset must contain a 'threshold' column to plot (-t, y) pairs.")

    n = min(int(sample_size), len(df))
    # Use a local generator so plotting does not consume the training RNG stream.
    rng = np.random.default_rng(seed)
    sample = df.iloc[rng.choice(len(df), size=n, replace=False)]
    sample.loc[:, ["threshold", "y"]].assign(
        neg_threshold=lambda x: -x["threshold"]
    ).to_csv(os.path.splitext(output_path)[0] + ".csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(-sample["threshold"].to_numpy(), sample["y"].to_numpy(),
                s=10, alpha=0.5, edgecolors="none")
    plt.xlabel("-t")
    plt.ylabel("y")
    plt.title(f"Random sample of DeepDEL dataset (n={n:,})\n{title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved dataset sample plot to {output_path}", flush=True)

def smiles_to_morgan_bits(smiles: str, n_bits: int = 2048, radius: int = 2, dtype=np.float32,
                          append_molecular_weight: bool = False):
    """Convert a SMILES string to a dense Morgan fingerprint bit vector.
    Returns ``None`` for invalid SMILES.

    When ``append_molecular_weight=True``, the molecular weight (computed by
    RDKit's ``Descriptors.MolWt``) is appended as an additional scalar feature,
    increasing the output dimension from ``n_bits`` to ``n_bits + 1``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=dtype)
    DataStructs.ConvertToNumpyArray(fp, arr)
    if append_molecular_weight:
        from rdkit.Chem import Descriptors
        mw = float(Descriptors.MolWt(mol))
        arr = np.append(arr, np.array(mw, dtype=dtype))
    return arr

def _parse_id_string(s: str) -> np.ndarray:
    """Parse a pipe-delimited ID string (e.g. ``"101|205|340"``) into an
    array of int64 indices.
    """
    s = str(s)
    if not s or s.lower() == "nan":
        return np.array([], dtype=np.int64)
    return np.array([int(x) for x in s.split("|") if x != ""], dtype=np.int64)

def _pretokenize_id_chunk(args):
    """Parse B1/B2/B3 ID strings for a chunk of the dataset and map BB IDs
    to their row indices in ``X_all``.

    This is a top-level function so it is picklable by ``ProcessPoolExecutor``.
    """
    b1, b2, b3, id2idx, store_as_indices = args

    def parse_col(values):
        out = []
        for s in values:
            arr = _parse_id_string(s)
            if store_as_indices:
                try:
                    arr = np.asarray([id2idx[int(x)] for x in arr], dtype=np.int64)
                except KeyError as e:
                    raise KeyError(f"ID {e} from dataset not found in bbs.csv 'ID'") from e
            out.append(arr)
        return out

    return parse_col(b1), parse_col(b2), parse_col(b3)

def _chunk_bounds(n: int, n_chunks: int):
    """Yield ``(start, end)`` slice bounds dividing ``n`` items into
    ``n_chunks`` roughly equal chunks.
    """
    n_chunks = max(1, min(int(n_chunks), int(n))) if n > 0 else 1
    step = (n + n_chunks - 1) // n_chunks
    for start in range(0, n, step):
        yield start, min(n, start + step)

def _resolve_cpu_count_from_slurm(default: int = 0) -> int:
    """Return ``$SLURM_CPUS_PER_TASK`` if available, otherwise *default*."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None and str(slurm_cpus).isdigit():
        return int(slurm_cpus)
    return int(default)

def _elapsed_msg(label: str, t0: float) -> float:
    """Print a timing message with elapsed seconds and return the current
    time.
    """
    now = time.time()
    print(f"[Timing] {label}: {now - t0:.2f}s", flush=True)
    return now

def load_bbs(csv_path: str) -> pd.DataFrame:
    """Load building blocks CSV and validate that it contains ``SMILES``
    and ``ID`` columns.  Adds a ``Name`` column if missing.
    """
    df = pd.read_csv(csv_path)
    if "SMILES" not in df.columns:
        raise ValueError("bbs.csv must include 'SMILES'.")
    if "ID" not in df.columns:
        raise ValueError("bbs.csv must include 'ID'.")
    if "Name" not in df.columns:
        df["Name"] = [f"BB_{i}" for i in range(len(df))]
    return df

def featurize_bbs(df: pd.DataFrame, n_bits: int, radius: int,
                  append_molecular_weight: bool = False):
    """Featurize all rows of a BB DataFrame into a feature matrix ``X`` and
    a ``smiles`` array.
    """
    X, smiles = [], []
    for _, row in df.iterrows():
        smi = str(row["SMILES"])
        arr = smiles_to_morgan_bits(smi, n_bits=n_bits, radius=radius, dtype=np.float32,
                                    append_molecular_weight=append_molecular_weight)
        if arr is not None:
            X.append(arr); smiles.append(smi)
    if not X:
        raise ValueError("No valid BBs after featurization.")
    return np.stack(X, 0), np.array(smiles, dtype=object)

def load_or_create_bb_cache(csv_path: str, cache_path: str, n_bits: int, radius: int,
                            recompute: bool, append_molecular_weight: bool = False):
    """Load BB fingerprints from a cached ``.npz`` file, or compute and
    save them if the cache is missing or recomputation is requested.
    """
    if os.path.exists(cache_path) and not recompute:
        z = np.load(cache_path, allow_pickle=True)
        return z["X"], z["smiles"]
    df = load_bbs(csv_path)
    X, smiles = featurize_bbs(df, n_bits=n_bits, radius=radius,
                              append_molecular_weight=append_molecular_weight)
    np.savez_compressed(cache_path, X=X, smiles=smiles)
    return X, smiles

# ----------------------- Dataset (ID-based) --------------------------

class PrecomputedTripleDataset(Dataset):
    """PyTorch Dataset consuming a DataFrame with ``B1_id``, ``B2_id``,
    ``B3_id``, ``y`` columns (and optionally ``threshold``) and lazily
    constructing ``(X1, X2, X3), threshold, y`` samples by indexing into a
    shared ``X_all`` fingerprint matrix.
    """
    def __init__(
        self,
        X_all: np.ndarray,
        df: pd.DataFrame,
        id2idx: Dict[int, int],
        *,
        pretokenize: bool = True,
        store_as_indices: bool = True,
        pretokenize_workers: int = 0,
        default_threshold: float = 0.0,
    ):
        if df is None or df.empty:
            raise ValueError("Empty dataset.")
        req = {"B1_id","B2_id","B3_id","y"}
        if not req.issubset(df.columns):
            raise ValueError(f"Dataset missing columns: {req - set(df.columns)}")
        self.X_all = X_all
        self.id2idx = {int(k): int(v) for k, v in id2idx.items()}
        self.store_as_indices = bool(store_as_indices)

        # Store y as contiguous float32 for speed.
        self.y = df["y"].astype(np.float32).to_numpy(copy=True)

        # Threshold conditioning column (optional for legacy datasets).
        if "threshold" in df.columns:
            self.threshold = df["threshold"].astype(np.float32).to_numpy(copy=True)
        else:
            self.threshold = np.full(len(self.y), float(default_threshold), dtype=np.float32)

        # Parsing ID lists (B1_id etc.) from strings inside __getitem__ is a major
        # CPU bottleneck on clusters (esp. with small set sizes like 6). We parse
        # once here and keep compact int arrays.
        if pretokenize:
            b1 = df["B1_id"].astype(str).to_list()
            b2 = df["B2_id"].astype(str).to_list()
            b3 = df["B3_id"].astype(str).to_list()
            pretokenize_workers = max(0, int(pretokenize_workers or 0))
            if pretokenize_workers > 1 and len(self.y) > 1:
                chunks = [
                    (b1[s:e], b2[s:e], b3[s:e], self.id2idx, self.store_as_indices)
                    for s, e in _chunk_bounds(len(self.y), pretokenize_workers)
                ]
                with ProcessPoolExecutor(max_workers=pretokenize_workers) as ex:
                    parts = list(ex.map(_pretokenize_id_chunk, chunks))
                self.B1 = [arr for part in parts for arr in part[0]]
                self.B2 = [arr for part in parts for arr in part[1]]
                self.B3 = [arr for part in parts for arr in part[2]]
            else:
                self.B1, self.B2, self.B3 = _pretokenize_id_chunk(
                    (b1, b2, b3, self.id2idx, self.store_as_indices)
                )
        else:
            # Keep original strings if user wants minimal upfront work/memory.
            self.rows = df[["B1_id", "B2_id", "B3_id"]].astype(str).to_dict("records")
            self.B1 = self.B2 = self.B3 = None

        # Compatibility for __len__
        if not hasattr(self, "rows"):
            self.rows = [None] * len(self.y)

    def __len__(self): return len(self.rows)

    @staticmethod
    def _parse_ids(s: str) -> np.ndarray:
        return _parse_id_string(s)

    def __getitem__(self, idx):
        if self.B1 is None:
            # Slow-path (no pretokenization)
            r = self.rows[idx]
            I_ids = self._parse_ids(r["B1_id"])
            J_ids = self._parse_ids(r["B2_id"])
            K_ids = self._parse_ids(r["B3_id"])
            if len(I_ids) == 0 or len(J_ids) == 0 or len(K_ids) == 0:
                raise ValueError("Encountered empty subset.")
            try:
                I = np.array([self.id2idx[int(x)] for x in I_ids], dtype=np.int64)
                J = np.array([self.id2idx[int(x)] for x in J_ids], dtype=np.int64)
                K = np.array([self.id2idx[int(x)] for x in K_ids], dtype=np.int64)
            except KeyError as e:
                raise KeyError(f"ID {e} from dataset not found in bbs.csv 'ID'") from e
        else:
            I = self.B1[idx]
            J = self.B2[idx]
            K = self.B3[idx]

        X1 = torch.from_numpy(self.X_all[I]).float()
        X2 = torch.from_numpy(self.X_all[J]).float()
        X3 = torch.from_numpy(self.X_all[K]).float()
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        threshold = torch.tensor(self.threshold[idx], dtype=torch.float32)
        return (X1, X2, X3), threshold, y

# ----------------------- Collate (pad sets) --------------------------

def collate_precomputed(batch):
    """Collate a list of ``((X1, X2, X3), threshold, y)`` tuples into padded
    tensors with masks.  Handles variable-sized BB subsets within a batch.
    """
    X1s, X2s, X3s, thresholds, ys = [], [], [], [], []
    for (x1,x2,x3), threshold, y in batch:
        X1s.append(x1); X2s.append(x2); X3s.append(x3)
        thresholds.append(threshold); ys.append(y)

    def pad_stack(tensors: List[torch.Tensor]):
        sizes = [t.shape[0] for t in tensors]
        smax = max(sizes)
        feats, masks = [], []
        for t in tensors:
            s = t.shape[0]; pad = smax - s
            if pad > 0:
                tp = torch.nn.functional.pad(t, (0,0,0,pad))
                mask = torch.cat([torch.ones(s), torch.zeros(pad)])
            else:
                tp = t; mask = torch.ones(s)
            feats.append(tp); masks.append(mask)
        return torch.stack(feats,0).float(), torch.stack(masks,0).float()

    X1, M1 = pad_stack(X1s)
    X2, M2 = pad_stack(X2s)
    X3, M3 = pad_stack(X3s)
    threshold = torch.stack(thresholds, 0).float()
    y = torch.stack(ys,0).float()
    return ((X1,M1),(X2,M2),(X3,M3)), threshold, y

# ----------------------- Model --------------------------
# Phi and TripleDeepSet are imported from deepdelgfn.models.deepsets.


# ----------------------- Train / Eval --------------------------

def _loss_label(loss_name: str) -> str:
    return "BCEWithLogitsLoss" if loss_name == "bce_with_logits" else "MSELoss"


def _compute_loss_and_mse(model, triple, threshold, y, loss_name, loss_fn):
    """Return ``(train_loss, regression_mse, yhat)`` for one batch.

    - ``mse``: ``train_loss`` and ``regression_mse`` are identical
      (``F.mse_loss(yhat, y)``); behavior is unchanged from the historical
      objective.
    - ``bce_with_logits``: ``train_loss`` is the BCE objective on the raw
      logits against ``y / log(1 + k^3)``; ``regression_mse`` is the mean
      squared error of the model's *actual bounded output*
      ``log(1 + k^3) * sigmoid(logits)`` vs ``y`` -- i.e. exactly the metric
      that ``--loss mse`` runs report, so the two variants are directly
      comparable for the same ``--lib-size``.

    ``yhat`` is the model output in log-reward space.
    """
    if loss_name == "bce_with_logits":
        logits = model.forward_logits(triple, threshold)
        target = (y / float(model.log_reward_bound)).clamp(0.0, 1.0)
        loss = loss_fn(logits, target)
        yhat = model.log_reward_bound * torch.sigmoid(logits)
        return loss, F.mse_loss(yhat, y), yhat
    yhat = model(triple, threshold)
    loss = loss_fn(yhat, y)
    return loss, loss, yhat


def _compute_loss(model, triple, threshold, y, loss_name, loss_fn):
    """Compute only the configured objective (backward-compatible wrapper)."""
    return _compute_loss_and_mse(model, triple, threshold, y, loss_name, loss_fn)[0]


def train_one_epoch(model, loader, opt, device, loss_name="mse"):
    """Train the model for one epoch.  Supports CUDA AMP and returns
    ``(train_losses, train_mses)``: per-batch configured objectives and the
    matching per-batch regression MSEs of the model output vs ``y`` in
    log-reward space (identical for ``loss_name == "mse"``).
    """
    model.train()
    loss_fn = nn.BCEWithLogitsLoss() if loss_name == "bce_with_logits" else nn.MSELoss()
    losses, mses = [], []
    scaler = getattr(train_one_epoch, "_scaler", None)
    use_amp = getattr(train_one_epoch, "_use_amp", False)
    if use_amp and scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        train_one_epoch._scaler = scaler

    for triple, threshold, y in loader:
        triple = tuple((X.to(device, non_blocking=True), M.to(device, non_blocking=True)) for (X, M) in triple)
        threshold = threshold.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                loss, mse, _ = _compute_loss_and_mse(model, triple, threshold, y, loss_name, loss_fn)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss, mse, _ = _compute_loss_and_mse(model, triple, threshold, y, loss_name, loss_fn)
            loss.backward()
            opt.step()

        losses.append(float(loss.item()))
        mses.append(float(mse.item()))
    return losses, mses

@torch.no_grad()
def evaluate(model, loader, device, compute_tau=False, loss_name="mse"):
    """Evaluate the model on a DataLoader.

    Returns a dict with:
      - ``loss``: mean configured validation objective (BCE or MSE);
      - ``mse``: mean regression MSE of the model output vs ``y`` in
        log-reward space (for ``loss_name == "mse"`` this equals ``loss``);
      - ``tau``: Kendall's τ between model outputs and targets when
        ``compute_tau=True``, else NaN.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss() if loss_name == "bce_with_logits" else nn.MSELoss()
    losses, mses, y_true, y_pred = [], [], [], []
    for triple, threshold, y in loader:
        triple = tuple((X.to(device), M.to(device)) for (X,M) in triple)
        threshold = threshold.to(device)
        y = y.to(device)
        loss, mse, yhat = _compute_loss_and_mse(model, triple, threshold, y, loss_name, loss_fn)
        losses.append(float(loss.item()))
        mses.append(float(mse.item()))
        if compute_tau:
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(yhat.detach().cpu().numpy())
    value = float(np.mean(losses)) if losses else np.nan
    mse_val = float(np.mean(mses)) if mses else np.nan
    out = {"loss": value, "mse": mse_val, "tau": np.nan}
    if compute_tau and y_true:
        if kendalltau is None:
            print("Warning: scipy not installed; τ unavailable.")
        else:
            yt = np.concatenate(y_true); yp = np.concatenate(y_pred)
            t,_ = kendalltau(yt, yp); out["tau"] = float(t) if t is not None else np.nan
    return out

def save_checkpoint(path, model, opt, epoch, args, best_mse=None,
                    d_in: Optional[int] = None):
    """Save a training checkpoint containing model state, optimizer state,
    epoch, best MSE, CLI args, and optional ``d_in`` (actual input
    dimension).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "epoch": int(epoch),
        "best_mse": (None if best_mse is None else float(best_mse)),
        "args": vars(args),
    }
    if d_in is not None:
        ckpt["d_in"] = int(d_in)
    ckpt["checkpoint_schema_version"] = 1
    ckpt["model_config"] = DeepSetConfig(
        d_in=int(d_in if d_in is not None else model.rho[0].in_features),
        d_hidden=int(model.phi.net[0].out_features),
        d_rho=int(model.rho[0].out_features),
        dropout=float(model.phi.net[2].p),
        shared_phi=model.phi is model.phi2 is model.phi3,
        pooling=model.pooling, output_head=model.output_head,
        reward_bound_k=model.reward_bound_k,
        fourier_n_freqs=model.fourier_n_freqs,
        fourier_freq_scale=model.fourier_freq_scale,
        fourier_linear=model.fourier_linear,
        fourier_append_raw=model.fourier_append_raw,
        fourier_condition=model.fourier_condition,
        fourier_threshold_center=model.fourier_threshold_center,
        fourier_threshold_scale=model.fourier_threshold_scale,
    ).to_dict()
    torch.save(ckpt, path)

def load_checkpoint(path, model, opt=None, map_location="cpu", nonstrict=False):
    """Load a training checkpoint.  Returns a dict with ``epoch``,
    ``best_mse``, ``missing_keys``, ``unexpected_keys``.
    """
    ckpt = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=not nonstrict)
    if opt is not None and "opt_state" in ckpt:
        try: opt.load_state_dict(ckpt["opt_state"])
        except Exception: pass
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_mse": ckpt.get("best_mse", None),
        "missing_keys": missing if isinstance(missing, list) else [],
        "unexpected_keys": unexpected if isinstance(unexpected, list) else [],
    }

# ----------------------- CSV split helpers --------------------------

def build_csv_split_random(df: pd.DataFrame, val_frac: float, seed: Optional[int]):
    """Split the dataset DataFrame into train/val with random shuffling."""
    rs = np.random.RandomState(seed)
    idx = np.arange(len(df)); rs.shuffle(idx)
    k = max(1, int(round(len(df) * val_frac)))
    val_idx = set(idx[:k].tolist())
    train_idx = [i for i in range(len(df)) if i not in val_idx]
    cols = ["B1_id","B2_id","B3_id","y"] + (["threshold"] if "threshold" in df.columns else [])
    df_val = df.iloc[list(val_idx)][cols].reset_index(drop=True)
    df_train = df.iloc[train_idx][cols].reset_index(drop=True)
    return df_train, df_val

def build_csv_split_stratified(df: pd.DataFrame, val_frac: float, n_bins: int, seed: Optional[int]):
    """Split the dataset DataFrame into train/val using stratified sampling
    based on y-binned quantiles.
    """
    val_frac = float(val_frac)
    n_bins = max(2, int(n_bins))
    try:
        df = df.copy()
        df["_bin"] = pd.qcut(df["y"], q=min(n_bins, max(2, len(df))), duplicates="drop")
    except Exception:
        df = df.copy(); df["_bin"] = 0

    # groupby.sample is much faster/cleaner than manual Python loops over bins.
    # For tiny bins, keep at least one validation row when possible.
    def sample_group(g):
        n = min(len(g), max(1, int(round(len(g) * val_frac))))
        return g.sample(n=n, random_state=seed)

    df_val = df.groupby("_bin", observed=False, dropna=False, group_keys=False).apply(sample_group)
    val_idx = df_val.index
    df_train = df.drop(index=val_idx)
    cols = ["B1_id","B2_id","B3_id","y"] + (["threshold"] if "threshold" in df.columns else [])
    df_val = df_val[cols].reset_index(drop=True)
    df_train = df_train[cols].reset_index(drop=True)
    return df_train, df_val

# ----------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser("Offline training for DeepDel (ID-only dataset)")
    ap.add_argument("--bbs", required=True, help="bbs.csv with SMILES and ID")
    ap.add_argument("--dataset", required=True, help="CSV with B1_id,B2_id,B3_id,y")
    ap.add_argument("--plot-sample-size", type=int, default=10000,
                    help="Number of random (-threshold, y) dataset pairs to plot at startup.")
    ap.add_argument("--bb-fp-bits", type=int, default=2048)
    ap.add_argument("--bb-fp-radius", type=int, default=2)
    ap.add_argument("--recompute-bb-cache", action="store_true")
    ap.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help=(
            "Optional external held-out CSV (B1_id,B2_id,B3_id[,threshold],y) "
            "evaluated on the final (--save-last) checkpoint after training. "
            "Independent of the internal --val-frac split used for early stopping."
        ),
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Do not train. Load --resume and evaluate on --eval-dataset, then write "
            "a --stats-json summary (with external_val_mse) and exit. Used to score "
            "a pretrained checkpoint at dd-init."
        ),
    )

    # split
    ap.add_argument("--validation-type", choices=["stratified","random"], default="stratified")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-bins", type=int, default=10)
    ap.add_argument(
        "--compute-tau",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute Kendall's tau between model outputs and targets on the "
            "validation set every epoch (default: enabled; use --no-compute-tau "
            "to disable)."
        ),
    )

    # training
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Early stopping patience on the validation objective (MSE for --loss mse, "
            "BCE for --loss bce_with_logits). Set <=0 to disable (default: 0)."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument(
        "--loss",
        choices=["mse", "bce_with_logits"],
        default="mse",
        help=(
            "Training/evaluation loss. bce_with_logits uses the raw sigmoid logit "
            "and y/log(1+k^3) as the target; it requires --output-head sigmoid_scaled."
        ),
    )
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--rho-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--shared-phi", action="store_true")
    ap.add_argument(
        "--pooling",
        choices=["mean", "sum"],
        default="mean",
        help=(
            "Permutation-invariant pooling over per-building-block phi embeddings. "
            "'mean' preserves legacy checkpoints/behavior; 'sum' matches the "
            "canonical DeepSets sum aggregation."
        ),
    )
    ap.add_argument(
        "--log-target",
        action="store_true",
        help=(
            "Marker flag: indicates that the dataset's `y` column is log R(x) "
            "(see generate_dataset.py --log-target). The model itself is the "
            "same plain linear-head regressor; this flag is only persisted into "
            "ckpt['args'] so downstream consumers (e.g. GFN training) know to "
            "interpret model(x) as log R rather than R, and skip the unsafe "
            "log() they would otherwise apply."
        ),
    )
    ap.add_argument(
        "--output-head",
        choices=["linear", "sigmoid_scaled", "softplus_scaled"],
        default="linear",
        help=(
            "Final activation applied to the raw rho logit. 'linear' preserves "
            "legacy behavior. 'sigmoid_scaled' outputs log(1+k^3)*sigmoid(z), "
            "bounded in [0, log(1+k^3)] for a k^3-molecule threshold reward; "
            "'softplus_scaled' outputs log(1+k^3)*softplus(z) (non-negative, "
            "unbounded above). Non-linear heads require --lib-size."
        ),
    )
    ap.add_argument(
        "--lib-size",
        type=int,
        default=None,
        help=(
            "k = number of building blocks per position. Used only with a "
            "non-linear --output-head to compute the constant log(1+k^3) "
            "bounding a 1+sum-of-sigmoids threshold reward."
        ),
    )


    # resume / saves
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-optimizer", action="store_true")
    ap.add_argument("--resume-nonstrict", action="store_true")
    ap.add_argument("--save-best", type=str, default=None)
    ap.add_argument("--save-last", type=str, default="models/deepdel.pt")

    # performance / Narval
    ap.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers. Use -1 to auto-use $SLURM_CPUS_PER_TASK (else 8).",
    )
    ap.add_argument("--prefetch-factor", type=int, default=2, help="prefetch_factor for DataLoader workers")
    ap.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive across epochs")
    ap.add_argument("--pin-memory", action="store_true", help="Enable pin_memory (GPU recommended)")
    ap.add_argument("--max-workers", type=int, default=64, help="Maximum number of DataLoader workers allowed")
    ap.add_argument(
        "--pretokenize-workers",
        type=int,
        default=-1,
        help=(
            "Workers used to parse/map B1_id/B2_id/B3_id before training. "
            "Use -1 to match resolved DataLoader workers / SLURM_CPUS_PER_TASK; "
            "0 or 1 parses serially."
        ),
    )
    ap.add_argument("--amp", action="store_true", help="Enable CUDA AMP (recommended on A100)")
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn (recommended on A100)")
    ap.add_argument("--compile", action="store_true", help="torch.compile(model) (PyTorch 2.x)")
    ap.add_argument("--log-interval", type=int, default=50, help="Batches between progress prints")
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override torch.set_num_threads on CPU. Default: SLURM_CPUS_PER_TASK/4 capped at 32.",
    )

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stats-json", type=str, default=None,
                    help="If provided, write a JSON summary of training metrics to this path.")
    ap.add_argument(
        "--append-molecular-weight",
        action="store_true",
        help=(
            "Append the molecular weight of each building block as an extra "
            "scalar feature to the ECFP fingerprint, increasing the input "
            "dimension from bb_fp_bits to bb_fp_bits + 1."
        ),
    )

    # Fourier threshold conditioning
    ap.add_argument("--fourier-n-freqs", type=int, default=0,
                    help="Number of Fourier frequencies for threshold conditioning (0=disabled).")
    ap.add_argument("--fourier-freq-scale", type=float, default=1.5,
                    help="Geometric frequency scale for Fourier encoding.")
    ap.add_argument("--fourier-linear", action="store_true",
                    help="Use linear frequencies (i+1) instead of geometric (freq_scale^i).")
    ap.add_argument("--fourier-append-raw", action="store_true",
                    help="Append the raw threshold value to the Fourier encoding.")
    ap.add_argument("--fourier-condition", choices=["rho", "phi"], default="rho",
                    help="Where to inject the Fourier encoding: 'rho' (before rho head) or 'phi' (per-BB before phi).")
    ap.add_argument("--fourier-threshold-center", type=float, default=None,
                    help="Optional center subtracted from thresholds before Fourier encoding.")
    ap.add_argument("--fourier-threshold-scale", type=float, default=None,
                    help="Optional positive scale dividing thresholds before Fourier encoding.")
    ap.add_argument("--default-threshold", type=float, default=0.0,
                    help="Threshold used for legacy datasets without a 'threshold' column.")

    args = ap.parse_args()
    set_seed(args.seed)

    startup_t0 = time.time()
    t_phase = startup_t0

    allocated_cpus = _resolve_cpu_count_from_slurm(default=os.cpu_count() or 1)
    requested_num_workers = int(args.num_workers)
    if args.num_workers < 0:
        args.num_workers = allocated_cpus
    args.num_workers = max(0, min(int(args.num_workers), int(args.max_workers)))

    if args.pretokenize_workers < 0:
        args.pretokenize_workers = args.num_workers if args.num_workers > 0 else allocated_cpus
    args.pretokenize_workers = max(0, min(int(args.pretokenize_workers), int(args.max_workers), allocated_cpus))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep the main process from spawning large BLAS/OpenMP pools while DataLoader
    # workers and pretokenization workers are also using the allocated CPUs.
    if args.cpu_threads is not None:
        torch_threads = max(1, int(args.cpu_threads))
    elif device.type == "cuda" and (args.num_workers > 0 or args.pretokenize_workers > 1):
        torch_threads = 1
    elif device.type == "cpu":
        torch_threads = max(1, min(32, allocated_cpus // 4 if allocated_cpus > 1 else 1))
    else:
        torch_threads = max(1, min(4, allocated_cpus))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, torch_threads)))
    except RuntimeError:
        # Can only be set before parallel work starts; ignore if a library already used it.
        pass

    print(
        "[CPU] "
        f"SLURM_CPUS_PER_TASK/os={allocated_cpus} | "
        f"requested_num_workers={requested_num_workers} | "
        f"resolved_num_workers={args.num_workers} | "
        f"pretokenize_workers={args.pretokenize_workers} | "
        f"torch_threads={torch.get_num_threads()}",
        flush=True,
    )

    if psutil:
        print(f"[Mem] Initial RAM: {psutil.virtual_memory().used / 1e9:.2f} GB", flush=True)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.join("outputs","deepdel"), exist_ok=True)
    t_phase = _elapsed_msg("created output directories", t_phase)

    # Load BBs and build ID map
    df = load_bbs(args.bbs)
    id_map = df["ID"].to_numpy()
    id2idx = {int(bb_id): int(i) for i, bb_id in enumerate(id_map.tolist())}
    t_phase = _elapsed_msg(f"loaded BBs ({len(df):,} rows)", t_phase)

    # BB cache (fingerprints)
    mw_suffix = "_mw" if args.append_molecular_weight else ""
    cache_path = os.path.join("outputs","deepdel", f"bbs_fp{args.bb_fp_bits}_r{args.bb_fp_radius}{mw_suffix}.npz")
    X_all, _ = load_or_create_bb_cache(args.bbs, cache_path, args.bb_fp_bits, args.bb_fp_radius,
                                       args.recompute_bb_cache,
                                       append_molecular_weight=args.append_molecular_weight)
    t_phase = _elapsed_msg(f"loaded/created BB fingerprint cache X_all={tuple(X_all.shape)}", t_phase)

    # Dataset CSV
    df_all = pd.read_csv(args.dataset).dropna(subset=["y"])
    if df_all.empty:
        raise ValueError("Dataset CSV has no rows with non-NaN y.")
    t_phase = _elapsed_msg(f"read dataset CSV ({len(df_all):,} non-NaN rows)", t_phase)
    plot_dir = os.path.dirname(args.stats_json) if args.stats_json else os.path.join("outputs", "deepdel")
    os.makedirs(plot_dir or ".", exist_ok=True)
    dataset_plot_path = os.path.join(plot_dir, "dataset_sample.png")
    title_suffix = (
        f"hidden={args.hidden_dim} rho={args.rho_dim} | lib={args.lib_size} | "
        f"Fourier n_freqs={args.fourier_n_freqs}"
    )
    plot_dataset_sample(df_all, dataset_plot_path, args.plot_sample_size, args.seed, title_suffix)
    t_phase = _elapsed_msg("plotted random dataset sample", t_phase)

    print(
        f"[Split] type={args.validation_type} val_frac={args.val_frac} val_bins={args.val_bins} rows={len(df_all):,}",
        flush=True,
    )
    if args.validation_type == "stratified":
        df_train, df_val = build_csv_split_stratified(df_all, args.val_frac, args.val_bins, args.seed)
    else:
        df_train, df_val = build_csv_split_random(df_all, args.val_frac, args.seed)
    t_phase = _elapsed_msg(f"built {args.validation_type} split (train={len(df_train):,}, val={len(df_val):,})", t_phase)

    # Datasets / loaders
    train_ds = PrecomputedTripleDataset(
        X_all, df_train, id2idx,
        pretokenize=True, store_as_indices=True,
        pretokenize_workers=args.pretokenize_workers,
        default_threshold=args.default_threshold,
    )
    t_phase = _elapsed_msg("pretokenized train dataset", t_phase)
    val_ds = PrecomputedTripleDataset(
        X_all, df_val, id2idx,
        pretokenize=True, store_as_indices=True,
        pretokenize_workers=args.pretokenize_workers,
        default_threshold=args.default_threshold,
    )
    t_phase = _elapsed_msg("pretokenized validation dataset", t_phase)

    # Device-based configuration before creating DataLoaders so pin_memory is honored.
    if device.type == "cuda":
        # Enable GPU optimizations if not specified
        if not args.amp:
            args.amp = True
            print("[Auto] Enabled AMP for CUDA", flush=True)
        if not args.tf32:
            args.tf32 = True
            print("[Auto] Enabled TF32 for CUDA", flush=True)
        if not args.pin_memory:
            args.pin_memory = True
            print("[Auto] Enabled pin_memory for CUDA", flush=True)
    else:
        print(f"[Auto] Restricted PyTorch to {torch.get_num_threads()} threads for CPU efficiency.", flush=True)

    if args.persistent_workers and args.epochs <= 1:
        print("[DataLoader] persistent_workers requested for one epoch; disabling to avoid extra worker lifetime overhead.", flush=True)
        args.persistent_workers = False

    pin_memory = bool(args.pin_memory)

    # DataLoader tuning: on Narval (A100) the main bottleneck for this script is
    # CPU-side __getitem__ work + host->device transfers.
    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_precomputed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
    )
    # prefetch_factor only valid when num_workers>0
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    print(f"[DataLoader] {loader_kwargs}", flush=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    t_phase = _elapsed_msg("created DataLoaders", t_phase)

    # External held-out dataset (evaluated independently of the internal split).
    eval_loader = None
    if args.eval_dataset:
        df_eval = pd.read_csv(args.eval_dataset).dropna(subset=["y"])
        if df_eval.empty:
            raise ValueError("Eval dataset CSV has no rows with non-NaN y.")
        eval_ds = PrecomputedTripleDataset(
            X_all, df_eval, id2idx,
            pretokenize=True, store_as_indices=True,
            pretokenize_workers=args.pretokenize_workers,
            default_threshold=args.default_threshold,
        )
        eval_loader = DataLoader(eval_ds, shuffle=False, **loader_kwargs)
        t_phase = _elapsed_msg(
            f"pretokenized external eval dataset ({len(df_eval):,} rows)", t_phase
        )

    # Model / opt

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = TripleDeepSet(d_in=X_all.shape[1], d_hidden=args.hidden_dim, d_rho=args.rho_dim,
                          dropout=args.dropout, shared_phi=args.shared_phi,
                          pooling=args.pooling, output_head=args.output_head,
                          reward_bound_k=args.lib_size,
                          fourier_n_freqs=args.fourier_n_freqs,
                          fourier_freq_scale=args.fourier_freq_scale,
                          fourier_linear=args.fourier_linear,
                          fourier_append_raw=args.fourier_append_raw,
                          fourier_condition=args.fourier_condition,
                          fourier_threshold_center=args.fourier_threshold_center,
                          fourier_threshold_scale=args.fourier_threshold_scale).to(device)
    if args.loss == "bce_with_logits" and args.output_head != "sigmoid_scaled":
        raise ValueError("--loss bce_with_logits requires --output-head sigmoid_scaled")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] DeepDEL TripleDeepSet pooling={args.pooling} output_head={args.output_head}", flush=True)
    print(f"[Model] Trainable parameters: {n_trainable:,}", flush=True)
    t_phase = _elapsed_msg("created model", t_phase)


    if args.compile:
        try:
            model = torch.compile(model)
            print("[Perf] torch.compile enabled")
        except Exception as e:
            print(f"[Perf] torch.compile failed: {e}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t_phase = _elapsed_msg("created optimizer", t_phase)

    # AMP setup (train_one_epoch reads these attributes)
    train_one_epoch._use_amp = bool(args.amp and device.type == "cuda")
    train_one_epoch._scaler = torch.amp.GradScaler('cuda', enabled=train_one_epoch._use_amp)

    # Optional resume
    start_epoch = 0
    best_mse = float("inf")
    best_val_loss = float("inf")
    if args.resume:
        meta = load_checkpoint(args.resume, model, opt if args.resume_optimizer else None,
                               map_location=device, nonstrict=args.resume_nonstrict)
        start_epoch = int(meta.get("epoch", 0))
        if meta.get("best_mse") is not None:
            best_mse = float(meta["best_mse"])
            if args.loss == "mse":
                best_val_loss = best_mse  # objective == regression MSE for --loss mse
        if meta.get("missing_keys"):   print(f"[Resume] Missing keys: {len(meta['missing_keys'])}")
        if meta.get("unexpected_keys"):print(f"[Resume] Unexpected keys: {len(meta['unexpected_keys'])}")
        print(f"[Resume] Loaded {args.resume} (epoch={start_epoch}, best_mse={best_mse})", flush=True)
        t_phase = _elapsed_msg("loaded checkpoint", t_phase)

    print(f"[Timing] total startup before training: {time.time() - startup_t0:.2f}s", flush=True)

    # Eval-only mode: score a resumed checkpoint on the external eval dataset.
    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume (checkpoint path).")
        if eval_loader is None:
            raise ValueError("--eval-only requires --eval-dataset.")
        model.eval()
        ext_metrics = evaluate(model, eval_loader, device,
                               compute_tau=bool(args.compute_tau), loss_name=args.loss)
        external_val_mse = float(ext_metrics["mse"])
        print(f"[Eval-only] External validation MSE = {external_val_mse:.6f}", flush=True)
        if args.stats_json:
            import json
            stats = {
                "loss": args.loss,
                "external_val_mse": external_val_mse,
                "external_val_tau": float(ext_metrics["tau"]),
                "last_val_mse": external_val_mse,
                "best_val_mse": external_val_mse,
                "epochs_completed": 0,
                "train_mse": [],
                "val_mse": [external_val_mse],
                "train_loss": [],
                "val_loss": [],
                "val_tau": [],
            }
            os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)
            with open(args.stats_json, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Saved stats JSON to {args.stats_json}")
        return

    # Train
    train_hist, val_hist = [], []            # regression MSE per epoch
    train_loss_hist, val_loss_hist = [], []  # training objective per epoch
    val_tau_hist = []                        # Kendall's τ per epoch
    bad_epochs = 0
    last_epoch = start_epoch
    for ep in range(start_epoch+1, start_epoch + args.epochs + 1):
        last_epoch = ep
        t0 = time.time()
        print(f"\nEpoch {ep - start_epoch}/{args.epochs} (global epoch {ep})")
        train_losses, train_mses = train_one_epoch(model, train_loader, opt, device, args.loss)
        train_mse = float(np.mean(train_mses)) if train_mses else np.nan
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        train_hist.append(train_mse)
        train_loss_hist.append(train_loss)

        val_metrics = evaluate(model, val_loader, device,
                               compute_tau=bool(args.compute_tau), loss_name=args.loss)
        val_loss = val_metrics["loss"]
        val_mse = val_metrics["mse"]
        val_tau = val_metrics["tau"]
        val_hist.append(val_mse)
        val_loss_hist.append(val_loss)
        val_tau_hist.append(val_tau)
        if args.loss == "bce_with_logits":
            print(f"Validation {_loss_label(args.loss)} = {val_loss:.6f}")
            # Regression MSE in log-R space: the same metric --loss mse runs
            # report, so the two variants are directly comparable for the same
            # --lib-size.
            print(f"Validation MSE = {val_mse:.6f}")
        else:
            print(f"Validation MSE = {val_mse:.6f}")
        if args.compute_tau and np.isfinite(val_tau):
            print(f"Validation Kendall τ = {val_tau:.6f}")

        dt = time.time() - t0
        if len(train_loader) > 0:
            sec_per_batch = dt / len(train_loader)
            samples_per_sec = (len(train_loader.dataset) / dt) if dt > 0 else float("nan")
            print(f"[Timing] epoch seconds={dt:.2f} | sec/batch={sec_per_batch:.4f} | samples/sec={samples_per_sec:.1f}")

        # Best-model selection / early stopping use the validation objective the
        # model is actually optimizing (BCE for bce_with_logits, MSE otherwise);
        # best_mse stores the corresponding regression MSE (identical for mse).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mse = val_mse
            bad_epochs = 0
            if args.save_best:
                save_checkpoint(args.save_best, model, opt, ep, args, best_mse,
                                d_in=X_all.shape[1])
                print(f"✓ Saved BEST to {args.save_best} (MSE={best_mse:.6f})")
        else:
            bad_epochs += 1
            if args.patience and int(args.patience) > 0 and bad_epochs >= int(args.patience):
                print(f"[Early stopping] No validation {_loss_label(args.loss)} improvement for {bad_epochs} epoch(s) "
                      f"(patience={args.patience}); stopping.")
                break

    if args.save_last:
        save_checkpoint(args.save_last, model, opt, last_epoch, args, best_mse,
                        d_in=X_all.shape[1])
        print(f"✓ Saved LAST to {args.save_last} (MSE={best_mse:.6f})")

    # External held-out evaluation on the final (saved) checkpoint.
    external_val_mse = float("nan")
    external_val_tau = float("nan")
    if eval_loader is not None:
        ext_metrics = evaluate(model, eval_loader, device,
                               compute_tau=bool(args.compute_tau), loss_name=args.loss)
        external_val_mse = float(ext_metrics["mse"])
        external_val_tau = float(ext_metrics["tau"])
        print(f"[Eval] External validation MSE = {external_val_mse:.6f}", flush=True)

    # Plots
    plt.figure(figsize=(10,5))
    if train_hist: plt.plot(train_hist, label="Train MSE")
    if val_hist:   plt.plot(val_hist, label="Val MSE")
    if args.loss == "bce_with_logits":
        if train_loss_hist:
            plt.plot(train_loss_hist, label=f"Train {_loss_label(args.loss)}", linestyle="--", alpha=0.6)
        if val_loss_hist:
            plt.plot(val_loss_hist, label=f"Val {_loss_label(args.loss)}", linestyle="--", alpha=0.6)
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
    if val_tau_hist and np.any(np.isfinite(np.asarray(val_tau_hist, dtype=float))):
        ax2 = plt.gca().twinx()
        ax2.plot(range(1, len(val_tau_hist) + 1), val_tau_hist,
                 label="Val Kendall τ", color="green", marker="o", alpha=0.7)
        ax2.set_ylabel("Kendall τ")
        ax2.legend(loc="lower right")
    plt.title(f"DeepDEL train/validation MSE\n{title_suffix}")
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "offline_train_val.png")
    pd.DataFrame({
        "epoch": np.arange(1, max(len(train_hist), len(val_hist)) + 1),
        "train_mse": pd.Series(train_hist),
        "val_mse": pd.Series(val_hist),
        "train_loss": pd.Series(train_loss_hist),
        "val_loss": pd.Series(val_loss_hist),
        "val_tau": pd.Series(val_tau_hist),
    }).to_csv(os.path.splitext(plot_path)[0] + ".csv", index=False)
    plt.savefig(plot_path, dpi=180); plt.close()
    print(f"Saved curve to {plot_path}")

    if args.stats_json:
        import json
        stats = {
            "loss": args.loss,
            "last_val_mse": float(val_hist[-1]) if val_hist else float("nan"),
            "best_val_mse": float(best_mse),
            "external_val_mse": external_val_mse,
            "external_val_tau": external_val_tau,
            "epochs_completed": len(train_hist),
            "train_mse": [float(v) for v in train_hist],
            "val_mse": [float(v) for v in val_hist],
            "train_loss": [float(v) for v in train_loss_hist],
            "val_loss": [float(v) for v in val_loss_hist],
            "val_tau": [float(v) for v in val_tau_hist],
        }
        os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats JSON to {args.stats_json}")

    if psutil:
        print(f"[Mem] Final RAM: {psutil.virtual_memory().used / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
