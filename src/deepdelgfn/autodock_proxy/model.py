"""src.autodock_proxy

Unified loader/predictor for the AutoDock proxy.

Backwards compatible with the existing RandomForest joblib artifact and supports
the new PyTorch NN proxy checkpoint.

Supported artifacts (extension-based detection):
  - *.joblib : joblib dict with at least {"model": sklearn_model} and optional
               {"radius": int, "n_bits": int}
  - *.pt / *.pth : torch checkpoint dict produced by train_autodock_proxy_nn.py
                  with keys:
                    - model_state_dict
                    - model_config (arch,n_bits,hidden_dim,n_layers,dropout)
                    - fp_config (radius,n_bits)
                    - y_normalization (enabled, mean, std)

The public API is a single function:
  load_autodock_proxy(path, device=None)

which returns an object with:
  - predict_ecfp(X_np) -> np.ndarray
  - predict_smiles(smiles_list) -> np.ndarray

This module intentionally avoids non-standard dependencies beyond what the
repo already uses (rdkit, numpy, torch/joblib depending on artifact type).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np


def _smiles_to_ecfp_bits(smiles: str, *, radius: int, n_bits: int) -> Optional[np.ndarray]:
    """Convert a single SMILES string to a dense ECFP (Morgan fingerprint)
    bit vector.  Returns ``None`` if the SMILES cannot be parsed by RDKit.
    """
    # Local import to keep module importable in environments without RDKit
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    arr = np.zeros((int(n_bits),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class AutodockProxy(Protocol):
    """Protocol interface defining the public API for all proxy objects.

    Guarantees ``.radius``, ``.n_bits``, ``predict_ecfp(X) -> np.ndarray``,
    and ``predict_smiles(smiles_list) -> np.ndarray``.
    """

    radius: int
    n_bits: int

    def predict_ecfp(self, X: np.ndarray) -> np.ndarray: ...

    def predict_smiles(self, smiles_list: List[str], *, batch_size: int = 4096) -> np.ndarray: ...


@dataclass
class RFProxy:
    """Random Forest proxy backed by a scikit-learn model loaded from a
    ``.joblib`` artifact.

    Provides ``predict_ecfp`` (numpy densify → sklearn predict) and
    ``predict_smiles`` (RDKit featurize → ``predict_ecfp``).
    """

    model: object
    radius: int
    n_bits: int

    def predict_ecfp(self, X: np.ndarray) -> np.ndarray:
        y = np.asarray(self.model.predict(X), dtype=np.float32)
        return y

    def predict_smiles(self, smiles_list: List[str], *, batch_size: int = 4096) -> np.ndarray:
        feats: List[np.ndarray] = []
        keep: List[int] = []
        for i, smi in enumerate(smiles_list):
            arr = _smiles_to_ecfp_bits(smi, radius=self.radius, n_bits=self.n_bits)
            if arr is None:
                continue
            feats.append(arr)
            keep.append(i)
        if not feats:
            return np.array([], dtype=np.float32)
        X = np.stack(feats, axis=0)
        return self.predict_ecfp(X)


@dataclass
class NNProxy:
    """Neural network proxy backed by a PyTorch model loaded from a
    ``.pt`` / ``.pth`` checkpoint.

    Supports optional y-normalization (standardization that is undone at
    inference time).  Provides ``predict_ecfp`` (batched GPU/CPU inference
    via the stored model) and ``predict_smiles``.
    """

    model: "torch.nn.Module"
    radius: int
    n_bits: int
    y_mean: float
    y_std: float
    y_norm_enabled: bool
    device: "torch.device"

    def predict_ecfp(self, X: np.ndarray) -> np.ndarray:
        import torch

        if X.ndim != 2 or X.shape[1] != int(self.n_bits):
            raise ValueError(f"Expected X shape [N,{self.n_bits}] got {tuple(X.shape)}")

        self.model.eval()
        outs: List[np.ndarray] = []
        bs = 8192
        with torch.no_grad():
            for start in range(0, X.shape[0], bs):
                xb = torch.from_numpy(X[start : start + bs]).float().to(self.device)
                yb = self.model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
                if self.y_norm_enabled:
                    yb = yb * float(self.y_std) + float(self.y_mean)
                outs.append(yb)
        return np.concatenate(outs, axis=0) if outs else np.array([], dtype=np.float32)

    def predict_smiles(self, smiles_list: List[str], *, batch_size: int = 4096) -> np.ndarray:
        # Featurize then predict in batches.
        feats: List[np.ndarray] = []
        for smi in smiles_list:
            arr = _smiles_to_ecfp_bits(smi, radius=self.radius, n_bits=self.n_bits)
            if arr is None:
                continue
            feats.append(arr)
        if not feats:
            return np.array([], dtype=np.float32)
        X = np.stack(feats, axis=0).astype(np.float32, copy=False)
        return self.predict_ecfp(X)


def _build_ecfp_mlp(*, n_bits: int, hidden_dim: int, n_layers: int, dropout: float):
    """Construct a multi-layer perceptron (MLP) with ReLU activations and
    optional dropout that maps ECFP bits to a scalar docking score.
    """
    import torch.nn as nn

    layers: List[nn.Module] = []
    in_dim = int(n_bits)
    for _ in range(int(n_layers)):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def _strip_state_dict_prefix(state_dict: dict, prefix: str) -> dict:
    """Return a copy of *state_dict* with *prefix* removed from all keys
    that have it.

    Used to strip ``ECFPMLP`` wrapping (e.g. ``"net."``) from checkpoint
    keys when loading into a bare ``nn.Sequential``.
    """
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def load_autodock_proxy(path: str, *, device: Optional[str] = None) -> AutodockProxy:
    """Load either RF (.joblib) or NN (.pt/.pth) autodock proxy based on file extension."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suf = p.suffix.lower()

    if suf == ".joblib":
        from joblib import load as joblib_load

        art = joblib_load(str(p))
        if not isinstance(art, dict) or "model" not in art:
            raise ValueError("RF artifact must be a dict with key 'model'.")
        radius = int(art.get("radius", 2))
        n_bits = int(art.get("n_bits", 2048))
        return RFProxy(model=art["model"], radius=radius, n_bits=n_bits)

    if suf in (".pt", ".pth"):
        import torch

        ckpt = torch.load(str(p), map_location="cpu")
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError("NN artifact must be a torch checkpoint dict with key 'model_state_dict'.")

        mcfg = ckpt.get("model_config", {})
        fcfg = ckpt.get("fp_config", {})
        yn = ckpt.get("y_normalization", {})

        n_bits = int(fcfg.get("n_bits", mcfg.get("n_bits", 2048)))
        radius = int(fcfg.get("radius", 2))
        hidden_dim = int(mcfg.get("hidden_dim", 512))
        n_layers = int(mcfg.get("n_layers", 3))
        dropout = float(mcfg.get("dropout", 0.1))

        # Checkpoint compatibility:
        # - Newer checkpoints from train_autodock_proxy_nn.py save ECFPMLP with keys like "net.0.weight".
        # - Older or alternate formats may save a raw nn.Sequential with keys like "0.weight".
        sd = ckpt["model_state_dict"]
        if not isinstance(sd, dict):
            raise ValueError("NN checkpoint model_state_dict must be a dict")

        model = _build_ecfp_mlp(n_bits=n_bits, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

        # If keys look like ECFPMLP.net.* then strip "net." to match our Sequential.
        # This keeps strict=True so true shape mismatches still error loudly.
        if any(isinstance(k, str) and k.startswith("net.") for k in sd.keys()):
            sd = _strip_state_dict_prefix(sd, "net.")

        model.load_state_dict(sd, strict=True)

        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)

        y_norm_enabled = bool(yn.get("enabled", False))
        y_mean = float(yn.get("mean", 0.0))
        y_std = float(yn.get("std", 1.0))

        return NNProxy(
            model=model,
            radius=radius,
            n_bits=n_bits,
            y_mean=y_mean,
            y_std=y_std,
            y_norm_enabled=y_norm_enabled,
            device=dev,
        )

    raise ValueError(f"Unsupported autodock proxy extension: {suf} (expected .joblib or .pt/.pth)")
