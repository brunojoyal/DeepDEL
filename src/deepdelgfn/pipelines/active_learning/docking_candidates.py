#!/usr/bin/env python3
"""Active-learning docking candidate selection.

This module was extracted from active_learning_stage.py.  It provides
"top" and "leaders" selection strategies that choose which building-block
combinations to dock after each inner loop, together with the DeepDEL φ
context helpers needed for leader diversity.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    BBS_CSV,
    CFG,
    PROJECT_ROOT,
    _cfg_get,
    _csv_data_row_count,
    RUN_ROOT,
)

# DOCK score column candidates used for title lookup only.
_DOCK_SCORE_COL_CANDIDATES = ("docking_score", "trimer_score", "score")


# ---------------------------------------------------------------------------
# Utility: canonical SMILES
# ---------------------------------------------------------------------------


def _canonical_smiles(smiles) -> str:
    """Canonicalize a SMILES string for score reuse; fall back to stripped text."""
    if pd.isna(smiles):
        return ""
    text = str(smiles).strip()
    if not text:
        return ""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        pass
    return text


# ---------------------------------------------------------------------------
# Prior docking score lookup
# ---------------------------------------------------------------------------


def _iter_autodock_proxy_input_csvs() -> List[Path]:
    paths = _cfg_get("paths.autodock_proxy_inputs", ["data/scored_libraries"])
    csvs: List[Path] = []
    for raw in paths:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file() and p.suffix.lower() == ".csv":
            csvs.append(p)
        elif p.is_dir():
            csvs.extend(sorted(p.rglob("*.csv")))
    return sorted(set(csvs))


def _load_previous_docking_scores() -> Dict[str, float]:
    """Build canonical-SMILES -> score from autodock proxy inputs.

    This intentionally keeps 0.0 values, because they are meaningful for the
    user's requested reuse semantics and may represent prior failed/no-hit dock
    attempts that should not be repeated.
    """
    if not bool(_cfg_get("docking.reuse_previous_scores", True)):
        return {}
    smiles_col = str(
        _cfg_get("docking.reuse_smiles_col", _cfg_get("autodock_proxy.smiles_col", "smiles"))
    )
    score_col_cfg = str(
        _cfg_get("docking.reuse_score_col", _cfg_get("autodock_proxy.target_col", "docking_score"))
    )
    lookup: Dict[str, float] = {}
    scanned = 0
    for csv_path in _iter_autodock_proxy_input_csvs():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[reuse] Skipping unreadable CSV {csv_path}: {e}")
            continue
        if smiles_col not in df.columns:
            continue
        score_col = score_col_cfg if score_col_cfg in df.columns else None
        if score_col is None:
            for candidate in _DOCK_SCORE_COL_CANDIDATES:
                if candidate in df.columns:
                    score_col = candidate
                    break
        if score_col is None:
            continue
        scanned += 1
        scores = pd.to_numeric(df[score_col], errors="coerce")
        for smi, score in zip(df[smiles_col], scores):
            if pd.isna(score):
                continue
            key = _canonical_smiles(smi)
            if key and key not in lookup:
                lookup[key] = float(score)
    print(f"[reuse] Loaded {len(lookup)} prior molecule score(s) from {scanned} autodock proxy input CSV(s).")
    return lookup


# ---------------------------------------------------------------------------
# Docking selection configuration
# ---------------------------------------------------------------------------


def _docking_selection_cfg() -> dict:
    cfg = CFG.get("docking_selection", {}) or {}
    legacy_n = int(CFG.get("topn_to_dock_per_inner", 20))
    return {
        "mode": str(cfg.get("mode", "top")).lower(),
        "n_to_dock_per_inner": int(cfg.get("n_to_dock_per_inner", legacy_n)),
        "n_leaders_to_dock": int(cfg.get("n_leaders_to_dock", legacy_n)),
        "leader_similarity": str(cfg.get("leader_similarity", "cosine")).lower(),
        "leader_binary_search_iters": int(cfg.get("leader_binary_search_iters", 24)),
    }


# ---------------------------------------------------------------------------
# Top-N selection
# ---------------------------------------------------------------------------


def select_top_libraries_from_csv(topm_csv: Path, *, topn: int) -> pd.DataFrame:
    if not topm_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(topm_csv)
    score_col = "autodock_proxy_value" if "autodock_proxy_value" in df.columns else "true_value"
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])
    key_cols = [c for c in ["B1_id", "B2_id", "B3_id"] if c in df.columns]
    if len(key_cols) == 3:
        df = df.drop_duplicates(subset=key_cols)
    return df.sort_values(by=score_col, ascending=False).head(int(topn)).reset_index(drop=True)


def gather_top_candidates_by_inner_loop(
    outer_dir: Path, *, topn_per_inner: int, only_inner_loop: Optional[int] = None
) -> pd.DataFrame:
    inner_root = outer_dir / "inners"
    if not inner_root.exists():
        return pd.DataFrame()
    rows = []
    inner_dirs = (
        [inner_root / f"inner_{int(only_inner_loop)}"]
        if only_inner_loop is not None
        else sorted(inner_root.glob("inner_*"))
    )
    for inner_dir in inner_dirs:
        try:
            inner_idx = int(inner_dir.name.split("_")[-1])
        except Exception:
            continue
        topm_out = inner_dir / "topm_actual_scores.csv"
        if not topm_out.exists():
            continue
        df = select_top_libraries_from_csv(topm_out, topn=topn_per_inner)
        if len(df) > 0:
            df = df.copy()
            df["inner_loop"] = inner_idx
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Leader (diversity) selection
# ---------------------------------------------------------------------------


def _parse_id_list(value) -> List[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("|") if part != ""]


def _load_deepdel_phi_context(
    deepdel_model_path: Path,
) -> tuple[Dict[int, int], np.ndarray, np.ndarray, np.ndarray, str]:
    """Reconstruct DeepDEL φ tables on CPU for leader selection.

    Candidate CSVs store external BB IDs. This helper loads the same combined BB
    CSV used by GFN training, maps external IDs back to row indices, reloads the
    DeepDEL checkpoint, and precomputes phi1/phi2/phi3 tables for every BB.
    """
    import torch
    from deepdelgfn.deepdel.train_offline import TripleDeepSet, smiles_to_morgan_bits

    if not deepdel_model_path.exists():
        raise FileNotFoundError(
            f"DeepDEL checkpoint not found for leader selection: {deepdel_model_path}"
        )
    bbs = pd.read_csv(BBS_CSV)
    if "ID" not in bbs.columns or "SMILES" not in bbs.columns:
        raise ValueError(f"Combined BB CSV must include ID and SMILES columns: {BBS_CSV}")
    id_to_idx = {int(bb_id): int(i) for i, bb_id in enumerate(bbs["ID"].to_numpy())}

    ckpt = torch.load(str(deepdel_model_path), map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}
    d_in = int(ckpt_args.get("bb_fp_bits", 2048))
    d_h = int(ckpt_args.get("hidden_dim", 256))
    d_rho = int(ckpt_args.get("rho_dim", 256))
    dropout = float(ckpt_args.get("dropout", 0.1))
    shared_phi = bool(ckpt_args.get("shared_phi", False))
    pooling = str(ckpt_args.get("pooling", "mean"))
    output_head = str(ckpt_args.get("output_head", "linear"))
    lib_size = ckpt_args.get("lib_size", None)
    radius = int(ckpt_args.get("bb_fp_radius", 2))

    model = TripleDeepSet(
        d_in=d_in,
        d_hidden=d_h,
        d_rho=d_rho,
        dropout=dropout,
        shared_phi=shared_phi,
        pooling=pooling,
        output_head=output_head,
        reward_bound_k=lib_size,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    feats = []
    for smi in bbs["SMILES"].astype(str).tolist():
        arr = smiles_to_morgan_bits(smi, n_bits=d_in, radius=radius, dtype=np.float32)
        if arr is None:
            raise ValueError(f"Invalid BB SMILES while building φ table: {smi}")
        feats.append(arr)
    X_all = torch.from_numpy(np.stack(feats, axis=0)).float()
    with torch.no_grad():
        phi1 = model.phi(X_all).cpu().numpy().astype(np.float32, copy=False)
        phi2 = model.phi2(X_all).cpu().numpy().astype(np.float32, copy=False)
        phi3 = model.phi3(X_all).cpu().numpy().astype(np.float32, copy=False)
    print(
        f"[leaders] Built DeepDEL φ tables from {deepdel_model_path}: "
        f"N={len(bbs)} d_h={d_h} pooling={pooling}"
    )
    return id_to_idx, phi1, phi2, phi3, pooling


def _pool_phi(phi_table: np.ndarray, indices: List[int], *, pooling: str) -> np.ndarray:
    if not indices:
        return np.zeros((phi_table.shape[1],), dtype=np.float32)
    pooled = phi_table[np.asarray(indices, dtype=np.int64)].sum(axis=0)
    if str(pooling).lower() == "sum":
        return pooled
    return pooled / max(1, len(indices))


def _candidate_phi_representations(
    df: pd.DataFrame,
    *,
    id_to_idx: Dict[int, int],
    phi1: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    pooling: str = "mean",
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    reps = []
    for _, row in df.iterrows():
        try:
            b1 = [id_to_idx[x] for x in _parse_id_list(row["B1_id"])]
            b2 = [id_to_idx[x] for x in _parse_id_list(row["B2_id"])]
            b3 = [id_to_idx[x] for x in _parse_id_list(row["B3_id"])]
        except Exception as e:
            print(f"[leaders] Skipping candidate with unmappable BB IDs: {e}")
            continue
        rep = np.concatenate(
            [
                _pool_phi(phi1, b1, pooling=pooling),
                _pool_phi(phi2, b2, pooling=pooling),
                _pool_phi(phi3, b3, pooling=pooling),
            ],
            axis=0,
        )
        rows.append(row)
        reps.append(rep)
    if not rows:
        return pd.DataFrame(columns=df.columns), np.empty((0, phi1.shape[1] * 3), dtype=np.float32)
    out_df = pd.DataFrame(rows).reset_index(drop=True)
    out_reps = np.stack(reps, axis=0).astype(np.float32, copy=False)
    return out_df, out_reps


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _greedy_leader_indices(
    normalized_reps: np.ndarray, *, threshold: float, limit: Optional[int] = None
) -> List[int]:
    selected: List[int] = []
    for i in range(normalized_reps.shape[0]):
        if selected:
            max_sim = float(np.max(normalized_reps[np.asarray(selected)] @ normalized_reps[i]))
            if max_sim > float(threshold):
                continue
        selected.append(i)
        if limit is not None and len(selected) >= int(limit):
            break
    return selected


def select_leader_libraries_from_csv(
    topm_csv: Path,
    *,
    n_leaders: int,
    id_to_idx: Dict[int, int],
    phi1: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    pooling: str = "mean",
    similarity: str = "cosine",
    binary_search_iters: int = 24,
) -> pd.DataFrame:
    if similarity != "cosine":
        raise ValueError(
            f"Unsupported leader similarity '{similarity}'. Currently only 'cosine' is implemented."
        )
    if not topm_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(topm_csv)
    score_col = "autodock_proxy_value" if "autodock_proxy_value" in df.columns else "true_value"
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])
    key_cols = [c for c in ["B1_id", "B2_id", "B3_id"] if c in df.columns]
    if len(key_cols) == 3:
        df = df.drop_duplicates(subset=key_cols)
    df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    if df.empty:
        return df

    df, reps = _candidate_phi_representations(
        df, id_to_idx=id_to_idx, phi1=phi1, phi2=phi2, phi3=phi3, pooling=pooling
    )
    if df.empty:
        return df
    target = min(int(n_leaders), len(df))
    reps = _normalize_rows(reps)

    lo, hi = -1.0, 1.0
    best_threshold = hi
    for _ in range(max(1, int(binary_search_iters))):
        mid = (lo + hi) / 2.0
        count = len(_greedy_leader_indices(reps, threshold=mid, limit=target))
        if count >= target:
            best_threshold = mid
            hi = mid
        else:
            lo = mid

    selected = _greedy_leader_indices(reps, threshold=best_threshold, limit=target)
    if len(selected) < target:
        selected_set = set(selected)
        selected.extend(
            [i for i in range(len(df)) if i not in selected_set][: target - len(selected)]
        )

    out = df.iloc[selected[:target]].copy().reset_index(drop=True)
    out["selection_mode"] = "leaders"
    out["leader_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["leader_similarity_threshold"] = float(best_threshold)
    return out


def gather_docking_candidates_by_inner_loop(
    outer_dir: Path, *, deepdel_model_path: Path, only_inner_loop: Optional[int] = None
) -> pd.DataFrame:
    cfg = _docking_selection_cfg()
    mode = cfg["mode"]
    if mode == "top":
        df = gather_top_candidates_by_inner_loop(
            outer_dir, topn_per_inner=cfg["n_to_dock_per_inner"], only_inner_loop=only_inner_loop
        )
        if len(df) > 0:
            df = df.copy()
            df["selection_mode"] = "top"
        return df
    if mode != "leaders":
        raise ValueError(f"Unknown docking_selection.mode='{mode}' (expected 'top' or 'leaders')")

    inner_root = outer_dir / "inners"
    if not inner_root.exists():
        return pd.DataFrame()
    id_to_idx, phi1, phi2, phi3, pooling = _load_deepdel_phi_context(deepdel_model_path)
    rows = []
    inner_dirs = (
        [inner_root / f"inner_{int(only_inner_loop)}"]
        if only_inner_loop is not None
        else sorted(inner_root.glob("inner_*"))
    )
    for inner_dir in inner_dirs:
        try:
            inner_idx = int(inner_dir.name.split("_")[-1])
        except Exception:
            continue
        topm_out = inner_dir / "topm_actual_scores.csv"
        if not topm_out.exists():
            continue
        df = select_leader_libraries_from_csv(
            topm_out,
            n_leaders=cfg["n_leaders_to_dock"],
            id_to_idx=id_to_idx,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            pooling=pooling,
            similarity=cfg["leader_similarity"],
            binary_search_iters=cfg["leader_binary_search_iters"],
        )
        if len(df) > 0:
            df = df.copy()
            df["inner_loop"] = inner_idx
            rows.append(df)
            print(f"[leaders] inner_{inner_idx}: selected {len(df)} leaders from {topm_out}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()