"""Utilities for computing molecular weights for DEL trimers."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


@lru_cache(maxsize=100_000)
def smiles_exact_molwt(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return float(rdMolDescriptors.CalcExactMolWt(mol))


def smiles_weight_array(smiles_list: Sequence[str]) -> np.ndarray:
    """Return an array of exact molecular weights for the given SMILES list."""
    return np.asarray([smiles_exact_molwt(s) for s in smiles_list], dtype=float)


def bb_weight_lookup(smiles_by_id: Mapping[int, str]) -> dict[int, float]:
    return {int(idx): smiles_exact_molwt(str(smi)) for idx, smi in smiles_by_id.items()}


def bb_sum_weights(
    ids_list: Iterable[Sequence[int]],
    *,
    bb_weights: dict[int, float],
) -> np.ndarray:
    """Compute additive molecular weights by summing BB weights for each triple."""
    out = []
    for ids in ids_list:
        weight = 0.0
        for idx in ids:
            weight += float(bb_weights.get(int(idx), 0.0))
        out.append(weight)
    return np.asarray(out, dtype=float)