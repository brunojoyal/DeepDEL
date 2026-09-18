"""Class-based architecture for building amide-coupled trimers.

Usage (CLI):
    python trimer_library_classes.py --input bifunctional_filtered.csv --outdir trimers --N 2 --save

Programmatic:
    from trimer_library_classes import PoolIO, TrimerBuilder, LibraryGenerator
    df = PoolIO.load_pool("bifunctional_filtered.csv")
    builder = TrimerBuilder(df)
    gen = LibraryGenerator(builder, outdir="trimers", save=True)
    for rec in gen.generate_first_N(N=2):
        print(rec["label"], rec["smi"], rec["mw"]) 
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions, rdMolDescriptors

# ------------------------------ Defaults ------------------------------
DEFAULT_INPUT = "bifunctional_filtered.csv"
DEFAULT_OUTDIR = "trimers"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SULFONAMIDE_RXN = str(PROJECT_ROOT / "data" / "AmpC" / "NH2-to-sulfoamide.rxn")

# ------------------------------ Chemistry primitives ------------------------------
class FunctionalGroups:
    """SMARTS patterns and simple counters used throughout the build."""
    ACID_H_SMARTS = Chem.MolFromSmarts("[CX3](=O)[O;H1]")   # -C(=O)OH
    ACID_AN_SMARTS = Chem.MolFromSmarts("[CX3](=O)[O-]")     # -C(=O)O-
    AMINE_SMARTS = Chem.MolFromSmarts("[NX3;H2]")            # primary amine
    SULFONYL_CHLORIDE_SMARTS = Chem.MolFromSmarts("[S](=[O])(=[O])[Cl]")

    @staticmethod
    def count_carboxyl(m: Chem.Mol) -> int:
        return len(m.GetSubstructMatches(FunctionalGroups.ACID_H_SMARTS)) + \
               len(m.GetSubstructMatches(FunctionalGroups.ACID_AN_SMARTS))

    @staticmethod
    def count_primary_amines(m: Chem.Mol) -> int:
        return len(m.GetSubstructMatches(FunctionalGroups.AMINE_SMARTS))

    @staticmethod
    def count_sulfonyl_chlorides(m: Chem.Mol) -> int:
        return len(m.GetSubstructMatches(FunctionalGroups.SULFONYL_CHLORIDE_SMARTS))


class Standardizer:
    """Gentle standardization layer with graceful fallback when rdMolStandardize is unavailable."""
    _has_std = False
    _normalizer = None
    _reionizer = None
    _uncharger = None

    @classmethod
    def _init(cls) -> None:
        if cls._normalizer is not None:
            return
        try:
            from rdkit.Chem import rdMolStandardize  # type: ignore
            cls._normalizer = rdMolStandardize.Normalizer()
            cls._reionizer = rdMolStandardize.Reionizer()
            cls._uncharger = rdMolStandardize.Uncharger()
            cls._has_std = True
        except Exception:
            cls._has_std = False

    @classmethod
    def standardize(cls, m: Chem.Mol) -> Chem.Mol:
        cls._init()
        m2 = Chem.Mol(m)
        if cls._has_std:
            m2 = cls._normalizer.normalize(m2)
            m2 = cls._reionizer.reionize(m2)
            m2 = cls._uncharger.uncharge(m2)
        Chem.SanitizeMol(m2)
        return m2


class MolOps:
    """SMILES ↔ Mol helpers and validation of building blocks."""
    @staticmethod
    def smi_to_std_mol(smi: str) -> Chem.Mol:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        return Standardizer.standardize(m)

    @staticmethod
    def validate_bb(m: Chem.Mol, tag: str = "") -> None:
        n_carbox = FunctionalGroups.count_carboxyl(m)
        n_nh2 = FunctionalGroups.count_primary_amines(m)
        if n_carbox != 1:
            raise ValueError(f"BB {tag} must have exactly one COOH/COO- (found {n_carbox}).")
        if n_nh2 != 1:
            raise ValueError(f"BB {tag} must have exactly one primary NH2 (found {n_nh2}).")

    @staticmethod
    def validate_amino_acid_bb(m: Chem.Mol, tag: str = "") -> None:
        MolOps.validate_bb(m, tag)

    @staticmethod
    def validate_primary_amine_bb(m: Chem.Mol, tag: str = "") -> None:
        """Validate a capped/non-acid BB that contributes only its primary amine."""
        n_nh2 = FunctionalGroups.count_primary_amines(m)
        if n_nh2 != 1:
            raise ValueError(f"BB {tag} must have exactly one primary NH2 (found {n_nh2}).")

    @staticmethod
    def validate_sulfonyl_chloride_bb(m: Chem.Mol, tag: str = "") -> None:
        n_so2cl = FunctionalGroups.count_sulfonyl_chlorides(m)
        if n_so2cl != 1:
            raise ValueError(f"BB {tag} must have exactly one sulfonyl chloride (found {n_so2cl}).")


class ReactiveSiteFinder:
    """Locates reactive atoms (acid carbonyl and amine nitrogen)."""
    _ACID_SMARTS = Chem.MolFromSmarts("[C:1](=[O:2])[O;H1:3]")
    _ANION_SMARTS = Chem.MolFromSmarts("[C:1](=[O:2])[O-:3]")
    _NPLUS_H3 = Chem.MolFromSmarts("[N+;H3]")
    _SULFONYL_CHLORIDE_SMARTS = Chem.MolFromSmarts("[S:1](=[O:2])(=[O:3])[Cl:4]")

    @staticmethod
    def find_primary_amine_n(m: Chem.Mol, prefer_exocyclic: bool = False) -> int:
        hits = m.GetSubstructMatches(FunctionalGroups.AMINE_SMARTS)
        if hits:
            candidates = [h[0] for h in hits]
            if prefer_exocyclic:
                ranked = sorted(
                    candidates,
                    key=lambda idx: (
                        ReactiveSiteFinder._amine_priority(m, idx),
                        idx,
                    ),
                )
                return ranked[0]
            return candidates[0]
        plus = m.GetSubstructMatches(ReactiveSiteFinder._NPLUS_H3)
        if plus:
            return plus[0][0]
        raise RuntimeError("No primary amine (NH2) found.")

    @staticmethod
    def _amine_priority(m: Chem.Mol, n_idx: int) -> Tuple[int, int, int, int]:
        """Lower tuple sorts first; prefer the leftover exocyclic, non-ring NH2."""
        atom = m.GetAtomWithIdx(n_idx)
        heavy_neighbors = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1]
        carbonyl_neighbor = 0
        sulfonyl_neighbor = 0
        carbon_sp3_neighbor = 1

        for nb in heavy_neighbors:
            if nb.GetAtomicNum() == 6:
                for bond in nb.GetBonds():
                    other = bond.GetOtherAtom(nb)
                    if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
                        carbonyl_neighbor = 1
                if nb.GetHybridization() == Chem.HybridizationType.SP3:
                    carbon_sp3_neighbor = 0
            if nb.GetAtomicNum() == 16:
                dbl_o = sum(
                    1
                    for bond in nb.GetBonds()
                    if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(nb).GetAtomicNum() == 8
                )
                if dbl_o >= 2:
                    sulfonyl_neighbor = 1

        return (
            1 if atom.IsInRing() else 0,
            carbonyl_neighbor,
            sulfonyl_neighbor,
            carbon_sp3_neighbor,
        )

    @staticmethod
    def find_acid_site(m: Chem.Mol) -> Dict[str, Optional[int]]:
        """
        Returns indices for carbonyl carbon (c), carbonyl oxygen to keep (o_keep),
        leaving oxygen (o_leave), and its hydrogen if present (h_leave).
        Works for COOH and COO−.
        """
        for pat, is_anion in ((ReactiveSiteFinder._ACID_SMARTS, False), (ReactiveSiteFinder._ANION_SMARTS, True)):
            matches = m.GetSubstructMatches(pat)
            if matches:
                c, o_keep, o_leave = matches[0]
                h_leave: Optional[int] = None
                if not is_anion:
                    o_atom = m.GetAtomWithIdx(o_leave)
                    for nb in o_atom.GetNeighbors():
                        if nb.GetAtomicNum() == 1:
                            h_leave = nb.GetIdx()
                            break
                return {"c": c, "o_keep": o_keep, "o_leave": o_leave, "h_leave": h_leave}
        raise RuntimeError("No suitable carboxyl (COOH/COO−) found.")

    @staticmethod
    def find_sulfonyl_chloride_site(m: Chem.Mol) -> Dict[str, int]:
        matches = m.GetSubstructMatches(ReactiveSiteFinder._SULFONYL_CHLORIDE_SMARTS)
        if not matches:
            raise RuntimeError("No suitable sulfonyl chloride found.")
        s, o1, o2, cl = matches[0]
        return {"s": s, "o1": o1, "o2": o2, "cl": cl}


class Merger:
    """Molecule merging utilities that preserve atom indices and stereo."""
    @staticmethod
    def combine_two_mols(m1: Chem.Mol, m2: Chem.Mol) -> Tuple[Chem.RWMol, List[int], List[int]]:
        nm = Chem.RWMol(Chem.CombineMols(m1, m2))
        n1 = m1.GetNumAtoms()
        map1 = list(range(n1))
        map2 = list(range(n1, n1 + m2.GetNumAtoms()))
        return nm, map1, map2


class AmideCoupler:
    """Implements manual amide coupling between an acid and a primary amine."""
    @staticmethod
    def couple(acid_mol: Chem.Mol, amine_mol: Chem.Mol) -> Chem.Mol:
        mol, _ = AmideCoupler.couple_with_metadata(acid_mol, amine_mol)
        return mol

    @staticmethod
    def couple_with_metadata(
        acid_mol: Chem.Mol,
        amine_mol: Chem.Mol,
        amine_target_n: Optional[int] = None,
    ) -> Tuple[Chem.Mol, Dict[str, int]]:
        """
        Couple acid_mol (COOH) with amine_mol (NH2).
        If amine_target_n is given, only that specific NH2 in amine_mol is allowed to react
        (all other N atoms are temporarily masked with Si before coupling).
        """
        acid = Chem.Mol(acid_mol)
        amine = Chem.Mol(amine_mol)

        # If a specific amine site is specified, mask all other N atoms in amine
        _N_MASK = 14  # Si
        _amine_masked_indices: List[int] = []
        if amine_target_n is not None:
            rw = Chem.RWMol(amine)
            for atom in rw.GetAtoms():
                if atom.GetIdx() == amine_target_n:
                    continue
                if atom.GetAtomicNum() == 7:
                    atom.SetAtomicNum(_N_MASK)
                    atom.SetFormalCharge(0)
                    _amine_masked_indices.append(atom.GetIdx())
            amine = rw.GetMol()

        site = ReactiveSiteFinder.find_acid_site(acid)
        c_idx_a, o_keep_a, o_leave_a, h_leave_a = site["c"], site["o_keep"], site["o_leave"], site["h_leave"]
        n_idx_b = ReactiveSiteFinder.find_primary_amine_n(amine)

        # The acid-side NH2 is the amine that survives into the product for step 2.
        acid_amine_hits = [h[0] for h in acid.GetSubstructMatches(FunctionalGroups.AMINE_SMARTS)]
        acid_amine_idx_a = acid_amine_hits[0] if acid_amine_hits else -1

        rw, mapA, mapB = Merger.combine_two_mols(acid, amine)
        c = mapA[c_idx_a]
        o_keep = mapA[o_keep_a]
        o_leave = mapA[o_leave_a]
        h_leave = mapA[h_leave_a] if h_leave_a is not None else None
        n = mapB[n_idx_b]
        surviving_amine = mapA[acid_amine_idx_a] if acid_amine_idx_a >= 0 else -1

        # enforce C=O double bond
        b_ck = rw.GetBondBetweenAtoms(c, o_keep)
        if b_ck is None:
            rw.AddBond(c, o_keep, Chem.BondType.DOUBLE)
        else:
            b_ck.SetBondType(Chem.BondType.DOUBLE)

        # break C–O(leaving) if present
        b_cl = rw.GetBondBetweenAtoms(c, o_leave)
        if b_cl is not None:
            rw.RemoveBond(c, o_leave)

        # delete in descending order to maintain indices
        to_delete: List[int] = []
        if h_leave is not None:
            to_delete.append(h_leave)
        to_delete.append(o_leave)
        for idx in sorted(to_delete, reverse=True):
            rw.RemoveAtom(idx)
            if n >= idx: n -= 1
            if c >= idx: c -= 1
            if o_keep >= idx: o_keep -= 1
            if surviving_amine >= idx and surviving_amine >= 0: surviving_amine -= 1

        # add new amide bond
        rw.AddBond(c, n, Chem.BondType.SINGLE)

        # neutralize ammonium if needed
        n_atom = rw.GetAtomWithIdx(n)
        if n_atom.GetFormalCharge() > 0:
            n_atom.SetFormalCharge(0)

        # Tag the surviving acid-side amine with a unique atom-map number so we can
        # re-locate it after RemoveHs potentially renumbers atoms.
        _AMINE_TAG = 99
        if surviving_amine >= 0:
            rw.GetAtomWithIdx(surviving_amine).SetAtomMapNum(_AMINE_TAG)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)

        # Re-locate the tagged atom and strip the tag
        meta: Dict[str, int] = {}
        rw2 = Chem.RWMol(mol)
        for atom in rw2.GetAtoms():
            if atom.GetAtomMapNum() == _AMINE_TAG:
                meta["surviving_acid_amine_idx"] = atom.GetIdx()
                # Backward-compatible alias for existing callers.
                meta["surviving_bb1_amine_idx"] = atom.GetIdx()
                atom.SetAtomMapNum(0)
                break
        mol = rw2.GetMol()

        # Restore any masked N atoms in the product
        if _amine_masked_indices:
            rw3 = Chem.RWMol(mol)
            for atom in rw3.GetAtoms():
                if atom.GetAtomicNum() == _N_MASK:
                    atom.SetAtomicNum(7)
                    atom.SetFormalCharge(0)
            mol = rw3.GetMol()

        Chem.SanitizeMol(mol)
        return mol, meta


class SulfonamideCoupler:
    """Implements NH2-to-sulfonamide coupling using the provided RXN file."""
    _rxn_cache: Dict[str, rdChemReactions.ChemicalReaction] = {}
    _NH2_MASK_ATOMIC_NUM = 14

    @classmethod
    def _load_reaction(cls, rxn_path: str | Path = DEFAULT_SULFONAMIDE_RXN) -> rdChemReactions.ChemicalReaction:
        rxn_path = str(rxn_path)
        if rxn_path not in cls._rxn_cache:
            rxn = rdChemReactions.ReactionFromRxnFile(rxn_path)
            if rxn is None:
                raise ValueError(f"Could not load reaction from {rxn_path}")
            rdChemReactions.SanitizeRxn(rxn)
            cls._rxn_cache[rxn_path] = rxn
        return cls._rxn_cache[rxn_path]

    @classmethod
    def couple(
        cls,
        amine_mol: Chem.Mol,
        sulfonyl_chloride_mol: Chem.Mol,
        rxn_path: str | Path = DEFAULT_SULFONAMIDE_RXN,
        target_n: Optional[int] = None,
    ) -> Chem.Mol:
        amine = Chem.Mol(amine_mol)
        sulfonyl = Chem.Mol(sulfonyl_chloride_mol)
        rxn = cls._load_reaction(rxn_path)

        if target_n is None:
            target_n = ReactiveSiteFinder.find_primary_amine_n(amine, prefer_exocyclic=True)
        masked_amine = cls._mask_non_target_primary_amines(amine, target_n)

        products = rxn.RunReactants((masked_amine, sulfonyl))
        if not products:
            raise RuntimeError("Sulfonamide reaction produced no products.")

        sanitized: List[Chem.Mol] = []
        for product_tuple in products:
            for product in product_tuple:
                mol = Chem.Mol(product)
                try:
                    mol = cls._unmask_atoms(mol)
                    Chem.SanitizeMol(mol)
                    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
                    mol = Chem.RemoveHs(mol)
                    Chem.SanitizeMol(mol)
                    sanitized.append(mol)
                except Exception:
                    continue

        if not sanitized:
            raise RuntimeError("Sulfonamide reaction generated only invalid products.")

        sanitized.sort(key=lambda m: Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))
        return sanitized[0]

    @classmethod
    def _mask_non_target_primary_amines(cls, mol: Chem.Mol, target_idx: int) -> Chem.Mol:
        """Mask ALL nitrogen atoms except the target so only the intended NH2 can react."""
        rw = Chem.RWMol(Chem.Mol(mol))
        for atom in rw.GetAtoms():
            if atom.GetIdx() == target_idx:
                continue
            if atom.GetAtomicNum() != 7:
                continue
            # Mask every N (including amide NH) so the RXN cannot fire on the wrong site.
            atom.SetAtomicNum(cls._NH2_MASK_ATOMIC_NUM)
            atom.SetFormalCharge(0)
        masked = rw.GetMol()
        return masked

    @classmethod
    def _unmask_atoms(cls, mol: Chem.Mol) -> Chem.Mol:
        rw = Chem.RWMol(Chem.Mol(mol))
        changed = False
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == cls._NH2_MASK_ATOMIC_NUM:
                atom.SetAtomicNum(7)
                atom.SetFormalCharge(0)
                changed = True
        if not changed:
            return Chem.Mol(mol)
        out = rw.GetMol()
        return out


# ------------------------------ I/O and auditing ------------------------------
class PoolIO:
    @staticmethod
    def split_by_pool(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a combined BBs frame into per-cycle (B1, B2, B3) DataFrames.

        If `df` has a `pool` column tagging origin (1, 2, 3) we partition on it.
        Otherwise we return the same frame three times (legacy single-pool behaviour).
        Each returned frame keeps its row indexing reset so TrimerBuilder works
        cleanly on it.
        """
        if "pool" in df.columns:
            d1 = df.loc[df["pool"].astype(int) == 1].reset_index(drop=True)
            d2 = df.loc[df["pool"].astype(int) == 2].reset_index(drop=True)
            d3 = df.loc[df["pool"].astype(int) == 3].reset_index(drop=True)
            if len(d1) == 0 or len(d2) == 0 or len(d3) == 0:
                # If the pool column exists but has no 1/2/3 tags (e.g. all 0),
                # fall back to legacy single-pool behaviour rather than failing.
                if len(df) > 0:
                    return df.reset_index(drop=True), df.reset_index(drop=True), df.reset_index(drop=True)
                raise ValueError(
                    f"PoolIO.split_by_pool: combined CSV has empty pool(s): "
                    f"|B1|={len(d1)}, |B2|={len(d2)}, |B3|={len(d3)}"
                )
            return d1, d2, d3
        return df.reset_index(drop=True), df.reset_index(drop=True), df.reset_index(drop=True)

    @staticmethod
    def load_pool(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        smiles_col = PoolIO.resolve_smiles_col(df)
        if smiles_col != "SMILES":
            df = df.rename(columns={smiles_col: "SMILES"})
        if "Name" not in df.columns:
            if "Catalog_ID" in df.columns:
                df["Name"] = df["Catalog_ID"].astype(str)
            else:
                df["Name"] = [f"BB_{i}" for i in range(len(df))]
        return df

    @staticmethod
    def resolve_smiles_col(df: pd.DataFrame) -> str:
        for col in ("SMILES", "Smiles", "smiles"):
            if col in df.columns:
                return col
        raise ValueError("CSV must contain a SMILES-like column ('SMILES', 'Smiles', or 'smiles').")

    @staticmethod
    def save_outputs(mol: Chem.Mol, smi: str, outdir: Path, label: str) -> Dict[str, str]:
        outdir.mkdir(parents=True, exist_ok=True)
        AllChem.Compute2DCoords(mol)
        sdf_path = outdir / f"trimer_{label}.sdf"
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
        png_path = outdir / f"trimer_{label}.png"
        Draw.MolToFile(mol, str(png_path), size=(720, 520), kekulize=True)
        smi_path = outdir / f"trimer_{label}.smi"
        smi_path.write_text(smi + "\n", encoding="utf-8")
        return {"sdf": str(sdf_path), "png": str(png_path), "smi": str(smi_path)}

    @staticmethod
    def audit_pool(df: pd.DataFrame, limit: int = 50) -> List[Dict[str, object]]:
        """Returns a list of problematic blocks (and prints a brief summary)."""
        bad: List[Dict[str, object]] = []
        for idx, row in df.iterrows():
            try:
                m = MolOps.smi_to_std_mol(row["SMILES"])
            except Exception:
                bad.append({"idx": idx, "Name": row.get("Name", f"BB_{idx}"), "reason": "Invalid SMILES"})
                continue
            n_carbox = FunctionalGroups.count_carboxyl(m)
            n_nh2 = FunctionalGroups.count_primary_amines(m)
            if n_carbox != 1 or n_nh2 != 1:
                bad.append({
                    "idx": idx,
                    "Name": row.get("Name", f"BB_{idx}"),
                    "SMILES": row["SMILES"],
                    "n_carboxyl": n_carbox,
                    "n_primary_NH2": n_nh2,
                })
        if bad:
            print(f"Found {len(bad)} blocks that fail the check (showing up to {limit}):")
            for b in bad[:limit]:
                print(b)
        else:
            print("All good: exactly one carboxyl (acid/anion) and one primary NH2 per block.")
        return bad


# ------------------------------ Core builder ------------------------------
@dataclass
class TrimerRecord:
    mol: Chem.Mol
    smi: str
    mw: float
    meta: Dict[str, str]
    label: str


# Valid reaction modes
REACTION_MODE_AMIDE_SULFONAMIDE = "amide_sulfonamide"   # AmpC: step1=amide, step2=sulfonamide (BB3=sulfonyl chloride)
REACTION_MODE_AMIDE_AMIDE       = "amide_amide"          # sEH:  step1=amide, step2=amide       (BB3=amino acid)
REACTION_MODE_AMIDE_AMIDE_LEGACY = "amide_amide_legacy" # sEH legacy: BB1 acid + BB2 NH2, then BB2 acid + BB3 NH2
VALID_REACTION_MODES = (
    REACTION_MODE_AMIDE_SULFONAMIDE,
    REACTION_MODE_AMIDE_AMIDE,
    REACTION_MODE_AMIDE_AMIDE_LEGACY,
)


class TrimerBuilder:
    """
    reaction_mode controls the second coupling step:
      "amide_sulfonamide"  (default/AmpC): BB3 is a sulfonyl chloride; step 2 uses SulfonamideCoupler
      "amide_amide"        (tracked-N):    BB3 is an amino acid acid donor; BB2's tracked NH2 survives to step 2
      "amide_amide_legacy" (old sEH):      BB1 acid + BB2 NH2, then BB2 acid + BB3 NH2
    """
    def __init__(
        self,
        pool_df: pd.DataFrame,
        pool_df_2: Optional[pd.DataFrame] = None,
        pool_df_3: Optional[pd.DataFrame] = None,
        reaction_mode: str = REACTION_MODE_AMIDE_SULFONAMIDE,
    ):
        self.df1 = pool_df.reset_index(drop=True)
        self.df2 = (pool_df_2 if pool_df_2 is not None else pool_df).reset_index(drop=True)
        self.df3 = (pool_df_3 if pool_df_3 is not None else (pool_df_2 if pool_df_2 is not None else pool_df)).reset_index(drop=True)
        self.df = self.df1
        if reaction_mode not in VALID_REACTION_MODES:
            raise ValueError(f"reaction_mode must be one of {VALID_REACTION_MODES}, got {reaction_mode!r}")
        self.reaction_mode = reaction_mode

    @staticmethod
    def build_trimer_from_smiles(
        smi_i: str,
        smi_j: str,
        smi_k: str,
        reaction_mode: str = REACTION_MODE_AMIDE_SULFONAMIDE,
    ) -> "TrimerRecord":
        """
        Build a trimer directly from three SMILES.
        reaction_mode controls the second step (see class docstring).
        """
        mi, mj, mk = map(MolOps.smi_to_std_mol, (smi_i, smi_j, smi_k))

        if reaction_mode == REACTION_MODE_AMIDE_AMIDE_LEGACY:
            # Historical sEH topology from the old dels.py:
            #   step 1: BB1 COOH + BB2 NH2 -> BB1-amide-BB2, leaving BB2 COOH
            #   step 2: remaining BB2 COOH + BB3 NH2 -> final diamide
            MolOps.validate_amino_acid_bb(mi, "i")
            MolOps.validate_amino_acid_bb(mj, "j")
            MolOps.validate_amino_acid_bb(mk, "k")
            ij = AmideCoupler.couple(mi, mj)
            if FunctionalGroups.count_carboxyl(ij) == 0:
                ij = Standardizer.standardize(ij)
            ijk = AmideCoupler.couple(ij, mk)
            smi_ijk = Chem.MolToSmiles(ijk, isomericSmiles=True, canonical=True)
            mw = rdMolDescriptors.CalcExactMolWt(ijk)
            return TrimerRecord(ijk, smi_ijk, mw, meta={"i": "", "j": "", "k": ""}, label="custom")

        MolOps.validate_primary_amine_bb(mi, "i")
        MolOps.validate_amino_acid_bb(mj, "j")

        # Step 1: BB2 (mj) COOH + BB1 (mi) NH2 → amide; BB2's NH2 survives for step 2
        ij, meta = AmideCoupler.couple_with_metadata(mj, mi)
        target = meta.get("surviving_acid_amine_idx")

        if reaction_mode == REACTION_MODE_AMIDE_SULFONAMIDE:
            MolOps.validate_sulfonyl_chloride_bb(mk, "k")
            ijk = SulfonamideCoupler.couple(ij, mk, target_n=target)
        elif reaction_mode == REACTION_MODE_AMIDE_AMIDE:
            MolOps.validate_amino_acid_bb(mk, "k")
            # acid=mk (BB3 provides COOH), amine=ij (intermediate provides the tracked NH2)
            ijk, _ = AmideCoupler.couple_with_metadata(mk, ij, amine_target_n=target)
        else:
            raise ValueError(f"Unknown reaction_mode: {reaction_mode!r}")

        smi_ijk = Chem.MolToSmiles(ijk, isomericSmiles=True, canonical=True)
        mw = rdMolDescriptors.CalcExactMolWt(ijk)
        return TrimerRecord(ijk, smi_ijk, mw, meta={"i": "", "j": "", "k": ""}, label="custom")

    def build_trimer_by_ids(self, id_i: int, id_j: int, id_k: int, id_col: str = "ID") -> "TrimerRecord":
        """Look up SMILES by an ID column and build the trimer."""
        def smi_for(df: pd.DataFrame, x: int) -> str:
            if id_col in df.columns:
                hits = df.index[df[id_col] == x]
                if len(hits) != 1:
                    raise ValueError(f"BB ID {x} not found or not unique in column '{id_col}'.")
                return str(df.loc[hits[0], "SMILES"])
            else:
                return str(df.loc[x, "SMILES"])
        return TrimerBuilder.build_trimer_from_smiles(
            smi_for(self.df1, id_i),
            smi_for(self.df2, id_j),
            smi_for(self.df3, id_k),
            reaction_mode=self.reaction_mode,
        )

    def build_trimer(self, i: int, j: int, k: int, debug: bool = False) -> TrimerRecord:
        """
        Step 1: BB2 (j) COOH + BB1 (i) NH2 → amide; BB2's NH2 survives for step 2.
        Step 2 depends on reaction_mode:
          "amide_sulfonamide": BB3 sulfonyl chloride + surviving NH2 → sulfonamide
          "amide_amide":       BB3 amino acid COOH  + surviving NH2 → amide
          "amide_amide_legacy": BB1 COOH + BB2 NH2, then BB2 COOH + BB3 NH2
        """
        si, sj, sk = self.df1.loc[i, "SMILES"], self.df2.loc[j, "SMILES"], self.df3.loc[k, "SMILES"]
        ni, nj, nk = self.df1.loc[i, "Name"],   self.df2.loc[j, "Name"],   self.df3.loc[k, "Name"]

        mi, mj, mk = map(MolOps.smi_to_std_mol, (si, sj, sk))

        if self.reaction_mode == REACTION_MODE_AMIDE_AMIDE_LEGACY:
            MolOps.validate_amino_acid_bb(mi, "i")
            MolOps.validate_amino_acid_bb(mj, "j")
            MolOps.validate_amino_acid_bb(mk, "k")
            ij = AmideCoupler.couple(mi, mj)
            if FunctionalGroups.count_carboxyl(ij) == 0:
                ij = Standardizer.standardize(ij)
            ijk = AmideCoupler.couple(ij, mk)
            smi_ijk = Chem.MolToSmiles(ijk, isomericSmiles=True, canonical=True)
            mw = rdMolDescriptors.CalcExactMolWt(ijk)
            meta = {"i": ni, "j": nj, "k": nk}
            label = f"{i}_{j}_{k}"
            return TrimerRecord(ijk, smi_ijk, mw, meta, label)

        MolOps.validate_primary_amine_bb(mi, "i")
        MolOps.validate_amino_acid_bb(mj, "j")

        # Step 1: BB2 COOH + BB1 NH2 → amide; BB2's NH2 survives as reactive site for step 2
        ij, step1_meta = AmideCoupler.couple_with_metadata(mj, mi)
        target = step1_meta.get("surviving_acid_amine_idx")

        # Step 2
        if self.reaction_mode == REACTION_MODE_AMIDE_SULFONAMIDE:
            MolOps.validate_sulfonyl_chloride_bb(mk, "k")
            ijk = SulfonamideCoupler.couple(ij, mk, target_n=target)
        else:  # REACTION_MODE_AMIDE_AMIDE
            MolOps.validate_amino_acid_bb(mk, "k")
            ijk, _ = AmideCoupler.couple_with_metadata(mk, ij, amine_target_n=target)

        smi_ijk = Chem.MolToSmiles(ijk, isomericSmiles=True, canonical=True)
        mw = rdMolDescriptors.CalcExactMolWt(ijk)
        meta = {"i": ni, "j": nj, "k": nk}
        label = f"{i}_{j}_{k}"
        return TrimerRecord(ijk, smi_ijk, mw, meta, label)


# ------------------------------ Library generation ------------------------------
class LibraryGenerator:
    def __init__(self, builder: TrimerBuilder, outdir: str | Path = DEFAULT_OUTDIR, save: bool = False):
        self.builder = builder
        self.outdir = Path(outdir)
        self.save = save

    def generate_first_N(self, N: int) -> Iterable[Dict[str, object]]:
        n1 = min(N, len(self.builder.df1))
        n2 = min(N, len(self.builder.df2))
        n3 = min(N, len(self.builder.df3))
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    rec = self.builder.build_trimer(i, j, k)
                    paths = None
                    if self.save:
                        paths = PoolIO.save_outputs(rec.mol, rec.smi, self.outdir, rec.label)
                    yield {
                        "label": rec.label,
                        "meta": rec.meta,
                        "smi": rec.smi,
                        "mw": rec.mw,
                        "paths": paths,
                    }


# ------------------------------ CLI ------------------------------
class CLI:
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Build trimers from cycle-specific pools using amide then sulfonamide coupling.")
        p.add_argument("--input", default=DEFAULT_INPUT, help="Cycle 1 CSV with SMILES and optional Name column")
        p.add_argument("--input2", default=None, help="Cycle 2 CSV with SMILES and optional Name column (defaults to --input)")
        p.add_argument("--input3", default=None, help="Cycle 3 CSV with SMILES and optional Name column (defaults to --input2 or --input)")
        p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory for outputs (if --save)")
        p.add_argument("--N", type=int, default=20, help="Use the first N building blocks for i, j, k")
        p.add_argument("--save", action="store_true", help="Write SDF/PNG/SMI files")
        p.add_argument("--audit", action="store_true", help="Audit the input pool and exit")
        return p

    @staticmethod
    def run():
        parser = CLI.build_parser()
        args = parser.parse_args()

        df1 = PoolIO.load_pool(args.input)
        df2 = PoolIO.load_pool(args.input2) if args.input2 else df1.copy()
        df3 = PoolIO.load_pool(args.input3) if args.input3 else df2.copy()
        if args.audit:
            print("[audit] Cycle 1 pool")
            PoolIO.audit_pool(df1)
            print("[audit] Cycle 2 pool")
            PoolIO.audit_pool(df2)
            print("[audit] Cycle 3 pool")
            bad = []
            for idx, row in df3.iterrows():
                try:
                    m = MolOps.smi_to_std_mol(row["SMILES"])
                    MolOps.validate_sulfonyl_chloride_bb(m, row.get("Name", f"BB_{idx}"))
                except Exception as exc:
                    bad.append({"idx": idx, "Name": row.get("Name", f"BB_{idx}"), "reason": str(exc)})
            if bad:
                print(f"Found {len(bad)} cycle-3 blocks that fail the sulfonyl chloride check (showing up to 50):")
                for b in bad[:50]:
                    print(b)
            else:
                print("All good for cycle 3: exactly one sulfonyl chloride per block.")
            return

        builder = TrimerBuilder(df1, df2, df3)
        gen = LibraryGenerator(builder, outdir=args.outdir, save=args.save)
        for rec in gen.generate_first_N(args.N):
            i, j, k = rec["label"].split("_")
            meta = rec["meta"]
            print(f"Built trimer ({i}, {j}, {k})  =>  {meta['i']} – {meta['j']} – {meta['k']}")
            print(f"SMILES: {rec['smi']}")
            print(f"Exact mass: {rec['mw']:.5f}")
            if rec["paths"]:
                print("Outputs:")
                for k2, v2 in rec["paths"].items():
                    print(f"  {k2}: {v2}")
            print()


def main():
    CLI.run()


if __name__ == "__main__":
    main()
